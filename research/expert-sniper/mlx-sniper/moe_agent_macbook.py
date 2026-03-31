#!/usr/bin/env python3
"""
MoE Sniper Agent — Qwen3.5-35B-A3B (22 GB) on 8 GB M2 MacBook.
Modeled after agent.py: intent classification → tool routing → LLM response.
"""

import json, sys, os, time, gc, random, subprocess, re
from datetime import datetime
from pathlib import Path
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding

from expert_io import MoEExpertReader

console = Console()

MODEL_DIR = "/Volumes/USB DISK/qwen35-35b-moe-stream"
BITS = 4
GROUP_SIZE = 64
MAX_TOKENS = 1024

CREATURES = [
    ["   ⚡( ᐛ )⚡  ", "  ⚡( ᐛ )⚡   ", " ⚡( ᐛ )⚡    ", "  ⚡( ᐛ )⚡   "],
    ["  ⠋  ", "  ⠙  ", "  ⠹  ", "  ⠸  ", "  ⠼  ", "  ⠴  ", "  ⠦  ", "  ⠧  "],
]
CREATURE = CREATURES[random.randint(0, len(CREATURES) - 1)]

TOOL_KEYWORDS = [
    "search", "find", "look up", "google", "what time", "when do",
    "when is", "who is", "who won", "weather", "news", "latest",
    "price", "stock", "score", "tonight", "today", "tomorrow",
]
SHELL_KEYWORDS = [
    "list files", "show files", "what's on my", "disk space", "run ",
    "execute", "find files", "read file", "create file",
]


def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    """gather_qmm fusion — matches native SwitchGLU math exactly."""
    active_ids = sorted(expert_data.keys())
    id_to_local = {eid: i for i, eid in enumerate(active_ids)}

    inds_np = np.array(top_k_indices)
    local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
    local_indices = mx.array(local_np)

    def stack_proj(proj):
        w = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj}.weight"] for eid in active_ids])
        s = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj}.scales"] for eid in active_ids])
        b = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    x_exp = mx.expand_dims(x, (-2, -3))

    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)

    hidden = nn.silu(gate_out) * up_out

    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)

    out = down_out.squeeze(-2)
    out = (out * top_k_weights[..., None]).sum(axis=-2)
    return out


class MoESniperEngine:
    def __init__(self):
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 40

    def load(self):
        with open(f"{MODEL_DIR}/config.json") as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]
        streaming = config["streaming"]

        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
        args = TextModelArgs(
            model_type=config.get("model_type"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            max_position_embeddings=config["max_position_embeddings"],
            head_dim=config.get("head_dim"),
            tie_word_embeddings=config["tie_word_embeddings"],
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            shared_expert_intermediate_size=config["shared_expert_intermediate_size"],
            moe_intermediate_size=config["moe_intermediate_size"],
            linear_num_value_heads=config.get("linear_num_value_heads"),
            linear_num_key_heads=config.get("linear_num_key_heads"),
            linear_key_head_dim=config.get("linear_key_head_dim"),
            linear_value_head_dim=config.get("linear_value_head_dim"),
            linear_conv_kernel_dim=config.get("linear_conv_kernel_dim"),
            full_attention_interval=config.get("full_attention_interval"),
            rope_parameters=config.get("rope_parameters"),
        )

        self.model = TextModel(args)
        from mlx_lm.models.switch_layers import SwitchLinear
        SSM_PROTECT = {"conv1d"}
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding): return True
            if isinstance(module, SwitchLinear): return True
            if not isinstance(module, nn.Linear): return False
            if any(k in path for k in SSM_PROTECT): return False
            if module.weight.shape[-1] < GROUP_SIZE: return False
            return True
        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS, class_predicate=should_quantize)

        mx.set_memory_limit(6 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)

        pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
        self.model.load_weights(list(pinned.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        # LRU cache: 300 experts ≈ 500 MB, fits in 8 GB alongside 1.4 GB pinned
        self.reader = MoEExpertReader(f"{MODEL_DIR}/{streaming['expert_dir']}", self.num_layers, num_workers=4, cache_size=300)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        return pinned_gb

    def reset_cache(self):
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        h = self.model.model.embed_tokens(input_ids)
        fa_mask = create_attention_mask(h, self.cache[self.model.model.fa_idx])
        ssm_mask = create_ssm_mask(h, self.cache[self.model.model.ssm_idx])

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=self.cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=self.cache[i])
            h = h + attn_out
            mx.eval(h)

            normed = layer.post_attention_layernorm(h)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            active_ids = list(set(int(e) for e in np.array(inds).flatten()))
            # Profiling hook: record activations if profiler is attached
            if hasattr(self, '_profiler') and self._profiler is not None:
                try:
                    weights = [float(x) for x in mx.reshape(scores, (-1,)).tolist()]
                except:
                    weights = []
                self._profiler(i, active_ids, weights)
            if i + 1 < self.num_layers:
                self.reader.prefetch_experts(i + 1, active_ids)
            expert_data = self.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def generate(self, messages, max_tokens=MAX_TOKENS, temperature=0.7, stream=True):
        """Generate response. Yields chunks if stream=True, returns full text if False."""
        self.reset_cache()

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        logits = self.forward(input_ids)
        mx.eval(logits)

        generated = []
        full_text = ""
        in_think = False  # Yield everything — model "thinks" directly as text

        for _ in range(max_tokens):
            next_logits = logits[:, -1, :]
            if generated:
                seen = mx.array(list(set(generated[-100:])))
                pl = next_logits[:, seen]
                pl = mx.where(pl > 0, pl / 1.15, pl * 1.15)
                next_logits[:, seen] = pl

            probs = mx.softmax(next_logits / temperature, axis=-1)
            sorted_idx = mx.argsort(-probs, axis=-1)
            sorted_p = mx.take_along_axis(probs, sorted_idx, axis=-1)
            cumsum = mx.cumsum(sorted_p, axis=-1)
            mask = (cumsum - sorted_p) <= 0.9
            sorted_p = sorted_p * mask
            sorted_p = sorted_p / (sorted_p.sum(axis=-1, keepdims=True) + 1e-10)
            token = mx.random.categorical(mx.log(sorted_p + 1e-10))
            token = mx.take_along_axis(sorted_idx, token[:, None], axis=-1).squeeze(-1)
            mx.eval(token)
            token_id = token.item()

            # Stop conditions
            if token_id in (248044, 248045):
                break

            generated.append(token_id)
            chunk = self.tokenizer.decode([token_id])

            # Stop on special tokens in text
            if "<|im_end|>" in chunk or "<|endoftext|>" in chunk:
                break

            # Track think state — model thinks silently, yields after </think>
            if "</think>" in chunk:
                in_think = False
            elif not in_think and chunk:
                full_text += chunk
                if stream:
                    yield chunk

            # ALWAYS advance the model (never skip forward pass)
            logits = self.forward(mx.array([[token_id]]))
            mx.eval(logits)

        if not stream:
            return full_text

    def quick_call(self, messages, max_tokens=50, temperature=0.0):
        """Non-streaming LLM call for intent classification etc."""
        result = ""
        for chunk in self.generate(messages, max_tokens=max_tokens, temperature=temperature, stream=True):
            result += chunk
        return result.strip()


# ── Tool functions ────────────────────────────────

def classify_intent(engine, message):
    lower = message.lower()
    if any(k in lower for k in SHELL_KEYWORDS):
        return "shell"
    if any(k in lower for k in TOOL_KEYWORDS):
        return "search"
    return "chat"

def quick_search(engine, query):
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    today = datetime.now().strftime("%A, %B %d, %Y")

    # Rewrite query
    search_query = engine.quick_call([
        {"role": "system", "content": f"Today is {today}. Rewrite the user's question into an optimal web search query. Output ONLY the search query, nothing else."},
        {"role": "user", "content": query},
    ], max_tokens=30, temperature=0.0)
    if not search_query:
        search_query = query

    # Search
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=5):
                results.append(f"- {r['title']}: {r['body']}")
    except:
        return None

    if not results:
        return None

    context = "\n".join(results)
    return context, search_query

def run_shell(engine, query):
    home = os.path.expanduser("~")
    cmd = engine.quick_call([
        {"role": "system", "content": f"Generate a single macOS shell command for the user's request. Home dir: {home}. Output ONLY the command."},
        {"role": "user", "content": query},
    ], max_tokens=100, temperature=0.0)
    if not cmd:
        return None, None

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout[:4000]
        if result.stderr:
            output += f"\n{result.stderr[:1000]}"
    except Exception as e:
        output = f"Error: {e}"

    return cmd, output


# ── UI ────────────────────────────────────────────

class ThinkingDisplay:
    def __init__(self, phase="thinking"):
        self.frame = 0
        self.start = time.time()
        self.phase = phase
    def render(self):
        self.frame += 1
        cf = CREATURE[self.frame % len(CREATURE)]
        t = Text()
        t.append(f"  {cf}", style="bright_cyan")
        t.append(f"  {self.phase}", style="bold bright_cyan")
        t.append(f"  {time.time()-self.start:.0f}s", style="dim")
        return t


def print_banner(pinned_gb):
    console.print()
    logo = Text()
    logo.append("  moe", style="bold bright_cyan")
    logo.append("-", style="dim")
    logo.append("sniper", style="bold bright_yellow")
    console.print(logo)
    sub = Text()
    sub.append("  35B model · 1.4 GB RAM · experts sniped from SSD", style="dim italic")
    console.print(sub)
    console.print()
    rows = [
        ("model", "Qwen3.5-35B-A3B", "Q4_K_M · 22 GB · 256 experts"),
        ("pinned", f"{pinned_gb:.1f} GB", "attention + SSM + router + shared"),
        ("tools", "search · shell · chat", "LLM-as-router"),
        ("cost", "$0.00/hr", "Apple Silicon · local"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:8s} ", style="bold dim")
        line.append(value, style="bold white")
        line.append(f"  {extra}", style="dim")
        console.print(line)
    console.print()
    console.print(Rule(style="dim"))
    console.print()


def main():
    console.clear()
    with console.status("[bold bright_cyan]  Loading MoE Sniper...", spinner="dots"):
        engine = MoESniperEngine()
        pinned_gb = engine.load()
    print_banner(pinned_gb)

    session_tokens = 0
    session_time = 0.0

    while True:
        try:
            console.print("  [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]goodbye.[/]\n")
            break

        if not user_input.strip():
            continue
        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/clear":
            console.clear()
            print_banner(pinned_gb)
            continue
        elif cmd == "/stats":
            avg = session_tokens / session_time if session_time > 0 else 0
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_column(style="bold bright_cyan", width=12)
            t.add_column()
            t.add_row("tokens", f"{session_tokens:,}")
            t.add_row("time", f"{session_time:.1f}s")
            t.add_row("speed", f"{avg:.2f} tok/s")
            t.add_row("memory", f"{mx.get_active_memory()/1e9:.2f} GB")
            console.print(t)
            console.print()
            continue
        elif cmd in ("/help", "/?"):
            for c, d in [("/clear", "Reset"), ("/stats", "Stats"), ("/quit", "Exit")]:
                console.print(f"  [bold bright_cyan]{c:10s}[/] [dim]{d}[/]")
            console.print()
            continue

        console.print()

        # Classify intent
        intent = classify_intent(engine, user_input)

        start = time.time()
        tokens = 0

        if intent == "search":
            # Web search
            display = ThinkingDisplay("searching the web")
            with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
                search_result = quick_search(engine, user_input)
                while search_result is None:
                    break

            if search_result:
                context, query = search_result
                console.print(f"  [dim]searched: {query}[/]\n")
                messages = [
                    {"role": "system", "content": f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. Answer the user's question using these search results. Be concise.\n\nSearch results:\n{context}"},
                    {"role": "user", "content": user_input},
                ]
                console.print("  ", end="")
                for chunk in engine.generate(messages, temperature=0.3):
                    console.print(chunk, end="", highlight=False)
                    tokens += 1
            else:
                console.print("  [dim]search unavailable, answering directly[/]\n")
                messages = [
                    {"role": "system", "content": "Be concise and helpful."},
                    {"role": "user", "content": user_input},
                ]
                console.print("  ", end="")
                for chunk in engine.generate(messages):
                    console.print(chunk, end="", highlight=False)
                    tokens += 1

        elif intent == "shell":
            # Shell command
            display = ThinkingDisplay("running command")
            with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
                cmd_str, output = run_shell(engine, user_input)

            if cmd_str:
                console.print(f"  [dim]$ {cmd_str}[/]\n")
                messages = [
                    {"role": "system", "content": "Present these shell command results clearly. Be concise."},
                    {"role": "user", "content": f"Command: {cmd_str}\nOutput:\n{output}\n\nOriginal question: {user_input}"},
                ]
                console.print("  ", end="")
                for chunk in engine.generate(messages, temperature=0.3):
                    console.print(chunk, end="", highlight=False)
                    tokens += 1

        else:
            # Chat
            display = ThinkingDisplay("thinking")
            first_token = True
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Be concise and helpful."},
                {"role": "user", "content": user_input},
            ]
            with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
                for chunk in engine.generate(messages):
                    if first_token:
                        first_token = False
                        live.stop()
                        console.print("  ", end="")
                    console.print(chunk, end="", highlight=False)
                    tokens += 1

        elapsed = time.time() - start
        if tokens > 0:
            speed = tokens / elapsed
            clr = "bright_green" if speed > 1 else "yellow"
            s = Text()
            s.append(f"\n\n  {speed:.2f} tok/s", style=f"bold {clr}")
            s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
            console.print(s)
        console.print()

        session_tokens += tokens
        session_time += elapsed


if __name__ == "__main__":
    main()
