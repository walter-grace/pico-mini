#!/usr/bin/env python3
"""
Interactive Demo: Qwen3.5-35B-A3B on 8 GB RAM via MoE Expert Streaming.

This model is 22 GB. It should not fit. It does.
llama.cpp OOMs on this hardware. We don't.

Usage:
    python3 interactive_demo.py              # Expert streaming (works on 8GB)
    python3 interactive_demo.py --llama      # llama.cpp IQ2_M (needs 16GB)
    python3 interactive_demo.py --compare    # Side-by-side comparison
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

# ── Pretty output ──────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def banner(mode):
    print()
    print(f"  {BOLD}{CYAN}moe{RESET}{DIM}-{RESET}{BOLD}{YELLOW}sniper{RESET}  {DIM}interactive demo{RESET}")
    print()
    if mode == "stream":
        print(f"  {BOLD}Model{RESET}     Qwen3.5-35B-A3B {DIM}(22 GB, Q4_K_M 4-bit){RESET}")
        print(f"  {BOLD}Method{RESET}    Expert streaming from SSD")
        print(f"  {BOLD}RAM{RESET}       1.4 GB pinned {DIM}(on 8 GB machine){RESET}")
        print(f"  {BOLD}Experts{RESET}   8 of 256 per token × 40 layers")
    elif mode == "llama":
        print(f"  {BOLD}Model{RESET}     Qwen3.5-35B-A3B {DIM}(10.6 GB, IQ2_M 2-bit){RESET}")
        print(f"  {BOLD}Method{RESET}    llama.cpp (full model in RAM)")
        print(f"  {BOLD}RAM{RESET}       ~10.6 GB required")
    print()
    print(f"  {DIM}Type a question. Type /quit to exit. Type /stats for stats.{RESET}")
    print(f"  {DIM}{'─' * 56}{RESET}")
    print()


# ── Expert Streaming Backend ───────────────────────────

class StreamingBackend:
    def __init__(self):
        self.engine = None
        self.total_tokens = 0
        self.total_time = 0.0

    def load(self):
        import mlx.core as mx
        from moe_agent_macbook import MoESniperEngine
        self.engine = MoESniperEngine()
        pinned_gb = self.engine.load()
        self.mem_active = mx.get_active_memory() / 1e9
        self.mem_peak = mx.get_peak_memory() / 1e9
        return pinned_gb

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Be concise and clear. Answer directly in 2-4 sentences. Do not think step by step."},
            {"role": "user", "content": prompt},
        ]
        start = time.time()
        tokens = 0
        for chunk in self.engine.generate(messages, temperature=0.4):
            yield chunk
            tokens += 1
        elapsed = time.time() - start
        self.total_tokens += tokens
        self.total_time += elapsed
        yield None  # signal done
        # Store last stats
        self._last_tokens = tokens
        self._last_time = elapsed

    def stats(self):
        import mlx.core as mx
        avg = self.total_tokens / self.total_time if self.total_time > 0 else 0
        return {
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_speed": avg,
            "memory_gb": mx.get_active_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
        }


# ── llama.cpp Backend ──────────────────────────────────

class LlamaBackend:
    def __init__(self, model_path):
        self.model_path = model_path
        self.total_tokens = 0
        self.total_time = 0.0

    def load(self):
        import subprocess
        # Start llama-server in background
        self.proc = subprocess.Popen(
            ["llama-server", "-m", self.model_path,
             "--port", "8199", "-ngl", "999", "-c", "4096",
             "--flash-attn"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for server to be ready
        import urllib.request
        for _ in range(60):
            try:
                urllib.request.urlopen("http://localhost:8199/health", timeout=1)
                return
            except:
                time.sleep(1)
        raise RuntimeError("llama-server failed to start")

    def generate(self, prompt):
        import json, urllib.request
        body = json.dumps({
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Be concise and clear. Answer in 2-4 sentences unless more detail is needed."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": 0.4,
            "max_tokens": 512,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:8199/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        tokens = 0
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                            tokens += 1
                    except:
                        pass

        elapsed = time.time() - start
        self.total_tokens += tokens
        self.total_time += elapsed
        yield None
        self._last_tokens = tokens
        self._last_time = elapsed

    def stats(self):
        avg = self.total_tokens / self.total_time if self.total_time > 0 else 0
        return {
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_speed": avg,
        }

    def __del__(self):
        if hasattr(self, 'proc') and self.proc:
            self.proc.terminate()


# ── Demo Questions ─────────────────────────────────────

DEMO_QUESTIONS = [
    "What makes mixture-of-experts models more efficient than dense transformers?",
    "Explain how an SSD can be used as extended memory for neural network inference.",
    "Write a Python function that implements an LRU cache in 10 lines.",
    "What are the key differences between Apple's M2 and M4 chips?",
    "Explain the halting problem to a 10 year old.",
]


# ── Main Loop ──────────────────────────────────────────

def run_interactive(backend, mode):
    banner(mode)

    print(f"  {DIM}Loading model...{RESET}", end="", flush=True)
    t0 = time.time()
    result = backend.load()
    print(f" {GREEN}done{RESET} {DIM}({time.time()-t0:.1f}s){RESET}")

    if mode == "stream" and result:
        print(f"  {DIM}Pinned: {result:.1f} GB{RESET}")
    print()

    # Show suggested questions
    print(f"  {DIM}Try asking:{RESET}")
    for i, q in enumerate(DEMO_QUESTIONS[:3]):
        print(f"  {DIM}  {i+1}. {q}{RESET}")
    print()

    while True:
        try:
            user = input(f"  {BOLD}{YELLOW}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}goodbye.{RESET}\n")
            break

        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break

        if user in ("/stats", "/s"):
            s = backend.stats()
            print(f"\n  {CYAN}Tokens{RESET}  {s['total_tokens']:,}")
            print(f"  {CYAN}Time{RESET}    {s['total_time']:.1f}s")
            print(f"  {CYAN}Speed{RESET}   {s['avg_speed']:.2f} tok/s")
            if "memory_gb" in s:
                print(f"  {CYAN}Memory{RESET}  {s['memory_gb']:.2f} GB active, {s['peak_gb']:.2f} GB peak")
            print()
            continue

        # Allow numbered shortcuts for demo questions
        if user.isdigit() and 1 <= int(user) <= len(DEMO_QUESTIONS):
            user = DEMO_QUESTIONS[int(user) - 1]
            print(f"  {DIM}{user}{RESET}")

        print()
        print(f"  ", end="", flush=True)

        last_tokens = 0
        last_time = 0
        for chunk in backend.generate(user):
            if chunk is None:
                last_tokens = backend._last_tokens
                last_time = backend._last_time
                break
            print(chunk, end="", flush=True)

        speed = last_tokens / last_time if last_time > 0 else 0
        color = GREEN if speed > 1 else YELLOW if speed > 0.1 else RED
        print(f"\n\n  {color}{BOLD}{speed:.2f} tok/s{RESET}  {DIM}{last_tokens} tokens in {last_time:.1f}s{RESET}")
        print()


def main():
    parser = argparse.ArgumentParser(description="MoE Sniper Interactive Demo")
    parser.add_argument("--llama", action="store_true", help="Use llama.cpp backend (IQ2_M)")
    parser.add_argument("--model", type=str, help="Path to GGUF model for llama.cpp")
    args = parser.parse_args()

    if args.llama:
        model_path = args.model or "/Volumes/USB DISK/gguf/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
        if not os.path.exists(model_path):
            print(f"  {RED}GGUF not found: {model_path}{RESET}")
            print(f"  {DIM}Download it first or use --model to specify path{RESET}")
            sys.exit(1)
        backend = LlamaBackend(model_path)
        run_interactive(backend, "llama")
    else:
        backend = StreamingBackend()
        run_interactive(backend, "stream")


if __name__ == "__main__":
    main()
