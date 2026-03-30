#!/usr/bin/env python3
"""
MoE Sniper Agent — Interactive shell for Qwen3.5-35B-A3B on low-RAM machines.
Uses our custom llama.cpp build with expert-aware LRU cache.

Supports:
  - Chat with a 35B model on 8-16 GB machines
  - Web search via DuckDuckGo
  - Shell command execution
  - Model switching (Q4_K_M for quality, IQ2_M for speed)
  - Expert cache stats

Usage:
  python3 agent.py                          # auto-detect best model
  python3 agent.py --model q4               # force Q4_K_M (21 GB, needs cache)
  python3 agent.py --model iq2              # force IQ2_M (10.6 GB, fits in 16 GB)
  python3 agent.py --cache 3000             # expert cache size in MB
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import urllib.request
from datetime import datetime

# ── Config ─────────────────────────────────────────

LLAMA_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODEL_DIR = "/Volumes/USB DISK/gguf"
MMPROJ = "mmproj-F16.gguf"
MODELS = {
    "q4":  "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "iq2": "Qwen3.5-35B-A3B-UD-IQ2_M.gguf",
    "9b":  "Qwen3.5-9B-Q4_K_M.gguf",
}
PORT = 8199
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and clear. Answer directly without thinking step by step."

# ── Colors ─────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# ── Intent Classification ──────────────────────────

SEARCH_KEYWORDS = [
    "search", "find", "look up", "google", "what time", "when do",
    "when is", "who is", "who won", "weather", "news", "latest",
    "price", "stock", "score", "tonight", "today", "tomorrow",
]
SHELL_KEYWORDS = [
    "list files", "show files", "disk space", "run ",
    "execute", "find files", "read file", "create file",
]

def classify_intent(text):
    lower = text.lower()
    if any(k in lower for k in SHELL_KEYWORDS):
        return "shell"
    if any(k in lower for k in SEARCH_KEYWORDS):
        return "search"
    return "chat"

# ── LLM Backend (llama-server) ─────────────────────

class LlamaServer:
    def __init__(self, model_path, cache_mb=0, ngl=0, ctx=2048):
        self.model_path = model_path
        self.cache_mb = cache_mb
        self.ngl = ngl
        self.ctx = ctx
        self.proc = None
        self.total_tokens = 0
        self.total_time = 0.0

    def start(self):
        cmd = [
            LLAMA_BIN,
            "-m", self.model_path,
            "--port", str(PORT),
            "-ngl", str(self.ngl),
            "-c", str(self.ctx),
            "--no-warmup",
        ]
        if self.cache_mb > 0:
            cmd += ["--expert-cache-size", str(self.cache_mb)]

        # Enable vision if mmproj exists
        mmproj_path = os.path.join(MODEL_DIR, MMPROJ)
        if os.path.exists(mmproj_path):
            cmd += ["--mmproj", mmproj_path]

        # Disable reasoning/thinking to get direct answers (faster)
        cmd += ["--reasoning", "off"]

        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Wait for server to be ready
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1)
                return True
            except:
                time.sleep(1)
        return False

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
            self.proc = None

    def chat(self, messages, max_tokens=512, temperature=0.4, stream=True):
        body = json.dumps({
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        tokens = 0
        full_text = ""

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0]["delta"]
                            # Try content first, fall back to reasoning_content
                            text = delta.get("content", "") or delta.get("reasoning_content", "")
                            if text:
                                # Skip think tags
                                if "<think>" in text or "</think>" in text:
                                    text = text.replace("<think>", "").replace("</think>", "").strip()
                                    if not text:
                                        continue
                                yield text
                                full_text += text
                                tokens += 1
                        except:
                            pass
        except Exception as e:
            yield f"\n{RED}Error: {e}{RESET}"

        elapsed = time.time() - start
        self.total_tokens += tokens
        self.total_time += elapsed

    def quick_call(self, messages, max_tokens=50):
        result = ""
        for chunk in self.chat(messages, max_tokens=max_tokens, temperature=0.0):
            result += chunk
        return result.strip()

# ── Tools ──────────────────────────────────────────

def web_search(server, query):
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None, None

    # Use raw query directly — skip LLM rewrite to save 10-30s
    search_query = query

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=5):
                results.append(f"- {r['title']}: {r['body']}")
    except:
        return None, None

    if not results:
        return None, None

    return "\n".join(results), search_query

def shell_exec(server, query):
    cmd = server.quick_call([
        {"role": "system", "content": f"Generate a single macOS shell command. Output ONLY the command, nothing else."},
        {"role": "user", "content": query},
    ])
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

# ── Main Loop ──────────────────────────────────────

def banner(model_name, cache_mb):
    print()
    print(f"  {BOLD}{CYAN}moe{RESET}{DIM}-{RESET}{BOLD}{YELLOW}sniper{RESET}  {DIM}agent{RESET}")
    print()
    print(f"  {BOLD}Model{RESET}     {model_name}")
    print(f"  {BOLD}Cache{RESET}     {cache_mb} MB expert LRU cache")
    print(f"  {BOLD}Engine{RESET}    llama.cpp + expert-aware cache")
    mmproj_path = os.path.join(MODEL_DIR, MMPROJ)
    has_vision = os.path.exists(mmproj_path)
    print(f"  {BOLD}Tools{RESET}     chat, search, shell{', vision' if has_vision else ''}")
    if has_vision:
        print(f"  {BOLD}Vision{RESET}    {MMPROJ} ({os.path.getsize(mmproj_path) // (1024*1024)} MB)")
    print()
    print(f"  {DIM}Commands: /search, /shell, /image <path>, /stats, /quit{RESET}")
    print(f"  {DIM}{'─' * 50}{RESET}")
    print()

def main():
    parser = argparse.ArgumentParser(description="MoE Sniper Agent")
    parser.add_argument("--model", choices=["q4", "iq2", "9b"], default="iq2",
                        help="Model variant (default: iq2)")
    parser.add_argument("--cache", type=int, default=3000,
                        help="Expert cache size in MB (default: 3000)")
    parser.add_argument("--ngl", type=int, default=0,
                        help="GPU layers (default: 0)")
    parser.add_argument("--ctx", type=int, default=2048,
                        help="Context size (default: 2048)")
    args = parser.parse_args()

    model_file = MODELS[args.model]
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        print(f"  {RED}Model not found: {model_path}{RESET}")
        sys.exit(1)

    if not os.path.exists(LLAMA_BIN):
        print(f"  {RED}llama-server not found: {LLAMA_BIN}{RESET}")
        print(f"  {DIM}Build with: cd ~/llama.cpp && cmake --build build --target llama-server{RESET}")
        sys.exit(1)

    banner(model_file, args.cache)

    print(f"  {DIM}Starting server...{RESET}", end="", flush=True)
    server = LlamaServer(model_path, cache_mb=args.cache, ngl=args.ngl, ctx=args.ctx)

    # Handle Ctrl+C gracefully
    def cleanup(sig, frame):
        print(f"\n  {DIM}shutting down...{RESET}")
        server.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if not server.start():
        print(f" {RED}failed{RESET}")
        print(f"  {DIM}Server didn't start. Try reducing --ctx or --cache.{RESET}")
        server.stop()
        sys.exit(1)

    print(f" {GREEN}ready{RESET}")
    print()

    conversation = []

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

        if user == "/stats":
            avg = server.total_tokens / server.total_time if server.total_time > 0 else 0
            print(f"\n  {CYAN}Tokens{RESET}  {server.total_tokens:,}")
            print(f"  {CYAN}Time{RESET}    {server.total_time:.1f}s")
            print(f"  {CYAN}Speed{RESET}   {avg:.2f} tok/s")
            print()
            continue

        if user == "/clear":
            conversation.clear()
            print(f"  {DIM}conversation cleared{RESET}\n")
            continue

        # Force intent with slash commands
        if user.startswith("/search "):
            intent = "search"
            user = user[8:]
        elif user.startswith("/shell "):
            intent = "shell"
            user = user[7:]
        elif user.startswith("/image "):
            intent = "image"
            user = user[7:]
        else:
            intent = classify_intent(user)

        print()
        start = time.time()
        tokens = 0

        if intent == "search":
            print(f"  {CYAN}[intent]{RESET} search")
            print(f"  {CYAN}[rewriting query...]{RESET}", flush=True)
            context, query = web_search(server, user)
            if context:
                print(f"  {CYAN}[query]{RESET} {query}")
                print(f"  {CYAN}[results]{RESET}")
                for line in context.split("\n")[:5]:
                    print(f"  {DIM}  {line[:100]}{RESET}")
                print(f"  {CYAN}[synthesizing answer...]{RESET}\n")
                messages = [
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\nToday is {datetime.now().strftime('%A, %B %d, %Y')}. Answer using these search results:\n{context}"},
                    {"role": "user", "content": user},
                ]
            else:
                print(f"  {YELLOW}[search unavailable, answering directly]{RESET}\n")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ]
            print(f"  ", end="", flush=True)
            for chunk in server.chat(messages):
                print(chunk, end="", flush=True)
                tokens += 1

        elif intent == "shell":
            print(f"  {CYAN}[intent]{RESET} shell")
            print(f"  {CYAN}[generating command...]{RESET}", flush=True)
            cmd, output = shell_exec(server, user)
            if cmd:
                print(f"  {CYAN}[command]{RESET} $ {cmd}")
                if output.strip():
                    for line in output.strip().split("\n")[:10]:
                        print(f"  {DIM}  {line[:120]}{RESET}")
                print(f"  {CYAN}[summarizing...]{RESET}\n")
                messages = [
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\nPresent these shell results clearly."},
                    {"role": "user", "content": f"Command: {cmd}\nOutput:\n{output}\n\nOriginal question: {user}"},
                ]
                print(f"  ", end="", flush=True)
                for chunk in server.chat(messages):
                    print(chunk, end="", flush=True)
                    tokens += 1

        elif intent == "image":
            # Vision: analyze an image
            import base64, glob
            img_path = os.path.expanduser(user.strip())
            if not os.path.exists(img_path):
                # Try glob
                matches = glob.glob(img_path)
                if matches:
                    img_path = matches[0]
                else:
                    print(f"  {RED}Image not found: {user}{RESET}\n")
                    continue

            print(f"  {CYAN}[intent]{RESET} vision")
            print(f"  {CYAN}[image]{RESET} {os.path.basename(img_path)} ({os.path.getsize(img_path) // 1024} KB)")
            print(f"  {CYAN}[sending to 35B vision model...]{RESET}")

            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Describe what you see in this image in detail."}
            ]}]
            print(f"\n  ", end="", flush=True)
            for chunk in server.chat(messages, max_tokens=200):
                print(chunk, end="", flush=True)
                tokens += 1

        else:
            # Regular chat
            print(f"  {CYAN}[intent]{RESET} chat")
            conversation.append({"role": "user", "content": user})
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-10:]
            print(f"  ", end="", flush=True)
            response = ""
            for chunk in server.chat(messages):
                print(chunk, end="", flush=True)
                response += chunk
                tokens += 1
            conversation.append({"role": "assistant", "content": response})

        elapsed = time.time() - start
        if tokens > 0:
            speed = tokens / elapsed
            color = GREEN if speed > 1 else YELLOW if speed > 0.1 else RED
            print(f"\n\n  {color}{BOLD}{speed:.2f} tok/s{RESET}  {DIM}{tokens} tokens in {elapsed:.1f}s{RESET}")
        print()

    server.stop()

if __name__ == "__main__":
    main()
