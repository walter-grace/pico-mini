#!/usr/bin/env python3
"""
MoE Sniper Router — Run the agent UI locally, inference remotely.

Your MacBook runs the pretty agent. Your Mac Mini runs the heavy model.
No model files needed on the client — just HTTP to the remote server.

Usage:
  # On Mac Mini (server):
  ~/llama.cpp/build/bin/llama-server \
    -m ~/models/gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf \
    --mmproj ~/models/gguf/mmproj-F16.gguf \
    -ngl 1 -c 2048 --expert-cache-size 5000 \
    --reasoning off --port 8201 --host 0.0.0.0

  # On MacBook (client):
  python3 router.py --server your-server.local:8201
  python3 router.py --server 10.0.0.100:8201
"""

import os, sys, json, time, signal, subprocess, argparse, threading, ssl
import urllib.request, base64, glob
from datetime import datetime

# Allow HTTPS without certificate verification (for RunPod proxy etc)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Colors ─────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
C = "\033[96m"
Y = "\033[93m"
G = "\033[92m"
R = "\033[91m"
W = "\033[97m"
X = "\033[0m"

# ── Spinner ────────────────────────────────────────────

class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    def __init__(self, message=""):
        self.message = message
        self.running = False
        self.thread = None
        self.frame = 0
        self.start_time = 0
    def _spin(self):
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.FRAMES[self.frame % len(self.FRAMES)]
            sys.stdout.write(f"\r  {C}{frame}{X} {self.message} {D}{elapsed:.0f}s{X}  ")
            sys.stdout.flush()
            self.frame += 1
            time.sleep(0.1)
    def start(self, message=None):
        if message: self.message = message
        self.running = True
        self.start_time = time.time()
        self.frame = 0
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self
    def stop(self, final_message=None):
        self.running = False
        if self.thread: self.thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        sys.stdout.write("\r" + " " * 60 + "\r")
        if final_message:
            print(f"  {G}✓{X} {final_message} {D}({elapsed:.1f}s){X}")

# ── Remote Server ──────────────────────────────────────

class RemoteServer:
    def __init__(self, host):
        self.host = host
        if not host.startswith("http"):
            self.host = f"http://{host}"
        self.tok_total = 0
        self.time_total = 0.0

    def health(self):
        try:
            resp = urllib.request.urlopen(f"{self.host}/health", timeout=5, context=SSL_CTX)
            return json.load(resp).get("status") == "ok"
        except:
            return False

    def server_info(self):
        """Get model info from the remote server."""
        try:
            resp = urllib.request.urlopen(f"{self.host}/props", timeout=5, context=SSL_CTX)
            return json.load(resp)
        except:
            return {}

    def stream(self, messages, max_tokens=512, temperature=0.4):
        body = json.dumps({
            "messages": messages, "stream": True,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        start = time.time()
        tokens = 0
        try:
            with urllib.request.urlopen(req, timeout=600, context=SSL_CTX) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            delta = json.loads(line[6:])["choices"][0]["delta"]
                            text = delta.get("content", "") or delta.get("reasoning_content", "")
                            if text:
                                text = text.replace("<think>", "").replace("</think>", "")
                                if text.strip():
                                    tokens += 1
                                    yield text
                        except: pass
        except Exception as e:
            yield f"\n{R}Error: {e}{X}"
        elapsed = time.time() - start
        self.tok_total += tokens
        self.time_total += elapsed
        yield None
        self._last = (tokens, elapsed)

    def quick(self, messages, max_tokens=50):
        result = ""
        for chunk in self.stream(messages, max_tokens=max_tokens, temperature=0.0):
            if chunk is None: break
            result += chunk
        return result.strip()

# ── Intent + Search ────────────────────────────────────

SEARCH_KW = [
    "search", "find", "look up", "google", "what time", "when do",
    "when is", "who is", "who won", "weather", "news", "latest",
    "price", "stock", "score", "tonight", "today", "tomorrow",
]
SHELL_KW = ["list files", "show files", "disk space", "run ", "execute", "find files"]
SYSTEM = "You are a helpful AI assistant. Be concise and clear. Answer directly."

def classify(text):
    lower = text.lower()
    if any(k in lower for k in SHELL_KW): return "shell"
    if any(k in lower for k in SEARCH_KW): return "search"
    return "chat"

def web_search(query):
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r)
    except:
        return None
    return results if results else None

# ── Banner ─────────────────────────────────────────────

def banner(server_host, remote_info):
    ram = 0
    chip = "unknown"
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "hw.memsize"], text=True)
        ram = int(out.split(":")[1].strip()) / (1024**3)
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except: pass

    print()
    print(f"  {B}{C}  moe{X}{D}-{X}{B}{Y}sniper{X}  {D}remote{X}")
    print(f"  {D}  35B AI — local agent, remote inference{X}")
    print()
    print(f"  {D}{'─' * 52}{X}")
    print(f"  {B}{W}Server{X}    {server_host}")
    print(f"  {B}{W}Client{X}    {chip} ({ram:.0f} GB RAM)")
    print(f"  {B}{W}Tools{X}     chat, search, shell, vision")
    print(f"  {D}{'─' * 52}{X}")
    print()
    print(f"  {D}Commands:{X}")
    print(f"  {D}  /search <query>    web search + AI synthesis{X}")
    print(f"  {D}  /image <path>      describe an image (vision){X}")
    print(f"  {D}  /screenshot        capture + analyze your screen{X}")
    print(f"  {D}  /shell <task>      run a shell command{X}")
    print(f"  {D}  /stats             performance stats{X}")
    print(f"  {D}  /ping              check server status{X}")
    print(f"  {D}  /quit              exit{X}")
    print()

# ── Main ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Sniper Router — remote inference")
    parser.add_argument("--server", type=str, required=True,
                        help="Remote server address (e.g., your-server.local:8201)")
    args = parser.parse_args()

    server = RemoteServer(args.server)

    spin = Spinner()
    spin.start(f"Connecting to {args.server}...")
    if not server.health():
        spin.stop(f"{R}Cannot reach {args.server}{X}")
        print(f"  {D}Make sure llama-server is running on the remote machine:{X}")
        print(f"  {D}  llama-server -m model.gguf --host 0.0.0.0 --port 8201{X}\n")
        sys.exit(1)
    info = server.server_info()
    spin.stop(f"Connected to {args.server}")

    banner(args.server, info)

    signal.signal(signal.SIGINT, lambda s, f: (print(f"\n  {D}goodbye.{X}\n"), sys.exit(0)))

    conversation = []

    while True:
        try:
            user = input(f"  {B}{Y}>{X} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {D}goodbye.{X}\n")
            break

        if not user: continue
        if user in ("/quit", "/exit", "/q"): break

        if user == "/stats":
            avg = server.tok_total / server.time_total if server.time_total > 0 else 0
            print(f"\n  {C}Tokens{X}  {server.tok_total:,}")
            print(f"  {C}Time{X}    {server.time_total:.1f}s")
            print(f"  {C}Speed{X}   {avg:.2f} tok/s\n")
            continue

        if user == "/clear":
            conversation.clear()
            print(f"  {D}cleared{X}\n")
            continue

        if user == "/ping":
            spin = Spinner()
            spin.start("Pinging server...")
            ok = server.health()
            spin.stop(f"{'Server OK' if ok else R + 'Server unreachable' + X}")
            print()
            continue

        # Parse intent
        if user == "/screenshot":
            import tempfile
            tmp = tempfile.mktemp(suffix=".png")
            print(f"  {C}[screenshot]{X} capturing screen...")
            subprocess.run(["screencapture", "-x", tmp], capture_output=True)
            subprocess.run(["sips", "-z", "480", "640", tmp, "--out", tmp], capture_output=True)
            intent, user = "image", tmp
        elif user.startswith("/search "):
            intent, user = "search", user[8:]
        elif user.startswith("/shell "):
            intent, user = "shell", user[7:]
        elif user.startswith("/image "):
            intent, user = "image", user[7:]
        else:
            intent = classify(user)

        print()
        start = time.time()
        tokens = 0

        # ── Search ──
        if intent == "search":
            today = datetime.now().strftime("%A, %B %d, %Y")
            spin = Spinner()
            spin.start(f"Searching: {user[:40]}...")
            results = web_search(user)
            if results:
                spin.stop(f"Found {len(results)} results")
                for r in results[:5]:
                    print(f"  {D}  {r['title'][:70]}{X}")
                context = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
                print()
                spin2 = Spinner()
                spin2.start("Remote 35B model synthesizing...")
                messages = [
                    {"role": "system", "content": f"{SYSTEM}\nToday is {today}. Answer using these search results:\n{context}"},
                    {"role": "user", "content": user},
                ]
                first = True
                for chunk in server.stream(messages):
                    if chunk is None: break
                    if first:
                        spin2.stop("Generating")
                        print(f"\n  ", end="", flush=True)
                        first = False
                    print(chunk, end="", flush=True)
                    tokens += 1
            else:
                spin.stop(f"{Y}No results{X}")
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                print(f"\n  ", end="", flush=True)
                for chunk in server.stream(messages):
                    if chunk is None: break
                    print(chunk, end="", flush=True)
                    tokens += 1

        # ── Shell ──
        elif intent == "shell":
            spin = Spinner()
            spin.start("Generating command...")
            cmd = server.quick([
                {"role": "system", "content": "Generate a single macOS shell command. Output ONLY the command."},
                {"role": "user", "content": user},
            ])
            if cmd:
                spin.stop(f"$ {cmd}")
                spin2 = Spinner()
                spin2.start("Executing locally...")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    output = (result.stdout + result.stderr)[:4000]
                except Exception as e:
                    output = str(e)
                spin2.stop("Done")
                for line in output.strip().split("\n")[:8]:
                    print(f"  {D}  {line[:100]}{X}")
                print()
                spin3 = Spinner()
                spin3.start("Summarizing...")
                messages = [
                    {"role": "system", "content": f"{SYSTEM}\nPresent these results clearly."},
                    {"role": "user", "content": f"$ {cmd}\n{output}\n\nQuestion: {user}"},
                ]
                first = True
                for chunk in server.stream(messages):
                    if chunk is None: break
                    if first:
                        spin3.stop("Generating")
                        print(f"\n  ", end="", flush=True)
                        first = False
                    print(chunk, end="", flush=True)
                    tokens += 1
            else:
                spin.stop(f"{R}Failed{X}")

        # ── Vision ──
        elif intent == "image":
            img_path = os.path.expanduser(user.strip().strip("'\""))
            if not os.path.exists(img_path):
                search_name = os.path.basename(user.strip())
                matches = (
                    glob.glob(img_path) or
                    glob.glob(os.path.expanduser(f"~/Desktop/{search_name}")) or
                    glob.glob(os.path.expanduser(f"~/Desktop/*{search_name}*")) or
                    glob.glob(os.path.expanduser("~/Desktop/*.png"))
                )
                if matches:
                    img_path = matches[0]
                    print(f"  {D}Found: {os.path.basename(img_path)}{X}")
                else:
                    img_path = None
            if not img_path or not os.path.exists(img_path):
                print(f"  {R}Image not found: {user.strip()}{X}\n")
                continue

            size_kb = os.path.getsize(img_path) // 1024
            spin = Spinner()
            spin.start(f"Sending {os.path.basename(img_path)} ({size_kb} KB) to remote model...")

            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            ext = img_path.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")

            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": "Describe what you see in this image in detail."}
            ]}]
            first = True
            for chunk in server.stream(messages, max_tokens=200):
                if chunk is None: break
                if first:
                    spin.stop(f"Image received by server")
                    print(f"\n  ", end="", flush=True)
                    first = False
                print(chunk, end="", flush=True)
                tokens += 1

        # ── Chat ──
        else:
            conversation.append({"role": "user", "content": user})
            messages = [{"role": "system", "content": SYSTEM}] + conversation[-10:]
            spin = Spinner()
            spin.start("Remote model thinking...")
            first = True
            response = ""
            for chunk in server.stream(messages):
                if chunk is None: break
                if first:
                    spin.stop("Generating")
                    print(f"\n  ", end="", flush=True)
                    first = False
                print(chunk, end="", flush=True)
                response += chunk
                tokens += 1
            conversation.append({"role": "assistant", "content": response})

        elapsed = time.time() - start
        if tokens > 0:
            speed = tokens / elapsed
            color = G if speed > 5 else Y if speed > 0.5 else R
            print(f"\n\n  {color}{B}{speed:.2f} tok/s{X}  {D}{tokens} tokens in {elapsed:.1f}s{X}")
        print()

if __name__ == "__main__":
    main()
