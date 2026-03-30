#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  MoE SNIPER — 35B AI on 8 GB RAM                        ║
║  Text • Vision • Web Search • Shell                      ║
║  github.com/walter-grace/mac-code                        ║
╚══════════════════════════════════════════════════════════╝

A 35B multimodal model running on hardware that shouldn't support it.
Models stored on USB drive or internal SSD. Expert-aware LRU cache
prevents mmap thrashing. Vision encoder describes images.

Usage:
  python3 sniper.py                     # auto-detect model
  python3 sniper.py --model q4          # 21 GB Q4_K_M (needs cache)
  python3 sniper.py --model iq2         # 10.6 GB IQ2_M
  python3 sniper.py --cache 3000        # 3 GB expert cache
  python3 sniper.py --model-dir /path   # custom model directory
"""

import os, sys, json, time, signal, subprocess, argparse, threading
import urllib.request, base64, glob
from datetime import datetime

# ── Config ─────────────────────────────────────────────

LLAMA_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
DEFAULT_DIRS = [
    "/Volumes/USB DISK/gguf",
    os.path.expanduser("~/models/gguf"),
    os.path.expanduser("~/models"),
]
MMPROJ = "mmproj-F16.gguf"
MODELS = {
    "q4":  "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "iq2": "Qwen3.5-35B-A3B-UD-IQ2_M.gguf",
    "9b":  "Qwen3.5-9B-Q4_K_M.gguf",
}
PORT = 8199
SYSTEM = "You are a helpful AI assistant. Be concise and clear. Answer directly."

# ── Colors ─────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
C = "\033[96m"
Y = "\033[93m"
G = "\033[92m"
R = "\033[91m"
W = "\033[97m"
X = "\033[0m"

# ── Spinner ─────────────────────────────────────────────

class Spinner:
    """Animated spinner that shows what's happening."""
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

    def update(self, message):
        self.message = message

    def stop(self, final_message=None):
        self.running = False
        if self.thread: self.thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        sys.stdout.write("\r" + " " * 60 + "\r")  # clear line
        if final_message:
            print(f"  {G}✓{X} {final_message} {D}({elapsed:.1f}s){X}")

# ── Utilities ──────────────────────────────────────────

def find_model_dir():
    for d in DEFAULT_DIRS:
        if os.path.isdir(d):
            for m in MODELS.values():
                if os.path.exists(os.path.join(d, m)):
                    return d
    return None

def get_ram_gb():
    try:
        out = subprocess.check_output(["sysctl", "hw.memsize"], text=True)
        return int(out.split(":")[1].strip()) / (1024**3)
    except:
        return 0

def get_chip():
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        return out
    except:
        return "Apple Silicon"

def get_storage_type(path):
    if "/Volumes/USB" in path:
        return "USB Drive"
    elif "/Volumes/" in path:
        return "External Drive"
    return "Internal SSD"

# ── Search ─────────────────────────────────────────────

SEARCH_KW = [
    "search", "find", "look up", "google", "what time", "when do",
    "when is", "who is", "who won", "weather", "news", "latest",
    "price", "stock", "score", "tonight", "today", "tomorrow",
]
SHELL_KW = ["list files", "show files", "disk space", "run ", "execute", "find files"]

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

# ── LLM Server ─────────────────────────────────────────

class Server:
    def __init__(self, model_path, mmproj_path=None, cache_mb=0, ngl=0, ctx=2048):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.cache_mb = cache_mb
        self.ngl = ngl
        self.ctx = ctx
        self.proc = None
        self.tok_total = 0
        self.time_total = 0.0

    def start(self):
        cmd = [LLAMA_BIN, "-m", self.model_path, "--port", str(PORT),
               "-ngl", str(self.ngl), "-c", str(self.ctx), "--no-warmup"]
        if self.cache_mb > 0:
            cmd += ["--expert-cache-size", str(self.cache_mb)]
        if self.mmproj_path and os.path.exists(self.mmproj_path):
            cmd += ["--mmproj", self.mmproj_path]
        cmd += ["--reasoning", "off"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1)
                return True
            except:
                time.sleep(1)
        return False

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try: self.proc.wait(timeout=5)
            except: self.proc.kill()
            self.proc = None

    def stream(self, messages, max_tokens=512, temperature=0.4):
        body = json.dumps({
            "messages": messages, "stream": True,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        start = time.time()
        tokens = 0
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
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
        yield None  # signal done
        self._last = (tokens, elapsed)

    def quick(self, messages, max_tokens=50):
        result = ""
        for chunk in self.stream(messages, max_tokens=max_tokens, temperature=0.0):
            if chunk is None: break
            result += chunk
        return result.strip()

# ── Banner ─────────────────────────────────────────────

def banner(model_name, model_dir, cache_mb, has_vision):
    ram = get_ram_gb()
    chip = get_chip()
    storage = get_storage_type(model_dir)
    model_size = 0
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024**3)

    print()
    print(f"  {B}{C}  moe{X}{D}-{X}{B}{Y}sniper{X}")
    print(f"  {D}  35B multimodal AI on consumer hardware{X}")
    print()
    print(f"  {D}{'─' * 52}{X}")
    print(f"  {B}{W}Model{X}     {model_name}")
    print(f"  {B}{W}Size{X}      {model_size:.1f} GB {D}({storage}){X}")
    print(f"  {B}{W}Hardware{X}  {chip} {D}({ram:.0f} GB RAM){X}")
    print(f"  {B}{W}Cache{X}     {cache_mb} MB expert LRU cache")
    tools = "chat, search, shell"
    if has_vision:
        tools += f", {G}vision{X}"
    print(f"  {B}{W}Tools{X}     {tools}")
    print(f"  {D}{'─' * 52}{X}")
    print()
    if model_size > ram:
        print(f"  {G}{B}  Model is {model_size/ram:.1f}x larger than RAM.{X}")
        print(f"  {G}{B}  Expert cache makes this possible.{X}")
        print()
    print(f"  {D}Commands:{X}")
    print(f"  {D}  /search <query>    web search + AI synthesis{X}")
    print(f"  {D}  /image <path>      describe an image (vision){X}")
    print(f"  {D}  /screenshot        capture + analyze your screen{X}")
    print(f"  {D}  /shell <task>      run a shell command{X}")
    print(f"  {D}  /stats             show performance stats{X}")
    print(f"  {D}  /quit              exit{X}")
    print()

# ── Main ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Sniper — 35B AI Agent")
    parser.add_argument("--model", choices=["q4", "iq2", "9b"], default="iq2")
    parser.add_argument("--cache", type=int, default=3000)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir or find_model_dir()
    if not model_dir:
        print(f"  {R}No model directory found. Use --model-dir{X}")
        sys.exit(1)

    model_file = MODELS[args.model]
    model_path = os.path.join(model_dir, model_file)
    mmproj_path = os.path.join(model_dir, MMPROJ)
    has_vision = os.path.exists(mmproj_path)

    if not os.path.exists(model_path):
        print(f"  {R}Model not found: {model_path}{X}")
        sys.exit(1)
    if not os.path.exists(LLAMA_BIN):
        print(f"  {R}llama-server not found at {LLAMA_BIN}{X}")
        sys.exit(1)

    banner(model_file, model_dir, args.cache, has_vision)

    server = Server(model_path, mmproj_path, args.cache, args.ngl, args.ctx)

    signal.signal(signal.SIGINT, lambda s, f: (print(f"\n  {D}goodbye.{X}\n"), server.stop(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (server.stop(), sys.exit(0)))

    spin = Spinner()
    spin.start("Loading model weights...")
    if not server.start():
        spin.stop(f"{R}Engine failed to start{X}")
        server.stop()
        sys.exit(1)
    spin.stop(f"Engine ready — 35B model loaded")
    print()

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
            print(f"  {C}Speed{X}   {avg:.2f} tok/s")
            print()
            continue

        if user == "/clear":
            conversation.clear()
            print(f"  {D}cleared{X}\n")
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
                spin2.start("35B model synthesizing answer...")
                # Get first token to stop spinner
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
                spin.stop(f"{Y}No results, answering directly{X}")
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                print(f"\n  ", end="", flush=True)
                for chunk in server.stream(messages):
                    if chunk is None: break
                    print(chunk, end="", flush=True)
                    tokens += 1

        # ── Shell ──
        elif intent == "shell":
            spin = Spinner()
            spin.start("Generating shell command...")
            cmd = server.quick([
                {"role": "system", "content": "Generate a single macOS shell command. Output ONLY the command."},
                {"role": "user", "content": user},
            ])
            if cmd:
                spin.stop(f"Command: $ {cmd}")
                spin2 = Spinner()
                spin2.start("Executing...")
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
                spin3.start("Summarizing results...")
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
                spin.stop(f"{R}Could not generate command{X}")

        # ── Vision ──
        elif intent == "image":
            img_path = os.path.expanduser(user.strip().strip("'\""))
            if not os.path.exists(img_path):
                # Try common locations and partial matches
                search_name = os.path.basename(user.strip())
                matches = (
                    glob.glob(img_path) or
                    glob.glob(os.path.expanduser(f"~/Desktop/{search_name}")) or
                    glob.glob(os.path.expanduser(f"~/Desktop/*{search_name}*")) or
                    glob.glob(os.path.expanduser(f"~/Downloads/{search_name}")) or
                    glob.glob(os.path.expanduser("~/Desktop/*.png")) or
                    glob.glob(os.path.expanduser("~/Desktop/*.jpg"))
                )
                if matches:
                    img_path = matches[0]
                    print(f"  {D}Found: {os.path.basename(img_path)}{X}")
                else:
                    img_path = None
            if not img_path or not os.path.exists(img_path):
                print(f"  {R}Image not found: {user.strip()}{X}")
                print(f"  {D}Tip: use full path like /Users/you/Desktop/photo.png{X}\n")
                continue
            if not has_vision:
                print(f"  {R}Vision not available (mmproj-F16.gguf not found in model dir){X}\n")
                continue

            size_kb = os.path.getsize(img_path) // 1024
            spin = Spinner()
            spin.start(f"Analyzing {os.path.basename(img_path)} ({size_kb} KB)...")

            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            ext = img_path.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")

            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": "Describe what you see in this image in detail."}
            ]}]
            first = True
            for chunk in server.stream(messages, max_tokens=200):
                if chunk is None: break
                if first:
                    spin.stop(f"Image processed ({size_kb} KB)")
                    print(f"\n  ", end="", flush=True)
                    first = False
                print(chunk, end="", flush=True)
                tokens += 1

        # ── Chat ──
        else:
            conversation.append({"role": "user", "content": user})
            messages = [{"role": "system", "content": SYSTEM}] + conversation[-10:]
            spin = Spinner()
            spin.start("Thinking...")
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

    server.stop()

if __name__ == "__main__":
    main()
