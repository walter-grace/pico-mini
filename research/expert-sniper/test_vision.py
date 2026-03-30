#!/usr/bin/env python3
"""Test the 35B multimodal model with an image via llama-server."""
import sys, json, urllib.request, time, base64, os, glob

# Find an image to test with
img_path = sys.argv[1] if len(sys.argv) > 1 else None

if not img_path:
    # Auto-find a screenshot on Desktop
    screenshots = glob.glob(os.path.expanduser("~/Desktop/*.png"))
    if screenshots:
        img_path = screenshots[0]
    else:
        print("Usage: python3 test_vision.py <image_path>")
        print("No images found on Desktop")
        sys.exit(1)

with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

print(f"[image] {os.path.basename(img_path)}")
print(f"[size] {len(img_b64) // 1024} KB base64")
print(f"[sending to 35B vision model on USB drive...]")
print(flush=True)

body = json.dumps({
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "What do you see in this image? Describe it briefly."}
        ]}
    ],
    "temperature": 0.4,
    "max_tokens": 100,
    "stream": True,
}).encode()

req = urllib.request.Request(
    "http://localhost:8199/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)

start = time.time()
tokens = 0
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    text = delta.get("content", "") or delta.get("reasoning_content", "")
                    if text:
                        text = text.replace("<think>", "").replace("</think>", "")
                        if text.strip():
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            tokens += 1
                except:
                    pass
except Exception as e:
    print(f"\nError: {e}")

elapsed = time.time() - start
speed = tokens / elapsed if elapsed > 0 else 0
print(f"\n\n--- {tokens} tokens | {elapsed:.1f}s | {speed:.2f} tok/s ---")
print(f"--- 35B multimodal MoE on USB drive, 8 GB RAM ---")
