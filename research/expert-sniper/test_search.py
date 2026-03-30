#!/usr/bin/env python3
"""Test the web search pipeline with llama-server."""
import sys, json, urllib.request, time
from ddgs import DDGS
from datetime import datetime

today = datetime.now().strftime("%A, %B %d, %Y")
query = f"SpaceX latest news {datetime.now().strftime('%B %Y')}"

print(f"[intent] search")
print(f"[date] {today}")
print(f"[query] {query}")
print(f"[searching DuckDuckGo...]")

results = []
with DDGS() as ddgs:
    for r in ddgs.text(query, max_results=5):
        results.append(f"- {r['title']}: {r['body']}")
        print(f"  >> {r['title'][:80]}")

context = "\n".join(results)
print(f"[found {len(results)} results]")
print(f"[sending to 35B model for synthesis...]")
print(flush=True)

body = json.dumps({
    "messages": [
        {"role": "system", "content": f"Be concise and helpful. Today is {today}. Answer using these search results:\n{context}"},
        {"role": "user", "content": "What is the latest SpaceX news?"},
    ],
    "temperature": 0.4,
    "max_tokens": 150,
    "stream": True,
}).encode()

req = urllib.request.Request(
    "http://localhost:8199/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)

start = time.time()
tokens = 0
with urllib.request.urlopen(req, timeout=300) as resp:
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

elapsed = time.time() - start
speed = tokens / elapsed if elapsed > 0 else 0
print(f"\n\n--- {tokens} tokens | {elapsed:.1f}s | {speed:.2f} tok/s ---")
