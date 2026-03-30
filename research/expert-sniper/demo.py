#!/usr/bin/env python3
"""
Demo: Qwen3.5-35B-A3B running on 8 GB RAM via MoE expert streaming.
This model requires 22 GB — llama.cpp OOMs on this hardware.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from moe_agent_macbook import MoESniperEngine
import mlx.core as mx

print("\n" + "="*60)
print("  MoE SNIPER — 35B model on 8 GB RAM")
print("  Qwen3.5-35B-A3B (22 GB) via expert streaming from SSD")
print("="*60)

print("\n  Loading model...")
t0 = time.time()
engine = MoESniperEngine()
pinned_gb = engine.load()
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Pinned memory: {pinned_gb:.1f} GB")
print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")
print(f"  Peak memory:   {mx.get_peak_memory()/1e9:.2f} GB")

prompts = [
    "Explain in two sentences why mixture-of-experts models are efficient.",
]

for prompt in prompts:
    print(f"\n  Prompt: {prompt}")
    print(f"  {'—'*50}")
    print("  ", end="", flush=True)

    messages = [
        {"role": "system", "content": "Be concise. Answer in 2-3 sentences max. Do not show your thinking process."},
        {"role": "user", "content": prompt},
    ]

    start = time.time()
    tokens = 0
    for chunk in engine.generate(messages, temperature=0.3):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        tokens += 1

    elapsed = time.time() - start
    speed = tokens / elapsed if elapsed > 0 else 0

    print(f"\n\n  Tokens: {tokens} | Time: {elapsed:.1f}s | Speed: {speed:.2f} tok/s")
    print(f"  Memory: {mx.get_active_memory()/1e9:.2f} GB active, {mx.get_peak_memory()/1e9:.2f} GB peak")

print(f"\n  {'='*50}")
print(f"  35B model. 8 GB RAM. No OOM. That's the point.")
print(f"  {'='*50}\n")
