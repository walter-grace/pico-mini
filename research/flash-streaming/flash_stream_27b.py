"""
Flash Stream for Qwen3.5-27B Dense — Proving the method scales.

16.1 GB model on 16 GB Mac. Pinned attention+SSM in RAM,
FFN streamed from SSD per token. Same architecture as 32B proof,
now applied to a different model family.

Architecture: 64 layers, hybrid 3:1 linear_attention:full_attention
FFN: standard dense SwiGLU (no MoE)
"""

import time
import os
import sys
import json
import gc

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

MODEL_DIR = "/Users/bigneek/models/qwen35-27b-flash-stream"
BITS = 4
GROUP_SIZE = 64


def run_ffn(x, ffn_data):
    """SwiGLU FFN with quantized matmul from streamed weights."""
    gate = mx.quantized_matmul(
        x, ffn_data["mlp.gate_proj.weight"],
        scales=ffn_data["mlp.gate_proj.scales"],
        biases=ffn_data["mlp.gate_proj.biases"],
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    up = mx.quantized_matmul(
        x, ffn_data["mlp.up_proj.weight"],
        scales=ffn_data["mlp.up_proj.scales"],
        biases=ffn_data["mlp.up_proj.biases"],
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    hidden = nn.silu(gate) * up
    out = mx.quantized_matmul(
        hidden, ffn_data["mlp.down_proj.weight"],
        scales=ffn_data["mlp.down_proj.scales"],
        biases=ffn_data["mlp.down_proj.biases"],
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    return out


def main():
    print("=" * 60)
    print("  FLASH STREAM — Qwen3.5-27B Dense")
    print("  16.1 GB model · 16 GB Mac · FFN from SSD")
    print("=" * 60)

    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from mlx_lm.models.switch_layers import SwitchLinear

    args = TextModelArgs(
        model_type=config.get("model_type"),
        hidden_size=config["hidden_size"],
        num_hidden_layers=num_layers,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        head_dim=config.get("head_dim"),
        tie_word_embeddings=config["tie_word_embeddings"],
        linear_num_value_heads=config.get("linear_num_value_heads"),
        linear_num_key_heads=config.get("linear_num_key_heads"),
        linear_key_head_dim=config.get("linear_key_head_dim"),
        linear_value_head_dim=config.get("linear_value_head_dim"),
        linear_conv_kernel_dim=config.get("linear_conv_kernel_dim"),
        full_attention_interval=config.get("full_attention_interval"),
        rope_parameters=config.get("rope_parameters"),
    )

    model = TextModel(args)

    # Quantize — protect conv1d only
    def should_quantize(path, module):
        if isinstance(module, nn.Embedding): return True
        if isinstance(module, SwitchLinear): return True
        if not isinstance(module, nn.Linear): return False
        if "conv1d" in path: return False
        if module.weight.shape[-1] < GROUP_SIZE: return False
        return True

    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS, class_predicate=should_quantize)

    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)

    print("\nLoading pinned weights...")
    t0 = time.time()
    pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
    model.load_weights(list(pinned.items()), strict=False)
    params = [p for name, p in tree_flatten(model.parameters()) if "mlp" not in name]
    mx.eval(*params)
    del pinned; gc.collect(); mx.clear_cache()

    pinned_gb = sum(p.nbytes for p in params) / 1e9
    print(f"  {pinned_gb:.2f} GB in {time.time()-t0:.1f}s")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")
    print(f"  Layers: {num_layers}")

    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

    cache = model.make_cache()
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    def forward_streaming(input_ids):
        h = model.model.embed_tokens(input_ids)
        fa_mask = create_attention_mask(h, cache[model.model.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[model.model.ssm_idx])

        for i in range(num_layers):
            layer = model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask

            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
            h = h + attn_out
            mx.eval(h)

            # FFN from SSD
            ffn_path = f"{MODEL_DIR}/ffn/layer_{i:02d}.safetensors"
            ffn_data = mx.load(ffn_path)

            normed = layer.post_attention_layernorm(h)
            ffn_out = run_ffn(normed, ffn_data)
            h = h + ffn_out
            mx.eval(h)

            del ffn_data, ffn_out, normed, attn_out
            mx.clear_cache()

        h = model.model.norm(h)
        return model.lm_head(h)

    # Test
    prompt = "What is the capital of France?"
    messages = [
        {"role": "system", "content": "Think briefly, answer directly."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    import subprocess
    subprocess.run(["sudo", "purge"], capture_output=True)

    print(f"\n--- Prefill ({len(tokens)} tokens) ---")
    t0 = time.time()
    logits = forward_streaming(input_ids)
    mx.eval(logits)
    print(f"  {time.time()-t0:.2f}s")
    print(f"  Memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Decode
    print(f"\n--- Decode (max 30) ---")
    generated = []
    t_decode = time.time()

    for step in range(30):
        next_logits = logits[:, -1, :]
        token = mx.argmax(next_logits, axis=-1)
        mx.eval(token)
        tid = token.item()
        if tid in (248044, 248045, 248046): break

        generated.append(tid)
        chunk = tokenizer.decode([tid])
        if "<|im_end|>" in chunk: break
        print(chunk, end="", flush=True)

        logits = forward_streaming(mx.array([[tid]]))
        mx.eval(logits)

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t_decode
            tps = (step + 1) / elapsed
            mem = mx.get_active_memory() / 1e9
            print(f" [{tps:.3f} tok/s, {mem:.1f}GB]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    print(f"\n\nDecode: {n} tokens in {t_total:.1f}s ({tps:.3f} tok/s)")
    print(f"Memory: {mx.get_active_memory()/1e9:.2f} GB")

    print(f"\n{'='*60}")
    print(f"Q: {prompt}")
    print(f"A: {output}")
    print(f"{'='*60}")
    print(f"\n  Model: Qwen3.5-27B Dense (16.1 GB, full 4-bit)")
    print(f"  RAM: {mx.get_active_memory()/1e9:.1f} GB pinned")
    print(f"  FFN: streamed from SSD per token")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  METHOD SCALES: Dense model, different architecture, same technique.")


if __name__ == "__main__":
    main()
