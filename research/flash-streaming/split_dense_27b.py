"""
Split Qwen3.5-27B (dense, 16.1 GB) into Flash Stream format.

This model has NO MoE — standard dense FFN per layer.
Split: pinned (attention+SSM+embed) + per-layer FFN files.

Proves our Flash Streaming method works on dense models too.
"""

import os
import sys
import gc
import json
import time
import glob
import numpy as np
import mlx.core as mx

MLX_MODEL_DIR = "/Users/bigneek/models/qwen35-27b-mlx-4bit"
OUTPUT_DIR = "/Users/bigneek/models/qwen35-27b-flash-stream"
PAGE_SIZE = 16384
NUM_LAYERS = 64  # Qwen3.5-27B has 64 layers


def align_up(offset):
    return (offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def main():
    os.makedirs(f"{OUTPUT_DIR}/ffn", exist_ok=True)

    print("=" * 60)
    print("  Split Qwen3.5-27B Dense → Flash Stream")
    print("  16.1 GB model on 16 GB Mac")
    print("=" * 60)

    # Load all safetensors shards
    print("\nLoading MLX model...")
    t0 = time.time()
    shard_files = sorted(glob.glob(f"{MLX_MODEL_DIR}/model-*.safetensors"))
    print(f"  {len(shard_files)} shards")

    all_weights = {}
    for sf in shard_files:
        shard = mx.load(sf)
        all_weights.update(shard)
        print(f"  Loaded {os.path.basename(sf)}: {len(shard)} arrays")
        del shard

    print(f"  Total: {len(all_weights)} arrays in {time.time()-t0:.1f}s")

    # Strip language_model. prefix if present
    cleaned = {}
    for k, v in all_weights.items():
        name = k.replace("language_model.", "") if k.startswith("language_model.") else k
        cleaned[name] = v
    all_weights = cleaned

    # Categorize: pinned vs FFN
    pinned = {}
    ffn_layers = {}

    for name, arr in all_weights.items():
        # Dense FFN: gate_proj, up_proj, down_proj per layer
        is_ffn = any(p in name for p in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])

        if is_ffn and "layers." in name:
            parts = name.split(".")
            layer_idx = int(parts[2])
            local_name = ".".join(parts[3:])

            if layer_idx not in ffn_layers:
                ffn_layers[layer_idx] = {}
            ffn_layers[layer_idx][local_name] = arr
        else:
            pinned[name] = arr

    print(f"\n  Pinned: {len(pinned)} arrays")
    print(f"  FFN layers: {len(ffn_layers)}")

    # Save pinned
    print("\n  Saving pinned.safetensors...")
    mx.save_safetensors(f"{OUTPUT_DIR}/pinned.safetensors", pinned)
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    print(f"    {pinned_bytes/1e9:.2f} GB")

    # Verify SSM tensors
    print("\n  Verifying key weights...")
    for k in ["model.layers.0.linear_attn.A_log",
              "model.layers.0.linear_attn.dt_bias",
              "model.layers.0.input_layernorm.weight"]:
        if k in pinned:
            v = pinned[k]
            mx.eval(v)
            nz = mx.abs(v.astype(mx.float32)).sum().item() > 0
            print(f"    {'✓' if nz else '✗'} {k}: {v.shape} nonzero={nz}")

    del pinned
    gc.collect()

    # Save FFN layers as safetensors (simpler than binary for dense)
    print(f"\n  Saving {len(ffn_layers)} FFN layers...")
    total_ffn = 0

    for layer_idx in sorted(ffn_layers.keys()):
        data = ffn_layers[layer_idx]
        fname = f"{OUTPUT_DIR}/ffn/layer_{layer_idx:02d}.safetensors"
        mx.save_safetensors(fname, data)
        layer_bytes = sum(v.nbytes for v in data.values())
        total_ffn += layer_bytes
        if layer_idx % 16 == 0:
            print(f"    Layer {layer_idx}: {layer_bytes/1e6:.1f} MB")

    print(f"    Total FFN: {total_ffn/1e9:.2f} GB")

    # Save config
    config_src = f"{MLX_MODEL_DIR}/config.json"
    if os.path.exists(config_src):
        with open(config_src) as f:
            orig = json.load(f)

        tc = orig.get("text_config", orig)
        stream_config = {
            "model_type": tc.get("model_type", "qwen3_5_text"),
            "hidden_size": tc.get("hidden_size"),
            "num_hidden_layers": tc.get("num_hidden_layers"),
            "intermediate_size": tc.get("intermediate_size"),
            "num_attention_heads": tc.get("num_attention_heads"),
            "num_key_value_heads": tc.get("num_key_value_heads"),
            "rms_norm_eps": tc.get("rms_norm_eps"),
            "vocab_size": tc.get("vocab_size"),
            "max_position_embeddings": tc.get("max_position_embeddings"),
            "head_dim": tc.get("head_dim"),
            "tie_word_embeddings": orig.get("tie_word_embeddings", False),
            "linear_num_value_heads": tc.get("linear_num_value_heads"),
            "linear_num_key_heads": tc.get("linear_num_key_heads"),
            "linear_key_head_dim": tc.get("linear_key_head_dim"),
            "linear_value_head_dim": tc.get("linear_value_head_dim"),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim"),
            "full_attention_interval": tc.get("full_attention_interval"),
            "rope_parameters": tc.get("rope_parameters"),
            "quantization": orig.get("quantization", {"bits": 4, "group_size": 64}),
            "streaming": {
                "pinned_file": "pinned.safetensors",
                "ffn_dir": "ffn/",
                "ffn_layers": len(ffn_layers),
                "model_class": "dense",
            }
        }
        with open(f"{OUTPUT_DIR}/config.json", "w") as f:
            json.dump(stream_config, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s!")
    print(f"  Pinned: {pinned_bytes/1e9:.2f} GB")
    print(f"  FFN: {total_ffn/1e9:.2f} GB")
    print(f"  Total: {(pinned_bytes+total_ffn)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
