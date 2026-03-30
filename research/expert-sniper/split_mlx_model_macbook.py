"""
Split a pre-converted MLX model into Flash Stream format.
Ultra memory-efficient version for 8 GB MacBooks.

Uses safetensors to load individual tensors (not whole shards).
Processes one expert layer at a time: load 9 tensors, write .bin, free.
Peak memory: ~1.5 GB pinned + ~0.5 GB per expert layer = ~2 GB.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import mlx.core as mx
from safetensors import safe_open

MLX_MODEL_DIR = "/Volumes/USB DISK/qwen35"
OUTPUT_DIR = "/Volumes/USB DISK/qwen35-35b-moe-stream"
PAGE_SIZE = 16384
NUM_LAYERS = 40
NUM_EXPERTS = 256


def align_up(offset):
    return (offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def strip_prefix(name):
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def write_expert_layer(layer_idx, data):
    """Write one expert layer as a 16KB-aligned binary file."""
    tensor_order = []
    for proj in ["mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj"]:
        for suffix in [".weight", ".scales", ".biases"]:
            key = f"{proj}{suffix}"
            if key in data:
                tensor_order.append(key)

    expert_block_size = 0
    tensor_sizes = {}
    for key in tensor_order:
        arr = data[key]
        mx.eval(arr)
        per_expert = arr[0].nbytes
        tensor_sizes[key] = per_expert
        expert_block_size += per_expert
    expert_block_size = align_up(expert_block_size)

    inner_offset = 0
    layout_tensors = {}
    for key in tensor_order:
        arr = data[key]
        layout_tensors[key] = {
            "inner_offset": inner_offset,
            "nbytes": tensor_sizes[key],
            "dtype": str(arr[0].dtype),
            "shape_per_expert": list(arr[0].shape),
        }
        inner_offset += tensor_sizes[key]

    total_size = PAGE_SIZE + NUM_EXPERTS * expert_block_size

    header = json.dumps({
        "format": "moe_flash_v1",
        "page_size": PAGE_SIZE,
        "layer_idx": layer_idx,
        "total_size": total_size,
        "layout": {
            "num_experts": NUM_EXPERTS,
            "expert_block_size": expert_block_size,
            "data_start": PAGE_SIZE,
            "tensors": layout_tensors,
        }
    }).encode()

    out_path = f"{OUTPUT_DIR}/experts/layer_{layer_idx:02d}.bin"
    with open(out_path, "wb") as f:
        f.write(header + b"\x00" * (PAGE_SIZE - len(header)))

        for expert_id in range(NUM_EXPERTS):
            block_start = PAGE_SIZE + expert_id * expert_block_size
            if f.tell() < block_start:
                f.write(b"\x00" * (block_start - f.tell()))

            for key in tensor_order:
                arr = data[key]
                expert_slice = arr[expert_id]
                mx.eval(expert_slice)
                raw_bytes = memoryview(expert_slice)
                f.write(bytes(raw_bytes))

        if f.tell() < total_size:
            f.write(b"\x00" * (total_size - f.tell()))

    return total_size


def main():
    os.makedirs(f"{OUTPUT_DIR}/experts", exist_ok=True)

    print("=" * 60)
    print("  Split MLX Model -> Flash Stream Format")
    print("  Source: mlx-community/Qwen3.5-35B-A3B-4bit")
    print("  Ultra memory-efficient mode (8 GB MacBook)")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Load weight index
    print("\nLoading weight index...")
    index_path = f"{MLX_MODEL_DIR}/model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    print(f"  {len(weight_map)} weights")

    # Categorize weights
    pinned_weights = {}  # stripped_name -> (raw_name, shard_file)
    expert_weights = {}  # layer_idx -> {local_name: (raw_name, shard_file)}

    for raw_name, shard_file in weight_map.items():
        stripped = strip_prefix(raw_name)
        if "switch_mlp" in stripped:
            parts = stripped.split(".")
            layer_idx = int(parts[2])
            local_name = ".".join(parts[3:])
            if layer_idx not in expert_weights:
                expert_weights[layer_idx] = {}
            expert_weights[layer_idx][local_name] = (raw_name, shard_file)
        else:
            pinned_weights[stripped] = (raw_name, shard_file)

    print(f"  {len(pinned_weights)} pinned, {sum(len(v) for v in expert_weights.values())} expert")
    print(f"  {len(expert_weights)} expert layers")

    # Use mlx.core to load individual tensors from safetensors (handles bfloat16)
    # Load shards lazily with mx.load which returns a lazy dict
    def load_tensor(raw_name, shard_file):
        """Load a single tensor from a shard using mx.load (lazy)."""
        shard = mx.load(f"{MLX_MODEL_DIR}/{shard_file}", return_metadata=False)
        arr = shard[raw_name]
        mx.eval(arr)
        del shard
        return arr

    # Group pinned weights by shard so we load each shard only once
    pinned_by_shard = {}
    for stripped, (raw_name, shard_file) in pinned_weights.items():
        if shard_file not in pinned_by_shard:
            pinned_by_shard[shard_file] = []
        pinned_by_shard[shard_file].append((stripped, raw_name))

    # Step 2: Load pinned weights one shard at a time
    print(f"\n  Loading {len(pinned_weights)} pinned weights...")
    pinned = {}
    for shard_file in sorted(pinned_by_shard.keys()):
        shard = mx.load(f"{MLX_MODEL_DIR}/{shard_file}", return_metadata=False)
        for stripped, raw_name in pinned_by_shard[shard_file]:
            arr = shard[raw_name]
            mx.eval(arr)
            pinned[stripped] = arr
        # Only keep pinned refs, free shard dict
        del shard
        gc.collect()
        mx.clear_cache()
        print(f"    Loaded pinned from {shard_file}")
    mx.eval(*pinned.values())

    pinned_bytes = sum(v.nbytes for v in pinned.values())
    print(f"    {pinned_bytes/1e9:.2f} GB pinned loaded")

    # Save pinned
    print("  Saving pinned.safetensors...")
    mx.save_safetensors(f"{OUTPUT_DIR}/pinned.safetensors", pinned)

    # Verify SSM tensors
    for k in ["model.layers.0.linear_attn.A_log",
              "model.layers.0.linear_attn.dt_bias"]:
        if k in pinned:
            v = pinned[k]
            mx.eval(v)
            nz = mx.abs(v.astype(mx.float32)).sum().item() > 0
            print(f"    {'OK' if nz else 'ZERO'} {k}: {v.shape}")

    del pinned
    gc.collect()
    mx.clear_cache()

    # Step 3: Process expert layers one at a time
    print(f"\n  Processing {len(expert_weights)} expert layers...")
    total_expert_bytes = 0

    # Group expert layers by shard to minimize shard reloads
    expert_shard_map = {}  # shard_file -> [(layer_idx, local_name, raw_name)]
    for layer_idx in sorted(expert_weights.keys()):
        for local_name, (raw_name, shard_file) in expert_weights[layer_idx].items():
            if shard_file not in expert_shard_map:
                expert_shard_map[shard_file] = []
            expert_shard_map[shard_file].append((layer_idx, local_name, raw_name))

    # Load all expert data grouped by shard, then write complete layers
    all_layer_data = {}  # layer_idx -> {local_name: arr}
    expected_per_layer = {li: len(v) for li, v in expert_weights.items()}

    for shard_file in sorted(expert_shard_map.keys()):
        print(f"    Loading experts from {shard_file}...")
        shard = mx.load(f"{MLX_MODEL_DIR}/{shard_file}", return_metadata=False)
        for layer_idx, local_name, raw_name in expert_shard_map[shard_file]:
            if layer_idx not in all_layer_data:
                all_layer_data[layer_idx] = {}
            arr = shard[raw_name]
            mx.eval(arr)
            all_layer_data[layer_idx][local_name] = arr
        del shard
        gc.collect()
        mx.clear_cache()

        # Write any layers that are now complete
        completed = []
        for layer_idx in sorted(all_layer_data.keys()):
            if len(all_layer_data[layer_idx]) == expected_per_layer[layer_idx]:
                completed.append(layer_idx)

        for layer_idx in completed:
            layer_data = all_layer_data.pop(layer_idx)
            size = write_expert_layer(layer_idx, layer_data)
            total_expert_bytes += size
            del layer_data
            gc.collect()
            mx.clear_cache()
            if layer_idx % 5 == 0 or layer_idx == max(expert_weights.keys()):
                mem = mx.get_active_memory() / 1e9
                print(f"    Layer {layer_idx:2d}: {size/1e6:.0f} MB  (mem: {mem:.1f} GB)")

    # Step 4: Write config
    config_src = f"{MLX_MODEL_DIR}/config.json"
    if os.path.exists(config_src):
        with open(config_src) as f:
            orig_config = json.load(f)

        tc = orig_config.get("text_config", orig_config)
        stream_config = {
            "model_type": tc.get("model_type", "qwen3_5_moe_text"),
            "hidden_size": tc.get("hidden_size", 2048),
            "num_hidden_layers": tc.get("num_hidden_layers", 40),
            "num_attention_heads": tc.get("num_attention_heads", 16),
            "num_key_value_heads": tc.get("num_key_value_heads", 2),
            "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
            "vocab_size": tc.get("vocab_size", 248320),
            "max_position_embeddings": tc.get("max_position_embeddings", 262144),
            "head_dim": tc.get("head_dim", 256),
            "tie_word_embeddings": orig_config.get("tie_word_embeddings", False),
            "num_experts": tc.get("num_experts", 256),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 512),
            "moe_intermediate_size": tc.get("moe_intermediate_size", 512),
            "linear_num_value_heads": tc.get("linear_num_value_heads", 32),
            "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
            "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
            "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
            "full_attention_interval": tc.get("full_attention_interval", 4),
            "rope_parameters": tc.get("rope_parameters"),
            "quantization": orig_config.get("quantization", {"bits": 4, "group_size": 64}),
            "streaming": {
                "pinned_file": "pinned.safetensors",
                "expert_dir": "experts/",
                "num_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "experts_per_tok": 8,
            }
        }
        with open(f"{OUTPUT_DIR}/config.json", "w") as f:
            json.dump(stream_config, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s!")
    print(f"  Experts: {total_expert_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
