"""
Split Qwen3.5-122B-A10B-4bit into Flash Stream format for PyTorch.

Input: mlx-community/Qwen3.5-122B-A10B-4bit safetensors (69.6 GB)
Output: pinned.safetensors + per-layer expert files

The MLX 4-bit format stores:
  - weight: uint32 packed (4 bits per param, group_size=64)
  - scales: float16 (1 per group)
  - biases: float16 (1 per group)

We keep this format on disk. The sniper dequantizes on the GPU at runtime.

Usage:
    python3 split_122b.py [--model-dir /path/to/model] [--output-dir /path/to/output]
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

NUM_LAYERS = 48
NUM_EXPERTS = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.path.expanduser("~/models/qwen35-122b-a10b-4bit"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/models/qwen35-122b-stream"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experts").mkdir(exist_ok=True)

    # Copy config
    config_src = model_dir / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy(config_src, output_dir / "config.json")
        print(f"Copied config.json")

    # Index all safetensors files
    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    print(f"Found {len(shard_files)} safetensors shards")

    # Build key → shard mapping from index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        key_to_shard = index["weight_map"]
        print(f"Loaded weight map: {len(key_to_shard)} keys")
    else:
        # Scan all shards
        print("No index found, scanning shards...")
        key_to_shard = {}
        for sf in shard_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    key_to_shard[k] = sf.name

    # Classify keys: pinned vs expert
    pinned_keys = []
    expert_keys = {}  # layer_idx -> expert_idx -> [keys]

    # The weight names follow the pattern:
    # language_model.model.layers.{i}.mlp.switch_mlp.experts.{j}.{gate_proj|up_proj|down_proj}.{weight|scales|biases}
    # OR for the text-only model:
    # model.layers.{i}.mlp.switch_mlp.experts.{j}.{gate_proj|up_proj|down_proj}.{weight|scales|biases}

    for key in sorted(key_to_shard.keys()):
        is_expert = False
        for prefix in ["language_model.model.layers.", "model.layers."]:
            if prefix in key and ".mlp.switch_mlp.experts." in key:
                # Parse layer and expert index
                after_prefix = key.split(prefix)[1]
                parts = after_prefix.split(".")
                layer_idx = int(parts[0])
                # Find expert index after "experts."
                exp_start = key.index(".experts.") + len(".experts.")
                rest = key[exp_start:]
                expert_idx = int(rest.split(".")[0])

                if layer_idx not in expert_keys:
                    expert_keys[layer_idx] = {}
                if expert_idx not in expert_keys[layer_idx]:
                    expert_keys[layer_idx][expert_idx] = []
                expert_keys[layer_idx][expert_idx].append(key)
                is_expert = True
                break

        if not is_expert:
            pinned_keys.append(key)

    num_expert_layers = len(expert_keys)
    total_experts = sum(len(v) for v in expert_keys.values())
    print(f"\nPinned keys: {len(pinned_keys)}")
    print(f"Expert layers: {num_expert_layers}")
    print(f"Total expert entries: {total_experts}")
    print(f"Experts per layer: {total_experts // max(num_expert_layers, 1)}")

    # Save pinned weights
    print(f"\n--- Saving pinned weights ---")
    pinned = {}
    t0 = time.time()
    shards_opened = {}

    for i, key in enumerate(pinned_keys):
        shard_name = key_to_shard[key]
        shard_path = str(model_dir / shard_name)

        if shard_path not in shards_opened:
            shards_opened[shard_path] = safe_open(shard_path, framework="pt", device="cpu")

        tensor = shards_opened[shard_path].get_tensor(key)
        pinned[key] = tensor

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pinned_keys)} pinned keys loaded...")

    pinned_size = sum(t.nbytes for t in pinned.values()) / 1e9
    print(f"  Pinned: {len(pinned)} tensors, {pinned_size:.2f} GB")

    save_file(pinned, str(output_dir / "pinned.safetensors"))
    print(f"  Saved in {time.time()-t0:.1f}s")
    del pinned
    gc.collect()

    # Save expert weights per layer
    print(f"\n--- Saving expert weights per layer ---")
    for layer_idx in sorted(expert_keys.keys()):
        t0 = time.time()
        layer_experts = expert_keys[layer_idx]

        layer_tensors = {}
        for expert_idx in sorted(layer_experts.keys()):
            for key in layer_experts[expert_idx]:
                shard_name = key_to_shard[key]
                shard_path = str(model_dir / shard_name)

                if shard_path not in shards_opened:
                    shards_opened[shard_path] = safe_open(shard_path, framework="pt", device="cpu")

                tensor = shards_opened[shard_path].get_tensor(key)
                # Simplify key: just expert_idx.proj.component
                # e.g., "0.gate_proj.weight", "0.gate_proj.scales"
                short_key = key.split(".experts.")[1]
                layer_tensors[short_key] = tensor

        layer_size = sum(t.nbytes for t in layer_tensors.values()) / 1e9

        out_path = output_dir / "experts" / f"layer_{layer_idx:02d}.safetensors"
        save_file(layer_tensors, str(out_path))

        elapsed = time.time() - t0
        print(f"  Layer {layer_idx:2d}: {len(layer_experts)} experts, "
              f"{len(layer_tensors)} tensors, {layer_size:.2f} GB [{elapsed:.1f}s]")

        del layer_tensors
        gc.collect()

    # Close all shard handles
    shards_opened.clear()

    print(f"\n=== Split complete ===")
    print(f"  Output: {output_dir}")
    print(f"  Pinned: {output_dir / 'pinned.safetensors'}")
    print(f"  Experts: {output_dir / 'experts' / 'layer_*.safetensors'}")


if __name__ == "__main__":
    main()
