#!/usr/bin/env python3
"""
Split 35B MoE expert files for distributed inference.

Takes the existing per-layer expert files and copies the specified
layer range to an output directory.

Usage:
    # For Mac mini (layers 0-23):
    python3 split_for_distributed.py --source ~/models/qwen35-35b-moe-stream/ \
        --layers 0-23 --output ~/models/local-experts/

    # For MacBook (layers 24-39):
    python3 split_for_distributed.py --source ~/models/qwen35-35b-moe-stream/ \
        --layers 24-39 --output ~/models/remote-experts/
"""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source directory with layer_XX.bin files")
    parser.add_argument("--layers", required=True, help="Layer range, e.g., '0-23' or '24-39'")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    # Parse layer range
    start, end = args.layers.split("-")
    start, end = int(start), int(end)
    layers = list(range(start, end + 1))
    print(f"Splitting layers {start}-{end} ({len(layers)} layers)")

    os.makedirs(args.output, exist_ok=True)

    # Copy layer files
    total_size = 0
    for layer_idx in layers:
        # Try different naming patterns
        for pattern in [f"layer_{layer_idx:02d}.bin", f"layer_{layer_idx:02d}.safetensors"]:
            src = os.path.join(args.source, pattern)
            if os.path.exists(src):
                dst = os.path.join(args.output, pattern)
                size = os.path.getsize(src)
                total_size += size
                print(f"  {pattern}: {size/1e6:.1f} MB")
                shutil.copy2(src, dst)
                break
        else:
            print(f"  WARNING: No file found for layer {layer_idx}")

    # Copy header/config if present
    for extra in ["header.json", "config.json"]:
        src = os.path.join(args.source, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, extra))
            print(f"  {extra}: copied")

    print(f"\nTotal: {total_size/1e9:.2f} GB in {args.output}")
    print(f"\nTo copy to MacBook:")
    print(f"  scp -r {args.output} macbook:~/models/remote-experts/")


if __name__ == "__main__":
    main()
