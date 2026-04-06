#!/usr/bin/env python3
"""
Split Gemma 4 SwitchLinear stacked experts into per-expert bin files.

Gemma 4 stores experts as (128, out, in) stacked tensors.
This script unstacks them into the layer_XX.bin format that expert_io.py reads.
"""
import os, json, gc, time, glob, argparse
import numpy as np
import mlx.core as mx

PAGE_SIZE = 16384

def main():
    parser = argparse.ArgumentParser(description="Split Gemma 4 for Expert Sniper")
    parser.add_argument("--input", "-i", default="~/models/gemma4-26b-4bit")
    parser.add_argument("--output", "-o", default="~/models/gemma4-stream")
    args = parser.parse_args()

    INPUT_DIR = os.path.expanduser(args.input)
    OUTPUT_DIR = os.path.expanduser(args.output)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/bin", exist_ok=True)

    config = json.load(open(f"{INPUT_DIR}/config.json"))
    tc = config.get("text_config", config)
    NUM_LAYERS = tc["num_hidden_layers"]
    NUM_EXPERTS = tc["num_experts"]

    print(f"Gemma 4 Split (SwitchLinear unstack)")
    print(f"  Input:   {INPUT_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Layers:  {NUM_LAYERS}, Experts: {NUM_EXPERTS}")
    print()

    # Load all weights
    print("Loading safetensors...")
    t0 = time.time()
    all_weights = {}
    for sf in sorted(glob.glob(f"{INPUT_DIR}/model-*.safetensors")):
        print(f"  {os.path.basename(sf)}")
        all_weights.update(mx.load(sf))

    # Identify expert and pinned keys
    pinned = {}
    expert_tensors = {}  # layer_idx -> {tensor_name: (128, ...)}

    EXPERT_PREFIX = "language_model.model.layers.{}.experts.switch_glu.{}.{}"
    PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]
    COMP_NAMES = ["weight", "scales", "biases"]

    for key, val in all_weights.items():
        is_expert = False
        for li in range(NUM_LAYERS):
            for proj in PROJ_NAMES:
                for comp in COMP_NAMES:
                    expected = EXPERT_PREFIX.format(li, proj, comp)
                    if key == expected:
                        if li not in expert_tensors:
                            expert_tensors[li] = {}
                        # Store with the name format expert_io expects
                        tensor_name = f"switch_mlp.{proj}.{comp}"
                        expert_tensors[li][tensor_name] = val
                        is_expert = True
                        break
                if is_expert:
                    break
            if is_expert:
                break
        if not is_expert:
            pinned[key] = val

    print(f"\n  Expert layers: {len(expert_tensors)}")
    print(f"  Pinned keys: {len(pinned)}")

    # Determine per-expert block layout from first layer
    first_layer = expert_tensors[0]
    tensor_layout = {}
    inner_offset = 0

    for tname in sorted(first_layer.keys()):
        arr = first_layer[tname]
        # Shape is (128, ...) — per-expert shape is arr.shape[1:]
        per_expert_shape = list(arr.shape[1:])
        # dtype
        if arr.dtype == mx.uint32:
            dtype_str = "uint32"
            elem_size = 4
        elif arr.dtype == mx.bfloat16:
            dtype_str = "bfloat16"
            elem_size = 2
        elif arr.dtype == mx.float16:
            dtype_str = "float16"
            elem_size = 2
        elif arr.dtype == mx.float32:
            dtype_str = "float32"
            elem_size = 4
        else:
            dtype_str = str(arr.dtype).replace("mlx.core.", "")
            elem_size = 2

        nbytes = 1
        for d in per_expert_shape:
            nbytes *= d
        nbytes *= elem_size

        tensor_layout[tname] = {
            "inner_offset": inner_offset,
            "nbytes": nbytes,
            "shape_per_expert": per_expert_shape,
            "dtype": dtype_str,
        }
        inner_offset += nbytes

    expert_block_size = inner_offset
    data_start = PAGE_SIZE

    print(f"  Expert block: {expert_block_size} bytes ({expert_block_size/1024:.1f} KB)")
    print()

    # Write layer files
    total_expert_bytes = 0
    for layer_idx in range(NUM_LAYERS):
        lt = time.time()
        layer_data = expert_tensors[layer_idx]

        header = {
            "format": "expert_sniper_v1",
            "model": "gemma4-26b-a4b",
            "layer_idx": layer_idx,
            "num_experts": NUM_EXPERTS,
            "layout": {
                "expert_block_size": expert_block_size,
                "data_start": data_start,
                "tensors": tensor_layout,
            }
        }
        header_bytes = json.dumps(header, indent=2).encode("utf-8")
        assert len(header_bytes) < PAGE_SIZE
        header_padded = header_bytes + b"\x00" * (PAGE_SIZE - len(header_bytes))

        layer_path = f"{OUTPUT_DIR}/bin/layer_{layer_idx:02d}.bin"
        with open(layer_path, "wb") as f:
            f.write(header_padded)

            for eid in range(NUM_EXPERTS):
                expert_data = bytearray()
                for tname in sorted(tensor_layout.keys()):
                    stacked = layer_data[tname]  # (128, ...)
                    single = stacked[eid]  # (...)
                    mx.eval(single)

                    if single.dtype == mx.uint32:
                        np_arr = np.array(single).view(np.uint32)
                    elif single.dtype == mx.bfloat16:
                        np_arr = np.array(single.view(mx.uint16))
                    elif single.dtype == mx.float32:
                        np_arr = np.array(single).view(np.float32)
                    elif single.dtype == mx.float16:
                        np_arr = np.array(single).view(np.uint16)
                    else:
                        np_arr = np.array(single)
                    expert_data.extend(np_arr.tobytes())

                # Pad to exact block size
                if len(expert_data) < expert_block_size:
                    expert_data.extend(b"\x00" * (expert_block_size - len(expert_data)))
                f.write(bytes(expert_data[:expert_block_size]))

        file_size = os.path.getsize(layer_path)
        total_expert_bytes += file_size
        elapsed = time.time() - lt
        print(f"  Layer {layer_idx:2d}/{NUM_LAYERS}: {file_size/1e6:.1f} MB ({elapsed:.0f}s)")

        # Free this layer's expert data
        del expert_tensors[layer_idx]
        gc.collect()

    # Save pinned
    pinned_path = f"{OUTPUT_DIR}/pinned.safetensors"
    mx.save_safetensors(pinned_path, pinned)
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    print(f"\nSaved pinned.safetensors: {pinned_bytes/1e9:.2f} GB ({len(pinned)} keys)")

    # Config for streaming
    stream_config = dict(tc)
    stream_config["quantization"] = config.get("quantization", {"bits": 4, "group_size": 64})
    stream_config["streaming"] = {"pinned_file": "pinned.safetensors", "expert_dir": "bin"}
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(stream_config, f, indent=2)

    # Copy tokenizer files
    import shutil
    for tf in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json", "processor_config.json"]:
        src = f"{INPUT_DIR}/{tf}"
        if os.path.exists(src):
            shutil.copy(src, f"{OUTPUT_DIR}/{tf}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"Pinned: {pinned_bytes/1e9:.2f} GB, Experts: {total_expert_bytes/1e9:.2f} GB")

if __name__ == "__main__":
    main()
