#!/usr/bin/env python3
"""
Ternary Down-Proj Fallback Buffer

Stores down_proj per expert as ternary {-scale, 0, +scale}.
Threshold: abs(w) < 0.5 * scale → 0 (captures sparsity).
Packing: 2 bits per value, 4 values per byte.
  00 = zero, 01 = +scale, 10 = -scale

Per-projection cosine: 0.89 (vs 0.79 for 1-bit)
SwiGLU cosine: 0.74 (vs 0.52 for 1-bit)
"""
import json, os, sys, time, gc
import numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

GROUP_SIZE = 128
THRESHOLD = 0.5
INPUT_DIR = os.path.expanduser("~/models/qwen35-35b-moe-stream/experts/")
OUTPUT_PATH = "/Volumes/USB DISK/expert_fallback_down_ternary.bin"
DOWN_PROJ_PREFIX = "mlp.switch_mlp.down_proj"
VALUES_PER_DOWN = 1048576  # 2048 * 512


def parse_header(path):
    with open(path, "rb") as f:
        raw = f.read(16384)
    depth = 0
    for i, b in enumerate(raw):
        if b == ord("{"): depth += 1
        elif b == ord("}"):
            depth -= 1
            if depth == 0: return json.loads(raw[:i+1])


def read_bf16_as_f32(f, nbytes):
    u16 = np.frombuffer(f.read(nbytes), dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def dequant_down_proj(layer_path, expert_offset, tensors):
    w_info = tensors[f"{DOWN_PROJ_PREFIX}.weight"]
    s_info = tensors[f"{DOWN_PROJ_PREFIX}.scales"]
    b_info = tensors[f"{DOWN_PROJ_PREFIX}.biases"]
    with open(layer_path, "rb") as f:
        f.seek(expert_offset + w_info["inner_offset"])
        raw_w = np.frombuffer(f.read(w_info["nbytes"]), dtype=np.uint32)
        f.seek(expert_offset + s_info["inner_offset"])
        scales = read_bf16_as_f32(f, s_info["nbytes"])
        f.seek(expert_offset + b_info["inner_offset"])
        biases = read_bf16_as_f32(f, b_info["nbytes"])
    unpacked = np.zeros(len(raw_w) * 8, dtype=np.uint8)
    for bit in range(8):
        unpacked[bit::8] = ((raw_w >> (bit * 4)) & 0xF).astype(np.uint8)
    gs = len(unpacked) // len(scales)
    groups = unpacked.reshape(len(scales), gs).astype(np.float32)
    return (groups * scales[:, None] + biases[:, None]).flatten()


def quantize_ternary(float_weights):
    """Ternary quantization: {-scale, 0, +scale} with 2 bits per value.

    Packing: 4 values per byte
      00 = zero, 01 = +scale, 10 = -scale
    """
    flat = float_weights.astype(np.float32)
    r = len(flat) % GROUP_SIZE
    if r:
        flat = np.pad(flat, (0, GROUP_SIZE - r))

    ng = len(flat) // GROUP_SIZE
    groups = flat.reshape(ng, GROUP_SIZE)

    scales = np.mean(np.abs(groups), axis=1).astype(np.float16)
    safe_s = np.where(scales > 0, scales, 1e-7).astype(np.float32)

    # Normalize and threshold
    normalized = groups / safe_s[:, None]
    # 0 = zero, 1 = +scale, 2 = -scale
    codes = np.zeros_like(normalized, dtype=np.uint8)
    codes[normalized >= THRESHOLD] = 1   # +scale
    codes[normalized <= -THRESHOLD] = 2  # -scale
    # else stays 0 (zero)

    # Pack 4 values per byte: val0 in bits 0-1, val1 in bits 2-3, etc.
    codes_flat = codes.reshape(-1)
    # Pad to multiple of 4
    pad4 = (-len(codes_flat)) % 4
    if pad4:
        codes_flat = np.pad(codes_flat, (0, pad4))
    packed = (codes_flat[0::4] |
              (codes_flat[1::4] << 2) |
              (codes_flat[2::4] << 4) |
              (codes_flat[3::4] << 6)).astype(np.uint8)

    return scales, packed


def main():
    layer_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith('.bin'))
    n_layers = len(layer_files)

    header = parse_header(os.path.join(INPUT_DIR, layer_files[0]))
    layout = header["layout"]
    n_experts = layout["num_experts"]

    # Calculate ternary size for down_proj only
    padded = VALUES_PER_DOWN + (-VALUES_PER_DOWN % GROUP_SIZE)
    ng = padded // GROUP_SIZE
    scales_bytes = ng * 2  # fp16
    # 2 bits per value, 4 per byte, padded to multiple of 4
    total_codes = ng * GROUP_SIZE
    packed_bytes = (total_codes + 3) // 4
    expert_ternary_size = scales_bytes + packed_bytes

    total_data = n_layers * n_experts * expert_ternary_size
    header_size = 16384

    print(f"Ternary down-proj buffer")
    print(f"  Layers: {n_layers}, Experts: {n_experts}")
    print(f"  Per expert: {expert_ternary_size:,} bytes ({expert_ternary_size/1024:.1f} KB)")
    print(f"  Total: {(header_size + total_data)/1024/1024:.1f} MB")
    print(f"  Output: {OUTPUT_PATH}")

    out_header = {
        "format": "expert_fallback_down_ternary_v1",
        "group_size": GROUP_SIZE,
        "threshold": THRESHOLD,
        "num_layers": n_layers,
        "num_experts": n_experts,
        "expert_ternary_size": expert_ternary_size,
        "scales_bytes": scales_bytes,
        "packed_bytes": packed_bytes,
        "data_start": header_size,
        "projection": "down_proj",
        "values_per_projection": VALUES_PER_DOWN,
        "shape": [2048, 512],
        "packing": "2bit: 00=zero, 01=+scale, 10=-scale, 4 vals/byte LSB first",
        "reconstruction": "codes[0]=0, codes[1]=+scale, codes[2]=-scale",
    }

    out_bytes = json.dumps(out_header, indent=2).encode("utf-8")
    padded_header = out_bytes + b"\x00" * (header_size - len(out_bytes))

    with open(OUTPUT_PATH, "wb") as f:
        f.write(padded_header)

    t_start = time.time()

    for layer_idx, layer_file in enumerate(layer_files):
        t_layer = time.time()
        layer_path = os.path.join(INPUT_DIR, layer_file)
        lh = parse_header(layer_path)
        lt = lh["layout"]["tensors"]
        lds = lh["layout"]["data_start"]
        les = lh["layout"]["expert_block_size"]

        file_size = os.path.getsize(layer_path)
        n_full = (file_size - lds) // les

        with open(OUTPUT_PATH, "ab") as out_f:
            for ei in range(n_experts):
                if ei < n_full:
                    eo = lds + ei * les
                    fw = dequant_down_proj(layer_path, eo, lt)
                    s, p = quantize_ternary(fw)
                    out_f.write(s.tobytes())
                    out_f.write(p.tobytes())
                    del fw, s, p
                else:
                    out_f.write(b'\x00' * expert_ternary_size)

                if ei % 64 == 63:
                    out_f.flush()

        gc.collect()
        elapsed = time.time() - t_layer
        print(f"  Layer {layer_idx:2d}/{n_layers} in {elapsed:.1f}s "
              f"({n_full} real + {n_experts - n_full} zero-filled)")

    total_time = time.time() - t_start
    final_size = os.path.getsize(OUTPUT_PATH)
    expected = header_size + n_layers * n_experts * expert_ternary_size
    print(f"\nDone in {total_time:.0f}s")
    print(f"File: {final_size:,} bytes ({final_size/1024/1024:.1f} MB)")
    print(f"Expected: {expected:,} bytes")
    print(f"Match: {final_size == expected}")


if __name__ == "__main__":
    main()
