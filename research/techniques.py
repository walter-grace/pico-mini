#!/usr/bin/env python3
"""
KV Cache Compression Techniques Library.
This is the file that kv-lab modifies during experiments.

Each technique is a (compress, decompress) function pair with a standard signature.
The harness calls compress() → decompress() → measure_quality().
"""

import mlx.core as mx
import numpy as np
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class CompressedTensor:
    data: Any
    metadata: Dict
    original_shape: tuple
    original_dtype: str
    technique: str
    bits: int


# ══════════════════════════════════════════════════════
# TECHNIQUE 1: baseline_minmax
# Standard per-group asymmetric min-max quantization
# This is what turboquant.py currently does
# ══════════════════════════════════════════════════════

def baseline_minmax_compress(tensor, bits=4, group_size=64):
    x = tensor.astype(mx.float32)
    shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    rows, cols = x_flat.shape

    pad = (group_size - cols % group_size) % group_size
    if pad > 0:
        x_flat = mx.pad(x_flat, [(0, 0), (0, pad)])
        cols = x_flat.shape[-1]

    n_groups = cols // group_size
    x_groups = x_flat.reshape(rows, n_groups, group_size)

    g_min = mx.min(x_groups, axis=-1, keepdims=True)
    g_max = mx.max(x_groups, axis=-1, keepdims=True)

    max_int = (1 << bits) - 1
    scale = (g_max - g_min) / max_int
    scale = mx.where(scale == 0, mx.ones_like(scale), scale)

    x_quant = mx.round((x_groups - g_min) / scale).astype(mx.uint8)
    x_quant = mx.clip(x_quant, 0, max_int)

    return CompressedTensor(
        data=x_quant,
        metadata={"scales": scale.squeeze(-1), "zeros": g_min.squeeze(-1), "group_size": group_size, "pad": pad},
        original_shape=shape,
        original_dtype=str(tensor.dtype),
        technique="baseline_minmax",
        bits=bits,
    )


def baseline_minmax_decompress(ct):
    x = ct.data.astype(mx.float32) * ct.metadata["scales"][..., None] + ct.metadata["zeros"][..., None]
    x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    target = 1
    for s in ct.original_shape:
        target *= s
    x = x.reshape(-1)[:target].reshape(ct.original_shape)
    if "bfloat16" in ct.original_dtype:
        x = x.astype(mx.bfloat16)
    elif "float16" in ct.original_dtype:
        x = x.astype(mx.float16)
    return x


# ══════════════════════════════════════════════════════
# TECHNIQUE 2: polar_quant
# Convert to polar coordinates — angles get more bits
# Attention cares about direction, not magnitude
# ══════════════════════════════════════════════════════

def polar_quant_compress(tensor, bits=3, group_size=64):
    x = tensor.astype(mx.float32)
    shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])

    # Separate magnitude and direction
    norms = mx.linalg.norm(x_flat, axis=-1, keepdims=True)
    norms = mx.where(norms == 0, mx.ones_like(norms), norms)
    directions = x_flat / norms

    # Quantize directions with N bits (per-group min-max)
    dir_ct = baseline_minmax_compress(directions.reshape(shape), bits=bits, group_size=group_size)

    # Quantize norms with 8 bits (cheap — one per vector)
    n_min = mx.min(norms)
    n_max = mx.max(norms)
    n_scale = (n_max - n_min) / 255
    n_scale = mx.where(n_scale == 0, mx.ones_like(n_scale), n_scale)
    norms_quant = mx.round((norms - n_min) / n_scale).astype(mx.uint8)

    return CompressedTensor(
        data=dir_ct.data,
        metadata={
            **dir_ct.metadata,
            "norms_quant": norms_quant,
            "n_min": n_min, "n_max": n_max, "n_scale": n_scale,
        },
        original_shape=shape,
        original_dtype=str(tensor.dtype),
        technique="polar_quant",
        bits=bits,
    )


def polar_quant_decompress(ct):
    # Reconstruct directions
    dir_ct = CompressedTensor(
        data=ct.data, metadata=ct.metadata,
        original_shape=ct.original_shape, original_dtype=ct.original_dtype,
        technique="baseline_minmax", bits=ct.bits,
    )
    directions = baseline_minmax_decompress(dir_ct)
    d_flat = directions.reshape(-1, directions.shape[-1])

    # Reconstruct norms
    norms = ct.metadata["norms_quant"].astype(mx.float32) * ct.metadata["n_scale"] + ct.metadata["n_min"]

    # Recombine
    x = d_flat * norms
    x = x.reshape(ct.original_shape)
    if "bfloat16" in ct.original_dtype:
        x = x.astype(mx.bfloat16)
    return x


# ══════════════════════════════════════════════════════
# TECHNIQUE 3: qjl_1bit
# Random Gaussian projection → sign bit → asymmetric estimator
# Zero memory overhead — the most aggressive compression
# ══════════════════════════════════════════════════════

def qjl_1bit_compress(tensor, projection_dim=256, seed=42):
    x = tensor.astype(mx.float32)
    shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    d = x_flat.shape[-1]

    # Generate random projection matrix (deterministic from seed)
    mx.random.seed(seed)
    R = mx.random.normal(shape=(d, projection_dim)) / math.sqrt(projection_dim)

    # Project and take sign
    projected = x_flat @ R
    signs = mx.where(projected >= 0, mx.array(1, dtype=mx.int8), mx.array(-1, dtype=mx.int8))

    # Store norms for asymmetric reconstruction
    norms = mx.linalg.norm(x_flat, axis=-1, keepdims=True)

    return CompressedTensor(
        data=signs,
        metadata={"norms": norms, "seed": seed, "projection_dim": projection_dim, "d": d},
        original_shape=shape,
        original_dtype=str(tensor.dtype),
        technique="qjl_1bit",
        bits=1,
    )


def qjl_1bit_decompress(ct):
    signs = ct.data.astype(mx.float32)
    norms = ct.metadata["norms"]
    d = ct.metadata["d"]
    projection_dim = ct.metadata["projection_dim"]

    # Regenerate same projection matrix
    mx.random.seed(ct.metadata["seed"])
    R = mx.random.normal(shape=(d, projection_dim)) / math.sqrt(projection_dim)

    # Asymmetric reconstruction: x_approx = sqrt(pi/2) * norm * R^T @ sign(R @ x) / m
    x_approx = math.sqrt(math.pi / 2) * norms * (signs @ R.T) / projection_dim
    x_approx = x_approx.reshape(ct.original_shape)

    if "bfloat16" in ct.original_dtype:
        x_approx = x_approx.astype(mx.bfloat16)
    return x_approx


# ══════════════════════════════════════════════════════
# TECHNIQUE 4: hadamard_rotate
# Walsh-Hadamard rotation before quantization
# Spreads outliers across dimensions for better quantization
# ══════════════════════════════════════════════════════

def _hadamard_matrix(n):
    """Generate normalized Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return mx.array([[1.0]])
    half = _hadamard_matrix(n // 2)
    top = mx.concatenate([half, half], axis=1)
    bot = mx.concatenate([half, -half], axis=1)
    H = mx.concatenate([top, bot], axis=0) / math.sqrt(2)
    return H


def hadamard_rotate_compress(tensor, bits=4, group_size=64):
    x = tensor.astype(mx.float32)
    shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    d = x_flat.shape[-1]

    # Pad to next power of 2
    d_pad = 1 << (d - 1).bit_length()
    if d_pad > d:
        x_flat = mx.pad(x_flat, [(0, 0), (0, d_pad - d)])

    # Apply Hadamard rotation (only if dimension is manageable)
    if d_pad <= 512:
        H = _hadamard_matrix(d_pad)
        x_rot = x_flat @ H
    else:
        # For large dimensions, use random sign flip + shuffle (approximate Hadamard)
        mx.random.seed(42)
        signs = mx.where(mx.random.uniform(shape=(d_pad,)) > 0.5, 1.0, -1.0)
        x_rot = x_flat * signs

    # Quantize the rotated data
    rot_shape = list(shape)
    rot_shape[-1] = d_pad
    ct = baseline_minmax_compress(x_rot.reshape(rot_shape), bits=bits, group_size=group_size)

    ct.metadata["original_d"] = d
    ct.metadata["d_pad"] = d_pad
    ct.metadata["use_full_hadamard"] = d_pad <= 512
    ct.technique = "hadamard_rotate"
    ct.original_shape = shape
    return ct


def hadamard_rotate_decompress(ct):
    # Decompress quantized data
    base_ct = CompressedTensor(
        data=ct.data, metadata=ct.metadata,
        original_shape=list(ct.original_shape)[:-1] + [ct.metadata["d_pad"]],
        original_dtype=ct.original_dtype,
        technique="baseline_minmax", bits=ct.bits,
    )
    x_rot = baseline_minmax_decompress(base_ct)
    x_flat = x_rot.reshape(-1, ct.metadata["d_pad"])

    # Inverse Hadamard rotation
    d_pad = ct.metadata["d_pad"]
    if ct.metadata["use_full_hadamard"]:
        H = _hadamard_matrix(d_pad)
        x_flat = x_flat @ H.T
    else:
        mx.random.seed(42)
        signs = mx.where(mx.random.uniform(shape=(d_pad,)) > 0.5, 1.0, -1.0)
        x_flat = x_flat * signs

    # Remove padding
    x_flat = x_flat[:, :ct.metadata["original_d"]]
    x = x_flat.reshape(ct.original_shape)
    if "bfloat16" in ct.original_dtype:
        x = x.astype(mx.bfloat16)
    return x


# ══════════════════════════════════════════════════════
# TECHNIQUE 5: mixed_kv
# Different bits for K cache vs V cache
# K needs direction (angles), V needs magnitude (values)
# ══════════════════════════════════════════════════════

def mixed_kv_compress(tensor, k_bits=3, v_bits=4, group_size=64, is_key=True):
    bits = k_bits if is_key else v_bits
    return baseline_minmax_compress(tensor, bits=bits, group_size=group_size)

def mixed_kv_decompress(ct):
    return baseline_minmax_decompress(ct)


# ══════════════════════════════════════════════════════
# TECHNIQUE 6: per_layer_adaptive
# Measure sensitivity per layer, assign bits accordingly
# ══════════════════════════════════════════════════════

def per_layer_adaptive_compress(tensor, layer_idx=0, total_layers=32, group_size=64):
    # Heuristic: first and last layers get 4 bits, middle gets 3
    ratio = layer_idx / max(total_layers - 1, 1)
    if ratio < 0.15 or ratio > 0.85:
        bits = 4  # Important layers
    else:
        bits = 3  # Middle layers can handle more compression
    return baseline_minmax_compress(tensor, bits=bits, group_size=group_size)

def per_layer_adaptive_decompress(ct):
    return baseline_minmax_decompress(ct)


# ══════════════════════════════════════════════════════
# TECHNIQUE 7: residual_correction
# Two-pass: N-bit base + 1-bit QJL on residual error
# Inspired by TurboQuant's PolarQuant + QJL pipeline
# ══════════════════════════════════════════════════════

def residual_correction_compress(tensor, base_bits=3, group_size=64, projection_dim=128):
    # Pass 1: Base quantization
    base_ct = baseline_minmax_compress(tensor, bits=base_bits, group_size=group_size)
    reconstructed = baseline_minmax_decompress(base_ct)

    # Compute residual
    residual = tensor.astype(mx.float32) - reconstructed.astype(mx.float32)

    # Pass 2: 1-bit QJL on residual
    residual_ct = qjl_1bit_compress(residual, projection_dim=projection_dim)

    return CompressedTensor(
        data=base_ct.data,
        metadata={
            **base_ct.metadata,
            "residual_signs": residual_ct.data,
            "residual_norms": residual_ct.metadata["norms"],
            "residual_seed": residual_ct.metadata["seed"],
            "residual_dim": projection_dim,
            "residual_d": residual_ct.metadata["d"],
        },
        original_shape=base_ct.original_shape,
        original_dtype=base_ct.original_dtype,
        technique="residual_correction",
        bits=base_bits,
    )


def residual_correction_decompress(ct):
    # Reconstruct base
    base_ct = CompressedTensor(
        data=ct.data, metadata=ct.metadata,
        original_shape=ct.original_shape, original_dtype=ct.original_dtype,
        technique="baseline_minmax", bits=ct.bits,
    )
    base = baseline_minmax_decompress(base_ct).astype(mx.float32)

    # Reconstruct residual
    residual_ct = CompressedTensor(
        data=ct.metadata["residual_signs"],
        metadata={
            "norms": ct.metadata["residual_norms"],
            "seed": ct.metadata["residual_seed"],
            "projection_dim": ct.metadata["residual_dim"],
            "d": ct.metadata["residual_d"],
        },
        original_shape=ct.original_shape,
        original_dtype=ct.original_dtype,
        technique="qjl_1bit", bits=1,
    )
    residual = qjl_1bit_decompress(residual_ct).astype(mx.float32)

    # Combine
    x = base + residual
    if "bfloat16" in ct.original_dtype:
        x = x.astype(mx.bfloat16)
    return x


# ══════════════════════════════════════════════════════
# TECHNIQUE 8: lloyd_max
# Non-uniform quantizer using iterative optimization
# Finds optimal quantization levels for actual data distribution
# ══════════════════════════════════════════════════════

def lloyd_max_compress(tensor, bits=3, group_size=64, iterations=10):
    x = tensor.astype(mx.float32)
    shape = x.shape
    x_flat = x.reshape(-1)
    n_levels = 1 << bits

    # Initialize levels uniformly
    x_np = np.array(x_flat)
    vmin, vmax = float(np.min(x_np)), float(np.max(x_np))
    levels = np.linspace(vmin, vmax, n_levels)

    # Lloyd-Max iteration
    for _ in range(iterations):
        # Assign each value to nearest level
        diffs = np.abs(x_np[:, None] - levels[None, :])
        assignments = np.argmin(diffs, axis=1)

        # Update levels to centroid of assigned values
        for i in range(n_levels):
            mask = assignments == i
            if np.any(mask):
                levels[i] = np.mean(x_np[mask])

    # Final assignment
    diffs = np.abs(x_np[:, None] - levels[None, :])
    assignments = np.argmin(diffs, axis=1).astype(np.uint8)

    return CompressedTensor(
        data=mx.array(assignments.reshape(shape)),
        metadata={"levels": mx.array(levels)},
        original_shape=shape,
        original_dtype=str(tensor.dtype),
        technique="lloyd_max",
        bits=bits,
    )


def lloyd_max_decompress(ct):
    levels = ct.metadata["levels"]
    indices = ct.data.astype(mx.int32)
    x = levels[indices]
    if "bfloat16" in ct.original_dtype:
        x = x.astype(mx.bfloat16)
    return x


# ══════════════════════════════════════════════════════
# REGISTRY — maps technique names to (compress, decompress) pairs
# ══════════════════════════════════════════════════════

TECHNIQUES = {
    "baseline_minmax": (baseline_minmax_compress, baseline_minmax_decompress),
    "polar_quant": (polar_quant_compress, polar_quant_decompress),
    "qjl_1bit": (qjl_1bit_compress, qjl_1bit_decompress),
    "hadamard_rotate": (hadamard_rotate_compress, hadamard_rotate_decompress),
    "mixed_kv": (mixed_kv_compress, mixed_kv_decompress),
    "per_layer_adaptive": (per_layer_adaptive_compress, per_layer_adaptive_decompress),
    "residual_correction": (residual_correction_compress, residual_correction_decompress),
    "lloyd_max": (lloyd_max_compress, lloyd_max_decompress),
}
