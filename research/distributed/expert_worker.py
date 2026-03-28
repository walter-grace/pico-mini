#!/usr/bin/env python3
"""
Expert Worker — runs on the MacBook (or any second Mac).

Receives hidden states + expert IDs over TCP.
Loads experts from local RAM. Runs SwiGLU matmul on local GPU.
Returns expert output.

Usage:
    python3 expert_worker.py --expert-dir ~/models/remote-experts/ --port 9000
"""

import argparse
import json
import os
import struct
import socket
import time
import gc

import numpy as np
import mlx.core as mx
import mlx.nn as nn

BITS = 4
GROUP_SIZE = 64

# Message types
MSG_COMPUTE = 0
MSG_RESULT = 1
MSG_PING = 2
MSG_PONG = 3
MSG_SHUTDOWN = 255

# Header: type(1) + layer(1) + seq_len(2) + num_experts(1) + reserved(3) + payload_size(4) = 12 bytes
HEADER_SIZE = 12
HEADER_FMT = '<BBHBxxxI'


class ExpertStore:
    """Pre-loads expert weights into RAM for instant access."""

    def __init__(self, expert_dir, num_layers):
        self.expert_dir = expert_dir
        self.layers = {}
        self.num_layers = num_layers

    def load_all(self):
        """Load all expert layers into RAM."""
        print(f"Loading experts from {self.expert_dir}...")
        t0 = time.time()
        total_bytes = 0

        for fname in sorted(os.listdir(self.expert_dir)):
            if not fname.startswith("layer_") or not fname.endswith(".safetensors"):
                continue
            layer_idx = int(fname.split("_")[1].split(".")[0])
            path = os.path.join(self.expert_dir, fname)

            # Load as MLX arrays
            data = mx.load(path)
            self.layers[layer_idx] = data
            layer_bytes = sum(v.nbytes for v in data.values())
            total_bytes += layer_bytes
            print(f"  Layer {layer_idx}: {len(data)} tensors, {layer_bytes/1e6:.1f} MB")

        elapsed = time.time() - t0
        print(f"  Total: {len(self.layers)} layers, {total_bytes/1e9:.2f} GB in {elapsed:.1f}s")
        mx.eval(*[v for d in self.layers.values() for v in d.values()])
        print(f"  All weights evaluated and pinned in RAM")

    def get_expert_data(self, layer_idx, expert_ids):
        """Get weight/scales/biases for specific experts from a layer."""
        data = self.layers[layer_idx]
        result = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for comp in ["weight", "scales", "biases"]:
                key = f"{proj}.{comp}"
                full = data[key]  # [256, out, in]
                # Index active experts
                slices = mx.take(full, mx.array(expert_ids), axis=0)
                result[f"{proj}.{comp}"] = slices
        return result


def run_expert_ffn(x, expert_data, expert_ids, routing_weights):
    """
    Compute MoE FFN for the given hidden states using specified experts.
    Same math as moe_agent.py but on the worker device.
    """
    num_experts = len(expert_ids)
    output = mx.zeros_like(x)

    for i in range(num_experts):
        # Dequantize via mx.quantized_matmul
        gate = mx.quantized_matmul(
            x, expert_data[f"gate_proj.weight"][i],
            scales=expert_data[f"gate_proj.scales"][i],
            biases=expert_data[f"gate_proj.biases"][i],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        up = mx.quantized_matmul(
            x, expert_data[f"up_proj.weight"][i],
            scales=expert_data[f"up_proj.scales"][i],
            biases=expert_data[f"up_proj.biases"][i],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        hidden = nn.silu(gate) * up
        down = mx.quantized_matmul(
            hidden, expert_data[f"down_proj.weight"][i],
            scales=expert_data[f"down_proj.scales"][i],
            biases=expert_data[f"down_proj.biases"][i],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        output = output + routing_weights[i] * down

    return output


def handle_compute(store, payload):
    """
    Process a compute request.

    Payload format:
      - expert_ids: [num_experts] int32
      - routing_weights: [num_experts] float32
      - hidden_state: [seq_len, hidden_dim] float16
    """
    # Parse expert IDs (first 4 bytes per expert)
    offset = 0
    num_experts = struct.unpack_from('<I', payload, offset)[0]
    offset += 4
    expert_ids = list(struct.unpack_from(f'<{num_experts}I', payload, offset))
    offset += num_experts * 4

    # Parse routing weights
    routing_weights_raw = struct.unpack_from(f'<{num_experts}f', payload, offset)
    offset += num_experts * 4
    routing_weights = mx.array(list(routing_weights_raw))

    # Parse layer index
    layer_idx = struct.unpack_from('<I', payload, offset)[0]
    offset += 4

    # Parse hidden state dimensions
    seq_len, hidden_dim = struct.unpack_from('<II', payload, offset)
    offset += 8

    # Parse hidden state (float16)
    hidden_bytes = seq_len * hidden_dim * 2
    hidden_np = np.frombuffer(payload[offset:offset + hidden_bytes], dtype=np.float16)
    hidden_state = mx.array(hidden_np.reshape(seq_len, hidden_dim))

    # Load expert data
    expert_data = store.get_expert_data(layer_idx, expert_ids)

    # Compute
    result = run_expert_ffn(hidden_state, expert_data, expert_ids, routing_weights)
    mx.eval(result)

    # Serialize result back to bytes (float16)
    result_np = np.array(result, dtype=np.float16)
    return result_np.tobytes()


def serve(store, host, port):
    """Run the expert worker TCP server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    print(f"\nExpert Worker listening on {host}:{port}")
    print(f"Layers available: {sorted(store.layers.keys())}")
    print(f"Waiting for orchestrator connection...")

    while True:
        conn, addr = sock.accept()
        print(f"Connected: {addr}")

        try:
            while True:
                # Read header
                header_data = b''
                while len(header_data) < HEADER_SIZE:
                    chunk = conn.recv(HEADER_SIZE - len(header_data))
                    if not chunk:
                        raise ConnectionError("Connection closed")
                    header_data += chunk

                msg_type, layer_idx, seq_len, num_experts, payload_size = struct.unpack(HEADER_FMT, header_data)

                if msg_type == MSG_PING:
                    # Respond with pong
                    conn.sendall(struct.pack(HEADER_FMT, MSG_PONG, 0, 0, 0, 0))
                    continue

                if msg_type == MSG_SHUTDOWN:
                    print("Shutdown requested")
                    conn.close()
                    return

                if msg_type == MSG_COMPUTE:
                    # Read payload
                    payload = b''
                    while len(payload) < payload_size:
                        chunk = conn.recv(min(65536, payload_size - len(payload)))
                        if not chunk:
                            raise ConnectionError("Connection closed during payload")
                        payload += chunk

                    # Compute
                    t0 = time.time()
                    result_bytes = handle_compute(store, payload)
                    compute_time = time.time() - t0

                    # Send response
                    resp_header = struct.pack(HEADER_FMT, MSG_RESULT, layer_idx, seq_len, 0, len(result_bytes))
                    conn.sendall(resp_header + result_bytes)

        except (ConnectionError, BrokenPipeError) as e:
            print(f"Connection lost: {e}")
            conn.close()
            print("Waiting for reconnection...")


def main():
    parser = argparse.ArgumentParser(description="Expert Worker for Distributed MoE Sniper")
    parser.add_argument("--expert-dir", required=True, help="Directory with layer_XX.safetensors files")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=9000, help="Listen port")
    args = parser.parse_args()

    # Count layers
    layer_files = [f for f in os.listdir(args.expert_dir) if f.startswith("layer_") and f.endswith(".safetensors")]
    num_layers = len(layer_files)
    print(f"Found {num_layers} layer files in {args.expert_dir}")

    store = ExpertStore(args.expert_dir, num_layers)
    store.load_all()

    serve(store, args.host, args.port)


if __name__ == "__main__":
    main()
