"""
Expert Client — used by the Mac mini orchestrator to send work to remote workers.

Sends hidden states + expert IDs to a worker over TCP.
Receives computed expert output.

Usage:
    from expert_client import RemoteExpertClient

    client = RemoteExpertClient("macbook.local", 9000)
    client.connect()
    result = client.compute(layer_idx=24, hidden_state=h, expert_ids=[3,7,12,...], routing_weights=[0.1,...])
"""

import socket
import struct
import time
import numpy as np
import mlx.core as mx

MSG_COMPUTE = 0
MSG_RESULT = 1
MSG_PING = 2
MSG_PONG = 3
MSG_SHUTDOWN = 255

HEADER_SIZE = 12
HEADER_FMT = '<BBHBxxxI'


class RemoteExpertClient:
    """Client for sending expert computation to a remote worker."""

    def __init__(self, host, port, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.connected = False
        self.total_compute_time = 0
        self.total_network_time = 0
        self.total_calls = 0

    def connect(self):
        """Connect to the remote worker."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))
        self.connected = True
        print(f"Connected to worker at {self.host}:{self.port}")

        # Ping to verify
        self.sock.sendall(struct.pack(HEADER_FMT, MSG_PING, 0, 0, 0, 0))
        resp = self.sock.recv(HEADER_SIZE)
        msg_type = struct.unpack_from('<B', resp, 0)[0]
        if msg_type == MSG_PONG:
            print(f"  Worker responded: PONG")
        return True

    def compute(self, layer_idx, hidden_state, expert_ids, routing_weights):
        """
        Send a compute request to the remote worker.

        Args:
            layer_idx: int — which layer
            hidden_state: mx.array [seq_len, hidden_dim] — the hidden state
            expert_ids: list of int — which experts to compute
            routing_weights: list/mx.array of float — routing weights per expert

        Returns:
            mx.array [seq_len, hidden_dim] — expert output
        """
        if not self.connected:
            raise RuntimeError("Not connected to worker")

        t_start = time.time()

        # Serialize hidden state to float16 bytes
        hidden_np = np.array(hidden_state, dtype=np.float16)
        seq_len, hidden_dim = hidden_np.shape
        hidden_bytes = hidden_np.tobytes()

        # Build payload
        num_experts = len(expert_ids)
        payload = b''
        payload += struct.pack('<I', num_experts)
        payload += struct.pack(f'<{num_experts}I', *expert_ids)
        payload += struct.pack(f'<{num_experts}f', *[float(w) for w in routing_weights])
        payload += struct.pack('<I', layer_idx)
        payload += struct.pack('<II', seq_len, hidden_dim)
        payload += hidden_bytes

        # Send header + payload
        header = struct.pack(HEADER_FMT, MSG_COMPUTE, layer_idx, seq_len, num_experts, len(payload))

        t_send = time.time()
        self.sock.sendall(header + payload)

        # Receive response
        resp_header = b''
        while len(resp_header) < HEADER_SIZE:
            chunk = self.sock.recv(HEADER_SIZE - len(resp_header))
            if not chunk:
                raise ConnectionError("Worker disconnected")
            resp_header += chunk

        msg_type, _, _, _, result_size = struct.unpack(HEADER_FMT, resp_header)

        result_data = b''
        while len(result_data) < result_size:
            chunk = self.sock.recv(min(65536, result_size - len(result_data)))
            if not chunk:
                raise ConnectionError("Worker disconnected during result")
            result_data += chunk

        t_recv = time.time()

        # Deserialize result
        result_np = np.frombuffer(result_data, dtype=np.float16).reshape(seq_len, hidden_dim)
        result = mx.array(result_np)

        # Stats
        self.total_calls += 1
        self.total_network_time += (t_recv - t_send)
        self.total_compute_time += (t_recv - t_start)

        return result

    def stats(self):
        """Return timing statistics."""
        if self.total_calls == 0:
            return {"calls": 0}
        return {
            "calls": self.total_calls,
            "avg_network_ms": (self.total_network_time / self.total_calls) * 1000,
            "avg_total_ms": (self.total_compute_time / self.total_calls) * 1000,
        }

    def disconnect(self):
        """Gracefully disconnect."""
        if self.sock:
            try:
                self.sock.sendall(struct.pack(HEADER_FMT, MSG_SHUTDOWN, 0, 0, 0, 0))
            except:
                pass
            self.sock.close()
            self.connected = False

    def __del__(self):
        self.disconnect()
