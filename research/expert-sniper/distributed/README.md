# mac-tensor

**Run 35B parameter LLMs across multiple Macs over the network.**

Each Mac only needs 10-13 GB RAM. No GPU cluster. No CUDA. Just Macs and Python.

```
┌─────────────────────┐
│   COORDINATOR       │  Mac 1 — runs attention + routing
│   1.5 GB RAM        │  
└─────────┬───────────┘
          │ HTTP
    ┌─────┴─────┐
    ▼           ▼
┌──────────┐ ┌──────────┐
│  NODE A  │ │  NODE B  │  Mac 2 & 3 — hold experts in RAM
│  ~10 GB  │ │  ~10 GB  │  compute FFN, return results
└──────────┘ └──────────┘
```

## Why This Works

MoE (Mixture-of-Experts) models only activate ~3% of parameters per token. The other 97% sit idle. Instead of cramming everything into one machine, we split the expert weights across Macs and compute in parallel.

- **Coordinator** holds the small always-needed weights (attention, embeddings, router) ~1.5 GB
- **Expert nodes** each hold a partition of the FFN experts in RAM
- Router picks which experts to activate → coordinator sends hidden state to nodes → nodes compute and return → coordinator sums results

## Current Results

| Model | Total Params | Active/Token | Nodes | Speed |
|-------|-------------|-------------|-------|-------|
| Qwen 3.5-35B-A3B | 35B | 3.5B | 3 Mac Minis | **1.30 tok/s** |
| Gemma 4-26B-A4B | 26B | 3.8B | 3 Mac Minis | **1.23 tok/s** |

Tested on Scaleway Mac Mini M2 (16 GB, ~$0.13/hr each).

> **These speeds are not fast.** We know. The bottleneck is 30-40 sequential HTTP round trips per token over the network. We're actively optimizing (raw TCP, pipelining, better batching) and **open-sourcing so the community can help push this further.** If you have ideas — open an issue or PR.

## Single Mac? Use mlx-sniper Instead

If you only have **one Mac**, you don't need the distributed setup. The single-machine Expert Sniper streams experts from SSD and is significantly faster:

| Setup | Speed | Hardware |
|-------|-------|----------|
| **Single Mac (mlx-sniper)** | **5.37 tok/s** | 1x M4 Mac Mini 16 GB |
| Distributed (mac-tensor) | 1.30 tok/s | 3x Mac Mini M2 16 GB |

```bash
# Single Mac — just install and go
cd ../mlx-sniper && pip install -e .
mlx-sniper chat ~/models/qwen3-30b
```

See [mlx-sniper/README.md](../mlx-sniper/README.md) for the single-machine setup. It supports Qwen3-30B, Qwen3.5-35B, and more via SSD expert streaming with LRU cache + routing bias.

**When to use distributed (mac-tensor) instead:**
- You have multiple cheap Macs and want to pool their RAM
- The model doesn't fit on a single machine (even with SSD streaming)
- You want all experts in RAM (zero SSD latency, no cache misses)
- You're running a cloud fleet (e.g., Scaleway Mac Minis at $0.13/hr each)

## Install

```bash
cd distributed/
pip install -e .
```

This gives you the `mac-tensor` CLI. Or run directly with `python3 -m mac_tensor`.

## Quick Start (CLI)

```bash
# See supported models
mac-tensor info

# Download model (on each Mac)
mac-tensor download --model qwen35

# Split for distributed use (on each Mac, Qwen only)
mac-tensor split --model qwen35

# Start expert nodes
mac-tensor node --model qwen35 --partition 0-127 --port 8301    # Mac 2
mac-tensor node --model qwen35 --partition 128-255 --port 8301  # Mac 3

# Check nodes are ready
mac-tensor health --nodes http://mac2:8301,http://mac3:8301

# Chat!
mac-tensor chat --model qwen35 --nodes http://mac2:8301,http://mac3:8301
```

## Quick Start (Manual)

### Requirements
- 2-3 Macs with Apple Silicon (M1/M2/M3/M4), 16 GB RAM each
- Network connection between them (LAN, same datacenter, etc.)
- Python 3.9+

```bash
pip install mlx mlx-lm numpy fastapi uvicorn requests transformers tokenizers huggingface_hub
```

### Qwen 3.5-35B (256 experts, 40 layers)

**1. Download model (on each Mac):**
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-35B-A3B-4bit', local_dir='~/models/qwen35-4bit')
"
```

**2. Split model (on each Mac):**
```bash
python3 split_qwen.py
# Creates: ~/models/qwen35-stream/{pinned.safetensors, bin/layer_XX.bin}
```

**3. Start expert nodes:**
```bash
# Mac 2:
python3 expert_node_fast.py --partition 0-127 --model-dir ~/models/qwen35-stream --port 8301

# Mac 3:
python3 expert_node_fast.py --partition 128-255 --model-dir ~/models/qwen35-stream --port 8301
```

**4. Wait ~90 seconds, then start chatting:**
```bash
# Mac 1 (coordinator):
python3 distributed_interactive.py \
  --nodes http://<MAC2>:8301,http://<MAC3>:8301 \
  --max-tokens 300
```

### Gemma 4-26B (128 experts, 30 layers)

**1. Download model (on each Mac):**
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit', local_dir='~/models/gemma4-4bit')
"
```

**2. Start expert nodes (no split step needed):**
```bash
# Mac 2:
python3 gemma4_expert_node.py --partition 0-63 --model-dir ~/models/gemma4-4bit --port 8401

# Mac 3:
python3 gemma4_expert_node.py --partition 64-127 --model-dir ~/models/gemma4-4bit --port 8401
```

**3. Wait ~80 seconds, then start chatting:**
```bash
# Mac 1 (coordinator):
python3 gemma4_distributed.py \
  --nodes http://<MAC2>:8401,http://<MAC3>:8401 \
  --max-tokens 200
```

## How It Works

### The Split

MoE models have two types of weights:
- **Pinned** (~1.5 GB): attention layers, embeddings, layer norms, router — needed for every token
- **Experts** (~13-18 GB): FFN expert weights — only ~8 of 128-256 activated per token per layer

The coordinator loads pinned weights. Expert nodes load their partition of experts into RAM.

### The Forward Pass (per token)

For each of 30-40 layers:
1. **Attention** (coordinator, local) — standard self-attention with KV cache
2. **Router** (coordinator, local) — softmax gate selects top-8 experts
3. **Expert FFN** (distributed) — coordinator sends hidden state to all nodes in parallel, each node computes its partition's experts via `gather_qmm`, returns weighted output
4. **Sum** (coordinator, local) — add all partial results + shared/dense MLP output

### Binary Transport Protocol

Expert nodes serve a `/compute_bin` endpoint with raw binary payloads (no JSON/base64 overhead):
```
Request:  [layer_idx:u16][n_experts:u16][expert_ids:u16*N]
          [shapes:u16*9][hidden_state:f16][indices:i32][weights:f32]

Response: [n_computed:u16][ndim:u16][shape:u16*ndim][result:f16]
```

Connection pooling via `requests.Session` keeps TCP connections alive across requests.

## File Structure

```
distributed/
├── expert_node_fast.py          # Qwen expert partition server
├── gemma4_expert_node.py        # Gemma 4 expert partition server
├── distributed_reader_fast.py   # Client-side reader (connection pooling + binary)
├── distributed_interactive.py   # Qwen coordinator + interactive chat
├── gemma4_distributed.py        # Gemma 4 coordinator + interactive chat
├── split_qwen.py                # Split Qwen MLX model into streaming format
├── split_gemma4.py              # Split Gemma 4 MLX model (optional)
├── models_gemma4.py             # Custom Gemma 4 model definition for MLX
├── CLAUDE.md                    # Instructions for Claude Code to spin this up
└── README.md                    # This file
```

## Network Setup — LAN, Thunderbolt, and Cloud

mac-tensor works over any TCP network. The faster your connection, the faster inference runs. Here's how different setups compare:

| Connection | Latency | Expected Boost | Setup |
|-----------|---------|----------------|-------|
| Cloud (Scaleway) | ~6ms RTT | Baseline (1.3 tok/s) | Default — just use public IPs |
| Home LAN (Ethernet) | ~0.3ms RTT | **~3-4x faster** | Connect Macs to same switch/router |
| Thunderbolt Bridge | ~0.05ms RTT | **~5-8x faster** | Direct Thunderbolt cable between Macs |

### Home LAN Setup
Just plug your Macs into the same router/switch via Ethernet. Find each Mac's IP:
```bash
# On each Mac:
ipconfig getifaddr en0    # Ethernet
ipconfig getifaddr en1    # Wi-Fi (slower, not recommended)
```
Then use those IPs in the `--nodes` argument. Gigabit Ethernet is fine — each layer only transfers ~5 KB.

### Thunderbolt Bridge (Fastest)
Connect two Macs directly via a Thunderbolt cable for near-zero latency:

1. Connect Thunderbolt cable between Macs
2. On both Macs: **System Settings → Network → Thunderbolt Bridge**
3. Set manual IPs (e.g., Mac A: `169.254.1.1`, Mac B: `169.254.1.2`)
4. Use those IPs in `--nodes`:
```bash
python3 distributed_interactive.py --nodes http://169.254.1.1:8301,http://169.254.1.2:8301
```

Thunderbolt 3/4 gives 40 Gbps with sub-millisecond latency. Since our bottleneck is round-trip latency (not bandwidth), this is the single biggest speed improvement you can make.

### Wi-Fi
It works, but Wi-Fi adds jitter and latency spikes. Expect ~50-70% of Ethernet speed. Use Ethernet if possible.

## Known Limitations & Optimization Ideas

**Current bottleneck:** 30-40 sequential HTTP round trips per token (~12ms each on typical network = 360-480ms of network time per token).

Ideas we're exploring (PRs welcome!):
- **Raw TCP sockets** instead of HTTP to eliminate parsing overhead
- **Pipelining** attention layer i+1 while waiting for expert results from layer i
- **Larger partitions** — 4+ nodes to reduce per-node compute time
- **Mixed local+remote** — keep hot experts on coordinator, only dispatch cold ones
- **WebSocket** persistent connections instead of request/response per layer
- **Compression** — quantize hidden states to int8 for transport (5KB → 2.5KB per layer)

## Roadmap

- [x] CLI tool: `mac-tensor node`, `mac-tensor chat`, `mac-tensor health`
- [ ] Auto-discovery of Macs on local network
- [ ] Agent mode with tool use
- [ ] Support for more MoE models (Mixtral, DeepSeek, etc.)
- [ ] Performance dashboard / monitoring
- [ ] Benchmarking suite

## Contributing

This is early-stage research code. We're open-sourcing it because we believe distributed MoE inference on consumer hardware is an important problem, and we want the community's help pushing the speed further.

If you have ideas, benchmarks, or optimizations — **open an issue or PR.** We're especially interested in:
- Network transport optimizations
- New MoE model support
- Real-world usage reports on different hardware configs

## Built With

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [FastAPI](https://fastapi.tiangolo.com/) — expert node HTTP server
- [Hugging Face](https://huggingface.co/mlx-community) — MLX model weights

## License

MIT
