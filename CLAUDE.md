# mac code — Setup Instructions for Claude Code

This file tells Claude Code how to install and run mac code on any Mac with Apple Silicon.

## What This Project Is

mac code is a local AI coding agent (like Claude Code) that runs entirely on your Mac using a local LLM via llama.cpp. No cloud, no API keys, no cost.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ RAM recommended)
- Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- Python 3.10+

## Installation

### Quick install

```bash
bash setup.sh
```

### Manual install

```bash
# 1. Install dependencies
brew install llama.cpp
pip3 install huggingface-hub rich ddgs --break-system-packages

# 2. Download model (35B MoE — 30 tok/s on 16 GB)
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"

# 3. Start server
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 12288 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4

# 4. Run agent (in another terminal)
python3 agent.py
```

### Alternative: 9B model (64K context)

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4

python3 agent.py
```

## File Overview

- `agent.py` — Main agent TUI with auto-routing, slash commands, web search, tools
- `chat.py` — Lightweight streaming chat (no tools, direct to LLM)
- `dashboard.py` — Real-time server monitor (tok/s, slots, memory)
- `setup.sh` — One-command install script

## Architecture

The agent uses text-based intent routing. The LLM classifies each message as `search`, `shell`, or `chat`:

- **search** → DuckDuckGo search + LLM summarizes results (~5s)
- **shell** → LLM generates command → executes → LLM summarizes output (~3s)
- **chat** → Direct streaming response via llama.cpp

Two models:
- **35B MoE (IQ2_M, 10.6 GB)** — Default. 30 tok/s, fits in 16 GB RAM. 12K context.
- **9B (Q4_K_M)** — 64K context with quantized KV cache.

Switch with `/model 9b` or `/model 35b`.

## Common Issues

- **GPU OOM after long sessions**: Reboot the Mac to clear Metal GPU memory, then restart the server
- **Connection refused**: Make sure llama-server is running on port 8000 before starting agent.py
- **Model download fails**: Ensure `huggingface-hub` is installed and you have ~11 GB free disk space
- **Slow on 8 GB Mac**: Use a smaller model (Qwen3.5-4B or IQ2_XXS quantization)

## Key Paths

- Models: `~/models/`
- Interaction logs: `~/.mac-code/logs/`

---

## Expert Sniper (MoE inference larger than RAM)

### What it is
MLX-based MoE inference engine for Apple Silicon. Runs models
larger than RAM by only loading active experts from SSD.

### Quick reference
- CLI package: `research/expert-sniper/cli-agent/`
- Main engine: `research/expert-sniper/mlx-sniper/`
- llama.cpp patch: `research/expert-sniper/llama-cpp/`
- Ternary fallback research: `research/1bit-fallback/`

### Key files
- `cli-agent/src/mlx_expert_sniper/cli.py` — CLI entry point
- `cli-agent/src/mlx_expert_sniper/engine.py` — forward pass engine (35B)
- `cli-agent/src/mlx_expert_sniper/expert_io.py` — SSD streaming + LRU cache
- `cli-agent/src/mlx_expert_sniper/calibrate.py` — one-time calibration
- `cli-agent/src/mlx_expert_sniper/coactivation.py` — cross-layer prediction
- `cli-agent/src/mlx_expert_sniper/preprocess.py` — model split/preprocess

### Working commands
```
mlx-sniper calibrate <model-dir> [--quick] [--force]
mlx-sniper run <model-dir> -p "prompt" [-v]
```

### Not yet implemented
```
mlx-sniper download <model-name>  — needs preprocess integration
mlx-sniper chat <model-dir>       — interactive mode
mlx-sniper server <model-dir>     — OpenAI-compatible API
```

### Architecture
The engine streams expert weights from SSD using pread + F_NOCACHE,
bypassing the macOS page cache. A right-sized LRU cache keeps hot
experts in Metal unified memory. Co-activation prediction prefetches
likely experts. Cache-aware routing biases the router toward cached
experts.

Three optimizations give 6.9x speedup:
1. Right-sized cache (don't overfill RAM)
2. Co-activation prediction (70% cross-layer accuracy)
3. Routing bias (nudge router toward cached experts)

Calibrate runs once per model and saves the config.

### Model format
Models must be preprocessed into "sniper streaming format":
- `pinned.safetensors` — attention + norms (stays in RAM)
- `bin/layer_XX.bin` — expert weights per layer (streamed from SSD)
- `config.json` — model config
- `sniper_config.json` — calibration output
- `sniper_calibration.npz` — co-activation matrix + REAP scores

### How to add a new model
1. Download the MLX 4-bit version from HuggingFace
2. Write a split script (see `mlx-sniper/split_35b_v2.py` as template)
3. Run `mlx-sniper calibrate` on the processed model
4. Test with `mlx-sniper run`

### Dependencies
mlx, numpy, transformers, safetensors, huggingface_hub

### Hardware
Apple Silicon only (M1/M2/M3/M4). 16 GB RAM minimum for
the MLX path. 8 GB works with llama.cpp madvise path only.
