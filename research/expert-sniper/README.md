# MoE Sniper — 35B AI on 8 GB RAM

Run a 35B multimodal AI model on an 8 GB MacBook. Text, vision, and web search — from a USB drive.

```
  moe-sniper
  35B multimodal AI on consumer hardware
  ──────────────────────────────────────────────────
  Model     Qwen3.5-35B-A3B (10.6 GB)
  Hardware  Apple M2 (8 GB RAM)
  Storage   USB flash drive
  Tools     chat, search, shell, vision
  ──────────────────────────────────────────────────
  Model is 1.3x larger than RAM.
  Expert cache makes this possible.
```

## How It Works

Qwen3.5-35B-A3B is a Mixture-of-Experts model with 256 experts per layer, but only 8 activate per token — meaning 97% of the model is unused for any given computation.

We built an **expert-aware LRU cache** into llama.cpp that exploits this:
- Hot experts stay pinned in RAM (~1.4 GB)
- Cold experts live on disk (USB drive or SSD)
- The OS can reclaim unused mmap pages without affecting performance
- Result: a 10.6 GB model runs in 1.4 GB of RAM

Without our cache, llama.cpp mmap-thrashes for 15+ minutes with zero output. With it, the model generates coherent responses, describes images, and answers web search queries.

## Quick Start

### 1. Build llama.cpp with expert cache

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Copy our expert cache files into the source
cp /path/to/sniper/llama-expert-cache*.h src/
cp /path/to/sniper/llama-expert-cache*.cpp src/

# Apply patches (add source files to CMakeLists, add --expert-cache-size flag)
# See EXPERT_CACHE_PLAN.md for exact changes

cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server
```

### 2. Download model files

You need three files (~12 GB total). Put them anywhere — USB drive, external SSD, or internal storage:

```bash
# Using huggingface-cli:
pip install huggingface-hub

# The model (pick one):
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-IQ2_M.gguf --local-dir ./models
# IQ2_M = 10.6 GB, fits on 16 GB machines via mmap, works on 8 GB with cache

# Or for higher quality (needs more cache):
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir ./models
# Q4_K_M = 21 GB, needs expert cache on any machine under 24 GB

# Vision projector (required for /image command):
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  mmproj-F16.gguf --local-dir ./models

# Web search (required for /search command):
pip install ddgs
```

### 3. Run the agent

```bash
cd sniper

# Point to your model directory:
python3 sniper.py --model-dir /Volumes/USB\ DISK/gguf --model iq2 --cache 100

# Or if models are in ~/models:
python3 sniper.py --model-dir ~/models --model iq2 --cache 3000
```

### USB Drive Setup

Any USB drive works. Format as APFS (Mac) or exFAT (cross-platform). Just copy the GGUF files onto it:

```
USB Drive/
  Qwen3.5-35B-A3B-UD-IQ2_M.gguf   (10.6 GB)
  mmproj-F16.gguf                   (858 MB)
```

Plug it in, run:
```bash
python3 sniper.py --model-dir "/Volumes/YOUR_DRIVE" --model iq2 --cache 100
```

The expert cache compensates for USB read speeds by keeping hot experts in RAM.

## Commands

| Command | What it does |
|---------|-------------|
| (just type) | Chat with the 35B model |
| `/search <query>` | Web search via DuckDuckGo + AI synthesis |
| `/image <path>` | Describe an image (vision) |
| `/shell <task>` | AI generates + executes a shell command |
| `/stats` | Show token speed and counts |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

## GPU Offloading (`--ngl`)

The `--ngl` flag controls how many model layers run on GPU vs CPU.

**Tested on M2 MacBook Air (8 GB):** GPU offload OOMs even at `ngl 2`. The 10.6 GB model + GPU buffers exceed the 5.7 GB available VRAM. Use `--ngl 0` (CPU-only) on 8 GB machines.

```bash
# 8 GB machine — CPU only (tested, works)
python3 sniper.py --model iq2 --cache 100 --ngl 0

# 16+ GB machine — try GPU offload (untested, should help)
python3 sniper.py --model iq2 --cache 3000 --ngl 20
```

Machines with more RAM can try `--ngl 10`, `--ngl 20`, etc. Start low and increase until you see GPU memory errors, then back off.

## Tested Results

All measured on an **8 GB M2 MacBook Air** with model on **USB flash drive**:

| Capability | Result |
|-----------|--------|
| **Text** | "2+2 equals 4" — correct, 0.24 tok/s |
| **Text (Q4_K_M, 21 GB)** | "2+2 equals 4" — model is 2.6x larger than RAM |
| **Vision** | Correctly read IP address from screenshot, 0.7 tok/s |
| **Web search** | Found live SpaceX Transporter-16 launch info, 0.24 tok/s |
| **RAM usage** | 1,389 MB RSS for a 10,600 MB model |
| **llama.cpp baseline** | OOMs on 9B GPU, thrashes on 35B — our cache fixes both |

Settings used: `--model iq2 --cache 100 --ngl 0 --ctx 2048`

## Architecture

```
┌─────────────────────────────────────────────┐
│              sniper.py (agent)              │
│  chat • search • shell • vision             │
└──────────────┬──────────────────────────────┘
               │ HTTP (localhost:8199)
┌──────────────▼──────────────────────────────┐
│     llama-server + expert-aware cache       │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ LRU Cache   │  │ ggml_mul_mat_id      │  │
│  │ hot experts │◄─│ eval callback        │  │
│  │ in RAM      │  │ intercepts expert ops│  │
│  └──────┬──────┘  └──────────────────────┘  │
│         │ cache miss                         │
│  ┌──────▼──────┐                             │
│  │ mmap/pread  │                             │
│  │ from disk   │                             │
│  └──────┬──────┘                             │
└─────────┼───────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────┐
│  USB Drive / SSD / NVMe                     │
│  Qwen3.5-35B-A3B-UD-IQ2_M.gguf (10.6 GB)  │
│  mmproj-F16.gguf (858 MB vision)            │
└─────────────────────────────────────────────┘
```

## Research

See [RESEARCH.md](RESEARCH.md) for full technical details on the MoE expert streaming architecture, implementation, and measured results.

## License

MIT — [github.com/walter-grace/mac-code](https://github.com/walter-grace/mac-code)
