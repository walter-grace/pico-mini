# MoE Sniper — Claude Code Context

## What This Repo Is

This is a research project for running large MoE (Mixture-of-Experts) models on low-RAM Apple Silicon machines by streaming expert weights from SSD/USB instead of loading the full model into memory.

**Model:** Qwen3.5-35B-A3B — a 35B parameter multimodal MoE model (text + vision)
**Hardware:** 8 GB M2 MacBook Air, model files on USB flash drive
**Result:** 35B model generating coherent text and describing images on 8 GB RAM

## Key Files

| File | What it does |
|------|-------------|
| `agent.py` | Interactive agent using llama-server backend. Supports chat, web search (DuckDuckGo), shell commands, and vision (/image). Uses `--expert-cache-size` and `--mmproj` flags from our custom llama.cpp build. |
| `expert_io.py` | MoEExpertReader — reads individual experts from SSD via `F_NOCACHE` + `pread` with multi-threaded prefetch and LRU cache. |
| `moe_agent_macbook.py` | MLX-based inference engine — alternative path using MLX framework with gather_qmm fusion. Loads pinned weights (1.4 GB) + streams experts from split binary files. |
| `split_mlx_model_macbook.py` | Converts MLX safetensors model into flash stream format (pinned.safetensors + per-layer expert bins, 16KB aligned). |
| `flash_moe.py` | MoE engine with gather_qmm vectorized expert computation. |
| `batched_moe.py` | Union-of-experts batched verification + LRU cache for speculative decoding. |
| `direct_io.py` | F_NOCACHE direct I/O infrastructure for bypassing OS page cache. |
| `RESEARCH.md` | Full research writeup with measured results, architecture details, and scaling projections. |
| `test_search.py` | Web search test script (DuckDuckGo → LLM synthesis). |
| `test_vision.py` | Vision test script (sends image to multimodal model via llama-server API). |

## Two Inference Paths

### Path 1: llama.cpp + Expert Cache (production, recommended)
- Uses our custom llama.cpp build at `~/llama.cpp/` with `--expert-cache-size` flag
- GGUF model files on USB at `/Volumes/USB DISK/gguf/`
- Models: `Qwen3.5-35B-A3B-UD-IQ2_M.gguf` (10.6 GB), `Qwen3.5-35B-A3B-Q4_K_M.gguf` (21 GB)
- Vision projector: `mmproj-F16.gguf` (858 MB)
- Run: `python3 agent.py --model iq2 --cache 100`

### Path 2: MLX Expert Streaming (research, Apple-only)
- Custom inference using MLX framework with gather_qmm
- Split model files on USB at `/Volumes/USB DISK/qwen35-35b-moe-stream/`
- Run: `python3 moe_agent_macbook.py` or `python3 interactive_demo.py`

## Measured Results (verified on 8 GB M2 MacBook Air)

- **Text (IQ2_M via llama.cpp):** 0.24 tok/s, "2+2 equals 4" — correct
- **Text (Q4_K_M via llama.cpp):** Generates on 8 GB RAM (model is 21 GB = 2.6x RAM)
- **Vision (IQ2_M + mmproj):** 0.7 tok/s, correctly reads IP address from screenshot
- **Web search:** DuckDuckGo + LLM synthesis, returned live SpaceX Transporter-16 launch info
- **MLX expert streaming (Q4_K_M):** 0.03 tok/s from USB, "√69 ≈ 8.307" — correct
- **Memory:** 1.4 GB pinned (MLX path), 1.9 GB peak. Never exceeds 6 GB.
- **llama.cpp without our cache:** Q4_K_M mmap-thrashes for 15+ minutes with no output
- **llama.cpp 9B GPU:** OOMs on 8 GB even with partial GPU offload

## Custom llama.cpp Build

Located at `~/llama.cpp/`. Our additions:

| File | Purpose |
|------|---------|
| `src/llama-expert-cache.h` | LRU cache class — page-aligned alloc, pread disk loading, thread-safe |
| `src/llama-expert-cache.cpp` | Cache implementation — ensure(), get_or_alloc(), evict_until_free() |
| `src/llama-expert-cache-ctx.h` | Model integration — maps expert tensors, computes strides |
| `src/llama-expert-cache-ctx.cpp` | Eval callback — intercepts ggml_mul_mat_id, pre-caches active experts |
| `common/common.h` | Added `expert_cache_size` parameter |
| `common/arg.cpp` | Added `--expert-cache-size` CLI flag |
| `common/common.cpp` | Initializes cache at context creation, sets eval callback |
| `EXPERT_CACHE_PLAN.md` | Implementation plan with exact line numbers for all integration points |

Build: `cd ~/llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build -j4`

## Numerical Claims to Verify

If fact-checking RESEARCH.md, verify these against source code:

- **256 experts, 8 active per token:** Check config.json `num_experts=256`, `num_experts_per_tok=8`
- **~1.69 MB per expert:** Calculate from moe_intermediate_size=512, hidden_size=2048, 4-bit quantized (3 projections × 589,824 bytes each)
- **~540 MB per token read:** 8 × 1.69 MB × 40 layers
- **1.4 GB pinned:** Measured at runtime (moe_agent_macbook.py line 146)
- **6 GB memory limit:** `mx.set_memory_limit(6 * 1024**3)` in moe_agent_macbook.py line 137
- **4 workers (MacBook), 8 workers (16 GB):** moe_agent_macbook.py line 147 vs expert_io.py default
- **16 KB page alignment:** PAGE_SIZE = 16384 in split_mlx_model_macbook.py and expert_io.py
- **Top-p 0.9, repetition penalty 1.15:** moe_agent_macbook.py lines 241, 234
- **80% cache hit rate:** Estimated, not measured. Based on Zipf expert access patterns.
- **SSD throughput projections:** Hardware estimates, not code-derived. M2 ~3 GB/s, M4 Pro ~5.2 GB/s, Max ~7.4 GB/s.

## Dependencies

- Python 3.13 (`/opt/homebrew/bin/python3.13`)
- MLX (`mlx`, `mlx_lm`) — for Path 2 only
- transformers — tokenizer loading
- ddgs — web search (DuckDuckGo)
- rich — terminal UI (moe_agent_macbook.py only)
- llama.cpp — custom build at ~/llama.cpp/

## Common Commands

```bash
# Start the agent (text + vision + search)
cd ~/sniper && python3.13 agent.py --model iq2 --cache 100

# Test web search
python3.13 test_search.py

# Test vision
python3.13 test_vision.py ~/Desktop/screenshot.png

# Start llama-server manually
~/llama.cpp/build/bin/llama-server \
  -m "/Volumes/USB DISK/gguf/Qwen3.5-35B-A3B-UD-IQ2_M.gguf" \
  --mmproj "/Volumes/USB DISK/gguf/mmproj-F16.gguf" \
  -ngl 0 -c 2048 --expert-cache-size 100 --reasoning off --port 8199

# MLX path (slower on USB, research only)
python3.13 moe_agent_macbook.py
```
