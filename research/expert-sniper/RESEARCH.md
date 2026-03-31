# MoE Expert Streaming: Running 35B Models on 8 GB RAM

## Problem

Qwen3.5-35B-A3B is 21 GB at Q4_K_M (4-bit quantization, group_size=64). It will not fit in 8 GB RAM. llama.cpp and standard inference frameworks require the full model in memory and OOM on machines with less than 24 GB. The only workaround in llama.cpp is aggressive quantization to IQ2_M (2-bit, 10.6 GB), which degrades output quality and still requires 16 GB.

## Key Insight

Qwen3.5-35B-A3B is a Mixture-of-Experts model with 256 experts per layer, activating only 8 per token (top-8 routing with softmax gating and normalized top-k probabilities). This means only ~3.1% of FFN weights are needed for any given token.

Per-token read budget at Q4_K_M:
- Each expert: ~1.69 MB (gate_proj + up_proj + down_proj, each with weight/scales/biases)
- Per layer: 8 experts x 1.69 MB = 13.5 MB
- Per token (40 layers): ~540 MB
- At 3-5 GB/s NVMe throughput (varies by Apple Silicon model): 108-180 ms/token = ~5.6-9.3 tok/s theoretical

The core idea: keep the small, always-needed weights pinned in RAM and stream only the active experts from SSD on demand.

## Architecture

### Pinned in RAM (~1.4 GB)
- Embedding layer (`model.embed_tokens`)
- All attention weights (full attention every 4th layer, linear/SSM attention otherwise)
- SSM parameters (`A_log`, `dt_bias`, `conv1d`)
- Router gates (256-way classifiers, one per layer)
- Shared experts (one per layer, `shared_expert` + `shared_expert_gate`)
- Layer norms (`input_layernorm`, `post_attention_layernorm`)
- Final norm and `lm_head`

### Streamed from SSD (~20.6 GB)
- 256 experts x 40 layers = 10,240 expert blocks
- Each expert block contains 9 tensors: `{gate,up,down}_proj.{weight,scales,biases}`
- Stored in 40 binary files (`layer_00.bin` through `layer_39.bin`), one per layer
- 16 KB page-aligned layout for direct I/O compatibility

## Implementation

### Model Splitting: `split_mlx_model_macbook.py`

Converts a pre-converted MLX safetensors model into the flash stream format:

1. Reads `model.safetensors.index.json` to categorize all weights as pinned vs expert
2. Saves pinned weights to `pinned.safetensors`
3. For each of the 40 layers, writes a `layer_XX.bin` file containing:
   - 16 KB JSON header with layout metadata (expert_block_size, data_start, tensor offsets/shapes/dtypes)
   - 256 expert blocks, each 16 KB-aligned (`align_up` to `PAGE_SIZE = 16384`)
   - Expert data: raw bytes for each of the 9 quantized tensors

Peak memory during splitting: ~2 GB (1.5 GB pinned + 0.5 GB per expert layer being processed). Processes one layer at a time, groups expert weights by shard to minimize shard reloads.

### Expert I/O: `expert_io.py`

`MoEExpertReader` provides low-level SSD access to individual experts:

- Opens layer files with `os.O_RDONLY` and sets `F_NOCACHE = 48` via `fcntl` to bypass the OS page cache (direct I/O). This prevents 21 GB of expert data from evicting useful pages from the 8 GB unified memory.
- Reads expert blocks via `os.pread(fd, expert_block_size, offset)` where `offset = data_start + expert_id * expert_block_size`. Random access, no seeking.
- Multi-threaded parallel reads via `ThreadPoolExecutor` (4 workers on 8 GB MacBook, 8 workers on 16 GB machines) to saturate NVMe/USB queue depth.
- Asynchronous prefetch: `prefetch_experts(layer_idx+1, active_ids)` launches reads for the next layer while the current layer computes, overlapping I/O with compute.
- Parses raw bytes into MLX arrays: `np.frombuffer` to uint8, then `mx.array.view(dtype).reshape(shape)`.
- Tracks stats: read count, average latency, throughput (GB/s), total bytes read.

### Inference Engine: `moe_agent_macbook.py`

`MoESniperEngine` runs full inference on 8 GB hardware:

- Memory limits: `mx.set_memory_limit(6 GB)`, `mx.set_cache_limit(256 MB)`
- Loads `pinned.safetensors` with `strict=False` (expert weights intentionally missing)
- Per-layer forward pass:
  1. Layer norm + attention/SSM (from pinned RAM)
  2. `mx.eval(h)` to materialize before routing
  3. Router: softmax over 256-way gate, `argpartition` for top-8, normalize scores
  4. `mx.eval(inds, scores)` to get concrete expert IDs
  5. Prefetch next layer's experts (using current layer's active IDs as heuristic)
  6. Load active experts from SSD via `reader.get_experts()`
  7. `run_expert_ffn()`: gather_qmm fusion (see below)
  8. Add shared expert output (gated by sigmoid)
  9. Residual connection, `mx.eval(h)`, free expert data, `mx.clear_cache()`

### gather_qmm Fusion: `run_expert_ffn()` in `flash_moe.py` / `moe_agent_macbook.py`

Vectorized quantized matmul across active experts using MLX's `mx.gather_qmm`:

1. Stack 8 active experts into `[K, out, in]` mini-tensors (weight, scales, biases for each projection)
2. Remap global expert IDs (0-255) to local indices (0-7) via `np.vectorize`
3. Three `mx.gather_qmm` calls implement SwiGLU: `silu(gate_proj(x)) * up_proj(x)` then `down_proj(hidden)`
4. Squeeze expert dimension, weight by router scores, sum across experts

This matches the native `SwitchGLU` math exactly while operating only on the loaded subset of experts.

### Agent Layer: `moe_agent_macbook.py`

Full agent with:
- Keyword-based intent classification (search, shell, chat)
- Web search via DuckDuckGo (`ddgs` / `duckduckgo_search`): LLM rewrites query, searches, LLM synthesizes answer
- Shell tool: LLM generates macOS command, executes with 30s timeout, LLM presents results
- Streaming output with Rich console UI
- Top-p (0.9) sampling with repetition penalty (1.15x on recent tokens)
- Think-tag handling: thinking can be disabled via `enable_thinking=False` for faster time-to-first-token

### Batched Union-of-Experts: `batched_moe.py`

Processes K tokens per layer in one pass for speculative decoding verification:

1. Route all K positions through the router -> K x 8 expert selections
2. Compute `E_union`: the set of unique experts across all K positions (~40 unique out of K x 8 = 64 total for K=8)
3. Load `E_union` from SSD once (not K x 8 times)
4. Three `gather_qmm` calls with remapped indices handle all K positions simultaneously
5. Weight by per-position router scores, add shared expert

Expected I/O per batch: 40 layers x ~40 experts x 1.69 MB = 2.7 GB (vs 4.3 GB serial).

Includes `LRUExpertCache`: keeps ~100 experts in RAM (~170 MB). Estimated ~80% hit rate (based on Zipf-distributed expert access patterns) would reduce SSD reads ~8x.

## How It All Works

### The MoE Sniper Architecture

The system exploits a fundamental property of Mixture-of-Experts models: **sparsity**. In Qwen3.5-35B-A3B, only 8 of 256 experts activate per token — meaning 96.9% of the model's FFN weights are unused for any given computation. We split the model into two parts:

1. **Pinned weights (1.4 GB)** — attention, router, shared experts, embeddings. These are always needed and live permanently in RAM.
2. **Expert weights (20.6 GB)** — 10,240 expert blocks stored on SSD/USB. Only the 8 active experts per layer are loaded on demand.

### The llama.cpp Expert-Aware LRU Cache

Standard llama.cpp uses `mmap` to access model weights. When the model exceeds RAM, macOS pages expert weights in and out randomly, causing catastrophic thrashing — 15+ minutes with zero output.

Our custom llama.cpp build adds an **expert-aware LRU cache** (`--expert-cache-size`):
- An eval callback intercepts every `ggml_mul_mat_id` operation (the expert computation)
- Before each expert multiply, it copies the active expert slices into a user-space LRU cache
- Hot experts stay pinned in our cache; cold experts get evicted
- The OS can reclaim mmap pages for cold experts without affecting performance
- Result: the 21 GB Q4_K_M model generates output on 8 GB RAM instead of thrashing

### Vision (Multimodal)

Qwen3.5-35B-A3B includes a Vision Transformer (ViT) encoder. The vision encoder is small (~858 MB projector) and loads alongside the pinned weights. Image tokens flow through the same MoE expert routing as text — so the expert streaming technique works identically for vision tasks.

This means a single model handles **text + images + web search** on 8 GB hardware.

### USB Drive as Model Storage

All model files (GGUF weights, mmproj vision projector) can live on a **USB flash drive or external SSD**. The expert cache compensates for slower USB read speeds by keeping hot experts in RAM. This enables fully portable AI — plug in a USB drive and run a 35B multimodal model on any Mac.

## Measured Results

### Test: 8 GB M2 MacBook Air, model on USB flash drive

```
Prompt:  "What is the square root of 69?"
Answer:  "The square root of 69 is approximately 8.307."

Tokens:  16
Time:    474.7s
Speed:   0.03 tok/s
Memory:  1.42 GB active, 1.90 GB peak (on 8 GB machine)
```

- 35B parameter model running on 8 GB RAM — llama.cpp OOMs on this hardware
- 1.4 GB pinned memory; peak usage 1.90 GB, never exceeds 6 GB cap
- Coherent, correct output (√69 ≈ 8.3066)
- 0.03 tok/s bottlenecked by USB flash drive I/O (~300-500 MB/s), not compute
- Full Q4_K_M quality preserved — no quantization degradation beyond 4-bit

### Test: llama.cpp with Expert Cache (IQ2_M, 10.6 GB)

```
Prompt:  "What is mixture of experts in AI?"
Answer:  "A mixture of experts is an AI architecture that combines multiple
          specialized sub-models (experts) with a gating network to dynamically
          route inputs to the most relevant experts for efficient and scalable
          inference."

Speed:   0.24 tok/s (CPU-only, model on USB)
Cache:   100 MB expert LRU cache
```

### Test: llama.cpp with Expert Cache + Vision (IQ2_M + mmproj)

```
Image:   macOS network settings screenshot
Answer:  "This image shows a network configuration interface, likely from a
          macOS system preferences or network utility screen. It displays IPv4
          settings configured manually with a local IP address"

Prompt:  559 tokens at 2.3 tok/s (vision encoder)
Gen:     50 tokens at 0.7 tok/s
```

35B multimodal model correctly reading text from a screenshot on 8 GB RAM.

### Test: llama.cpp with Expert Cache (Q4_K_M, 21 GB — 2.6x larger than RAM)

```
Prompt:  "What is 2+2?"
Answer:  "2 + 2 equals **4**"

Model size: 21 GB on 8 GB machine
Without cache: mmap thrashing, 15+ minutes, no output
With cache: generates within minutes
```

### Comparison: All Methods Tested on 8 GB M2 MacBook Air

| Model | Method | Quant | Size | GPU | Result |
|-------|--------|-------|------|-----|--------|
| **Qwen3.5-9B** | llama.cpp | Q4_K_M | 5.3 GB | Full (ngl 999) | **OOM** |
| **Qwen3.5-9B** | llama.cpp | Q4_K_M | 5.3 GB | Partial (ngl 20) | **OOM** |
| **Qwen3.5-9B** | llama.cpp | Q4_K_M | 5.3 GB | CPU-only (ngl 0) | Works, ~0.1 tok/s |
| **Qwen3.5-35B-A3B** | llama.cpp | IQ2_M | 10.6 GB | Any | **OOM** |
| **Qwen3.5-35B-A3B** | Expert streaming | Q4_K_M | 21 GB | Full | **Works, 0.03 tok/s** |

On 8 GB hardware, llama.cpp cannot GPU-accelerate even a 9B model. Expert streaming runs a 35B model (4x larger) with full GPU acceleration and only 1.9 GB peak memory.

### Comparison: Inference Methods Across Hardware

| Method | Quant | Size | Min RAM | Speed | Quality |
|--------|-------|------|---------|-------|---------|
| llama.cpp (9B, CPU-only) | Q4_K_M | 5.3 GB | 8 GB | ~0.1 tok/s | Full (9B) |
| Expert streaming (USB) | Q4_K_M | 21 GB | 8 GB | 0.03 tok/s | Full (35B) |
| Expert streaming (NVMe) | Q4_K_M | 21 GB | 8 GB | 1-2 tok/s* | Full (35B) |
| Expert streaming (NVMe + LRU cache) | Q4_K_M | 21 GB | 8 GB | 5-9 tok/s* | Full (35B) |
| llama.cpp (35B) | IQ2_M | 10.6 GB | 16 GB | ~30 tok/s | Degraded |
| llama.cpp (35B) | Q4_K_M | 21 GB | 24+ GB | ~30 tok/s | Full (35B) |

*Projected, not yet measured on internal NVMe.

Expert streaming trades throughput for accessibility: full-quality 35B inference on hardware that cannot run the model at all with conventional approaches.

## Apple Silicon Scaling Projections

The bottleneck for expert streaming is SSD read throughput, not memory bandwidth or compute. Here is how the technique scales across Apple Silicon generations.

### Hardware Specs Relevant to Expert Streaming

| Chip | Memory BW | SSD Read | RAM Options | Notes |
|------|-----------|----------|-------------|-------|
| **M2** | 100 GB/s | ~3.0 GB/s | 8, 16, 24 GB | Current test hardware (8 GB) |
| **M2 Pro** | 200 GB/s | ~5.2 GB/s | 16, 32 GB | |
| **M2 Max** | 400 GB/s | ~7.4 GB/s | 32, 64, 96 GB | |
| **M3** | 100 GB/s | ~3.0 GB/s | 8, 16, 24 GB | |
| **M3 Pro** | 150 GB/s | ~5.2 GB/s | 18, 36 GB | |
| **M3 Max** | 400 GB/s | ~7.4 GB/s | 36-128 GB | |
| **M4** | 120 GB/s | ~3.2 GB/s | 16, 24, 32 GB | No more 8 GB configs |
| **M4 Pro** | 273 GB/s | ~5.2 GB/s | 24, 48 GB | |
| **M4 Max** | 546 GB/s | ~7.4 GB/s | 36-128 GB | SSD controller is the ceiling |
| **M4 Ultra** | 819 GB/s | ~7.4 GB/s | 128-512 GB | Ultra doesn't double SSD BW |
| **M5 (est.)** | ~600+ GB/s | ~8-10 GB/s? | TBD | PCIe 5.0 NVMe could unlock this |

### Projected Performance: Qwen3.5-35B-A3B (21 GB, Q4_K_M)

Per-token SSD read: **540 MB** (no cache) or **108 MB** (80% LRU cache hit rate)

| Hardware | SSD BW | No Cache | 80% Cache | 90% Cache | 95% Cache |
|----------|--------|----------|-----------|-----------|-----------|
| **M2 8 GB (USB)** | 0.3 GB/s | 0.03 tok/s | — | — | — |
| **M2 8 GB (NVMe)** | 3.0 GB/s | 5.6 tok/s | 27.8 tok/s | 55.6 tok/s | 111 tok/s |
| **M4 16 GB** | 3.2 GB/s | 5.9 tok/s | 29.6 tok/s | 59.3 tok/s | 118 tok/s |
| **M4 Pro 24 GB** | 5.2 GB/s | 9.6 tok/s | 48.1 tok/s | 96.3 tok/s | 193 tok/s |
| **M4 Max 64 GB** | 7.4 GB/s | 13.7 tok/s | 68.5 tok/s | 137 tok/s | 274 tok/s |
| **M5 Max (est.)** | ~10 GB/s | 18.5 tok/s | 92.6 tok/s | 185 tok/s | 370 tok/s |

**Notes:**
- These are theoretical upper bounds assuming SSD is the only bottleneck
- Real-world overhead (Metal buffer allocation, GPU sync, Python dispatch) reduces these by 30-60%
- Cache hit rate depends on available RAM for the LRU cache — more RAM = higher hit rate
- At 95% cache (achievable with ~2 GB cache on 16+ GB machines), expert streaming approaches llama.cpp speeds while running a higher-quality quantization

### RAM Budget: How Memory Is Allocated

The total RAM is split between four consumers. Here's how to size each one:

```
Total RAM
├── macOS + apps        ~2-3 GB (unavoidable overhead)
├── Pinned weights      ~1.4 GB (attention, router, shared experts, embeddings)
├── Compute buffers     ~0.5-1 GB (KV cache, activations, scratch)
├── Expert LRU cache    EVERYTHING ELSE (this is what you maximize)
└── Expert mmap pages   (OS manages, gets evicted when RAM is full)
```

**Recommended `--expert-cache-size` by hardware:**

| Machine | Total RAM | macOS + Pinned + Compute | Available for Cache | Recommended Flag |
|---------|-----------|--------------------------|--------------------|--------------------|
| M2 MacBook Air | 8 GB | ~4.5 GB | ~3.5 GB | `--expert-cache-size 100` (tight) |
| M4 MacBook | 16 GB | ~5 GB | ~11 GB | `--expert-cache-size 5000` |
| M4 Pro Mac Mini | 24 GB | ~5 GB | ~19 GB | `--expert-cache-size 8000` |
| M4 Max | 64 GB | ~5 GB | ~59 GB | `--expert-cache-size 22000` (full model cached) |

On 8 GB, the cache is very small — most of RAM goes to the OS and pinned weights. On 16 GB+, you can dedicate the majority to the cache, which dramatically increases hit rate and speed.

**For the llama.cpp path (GGUF + mmap):** The expert cache is on top of the mmap'd model. The OS manages mmap pages; our cache keeps hot experts from being evicted. Set the cache to roughly `(total_RAM - 5 GB) * 1024` in MB, leaving headroom for the OS.

**For the MLX path (split experts):** Set `mx.set_memory_limit()` to `total_RAM - 2 GB` and the LRU cache to fill the remaining space after pinned weights load. On 8 GB: `mx.set_memory_limit(6 GB)`, cache ~170 MB. On 16 GB: `mx.set_memory_limit(14 GB)`, cache ~2 GB.

### The Cache Hit Rate Is Everything

The single most impactful variable is not the chip generation — it's how much RAM you can dedicate to the expert LRU cache:

| RAM Config | Available for Cache | Cache Capacity | Estimated Hit Rate | Effective Read/tok |
|------------|--------------------|-----------------|--------------------|-------------------|
| 8 GB | ~170 MB | 100 experts | ~60-70% | 162-216 MB |
| 16 GB | ~2 GB | 1,200 experts | ~90-95% | 27-54 MB |
| 24 GB | ~8 GB | 4,700 experts | ~97-99% | 5-16 MB |
| 32 GB | ~16 GB | 9,500 experts | ~99%+ | <5 MB |
| 64+ GB | 22+ GB | All 10,240 | 100% (fully cached) | 0 MB (RAM-only) |

At 64 GB, the entire model fits in the LRU cache after warmup — expert streaming becomes equivalent to full in-memory inference with zero SSD reads.

### Scaling to Larger Models

The same technique applies to any MoE model. Here's the landscape:

| Model | Total Size (4-bit) | Experts | Active/Token | Pinned | Per-token Read | Min RAM |
|-------|-------------------|---------|--------------|--------|----------------|---------|
| Qwen3.5-35B-A3B | 21 GB | 256 × 40 | 8 | 1.4 GB | 540 MB | **8 GB** |
| Qwen3.5-122B-A10B | 70 GB | 256 × 48 | 10 | 2.9 GB | 1.2 GB | **8 GB** |
| Mixtral-8x22B | 80 GB | 8 × 32 | 2 | 12 GB | 2.5 GB | **16 GB** |
| DeepSeek-V3 (671B) | ~350 GB | 256 × 61 | 8 | ~15 GB | 3.8 GB | **24 GB** |

`sniper_122b.py` already implements the 122B variant on NVIDIA 24 GB GPUs. The same architecture — pin attention/router/embeddings, stream experts from NVMe — scales to any MoE model on any hardware where `pinned_size < available_RAM`.

### The Trend Favoring Expert Streaming

Three hardware trends make this technique more viable over time:

1. **SSD speeds are increasing faster than model sizes.** Apple's NVMe has gone from ~2.8 GB/s (M1) to ~7.4 GB/s (M4 Max). PCIe 5.0 could push this to 12-14 GB/s. Meanwhile, MoE expert sizes remain small (~1.7 MB at 4-bit) because they scale intermediate dimensions, not expert count.

2. **RAM is getting cheaper and larger.** M4 starts at 16 GB (no more 8 GB configs). M4 Max goes to 128 GB. More RAM means higher cache hit rates, which is the dominant performance factor.

3. **MoE is becoming the default architecture.** Qwen3.5, DeepSeek-V3, Mixtral, and most frontier models are MoE. The sparsity that makes expert streaming possible is a permanent feature of the model landscape, not a niche.

The crossover point — where expert streaming with caching matches full in-memory inference speed — occurs when `cache_hit_rate > 1 - (SSD_BW / mem_BW)`. On an M4 Pro (273 GB/s memory, 5.2 GB/s SSD), that's a 98% hit rate, achievable with ~4 GB of cache on 24 GB machines.

## GPU Offloading and the Expert Tensor Problem

### The Problem

When `ngl > 0`, llama.cpp offloads entire layers to GPU — including expert tensors (`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`). The Metal `mul_mat_id` kernel expects the full stacked `[n_embd, n_ff, n_expert]` tensor in GPU memory. For 256 experts per layer, that's the bulk of the model trying to fit in VRAM.

On Apple Silicon, `recommendedMaxWorkingSetSize` is ~75% of total RAM (e.g., ~6 GB on 8 GB, ~12 GB on 16 GB). With `ngl 5+`, expert tensors + attention + compute buffers exceed this limit and the GPU OOMs.

### Tested on 8 GB M2 MacBook Air (IQ2_M, 10.6 GB)

| Config | Prefill | Generate | Status |
|--------|---------|----------|--------|
| `ngl 0` (CPU only) | 0.26 tok/s | 0.24 tok/s | Works, fastest on 8 GB |
| `ngl 2` | 0.2 tok/s | ~0 tok/s | Barely works |
| `ngl 5+` | — | — | GPU OOM |
| `ngl 20` + `--override-tensor "ffn_.*_exps=CPU"` | 0.08 tok/s | 0.02 tok/s | Works but slow |

### Tested on RunPod RTX 3090 (24 GB VRAM, Q4_K_M 21 GB)

| Config | Prefill | Generate | Status |
|--------|---------|----------|--------|
| `ngl 999` (full offload) | — | — | OOM — 21 GB model + compute > 24 GB |
| `ngl 35` (35/41 layers on GPU) | **122.3 tok/s** | **26.0 tok/s** | 17.5 GB on GPU, 3.5 GB on CPU |

26 tok/s generation on a 35B MoE model — conversational speed. The model does NOT fit entirely on a 24 GB GPU, but partial offload (ngl 35) is highly effective. This is where expert-aware GPU offloading would help further: keep expert tensors on CPU, maximize attention layers on GPU.

### The `--override-tensor` Solution

llama.cpp has a per-tensor backend override mechanism (`src/llama-model-loader.cpp:1153-1172`). By forcing expert tensors to CPU while offloading attention layers to GPU:

```bash
llama-server -m model.gguf -ngl 20 \
  --override-tensor "ffn_gate_exps=CPU" \
  --override-tensor "ffn_up_exps=CPU" \
  --override-tensor "ffn_down_exps=CPU"
```

This prevents the GPU OOM because expert tensors (~90% of model weight) stay on CPU. Attention, router, and shared expert layers run on GPU.

**On 8 GB:** This is slower than pure CPU (0.02 vs 0.24 tok/s) because the cross-device data transfer overhead for every expert operation exceeds the GPU compute benefit. The scheduler must shuttle activations between GPU (attention) and CPU (experts) 40 times per token.

**On 16+ GB:** This split should perform better because:
- More GPU working set for attention/compute buffers
- Less memory pressure means fewer mmap page faults
- The cross-device overhead is fixed (~0.1ms per transfer) while GPU attention compute scales with model size

### Root Cause (from llama.cpp source analysis)

1. **Expert offloading follows layer assignment** (`llama-model.cpp:2646-2671`): `ffn_up_exps` gets the same backend as the layer's attention tensors. No per-op-type control.
2. **Metal `mul_mat_id` expects GPU buffers** (`ggml-metal-ops.cpp:2282`): The kernel binds `op->src[0]` as a Metal buffer. No CPU fallback.
3. **mmap page faults compete with GPU** on unified memory: macOS pages in expert data from the mmap'd file, competing with GPU buffer allocations for the same physical memory.

### Future Fix: Expert-Aware GPU Offloading

The proper solution is teaching llama.cpp to offload attention/router to GPU while keeping expert tensors on CPU with our LRU cache. This requires:

1. Automatic `--override-tensor` for expert tensors when `--expert-cache-size > 0`
2. Optimized CPU `mul_mat_id` path that reads from the LRU cache instead of mmap
3. Prefetching: while GPU computes attention for layer N, CPU pre-loads experts for layer N from cache

## Multi-Machine Expert Streaming

### The Disk I/O Wall

On a single machine, expert streaming speed is capped by storage bandwidth:

```
tok/s = SSD_bandwidth / bytes_per_token_read
      = 5 GB/s / 540 MB
      = ~9.3 tok/s (theoretical max on NVMe, no cache)
```

With caching this improves dramatically, but there's a fundamental limit: one machine's SSD can only feed experts so fast. For models larger than one machine's RAM + SSD can handle efficiently, you need to go multi-machine.

### Thunderbolt Mac Clustering

Connect multiple Macs via Thunderbolt to create a distributed expert pool:

```
┌─────────────────┐   Thunderbolt    ┌─────────────────┐
│  Mac A (16 GB)  │◄──── 5 GB/s ────►│  Mac B (16 GB)  │
│  Layers 0-19    │                   │  Layers 20-39   │
│  + LRU cache    │                   │  + LRU cache    │
│  + GPU (ngl 20) │                   │  + GPU (ngl 20) │
└─────────────────┘                   └─────────────────┘
        │                                      │
        └──────── Pipeline Parallelism ────────┘
        Token activations pass between machines
        Each machine holds half the experts
```

**Why this works for MoE:**
- Each machine only needs to hold experts for its layers (~10 GB for 20 layers)
- Activations between layers are tiny (~2 KB per token vs ~540 MB of expert data)
- Thunderbolt 4 gives ~5 GB/s, Thunderbolt 5 gives ~16 GB/s
- The activation transfer (~2 KB) takes <1 microsecond — negligible

**Concrete example: 2× Mac Mini (16 GB each)**
- Total RAM: 32 GB (vs 21 GB Q4_K_M model)
- Each machine holds 20 layers + 5 GB expert cache → ~95% hit rate per machine
- GPU offload: each machine does ngl 20 (full offload of its layers)
- Expected: 10-20 tok/s (vs 0.09 tok/s on single machine, CPU-only)

**Scaling table:**

| Machines | Total RAM | Model | Layers/Machine | Cache/Machine | GPU | Expected tok/s |
|----------|-----------|-------|---------------|--------------|-----|---------------|
| 1× 16 GB | 16 GB | Q4_K_M (21 GB) | 40 | 5 GB | partial | 1-5 |
| 2× 16 GB | 32 GB | Q4_K_M (21 GB) | 20 | 5 GB | full | 10-20 |
| 4× 16 GB | 64 GB | Q4_K_M (21 GB) | 10 | 10 GB | full | 20-30 |
| 2× 16 GB | 32 GB | 122B (70 GB) | 24 | 0 | partial | 0.5-2 |
| 4× 16 GB | 64 GB | 122B (70 GB) | 12 | 5 GB | full | 5-10 |

### Formal Verification of Multi-Machine Feasibility

The multi-machine setup introduces new latency components that can be formally verified:

```lean
structure ClusterHardware where
  n_machines : ℕ                    -- number of Macs in cluster
  ram_per_machine : ℝ               -- GB per machine
  ssd_bw_per_machine : ℝ            -- GB/s SSD bandwidth per machine
  gpu_bw_per_machine : ℝ            -- GB/s GPU memory bandwidth
  interconnect_bw : ℝ               -- GB/s between machines (Thunderbolt)
  interconnect_latency : ℝ          -- seconds per message (one-way)

structure ClusterModel where
  total_size : ℝ                    -- total model size in GB
  n_layers : ℕ                     -- total layers
  layers_per_machine : ℕ            -- layers assigned to each machine
  expert_size : ℝ                   -- bytes per expert
  experts_per_token : ℕ             -- K active experts per layer
  activation_size : ℝ               -- bytes per inter-layer activation

-- Per-token latency across the cluster
-- t_token = max over machines of:
--   t_compute(layers_per_machine) +    -- GPU matmul time
--   t_expert_load(cache_miss_rate) +   -- SSD reads for cache misses
--   t_transfer(activation_size) +      -- Thunderbolt transfer
--   t_sync                             -- barrier synchronization

theorem cluster_feasibility
  (h : ClusterHardware) (m : ClusterModel)
  (cache_hit_rate : ℝ)
  (target_tok_s : ℝ)
  -- Prove: per-token latency < 1/target_tok_s
  : t_token_cluster h m cache_hit_rate < 1 / target_tok_s := by
  sorry
```

The key insight for formal verification: **the activation transfer between machines is tiny compared to expert loading.** For Qwen3.5-35B-A3B:
- Activation per layer: hidden_size × sizeof(float16) = 2048 × 2 = **4 KB**
- Expert load per layer: 8 × 1.69 MB = **13.5 MB**
- Ratio: expert loading is **3,375x** more data than activation transfer

This means the interconnect latency for passing activations between machines is negligible compared to the SSD reads for experts. The bottleneck remains expert loading, not network — which means the multi-machine approach scales linearly with the number of machines (each machine's SSD contributes bandwidth to the pool).

A Lean proof could establish: "Given N machines with measured SSD bandwidth S and measured interconnect latency L, the system achieves target_tok_s if and only if `N * S * cache_hit_rate > bytes_per_token * target_tok_s`." This gives a concrete engineering target: how many machines do you need for a given speed target?

## Files (this repo)

| File | Purpose |
|------|---------|
| `sniper.py` | Interactive agent (chat, search, vision, shell) |
| `sniper-router/router.py` | Remote inference client |
| `expert_io.py` | F_NOCACHE + pread expert reader with LRU cache |
| `docker/` | Pre-built GPU inference server |
| `scripts/gpu_benchmark.sh` | Reproducible GPU benchmark |
| `RESEARCH.md` | This document |

Note: The MLX research files referenced in the Implementation section above (`split_mlx_model_macbook.py`, `moe_agent_macbook.py`, `flash_moe.py`, `batched_moe.py`, etc.) describe the Apple Silicon MLX path. They are archived separately and available in the [MLX expert sniper](https://huggingface.co/waltgrace/mlx-expert-sniper) project.

## Citation

```
MoE Expert Streaming: SSD-based inference for mixture-of-experts models
exceeding available RAM on Apple Silicon.
https://github.com/walter-grace/mac-code
```
