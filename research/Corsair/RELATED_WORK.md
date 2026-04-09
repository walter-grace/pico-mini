# Expert Sniper — Related Work & Academic Validation

## Key Papers That Validate Our Approach

### 1. Mixture of Cache-Conditional Experts (Dec 2024)
**THE closest paper to our routing bias.** Independently proves biasing routers toward cached experts works.
- >50% cache miss reduction, <0.1% accuracy loss on MMLU/GSM8K
- 2x speedup on Snapdragon
- Expert retention improved 5-9x (from ~22 to 58-76 tokens in cache)
- Router's top-2 prediction is optimal only 28% of the time → huge room for cache-aware steering
- Paper: https://arxiv.org/html/2412.00099

### 2. SAIL: SRAM-Accelerated LLM Inference (Oct 2025)
- 10.7x speedup via SRAM-based processing-in-memory
- 90% of token generation time is data movement, not compute
- LUT-GEMV for 2/3/4-bit quantized weights
- Dense models only — no MoE
- Paper: https://arxiv.org/abs/2509.25853

### 3. Fate: Cross-Layer Gate Prediction (Feb 2025)
- 97.15% prefetch accuracy using cross-layer gate similarity
- Adjacent layers' gate inputs have >83% cosine similarity
- 4.5x speedup over load-on-demand
- No retraining needed
- Paper: https://arxiv.org/html/2502.12224v1

### 4. Cache Management for MoE LLMs (Sep 2025)
- Formalizes MoE caching as "layered paging" problem
- Proves LRU achieves only k-competitive ratio (suboptimal)
- Proposes LLRU (Layered LRU): ~15% fewer misses
- Shared cache 3x better than per-layer distributed cache
- Paper: https://arxiv.org/html/2509.02408v1

### 5. "Not All Models Suit Expert Offloading" (May 2025)
- Studies routing locality across 20 MoE models
- Rankings: LLaMA-MoE-v2 (78) > OLMoE (51) > Mixtral (49) > DeepSeekMoE (37)
- Cache size 2x active experts achieves ~90% hit rate for good models
- Shared experts harm routing consistency
- Paper: https://arxiv.org/html/2505.16056v3

### 6. SambaNova SN40L — Composition of Experts (May 2024)
- Custom hardware with three-tier memory: 520 MB SRAM → 64 GB HBM → 1.5 TB DDR
- Experts live in DDR, active experts copied to HBM → SRAM for execution
- 19x smaller machine footprint than DGX, 3.7x overall speedup vs H100
- Their CoE architecture IS Expert Sniper in hardware
- Paper: https://arxiv.org/html/2405.07518v1

## Competing Systems

| System | Approach | Hardware | Speed | Diff from Expert Sniper |
|--------|----------|----------|-------|------------------------|
| FlashMoE (Jan 2026) | ML-based cache, SSD | RTX 5070 + NVMe | 2.6x over baselines | GPU+PCIe bottleneck |
| HOBBIT (Nov 2024) | Mixed-precision offload | Jetson Orin | 9.93x over MoE-Infinity | Mixed precision clever |
| DALI (Feb 2026) | Workload-aware offload | RTX 3090 | 3.97x over llama.cpp | GPU+CPU split |
| KTransformers (SOSP) | AMX CPU + async GPU | 24GB GPU + 382GB DRAM | 220 tok/s total | Datacenter-scale |
| mixtral-offloading | LRU + HQQ | T4/RTX 3060 | 2-3 tok/s | We beat them 2x |
| MoE-Infinity | Activation-aware prefetch | A5000 | 0.1s/token | General-purpose |
| llama.moe | Dynamic expert offload | llama.cpp | — | Fork of llama.cpp |

## Expert Sniper Differentiators

1. **Apple Silicon unified memory** — no PCIe copy penalty (GPU systems pay 30-50% overhead)
2. **MLX-native** — only system targeting MLX framework
3. **Training-free routing bias** — Pre-Gated MoE requires retraining; Cache-Conditional paper validates our bias-only approach
4. **6.41 tok/s** on consumer Mac Mini beats mixtral-offloading (2-3 tok/s) by 2x
5. **Corsair architectural match** — unified memory mirrors SRAM/LPDDR5X hierarchy

## Techniques to Adopt

1. **LLRU** (Layered LRU) — consider layer distance in eviction, ~15% fewer misses
2. **Fate cross-layer prediction** — 97% accurate next-layer expert prediction, augments our co-activation tracker
3. **Bit-sliced caching** (SliceMoE) — load MSB first for fast approximate, LSB later
4. **Mixed-precision fallback** (HOBBIT) — cache-miss experts at int4, hits at full precision

## Resource Lists
- Awesome MoE Inference: https://github.com/MoE-Inf/awesome-moe-inference/
- ACM Survey on MoE Inference: https://dl.acm.org/doi/10.1145/3794845
- MoE Inference Survey (arXiv): https://arxiv.org/abs/2412.14219
- HuggingFace MoE Offload Guide: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
