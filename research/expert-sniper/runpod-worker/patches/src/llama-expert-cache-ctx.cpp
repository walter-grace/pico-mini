#include "llama-expert-cache-ctx.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <algorithm>

#ifdef __APPLE__
#include <sys/mman.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#endif

// Initialize expert cache from model metadata
void llama_expert_cache_ctx::init(const llama_model & model, size_t cache_bytes) {
    const auto & hparams = model.hparams;

    n_expert      = (int)hparams.n_expert;
    n_expert_used = (int)hparams.n_expert_used;
    n_layers      = (int)hparams.n_layer;

    if (n_expert == 0 || n_expert_used == 0) {
        return;
    }

    // We still create the LRU cache for tracking which experts are hot,
    // but with minimal memory — just metadata, no data copies.
    cache = std::make_unique<llama_expert_cache>(cache_bytes);

    expert_tensors.resize(n_layers);
    expert_strides.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        const auto & layer = model.layers[il];

        expert_tensors[il] = {
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
        };

        for (int wt = 0; wt < 3; wt++) {
            ggml_tensor * t = expert_tensors[il][wt];
            if (t && t->ne[2] > 1) {
                expert_strides[il][wt] = t->nb[2];
            } else {
                expert_strides[il][wt] = 0;
            }
        }
    }

    // Small active buffer for build_active_buffer (kept for compatibility)
    size_t max_stride = 0;
    for (int il = 0; il < n_layers; il++) {
        for (int wt = 0; wt < 3; wt++) {
            max_stride = std::max(max_stride, expert_strides[il][wt]);
        }
    }
    active_buffer_size = (size_t)n_expert_used * max_stride;
    active_buffer = malloc(active_buffer_size);
    GGML_ASSERT(active_buffer != nullptr);

    fprintf(stderr, "llama_expert_cache_ctx: initialized for %d layers, %d experts (%d used), "
            "cache = %.1f MB, stride = %.2f MB [madvise prefetch mode]\n",
            n_layers, n_expert, n_expert_used,
            (double)cache_bytes / (1024*1024),
            (double)max_stride / (1024*1024));
}

std::pair<int, int> llama_expert_cache_ctx::identify_tensor(const ggml_tensor * t) const {
    for (int il = 0; il < n_layers; il++) {
        for (int wt = 0; wt < 3; wt++) {
            if (expert_tensors[il][wt] == t) {
                return {il, wt};
            }
        }
    }
    return {-1, -1};
}

void * llama_expert_cache_ctx::build_active_buffer(
        int layer, int weight_type,
        const int32_t * expert_ids, int n_ids) {

    const size_t stride = expert_strides[layer][weight_type];
    const ggml_tensor * stacked = expert_tensors[layer][weight_type];

    if (!stacked || stride == 0) return nullptr;

    char * dst = (char *)active_buffer;
    for (int i = 0; i < n_ids; i++) {
        int eid = expert_ids[i];
        if (eid < 0 || eid >= n_expert) continue;

        llama_expert_key key = {(int32_t)layer, (int32_t)eid, (int32_t)weight_type};

        const char * expert_src = nullptr;

        if (cache) {
            auto [buf, hit] = cache->get_or_alloc(key, stride);
            if (buf) {
                if (!hit) {
                    const char * src = (const char *)stacked->data + (size_t)eid * stride;
                    memcpy(buf, src, stride);
                }
                expert_src = (const char *)buf;
            }
        }

        if (!expert_src) {
            expert_src = (const char *)stacked->data + (size_t)eid * stride;
        }

        memcpy(dst, expert_src, stride);
        dst += stride;
    }

    return active_buffer;
}

// madvise-based eval callback.
//
// Strategy: guide the kernel's page cache instead of maintaining our own.
//   1. MADV_WILLNEED for active expert pages → kernel prefetches from SSD
//   2. MADV_DONTNEED for cold expert pages → kernel can reclaim memory
//
// This works with REPACK, doesn't allocate extra memory, and reduces
// memory pressure by proactively releasing pages the model won't need
// for the next few tokens.
//
// The LRU cache tracks which experts are "hot" — we don't release pages
// for recently-used experts, only for truly cold ones that got evicted.
bool llama_expert_cache_ctx::eval_callback(
        struct ggml_tensor * t,
        bool ask,
        void * user_data) {

    if (!ask) {
        return true;
    }

    // Only intercept MUL_MAT_ID operations
    if (t->op != GGML_OP_MUL_MAT_ID) {
        return true;
    }

    auto * ctx = (llama_expert_cache_ctx *)user_data;

    ggml_tensor * expert_weights = t->src[0];
    ggml_tensor * expert_indices = t->src[2];

    if (!expert_weights || !expert_indices || !ctx->cache) {
        return true;
    }

    auto [layer, weight_type] = ctx->identify_tensor(expert_weights);
    if (layer < 0) {
        return true;
    }

    const size_t stride = ctx->expert_strides[layer][weight_type];
    if (stride == 0) {
        return true;
    }

    // Read active expert indices
    if (!expert_indices->data) {
        return true;
    }

    const int32_t * ids = (const int32_t *)expert_indices->data;
    int n_ids = (int)(ggml_nelements(expert_indices));

    // Collect unique active experts
    std::set<int32_t> active_set;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= 0 && ids[i] < ctx->n_expert) {
            active_set.insert(ids[i]);
        }
    }

    // For each active expert:
    //   - Touch in LRU cache (mark as hot)
    //   - madvise WILLNEED to prefetch from SSD
    for (int eid : active_set) {
        llama_expert_key key = {(int32_t)layer, (int32_t)eid, (int32_t)weight_type};

        // Touch in LRU to track hotness (get_or_alloc updates LRU order)
        // Use a tiny allocation — we just need the LRU tracking, not the data
        ctx->cache->touch(key);

        // Prefetch this expert's mmap pages
#if defined(__APPLE__) || defined(__linux__)
        const char * expert_data = (const char *)expert_weights->data + (size_t)eid * stride;
        uintptr_t page_start = (uintptr_t)expert_data & ~(uintptr_t)4095;
        size_t page_len = ((uintptr_t)expert_data + stride - page_start + 4095) & ~(size_t)4095;
        madvise((void *)page_start, page_len, MADV_WILLNEED);
#endif
    }

    // Release pages for cold experts (not in LRU cache).
    // Only do this for weight_type 0 (up_proj) to avoid doing it 3x per layer.
    // This proactively tells the kernel it can reclaim memory for experts
    // that haven't been used recently.
    if (weight_type == 0) {
        for (int eid = 0; eid < ctx->n_expert; eid++) {
            if (active_set.count(eid)) continue;

            llama_expert_key key = {(int32_t)layer, (int32_t)eid, 0};

            if (!ctx->cache->contains(key)) {
                // This expert is cold — release all 3 weight types
                for (int wt = 0; wt < 3; wt++) {
                    const ggml_tensor * wt_tensor = ctx->expert_tensors[layer][wt];
                    if (!wt_tensor) continue;
                    size_t wt_stride = ctx->expert_strides[layer][wt];
                    if (wt_stride == 0) continue;

                    const char * data = (const char *)wt_tensor->data + (size_t)eid * wt_stride;
                    uintptr_t page_start = (uintptr_t)data & ~(uintptr_t)4095;
                    size_t page_len = ((uintptr_t)data + wt_stride - page_start + 4095) & ~(size_t)4095;

#if defined(__APPLE__) || defined(__linux__)
                    madvise((void *)page_start, page_len, MADV_DONTNEED);
#endif
                }
            }
        }
    }

    return true;
}
