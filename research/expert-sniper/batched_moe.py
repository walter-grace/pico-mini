"""
Batched MoE with Union-of-Experts — The Speed Engine.

Instead of K serial forward passes (each reading 8 experts from SSD),
this processes K tokens per layer in ONE pass:
  1. Run attention/SSM on all K hidden states
  2. Route all K positions → compute E_union (~40 unique experts)
  3. Load E_union from SSD ONCE (not K×8 times)
  4. gather_qmm with remapped indices — 3 kernel calls for SwiGLU
  5. Weight by router gates, sum, add shared expert

Expected: 40 layers × ~40 experts × 1.7 MB = 2.7 GB SSD per verify batch
vs serial: 40 layers × 64 experts × 1.7 MB = 4.4 GB
With LRU cache: ~0.5 GB SSD → 8+ tok/s
"""

import time
import os
import sys
import json
import gc

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_io import MoEExpertReader

BITS = 4
GROUP_SIZE = 64


class LRUExpertCache:
    """
    LRU cache for recently used experts.
    Keeps ~100 experts in RAM (~170 MB).
    80% hit rate → 8x fewer SSD reads.
    """

    def __init__(self, max_experts=100):
        self.max_experts = max_experts
        self.cache = {}  # (layer_idx, expert_id) → expert_data
        self.access_order = []  # most recent last
        self.hits = 0
        self.misses = 0

    def get(self, layer_idx, expert_id):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.hits += 1
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, layer_idx, expert_id, data):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_experts:
            evict = self.access_order.pop(0)
            del self.cache[evict]
        self.cache[key] = data
        self.access_order.append(key)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def stats(self):
        return f"LRU: {len(self.cache)}/{self.max_experts}, hit={self.hit_rate():.0%}"


def batched_moe_forward(
    hidden_states,  # [1, K, D] — K positions' hidden states after attention
    router_weight,  # the layer's router gate (nn.Linear or QuantizedLinear)
    shared_expert,  # the layer's shared expert MLP
    shared_expert_gate,  # the layer's shared expert gate
    norm_topk_prob,  # whether to normalize top-k probs
    top_k,  # number of experts per token
    reader,  # MoEExpertReader for SSD loading
    layer_idx,  # current layer index
    lru_cache=None,  # optional LRU cache
):
    """
    Batched MoE for K positions using Union-of-Experts + gather_qmm.

    ONE SSD read per unique expert per layer.
    THREE gather_qmm calls for SwiGLU (gate, up, down).
    """
    B, K, D = hidden_states.shape  # [1, K, D]

    # ── Step 1: Route all K positions ────────────────────
    gates = router_weight(hidden_states)  # [1, K, 256]
    gates = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]  # [1, K, 8]
    scores = mx.take_along_axis(gates, inds, axis=-1)  # [1, K, 8]
    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    mx.eval(inds, scores)

    # ── Step 2: Compute E_union ──────────────────────────
    flat_inds = np.array(inds).flatten()
    unique_ids = sorted(set(int(e) for e in flat_inds))
    E = len(unique_ids)

    # ── Step 3: Load unique experts (SSD or LRU cache) ───
    loaded = {}
    to_load = []
    for eid in unique_ids:
        if lru_cache is not None:
            cached = lru_cache.get(layer_idx, eid)
            if cached is not None:
                loaded[eid] = cached
                continue
        to_load.append(eid)

    if to_load:
        ssd_data = reader.get_experts(layer_idx, to_load)
        for eid, data in ssd_data.items():
            loaded[eid] = data
            if lru_cache is not None:
                lru_cache.put(layer_idx, eid, data)

    # ── Step 4: Pack into compact tensors ────────────────
    def stack_proj(proj_name):
        w = mx.stack([loaded[eid][f"mlp.switch_mlp.{proj_name}.weight"] for eid in unique_ids])
        s = mx.stack([loaded[eid][f"mlp.switch_mlp.{proj_name}.scales"] for eid in unique_ids])
        b = mx.stack([loaded[eid][f"mlp.switch_mlp.{proj_name}.biases"] for eid in unique_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    # ── Step 5: Remap indices global → local ─────────────
    id_to_local = {eid: i for i, eid in enumerate(unique_ids)}
    inds_np = np.array(inds)
    local_np = np.vectorize(lambda x: id_to_local.get(int(x), 0))(inds_np)
    local_indices = mx.array(local_np)  # [1, K, 8]

    # ── Step 6: gather_qmm × 3 (SwiGLU) ─────────────────
    x_exp = mx.expand_dims(hidden_states, (-2, -3))  # [1, K, 1, 1, D]

    gate_out = mx.gather_qmm(
        x_exp, gate_w, scales=gate_s, biases=gate_b,
        rhs_indices=local_indices, transpose=True,
        group_size=GROUP_SIZE, bits=BITS,
    )
    up_out = mx.gather_qmm(
        x_exp, up_w, scales=up_s, biases=up_b,
        rhs_indices=local_indices, transpose=True,
        group_size=GROUP_SIZE, bits=BITS,
    )

    intermediate = nn.silu(gate_out) * up_out  # SwiGLU activation

    # For down_proj: input is intermediate [1, K, 8, 1, I]
    # Reshape to [1, K*8, 1, 1, I] with repeated local indices
    K_pos = intermediate.shape[1]
    top_k_dim = intermediate.shape[2]
    flat_inter = intermediate.reshape(1, K_pos * top_k_dim, 1, 1, -1)
    flat_local = local_indices.reshape(1, K_pos * top_k_dim, 1)

    down_out = mx.gather_qmm(
        flat_inter, down_w, scales=down_s, biases=down_b,
        rhs_indices=flat_local, transpose=True,
        group_size=GROUP_SIZE, bits=BITS,
    )
    down_out = down_out.reshape(1, K_pos, top_k_dim, -1)  # [1, K, 8, D]

    # ── Step 7: Weight by router scores + sum ────────────
    expert_out = (down_out * scores[..., None]).sum(axis=2)  # [1, K, D]

    # ── Step 8: Shared expert ────────────────────────────
    shared_out = shared_expert(hidden_states)
    sg = mx.sigmoid(shared_expert_gate(hidden_states))
    if sg.ndim < shared_out.ndim:
        sg = sg[..., None]
    expert_out = expert_out + sg * shared_out

    return expert_out, E  # return union size for stats


def batched_verify(
    model,
    cache,
    reader,
    token_ids_list,
    num_layers,
    lru_cache=None,
):
    """
    Verify K draft tokens in ONE batched pass through 40 layers.
    Returns logits [K, vocab].

    Key: each layer reads experts ONCE for all K positions.
    """
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    K = len(token_ids_list)
    total_union = 0

    # Embed all K tokens
    input_ids = mx.array([token_ids_list])  # [1, K]
    h = model.model.embed_tokens(input_ids)  # [1, K, D]

    fa_mask = create_attention_mask(h, cache[model.model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[model.model.ssm_idx])

    for i in range(num_layers):
        layer = model.model.layers[i]
        mask = ssm_mask if layer.is_linear else fa_mask

        # ── Attention (from RAM) — handles K positions ───
        normed = layer.input_layernorm(h)
        if layer.is_linear:
            attn_out = layer.linear_attn(normed, mask=mask, cache=cache[i])
        else:
            attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
        h = h + attn_out
        mx.eval(h)

        # ── Batched MoE (union-of-experts from SSD) ──────
        normed = layer.post_attention_layernorm(h)
        moe_out, e_size = batched_moe_forward(
            normed,
            layer.mlp.gate,
            layer.mlp.shared_expert,
            layer.mlp.shared_expert_gate,
            layer.mlp.norm_topk_prob,
            layer.mlp.top_k,
            reader, i, lru_cache,
        )
        h = h + moe_out
        mx.eval(h)
        total_union += e_size

        del normed, attn_out, moe_out
        mx.clear_cache()

    h = model.model.norm(h)
    logits = model.lm_head(h)
    mx.eval(logits)

    avg_union = total_union / num_layers
    return logits[0], avg_union  # [K, vocab], avg E_union per layer


# ── Test ─────────────────────────────────────────────────

def test_batched():
    """Test batched verification with 8 tokens."""
    MODEL_DIR = os.environ.get("MOE_MODEL_DIR", os.path.expanduser("~/models/qwen35-35b-moe-stream"))

    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx.utils import tree_flatten

    args = TextModelArgs(
        model_type=config.get("model_type"), hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"], vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        head_dim=config.get("head_dim"), tie_word_embeddings=config["tie_word_embeddings"],
        num_experts=config["num_experts"], num_experts_per_tok=config["num_experts_per_tok"],
        shared_expert_intermediate_size=config["shared_expert_intermediate_size"],
        moe_intermediate_size=config["moe_intermediate_size"],
        linear_num_value_heads=config.get("linear_num_value_heads"),
        linear_num_key_heads=config.get("linear_num_key_heads"),
        linear_key_head_dim=config.get("linear_key_head_dim"),
        linear_value_head_dim=config.get("linear_value_head_dim"),
        linear_conv_kernel_dim=config.get("linear_conv_kernel_dim"),
        full_attention_interval=config.get("full_attention_interval"),
        rope_parameters=config.get("rope_parameters"),
    )

    model = TextModel(args)

    def should_quantize(path, module):
        if isinstance(module, nn.Embedding): return True
        if isinstance(module, SwitchLinear): return True
        if not isinstance(module, nn.Linear): return False
        if "conv1d" in path: return False
        if module.weight.shape[-1] < 64: return False
        return True

    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS, class_predicate=should_quantize)

    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)

    print("Loading pinned weights...")
    pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
    model.load_weights(list(pinned.items()), strict=False)
    params = [p for name, p in tree_flatten(model.parameters()) if "switch_mlp" not in name]
    mx.eval(*params)
    del pinned; gc.collect(); mx.clear_cache()

    reader = MoEExpertReader(f"{MODEL_DIR}/{config['streaming']['expert_dir']}",
                              config["num_hidden_layers"], num_workers=8)

    lru = LRUExpertCache(max_experts=100)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    cache = model.make_cache()

    # Prefill prompt
    messages = [
        {"role": "system", "content": "Think briefly, answer directly."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tok.encode(text)

    print(f"Prefilling {len(prompt_tokens)} tokens...")
    input_ids = mx.array([prompt_tokens])
    # Use serial prefill for prompt (same as working engine)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "qwen35-35b-moe"))
    from moe_agent import MoESniperEngine
    engine = MoESniperEngine()
    engine.load()
    engine.reset_cache()
    prefill_logits = engine.forward(mx.array([prompt_tokens]))
    mx.eval(prefill_logits)
    print("  Prefilled")

    # Generate 8 tokens serially to get draft token IDs
    serial_tokens = []
    logits = prefill_logits
    for i in range(8):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        serial_tokens.append(tid)
        if tid in (248044, 248045, 248046):
            break
        logits = engine.forward(mx.array([[tid]]))
        mx.eval(logits)

    print(f"  Serial tokens: {[tok.decode([t]) for t in serial_tokens]}")

    # Now test BATCHED verify on those 8 tokens
    # Reset cache first
    engine.reset_cache()
    prefill_logits = engine.forward(mx.array([prompt_tokens]))
    mx.eval(prefill_logits)

    print(f"\n=== BATCHED VERIFY (K={len(serial_tokens)}) ===")
    t0 = time.time()
    batch_logits, avg_union = batched_verify(
        engine.model, engine.cache, engine.reader,
        serial_tokens, engine.num_layers, lru,
    )
    elapsed = time.time() - t0
    mx.eval(batch_logits)

    batch_tokens = [mx.argmax(batch_logits[i]).item() for i in range(len(serial_tokens))]
    print(f"  Time: {elapsed:.2f}s for {len(serial_tokens)} tokens ({len(serial_tokens)/elapsed:.2f} tok/s)")
    print(f"  Avg E_union: {avg_union:.1f} experts/layer")
    print(f"  {lru.stats()}")
    print(f"  Batch tokens: {[tok.decode([t]) for t in batch_tokens]}")

    # Compare with serial
    print(f"\n  Serial: {[tok.decode([t]) for t in serial_tokens]}")
    print(f"  Batch:  {[tok.decode([t]) for t in batch_tokens]}")

    reader.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  BATCHED MOE — Union-of-Experts Verification")
    print("=" * 60)
    test_batched()
