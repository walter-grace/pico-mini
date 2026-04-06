#!/usr/bin/env python3
"""
MoE Expert Sniper — Gemma 4-26B-A4B via SSD streaming.

Architecture differences from Qwen3.5:
- 30 layers (not 40)
- 128 experts, top-8
- Dense MLP runs in parallel with MoE
- Router has per-dimension scale + per-expert scale
- Experts use gate_proj + up_proj + down_proj (separate, quantized)
- gelu_approx activation (not silu)
- Layer scalar per layer
- Sliding window + full attention hybrid
"""
import json, sys, os, time, gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Import expert I/O (F_NOCACHE + pread)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_io import MoEExpertReader
from coactivation import CoActivationTracker

MODEL_DIR = os.path.expanduser("~/models/gemma4-stream")
GROUP_SIZE = 64
# BITS is auto-detected from config quantization settings


def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    """Run expert FFN using gather_qmm with streamed expert weights.

    Gemma 4 uses gelu_approx activation (not silu like Qwen).
    Expert tensor names: switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
    """
    active_ids = sorted(expert_data.keys())
    if not active_ids:
        return mx.zeros_like(x)

    id_to_local = {eid: i for i, eid in enumerate(active_ids)}
    inds_np = np.array(top_k_indices)
    local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
    local_indices = mx.array(local_np)

    def stack_proj(proj):
        w = mx.stack([expert_data[eid][f"switch_mlp.{proj}.weight"] for eid in active_ids])
        s = mx.stack([expert_data[eid][f"switch_mlp.{proj}.scales"] for eid in active_ids])
        b = mx.stack([expert_data[eid][f"switch_mlp.{proj}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    x_exp = mx.expand_dims(x, (-2, -3))
    # Auto-detect bits from weight vs scales shape
    # scales shape[-1] = input_dim / group_size (num groups)
    # weight shape[-1] = input_dim / (32/bits) (packed uint32)
    # So: packed * (32/bits) = groups * group_size
    # → bits = 32 * packed / (groups * group_size)
    n_packed = gate_w.shape[-1]
    n_groups = gate_s.shape[-1]
    real_input = n_groups * GROUP_SIZE
    bits = round(32 * n_packed / real_input)
    if bits not in (4, 8):
        bits = 4  # fallback

    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=bits)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=bits)

    # Gemma 4 uses gelu_approx, not silu
    hidden = nn.gelu_approx(gate_out) * up_out

    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=bits)
    out = down_out.squeeze(-2)
    out = (out * top_k_weights[..., None]).sum(axis=-2)
    return out


class MoESniperEngineGemma4:
    def __init__(self, model_dir=None, cache_size=3000, enable_prediction=True):
        self.model_dir = model_dir or MODEL_DIR
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self._cache_size = cache_size
        self._enable_prediction = enable_prediction
        self.num_layers = 30
        self.coact = None

    def load(self):
        # Load config
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        text_config = config.get("text_config", config)
        self.num_layers = text_config["num_hidden_layers"]

        # Import Gemma 4 model class
        sys.path.insert(0, os.path.expanduser("~/cli-agent/src"))
        from mlx_expert_sniper.models.gemma4 import Model, ModelArgs

        args = ModelArgs.from_dict(text_config)
        self.model = Model(args)

        # Read quantization config from model
        quant_config = config.get("quantization", config.get("quantization_config", {}))
        default_bits = quant_config.get("bits", 4)
        default_gs = quant_config.get("group_size", GROUP_SIZE)

        # Mixed quantization: MLP/router at 8-bit, rest at 4-bit
        def _is_8bit(path, module):
            if not isinstance(module, nn.Linear):
                return False
            full_path = "language_model." + path
            if full_path in quant_config and isinstance(quant_config[full_path], dict):
                return quant_config[full_path].get("bits", default_bits) == 8
            return False

        def _q4(path, module):
            if isinstance(module, nn.Embedding): return True
            if not isinstance(module, nn.Linear): return False
            if _is_8bit(path, module): return False
            if module.weight.shape[-1] < default_gs: return False
            return True

        nn.quantize(self.model, group_size=default_gs, bits=default_bits,
                     class_predicate=_q4)
        nn.quantize(self.model, group_size=64, bits=8,
                     class_predicate=lambda p, m: isinstance(m, nn.Linear) and _is_8bit(p, m))

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        # Load pinned weights (non-expert)
        pinned_path = os.path.join(self.model_dir, "pinned.safetensors")
        pinned = mx.load(pinned_path)
        stripped = [(k.replace("language_model.", "", 1), v) for k, v in pinned.items()]
        self.model.load_weights(stripped, strict=False)

        # Eval only non-expert params
        params = [p for name, p in tree_flatten(self.model.parameters())
                  if "expert" not in name and "switch" not in name]
        mx.eval(*params)
        del pinned
        gc.collect()
        mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9

        # Expert reader (F_NOCACHE + pread)
        sniper_config_path = os.path.join(self.model_dir, "sniper_config.json")
        if os.path.exists(sniper_config_path):
            with open(sniper_config_path) as f:
                sc = json.load(f)
            expert_dir = os.path.join(self.model_dir, sc.get("streaming", {}).get("expert_dir", "bin"))
        else:
            expert_dir = os.path.join(self.model_dir, "bin")

        self.reader = MoEExpertReader(
            expert_dir, self.num_layers,
            num_workers=8, cache_size=self._cache_size
        )
        self.coact = CoActivationTracker(self.num_layers, warmup_tokens=3)

        # Tokenizer
        from tokenizers import Tokenizer
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            self.tokenizer = Tokenizer.from_file(tok_path)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # KV cache
        self.cache = self.model.make_cache()

        print(f"Gemma 4 Sniper loaded: {pinned_gb:.1f} GB pinned, "
              f"cache={self._cache_size}, layers={self.num_layers}")
        return pinned_gb

    def reset_cache(self):
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        """Forward pass with SSD-streamed experts."""
        from mlx_lm.models.base import create_attention_mask

        h = self.model.model.embed_tokens(input_ids)
        # Gemma scaling
        h = h * (self.model.args.hidden_size ** 0.5)

        mask = create_attention_mask(h, self.cache[0] if self.cache else None)

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            cache_i = self.cache[i] if self.cache else None

            # 1. Attention
            residual = h
            h_norm = layer.input_layernorm(h)
            h_attn = layer.self_attn(h_norm, mask=mask, cache=cache_i)
            h_attn = layer.post_attention_layernorm(h_attn)
            h = residual + h_attn
            mx.eval(h)

            # 2. Dense MLP
            residual = h
            h_ff = layer.pre_feedforward_layernorm(h)
            h_ff = layer.mlp(h_ff)

            if layer.enable_moe_block:
                h_dense = layer.post_feedforward_layernorm_1(h_ff)

                # 3. Router (local)
                B, L, D = residual.shape
                residual_flat = residual.reshape(-1, D)
                # Manual routing with bias (instead of layer.router)
                router = layer.router
                x_normed = router._inline_rms_norm(residual_flat)
                x_normed = x_normed * router.scale * (router.hidden_size ** -0.5)
                scores = router.proj(x_normed)

                # Routing bias: steer toward cached experts
                ROUTING_BIAS = 1.5
                if ROUTING_BIAS > 0 and self.reader.lru is not None:
                    bias_np = np.zeros(scores.shape[-1], dtype=np.float32)
                    for (li, eid) in self.reader.lru.cache.keys():
                        if li == i:
                            bias_np[eid] = ROUTING_BIAS
                    if bias_np.any():
                        scores = scores + mx.array(bias_np)

                probs = mx.softmax(scores, axis=-1)
                top_k_indices = mx.argpartition(-probs, kth=router.top_k - 1, axis=-1)[..., :router.top_k]
                top_k_weights = mx.take_along_axis(probs, top_k_indices, axis=-1)
                top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
                expert_scales = router.per_expert_scale[top_k_indices]
                top_k_weights = top_k_weights * expert_scales

                # Expert input
                moe_input = layer.pre_feedforward_layernorm_2(residual_flat)
                mx.eval(moe_input, top_k_indices, top_k_weights)

                top_k_indices_r = top_k_indices.reshape(B, L, -1)
                top_k_weights_r = top_k_weights.reshape(B, L, -1)

                active_ids = list(set(int(e) for e in np.array(top_k_indices_r).flatten()))
                self.coact.record_layer(i, active_ids)

                # Prefetch
                if self._enable_prediction and self.coact.ready and i + 1 < self.num_layers:
                    predicted = self.coact.predict_next_layer(i, active_ids, top_k=6)
                    if predicted:
                        to_fetch = [eid for eid in predicted
                                    if self.reader.lru and self.reader.lru.get(i + 1, eid) is None]
                        if to_fetch:
                            self.reader.prefetch_experts(i + 1, to_fetch)

                if i + 1 < self.num_layers:
                    self.reader.prefetch_experts(i + 1, active_ids)

                # 4. Expert FFN (from SSD)
                expert_data = self.reader.get_experts(i, active_ids)
                moe_input_r = moe_input.reshape(B, L, D)
                expert_out = run_expert_ffn(moe_input_r, expert_data, top_k_indices_r, top_k_weights_r)
                h_moe = layer.post_feedforward_layernorm_2(expert_out)

                h_ff = h_dense + h_moe

            # Final norm + residual + scalar
            h_ff = layer.post_feedforward_layernorm(h_ff)
            h = residual + h_ff
            h = h * layer.layer_scalar
            mx.eval(h)

            del expert_data, expert_out, moe_input
            mx.clear_cache()

        self.coact.end_token()
        h = self.model.model.norm(h)

        # Output head
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(h)
        else:
            return self.model.lm_head(h)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--cache-size", type=int, default=4000)
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    args = parser.parse_args()

    engine = MoESniperEngineGemma4(
        model_dir=args.model_dir,
        cache_size=args.cache_size,
    )
    engine.load()

    # Tokenize with Gemma 4 chat template
    NL = chr(10)
    if hasattr(engine.tokenizer, 'encode') and hasattr(engine.tokenizer.encode("test"), 'ids'):
        prompt_toks = engine.tokenizer.encode(args.prompt).ids
        user_toks = engine.tokenizer.encode("user" + NL).ids
        model_toks = engine.tokenizer.encode("model" + NL).ids
    else:
        prompt_toks = engine.tokenizer.encode(args.prompt)
        user_toks = engine.tokenizer.encode("user" + NL)
        model_toks = engine.tokenizer.encode("model" + NL)
    # bos=2, turn_start=105, turn_end=106, newline=107
    tokens = [2, 105] + user_toks + prompt_toks + [106, 107, 105] + model_toks

    input_ids = mx.array([tokens])

    # Prefill
    logits = engine.forward(input_ids)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[0, -1]).item())

    # Generate
    generated = [next_token]
    input_ids = mx.array([[next_token]])
    t_start = time.time()
    for step in range(args.tokens - 1):
        logits = engine.forward(input_ids)
        mx.eval(logits)
        nt = int(mx.argmax(logits[0, -1]).item())
        generated.append(nt)
        input_ids = mx.array([[nt]])
        # EOS: <eos>=1, <turn|>=106
        if nt in [1, 106]:
            break

    total = time.time() - t_start
    tps = len(generated) / total

    if hasattr(engine.tokenizer, 'decode'):
        text = engine.tokenizer.decode(generated[:50])
    else:
        text = str(generated[:20])

    hit_rate = engine.reader.lru.hit_rate() if engine.reader.lru else 0
    mem = mx.get_active_memory() / 1e9

    print(f"\nResults:")
    print(f"  tok/s: {tps:.2f}")
    print(f"  cache hit: {hit_rate:.1%}")
    print(f"  memory: {mem:.2f} GB")
    print(f"  output: {text[:150]}...")
    print(f"  stats: {engine.reader.stats()}")
