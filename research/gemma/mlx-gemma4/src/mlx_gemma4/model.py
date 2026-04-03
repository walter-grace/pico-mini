"""
MLX model implementation for Google Gemma 4 (26B-A4B) MoE.

Architecture: gemma4_text
- 30 decoder layers with mixed sliding/full attention
- 128 experts, top-8 routing per token
- Dense MLP runs in parallel with MoE on every layer
- K=V weight sharing (attention_k_eq_v)
- Two RoPE configs: sliding (theta=10k) vs full (theta=1M, partial_rotary_factor=0.25)
- Global attention uses larger head_dim (512) vs sliding (256)

Reference: HuggingFace transformers Gemma4TextModel
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def _default_layer_types() -> List[str]:
    """Default Gemma4 layer pattern: 5x sliding then 1x full, repeated."""
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    return (pattern * 5)[:30]  # 30 layers total


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"
    final_logit_softcapping: float = 30.0
    sliding_window: int = 1024
    max_position_embeddings: int = 262144
    attention_k_eq_v: bool = True
    enable_moe_block: bool = True
    tie_word_embeddings: bool = True
    layer_types: List[str] = field(default_factory=_default_layer_types)
    # KV sharing: last N layers share k_proj/k_norm with a base layer
    num_kv_shared_layers: int = 0
    # Per-layer embedding (PLE): injects per-layer input before attention
    hidden_size_per_layer_input: int = 0
    # Norm mode: "gguf" uses w*norm(x), "hf" uses (1+w)*norm(x)
    norm_mode: str = "gguf"
    # RoPE
    rope_theta_sliding: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    partial_rotary_factor: float = 0.25  # only for full attention layers


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #

# Global norm mode — set before creating model
_NORM_MODE = "gguf"  # "gguf" = w*norm(x), "hf" = (1+w)*norm(x)

class RMSNorm(nn.Module):
    """Gemma-style RMSNorm. Mode controls weight interpretation."""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if _NORM_MODE == "hf":
            return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
        else:
            # GGUF norm_shift=0.0: weights are final multipliers
            return mx.fast.rms_norm(x, self.weight, self.eps)


class BareRMSNorm(nn.Module):
    """RMSNorm without learnable scale (used for v_norm)."""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._dims = dims

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, mx.ones((self._dims,)), self.eps)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

class Attention(nn.Module):
    """
    Gemma4 attention with two modes:
      - sliding_attention: sliding window, standard RoPE, head_dim=256, 8 KV heads
      - full_attention: full context, partial RoPE, global_head_dim=512, 2 KV heads

    attention_k_eq_v: K and V share the same projection weights. The HF weights
    have separate k_proj and v_proj entries but they're identical. We keep both
    for weight loading compatibility but only use k_proj output for both K and V.
    """
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.is_kv_shared = False  # set by Gemma4TextModel after init
        self.store_kv_for_sharing = False  # set by Gemma4TextModel after init
        self._shared_kv = None  # storage for shared KV activations
        # K=V sharing only applies to full (non-sliding) attention layers
        self.use_kv_sharing = args.attention_k_eq_v and not self.is_sliding

        self.n_heads = args.num_attention_heads

        if self.is_sliding:
            self.n_kv_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            rope_dims = args.head_dim
            rope_theta = args.rope_theta_sliding
        else:
            self.n_kv_heads = args.num_global_key_value_heads or args.num_key_value_heads
            self.head_dim = args.global_head_dim or args.head_dim
            # Partial rotary: only rotate first partial_rotary_factor of dims
            rope_dims = int(args.global_head_dim * args.partial_rotary_factor)
            rope_theta = args.rope_theta_global

        self.scale = 1.0  # HF Gemma4 uses scaling=1.0; q_norm/k_norm handle magnitude

        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        # v_proj needed for sliding layers; dropped for full layers with K=V sharing
        if not self.use_kv_sharing:
            self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.v_norm = BareRMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(rope_dims, traditional=False, base=rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv_store: Optional[dict] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)

        if self.is_kv_shared and shared_kv_store and self.layer_idx in shared_kv_store:
            # Reuse KV activations from the base layer's cache
            keys, values = shared_kv_store[self.layer_idx]
        else:
            keys = self.k_proj(x)
            values = keys if self.use_kv_sharing else self.v_proj(x)
            keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys)
            values = self.v_norm(values)

            # Store KV for shared layers to reuse
            if self.store_kv_for_sharing and shared_kv_store is not None:
                shared_kv_store[self.layer_idx] = (keys, values)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# --------------------------------------------------------------------------- #
# Dense MLP
# --------------------------------------------------------------------------- #

class DenseMLP(nn.Module):
    """Standard gated MLP with gelu_pytorch_tanh activation."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

class Router(nn.Module):
    """
    Gemma4 MoE router with learnable scale and per-expert scales.

    Forward:
      1. Inline RMS norm (no learnable weight)
      2. Scale by self.scale * hidden_size^{-0.5}
      3. Linear projection to num_experts
      4. Softmax -> top-k selection
      5. Renormalize top-k weights, multiply by per_expert_scale
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.top_k = args.top_k_experts

        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        # Learnable per-dimension scale (shape matches hidden_size)
        self.scale = mx.ones((args.hidden_size,))
        # Per-expert scales
        self.per_expert_scale = mx.ones((args.num_experts,))

    def _inline_rms_norm(self, x: mx.array) -> mx.array:
        """RMS norm without learnable weight."""
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + 1e-6)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # x: [B, L, D] or [B*L, D]
        orig_shape = x.shape
        if x.ndim == 3:
            B, L, D = x.shape
            x = x.reshape(-1, D)
        else:
            D = x.shape[-1]

        # Inline RMS norm (no learnable weight)
        x_normed = self._inline_rms_norm(x)
        # Scale
        x_normed = x_normed * self.scale * (self.hidden_size ** -0.5)
        # Project to expert logits
        scores = self.proj(x_normed)
        # Softmax over experts
        probs = mx.softmax(scores, axis=-1)
        # Top-k
        top_k_indices = mx.argpartition(-probs, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        # Gather the weights for selected experts
        top_k_weights = mx.take_along_axis(probs, top_k_indices, axis=-1)
        # Renormalize
        top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
        # Per-expert scaling
        expert_scales = self.per_expert_scale[top_k_indices]
        top_k_weights = top_k_weights * expert_scales

        if len(orig_shape) == 3:
            top_k_weights = top_k_weights.reshape(B, L, self.top_k)
            top_k_indices = top_k_indices.reshape(B, L, self.top_k)

        return top_k_weights, top_k_indices


# --------------------------------------------------------------------------- #
# Expert Block
# --------------------------------------------------------------------------- #

class GELUGate(nn.Module):
    """GELU gated activation for MoE (replaces SwiGLU)."""
    def __call__(self, gate, up):
        return nn.gelu_approx(gate) * up


class Experts(nn.Module):
    """
    MoE experts using SwitchGLU from mlx-lm.

    Uses separate gate_proj + up_proj (not fused gate_up_proj) for
    compatibility with mlx-lm's SwitchLinear quantization.

    The sanitize method splits fused gate_up_proj → gate_proj + up_proj.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU
        self.switch_glu = SwitchGLU(
            input_dims=args.hidden_size,
            hidden_dims=args.moe_intermediate_size,
            num_experts=args.num_experts,
            activation=GELUGate(),
            bias=False,
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """
        Args:
            x: [B*L, D] or [B, L, D]
            indices: [B*L, top_k] or [B, L, top_k]

        Returns: [B*L, top_k, D] or [B, L, top_k, D]
        """
        if x.ndim == 3:
            B, L, D = x.shape
            out = self.switch_glu(x.reshape(-1, D), indices.reshape(-1, indices.shape[-1]))
            return out.reshape(B, L, indices.shape[-1], -1)
        return self.switch_glu(x, indices)


# --------------------------------------------------------------------------- #
# Decoder Layer
# --------------------------------------------------------------------------- #

class DecoderLayer(nn.Module):
    """
    Gemma4 decoder layer with dense MLP + MoE parallel residual structure.

    Forward pass:
      1. Attention with residual
      2. Dense MLP with pre/post norms
      3. MoE block (if enabled) added in parallel to dense output
      4. Residual connection + layer_scalar
    """
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.enable_moe_block = args.enable_moe_block

        # Attention
        self.self_attn = Attention(args, layer_idx)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Dense MLP
        self.mlp = DenseMLP(args)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if self.enable_moe_block:
            # MoE norms
            self.post_feedforward_layernorm_1 = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

            # Router + Experts
            self.router = Router(args)
            self.experts = Experts(args)

        # Per-layer learned scalar multiplier
        self.layer_scalar = mx.ones((1,))

        # PLE: per-layer embedding injection
        self.has_ple = (args.hidden_size_per_layer_input or 0) > 0
        if self.has_ple:
            ple_dim = args.hidden_size_per_layer_input
            # Gate: Linear(hidden_size, ple_dim) → sigmoid → element-wise gate
            self.per_layer_input_gate = nn.Linear(args.hidden_size, ple_dim, bias=False)
            # Projection: Linear(ple_dim, hidden_size) → project back to hidden
            self.per_layer_projection = nn.Linear(ple_dim, args.hidden_size, bias=False)
            self.post_per_layer_input_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
    ) -> mx.array:
        # 0. PLE injection (before attention)
        if self.has_ple and per_layer_input is not None:
            # Gate: hidden_states → sigmoid → element-wise multiply with PLE input
            gate = mx.sigmoid(self.per_layer_input_gate(x))  # [B, L, ple_dim]
            gated_input = gate * per_layer_input  # [B, L, ple_dim]
            ple_out = self.per_layer_projection(gated_input)  # [B, L, hidden_size]
            ple_out = self.post_per_layer_input_norm(ple_out)
            x = x + ple_out

        # 1. Attention with pre/post norms and residual
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        h = residual + h

        # 2. Feed-forward (dense MLP, optionally combined with MoE)
        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)

        if self.enable_moe_block:
            # Dense MLP output -> post_feedforward_layernorm_1
            h_dense = self.post_feedforward_layernorm_1(h)

            # MoE: router takes residual (pre-MLP hidden states), NOT normed
            B, L, D = residual.shape
            residual_flat = residual.reshape(-1, D)
            top_k_weights, top_k_indices = self.router(residual_flat)

            # Expert input: pre_feedforward_layernorm_2 applied to residual
            moe_input = self.pre_feedforward_layernorm_2(residual_flat)
            expert_out = self.experts(
                moe_input.reshape(B, L, D), top_k_indices.reshape(B, L, -1)
            )
            # Weighted sum over top-k experts
            top_k_weights_r = top_k_weights.reshape(B, L, -1)
            weighted_out = (expert_out * mx.expand_dims(top_k_weights_r, -1)).sum(axis=-2)
            h_moe = self.post_feedforward_layernorm_2(weighted_out)

            # Combine dense + MoE
            h = h_dense + h_moe

        # Final post-feedforward norm + residual
        h = self.post_feedforward_layernorm(h)
        h = residual + h

        h = h * self.layer_scalar

        return h


# --------------------------------------------------------------------------- #
# Full Model
# --------------------------------------------------------------------------- #

class Gemma4TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # KV sharing: last N layers reuse KV ACTIVATIONS from a base layer's cache
        # (NOT weight sharing — each layer keeps its own k_proj/v_proj weights)
        self.num_kv_shared = getattr(args, "num_kv_shared_layers", 0) or 0
        if self.num_kv_shared > 0:
            self.first_shared = args.num_hidden_layers - self.num_kv_shared
            # Mark shared layers
            for i in range(self.first_shared, args.num_hidden_layers):
                self.layers[i].self_attn.is_kv_shared = True
            # The base layer stores its KV for shared layers to reuse
            base = self.first_shared - 1
            self.layers[base].self_attn.store_kv_for_sharing = True

        # PLE embedding (separate from main embedding, no sqrt scaling)
        self.has_ple = (args.hidden_size_per_layer_input or 0) > 0
        if self.has_ple:
            self.per_layer_embed_tokens = nn.Embedding(
                args.vocab_size, args.hidden_size_per_layer_input
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        # Gemma scaling: multiply embeddings by sqrt(hidden_size)
        h = h * mx.array(self.args.hidden_size ** 0.5, dtype=h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        # Build masks: one for global attention, one for sliding window
        # Find first global layer to get its cache for mask creation
        first_global_idx = None
        first_sliding_idx = None
        for i, lt in enumerate(self.args.layer_types):
            if lt == "full_attention" and first_global_idx is None:
                first_global_idx = i
            if lt == "sliding_attention" and first_sliding_idx is None:
                first_sliding_idx = i
            if first_global_idx is not None and first_sliding_idx is not None:
                break

        global_mask = create_attention_mask(
            h, cache[first_global_idx] if first_global_idx is not None else None
        )
        sliding_mask = create_attention_mask(
            h,
            cache[first_sliding_idx] if first_sliding_idx is not None else None,
            window_size=self.args.sliding_window,
        )

        # PLE: compute per-layer input embedding (no sqrt scaling)
        ple_input = None
        if self.has_ple:
            ple_input = self.per_layer_embed_tokens(inputs)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = self.args.layer_types[i] == "full_attention"
            mask = global_mask if is_global else sliding_mask
            h = layer(h, mask, c, per_layer_input=ple_input)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        # Logit soft-capping
        cap = self.args.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = mx.tanh(logits / cap) * cap

        return logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        """Create KV caches: RotatingKVCache for sliding layers, KVCache for global."""
        caches = []
        for i in range(self.args.num_hidden_layers):
            if self.args.layer_types[i] == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Convert HuggingFace weight names to our model's expected names.

        HF prefix: model.language_model.layers.N.xxx
        Our prefix: model.layers.N.xxx

        Also handles:
        - Stripping "model.language_model." prefix -> "model."
        - v_proj weights are dropped when attention_k_eq_v (K=V sharing)
        - embed_tokens maps to model.embed_tokens
        - lm_head is dropped if tie_word_embeddings
        """
        new_weights = {}
        for k, v in weights.items():
            # Strip the "model.language_model." prefix
            new_key = k
            if new_key.startswith("model.language_model."):
                new_key = "model." + new_key[len("model.language_model."):]

            # layer_scalar.weight → layer_scalar (bare parameter, not a module)
            new_key = new_key.replace("layer_scalar.weight", "layer_scalar")

            # Drop v_proj only for full attention layers with K=V sharing
            # Sliding layers still need v_proj even when attention_k_eq_v is true
            if self.args.attention_k_eq_v and "v_proj" in new_key:
                # Extract layer index to check if it's a full attention layer
                layer_match = re.search(r'layers\.(\d+)\.', new_key)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if self.args.layer_types[layer_idx] != "sliding_attention":
                        continue  # Drop v_proj for full attention layers
                # If no layer index found, keep the weight

            # Drop lm_head when tied
            if self.args.tie_word_embeddings and new_key == "lm_head.weight":
                continue

            # Split fused gate_up_proj → gate_proj + up_proj for SwitchGLU
            if "experts.gate_up_proj" in new_key:
                # Map: experts.gate_up_proj.X → experts.switch_glu.gate_proj.X
                #                               + experts.switch_glu.up_proj.X
                suffix = new_key.split("experts.gate_up_proj")[-1]  # .weight/.scales/.biases
                base = new_key.split("experts.gate_up_proj")[0]
                # Split along first expert-local dim (dim 1 for [E, 2*I, D])
                half = v.shape[1] // 2
                gate_v = v[:, :half]
                up_v = v[:, half:]
                new_weights[base + "experts.switch_glu.gate_proj" + suffix] = gate_v
                new_weights[base + "experts.switch_glu.up_proj" + suffix] = up_v
                continue

            # Map per-expert scale to router
            if new_key.endswith("experts.down_proj.scale"):
                # This is the per-expert scale, not a quantization scale
                new_key = new_key.replace("experts.down_proj.scale", "router.per_expert_scale")
                new_weights[new_key] = v
                continue

            # Map experts.down_proj → experts.switch_glu.down_proj
            if "experts.down_proj" in new_key:
                new_key = new_key.replace("experts.down_proj", "experts.switch_glu.down_proj")

            new_weights[new_key] = v

        return new_weights
