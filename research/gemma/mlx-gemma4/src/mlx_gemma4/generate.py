"""Generate text with Gemma 4 on MLX."""
import os, sys, time, json, gc, re
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .model import Model, ModelArgs


def load_model(model_path, max_memory_gb=None):
    """Load a Gemma 4 model from a local path, HuggingFace repo, or GGUF file.

    Supports:
      - GGUF file (.gguf) — loads directly, extracts config from weights
      - Local directory with safetensors + config.json
      - HuggingFace repo ID (will download automatically)
    """
    is_gguf = model_path.endswith(".gguf")

    if is_gguf:
        return _load_from_gguf(model_path)

    # Download from HF if needed
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_path}...")
        model_path = snapshot_download(model_path)

    # Load config
    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path))
    tc = config.get("text_config", config)

    # Build args
    args = ModelArgs(
        model_type=tc.get("model_type", "gemma4"),
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc.get("num_key_value_heads") or tc.get("num_attention_heads", 16),
        num_global_key_value_heads=tc.get("num_global_key_value_heads") or tc.get("num_key_value_heads") or 2,
        head_dim=tc.get("head_dim", 256),
        global_head_dim=tc.get("global_head_dim", 512),
        intermediate_size=tc.get("intermediate_size", 2112),
        moe_intermediate_size=tc.get("moe_intermediate_size", 704),
        num_experts=tc.get("num_experts", 0) or 0,
        top_k_experts=tc.get("top_k_experts", 8),
        vocab_size=tc.get("vocab_size", 262144),
        rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
        sliding_window=tc.get("sliding_window", 1024),
        hidden_activation=tc.get("hidden_activation", "gelu_pytorch_tanh"),
        final_logit_softcapping=tc.get("final_logit_softcapping", 30.0),
        attention_k_eq_v=tc.get("attention_k_eq_v", True),
        enable_moe_block=tc.get("enable_moe_block", False),
        tie_word_embeddings=config.get("tie_word_embeddings", True),
        max_position_embeddings=tc.get("max_position_embeddings", 262144),
        layer_types=tc.get("layer_types", []),
        num_kv_shared_layers=tc.get("num_kv_shared_layers", 0) or 0,
        hidden_size_per_layer_input=tc.get("hidden_size_per_layer_input", 0) or 0,
    )

    # Set norm mode: GGUF-converted weights use raw multipliers
    from . import model as model_mod
    if config.get("quantization"):
        # Our converted models come from GGUF (norm_shift=0)
        model_mod._NORM_MODE = "gguf"
    else:
        # Direct HF models use offset from 1.0
        model_mod._NORM_MODE = "hf"

    # Create model
    model = Model(args)

    # Quantize if weights are quantized
    quant_config = config.get("quantization", tc.get("quantization"))
    if quant_config:
        bits = quant_config.get("bits", 4)
        group_size = quant_config.get("group_size", 64)
        from mlx_lm.models.switch_layers import SwitchLinear
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding): return True
            if isinstance(module, SwitchLinear): return True
            if not isinstance(module, nn.Linear): return False
            if "router" in path: return False
            return True
        nn.quantize(model, group_size=group_size, bits=bits,
                     class_predicate=should_quantize)

    # Load weights
    import glob
    weight_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Sanitize
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)

    # Eval all params
    params = tree_flatten(model.parameters())
    mx.eval(*[p for _, p in params])
    del weights; gc.collect()

    param_bytes = sum(p.nbytes for _, p in params)
    print(f"Loaded: {param_bytes/1e9:.2f} GB, {len(params)} params")

    # Load tokenizer (fix Gemma 4 extra_special_tokens bug in transformers 4.57)
    tc_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tc_path):
        import json as _json
        _tc = _json.load(open(tc_path))
        if "extra_special_tokens" in _tc and isinstance(_tc["extra_special_tokens"], list):
            del _tc["extra_special_tokens"]
            with open(tc_path, "w") as _f:
                _json.dump(_tc, _f, indent=2)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer


# GGUF name mapping: blk.N.xxx → model.layers.N.xxx
_GGUF_NAME_MAP = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_ffw_norm_1": "post_feedforward_layernorm_1",
    "post_ffw_norm_2": "post_feedforward_layernorm_2",
    "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
    "layer_output_scale": "layer_scalar",
    "ffn_gate_inp": "router.proj",
}

_GGUF_GLOBAL_MAP = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
}


def _map_gguf_key(key):
    """Map a GGUF key to our model's expected key."""
    # Global tensors
    for gguf_prefix, our_name in _GGUF_GLOBAL_MAP.items():
        if key.startswith(gguf_prefix):
            suffix = key[len(gguf_prefix):]
            return our_name + suffix

    # Layer tensors: blk.N.xxx → model.layers.N.xxx
    m = re.match(r"blk\.(\d+)\.(.*)", key)
    if m:
        layer_idx = m.group(1)
        tensor_name = m.group(2)
        # Split off .weight/.scales/.biases suffix
        parts = tensor_name.rsplit(".", 1)
        base = parts[0]
        suffix = "." + parts[1] if len(parts) > 1 else ""

        mapped = _GGUF_NAME_MAP.get(base, base)
        return f"model.layers.{layer_idx}.{mapped}{suffix}"

    return key  # fallback


def _load_from_gguf(gguf_path):
    """Load Gemma 4 directly from a GGUF file."""
    from . import model as model_mod
    model_mod._NORM_MODE = "gguf"

    print(f"Loading GGUF: {gguf_path}")
    t0 = time.time()

    weights_raw = mx.load(gguf_path)
    print(f"  Loaded {len(weights_raw)} tensors in {time.time()-t0:.1f}s")

    # Map GGUF names to our model names
    weights = {}
    for k, v in weights_raw.items():
        if "rope_freqs" in k:
            continue
        mapped = _map_gguf_key(k)
        weights[mapped] = v
    del weights_raw; gc.collect()

    # Detect model config from weight shapes
    # Find hidden_size from embed_tokens
    embed_w = weights.get("model.embed_tokens.weight")
    embed_s = weights.get("model.embed_tokens.scales")
    if embed_s is not None:
        # Quantized: hidden_size = weight_cols * 32 / bits
        # For Q8: weight (vocab, hidden/4), so hidden = cols * 4
        hidden_size = embed_w.shape[1] * 4  # Q8 packing
        vocab_size = embed_w.shape[0]
        bits = 8
        group_size = hidden_size // embed_s.shape[1]
    else:
        hidden_size = embed_w.shape[1]
        vocab_size = embed_w.shape[0]
        bits = 0
        group_size = 0

    # Count layers
    num_layers = 0
    for k in weights:
        m = re.match(r"model\.layers\.(\d+)\.", k)
        if m:
            num_layers = max(num_layers, int(m.group(1)) + 1)

    # Detect MoE
    has_moe = any("router" in k for k in weights)
    num_experts = 128 if has_moe else 0

    # Detect attention config from q_proj shape
    q_w = weights.get("model.layers.0.self_attn.q_proj.weight")
    k_w = weights.get("model.layers.0.self_attn.k_proj.weight")
    if q_w is not None:
        q_out = q_w.shape[0]  # n_heads * head_dim
        k_out = k_w.shape[0]  # n_kv_heads * head_dim
    else:
        q_out = hidden_size
        k_out = hidden_size

    # Detect layer types from attention weight sizes across layers
    layer_types = []
    head_dim = 256  # default
    global_head_dim = 512  # default
    for li in range(num_layers):
        q_key = f"model.layers.{li}.self_attn.q_proj.weight"
        if q_key in weights:
            qshape = weights[q_key].shape[0]
            # Full attention has larger q_proj (n_heads * global_head_dim)
            if qshape > q_out * 0.5:  # heuristic
                # Check if this layer's q_proj matches the first layer
                if qshape == weights["model.layers.0.self_attn.q_proj.weight"].shape[0]:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")
        else:
            layer_types.append("sliding_attention")

    # Better detection: full attention layers have different k_proj size
    layer_types = []
    first_k_size = weights["model.layers.0.self_attn.k_proj.weight"].shape[0]
    for li in range(num_layers):
        k_key = f"model.layers.{li}.self_attn.k_proj.weight"
        if k_key in weights:
            if weights[k_key].shape[0] != first_k_size:
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")
        else:
            layer_types.append("sliding_attention")

    # Infer n_heads and n_kv_heads from sliding layers
    n_kv_heads_sliding = first_k_size // head_dim if bits == 0 else (first_k_size * 4) // head_dim
    n_heads = (q_out // head_dim) if bits == 0 else (q_out * 4) // head_dim

    # For full attention layers
    full_layers = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    if full_layers:
        full_k = weights[f"model.layers.{full_layers[0]}.self_attn.k_proj.weight"]
        n_kv_heads_global = full_k.shape[0] // global_head_dim if bits == 0 else (full_k.shape[0] * 4) // global_head_dim
    else:
        n_kv_heads_global = n_kv_heads_sliding

    print(f"  Config: hidden={hidden_size}, layers={num_layers}, heads={n_heads}, "
          f"kv_sliding={n_kv_heads_sliding}, kv_global={n_kv_heads_global}, "
          f"moe={has_moe}, bits={bits}")

    args = ModelArgs(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads_sliding,
        num_global_key_value_heads=n_kv_heads_global,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        vocab_size=vocab_size,
        num_experts=num_experts,
        enable_moe_block=has_moe,
        attention_k_eq_v=has_moe,  # MoE models use K=V on full layers
        layer_types=layer_types,
    )

    model = Model(args)

    # Quantize if GGUF is quantized
    if bits > 0:
        from mlx_lm.models.switch_layers import SwitchLinear
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding): return True
            if isinstance(module, SwitchLinear): return True
            if not isinstance(module, nn.Linear): return False
            if "router" in path: return False
            return True
        nn.quantize(model, group_size=group_size, bits=bits,
                     class_predicate=should_quantize)

    # Sanitize and load
    sanitized = model.sanitize(weights)
    model.load_weights(list(sanitized.items()), strict=False)

    params = tree_flatten(model.parameters())
    mx.eval(*[p for _, p in params])
    del weights, sanitized; gc.collect()

    param_bytes = sum(p.nbytes for _, p in params)
    print(f"  Model: {param_bytes/1e9:.2f} GB, {len(params)} params")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load tokenizer from HF
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-4-26B-A4B-it", trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it", trust_remote_code=True)

    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.0,
             chat_format=True):
    """Generate text from a prompt.

    Args:
        model: loaded Gemma 4 Model
        tokenizer: loaded tokenizer
        prompt: text prompt
        max_tokens: max tokens to generate
        temperature: sampling temperature (0 = greedy)
        chat_format: wrap prompt in Gemma 4 turn format

    Yields:
        token strings as they're generated
    """
    if chat_format:
        formatted = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    else:
        formatted = prompt

    tokens = tokenizer.encode(formatted)
    input_ids = mx.array([tokens])

    cache = model.make_cache()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)

    eos_ids = {1, 106}  # Gemma 4 EOS tokens
    stop_strings = {"<|turn>", "<turn|>"}

    for _ in range(max_tokens):
        if temperature == 0:
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            probs = mx.softmax(logits[:, -1, :] / temperature, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        mx.eval(next_token)
        tid = int(next_token.item())

        if tid in eos_ids:
            break

        chunk = tokenizer.decode([tid])

        # Filter thinking tags and stop strings
        if any(s in chunk for s in stop_strings):
            break
        if "<|channel>" in chunk or "<channel|>" in chunk:
            continue  # skip thinking markers

        yield chunk

        logits = model(next_token.reshape(1, 1), cache=cache)
        mx.eval(logits)
