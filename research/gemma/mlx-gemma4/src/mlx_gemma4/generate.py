"""Generate text with Gemma 4 on MLX."""
import os, sys, time, json, gc
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .model import Model, ModelArgs


def load_model(model_path, max_memory_gb=None):
    """Load a Gemma 4 model from a local or HuggingFace path.

    Supports:
      - Local directory with safetensors + config.json
      - HuggingFace repo ID (will download automatically)
    """
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
    )

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
