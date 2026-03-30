"""
Flash MoE — The Agentic Speed Breakthrough.

Qwen3.5-35B-A3B at Q4_K_M (22 GB) on 16 GB Mac.
Only 1.4 GB pinned in RAM. 256 experts on SSD.
Router picks 8 active experts → pread only those → gather_qmm → discard.
"""

import time
import os
import sys
import json
import gc

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from expert_io import MoEExpertReader

MODEL_DIR = os.environ.get("MOE_MODEL_DIR", os.path.expanduser("~/models/qwen35-35b-moe-stream"))
BITS = 4
GROUP_SIZE = 64


def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    """
    Compute MoE FFN using gather_qmm — matches native SwitchGLU math exactly.

    Stacks 8 active experts into [K, out, in] mini-tensors, remaps indices
    to local [0..K-1], calls gather_qmm. Eliminates accumulator divergence.
    """
    active_ids = sorted(expert_data.keys())
    K = len(active_ids)

    # Build global→local index mapping
    id_to_local = {eid: i for i, eid in enumerate(active_ids)}

    # Remap indices: global expert IDs → local [0..K-1]
    inds_np = np.array(top_k_indices)
    local_np = np.vectorize(lambda x: id_to_local.get(int(x), 0))(inds_np)
    local_indices = mx.array(local_np)

    # Stack expert weights into [K, out, in] mini-tensors
    def stack_proj(proj_name):
        w = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj_name}.weight"] for eid in active_ids])
        s = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj_name}.scales"] for eid in active_ids])
        b = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj_name}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    # Expand x for gather: [B, L, 1, 1, D]
    x_exp = mx.expand_dims(x, (-2, -3))

    # SwiGLU via gather_qmm
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

    hidden = nn.silu(gate_out) * up_out

    down_out = mx.gather_qmm(
        hidden, down_w, scales=down_s, biases=down_b,
        rhs_indices=local_indices, transpose=True,
        group_size=GROUP_SIZE, bits=BITS,
    )

    # Squeeze and weight by router scores
    out = down_out.squeeze(-2)  # [B, L, K, D]
    out = (out * top_k_weights[..., None]).sum(axis=-2)  # [B, L, D]

    return out


def main():
    print("=" * 60)
    print("  FLASH MOE — 35B Model at Agentic Speed")
    print("  22 GB model · 16 GB RAM · gather_qmm expert fusion")
    print("=" * 60)

    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    streaming = config["streaming"]

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs(
        model_type=config.get("model_type"),
        hidden_size=config["hidden_size"],
        num_hidden_layers=num_layers,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        head_dim=config.get("head_dim"),
        tie_word_embeddings=config["tie_word_embeddings"],
        num_experts=config["num_experts"],
        num_experts_per_tok=config["num_experts_per_tok"],
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

    print(f"\nCreating model...")
    model = TextModel(args)

    from mlx_lm.models.switch_layers import SwitchLinear

    SSM_PROTECT = {"conv1d"}

    def should_quantize(path, module):
        if isinstance(module, nn.Embedding):
            return True
        if isinstance(module, SwitchLinear):
            return True
        if not isinstance(module, nn.Linear):
            return False
        if any(k in path for k in SSM_PROTECT):
            return False
        if module.weight.shape[-1] < GROUP_SIZE:
            return False
        return True

    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS, class_predicate=should_quantize)

    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)

    print("Loading pinned weights...")
    t0 = time.time()
    pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
    model.load_weights(list(pinned.items()), strict=False)

    params = [p for name, p in tree_flatten(model.parameters()) if "switch_mlp" not in name]
    mx.eval(*params)
    del pinned
    gc.collect()
    mx.clear_cache()

    pinned_gb = sum(p.nbytes for p in params) / 1e9
    print(f"  {pinned_gb:.2f} GB in {time.time()-t0:.1f}s")

    print("\nInitializing MoE expert reader (8 threads, F_NOCACHE)...")
    reader = MoEExpertReader(
        f"{MODEL_DIR}/{streaming['expert_dir']}",
        num_layers, num_workers=8,
    )

    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    cache = model.make_cache()

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    def forward_moe_streaming(input_ids):
        h = model.model.embed_tokens(input_ids)

        ssm_idx = model.model.ssm_idx
        fa_idx = model.model.fa_idx
        fa_mask = create_attention_mask(h, cache[fa_idx])
        ssm_mask = create_ssm_mask(h, cache[ssm_idx])

        for i in range(num_layers):
            layer = model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
            h = h + attn_out
            mx.eval(h)

            normed = layer.post_attention_layernorm(h)

            # Router
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            # Snipe active experts from SSD
            active_ids = list(set(int(e) for e in np.array(inds).flatten()))
            if i + 1 < num_layers:
                reader.prefetch_experts(i + 1, active_ids)

            expert_data = reader.get_experts(i, active_ids)

            # gather_qmm fusion — matches native SwitchGLU math
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            # Shared expert
            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

        h = model.model.norm(h)
        return model.lm_head(h)

    # Test
    prompt = "What is the capital of France?"
    print(f"\nPrompt: {prompt}")

    tokens = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "Be concise."},
             {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
    )
    input_ids = mx.array([tokens])

    import subprocess
    subprocess.run(["sudo", "purge"], capture_output=True)

    print(f"\n--- Prefill ({len(tokens)} tokens) ---")
    t0 = time.time()
    logits = forward_moe_streaming(input_ids)
    mx.eval(logits)
    print(f"  {time.time()-t0:.2f}s")

    print(f"\n--- Decode (max 100) ---")
    generated = []
    t_decode = time.time()
    in_think = True  # template starts in think mode

    for step in range(100):
        next_logits = logits[:, -1, :]
        if generated:
            seen = mx.array(list(set(generated[-50:])))
            pl = next_logits[:, seen]
            pl = mx.where(pl > 0, pl / 1.15, pl * 1.15)
            next_logits[:, seen] = pl

        probs = mx.softmax(next_logits / 0.7, axis=-1)
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cumsum = mx.cumsum(sorted_p, axis=-1)
        mask = (cumsum - sorted_p) <= 0.9
        sorted_p = sorted_p * mask
        sorted_p = sorted_p / (sorted_p.sum(axis=-1, keepdims=True) + 1e-10)
        token = mx.random.categorical(mx.log(sorted_p + 1e-10))
        token = mx.take_along_axis(sorted_idx, token[:, None], axis=-1).squeeze(-1)
        mx.eval(token)
        token_id = token.item()

        if token_id in (248044, 248045, 248046):
            break

        generated.append(token_id)
        chunk = tokenizer.decode([token_id])

        if "<|im_end|>" in chunk:
            break

        if "</think>" in chunk:
            in_think = False
            print("\n[Answer] ", end="", flush=True)
        elif not in_think:
            print(chunk, end="", flush=True)
        elif in_think and step % 10 == 0:
            print(".", end="", flush=True)  # show thinking progress

        logits = forward_moe_streaming(token.reshape(1, 1))
        mx.eval(logits)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    # Strip think content
    if "</think>" in output:
        output = output.split("</think>", 1)[1].strip()
    print(f"\n\nDecode: {n} tokens in {t_total:.1f}s ({tps:.2f} tok/s)")
    print(f"Memory: {mx.get_active_memory()/1e9:.2f} GB")

    print(f"\n{'='*60}")
    print(f"Q: {prompt}")
    print(f"A: {output}")
    print(f"{'='*60}")

    reader.close()


if __name__ == "__main__":
    main()
