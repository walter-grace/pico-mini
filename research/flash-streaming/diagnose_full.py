"""
Full diagnosis: buffers, double-injection, and layer-by-layer divergence.
Tests all remaining suspects in one run.

Test A: Does set_module_tensor_to_device work on non-meta tensors?
Test B: Buffer vs Parameter inventory — are A_log/dt_bias params or buffers?
Test C: Full injection + zero scan (both params AND buffers)
Test D: Layer-by-layer hidden state propagation (NaN/Inf/dead/exploding)
"""

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from pathlib import Path

GROUP_SIZE = 64

def dequantize_mlx_4bit(weight, scales, biases, group_size=64):
    if weight.dtype not in (torch.uint32, torch.int32):
        return weight.to(torch.bfloat16)
    orig_shape = weight.shape
    if weight.ndim == 3:
        batch = orig_shape[0]
        weight = weight.reshape(-1, orig_shape[-1])
        scales = scales.reshape(-1, scales.shape[-1])
        biases = biases.reshape(-1, biases.shape[-1])
    else:
        batch = None
    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=w.device)
    unpacked = (w.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).float()
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    dq = unpacked * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    result = dq.reshape(out_features, in_features).to(torch.bfloat16)
    if batch is not None:
        result = result.reshape(batch, orig_shape[1], in_features)
    return result

def remap_key(k):
    if k.startswith("language_model."):
        return k[len("language_model."):]
    return k

original_dir = "/workspace/qwen35-122b-a10b-4bit"
pinned_path = "/workspace/qwen35-122b-stream/pinned.safetensors"
expert_dir = "/workspace/qwen35-122b-stream/experts"

config = AutoConfig.from_pretrained(original_dir, trust_remote_code=True)
text_cfg = config.text_config if hasattr(config, 'text_config') else config

# ============================================================
print("=" * 60)
print("TEST A: set_module_tensor_to_device double-call")
print("=" * 60)
m = torch.nn.Linear(4, 4, bias=False)
set_module_tensor_to_device(m, 'weight', device='cpu', value=torch.zeros(4, 4))
print(f"  After zeros: mean={m.weight.mean():.4f}")
try:
    set_module_tensor_to_device(m, 'weight', device='cpu', value=torch.ones(4, 4))
    print(f"  After ones:  mean={m.weight.mean():.4f}")
    if m.weight.mean().item() > 0.5:
        print("  >>> PASS: double-call works")
    else:
        print("  >>> FAIL: double-call silently dropped!")
except Exception as e:
    print(f"  >>> ERROR: {e}")
del m

# ============================================================
print("\n" + "=" * 60)
print("TEST B: Buffer vs Parameter inventory")
print("=" * 60)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        text_cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)

param_names = set(n for n, _ in model.named_parameters())
buffer_names = set(n for n, _ in model.named_buffers())

print(f"  Total parameters: {len(param_names)}")
print(f"  Total buffers: {len(buffer_names)}")

print("\n  Layer 0 BUFFERS:")
for name, buf in model.named_buffers():
    if 'layers.0.' in name:
        print(f"    {name}: shape={list(buf.shape)} dtype={buf.dtype}")

print("\n  Layer 0 PARAMS with A_log/dt_bias/D:")
for name, _ in model.named_parameters():
    if 'layers.0.' in name and any(x in name for x in ['A_log', 'dt_bias', '.D', 'A_b']):
        print(f"    {name}")

with safe_open(pinned_path, framework='pt', device='cpu') as f:
    raw_keys = [k for k in f.keys()
                if not k.endswith('.scales') and not k.endswith('.biases') and not k.endswith('.weight')]
    buf_mapped = par_mapped = nowhere = 0
    nowhere_list = []
    for k in raw_keys:
        mapped = remap_key(k)
        if mapped in buffer_names:
            buf_mapped += 1
        elif mapped in param_names:
            par_mapped += 1
        else:
            nowhere += 1
            if 'layers.0.' in k or 'layers.1.' in k:
                nowhere_list.append((k, mapped, list(f.get_tensor(k).shape)))
    print(f"\n  Raw key mapping: {buf_mapped} -> buffers, {par_mapped} -> params, {nowhere} -> NOWHERE")
    if nowhere_list:
        print("  UNMAPPED (layer 0-1 samples):")
        for orig, mapped, shape in nowhere_list:
            print(f"    {orig} -> {mapped} {shape}")

# ============================================================
print("\n" + "=" * 60)
print("TEST C: Full injection + zero scan (params AND buffers)")
print("=" * 60)

for i in range(text_cfg.num_hidden_layers):
    model.model.layers[i].mlp.experts = torch.nn.ModuleList()

for name, param in list(model.named_parameters()):
    if param.device == torch.device("meta"):
        set_module_tensor_to_device(model, name, device="cpu",
            value=torch.zeros(param.shape, dtype=torch.bfloat16))
for name, buf in list(model.named_buffers()):
    if buf.device == torch.device("meta"):
        set_module_tensor_to_device(model, name, device="cpu",
            value=torch.zeros(buf.shape, dtype=buf.dtype))

model_param_names = set(n for n, _ in model.named_parameters())
model_buffer_names = set(n for n, _ in model.named_buffers())
loaded = skipped = 0

with safe_open(pinned_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    bases = {}
    for k in keys:
        if k.endswith(".scales"):
            bases.setdefault(k[:-7], {})["scales"] = k
        elif k.endswith(".biases"):
            bases.setdefault(k[:-7], {})["biases"] = k
        elif k.endswith(".weight"):
            bases.setdefault(k[:-7], {})["weight"] = k
        else:
            bases.setdefault(k, {})["raw"] = k

    for base, parts in bases.items():
        if "raw" in parts:
            raw_key = parts["raw"]
            mapped = remap_key(raw_key)
            tensor = f.get_tensor(raw_key)
            if mapped in model_param_names or mapped in model_buffer_names:
                try:
                    val = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                    set_module_tensor_to_device(model, mapped, device="cpu", value=val)
                    loaded += 1
                except Exception:
                    skipped += 1
            else:
                if "vision" not in mapped:
                    skipped += 1
                else:
                    skipped += 1
        elif "weight" in parts and "scales" in parts:
            w = f.get_tensor(parts["weight"])
            s = f.get_tensor(parts["scales"])
            b = f.get_tensor(parts["biases"])
            dq = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)
            target = remap_key(base) + ".weight"
            if target in model_param_names:
                model_shape = dict(model.named_parameters())[target].shape
                if dq.shape != model_shape and dq.numel() == model_shape.numel():
                    dq = dq.reshape(model_shape)
                try:
                    set_module_tensor_to_device(model, target, device="cpu", value=dq)
                    loaded += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1
            del dq
        elif "weight" in parts:
            w = f.get_tensor(parts["weight"])
            target = remap_key(base) + ".weight"
            if target in model_param_names:
                model_shape = dict(model.named_parameters())[target].shape
                val = w.to(torch.bfloat16)
                if val.shape != model_shape and val.numel() == model_shape.numel():
                    val = val.reshape(model_shape)
                try:
                    set_module_tensor_to_device(model, target, device="cpu", value=val)
                    loaded += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1

print(f"  Loaded: {loaded}, Skipped: {skipped}")

zero_params = []
for name, param in model.named_parameters():
    if "expert" in name:
        continue
    if param.float().std() < 1e-7 and abs(param.float().mean()) < 1e-7:
        zero_params.append(("PARAM", name, list(param.shape)))

zero_bufs = []
for name, buf in model.named_buffers():
    if "expert" in name:
        continue
    if buf.float().std() < 1e-7 and abs(buf.float().mean()) < 1e-7:
        zero_bufs.append(("BUFFER", name, list(buf.shape), buf.dtype))

print(f"\n  Zero PARAMS: {len(zero_params)}")
for kind, name, shape in zero_params[:10]:
    print(f"    {name}: {shape}")

print(f"\n  Zero BUFFERS: {len(zero_bufs)}")
for kind, name, shape, dtype in zero_bufs[:20]:
    print(f"    {name}: {shape} dtype={dtype}")

# ============================================================
print("\n" + "=" * 60)
print("TEST D: Layer-by-layer hidden state check (first 8 layers)")
print("=" * 60)

model.eval()

for i in range(8):
    layer = model.model.layers[i]
    if hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
        moe = layer.mlp
        gate = moe.gate
        shared_expert = getattr(moe, 'shared_expert', None)
        shared_expert_gate = getattr(moe, 'shared_expert_gate', None)
        top_k = text_cfg.num_experts_per_tok

        def make_forward(gate, shared_expert, shared_expert_gate, layer_idx, top_k):
            def forward(hidden_states):
                B, L, D = hidden_states.shape
                x = hidden_states.reshape(-1, D)
                gate_out = gate(x)
                if isinstance(gate_out, tuple) and len(gate_out) == 3:
                    _, topk_w, topk_idx = gate_out
                    topk_w = topk_w.to(hidden_states.dtype)
                else:
                    scores = F.softmax(gate_out, dim=-1, dtype=torch.float32)
                    topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
                    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

                needed = topk_idx.unique().tolist()
                ep = f"{expert_dir}/layer_{layer_idx:02d}.safetensors"
                with safe_open(ep, framework="pt", device="cpu") as ef:
                    expert_w = {}
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        fw = torch.stack([ef.get_tensor(f"{proj}.weight")[e] for e in needed])
                        fs = torch.stack([ef.get_tensor(f"{proj}.scales")[e] for e in needed])
                        fb = torch.stack([ef.get_tensor(f"{proj}.biases")[e] for e in needed])
                        expert_w[proj] = dequantize_mlx_4bit(fw, fs, fb, GROUP_SIZE)

                output = torch.zeros_like(x)
                for local_idx, eid in enumerate(needed):
                    mask = (topk_idx == eid)
                    token_mask = mask.any(dim=-1)
                    tidx = token_mask.nonzero(as_tuple=True)[0]
                    if len(tidx) == 0:
                        continue
                    w = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1)
                    inp = x[tidx]
                    g = F.silu(inp @ expert_w["gate_proj"][local_idx].t())
                    u = inp @ expert_w["up_proj"][local_idx].t()
                    out = (g * u) @ expert_w["down_proj"][local_idx].t()
                    output[tidx] += w[tidx].unsqueeze(-1) * out

                if shared_expert is not None:
                    s_out = shared_expert(x)
                    if shared_expert_gate is not None:
                        s_out = s_out * torch.sigmoid(shared_expert_gate(x))
                    output = output + s_out
                del expert_w
                return output.reshape(B, L, D)
            return forward

        moe.forward = make_forward(gate, shared_expert, shared_expert_gate, i, top_k)

layer_stats = {}
def make_hook(idx):
    def hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        layer_stats[idx] = {
            'mean': x.float().mean().item(),
            'std': x.float().std().item(),
            'absmax': x.float().abs().max().item(),
            'has_nan': x.isnan().any().item(),
            'has_inf': x.isinf().any().item(),
        }
    return hook

hooks = []
for i in range(8):
    hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

tokenizer = AutoTokenizer.from_pretrained(original_dir, trust_remote_code=True)
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")
print(f"  Input: {input_ids.shape[1]} tokens")

with torch.no_grad():
    try:
        out = model(input_ids)
        logits = out.logits if hasattr(out, 'logits') else out[0]
        top5 = torch.topk(logits[0, -1], 5)
        print(f"  Top 5 predictions: {[tokenizer.decode([t]) for t in top5.indices]}")
        print(f"  Top 5 logits: {top5.values.tolist()}")
    except Exception as e:
        print(f"  Forward failed: {e}")
        import traceback; traceback.print_exc()

print(f"\n  {'Layer':>5} {'Mean':>12} {'Std':>12} {'AbsMax':>12} {'NaN':>5} {'Inf':>5}")
for i in sorted(layer_stats.keys()):
    s = layer_stats[i]
    flag = ""
    if s['has_nan']: flag += " NaN!"
    if s['has_inf']: flag += " Inf!"
    if s['std'] < 1e-6: flag += " DEAD!"
    if s['absmax'] > 1000: flag += " EXPLODING!"
    print(f"  {i:>5} {s['mean']:>12.6f} {s['std']:>12.6f} {s['absmax']:>12.4f} {str(s['has_nan']):>5} {str(s['has_inf']):>5}{flag}")

for h in hooks:
    h.remove()

print("\n" + "=" * 60)
print("  DIAGNOSIS COMPLETE")
print("=" * 60)
