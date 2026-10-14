"""Exact-name recipe for AutoRound's unstacked routed experts.

Small output dimensions that packing cannot represent remain unquantized.
Control modules always remain in source precision.
"""
import re


def linear_scheme(name, in_features, out_features):
    if any(word in name for word in ["hyper_connection", "norm", "conv", "embed_tokens", "ngram_embedding"]) or name.startswith("mtp.") or ".visual." in name or name.startswith("model.visual."):
        return {"bits": 16, "act_bits": 16}
    if name.endswith(("mlp.gate", "mlp.shared_expert_gate")):
        return {"bits": 16, "act_bits": 16}
    if in_features % 32 or out_features % 32:
        return {"bits": 16, "act_bits": 16}
    match = re.search(r"\.layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate|up|down)_proj$", name)
    if match:
        bits = 4 if int(match.group(1)) in (0, 1, 46, 47) else 3
        return {"bits": bits, "group_size": 128, "sym": True, "act_bits": 16}
    return {"bits": 4, "group_size": 32, "sym": True, "act_bits": 16}


def build_recipe(model):
    import torch
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            result[name] = linear_scheme(name, module.in_features, module.out_features)
        if name.endswith(".mlp.experts") and hasattr(module, "gate_up_proj"):
            shape = tuple(module.gate_up_proj.shape)
            if len(shape) != 3 or shape[1:] != (1280, 2560):
                raise ValueError(f"Unexpected fused expert layout: {name}: {shape}")
            if tuple(module.down_proj.shape) != (shape[0], 2560, 640):
                raise ValueError(f"Unexpected down expert layout: {name}")
            module.is_transposed = False
            for expert in range(shape[0]):
                for projection, n, k in [("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)]:
                    key = f"{name}.{expert}.{projection}_proj"
                    result[key] = linear_scheme(key, k, n)
    if not any(c.get("bits") == 3 for c in result.values()):
        raise ValueError("Recipe did not find middle-layer W3 experts")
    return result
