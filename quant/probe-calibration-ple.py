#!/usr/bin/env python3
"""Compare Transformers' GPU PLE lookup with the BF16 SSD calibration backend.

Uses a small random table with the real n-gram implementation and 128 shards.
This validates lookup integration, not full-model loading or answer quality.
"""
import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "quant"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(str(args.source_config_dir), local_files_only=True).text_config
    config.ngram_vocab_size_base = 5
    reference = Qwen4ExpTextNGramEmbedding(config, 2560, layer_idx=1).to(dtype=torch.bfloat16)
    weight = reference.ngram_embedding.weight.detach().cpu()
    buffers = {name: value.detach().cpu().clone() for name, value in reference.named_buffers()}
    rows, width = weight.shape
    if rows % 128:
        raise ValueError("Probe geometry does not split into 128 shards")
    source = args.output / "source"
    source.mkdir()
    header, data = {}, bytearray()
    for shard in range(128):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        raw = weight[shard * (rows // 128):(shard + 1) * (rows // 128)].contiguous().view(torch.uint8).numpy().tobytes()
        start = len(data)
        data.extend(raw)
        header[name] = {"dtype": "BF16", "shape": [rows // 128, width], "data_offsets": [start, len(data)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "model.safetensors" for n in header}}))
    spec = importlib.util.spec_from_file_location("prepare", ROOT / "tools/prepare-ssd-ple.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(source, args.output / "ple")
    ids = torch.randint(2, 128, (2, 16), device="cuda")
    reference = reference.to("cuda")
    expected = reference(ids, None)
    from calibration_ple import install
    reader = install(args.output / "ple/ple-manifest.json")
    actual_module = Qwen4ExpTextNGramEmbedding(config, 2560, layer_idx=1)
    actual_module.load_state_dict(buffers, strict=True, assign=True)
    actual_module.to("cuda")
    actual = actual_module(ids, None)
    if not torch.equal(expected, actual):
        raise ValueError("SSD PLE lookup differs from the original GPU embedding")
    result = {"status": "passed", "scope": "small-table BF16 Transformers GPU/SSD lookup equality", "table_shape": list(weight.shape), "lookup_shape": list(actual.shape), "bitwise_equal": True, "full_model_qualified": False, "disk_bytes_read": reader.reader.disk_bytes}
    (args.output / "probe-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
