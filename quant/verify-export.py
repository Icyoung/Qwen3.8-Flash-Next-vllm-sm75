#!/usr/bin/env python3
"""CPU-only checkpoint audit: headers, exact recipe, preserved bytes and hashes.

This does not load a model, touch CUDA, or establish inference quality.
MTP is inventoried separately because the first deployment excludes it.
"""
import argparse
import collections
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path

WIDTH = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
         "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
         "F64": 8, "I64": 8, "U64": 8}


def header(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 100_000_000:
            raise ValueError(f"Oversized header: {path}")
        data = json.loads(f.read(size))
    return {k: v for k, v in data.items() if k != "__metadata__"}, size + 8


def digest_range(path, start=0, length=None):
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        while length is None or length:
            chunk = f.read(8 * 1024**2 if length is None else min(length, 8 * 1024**2))
            if not chunk:
                if length:
                    raise ValueError(f"Truncated tensor: {path}")
                break
            h.update(chunk)
            if length is not None:
                length -= len(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start = time.time()
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    index = json.loads((args.export / "model.safetensors.index.json").read_text())
    tensors, locations, headers, bases = {}, {}, {}, {}
    payload_bytes = 0
    for path in sorted(args.export.glob("*.safetensors")):
        h, base = header(path)
        headers[path.name], bases[path.name] = h, base
        offset = 0
        for name, t in sorted(h.items(), key=lambda item: item[1]["data_offsets"]):
            a, b = t["data_offsets"]
            check(a == offset and b >= a, f"Offsets: {path.name}/{name}")
            check(b - a == math.prod(t["shape"]) * WIDTH[t["dtype"]], f"Size: {name}")
            check(name not in tensors, f"Duplicate: {name}")
            tensors[name], locations[name] = t, path.name
            offset = b
        check(path.stat().st_size == base + offset, f"File size: {path.name}")
        payload_bytes += offset
    check(locations == index["weight_map"], "Index/header mapping differs")
    check(payload_bytes == index["metadata"]["total_size"], "Index total_size differs from payload")

    source_map = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
    source_headers = {}

    def source_tensor(name):
        filename = source_map[name]
        if filename not in source_headers:
            source_headers[filename] = header(args.source / filename)
        h, base = source_headers[filename]
        return h[name], args.source / filename, base

    recipe = json.loads(args.recipe.read_text())
    config = json.loads((args.export / "config.json").read_text())["quantization_config"]
    standalone = json.loads((args.export / "quantization_config.json").read_text())
    check(config == standalone, "Standalone quantization config differs")
    counts = collections.Counter()
    expected_packed = set()
    for name, rule in recipe.items():
        if rule["bits"] == 16:
            check(name + ".qweight" not in tensors, f"Control unexpectedly packed: {name}")
            continue
        bits, group = rule["bits"], rule["group_size"]
        counts[f"W{bits}/G{group}"] += 1
        override = config["extra_config"].get(name, {})
        check(override.get("bits", config["bits"]) == bits and
              override.get("group_size", config["group_size"]) == group,
              f"Quant metadata recipe mismatch: {name}")
        expert = re.search(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj$", name)
        if expert:
            n, k = (2560, 640) if expert[1] == "down" else (640, 2560)
        else:
            n, k = source_tensor(name + ".weight")[0]["shape"]
        expected = {"qweight": ([k // 32 * bits, n], "I32"),
                    "qzeros": ([math.ceil(k / group), n // 32 * bits], "I32"),
                    "scales": ([math.ceil(k / group), n], None)}
        for suffix, (shape, dtype) in expected.items():
            key = name + "." + suffix
            expected_packed.add(key)
            t = tensors.get(key)
            check(t is not None and t["shape"] == shape and
                  (t["dtype"] == dtype if dtype else t["dtype"] in ("BF16", "F16")),
                  f"Packed geometry/dtype: {key}")
    main_packed = {k for k in tensors if not k.startswith("mtp.") and k.endswith((".qweight", ".qzeros", ".scales"))}
    check(main_packed == expected_packed, "Main packed tensor coverage differs from recipe")
    check(not any("ngram_embedding" in k for k in tensors), "PLE table embedded in export")

    preserved = 0
    preserved_bytes = 0
    for name, t in tensors.items():
        if name.startswith("mtp.") or name in expected_packed:
            continue
        check(name in source_map, f"Preserved tensor absent from source: {name}")
        if name not in source_map:
            continue
        original, path, base = source_tensor(name)
        check(t["shape"] == original["shape"] and t["dtype"] == original["dtype"], f"Preserved shape/dtype: {name}")
        a, b = t["data_offsets"]
        x, y = original["data_offsets"]
        check(digest_range(args.export / locations[name], bases[locations[name]] + a, b - a) ==
              digest_range(path, base + x, y - x), f"Preserved bytes changed: {name}")
        preserved += 1
        preserved_bytes += b - a

    # Full main-model coverage, including original fused expert banks.
    for name in source_map:
        if name.startswith("mtp.") or "ngram_embedding" in name:
            continue
        if name in tensors or name.removesuffix(".weight") + ".qweight" in tensors:
            continue
        match = re.match(r"(.+\.mlp\.experts)\.(gate_up_proj|down_proj)$", name)
        if match:
            shape = source_tensor(name)[0]["shape"]
            projs = ("gate", "up") if match[2] == "gate_up_proj" else ("down",)
            check(all(f"{match[1]}.{i}.{p}_proj.qweight" in tensors for i in range(shape[0]) for p in projs), f"Missing fused expert bank: {name}")
        else:
            errors.append(f"Source tensor missing: {name}")

    hashes = []
    for path in sorted(args.export.iterdir()):
        if path.is_file():
            hashes.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest_range(path)})
            print(json.dumps({"hashed": path.name}), flush=True)
    mtp = {k: v for k, v in tensors.items() if k.startswith("mtp.")}
    result = {"schema": 1, "passed": not errors, "scope": "CPU export structure, packing recipe, source-preserved bytes and SHA256",
              "quality_validated": False, "sm75_inference_validated": False,
              "elapsed_seconds": time.time() - start, "errors": errors,
              "recipe_counts": dict(counts), "tensors": len(tensors), "payload_bytes": payload_bytes,
              "preserved_tensors_byte_identical": preserved, "preserved_bytes": preserved_bytes,
              "mtp_tensors": len(mtp), "mtp_payload_bytes": sum(t["data_offsets"][1] - t["data_offsets"][0] for t in mtp.values()),
              "files": hashes}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
