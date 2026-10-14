#!/usr/bin/env python3
"""Build a separate no-MTP view; keep the original AutoRound export intact.

Only reviewed MTP metadata differences may be corrected. Weight files are
read-only symlinks, to be dereferenced when archiving. Audit this view separately.
"""
import argparse
import copy
import json
from pathlib import Path


def prepare(source, output, audit):
    report = json.loads(audit.read_text())
    allowed = {"Index total_size differs from payload", "Standalone quantization config differs"}
    if set(report["errors"]) - allowed or not report.get("files"):
        raise ValueError("Run the full CPU export audit first; unreviewed errors remain")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mtp = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    main = {k: v for k, v in index["weight_map"].items() if not k.startswith("mtp.")}
    if not mtp or set(mtp.values()) & set(main.values()):
        raise ValueError("Expected dedicated MTP-only files; mixed shards need repacking")
    q = json.loads((source / "quantization_config.json").read_text())
    c = json.loads((source / "config.json").read_text())
    normalized = copy.deepcopy(c["quantization_config"])
    normalized["block_name_to_quantize"] = q["block_name_to_quantize"]
    normalized["extra_config"] = {k: v for k, v in normalized["extra_config"].items() if not k.startswith("mtp.")}
    if normalized != q or any(k.startswith("mtp.") for k in q["extra_config"]):
        raise ValueError("Quantization configs differ beyond reviewed MTP metadata")
    c["quantization_config"] = q
    index["weight_map"] = main
    index["metadata"]["total_size"] = report["payload_bytes"] - report["mtp_payload_bytes"]
    index["metadata"]["total_shards"] = len(set(main.values()))
    # AutoRound's parameter count describes packed storage and included MTP;
    # omit it instead of presenting it as the original parameter count.
    index["metadata"].pop("total_parameters", None)
    output.mkdir(exist_ok=False)
    changed = {"config.json": c, "quantization_config.json": q, "model.safetensors.index.json": index}
    for path in source.iterdir():
        if path.is_file() and path.name not in set(mtp.values()) | set(changed):
            (output / path.name).symlink_to(path.resolve())
    for name, value in changed.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.audit.resolve())
