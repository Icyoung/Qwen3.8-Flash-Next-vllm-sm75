#!/usr/bin/env python3
"""Attach dedicated PLE files to a freshly quantized checkpoint index.

Preserves an index backup. Relative symlinks allow the model/PLE directory tree
to move between NAS and local SSD. No ordinary weights are quantized here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from q38ssd.store import PLE_NAME, read_manifest


def attach(model, manifest_path):
    model = Path(model).resolve()
    manifest_path, manifest = read_manifest(manifest_path)
    path = model / "model.safetensors.index.json"
    backup = model / "model.safetensors.index.before-ssd.json"
    if backup.exists():
        raise ValueError("Index backup already exists; do not overwrite an attached checkpoint")
    original = path.read_bytes()
    index = json.loads(original)
    mapping = index["weight_map"]
    # Allow export recipes that exclude PLE. If they retain PLE, require the
    # original complete name set so we cannot silently change model geometry.
    old = {n for n in mapping if PLE_NAME.fullmatch(n)}
    if old and old != set(manifest["tensors"]):
        raise ValueError("Quantized checkpoint PLE names differ from SSD table")
    links = {}
    for record in manifest["tensors"].values():
        target = manifest_path.parent / record["file"]
        link = model / record["file"]
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        links[link] = target
    new_mapping = {n: f for n, f in mapping.items() if n not in old}
    new_mapping.update({n: t["file"] for n, t in manifest["tensors"].items()})
    metadata = index.setdefault("metadata", {})
    # Avoid keeping the old total_size, which may describe a different PLE dtype.
    metadata.pop("total_size", None)
    metadata["q38_ple_ssd"] = "BF16"
    index["weight_map"] = new_mapping
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    created = []
    try:
        for link, target in links.items():
            link.symlink_to(os.path.relpath(target, model))
            created.append(link)
        with backup.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        with partial.open("x") as stream:
            stream.write(json.dumps(index, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
    except BaseException:
        for link in created:
            link.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        # Backup is intentionally retained for recovery if it was created.
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    attach(args.model, args.manifest)
    print("Attached dedicated BF16 PLE shards; original index backed up")
