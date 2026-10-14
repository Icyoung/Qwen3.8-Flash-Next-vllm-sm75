#!/usr/bin/env python3
"""Extract original BF16 PLE to dedicated safetensors using a bounded buffer.

This preserves source bytes. It does not quantize ordinary model weights.
The output directory must be new so a failed run cannot overwrite a model.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from q38ssd.store import PLE_NAME, read_header


def prepare(source, output, chunk_bytes=8 * 1024 * 1024, expected_shards=128):
    if not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("Copy buffer must be between 1 byte and 64 MiB")
    source, output = Path(source).resolve(), Path(output).resolve()
    index = json.loads((source / "model.safetensors.index.json").read_text())
    entries = {n: f for n, f in index["weight_map"].items() if PLE_NAME.fullmatch(n)}
    ids = sorted(int(PLE_NAME.fullmatch(n).group(2)) for n in entries)
    prefixes = {PLE_NAME.fullmatch(n).group(1) for n in entries}
    if ids != list(range(expected_shards)) or len(prefixes) != 1:
        raise ValueError(f"Expected one PLE table with {expected_shards} consecutive shards")
    metadata = {}
    for filename in set(entries.values()):
        if Path(filename).name != filename or (source / filename).resolve().parent != source:
            raise ValueError("Source shard must be inside source directory")
        metadata[filename] = read_header(source / filename)
    for name, filename in entries.items():
        tensor = metadata[filename][0][name]
        if tensor["dtype"] != "BF16" or len(tensor["shape"]) != 2:
            raise ValueError(f"Source PLE must be original BF16: {name}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "dtype": "BF16", "source": str(source), "tensors": {}}
    weight_map = {}
    total = 0
    for name in sorted(entries, key=lambda n: int(PLE_NAME.fullmatch(n).group(2))):
        shard = int(PLE_NAME.fullmatch(name).group(2))
        filename = entries[name]
        tensor = metadata[filename][0][name]
        start, end = tensor["data_offsets"]
        size = end - start
        header = json.dumps({name: {"dtype": "BF16", "shape": tensor["shape"], "data_offsets": [0, size]}}, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        prefix = struct.pack("<Q", len(header)) + header
        target_name = f"q38-ple-{shard:05d}.safetensors"
        target = output / target_name
        partial = target.with_suffix(target.suffix + ".partial")
        digest = hashlib.sha256()
        with (source / filename).open("rb") as original, partial.open("xb") as dest:
            original.seek(metadata[filename][1] + start)
            dest.write(prefix)
            remaining = size
            while remaining:
                block = original.read(min(chunk_bytes, remaining))
                if not block:
                    raise OSError(f"Source truncated: {filename}")
                digest.update(block)
                dest.write(block)
                remaining -= len(block)
            dest.flush()
            os.fsync(dest.fileno())
        partial.replace(target)
        manifest["tensors"][name] = {"file": target_name, "shape": tensor["shape"], "offset": len(prefix), "header_sha256": hashlib.sha256(prefix).hexdigest(), "data_sha256": digest.hexdigest()}
        weight_map[name] = target_name
        total += size
    manifest["data_bytes"] = total
    (output / "ple-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "ple.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(f"Extracted {len(manifest['tensors'])} BF16 shards, {manifest['data_bytes'] / 1024**3:.3f} GiB; copy buffer 8 MiB")
