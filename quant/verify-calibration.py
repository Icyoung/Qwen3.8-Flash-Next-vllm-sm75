#!/usr/bin/env python3
"""Independently validate saved calibration tokens and every sample checksum."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-file", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    args = parser.parse_args()
    import torch
    meta = json.loads(args.tensor_file.with_suffix(".json").read_text())
    if hashlib.sha256(args.tensor_file.read_bytes()).hexdigest() != meta["tensor_file_sha256"]:
        raise ValueError("Calibration file checksum differs")
    config = json.loads(args.source_config.read_text())
    vocab = config["text_config"]["vocab_size"]
    tokens = torch.load(args.tensor_file, map_location="cpu", weights_only=True)
    if len(tokens) != 128 or meta["samples"] != 128 or meta["sequence_length"] != 512:
        raise ValueError("Incorrect calibration geometry")
    expected = {s["sample_index"]: s["input_ids_sha256"] for source in meta["sources"] for s in source["samples"]}
    if set(expected) != set(range(128)):
        raise ValueError("Incomplete recorded calibration provenance")
    for i, sample in enumerate(tokens):
        ids, mask = sample["input_ids"], sample["attention_mask"]
        if ids.dtype != torch.long or tuple(ids.shape) != (512,) or tuple(mask.shape) != (512,) or not bool(torch.all(mask == 1)) or int(ids.min()) < 0 or int(ids.max()) >= vocab:
            raise ValueError(f"Invalid calibration token sample: {i}")
        if hashlib.sha256(ids.numpy().tobytes()).hexdigest() != expected[i]:
            raise ValueError(f"Calibration sample checksum differs: {i}")
    result = {"status": "passed", "samples": 128, "sequence_length": 512, "tensor_file_sha256": meta["tensor_file_sha256"], "scope": "saved artifact geometry, token bounds and provenance hashes"}
    ready = args.tensor_file.parent / "calibration-ready.json"
    ready.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
