#!/usr/bin/env python3
"""Prepare reproducible, recorded 128x512 tokens before model calibration."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCES = [
    ("NeelNanda/pile-10k", None, "train", 64),
    ("madao33/new-title-chinese", None, "train", 32),
    ("google-research-datasets/mbpp", "full", "train", 16),
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", 16),
]


def row_text(row):
    if isinstance(row.get("messages"), list):
        return "\n".join(f"{x.get('role', '')}: {x.get('content', '')}" for x in row["messages"])
    return "\n".join(str(row[k]) for k in ["title", "content", "text", "code", "test_list"] if k in row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from huggingface_hub import HfApi
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    api = HfApi()
    samples, sources = [], []
    for repo, config, split, count in SOURCES:
        revision = api.dataset_info(repo).sha
        dataset = load_dataset(repo, config, split=split, revision=revision, streaming=True).shuffle(seed=42, buffer_size=128)
        buffer, provenance, emitted = [], [], []
        for ordinal, row in enumerate(dataset):
            text = row_text(row)
            if not text.strip():
                continue
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not tokens:
                continue
            tokens.append(tokenizer.eos_token_id)
            buffer.extend(tokens)
            provenance.append({"stream_ordinal": ordinal, "row_id": str(row.get("id", row.get("prompt_id", ""))), "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
            while len(buffer) >= 512 and len(emitted) < count:
                selected, buffer = buffer[:512], buffer[512:]
                samples.append({"input_ids": torch.tensor(selected, dtype=torch.long), "attention_mask": torch.ones(512, dtype=torch.long)})
                emitted.append({"sample_index": len(samples) - 1, "input_ids_sha256": hashlib.sha256(samples[-1]["input_ids"].numpy().tobytes()).hexdigest()})
            if len(emitted) == count:
                break
            if ordinal >= 100000:
                raise ValueError(f"Insufficient usable calibration text: {repo}")
        if len(emitted) != count:
            raise ValueError(f"Insufficient calibration samples: {repo}")
        sources.append({"repo": repo, "revision": revision, "config": config, "split": split, "seed": 42, "shuffle_buffer": 128, "samples": emitted, "consumed_rows": provenance})
        print(json.dumps({"repo": repo, "revision": revision, "samples": len(emitted)}), flush=True)
    if len(samples) != 128:
        raise ValueError("Calibration sample count differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(samples, partial)
    partial.replace(args.output)
    metadata = {"samples": 128, "sequence_length": 512, "seed": 42, "sources": sources, "tensor_file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "held_out": False, "expert_coverage_measured": False}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
