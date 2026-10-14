#!/usr/bin/env python3
"""Apply the SSD-only adapter to the exact pinned SM75 engine sources.

This patches two files, retaining upstream EP filtering for ordinary weights.
Unrecognized or already patched sources fail before any file is modified.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path

EXPECTED = {
    "model_executor/model_loader/weight_utils.py": "7443e55184777dcc99f621b0d97d9149d52c2ea8b4304c6621eb4902e1c8ba5b",
    "models/qwen4_exp/nvidia/ngram_embedding.py": "68d7afa77b4a4c2a7ce02a325c825dbffa75b05da1ca3bfd6dea780566c31aae",
}
MARKER = "# Q38_SSD_ONLY_OVERLAY_V1"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch anchor: {old[:80]}")
    return text.replace(old, new, 1)


def patch_sources(sources):
    for relative, data in sources.items():
        if hashlib.sha256(data).hexdigest() != EXPECTED[relative]:
            raise ValueError(f"Unreviewed upstream source: {relative}")
    weight = sources["model_executor/model_loader/weight_utils.py"].decode()
    weight = replace_once(weight, "def safetensors_weights_iterator(\n", "def _q38_base_safetensors_weights_iterator(\n")
    weight += "\n\n" + MARKER + '''
def safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load,
                                safetensors_load_strategy=None,
                                local_expert_ids=None, **kwargs):
    from q38ssd.loader import weights_iterator
    yield from weights_iterator(
        _q38_base_safetensors_weights_iterator, hf_weights_files,
        use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, **kwargs)
'''
    ngram = sources["models/qwen4_exp/nvidia/ngram_embedding.py"].decode()
    start = ngram.index("        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(")
    end = ngram.index("        if params_dtype is None:", start)
    original = ngram[start:end]
    ngram = replace_once(ngram, original, '''        if os.environ.get("Q38_PLE_SSD") == "1":
            embedding_quant_method = Qwen4ExpPLEUnquantizedEmbeddingMethod()
        else:
''' + "".join("    " + line for line in original.splitlines(keepends=True)))
    ngram = replace_once(ngram, "        self.ngram_embedding = embedding_cls(\n", '''        if os.environ.get("Q38_PLE_SSD") == "1":
            from q38ssd.embedding import SSDPLEEmbedding
            embedding_cls = SSDPLEEmbedding
        self.ngram_embedding = embedding_cls(
''')
    ngram = replace_once(ngram, '''                    if loaded_weight.dtype != torch.float8_e4m3fn:
                        raise ValueError("Disk PLE store requires FP8 checkpoint shards")
''', '''                    expected_dtype = getattr(embedding, "checkpoint_dtype", torch.float8_e4m3fn)
                    if loaded_weight.dtype != expected_dtype:
                        raise ValueError("Disk PLE checkpoint dtype mismatch")
''')
    ngram += "\n" + MARKER + "\n"
    result = {"model_executor/model_loader/weight_utils.py": weight.encode(), "models/qwen4_exp/nvidia/ngram_embedding.py": ngram.encode()}
    for relative, data in result.items():
        ast.parse(data.decode(), filename=relative)
    return result


def install(package):
    sources = {relative: (package / relative).read_bytes() for relative in EXPECTED}
    # Never trust a marker as evidence that the rest of a file is reviewed.
    if any(MARKER.encode() in data for data in sources.values()):
        raise ValueError("Overlay already present; rebuild from the pinned base to replace it")
    result = patch_sources(sources)
    for relative, data in result.items():
        (package / relative).write_bytes(data)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    package = args.package or Path(importlib.util.find_spec("vllm").origin).parent
    for name, data in install(package).items():
        print(name, hashlib.sha256(data).hexdigest())
