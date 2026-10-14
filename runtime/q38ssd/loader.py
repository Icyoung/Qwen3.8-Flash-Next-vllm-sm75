"""PLE header-only iterator; ordinary weights retain upstream EP filtering."""
import os
from pathlib import Path
from .store import read_header, read_manifest


def weights_iterator(base_iterator, hf_weights_files, use_tqdm_on_load, safetensors_load_strategy=None, local_expert_ids=None, **kwargs):
    if os.environ.get("Q38_PLE_SSD") != "1":
        yield from base_iterator(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, **kwargs)
        return
    if safetensors_load_strategy != "lazy":
        raise ValueError("SSD PLE requires --safetensors-load-strategy lazy")
    _, manifest = read_manifest(os.environ["Q38_PLE_MANIFEST"])
    expected = {}
    for name, record in manifest["tensors"].items():
        expected.setdefault(record["file"], set()).add(name)
    ordinary, ple = [], []
    for path in hf_weights_files:
        filename = Path(path).name
        if filename in expected:
            ple.append(path)
        elif filename.startswith("q38-ple-"):
            raise ValueError(f"Unrecognized dedicated PLE shard: {filename}")
        else:
            ordinary.append(path)
    if len(ple) != len(expected) or {Path(p).name for p in ple} != set(expected):
        raise ValueError("Model index must reference all dedicated SSD PLE shards")
    # Verify selected model files before allowing any GPU loading to start.
    for path in ple:
        header, _, digest = read_header(path)
        filename = Path(path).name
        if {n for n in header if n != "__metadata__"} != expected[filename]:
            raise ValueError("Selected PLE file differs from manifest")
        if any(digest != manifest["tensors"][n]["header_sha256"] for n in expected[filename]):
            raise ValueError("Selected PLE header differs from manifest")
    if ordinary:
        yield from base_iterator(ordinary, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, **kwargs)
    import torch
    for path in ple:
        header, _, _ = read_header(path)
        for name in expected[Path(path).name]:
            yield name, torch.empty(header[name]["shape"], dtype=torch.bfloat16, device="meta")
