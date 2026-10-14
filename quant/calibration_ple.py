"""BF16 SSD PLE backend for Transformers calibration, without table allocation."""
from types import SimpleNamespace
import os
from q38ssd.store import PLE_NAME, RowStore


class CalibrationRows:
    # Intentionally not an nn.Module: not a quantization target or state_dict
    # entry. Transformers' original forward moves returned rows to its device.
    def __init__(self, manifest_path):
        import torch
        # Quantization runs on Z4's large-RAM host. This shared calibration
        # cache is separate from the 32 MiB/rank four-T10 inference cache.
        self.reader = RowStore(manifest_path, cache_bytes=int(os.environ.get("Q38_CALIB_PLE_CACHE_BYTES", str(4 * 1024**3))))
        self.shards = sorted(self.reader.tensors, key=lambda n: int(PLE_NAME.fullmatch(n).group(2)))
        if len(self.shards) != 128 or [int(PLE_NAME.fullmatch(n).group(2)) for n in self.shards] != list(range(128)):
            raise ValueError("Calibration PLE requires 128 consecutive shards")
        self.rows = sum(self.reader.tensors[n]["shape"][0] for n in self.shards)
        self.dim = self.reader.tensors[self.shards[0]]["shape"][1]
        self.shard_rows = (self.rows + 127) // 128
        for i, name in enumerate(self.shards):
            expected = min(self.shard_rows, self.rows - i * self.shard_rows)
            if self.reader.tensors[name]["shape"] != [expected, self.dim]:
                raise ValueError("Calibration PLE geometry mismatch")
        self.weight = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16, shape=(self.rows, self.dim))

    def __call__(self, ids):
        import torch
        rows = ids.detach().cpu().reshape(-1).tolist()
        raw = bytearray()
        for row in rows:
            if not 0 <= row < self.rows:
                raise IndexError("PLE ID out of range")
            shard, within = divmod(row, self.shard_rows)
            raw.extend(self.reader.rows(self.shards[shard], [within]))
        if not rows:
            return torch.empty((*ids.shape, self.dim), dtype=torch.bfloat16)
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(*ids.shape, self.dim)


def install(manifest_path):
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    cls = modeling.Qwen4ExpTextNGramEmbedding
    if getattr(cls, "_q38_ssd_calibration", False):
        raise ValueError("Calibration PLE already installed")
    original = cls.__init__
    CalibrationRows.__call__ = torch.compiler.disable(CalibrationRows.__call__)
    reader = CalibrationRows(manifest_path)

    def initialize(self, *args, **kwargs):
        # Also safe if a caller constructs outside HF's empty-weight context.
        with torch.device("meta"):
            original(self, *args, **kwargs)
        embedding = self.ngram_embedding
        if (embedding.num_embeddings, embedding.embedding_dim) != (reader.rows, reader.dim):
            raise ValueError("Transformers/source PLE table geometry differs")
        del self._modules["ngram_embedding"]
        object.__setattr__(self, "ngram_embedding", reader)

    cls.__init__ = initialize
    cls._q38_ssd_calibration = True
    return reader
