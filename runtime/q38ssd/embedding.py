"""Experimental eager-only PLE backend; GPU integration is not qualified yet."""
import os
import torch
from torch import nn
from .store import PLE_NAME, RowStore


class SSDPLEEmbedding(nn.Module):
    supports_prefetch = False
    is_disk_ple = True
    checkpoint_dtype = torch.bfloat16

    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        if cfg.parallel_config.tensor_parallel_size != 4 or cfg.parallel_config.pipeline_parallel_size != 1 or cfg.parallel_config.data_parallel_size != 1:
            raise ValueError("SSD PLE prototype requires TP4, PP1, DP1")
        if not cfg.model_config.enforce_eager:
            raise ValueError("SSD PLE prototype requires --enforce-eager")
        if os.environ.get("TP4_NEXT_RUNTIME") == "1":
            raise ValueError("Do not combine SSD-only backend with the expert hot-cache adapter")
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.reader = RowStore(os.environ["Q38_PLE_MANIFEST"], int(os.environ.get("Q38_PLE_CACHE_BYTES", str(32 * 1024 * 1024))))
        self.shards = sorted(self.reader.tensors, key=lambda n: int(PLE_NAME.fullmatch(n).group(2)))
        if len(self.shards) != 128 or [int(PLE_NAME.fullmatch(n).group(2)) for n in self.shards] != list(range(128)) or len({PLE_NAME.fullmatch(n).group(1) for n in self.shards}) != 1:
            raise ValueError("Expected one complete PLE table with 128 shards")
        self.shard_rows = (num_embeddings + 127) // 128
        for i, name in enumerate(self.shards):
            expected = max(0, min(self.shard_rows, num_embeddings - i * self.shard_rows))
            if self.reader.tensors[name]["shape"] != [expected, embedding_dim]:
                raise ValueError(f"PLE runtime/checkpoint geometry mismatch: {name}")
        # Empty placeholder satisfies the model's loaded-parameter accounting.
        # The complete table is never allocated in CPU or GPU memory.
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)

    def dequantize(self, embeddings, output_dtype):
        return embeddings.to(output_dtype)

    def forward(self, ids):
        if ids.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SSD PLE requires eager execution")
        rows = ids.detach().cpu().reshape(-1).tolist()
        if not rows:
            return torch.empty((*ids.shape, self.embedding_dim), device=ids.device, dtype=torch.bfloat16)
        # Request-size buffers scale with the prefill chunk, not the full table.
        raw = bytearray()
        for row in rows:
            if not 0 <= row < self.org_vocab_size:
                raise IndexError("PLE ID out of range")
            shard, within = divmod(row, self.shard_rows)
            raw.extend(self.reader.rows(self.shards[shard], [within]))
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(*ids.shape, self.embedding_dim).to(ids.device)
