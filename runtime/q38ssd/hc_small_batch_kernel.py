"""Eager HC fusion. Caller validates the exact SM75/fp16 HC=4 layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _up_gate(L, W, X, Y, H: tl.constexpr, K: tl.constexpr,
             ROWS: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(0)
    feature = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, BK)
    lora = tl.load(L + token * K + k, k < K, 0).to(tl.float32)
    acc = tl.zeros((ROWS,), tl.float32)
    for stream in tl.static_range(4):
        columns = stream * H + feature
        weight = tl.load(W + columns[:, None] * K + k[None, :],
                         (feature[:, None] < H) & (k[None, :] < K), 0).to(tl.float32)
        # F.linear originally materializes an fp16 gate. Do not remove that
        # rounding boundary when fusing its consumer.
        gate = tl.sum(weight * lora[None, :], axis=1).to(tl.float16).to(tl.float32)
        x = tl.load(X + token * (4 * H) + columns, feature < H, 0).to(tl.float32)
        acc += tl.sigmoid(gate) * x
    tl.store(Y + token * H + feature, acc / 4, feature < H)


def up_gate(lora, weight, xn):
    m = xn.shape[0]
    output = torch.empty((m, 2560), device=xn.device, dtype=xn.dtype)
    rows = 16 if m == 1 else 8
    _up_gate[(m, triton.cdiv(2560, rows))](
        lora, weight, xn, output, 2560, 320, rows, 512, num_warps=4)
    return output
