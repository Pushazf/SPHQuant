"""Torch side of the split-KV decode attention kernel (decode_attention.cu, built into the same sph_gemv.so).

decode_attention(q, k, v, mask, scale) -> [B, 1, Hq, D] for one query token per sequence: q [B, Hq, 1, D], k and v
[B, Hkv, L, D] (a static KV cache, last dim contiguous), mask [B, 1, 1, L] bool (True: attend) or None. Call
prepare(device) once before CUDA graph capture: it allocates the zeroed counters the kernel's blocks share.
"""
from typing import Optional

import torch

from cuda_kernel import packed_linear as pl

HEAD_DIMS = (64, 128, 256)
GROUPS = (1, 2, 4, 8)  # query heads per KV head
DTYPES = {torch.float16: 0, torch.bfloat16: 1}
TILE = 64  # keys per tile; a block covers a multiple of it
BLOCKS_PER_SM = 4  # the keys are split until about this many blocks per SM
MAX_SPLITS = 64  # blocks per (sequence, KV head), the kernel's limit
MAX_COUNTERS = 4096  # batch * KV heads

_counters = {}


def _index(device):
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def prepare(device):
    """Zeroed counters for DEVICE (the kernel leaves them zero); allocate before capturing CUDA graphs."""
    i = _index(device)
    if i not in _counters:
        _counters[i] = torch.zeros(MAX_COUNTERS, dtype=torch.int32, device=f"cuda:{i}")
    return _counters[i]


def supported(q, k, v, mask):
    """Whether the kernel takes these inputs (one query token, FP16/BF16, a head dim and group size it knows)."""
    if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape or q.shape[2] != 1 or not q.is_cuda:
        return False
    b, hq, _, d = q.shape
    hkv = k.shape[1]
    if k.shape[0] != b or k.shape[3] != d or q.dtype not in DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if d not in HEAD_DIMS or hq % hkv or hq // hkv not in GROUPS or b * hkv > MAX_COUNTERS:
        return False
    if q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        return False
    return mask is None or (mask.dtype == torch.bool and mask.dim() == 4 and mask.shape[1:3] == (1, 1)
                            and mask.shape[3] == k.shape[2] and mask.shape[0] in (1, b))


def split_plan(length, batch, kv_heads, sms):
    """(keys per block, blocks per KV head): about BLOCKS_PER_SM blocks per SM, at most MAX_SPLITS per KV head,
    whole tiles of 64 keys."""
    target = min(MAX_SPLITS, max(1, -(-BLOCKS_PER_SM * sms // (batch * kv_heads))))
    chunk = -(-max(TILE, -(-length // target)) // TILE) * TILE
    return chunk, -(-length // chunk)


@torch.library.custom_op("sphquant::decode_attention", mutates_args=(), device_types="cuda")
def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor],
                     scale: float) -> torch.Tensor:
    """softmax(scale * q k^T over the keys MASK allows) v -> [B, 1, Hq, D]; see the module docstring."""
    if pl._lib is None:
        raise RuntimeError("call cuda_kernel.packed_linear.load_library() first")
    if not supported(q, k, v, mask):
        raise ValueError(f"decode_attention: unsupported inputs q {tuple(q.shape)} {q.dtype}, k {tuple(k.shape)}")
    counters = _counters.get(_index(q.device))
    if counters is None:
        raise RuntimeError("call cuda_kernel.decode_attention.prepare(device) first")
    b, hq, _, d = q.shape
    hkv, length = k.shape[1], k.shape[2]
    chunk, splits = split_plan(length, b, hkv, pl.sm_count(q.device))
    out = torch.empty(b, 1, hq, d, dtype=q.dtype, device=q.device)
    work = torch.empty(b * hq * splits * (d + 2), dtype=torch.float32, device=q.device)
    mask_ptr, mask_stride = 0, 0
    if mask is not None:
        m = mask[:, 0, 0, :]
        if m.stride(1) != 1:
            m = m.contiguous()
        mask_ptr, mask_stride = m.data_ptr(), (m.stride(0) if m.shape[0] > 1 else 0)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    rc = pl._lib.sph_decode_attention(
        DTYPES[q.dtype], q.data_ptr(), q.stride(0), q.stride(1), k.data_ptr(), k.stride(0), k.stride(1), k.stride(2),
        v.data_ptr(), v.stride(0), v.stride(1), v.stride(2), mask_ptr, mask_stride, b, hq, hkv, length, d, chunk,
        splits, scale, work.data_ptr(), counters.data_ptr(), out.data_ptr(), stream)
    if rc != 0:
        raise RuntimeError(f"sph_decode_attention returned {rc} ({pl.RETURN_CODES.get(rc, 'unknown')})")
    return out


@decode_attention.register_fake
def _decode_attention_fake(q, k, v, mask, scale):
    return q.new_empty((q.shape[0], 1, q.shape[1], q.shape[3]))
