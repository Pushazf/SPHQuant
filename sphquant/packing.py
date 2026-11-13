"""Bit packing of an export form into the kernel layout and back (bit-exact)."""
import torch

from sphquant.sph_linear import decode_export
from sphquant.spherical import CODEBOOK, GROUP_VECS, RADIUS_MAX_CODE


def _as_signed(x, bits):
    """Reinterpret unsigned values in int64 x as two's complement of the given width."""
    return x - ((x >> (bits - 1)) << bits)


def pack(state):
    """Export form -> packed_codes int16 [N, K/8], radius_extra int32 [N, K/128], fp16 scales/mins/codebook."""
    signs, dir_idx, codes = (state[key].to(torch.int64) for key in ("signs", "dir_idx", "codes"))
    n, kv = codes.shape
    if kv % GROUP_VECS or int(dir_idx.max()) >= CODEBOOK or int(codes.max()) > RADIUS_MAX_CODE:
        raise ValueError("export state out of range for packing")
    word = signs | (dir_idx << 8) | ((codes & 3) << 14)
    shifts = 2 * torch.arange(GROUP_VECS, device=codes.device)
    extra = ((codes >> 2).view(n, -1, GROUP_VECS) << shifts).sum(-1)
    return {
        "packed_codes": _as_signed(word, 16).to(torch.int16),
        "radius_extra": _as_signed(extra, 32).to(torch.int32),
        "radius_scales": state["step"].to(torch.float16),
        "radius_mins": state["rmin"].to(torch.float16),
        "codebook": state["codebook"].to(torch.float16),
    }


def unpack(packed):
    """Packed dict -> export form."""
    word = packed["packed_codes"].to(torch.int64) & 0xFFFF
    extra = packed["radius_extra"].to(torch.int64) & 0xFFFFFFFF
    n, kv = word.shape
    shifts = 2 * torch.arange(GROUP_VECS, device=word.device)
    high = ((extra.unsqueeze(-1) >> shifts) & 3).view(n, kv)
    return {
        "signs": (word & 0xFF).to(torch.uint8),
        "dir_idx": ((word >> 8) & 0x3F).to(torch.uint8),
        "codes": (((word >> 14) & 3) | (high << 2)).to(torch.uint8),
        "rmin": packed["radius_mins"],
        "step": packed["radius_scales"],
        "codebook": packed["codebook"],
    }


def decode_packed(packed, dtype=torch.float32):
    """Decode a packed dict to W [N, K]."""
    return decode_export(unpack(packed), dtype)
