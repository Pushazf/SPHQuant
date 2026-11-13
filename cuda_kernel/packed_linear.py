"""Torch side of the fused SPHQuant W2 kernels (sph_gemv.so, loaded with ctypes).

The export (sphquant-w2-v1: packed_codes, radius_extra, radius_scales, radius_mins, codebook) is repacked at load
time into the runtime layout the kernels read (see sph_gemv.cu; same 2.5 bits per weight). PackedSPHLinear holds
that layout; fuse() stacks several of them (q, k, v or gate, up) into one module evaluated by one kernel launch.
"""
import ctypes
import os

import torch
from torch import nn

FIELDS = ("packed_codes", "radius_extra", "radius_scales", "radius_mins", "codebook")  # sphquant-w2-v1 export
RUN_FIELDS = ("planes_a", "planes_b", "planes_c", "codebooks")  # runtime layout
GROUP = 128  # inputs per group (16 vectors of 8)
MAX_SEGMENTS = 4
GEMV_MAX_ROWS = 8  # inputs of more rows (prompts) are multiplied by the dequantized FP16 weight instead
BLOCKS_PER_SM = 2  # GEMV work split: rows spread over about this many blocks per SM, one wave
# GEMV launches allow programmatic dependent launch (compute capability 9.0 and up): a block loads its first weight
# rows and the codebooks while the previous kernel finishes, and waits for it only before reading x
PDL = os.environ.get("SPH_PDL", "1") != "0"
# a fused gate/up module computes silu(gate) * up in its GEMV kernel (PackedSPHLinear.forward_glu)
GLU = os.environ.get("SPH_GLU", "1") != "0"
RETURN_CODES = {1: "bad shape or argument", 2: "null or misaligned pointer", 3: "launch error"}

_lib = None
_sm_count = {}


def cudart_paths(maps_text, soname):
    """Distinct mapped file paths whose name starts with SONAME, from /proc/<pid>/maps text."""
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and os.path.basename(fields[5]).startswith(soname):
            paths.add(fields[5])
    return paths


def load_library(path):
    """Load sph_gemv.so; torch's CUDA runtime must be the only libcudart of its major version in the process."""
    global _lib
    torch.cuda.init()
    lib = ctypes.CDLL(os.path.abspath(path))
    soname = f"libcudart.so.{torch.version.cuda.split('.')[0]}"
    with open("/proc/self/maps") as f:
        paths = cudart_paths(f.read(), soname)
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {soname} in the process, found {sorted(paths)}")
    ptr, i64, i32 = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
    ends = ctypes.POINTER(ctypes.c_int)
    lib.sph_gemv.argtypes = [ptr, i64, i64, ptr, ptr, ptr, ptr, i32, ends, i64, i32, i32, i32, ptr, ptr]
    lib.sph_gemv.restype = ctypes.c_int
    lib.sph_dequant.argtypes = [ptr, ptr, ptr, ptr, i32, ends, i64, i64, ptr, ptr]
    lib.sph_dequant.restype = ctypes.c_int
    # split-KV decode attention (cuda_kernel/decode_attention.py)
    lib.sph_decode_attention.argtypes = [i32, ptr, i64, i64, ptr, i64, i64, i64, ptr, i64, i64, i64, ptr, i64,
                                         i32, i32, i32, i32, i32, i32, i32, ctypes.c_float, ptr, ptr, ptr, ptr]
    lib.sph_decode_attention.restype = ctypes.c_int
    _lib = lib
    return lib


def _as_int32(x):
    """Unsigned 32-bit values in int64 x -> int32 with the same bits."""
    return (x - ((x >> 31) << 32)).to(torch.int32)


def repack(packed):
    """sphquant-w2-v1 fields of one linear layer -> runtime layout (int32 planes, codebooks [1, 64, 8] fp16)."""
    word = packed["packed_codes"].to(torch.int64) & 0xFFFF
    extra = packed["radius_extra"].to(torch.int64) & 0xFFFFFFFF
    n, kv = word.shape
    if kv % 16:
        raise ValueError(f"in_features {kv * 8} is not a multiple of {GROUP}")
    word = word.view(n, kv // 16, 16)
    j = torch.arange(16, device=word.device)
    signs, dirs = word & 0xFF, (word >> 8) & 63
    codes = ((word >> 14) & 3) | (((extra.unsqueeze(-1) >> (2 * j)) & 3) << 2)
    planes_a = [((((signs >> (2 * k)) & 1) << j) | (((signs >> (2 * k + 1)) & 1) << (16 + j))).sum(-1)
                for k in range(4)]
    d = [sum(dirs[..., 5 * r + i] << (6 * i) for i in range(5)) | (((dirs[..., 15] >> (2 * r)) & 3) << 30)
         for r in range(3)]
    q = [sum((codes[..., 8 * h + 2 * p] << (4 * p)) | (codes[..., 8 * h + 2 * p + 1] << (16 + 4 * p))
             for p in range(4)) for h in range(2)]
    bits = [packed[f].to(torch.float16).view(torch.int16).to(torch.int64) & 0xFFFF
            for f in ("radius_scales", "radius_mins")]
    return {
        "planes_a": _as_int32(torch.stack(planes_a, -1)),
        "planes_b": _as_int32(torch.stack(d + q[:1], -1)),
        "planes_c": _as_int32(torch.stack([q[1], bits[0] | (bits[1] << 16)], -1)),
        "codebooks": packed["codebook"].to(torch.float16).reshape(1, 64, 8).contiguous(),
    }


def decode_run(run, ends, rows=None, dtype=torch.float32):
    """Weight [N, K] (or rows ROWS, a range) decoded from the runtime layout in PyTorch, as a reference."""
    rows = rows or range(ends[-1])
    a, b, c = (run[f][rows.start:rows.stop].to(torch.int64) & 0xFFFFFFFF
               for f in ("planes_a", "planes_b", "planes_c"))
    n, g, _ = a.shape
    j = torch.arange(16, device=a.device)
    sign = torch.stack([(a[..., i // 2, None] >> (16 * (i % 2) + j)) & 1 for i in range(8)], -1)
    d15 = (b[..., 0] >> 30) | ((b[..., 1] >> 30) << 2) | ((b[..., 2] >> 30) << 4)
    dirs = torch.stack([(b[..., v // 5] >> (6 * (v % 5))) & 63 for v in range(15)] + [d15], -1)
    q = torch.stack([b[..., 3], c[..., 0]], -1)
    codes = torch.stack([(q[..., v // 8] >> (4 * ((v % 8) // 2) + 16 * (v % 2))) & 15 for v in range(16)], -1)
    half = lambda t: (t - ((t >> 15) << 16)).to(torch.int16).view(torch.float16).float()  # noqa: E731  16-bit
    step, rmin = half(c[..., 1] & 0xFFFF), half(c[..., 1] >> 16)
    radius = rmin[..., None] + codes.float() * step[..., None]
    seg = torch.searchsorted(torch.tensor(ends, device=a.device),
                             torch.arange(rows.start, rows.stop, device=a.device), right=True)
    cb = run["codebooks"].float()[seg]  # [n, 64, 8]
    vec = torch.gather(cb, 1, dirs.view(n, -1, 1).expand(-1, -1, 8)).view(n, g, 16, 8)
    return (radius[..., None] * vec * (1 - 2 * sign)).reshape(n, g * GROUP).to(dtype)


def sm_count(device):
    i = torch.device(device).index or 0
    if i not in _sm_count:
        _sm_count[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _sm_count[i]


def rows_per_block(n, k, sms, glu=False):
    """Outputs per block of the GEMV kernel (N rows, or N / 2 outputs with GLU) spread over BLOCKS_PER_SM blocks per
    SM, at least one per warp."""
    slices = -(-k // (32 * GROUP))
    outputs = n // 2 if glu else n
    rows = max(max(1, 8 // slices), -(-outputs // (BLOCKS_PER_SM * sms)))
    # partial sums of rows * slices * (2 with GLU) floats: at most 16 KB of shared memory
    return min(rows, 4096 // slices // (2 if glu else 1)) if slices > 1 else rows


def _check(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} returned {rc} ({RETURN_CODES.get(rc, 'unknown')})")


def _ends_array(ends):
    return (ctypes.c_int * len(ends))(*ends)


@torch.library.custom_op("sphquant::linear", mutates_args=(), device_types="cuda")
def sph_linear(x: torch.Tensor, planes_a: torch.Tensor, planes_b: torch.Tensor, planes_c: torch.Tensor,
               codebooks: torch.Tensor, ends: list[int], glu: bool = False) -> torch.Tensor:
    """y fp16 [M, N] = x fp16 [M, K] @ W^T, or with GLU [M, N / 2] = silu(first half) * second half; an opaque
    operator, so torch.compile and CUDA graphs keep the kernels."""
    if _lib is None:
        raise RuntimeError("call load_library() first")
    m, k = x.shape
    n = ends[-1]
    stream = torch.cuda.current_stream(x.device).cuda_stream
    bufs = (planes_a.data_ptr(), planes_b.data_ptr(), planes_c.data_ptr(), codebooks.data_ptr())
    if m <= GEMV_MAX_ROWS:
        y = torch.empty(m, n // 2 if glu else n, dtype=torch.float16, device=x.device)
        rows = rows_per_block(n, k, sm_count(x.device), glu)
        _check(_lib.sph_gemv(x.data_ptr(), m, k, *bufs, len(ends), _ends_array(ends), n, rows, int(glu), int(PDL),
                             y.data_ptr(), stream), "sph_gemv")
        return y
    w = torch.empty(n, k, dtype=torch.float16, device=x.device)
    _check(_lib.sph_dequant(*bufs, len(ends), _ends_array(ends), k, n, w.data_ptr(), stream), "sph_dequant")
    y = x @ w.T
    if glu:
        gate, up = y.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    return y


@sph_linear.register_fake
def _sph_linear_fake(x, planes_a, planes_b, planes_c, codebooks, ends, glu=False):
    return x.new_empty((x.shape[0], ends[-1] // 2 if glu else ends[-1]))


class PackedSPHLinear(nn.Module):
    """Linear layer evaluated by the fused kernels from the runtime layout (input cast to FP16 and back)."""

    def __init__(self, run, ends, bias=None):
        super().__init__()
        n, g, _ = run["planes_a"].shape
        if len(ends) > MAX_SEGMENTS or ends[-1] != n or run["codebooks"].shape != (len(ends), 64, 8):
            raise ValueError(f"inconsistent runtime layout: {n} rows, ends {ends}, codebooks "
                             f"{tuple(run['codebooks'].shape)}")
        for f in RUN_FIELDS:
            self.register_buffer(f, run[f].contiguous())
        self.ends = [int(e) for e in ends]
        self.out_features, self.in_features = n, g * GROUP
        self.register_parameter("bias", bias)

    @classmethod
    def from_export(cls, packed, bias=None):
        return cls(repack(packed), [packed["packed_codes"].shape[0]], bias)

    def run(self):
        return {f: getattr(self, f) for f in RUN_FIELDS}

    def forward(self, x):
        x2 = x.reshape(-1, self.in_features).to(torch.float16).contiguous()
        y = sph_linear(x2, *(getattr(self, f) for f in RUN_FIELDS), self.ends)
        y = y.to(x.dtype).reshape(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias

    def forward_glu(self, x):
        """silu(gate) * up for a fused gate/up module (first half of the rows the gate), in one kernel."""
        if self.bias is not None or self.out_features % 2:
            raise ValueError("forward_glu needs an even number of rows and no bias")
        x2 = x.reshape(-1, self.in_features).to(torch.float16).contiguous()
        y = sph_linear(x2, *(getattr(self, f) for f in RUN_FIELDS), self.ends, True)
        return y.to(x.dtype).reshape(*x.shape[:-1], self.out_features // 2)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, segments={self.ends}"


def fuse(modules):
    """One PackedSPHLinear computing the concatenated outputs of MODULES (same input), each with its codebook."""
    if any(m.in_features != modules[0].in_features for m in modules):
        raise ValueError("fused modules must share in_features")
    ends, total = [], 0
    for m in modules:
        if len(m.ends) != 1:
            raise ValueError("fuse() takes unfused modules")
        total += m.out_features
        ends.append(total)
    run = {f: torch.cat([getattr(m, f) for m in modules]) for f in RUN_FIELDS}
    biases = [m.bias for m in modules]
    bias = None if all(b is None for b in biases) else nn.Parameter(torch.cat(
        [b if b is not None else torch.zeros(m.out_features, dtype=modules[0].codebooks.dtype,
                                             device=modules[0].codebooks.device) for m, b in zip(modules, biases)]))
    return PackedSPHLinear(run, ends, bias)
