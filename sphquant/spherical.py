"""Spherical decomposition, angle-parameterized directions and group-wise radius fake quantization."""
import math

import torch

VEC = 8  # weights per vector
GROUP = 128  # weights per radius group
GROUP_VECS = GROUP // VEC  # vectors per radius group
RADIUS_BITS = 4
RADIUS_MAX_CODE = 2**RADIUS_BITS - 1
DIR_BITS = 6
CODEBOOK = 2**DIR_BITS
ANGLE_EPS = 1e-4
RADIUS_EPS = 1e-8


def decompose(w):
    """W [N, K] -> (signs_pm1 int8 [N, K/8, 8], mags [N, K/8, 8], radius [N, K/8]), FP32. Zero is positive."""
    n, k = w.shape
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    v = w.float().reshape(n, k // VEC, VEC)
    signs = (v < 0).to(torch.int8).mul(-2).add(1)
    mags = v.abs()
    return signs, mags, torch.linalg.vector_norm(mags, dim=-1)


def angles_to_unit(phi):
    """Angle logits [..., 7] -> nonnegative unit vectors [..., 8] (hyperspherical coordinates)."""
    theta = ANGLE_EPS + (math.pi / 2 - 2 * ANGLE_EPS) * torch.sigmoid(phi)
    sin_prod = torch.cumprod(torch.sin(theta), dim=-1)
    prefix = torch.cat([torch.ones_like(theta[..., :1]), sin_prod[..., :-1]], dim=-1)
    c = torch.cat([prefix * torch.cos(theta), sin_prod[..., -1:]], dim=-1)
    return c / torch.linalg.vector_norm(c, dim=-1, keepdim=True).clamp_min(1e-12)


def unit_to_logits(c):
    """Inverse of angles_to_unit for nonnegative vectors [..., 8] (scale invariant); computed in FP64."""
    c64 = c.double()
    suffix = torch.flip(torch.cumsum(torch.flip(c64.square(), [-1]), -1), [-1])
    tail = suffix[..., :-1].sqrt()
    ratio = (c64[..., :-1] / tail.clamp_min(1e-300)).clamp(0.0, 1.0)
    theta = torch.acos(ratio).clamp(ANGLE_EPS, math.pi / 2 - ANGLE_EPS)
    s = ((theta - ANGLE_EPS) / (math.pi / 2 - 2 * ANGLE_EPS)).clamp(1e-7, 1 - 1e-7)
    return torch.logit(s).to(c.dtype)


def softplus_inverse(x):
    """FP32 inverse of softplus, input clamped to >= 1e-8."""
    x = x.float().clamp_min(1e-8)
    return x + torch.log(-torch.expm1(-x))


def radius_grid(r):
    """Per-group (rmin, step) [N, G] from detached r [N, G, 16], rounded through FP16, returned as FP32."""
    r = r.detach()
    lo, hi = r.amin(-1), r.amax(-1)
    return lo.half().float(), ((hi - lo) / RADIUS_MAX_CODE).half().float()


def radius_codes(r, rmin, step):
    """Float radius codes in [0, 15]; code 0 wherever step == 0."""
    rmin, step = rmin.unsqueeze(-1), step.unsqueeze(-1)
    nonzero = step > 0
    safe = torch.where(nonzero, step, torch.ones_like(step))
    codes = torch.round((r.detach() - rmin) / safe).clamp(0, RADIUS_MAX_CODE)
    return torch.where(nonzero, codes, torch.zeros_like(codes))


def fake_quant_radius(r, grid="recompute", frozen_grid=None):
    """r [N, G, 16] -> rmin + codes * step with a straight-through (identity) gradient.

    grid="recompute" derives the grid from r; grid="frozen" uses frozen_grid = (rmin, step).
    """
    if grid == "recompute":
        rmin, step = radius_grid(r)
    elif grid == "frozen":
        rmin, step = frozen_grid
    else:
        raise ValueError(f"unknown grid mode {grid!r}")
    r_hat = rmin.unsqueeze(-1) + radius_codes(r, rmin, step) * step.unsqueeze(-1)
    return r_hat + (r - r.detach())  # value is exactly r_hat
