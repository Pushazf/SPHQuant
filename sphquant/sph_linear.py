"""SPHLinear: trainable fake-quantized linear layer (radius logits rho, codebook angle logits phi)."""
import torch
import torch.nn.functional as F
from torch import nn

from sphquant.kmeans import assign, spherical_kmeans
from sphquant.spherical import (
    CODEBOOK, GROUP_VECS, RADIUS_EPS, VEC, angles_to_unit, decompose, fake_quant_radius,
    radius_codes, radius_grid, softplus_inverse, unit_to_logits,
)


def segment_index(idx, k=CODEBOOK):
    """Stable sort permutation of idx [V] and segment boundaries [k + 1] of each value in the sorted order."""
    order = torch.sort(idx, stable=True).indices
    counts = torch.bincount(idx, minlength=k)
    return order, torch.cat([counts.new_zeros(1), counts.cumsum(0)])


class CodebookGather(torch.autograd.Function):
    """codebook[idx] whose backward sums row gradients by sorted segments (FP64 cumsum), no atomics."""

    @staticmethod
    def forward(ctx, codebook, idx, order, bounds):
        ctx.save_for_backward(order, bounds)
        return codebook.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        order, bounds = ctx.saved_tensors
        g = grad.index_select(0, order).double().T.contiguous()  # [8, V], scan along the last dim
        cs = torch.cat([g.new_zeros(g.shape[0], 1), g.cumsum(1)], 1)
        return (cs[:, bounds[1:]] - cs[:, bounds[:-1]]).T.to(grad.dtype), None, None, None


class SPHLinear(nn.Module):
    """W_hat[n, 8v:8v+8] = fq(softplus(rho[n, v])) * codebook(phi)[dir_idx[n, v]] * signs_pm1[n, v]."""

    def __init__(self, rho, phi, signs_pm1, dir_idx, bias=None, grid="recompute", frozen_grid=None):
        super().__init__()
        if grid not in ("recompute", "frozen") or (grid == "frozen") != (frozen_grid is not None):
            raise ValueError(f"bad grid mode {grid!r} / frozen_grid")
        self.out_features, kv = rho.shape
        self.in_features = kv * VEC
        self.grid = grid
        self.kmeans_stats = {}
        self.rho = nn.Parameter(rho.detach().float().clone())
        self.phi = nn.Parameter(phi.detach().float().clone())
        self.register_buffer("signs_pm1", signs_pm1.to(torch.int8))
        self.register_buffer("dir_idx", dir_idx.long())
        order, bounds = segment_index(self.dir_idx.view(-1))
        self.register_buffer("dir_order", order, persistent=False)
        self.register_buffer("dir_bounds", bounds, persistent=False)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        if frozen_grid is not None:
            self.register_buffer("grid_rmin", frozen_grid[0].float())
            self.register_buffer("grid_step", frozen_grid[1].float())

    @classmethod
    @torch.no_grad()
    def from_dense(cls, weight, bias, generator, kmeans_samples=65536, kmeans_iters=20, grid="recompute"):
        """Initialize from a dense weight [N, K]; generator is a CPU torch.Generator used by k-means."""
        if not torch.isfinite(weight).all():
            raise ValueError("non-finite weight")
        signs, mags, radius = decompose(weight.detach())
        n, kv = radius.shape
        nonzero = (radius > 0).unsqueeze(-1)
        dirs = torch.where(nonzero, mags / torch.where(nonzero, radius.unsqueeze(-1), 1.0), VEC**-0.5)
        flat = dirs.view(-1, VEC)
        centers, stats = spherical_kmeans(flat, radius.view(-1), CODEBOOK, kmeans_samples, kmeans_iters,
                                          generator=generator)
        dir_idx = assign(flat, centers).view(n, kv)
        r_tilde = (mags * centers[dir_idx]).sum(-1).clamp_min(RADIUS_EPS)
        rho = softplus_inverse(r_tilde)
        frozen = radius_grid(F.softplus(rho).view(n, -1, GROUP_VECS)) if grid == "frozen" else None
        module = cls(rho, unit_to_logits(centers), signs, dir_idx, bias, grid, frozen)
        module.kmeans_stats = stats
        return module

    def codebook(self):
        """Current codebook [64, 8], FP32 nonnegative unit rows."""
        return angles_to_unit(self.phi)

    def _frozen_grid(self):
        return (self.grid_rmin, self.grid_step) if self.grid == "frozen" else None

    def materialize(self):
        """Fake-quantized FP32 weight [N, K] with straight-through radius gradients."""
        n, kv = self.rho.shape
        r = F.softplus(self.rho).view(n, -1, GROUP_VECS)
        r_hat = fake_quant_radius(r, self.grid, self._frozen_grid()).view(n, kv, 1)
        d = CodebookGather.apply(self.codebook(), self.dir_idx.view(-1), self.dir_order, self.dir_bounds)
        return (r_hat * d.view(n, kv, VEC) * self.signs_pm1).view(n, kv * VEC)

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.materialize().to(x.dtype), bias)

    @torch.no_grad()
    def export_state(self):
        """Export form (CPU): signs/dir_idx/codes uint8 [N, K/8], rmin/step fp16 [N, K/128], codebook fp16 [64, 8]."""
        n, kv = self.rho.shape
        r = F.softplus(self.rho).view(n, -1, GROUP_VECS)
        rmin, step = radius_grid(r) if self.grid == "recompute" else self._frozen_grid()
        codes = radius_codes(r, rmin, step).view(n, kv)
        shifts = torch.arange(VEC, device=self.signs_pm1.device)
        signs = ((self.signs_pm1 < 0).to(torch.int32) << shifts).sum(-1)
        return {
            "signs": signs.to(torch.uint8).cpu(),
            "dir_idx": self.dir_idx.to(torch.uint8).cpu(),
            "codes": codes.to(torch.uint8).cpu(),
            "rmin": rmin.half().cpu(),
            "step": step.half().cpu(),
            "codebook": self.codebook().half().cpu(),
        }


def decode_export(state, dtype=torch.float32):
    """Reference decode of an export form to W [N, K] (FP32 arithmetic on the stored values, then dtype)."""
    signs, codes = state["signs"], state["codes"]
    n, kv = codes.shape
    rmin = state["rmin"].float().unsqueeze(-1)
    step = state["step"].float().unsqueeze(-1)
    r = rmin + codes.float().view(n, -1, GROUP_VECS) * step  # same operation order as materialize()
    bits = (signs.to(torch.int32).unsqueeze(-1) >> torch.arange(VEC, device=signs.device)) & 1
    w = r.view(n, kv, 1) * state["codebook"].float()[state["dir_idx"].long()] * (1 - 2 * bits)
    return w.view(n, kv * VEC).to(dtype)
