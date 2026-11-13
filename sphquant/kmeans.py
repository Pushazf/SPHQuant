"""Radius-weighted spherical k-means for the direction codebook.

Sampling and Lloyd iterations run on CPU in FP64 so the result depends only on the generator,
not on the device of the inputs. The generator must be a CPU torch.Generator.
"""
import torch

from sphquant.spherical import CODEBOOK


def sample_indices(radius, n, generator):
    """n indices drawn with replacement, probability proportional to radius**2 (uniform if all zero)."""
    w = radius.detach().double().cpu().square()
    if w.sum() == 0:
        w = torch.ones_like(w)
    cdf = w.cumsum(0)
    u = torch.rand(n, generator=generator, dtype=torch.float64) * cdf[-1]
    return torch.searchsorted(cdf, u, right=True).clamp_max(len(w) - 1)


def lloyd_step(points, centers):
    """One spherical Lloyd step; empty clusters keep their old center. Returns (centers, n_empty)."""
    a = (points @ centers.T).argmax(1)
    sums = torch.zeros_like(centers).index_add_(0, a, points)
    new = sums / torch.linalg.vector_norm(sums, dim=1, keepdim=True).clamp_min(1e-12)
    empty = torch.bincount(a, minlength=len(centers)) == 0
    return torch.where(empty.unsqueeze(1), centers, new), int(empty.sum())


def spherical_kmeans(dirs, radius, k=CODEBOOK, samples=65536, iters=20, *, generator):
    """Unit directions [V, 8] and radii [V] -> (centers [k, 8] FP32 on dirs.device, stats)."""
    idx = sample_indices(radius, min(samples, len(dirs)), generator)
    pts = dirs.detach()[idx.to(dirs.device)].double().cpu()
    uniq, inv = torch.unique(pts, dim=0, return_inverse=True)
    if len(uniq) < k:
        raise ValueError(f"only {len(uniq)} distinct sampled directions, need {k}")
    # first k distinct directions in a generator-chosen order of the samples
    order = torch.randperm(len(pts), generator=generator)
    first = torch.full((len(uniq),), len(pts), dtype=torch.long)
    first.scatter_reduce_(0, inv[order], torch.arange(len(pts)), "amin")
    centers = uniq[first.argsort()[:k]]
    empty_total = 0
    for _ in range(iters):
        centers, empty = lloyd_step(pts, centers)
        empty_total += empty
    stats = {"samples": len(pts), "distinct": len(uniq), "iters": iters, "empty_clusters": empty_total}
    return centers.float().to(dirs.device), stats


def assign(dirs, centers, chunk=1 << 18):
    """Index of the max-dot center (first on ties) for each row of dirs, FP64, chunked. int64 [V]."""
    c = centers.to(dirs.device, torch.float64).T
    return torch.cat([(d.double() @ c).argmax(1) for d in dirs.split(chunk)])
