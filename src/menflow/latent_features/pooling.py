"""Pooling strategies for MAISI-v2 latents.

Three pooling strategies cover the E2 chain:

- :func:`mask_anchored_mean` averages latent voxels inside the projected tumor
  mask, the lesion-localized representation.
- :func:`global_mean` averages all latent voxels, the cohort baseline.
- :func:`random_region_mean` averages a random region of matched spatial size,
  the negative control used in E2.3.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def mask_anchored_mean(
    z: Tensor,
    mask_lat: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Mean of ``z`` inside ``mask_lat`` along spatial dims.

    Parameters
    ----------
    z
        Latent tensor of shape ``(B, C, H', W', D')``.
    mask_lat
        Binary mask of shape ``(B, 1, H', W', D')`` (same spatial dims as ``z``).
    eps
        Numerical floor on the mask voxel count to avoid 0/0.

    Returns
    -------
    Tensor
        ``(B, C)`` mean inside the mask. Empty masks (all zeros) yield a zero
        vector and emit a warning.
    """
    if z.ndim != 5 or mask_lat.ndim != 5:
        raise ValueError(
            f"z and mask_lat must be 5-D; got {tuple(z.shape)}, {tuple(mask_lat.shape)}"
        )
    if z.shape[2:] != mask_lat.shape[2:]:
        raise ValueError(
            f"spatial mismatch: z={tuple(z.shape[2:])} vs mask={tuple(mask_lat.shape[2:])}"
        )
    m = mask_lat.to(z.dtype)
    n_voxels = m.sum(dim=(2, 3, 4))  # (B, 1)
    weighted = (z * m).sum(dim=(2, 3, 4))  # (B, C)
    out = weighted / n_voxels.clamp(min=eps)
    empty = n_voxels.squeeze(1) == 0
    if bool(empty.any()):
        logger.warning(
            "mask_anchored_mean: %d/%d items have empty masks; returning zeros for those.",
            int(empty.sum().item()),
            int(empty.numel()),
        )
        out = torch.where(empty[:, None], torch.zeros_like(out), out)
    return out


def global_mean(z: Tensor) -> Tensor:
    """Spatial mean of ``z``. Returns ``(B, C)``."""
    if z.ndim != 5:
        raise ValueError(f"z must be 5-D; got {tuple(z.shape)}")
    return z.mean(dim=(2, 3, 4))


def random_region_mean(
    z: Tensor,
    n_voxels: Tensor,
    *,
    seed: int = 0,
) -> Tensor:
    """Mean of ``z`` over a random region of size ``n_voxels`` per item.

    The random region is drawn uniformly without replacement from the latent
    grid for each item. Used as a negative control in E2.3 to verify that
    mask-anchored pooling beats an arbitrary spatial pooling of equal size.

    Parameters
    ----------
    z
        Latent tensor ``(B, C, H', W', D')``.
    n_voxels
        Per-item number of voxels to sample, shape ``(B,)``. Typically the
        latent-mask voxel count for that scan. Items with ``n_voxels == 0``
        receive a zero vector.
    seed
        RNG seed; set per-call so the experiment is reproducible.
    """
    if z.ndim != 5:
        raise ValueError(f"z must be 5-D; got {tuple(z.shape)}")
    if n_voxels.ndim != 1 or n_voxels.shape[0] != z.shape[0]:
        raise ValueError(
            f"n_voxels must be (B,) matching z; got {tuple(n_voxels.shape)} vs B={z.shape[0]}"
        )
    b, c, h, w, d = z.shape
    flat = z.reshape(b, c, h * w * d)  # (B, C, V)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.zeros((b, c), dtype=z.dtype, device=z.device)
    total = h * w * d
    for i in range(b):
        k = int(n_voxels[i].item())
        if k <= 0:
            continue
        k = min(k, total)
        idx = torch.randperm(total, generator=g)[:k]
        out[i] = flat[i, :, idx.to(flat.device)].mean(dim=1)
    return out
