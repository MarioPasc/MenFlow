"""Optional small-tumor dilation of the latent-resolution mask.

The §3 forecast for E2.4 flagged that stride-4 mask projection collapses small
tumors to 3-10 latent voxels, which can produce sharp boundary artefacts when
the decoder responds non-linearly at small support. A radius-1 spherical
dilation (default-off) softens that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dilate_mask_lat(mask_lat: Tensor, radius: int = 0) -> Tensor:
    """Dilate a binary latent-resolution mask by ``radius`` voxels.

    Parameters
    ----------
    mask_lat
        Mask of shape ``(H', W', D')`` / ``(1, H', W', D')`` / ``(1, 1, H', W', D')``.
        Treated as binary; non-zero voxels are 1.
    radius
        Non-negative dilation radius in latent voxels. ``radius=0`` is a no-op
        and returns the input as a contiguous bool tensor.

    Returns
    -------
    Tensor
        Bool mask in the same shape and device as ``mask_lat``.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0; got {radius}")
    orig_shape = mask_lat.shape
    if mask_lat.ndim == 3:
        m = mask_lat[None, None]
    elif mask_lat.ndim == 4:
        m = mask_lat[None]
    elif mask_lat.ndim == 5:
        m = mask_lat
    else:
        raise ValueError(f"mask_lat must be 3-, 4- or 5-D; got {tuple(mask_lat.shape)}")
    if radius == 0:
        return mask_lat.to(torch.bool).contiguous()

    k = 2 * radius + 1
    pooled = F.max_pool3d(m.to(torch.float32), kernel_size=k, stride=1, padding=radius)
    out = (pooled > 0.5).to(torch.bool)
    return out.view(orig_shape)
