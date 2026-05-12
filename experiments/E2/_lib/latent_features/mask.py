"""Project source-resolution tumor masks into MAISI-v2 latent resolution.

The MAISI-v2 autoencoder downsamples spatial dimensions by a factor of 4. To
align a binary tumor mask defined on the source grid (H, W, D) with the latent
grid (H', W', D') = (H/4, W/4, D/4), we apply a strided MaxPool3d followed by a
threshold. The MaxPool variant is the canonical choice in TCAV-style pipelines:
any voxel inside a stride-block that is part of the tumor turns the latent voxel
on. Average-pool would shrink small lesions disproportionately.
"""

from __future__ import annotations

import logging

import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def project_mask_to_latent(
    mask: Tensor,
    *,
    stride: int = 4,
    threshold: float = 0.5,
) -> Tensor:
    """Project a binary mask from source to latent resolution.

    Parameters
    ----------
    mask
        Binary mask of shape ``(B, 1, H, W, D)`` with values in ``{0, 1}``. May
        also be passed as ``(H, W, D)`` or ``(1, H, W, D)``; the function will
        unsqueeze it to ``(1, 1, H, W, D)``.
    stride
        Pooling stride along each spatial axis. The MAISI-v2 encoder uses
        ``stride=4`` (downsample 4x in each dimension).
    threshold
        Threshold applied to the pooled fractional mask. With MaxPool the
        per-cell value is already either 0 or 1 when the input is binary; the
        threshold is kept for forward compatibility with average-pool variants.

    Returns
    -------
    Tensor
        Boolean mask at latent resolution of shape ``(B, 1, H', W', D')`` where
        ``H' = H // stride``. dtype is ``torch.bool``.
    """
    if mask.ndim == 3:
        mask = mask[None, None]
    elif mask.ndim == 4:
        mask = mask[None]
    if mask.ndim != 5:
        raise ValueError(f"mask must be 3-, 4-, or 5-D (got shape {tuple(mask.shape)})")
    if mask.shape[1] != 1:
        raise ValueError(f"mask must have a single channel; got shape {tuple(mask.shape)}")

    pooled = F.max_pool3d(mask.float(), kernel_size=stride, stride=stride)
    return pooled > threshold
