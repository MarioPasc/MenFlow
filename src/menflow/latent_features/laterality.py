"""Laterality classification from 3-D segmentation masks.

Determines whether a tumor is left-lateralized, right-lateralized, or midline
by computing the mass centroid of the binary mask along the L-R axis.

Orientation assumption: BraTS uses LAS orientation, where axis 0 is the
left-right axis and index 0 corresponds to the *left* side of the patient.
Pass ``axis`` explicitly if a different orientation is used.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def compute_laterality_from_mask(
    mask: np.ndarray | _torch.Tensor,
    *,
    axis: int = 0,
    midline_tol: float = 0.05,
) -> str:
    """Classify tumor laterality from a binary segmentation mask.

    Parameters
    ----------
    mask:
        3-D binary (or boolean) array of shape ``(H, W, D)``. May be a
        :class:`torch.Tensor`; it will be converted to NumPy automatically.
    axis:
        Spatial axis corresponding to the left-right direction.  Defaults to
        ``0``, which is correct for BraTS LAS orientation (index 0 = left).
    midline_tol:
        Fractional tolerance around the geometric midpoint.  If the centroid
        satisfies ``|c - mid| / extent < midline_tol``, the tumor is
        classified as midline.

    Returns
    -------
    str
        One of ``"L"``, ``"R"``, or ``"midline"``.

    Notes
    -----
    **Orientation convention**: in LAS orientation (used by BraTS and stored
    in the unified MenFlow H5), axis 0 runs from left (index 0) to right
    (index H-1).  A centroid below the midpoint is therefore left-sided.
    Override ``axis`` for volumes in other orientations (e.g. RAS uses axis 0
    reversed: index 0 = right).

    An empty mask (all zeros) returns ``"midline"`` with a warning.
    """
    if _TORCH_AVAILABLE and isinstance(mask, _torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D; got shape {mask.shape}")

    total = float(mask.sum())
    if total == 0.0:
        logger.warning("compute_laterality_from_mask: empty mask (all zeros); returning 'midline'")
        return "midline"

    size = mask.shape[axis]
    coords = np.arange(size, dtype=np.float64)

    # Build a shape for broadcasting: 1 along all dims except axis
    reshape = [1, 1, 1]
    reshape[axis] = size
    coords_bc = coords.reshape(reshape)

    centroid = float((mask * coords_bc).sum()) / total
    mid = (size - 1) / 2.0
    extent = float(size - 1)

    if extent == 0.0 or abs(centroid - mid) / extent < midline_tol:
        return "midline"

    # axis 0 in LAS: lower index = left
    if centroid < mid:
        return "L"
    return "R"
