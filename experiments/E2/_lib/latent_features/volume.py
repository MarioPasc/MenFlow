"""Convert tumor voxel counts to clinical volume in cm³ and log-volume."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Floor on voxel count before log() to keep zero-volume scans finite.
_MIN_VOXELS = 1.0


def voxel_count_to_log_volume_cm3(
    n_voxels: int | float | np.ndarray,
    spacing_mm: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Compute ``log(volume_cm3)`` from a voxel count and isotropic-or-not spacing.

    A voxel volume is ``prod(spacing_mm)`` mm³, i.e. ``prod(spacing_mm) / 1000``
    cm³. The total tumor volume is ``n_voxels * voxel_volume_cm3``. To keep the
    log finite for empty masks we floor ``n_voxels`` at ``_MIN_VOXELS``.

    Parameters
    ----------
    n_voxels
        Scalar or 1-D array of voxel counts. Cast to float.
    spacing_mm
        Sequence of three voxel spacings in millimetres.

    Returns
    -------
    float or np.ndarray
        ``log(volume_cm3)`` matching the input shape.
    """
    sp = np.asarray(spacing_mm, dtype=np.float64)
    if sp.shape != (3,):
        raise ValueError(f"spacing_mm must have shape (3,); got {sp.shape}")
    voxel_vol_cm3 = float(np.prod(sp)) / 1000.0
    n = np.asarray(n_voxels, dtype=np.float64)
    n = np.maximum(n, _MIN_VOXELS)
    log_v = np.log(n * voxel_vol_cm3)
    if np.isscalar(n_voxels) or (hasattr(n_voxels, "ndim") and n_voxels.ndim == 0):
        return float(log_v)
    return log_v


def log_volume_cm3_to_voxel_count(
    log_v: float | np.ndarray,
    spacing_mm: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Inverse of :func:`voxel_count_to_log_volume_cm3`."""
    sp = np.asarray(spacing_mm, dtype=np.float64)
    voxel_vol_cm3 = float(np.prod(sp)) / 1000.0
    if voxel_vol_cm3 <= 0.0:
        raise ValueError("voxel_vol_cm3 must be positive")
    if np.isscalar(log_v):
        return math.exp(float(log_v)) / voxel_vol_cm3
    return np.exp(np.asarray(log_v, dtype=np.float64)) / voxel_vol_cm3
