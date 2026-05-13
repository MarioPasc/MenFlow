"""Naive volume direction derived directly from a linear probe coefficient vector."""

from __future__ import annotations

import numpy as np


def naive_volume_direction(coef: np.ndarray) -> np.ndarray:
    """Return the unit vector along the linear probe coefficient.

    Parameters
    ----------
    coef:
        1-D coefficient vector from a fitted Ridge probe, shape ``(C,)``.

    Returns
    -------
    np.ndarray
        Unit vector ``coef / ||coef||``, shape ``(C,)``.

    Raises
    ------
    ValueError
        If ``coef`` is not 1-D or has zero norm.
    """
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim != 1:
        raise ValueError(f"coef must be 1-D; got shape {coef.shape}")
    norm = float(np.linalg.norm(coef))
    if norm == 0.0:
        raise ValueError("coef has zero norm; cannot produce a unit direction")
    return coef / norm
