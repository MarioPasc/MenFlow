"""Reconstruct the E2.1 ridge probe from its saved direction.npz.

The probe was fit by ``experiments/E2/E2_1_recoverability`` on the per-modality
``z_tumor`` features (mask-pooled latent-channel means, shape ``(N, C)`` with
C=4 for MAISI-v2). The artifact stores the raw scikit-learn ``coef_`` and
``intercept_``; we wrap them in a lightweight callable that takes a pooled
feature vector and returns predicted ``log V`` in ``log(cm^3)`` units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeProbe:
    """Linear probe with shape ``(C,)`` coefficient vector and scalar intercept."""

    coef: np.ndarray
    intercept: float

    @classmethod
    def load_npz(cls, path: str | Path) -> RidgeProbe:
        """Load a probe from an E2.1 ``direction.npz`` artifact.

        ``coef_raw`` is the un-normalised Ridge coefficient (the ``direction``
        field is the unit vector; we keep ``coef_raw`` because it has the
        correct scale for prediction).
        """
        data = np.load(path, allow_pickle=False)
        if "coef_raw" not in data:
            raise KeyError(f"{path}: missing 'coef_raw' field (legacy npz?)")
        coef = np.asarray(data["coef_raw"], dtype=np.float64).ravel()
        intercept = float(data.get("intercept", np.float64(0.0)))
        return cls(coef=coef, intercept=intercept)

    def predict(self, pooled: np.ndarray) -> np.ndarray:
        """Predict ``log V`` for one or many pooled-feature vectors.

        Parameters
        ----------
        pooled
            Either ``(C,)`` for a single scan or ``(B, C)`` for a batch.

        Returns
        -------
        np.ndarray
            Scalar ``log V`` (or batch of scalars) in ``log(cm^3)`` units.
        """
        arr = np.asarray(pooled, dtype=np.float64)
        if arr.ndim == 1:
            return arr @ self.coef + self.intercept
        if arr.ndim == 2:
            return arr @ self.coef + self.intercept
        raise ValueError(f"pooled must be (C,) or (B, C); got {arr.shape}")
