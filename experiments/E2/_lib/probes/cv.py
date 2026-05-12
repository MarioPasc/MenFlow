"""Patient-grouped cross-validation utilities."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold


def patient_grouped_kfold_indices(
    groups: np.ndarray,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` from a patient-grouped K-fold.

    Wraps :class:`sklearn.model_selection.GroupKFold` so callers do not need to
    pass ``X`` and ``y`` arrays just to iterate splits.
    """
    cv = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(groups), 1))
    for tr, te in cv.split(dummy_X, groups=groups):
        yield tr, te
