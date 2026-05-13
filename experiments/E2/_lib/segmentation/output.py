"""Convert raw BraTS multi-class predictions into a binary whole-tumor mask.

BraTS challenge containers do not agree on output label semantics:

* **BraTS-MEN ground truth** (and BraTS23 predictions): ``{1: NETC, 2: SNFH, 3: ET}``.
* **BraTS25 predictions**: ``{1: SNFH, 2: ET}`` — NETC dropped (2-class).

The whole-tumor (WT) region is the union of all non-background labels in
either case. We expose a single :func:`wt_mask_from_prediction` that does the
right thing once told which convention applies.
"""

from __future__ import annotations

from typing import Final

import numpy as np

LABEL_MAPS: Final[dict[str, dict[str, list[int]]]] = {
    "brats25": {
        "wt": [1, 2],
        "tc": [2],
        "et": [2],
    },
    "brats23": {
        "wt": [1, 2, 3],
        "tc": [1, 3],
        "et": [3],
    },
    "ground_truth": {
        "wt": [1, 2, 3],
        "tc": [1, 3],
        "et": [3],
    },
}


def wt_mask_from_prediction(seg: np.ndarray, label_map_name: str) -> np.ndarray:
    """Reduce a multi-class prediction to a binary whole-tumor mask.

    Parameters
    ----------
    seg
        Integer-valued segmentation array (any shape).
    label_map_name
        Key into :data:`LABEL_MAPS` (e.g. ``"brats25"``, ``"brats23"``,
        ``"ground_truth"``).

    Returns
    -------
    np.ndarray
        Boolean mask of the same shape as ``seg``, ``True`` at every voxel
        whose label is in the WT union for the given convention.

    Raises
    ------
    KeyError
        If ``label_map_name`` is not recognised.
    """
    if label_map_name not in LABEL_MAPS:
        raise KeyError(f"unknown label map {label_map_name!r}; choose from {sorted(LABEL_MAPS)}")
    wt_labels = LABEL_MAPS[label_map_name]["wt"]
    return np.isin(seg, np.asarray(wt_labels, dtype=seg.dtype))
