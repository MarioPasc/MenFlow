"""Unit tests for wt_mask_from_prediction (label-map normalisation)."""

from __future__ import annotations

import numpy as np
import pytest

from menflow.segmentation.output import LABEL_MAPS, wt_mask_from_prediction


def test_brats25_includes_labels_1_and_2():
    seg = np.array([0, 1, 2, 3], dtype=np.int8)
    mask = wt_mask_from_prediction(seg, "brats25")
    assert mask.tolist() == [False, True, True, False]
    # 3 is not in the BraTS25 schema; it shouldn't be counted as WT.


def test_brats23_includes_labels_1_2_3():
    seg = np.array([0, 1, 2, 3], dtype=np.int8)
    mask = wt_mask_from_prediction(seg, "brats23")
    assert mask.tolist() == [False, True, True, True]


def test_ground_truth_matches_brats23():
    seg = np.array([0, 1, 2, 3], dtype=np.int8)
    assert np.array_equal(
        wt_mask_from_prediction(seg, "ground_truth"),
        wt_mask_from_prediction(seg, "brats23"),
    )


def test_empty_seg_returns_empty_mask():
    seg = np.zeros((4, 4, 4), dtype=np.int8)
    assert wt_mask_from_prediction(seg, "brats25").sum() == 0
    assert wt_mask_from_prediction(seg, "brats23").sum() == 0


def test_unknown_label_map_raises():
    seg = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(KeyError, match="unknown label map"):
        wt_mask_from_prediction(seg, "unknown")


def test_label_maps_keys():
    for name in ("brats25", "brats23", "ground_truth"):
        assert name in LABEL_MAPS
        for region in ("wt", "tc", "et"):
            assert region in LABEL_MAPS[name]
