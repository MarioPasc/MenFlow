"""Regression test: decode must copy nested split groups (MenGrowth).

The Picasso MenGrowth decode failed with ``TypeError: Accessing a group is
done with bytes or str, not <class 'slice'>`` because ``_initialise_output``
assumed every ``splits/<name>`` was a leaf Dataset. MenGrowth ships
``splits/kfold/k{N}/foldX/{train,val}`` — nested groups. The fix walks the
tree via :func:`_copy_splits_filtered`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from routines.decode.engine.decode_engine import _copy_splits_filtered


def _make_nested_splits(tmp_path: Path) -> Path:
    p = tmp_path / "src.h5"
    with h5py.File(p, "w") as f:
        sp = f.create_group("splits")
        sp.create_dataset("train", data=np.array([0, 1, 2, 5, 9], dtype=np.int32))
        sp.create_dataset("val", data=np.array([3, 4, 7], dtype=np.int32))
        # MenGrowth-style nested kfold splits
        kf = sp.create_group("kfold")
        k5 = kf.create_group("k5")
        for fold in range(2):
            fg = k5.create_group(f"fold{fold}")
            fg.create_dataset("train", data=np.array([0, 2, 6, 8], dtype=np.int32))
            fg.create_dataset("val", data=np.array([1, 7, 9], dtype=np.int32))
    return p


def test_flat_splits_copied(tmp_path: Path) -> None:
    src_path = _make_nested_splits(tmp_path)
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path / "out.h5", "w") as out:
        grp = out.create_group("splits")
        _copy_splits_filtered(src["splits"], grp, n_patients_kept=10)
        assert set(out["splits"]["train"][:]) == {0, 1, 2, 5, 9}


def test_nested_splits_copied(tmp_path: Path) -> None:
    src_path = _make_nested_splits(tmp_path)
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path / "out.h5", "w") as out:
        grp = out.create_group("splits")
        _copy_splits_filtered(src["splits"], grp, n_patients_kept=10)
        assert "kfold/k5/fold0/train" in out["splits"]
        assert set(out["splits/kfold/k5/fold0/train"][:]) == {0, 2, 6, 8}
        assert set(out["splits/kfold/k5/fold1/val"][:]) == {1, 7, 9}


def test_splits_filter_drops_out_of_range_indices(tmp_path: Path) -> None:
    src_path = _make_nested_splits(tmp_path)
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path / "out.h5", "w") as out:
        grp = out.create_group("splits")
        _copy_splits_filtered(src["splits"], grp, n_patients_kept=5)
        assert set(out["splits/train"][:]) == {0, 1, 2}
        assert set(out["splits/val"][:]) == {3, 4}
        assert set(out["splits/kfold/k5/fold0/train"][:]) == {0, 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
