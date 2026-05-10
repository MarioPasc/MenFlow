"""Unit tests for :mod:`menflow.data.kfold_splits`.

Builds synthetic patient cohorts and exercises the multi-k layout, the shared
held-out test set, the per-fold log_v summary, and HDF5 round-trip.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from menflow.data.kfold_splits import (
    KFoldLayout,
    build_kfold_splits,
    compute_log_volume_voxels,
    read_kfold_from_h5,
    write_kfold_to_h5,
)


def test_compute_log_volume_voxels_floor() -> None:
    seg = np.zeros((4, 4, 4), dtype=np.int8)
    assert compute_log_volume_voxels(seg) == pytest.approx(np.log(1e-3))


def test_compute_log_volume_voxels_label_filter() -> None:
    seg = np.zeros((4, 4, 4), dtype=np.int8)
    seg[0, 0, 0] = 1
    seg[1, 1, 1] = 2
    seg[2, 2, 2] = 3
    seg[3, 3, 3] = 7  # not in label set
    assert compute_log_volume_voxels(seg, label_set=(1, 2, 3)) == pytest.approx(np.log(3.0))


def _synthetic_cohort(n_train: int = 30, n_val: int = 5) -> dict:
    n = n_train + n_val
    patient_list = [f"P{i:03d}" for i in range(n)]
    scan_to_patient = [(f"S{i:03d}", f"P{i:03d}") for i in range(n)]
    scan_log_v = {
        **{f"S{i:03d}": float(np.log(100 + 50 * i)) for i in range(n_train)},
        **{f"S{i:03d}": float(np.log(1e-3)) for i in range(n_train, n)},
    }
    scan_subset = {
        **{f"S{i:03d}": "train" for i in range(n_train)},
        **{f"S{i:03d}": "val" for i in range(n_train, n)},
    }
    return {
        "patient_list": patient_list,
        "scan_to_patient": scan_to_patient,
        "scan_log_v": scan_log_v,
        "scan_subset": scan_subset,
    }


def test_build_kfold_splits_returns_one_layout_per_k() -> None:
    cohort = _synthetic_cohort(n_train=30, n_val=5)
    layouts = build_kfold_splits(
        **cohort,
        k_values=(1, 3, 5),
        test_pct=0.1,
        seed=42,
        n_strata=3,
    )
    assert set(layouts) == {1, 3, 5}
    for k, layout in layouts.items():
        assert isinstance(layout, KFoldLayout)
        assert layout.k == k
        assert len(layout.folds) == k


def test_test_set_shared_across_k() -> None:
    cohort = _synthetic_cohort()
    layouts = build_kfold_splits(
        **cohort,
        k_values=(1, 3, 5),
        test_pct=0.1,
        seed=42,
        n_strata=3,
    )
    test_set_1 = set(layouts[1].test.tolist())
    test_set_3 = set(layouts[3].test.tolist())
    test_set_5 = set(layouts[5].test.tolist())
    assert test_set_1 == test_set_3 == test_set_5
    for k, layout in layouts.items():
        for f in layout.folds:
            assert not (set(f.train.tolist()) & test_set_1)
            assert not (set(f.val.tolist()) & test_set_1)


def test_unannotated_cohort_excluded() -> None:
    cohort = _synthetic_cohort(n_train=30, n_val=5)
    layouts = build_kfold_splits(
        **cohort,
        k_values=(1, 3),
        test_pct=0.1,
        seed=42,
        n_strata=3,
    )
    for layout in layouts.values():
        unique = set(layout.test.tolist())
        for f in layout.folds:
            unique |= set(f.train.tolist())
            unique |= set(f.val.tolist())
        assert all(p < 30 for p in unique)


def test_per_fold_disjoint_within_k() -> None:
    cohort = _synthetic_cohort()
    layouts = build_kfold_splits(
        **cohort,
        k_values=(3, 5),
        test_pct=0.1,
        seed=42,
        n_strata=3,
    )
    for layout in layouts.values():
        for f in layout.folds:
            assert not (set(f.train.tolist()) & set(f.val.tolist()))


def test_seed_determinism() -> None:
    cohort = _synthetic_cohort()
    a = build_kfold_splits(**cohort, k_values=(1, 3), seed=42)
    b = build_kfold_splits(**cohort, k_values=(1, 3), seed=42)
    for k in (1, 3):
        np.testing.assert_array_equal(a[k].test, b[k].test)
        for fa, fb in zip(a[k].folds, b[k].folds):
            np.testing.assert_array_equal(fa.train, fb.train)
            np.testing.assert_array_equal(fa.val, fb.val)


def test_log_v_distribution_populated() -> None:
    cohort = _synthetic_cohort()
    layouts = build_kfold_splits(**cohort, k_values=(3,), seed=42, n_strata=3)
    dist = layouts[3].log_v_distribution
    assert "test" in dist
    assert "folds" in dist
    assert len(dist["folds"]) == 3
    assert dist["test"]["n"] >= 1


def test_h5_roundtrip(tmp_path) -> None:
    cohort = _synthetic_cohort()
    layouts = build_kfold_splits(**cohort, k_values=(1, 3), seed=42, n_strata=3)
    p = tmp_path / "k.h5"
    with h5py.File(p, "w") as f:
        write_kfold_to_h5(f.create_group("kfold"), layouts)
    with h5py.File(p, "r") as f:
        read = read_kfold_from_h5(f["kfold"])
    assert set(read) == set(layouts)
    for k in layouts:
        np.testing.assert_array_equal(read[k].test, layouts[k].test)
        for fa, fb in zip(read[k].folds, layouts[k].folds):
            np.testing.assert_array_equal(fa.train, fb.train)
            np.testing.assert_array_equal(fa.val, fb.val)


def test_too_few_eligible_patients_rejected() -> None:
    cohort = {
        "patient_list": [f"P{i}" for i in range(4)],
        "scan_to_patient": [(f"S{i}", f"P{i}") for i in range(4)],
        "scan_log_v": {f"S{i}": 2.0 + 0.1 * i for i in range(4)},
        "scan_subset": {f"S{i}": "train" for i in range(4)},
    }
    with pytest.raises(ValueError, match="eligible patients"):
        build_kfold_splits(**cohort, k_values=(10,), seed=42)
