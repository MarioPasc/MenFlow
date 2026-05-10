"""K-fold cross-validation splits for the MenFlow datasets.

Replaces the flat ``splits/e3_{train,val,test}`` layout with a self-describing
``splits/kfold/`` group that pre-computes splits for several `k` values:

* ``k=1`` — single holdout (train / val / test ≈ 80 / 10 / 10). Equivalent to
  the previous E3.1 layout; the FM finetuning routine reads this by default.
* ``k=3, 5, 10`` — proper cross-validation. The same held-out test set
  (≈10 % of patients) is shared across every k, so per-fold metrics are
  comparable across k. Each fold rotates a 1/k slice of the remaining 90 %
  through the val role; the other (k-1) slices form train.

Patients with no annotated tumor (BraTS challenge val cohort, or scans whose
segmentation is at the sentinel floor) are excluded from every split.

HDF5 layout
-----------

::

    splits/
    └── kfold/
        ├── k1/
        │   ├── test   [int32, n_test_pat]
        │   ├── fold_0/
        │   │   ├── train [int32, n_train_pat]
        │   │   └── val   [int32, n_val_pat]
        │   └── attrs:
        │       ├── n_folds              = 1
        │       ├── seed                 = 42
        │       ├── test_pct             = 0.1
        │       ├── log_v_distribution   = JSON-encoded {fold_i:{train:{}, val:{}}, test:{}}
        ├── k3/  ... (3 folds, same test set)
        ├── k5/  ...
        └── k10/ ...

The arrays index into ``longitudinal/patient_list``. Indices are int32,
non-negative, sorted ascending, and disjoint within a (k, fold) pair.

References
----------
Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning", §7.10.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


_DEFAULT_LABEL_SET: tuple[int, ...] = (1, 2, 3)
_LOGV_FLOOR: float = float(np.log(1e-3))  # = -6.9078


# ============================================================================
# Public API
# ============================================================================


@dataclass(frozen=True, slots=True)
class FoldArrays:
    """Train / val patient indices for one fold."""

    train: np.ndarray  # int32
    val: np.ndarray  # int32


@dataclass(frozen=True, slots=True)
class KFoldLayout:
    """Held-out test + per-fold (train, val) for a given k."""

    k: int
    test: np.ndarray  # int32, shared across all folds
    folds: list[FoldArrays]
    log_v_distribution: dict = field(default_factory=dict)


def compute_log_volume_voxels(
    segmentation: np.ndarray,
    label_set: Sequence[int] = _DEFAULT_LABEL_SET,
) -> float:
    """Natural-log tumor volume in voxel units, with floor at ln(1e-3)."""
    if segmentation.size == 0:
        return _LOGV_FLOOR
    n_voxels = int(np.isin(segmentation, list(label_set)).sum())
    return float(np.log(max(n_voxels, 1e-3)))


def build_kfold_splits(
    *,
    patient_list: Sequence[str],
    scan_to_patient: Iterable[tuple[str, str]],
    scan_log_v: dict[str, float],
    scan_subset: dict[str, str],
    k_values: Sequence[int] = (1, 3, 5, 10),
    test_pct: float = 0.1,
    seed: int = 42,
    n_strata: int = 5,
    excluded_subsets: Sequence[str] = ("val",),
    log_v_floor: float = -6.0,
) -> dict[int, KFoldLayout]:
    """Pre-compute multi-k splits with a shared held-out test set.

    Returns ``{k: KFoldLayout}``. The test set is identical across every k so
    rank-sweeps and k-sweeps remain directly comparable. ``k=1`` produces a
    single fold whose val is a stratified ``test_pct``-sized slice of the
    annotated cohort minus the test set (≈ 80 / 10 / 10 patient split for the
    default ``test_pct=0.1``).
    """
    eligible_idx, eligible_logv, n_excluded_subset, n_excluded_floor = _select_eligible(
        patient_list=patient_list,
        scan_to_patient=scan_to_patient,
        scan_log_v=scan_log_v,
        scan_subset=scan_subset,
        excluded_subsets=excluded_subsets,
        log_v_floor=log_v_floor,
    )
    logger.info(
        "kfold eligibility: %d/%d patients (excluded %d by subset, %d below log_v floor %.3f)",
        len(eligible_idx),
        len(patient_list),
        n_excluded_subset,
        n_excluded_floor,
        log_v_floor,
    )
    if len(eligible_idx) < max(k_values) + 1:
        raise ValueError(
            f"only {len(eligible_idx)} eligible patients; need at least "
            f"{max(k_values) + 1} to support every k in {tuple(k_values)}"
        )

    eligible_arr = np.asarray(eligible_idx, dtype=np.int64)
    logv_arr = np.asarray(eligible_logv, dtype=np.float64)

    # 1. Hold out the shared test set, stratified by log_v bin.
    test_strata = _quantile_bins(logv_arr, n_strata=n_strata)
    eligible_local, test_local = _stratified_holdout(
        n=len(eligible_arr),
        strata=test_strata,
        test_pct=test_pct,
        seed=seed,
    )
    test_global = eligible_arr[test_local].copy()
    test_global.sort()

    # 2. Per-k fold construction over the remaining patients (eligible_local).
    layouts: dict[int, KFoldLayout] = {}
    for k in sorted(set(int(x) for x in k_values)):
        layouts[k] = _build_one_k_layout(
            k=k,
            remaining_local=eligible_local,
            eligible_arr=eligible_arr,
            logv_arr=logv_arr,
            test_global=test_global,
            seed=seed + k,  # decoupled per-k seed → folds for k=3 and k=5 are independent
            n_strata=n_strata,
            test_pct=test_pct,
        )
    return layouts


def split_summary_for_log_v(values: np.ndarray) -> dict[str, float | int]:
    """Quantile summary of a per-patient log_v array, JSON-friendly."""
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "log_v_min": float(np.min(values)),
        "log_v_p05": float(np.quantile(values, 0.05)),
        "log_v_p50": float(np.quantile(values, 0.50)),
        "log_v_p95": float(np.quantile(values, 0.95)),
        "log_v_max": float(np.max(values)),
        "log_v_mean": float(np.mean(values)),
        "log_v_std": float(np.std(values)),
    }


# ============================================================================
# Internals
# ============================================================================


def _select_eligible(
    *,
    patient_list: Sequence[str],
    scan_to_patient: Iterable[tuple[str, str]],
    scan_log_v: dict[str, float],
    scan_subset: dict[str, str],
    excluded_subsets: Sequence[str],
    log_v_floor: float,
) -> tuple[list[int], list[float], int, int]:
    """Filter patients eligible for splitting; aggregate log_v at patient level."""
    pid_to_logv: dict[str, list[float]] = {pid: [] for pid in patient_list}
    pid_to_subsets: dict[str, set[str]] = {pid: set() for pid in patient_list}
    for scan_id, pid in scan_to_patient:
        if pid not in pid_to_logv:
            continue
        pid_to_logv[pid].append(scan_log_v.get(scan_id, _LOGV_FLOOR))
        pid_to_subsets[pid].add(scan_subset.get(scan_id, "unknown"))

    excluded = set(excluded_subsets)
    eligible_idx: list[int] = []
    eligible_logv: list[float] = []
    n_excluded_subset = 0
    n_excluded_floor = 0
    for i, pid in enumerate(patient_list):
        if pid_to_subsets.get(pid, set()) & excluded:
            n_excluded_subset += 1
            continue
        max_lv = max(pid_to_logv.get(pid, [_LOGV_FLOOR]))
        if max_lv <= log_v_floor:
            n_excluded_floor += 1
            continue
        eligible_idx.append(i)
        eligible_logv.append(max_lv)
    return eligible_idx, eligible_logv, n_excluded_subset, n_excluded_floor


def _quantile_bins(values: np.ndarray, *, n_strata: int) -> np.ndarray:
    """Quantile-bin a 1-D array; collapses to a single bin if degenerate."""
    n = values.size
    effective = max(1, min(int(n_strata), n))
    if effective <= 1:
        return np.zeros(n, dtype=np.int64)
    try:
        raw = pd.qcut(values, q=effective, labels=False, duplicates="drop")
        return np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0).astype(np.int64)
    except ValueError:
        return np.zeros(n, dtype=np.int64)


def _stratified_holdout(
    *,
    n: int,
    strata: np.ndarray,
    test_pct: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull a stratified test set; return ``(remaining_local, test_local)``."""
    indices = np.arange(n)
    # train_test_split with stratify=strata picks exactly test_pct fraction
    # while preserving the bin distribution. Falls back to non-stratified if a
    # bin has only one member.
    try:
        remaining_local, test_local = train_test_split(
            indices, test_size=test_pct, random_state=seed, stratify=strata
        )
    except ValueError:
        remaining_local, test_local = train_test_split(
            indices, test_size=test_pct, random_state=seed
        )
    return np.sort(remaining_local), np.sort(test_local)


def _build_one_k_layout(
    *,
    k: int,
    remaining_local: np.ndarray,
    eligible_arr: np.ndarray,
    logv_arr: np.ndarray,
    test_global: np.ndarray,
    seed: int,
    n_strata: int,
    test_pct: float,
) -> KFoldLayout:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_remaining = len(remaining_local)
    rem_strata = _quantile_bins_for_kfold(
        logv_arr[remaining_local], n_strata=n_strata, n_folds=max(k, 2)
    )

    folds: list[FoldArrays] = []
    if k == 1:
        # Stratified single holdout for val on the remaining patients. With
        # default test_pct=0.1, val_pct=0.1 / (1-0.1) ≈ 0.111 → ≈10 % of all
        # eligible patients ⇒ overall train/val/test ≈ 80/10/10.
        val_pct = test_pct / (1.0 - test_pct)
        try:
            tr_local, va_local = train_test_split(
                np.arange(n_remaining),
                test_size=val_pct,
                random_state=seed,
                stratify=rem_strata,
            )
        except ValueError:
            tr_local, va_local = train_test_split(
                np.arange(n_remaining),
                test_size=val_pct,
                random_state=seed,
            )
        train_global = np.sort(eligible_arr[remaining_local[tr_local]])
        val_global = np.sort(eligible_arr[remaining_local[va_local]])
        folds.append(
            FoldArrays(
                train=train_global.astype(np.int32),
                val=val_global.astype(np.int32),
            )
        )
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_local, va_local in skf.split(np.zeros(n_remaining), rem_strata):
            train_global = np.sort(eligible_arr[remaining_local[tr_local]])
            val_global = np.sort(eligible_arr[remaining_local[va_local]])
            folds.append(
                FoldArrays(
                    train=train_global.astype(np.int32),
                    val=val_global.astype(np.int32),
                )
            )

    distribution = {
        "test": split_summary_for_log_v(logv_arr[np.searchsorted(eligible_arr, test_global)]),
        "folds": [],
    }
    for f in folds:
        train_idx_local = np.searchsorted(eligible_arr, f.train)
        val_idx_local = np.searchsorted(eligible_arr, f.val)
        distribution["folds"].append(
            {
                "train": split_summary_for_log_v(logv_arr[train_idx_local]),
                "val": split_summary_for_log_v(logv_arr[val_idx_local]),
            }
        )

    return KFoldLayout(
        k=k,
        test=test_global.astype(np.int32),
        folds=folds,
        log_v_distribution=distribution,
    )


def _quantile_bins_for_kfold(
    values: np.ndarray,
    *,
    n_strata: int,
    n_folds: int,
) -> np.ndarray:
    """Stratify into log_v bins, clamped so every bin has >= n_folds members."""
    if values.size < n_folds:
        return np.zeros(values.size, dtype=np.int64)
    max_strata = max(1, values.size // n_folds)
    effective = max(1, min(int(n_strata), max_strata))
    return _quantile_bins(values, n_strata=effective)


def write_kfold_to_h5(group: h5py.Group, layouts: dict[int, KFoldLayout]) -> None:  # noqa: F821
    """Write a `{k: KFoldLayout}` mapping into an open HDF5 group named `kfold`.

    Creates ``kfold/k{N}/test`` (int32), ``kfold/k{N}/fold_{i}/{train,val}``
    (int32), and writes per-k attrs (n_folds, seed, log_v_distribution as a
    JSON string).
    """
    for k, layout in layouts.items():
        kg = group.create_group(f"k{k}")
        kg.attrs["n_folds"] = int(layout.k)
        kg.create_dataset("test", data=np.asarray(layout.test, dtype=np.int32))
        for i, f in enumerate(layout.folds):
            fg = kg.create_group(f"fold_{i}")
            fg.create_dataset("train", data=np.asarray(f.train, dtype=np.int32))
            fg.create_dataset("val", data=np.asarray(f.val, dtype=np.int32))
        kg.attrs["log_v_distribution"] = json.dumps(layout.log_v_distribution)


def read_kfold_from_h5(group: h5py.Group) -> dict[int, KFoldLayout]:  # noqa: F821
    """Inverse of :func:`write_kfold_to_h5`. Returns ``{k: KFoldLayout}``."""
    out: dict[int, KFoldLayout] = {}
    for key in group:
        if not key.startswith("k"):
            continue
        try:
            k = int(key[1:])
        except ValueError:
            continue
        kg = group[key]
        test = np.asarray(kg["test"][:], dtype=np.int32)
        folds: list[FoldArrays] = []
        i = 0
        while f"fold_{i}" in kg:
            fg = kg[f"fold_{i}"]
            folds.append(
                FoldArrays(
                    train=np.asarray(fg["train"][:], dtype=np.int32),
                    val=np.asarray(fg["val"][:], dtype=np.int32),
                )
            )
            i += 1
        try:
            distribution = json.loads(kg.attrs.get("log_v_distribution", "{}"))
        except (TypeError, json.JSONDecodeError):
            distribution = {}
        out[k] = KFoldLayout(k=k, test=test, folds=folds, log_v_distribution=distribution)
    return out
