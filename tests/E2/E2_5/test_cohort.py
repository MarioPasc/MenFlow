"""Unit tests for cohort.py — Patience-style monotone-3 + C4 ladder."""

from __future__ import annotations

import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis.cohort import (
    _longest_monotone_indices,
    _pick_triple_from_lis,
)


def test_lis_strict_monotone() -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    out = _longest_monotone_indices(vals)
    assert out == [0, 1, 2, 3]


def test_lis_with_ties_breaks_strict() -> None:
    """Equal values are not part of a strictly increasing subsequence."""
    vals = np.array([1.0, 2.0, 2.0, 3.0])
    out = _longest_monotone_indices(vals)
    assert len(out) == 3
    selected = vals[out]
    assert np.all(np.diff(selected) > 0)


def test_lis_non_monotone() -> None:
    vals = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    out = _longest_monotone_indices(vals)
    selected = vals[out]
    assert np.all(np.diff(selected) > 0)
    assert len(out) >= 4  # 1, 4, 5, 9 or 1, 4, 5, 6 etc.


def test_lis_with_nan_skipped() -> None:
    vals = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    out = _longest_monotone_indices(vals)
    selected = vals[out]
    assert np.all(np.diff(selected) > 0)
    assert not np.any(np.isnan(selected))


def test_pick_triple_balances_log_volume() -> None:
    vals = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    out = _pick_triple_from_lis([0, 1, 2, 3, 4], vals)
    assert out is not None
    first, mid, last = out
    assert first == 0
    assert last == 4
    # interior values 0.4, 0.5, 0.6; target = 0.5 → mid=2
    assert mid == 2


def test_pick_triple_short_lis_returns_none() -> None:
    assert _pick_triple_from_lis([0, 1], np.array([0.0, 1.0])) is None
