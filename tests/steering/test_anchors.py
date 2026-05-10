"""Unit tests for stratified_anchor_indices."""

from __future__ import annotations

import numpy as np
import pytest

from menflow.steering.anchors import stratified_anchor_indices


def _toy_log_v(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Spread across [-2, 4] log cm³ — 5 bins of width ~1.2.
    return rng.uniform(-2.0, 4.0, size=n).astype(np.float64)


def test_returns_n_per_bin_when_enough_candidates():
    log_v = _toy_log_v(n=200)
    cand = np.arange(200)
    edges = np.array([-2.5, -1.0, 0.5, 1.5, 2.5, 4.5])
    anchors = stratified_anchor_indices(log_v, cand, edges, n_per_bin=2, seed=0)
    assert len(anchors) == 5 * 2
    bins = [a.bin_id for a in anchors]
    assert sorted(bins) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_deterministic_under_seed():
    log_v = _toy_log_v(n=300)
    cand = np.arange(300)
    edges = np.array([-2.5, -1.0, 0.5, 1.5, 2.5, 4.5])
    a1 = stratified_anchor_indices(log_v, cand, edges, n_per_bin=3, seed=42)
    a2 = stratified_anchor_indices(log_v, cand, edges, n_per_bin=3, seed=42)
    assert [a.index for a in a1] == [a.index for a in a2]


def test_different_seeds_differ():
    log_v = _toy_log_v(n=300)
    cand = np.arange(300)
    edges = np.array([-2.5, -1.0, 0.5, 1.5, 2.5, 4.5])
    a1 = stratified_anchor_indices(log_v, cand, edges, n_per_bin=3, seed=0)
    a2 = stratified_anchor_indices(log_v, cand, edges, n_per_bin=3, seed=1)
    assert [a.index for a in a1] != [a.index for a in a2]


def test_empty_bin_raises():
    log_v = _toy_log_v(n=200)
    cand = np.arange(200)
    edges = np.array([-2.5, -1.0, 0.5, 1.5, 2.5, 100.0, 200.0])  # last bin empty
    with pytest.raises(ValueError, match="no eligible candidates"):
        stratified_anchor_indices(log_v, cand, edges, n_per_bin=1, seed=0)


def test_take_truncated_when_short():
    log_v = _toy_log_v(n=10)
    cand = np.arange(10)
    edges = np.array([-2.5, -1.0, 0.5, 1.5, 2.5, 4.5])
    anchors = stratified_anchor_indices(log_v, cand, edges, n_per_bin=10, seed=0)
    # Each bin will return at most as many candidates as it has — exact total
    # depends on the random draw but must equal len(cand).
    assert len(anchors) == 10


def test_non_monotone_edges_rejected():
    log_v = _toy_log_v(n=10)
    cand = np.arange(10)
    edges = np.array([0.0, 1.0, 0.5, 2.0])  # non-monotone
    with pytest.raises(ValueError, match="strictly increasing"):
        stratified_anchor_indices(log_v, cand, edges, n_per_bin=1, seed=0)
