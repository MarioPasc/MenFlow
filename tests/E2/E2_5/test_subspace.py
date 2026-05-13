"""Unit tests for the PCA longitudinal-direction test."""

from __future__ import annotations

import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis.subspace import (
    pca_longitudinal_direction,
)


def test_aligned_triples_give_high_variance() -> None:
    """All trajectories along the x-axis → PC1 captures ≥ 0.95 of variance."""
    rng = np.random.default_rng(0)
    triples = []
    for _ in range(20):
        a = rng.standard_normal(8) * 0.1
        d = np.zeros(8)
        d[0] = 1.0 + rng.normal(scale=0.05)  # scalar growth along axis 0
        b = a + 0.5 * d + rng.standard_normal(8) * 0.01
        c = a + d
        triples.append((a, b, c))
    res = pca_longitudinal_direction(triples)
    assert res.variance_explained > 0.95
    assert abs(res.cos_z3_minus_z1.mean()) > 0.95
    assert res.cos_z2_minus_z1.mean() > 0.9  # z2-z1 also points along PC1


def test_random_triples_give_low_variance() -> None:
    """Trajectories pointing in random directions → PC1 captures ≈ 1/D variance."""
    rng = np.random.default_rng(1)
    D = 8
    triples = []
    for _ in range(60):
        a = rng.standard_normal(D)
        c = rng.standard_normal(D)  # random direction, not aligned
        b = a + 0.5 * (c - a) + rng.standard_normal(D) * 0.5
        triples.append((a, b, c))
    res = pca_longitudinal_direction(triples)
    assert res.variance_explained < 0.5  # well below the R1 0.70 threshold
