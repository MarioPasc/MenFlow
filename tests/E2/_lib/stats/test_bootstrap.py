"""Tests for the patient-level bootstrap CI."""

from __future__ import annotations

import numpy as np

from experiments.E2._lib.probes.linear import fit_linear_probe
from experiments.E2._lib.stats.bootstrap import patient_level_bootstrap_r2


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    n, c = 200, 4
    y = rng.uniform(-1.0, 4.0, size=n)
    direction = np.ones(c) / np.sqrt(c)
    z = y[:, None] * direction[None] + 0.05 * rng.standard_normal((n, c))
    groups = np.arange(n)

    def fit_fn(zb, yb, gb):
        model, _, _ = fit_linear_probe(zb, yb, gb)
        return model

    ci_lo, ci_hi, scores = patient_level_bootstrap_r2(z, y, groups, fit_fn, n_boot=50, seed=0)
    assert ci_lo < ci_hi
    assert 0.5 <= ci_lo <= 1.0
    assert 0.5 <= ci_hi <= 1.0
    assert scores.shape == (50,)
    assert np.isfinite(scores).all()
