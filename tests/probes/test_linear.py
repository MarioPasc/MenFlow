"""Linear probe tests on synthetic data with known direction."""

from __future__ import annotations

import numpy as np

from menflow.probes.linear import fit_linear_probe
from menflow.probes.metrics import r2_score


def _synthetic_linear(n: int = 200, c: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(-1.0, 4.0, size=n)  # log V range
    direction = np.ones(c) / np.sqrt(c)  # unit-norm 1_C / sqrt(C)
    z = y[:, None] * direction[None] + 0.05 * rng.standard_normal((n, c))
    groups = np.arange(n)  # one scan per "patient"
    return z, y, groups, direction


def test_linear_probe_recovers_high_r2() -> None:
    z, y, groups, _ = _synthetic_linear()
    model, r2_cv, alpha = fit_linear_probe(z, y, groups)
    assert r2_cv > 0.95
    assert alpha in (0.1, 1.0, 10.0, 100.0)
    pred = model.predict(z)
    assert r2_score(pred, y) > 0.95


def test_linear_probe_direction_aligned_with_signal() -> None:
    z, y, groups, true_dir = _synthetic_linear()
    model, _, _ = fit_linear_probe(z, y, groups)
    coef = np.asarray(model.coef_).ravel()
    cos = float(np.dot(coef, true_dir) / (np.linalg.norm(coef) * np.linalg.norm(true_dir)))
    assert cos > 0.99


def test_linear_probe_low_r2_on_noise() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 4))
    y = rng.standard_normal(200)
    groups = np.arange(200)
    _, r2_cv, _ = fit_linear_probe(z, y, groups)
    assert r2_cv < 0.1
