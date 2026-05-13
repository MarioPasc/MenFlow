"""Unit tests for probe.py — load + predict."""

from __future__ import annotations

import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis.probe import RidgeProbe


def test_predict_single_vector(tmp_path):
    coef = np.array([1.0, -2.0, 0.5, 3.0])
    intercept = 0.7
    npz = tmp_path / "direction.npz"
    np.savez(
        npz,
        direction=coef / np.linalg.norm(coef),
        direction_norm=float(np.linalg.norm(coef)),
        coef_raw=coef,
        intercept=intercept,
    )
    probe = RidgeProbe.load_npz(npz)
    x = np.array([0.5, 0.5, 0.5, 0.5])
    expected = float(coef @ x + intercept)
    assert abs(probe.predict(x) - expected) < 1e-9


def test_predict_batch(tmp_path):
    coef = np.array([1.0, 0.0, 0.0, 0.0])
    intercept = 0.0
    npz = tmp_path / "direction.npz"
    np.savez(npz, coef_raw=coef, intercept=intercept)
    probe = RidgeProbe.load_npz(npz)
    X = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    out = probe.predict(X)
    assert out.shape == (2,)
    assert abs(out[0] - 1.0) < 1e-9
    assert abs(out[1] - 5.0) < 1e-9


def test_load_missing_coef_raw_raises(tmp_path):
    import pytest

    npz = tmp_path / "bad.npz"
    np.savez(npz, direction=np.zeros(4))
    with pytest.raises(KeyError):
        RidgeProbe.load_npz(npz)
