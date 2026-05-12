"""Unit tests for delta_to_alpha calibration."""

from __future__ import annotations

import math

import pytest

from experiments.E2._lib.steering.calibration import alpha_to_delta, delta_to_alpha


def test_delta_to_alpha_t1c_value():
    # t1c E2.1 norm = 20.846; Δ log V = 1 → α ≈ 0.04797
    alpha = delta_to_alpha(1.0, 20.846)
    assert math.isclose(alpha, 1.0 / 20.846, rel_tol=1e-9)


def test_delta_to_alpha_zero_step():
    assert delta_to_alpha(0.0, 5.0) == 0.0


def test_delta_to_alpha_negative_step():
    assert delta_to_alpha(-1.5, 10.0) == -0.15


def test_inverse_round_trip():
    for delta in (-1.5, -0.5, 0.0, 0.5, 1.5):
        alpha = delta_to_alpha(delta, 20.846)
        assert math.isclose(alpha_to_delta(alpha, 20.846), delta, rel_tol=1e-9)


def test_invalid_norm_raises():
    with pytest.raises(ValueError, match="positive"):
        delta_to_alpha(1.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        delta_to_alpha(1.0, -3.0)
