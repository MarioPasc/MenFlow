"""Tests for naive_volume_direction."""

from __future__ import annotations

import numpy as np
import pytest

from menflow.directions import naive_volume_direction


def test_random_nonzero_vector_unit_norm() -> None:
    """Result of normalizing a random vector must have unit norm."""
    rng = np.random.default_rng(42)
    coef = rng.standard_normal(8)
    result = naive_volume_direction(coef)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, rtol=1e-12)


def test_already_unit_input_unchanged() -> None:
    """A unit vector should map to itself (modulo floating-point noise)."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal(6)
    v /= np.linalg.norm(v)
    result = naive_volume_direction(v)
    np.testing.assert_allclose(result, v, rtol=1e-12)


def test_all_zeros_raises_value_error() -> None:
    """Zero-norm input must raise ValueError."""
    with pytest.raises(ValueError, match="zero norm"):
        naive_volume_direction(np.zeros(5))


def test_single_element_positive() -> None:
    """Single positive element → +1."""
    result = naive_volume_direction(np.array([3.0]))
    np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-12)


def test_single_element_negative() -> None:
    """Single negative element → -1."""
    result = naive_volume_direction(np.array([-2.5]))
    np.testing.assert_allclose(result, np.array([-1.0]), rtol=1e-12)


def test_2d_input_raises_value_error() -> None:
    """Non-1D input must raise ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        naive_volume_direction(np.ones((3, 3)))
