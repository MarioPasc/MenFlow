"""Unit tests for off_manifold_drift."""

from __future__ import annotations

import pytest
import torch

from experiments.E2._lib.steering.drift import off_manifold_drift


def test_drift_zero_when_identical():
    z = torch.randn(4, 8, 8, 8)
    assert off_manifold_drift(z, z) == 0.0


def test_drift_positive_when_perturbed():
    z = torch.randn(4, 8, 8, 8)
    z2 = z + 0.1 * torch.randn_like(z)
    assert off_manifold_drift(z, z2) > 0.0


def test_drift_relative_scale_invariant():
    z = torch.randn(4, 8, 8, 8)
    perturb = 0.05 * torch.randn_like(z)
    d1 = off_manifold_drift(z, z + perturb)
    d2 = off_manifold_drift(10 * z, 10 * (z + perturb))
    # ||10(z+p) - 10z|| / ||10z|| = ||p|| / ||z|| -> identical to d1.
    assert d1 == pytest.approx(d2, rel=1e-5)


def test_drift_shape_mismatch_raises():
    a = torch.randn(4, 8, 8, 8)
    b = torch.randn(4, 8, 8, 7)
    with pytest.raises(ValueError, match="shape mismatch"):
        off_manifold_drift(a, b)


def test_drift_handles_fp16_via_fp32_path():
    z = torch.randn(4, 4, 4, 4, dtype=torch.float16)
    z2 = z.clone() + torch.tensor(1e-3, dtype=torch.float16) * torch.ones_like(z)
    d = off_manifold_drift(z, z2)
    assert d > 0.0 and d < 1.0
