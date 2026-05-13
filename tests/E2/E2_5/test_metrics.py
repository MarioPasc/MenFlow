"""Unit tests for E2.5 metrics (CPU-only, no fixtures)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.E2.E2_5_ae_longitudinal.analysis.metrics import (
    DegenerateTripletError,
    linearity_residual,
    volume_matched_beta,
)


def test_linearity_residual_collinear_midpoint() -> None:
    """z2 = midpoint(z1, z3) ⇒ β*=0.5, ρ_lin=0."""
    z1 = np.array([0.0, 0.0, 0.0])
    z3 = np.array([2.0, 4.0, -2.0])
    z2 = 0.5 * (z1 + z3)
    res = linearity_residual(z1, z2, z3)
    assert abs(res.beta_star - 0.5) < 1e-9
    assert res.rho_lin < 1e-9


def test_linearity_residual_orthogonal() -> None:
    """z2 orthogonal to the line, midway in projection ⇒ β*=0.5, ρ_lin=||orth||/||d||."""
    z1 = np.array([0.0, 0.0, 0.0])
    z3 = np.array([4.0, 0.0, 0.0])
    z2 = np.array([2.0, 3.0, 0.0])  # midpoint + 3 along y
    res = linearity_residual(z1, z2, z3)
    assert abs(res.beta_star - 0.5) < 1e-9
    assert abs(res.rho_lin - 3.0 / 4.0) < 1e-9


def test_linearity_residual_outside_clips_beta() -> None:
    """β projects > 1 ⇒ clipped to 1."""
    z1 = np.array([0.0, 0.0])
    z3 = np.array([1.0, 0.0])
    z2 = np.array([2.0, 0.0])  # projects to β=2, clipped to 1
    res = linearity_residual(z1, z2, z3)
    assert res.beta_star == 1.0
    # residual = ||z2 - z3|| / ||d|| = 1.0
    assert abs(res.rho_lin - 1.0) < 1e-9


def test_linearity_residual_degenerate() -> None:
    z = np.zeros(8)
    with pytest.raises(DegenerateTripletError):
        linearity_residual(z, z + 1, z)


def test_volume_matched_beta_with_mock_probe() -> None:
    """A scalar latent + identity probe must recover β_vol = 0.5 when target = midpoint."""
    # z shape: (M=1, C=1, 4, 4, 4). The mask covers all voxels.
    z1 = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
    z3 = np.full((1, 1, 4, 4, 4), 1.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=bool)

    # Probe: log_v = coef * mean(z); coef = 1.0.
    class IdProbe:
        def __call__(self, pooled: np.ndarray) -> float:
            return float(pooled[0])

    target = 0.5
    beta_vol, betas, log_v_hat = volume_matched_beta(
        z1, z3, mask_latent=mask, log_v_target=target,
        probe_predict_per_modality=[IdProbe()], n_grid=11,
    )
    assert abs(beta_vol - 0.5) < 1e-6
    assert abs(log_v_hat[5] - 0.5) < 1e-6
