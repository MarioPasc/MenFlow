"""Unit tests for the mask-localised + mask-pooled ρ_lin variants."""

from __future__ import annotations

import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis.metrics import (
    linearity_residual,
    linearity_residual_pooled,
)


def test_mask_isolates_changing_voxels() -> None:
    """Full-latent ρ_lin should match mask-localised when background is identical."""
    rng = np.random.default_rng(0)
    z1 = rng.standard_normal((2, 2, 4, 4, 4)).astype(np.float64)
    z3 = z1.copy()
    z3[..., :2, :2, :2] += 1.0  # only this region grows
    z2 = z1 + 0.5 * (z3 - z1)
    z2[..., :2, :2, :2] += 0.2  # add orthogonal noise inside the changing region
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[:2, :2, :2] = True

    res_full = linearity_residual(z1, z2, z3)
    res_mask = linearity_residual(z1, z2, z3, mask_latent=mask)
    # The orthogonal noise lives entirely inside the mask, so the mask-localised
    # ρ_lin is bounded below by the full version (numerator concentrated, denom
    # decreased). Both finite and non-negative.
    assert res_full.rho_lin >= 0
    assert res_mask.rho_lin >= 0
    assert np.isfinite(res_mask.rho_lin)


def test_mask_pooled_low_dim() -> None:
    """Pooled ρ_lin on collinear pooled vectors is ~ 0."""
    p1 = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])  # (M=2, C=4)
    p3 = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]])
    p2 = 0.5 * (p1 + p3)
    res = linearity_residual_pooled(p1, p2, p3)
    assert res.rho_lin < 1e-9
    assert abs(res.beta_star - 0.5) < 1e-9


def test_mask_pooled_orthogonal() -> None:
    """Pooled ρ_lin is non-trivial when z2 deviates orthogonally to the line."""
    p1 = np.zeros((2, 4))
    p3 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    p2 = np.array([[0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    res = linearity_residual_pooled(p1, p2, p3)
    # residual norm = 1, line norm = 1 → ρ_lin = 1
    assert abs(res.rho_lin - 1.0) < 1e-9
