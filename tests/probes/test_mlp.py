"""Smoke test for the MLP probe."""

from __future__ import annotations

import numpy as np
import torch

from menflow.probes.mlp import fit_mlp_probe


def test_mlp_probe_recovers_linear_relation() -> None:
    rng = np.random.default_rng(0)
    n, c = 300, 4
    y = rng.uniform(-1.0, 4.0, size=n).astype(np.float32)
    direction = np.ones(c) / np.sqrt(c)
    z = (y[:, None] * direction[None] + 0.05 * rng.standard_normal((n, c))).astype(np.float32)
    groups = np.arange(n)

    torch.manual_seed(0)
    _, r2_cv = fit_mlp_probe(
        z,
        y,
        groups,
        hidden=64,
        max_epochs=80,
        patience=10,
        batch_size=32,
        n_splits=3,
        seed=0,
    )
    assert r2_cv > 0.85, f"expected R² > 0.85 on linear synthetic, got {r2_cv:.3f}"
