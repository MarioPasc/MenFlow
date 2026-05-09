"""Unit tests for pooling and volume utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from menflow.latent_features.pooling import (
    global_mean,
    mask_anchored_mean,
    random_region_mean,
)
from menflow.latent_features.volume import (
    log_volume_cm3_to_voxel_count,
    voxel_count_to_log_volume_cm3,
)


def test_global_mean_matches_torch_mean() -> None:
    z = torch.randn(2, 4, 6, 6, 6)
    out = global_mean(z)
    assert out.shape == (2, 4)
    assert torch.allclose(out, z.mean(dim=(2, 3, 4)))


def test_mask_anchored_mean_recovers_constant_signal() -> None:
    z = torch.zeros(1, 4, 6, 6, 6)
    z[..., 1, 1, 1] = 7.0
    mask = torch.zeros(1, 1, 6, 6, 6, dtype=torch.bool)
    mask[..., 1, 1, 1] = True
    out = mask_anchored_mean(z, mask)
    assert torch.allclose(out, torch.full((1, 4), 7.0))


def test_mask_anchored_mean_empty_mask_returns_zero() -> None:
    z = torch.randn(1, 4, 6, 6, 6)
    mask = torch.zeros(1, 1, 6, 6, 6, dtype=torch.bool)
    out = mask_anchored_mean(z, mask)
    assert torch.allclose(out, torch.zeros(1, 4))


def test_random_region_mean_reproducible_with_seed() -> None:
    z = torch.randn(2, 4, 4, 4, 4)
    n_voxels = torch.tensor([5, 7])
    a = random_region_mean(z, n_voxels, seed=42)
    b = random_region_mean(z, n_voxels, seed=42)
    assert torch.allclose(a, b)


def test_voxel_count_to_log_volume_known_values() -> None:
    # 1000 voxels at 1 mm³ each → 1 cm³ → log(1) == 0
    assert voxel_count_to_log_volume_cm3(1000, (1.0, 1.0, 1.0)) == pytest.approx(0.0)
    # Roundtrip
    spacing = (1.0, 1.0, 1.0)
    log_v = voxel_count_to_log_volume_cm3(2500.0, spacing)
    n_back = log_volume_cm3_to_voxel_count(log_v, spacing)
    assert n_back == pytest.approx(2500.0, rel=1e-9)


def test_voxel_count_array_input() -> None:
    arr = np.array([1000, 2000, 5000], dtype=np.float64)
    out = voxel_count_to_log_volume_cm3(arr, (1.0, 1.0, 1.0))
    assert out.shape == arr.shape
    assert out[0] == pytest.approx(np.log(1.0))
