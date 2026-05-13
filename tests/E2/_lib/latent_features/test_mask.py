"""Unit tests for mask projection."""

from __future__ import annotations

import torch

from experiments.E2._lib.latent_features.mask import project_mask_to_latent


def test_project_mask_uniform_block_returns_one_voxel() -> None:
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[0, 0, 0:4, 0:4, 0:4] = 1.0
    out = project_mask_to_latent(mask, stride=4)
    assert out.shape == (1, 1, 1, 1, 1)
    assert bool(out.item()) is True


def test_project_mask_zero_input_yields_zero() -> None:
    mask = torch.zeros(1, 1, 8, 8, 8)
    out = project_mask_to_latent(mask, stride=4)
    assert out.shape == (1, 1, 2, 2, 2)
    assert out.sum().item() == 0


def test_project_mask_single_voxel_propagates_to_one_latent_voxel() -> None:
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[0, 0, 1, 1, 1] = 1.0
    out = project_mask_to_latent(mask, stride=4)
    assert out.shape == (1, 1, 2, 2, 2)
    assert int(out.sum().item()) == 1
    assert bool(out[0, 0, 0, 0, 0].item())


def test_project_mask_accepts_3d_input() -> None:
    mask = torch.zeros(8, 8, 8)
    mask[0, 0, 0] = 1.0
    out = project_mask_to_latent(mask, stride=4)
    assert out.shape == (1, 1, 2, 2, 2)
