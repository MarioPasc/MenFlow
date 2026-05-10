"""Unit tests for dilate_mask_lat."""

from __future__ import annotations

import pytest
import torch

from menflow.steering.mask_dilation import dilate_mask_lat


def test_radius_zero_is_noop_returns_bool():
    m = torch.zeros(8, 8, 8, dtype=torch.bool)
    m[3, 3, 3] = True
    out = dilate_mask_lat(m, radius=0)
    assert out.dtype == torch.bool
    assert torch.equal(out, m)


def test_radius_one_grows_central_voxel():
    m = torch.zeros(7, 7, 7, dtype=torch.bool)
    m[3, 3, 3] = True
    out = dilate_mask_lat(m, radius=1)
    # 3x3x3 cube centred at (3,3,3) is on; rest off.
    expected = torch.zeros_like(m)
    expected[2:5, 2:5, 2:5] = True
    assert torch.equal(out, expected)


def test_preserves_input_shape_for_each_ndim():
    for ndim in (3, 4, 5):
        shape = [1] * (ndim - 3) + [5, 5, 5]
        m = torch.zeros(*shape, dtype=torch.bool)
        out = dilate_mask_lat(m, radius=1)
        assert out.shape == m.shape


def test_negative_radius_rejected():
    m = torch.zeros(4, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match=">= 0"):
        dilate_mask_lat(m, radius=-1)
