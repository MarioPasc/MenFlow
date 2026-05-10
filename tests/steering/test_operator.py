"""Unit tests for the steering operators."""

from __future__ import annotations

import pytest
import torch

from menflow.steering.operator import STEER_OPERATORS, global_steer, local_steer


def _make_inputs(c: int = 4, h: int = 4, w: int = 4, d: int = 4, dtype=torch.float32):
    z = torch.zeros(c, h, w, d, dtype=dtype)
    mask = torch.zeros(h, w, d, dtype=torch.bool)
    mask[1:3, 1:3, 1:3] = True  # 8-voxel cube
    u = torch.tensor([1.0, 0.0, -1.0, 0.5], dtype=dtype)
    return z, mask, u


def test_local_steer_only_inside_mask():
    z, mask, u = _make_inputs()
    alpha = 2.5
    z_out = local_steer(z, mask, u, alpha)
    inside = z_out[:, mask]
    outside = z_out[:, ~mask]
    expected = (alpha * u).view(-1, 1)
    assert torch.allclose(inside, expected.expand_as(inside))
    assert torch.allclose(outside, torch.zeros_like(outside))


def test_local_steer_dtype_preserved():
    z, mask, u = _make_inputs(dtype=torch.float16)
    z_out = local_steer(z, mask, u, 1.0)
    assert z_out.dtype == torch.float16


def test_local_steer_accepts_batched_singleton():
    z, mask, u = _make_inputs()
    z_b = z.unsqueeze(0)
    z_out_b = local_steer(z_b, mask, u, 0.5)
    z_out = local_steer(z, mask, u, 0.5)
    assert torch.allclose(z_out_b, z_out)


def test_local_steer_rejects_real_batch():
    z, mask, u = _make_inputs()
    bad = z.unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
    with pytest.raises(ValueError, match="single anchor"):
        local_steer(bad, mask, u, 1.0)


def test_local_steer_mismatched_mask_shape():
    z, _, u = _make_inputs()
    bad_mask = torch.zeros(3, 3, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="spatial shape"):
        local_steer(z, bad_mask, u, 1.0)


def test_local_steer_wrong_direction_length():
    z, mask, _ = _make_inputs()
    u_bad = torch.zeros(3)
    with pytest.raises(ValueError, match="latent channels"):
        local_steer(z, mask, u_bad, 1.0)


def test_global_steer_adds_everywhere():
    z, mask, u = _make_inputs()
    z_out = global_steer(z, mask, u, -1.5)
    expected = (-1.5 * u).view(-1, 1, 1, 1).expand_as(z)
    assert torch.allclose(z_out, expected)


def test_strongly_local_alias():
    assert STEER_OPERATORS["strongly_local"] is local_steer
    assert STEER_OPERATORS["local"] is local_steer
    assert STEER_OPERATORS["global"] is global_steer
