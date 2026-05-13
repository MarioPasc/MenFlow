"""Regression test: MS-SSIM must adapt its kernel to the smallest spatial dim.

BraTS volumes ship at ``(240, 240, 155)`` and the default MS-SSIM kernel of 11
with 5 weights requires ``min_spatial > 160`` — every BraTS scan triggered the
fallback NaN path. The fix shrinks the kernel to fit.
"""

from __future__ import annotations

import math

import pytest
import torch

from experiments.E1_pathology_preservation.analysis.metrics import _global_msssim


@pytest.mark.parametrize(
    "shape",
    [(240, 240, 155), (240, 240, 240), (155, 155, 155), (200, 200, 96)],
)
def test_msssim_returns_finite_for_realistic_volumes(shape: tuple[int, int, int]) -> None:
    torch.manual_seed(0)
    x = torch.rand((1, *shape), dtype=torch.float32)
    x_hat = x + 0.01 * torch.randn_like(x)
    val = _global_msssim(x, x_hat, data_range=1.0)
    assert not math.isnan(val), f"MS-SSIM is NaN for shape {shape}"
    assert 0.0 <= val <= 1.0


def test_msssim_returns_nan_for_too_small_volume() -> None:
    x = torch.rand((1, 16, 16, 16), dtype=torch.float32)
    val = _global_msssim(x, x.clone(), data_range=1.0)
    assert math.isnan(val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
