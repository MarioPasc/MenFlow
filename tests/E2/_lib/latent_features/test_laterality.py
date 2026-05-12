"""Tests for compute_laterality_from_mask."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from experiments.E2._lib.latent_features import compute_laterality_from_mask

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Basic laterality tests (axis=0, default)
# ---------------------------------------------------------------------------


def test_left_mask_returns_L() -> None:
    """Mask occupying only the low-index (left) side must return 'L'."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[0:5, :, :] = 1.0
    assert compute_laterality_from_mask(mask) == "L"


def test_right_mask_returns_R() -> None:
    """Mask occupying only the high-index (right) side must return 'R'."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[15:20, :, :] = 1.0
    assert compute_laterality_from_mask(mask) == "R"


def test_centered_mask_returns_midline() -> None:
    """Mask centered around the midpoint must return 'midline'."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[8:12, :, :] = 1.0
    assert compute_laterality_from_mask(mask) == "midline"


def test_all_zero_mask_returns_midline_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """All-zero mask returns 'midline' and logs a warning."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        result = compute_laterality_from_mask(mask)
    assert result == "midline"
    assert any("empty mask" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# Torch tensor input
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_torch_tensor_left_returns_L() -> None:
    """torch.Tensor input should be handled identically to numpy."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[0:5, :, :] = 1.0
    tensor = torch.from_numpy(mask)
    assert compute_laterality_from_mask(tensor) == "L"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_torch_tensor_right_returns_R() -> None:
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[15:20, :, :] = 1.0
    tensor = torch.from_numpy(mask)
    assert compute_laterality_from_mask(tensor) == "R"


# ---------------------------------------------------------------------------
# axis override
# ---------------------------------------------------------------------------


def test_axis2_override() -> None:
    """With axis=2, the depth dimension is treated as L-R."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    # Place mass at low indices of axis=2 → 'L'
    mask[:, :, 0:5] = 1.0
    assert compute_laterality_from_mask(mask, axis=2) == "L"

    # Place mass at high indices of axis=2 → 'R'
    mask2 = np.zeros((20, 20, 20), dtype=np.float32)
    mask2[:, :, 15:20] = 1.0
    assert compute_laterality_from_mask(mask2, axis=2) == "R"


# ---------------------------------------------------------------------------
# Edge: 2-D mask raises
# ---------------------------------------------------------------------------


def test_2d_mask_raises_value_error() -> None:
    """Non-3D mask must raise ValueError."""
    with pytest.raises(ValueError, match="3-D"):
        compute_laterality_from_mask(np.ones((20, 20)))
