"""Pre/post-process invariants for the MAISI-v2 helper transforms."""

from __future__ import annotations

import numpy as np
import pytest

from menflow.maisi_autoencoder.transforms import (
    PercentileNormalizer,
    build_postprocess,
    build_preprocess,
)


@pytest.fixture(scope="module")
def volume() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(50, 800, size=(40, 40, 27)).astype(np.float32)


def test_preprocess_pads_to_multiple(volume: np.ndarray) -> None:
    padded, _norm, original_shape, pad_widths = build_preprocess(volume, pad_multiple=16)
    assert original_shape == (40, 40, 27)
    assert all(s % 16 == 0 for s in padded.shape)
    assert pad_widths == ((0, 8), (0, 8), (0, 5))


def test_normalize_inverse_round_trip(volume: np.ndarray) -> None:
    padded, norm, _, _ = build_preprocess(volume)
    # Cropping a region inside the original window should round-trip exactly.
    inside = padded[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    restored = norm.inverse(inside)
    # Volumes are not clipped during forward, so any value inside the percentile
    # window is recovered up to floating-point noise.
    mask = (volume >= norm.lower_value) & (volume <= norm.upper_value)
    np.testing.assert_allclose(restored[mask], volume[mask], rtol=1e-5, atol=1e-3)


def test_postprocess_crops_back(volume: np.ndarray) -> None:
    padded, norm, original_shape, pad_widths = build_preprocess(volume)
    out = build_postprocess(
        padded, normalizer=norm, original_shape=original_shape, pad_widths=pad_widths
    )
    assert out.shape == original_shape


def test_normalizer_handles_degenerate_volume() -> None:
    flat = np.full((8, 8, 8), 42.0, dtype=np.float32)
    padded, norm, _, _ = build_preprocess(flat, pad_multiple=8)
    assert norm.upper_value > norm.lower_value  # degenerate guard kicked in
    assert padded.shape == (8, 8, 8)
