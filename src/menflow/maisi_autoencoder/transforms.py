"""Pre/post-processing transforms around the MAISI-v2 autoencoder.

Two stateless helpers produce the I/O wrappers needed by the inference pipeline:

* :func:`build_preprocess` — percentile-rescale a single-channel volume to [0, 1]
  and pad spatial dims to a multiple of *pad_multiple* (so the 3-stage encoder
  produces an integer-valued latent shape).
* :func:`build_postprocess` — inverse of the above: crop back to the original
  shape and rescale the reconstruction to source intensity units.

Per-volume percentile bounds are stored alongside the latent so decoding is
fully reversible. The transforms operate on numpy arrays; tensor handling is
the model wrapper's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PercentileNormalizer:
    """State-bearing percentile rescaler used during a single encode/decode pass."""

    lower_value: float
    upper_value: float
    b_min: float = 0.0
    b_max: float = 1.0

    def forward(self, volume: np.ndarray) -> np.ndarray:
        """Map ``volume`` from source units to [b_min, b_max]."""
        scale = (self.b_max - self.b_min) / max(self.upper_value - self.lower_value, 1e-8)
        return (volume - self.lower_value) * scale + self.b_min

    def inverse(self, volume: np.ndarray) -> np.ndarray:
        """Map a normalised reconstruction back to source units."""
        scale = (self.upper_value - self.lower_value) / max(self.b_max - self.b_min, 1e-8)
        return (volume - self.b_min) * scale + self.lower_value


def build_preprocess(
    volume: np.ndarray,
    *,
    lower_percentile: float = 0.0,
    upper_percentile: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
    pad_multiple: int = 16,
    pad_mode: str = "reflect",
) -> tuple[np.ndarray, PercentileNormalizer, tuple[int, int, int], tuple[tuple[int, int], ...]]:
    """Prepare a single-modality 3D volume for encoding.

    Parameters
    ----------
    volume
        Input array with shape ``(H, W, D)``.
    lower_percentile, upper_percentile
        Endpoints used for intensity rescaling.
    b_min, b_max
        Target intensity range after rescaling.
    pad_multiple
        Spatial dims will be reflection-padded to a multiple of this number on
        the trailing edge of each axis.

    Returns
    -------
    padded
        Float32 array of shape ``(H', W', D')`` with ``H' = ceil(H/m)*m`` etc.
    normalizer
        :class:`PercentileNormalizer` carrying the percentile values, so
        :meth:`PercentileNormalizer.inverse` can undo the rescaling.
    original_shape
        ``(H, W, D)`` — needed to crop the reconstruction back.
    pad_widths
        The exact pad widths applied per axis ``((0, pad_h), (0, pad_w), (0, pad_d))``.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {volume.shape}")

    arr = volume.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    lo = float(np.percentile(arr, lower_percentile))
    hi = float(np.percentile(arr, upper_percentile))
    if hi <= lo:
        hi = lo + 1.0  # degenerate-volume guard

    norm = PercentileNormalizer(lower_value=lo, upper_value=hi, b_min=b_min, b_max=b_max)
    rescaled = norm.forward(arr)

    original_shape = tuple(arr.shape)
    pad_widths = tuple(
        (0, _ceil_to_multiple(s, pad_multiple) - s) for s in original_shape
    )
    if any(p[1] for p in pad_widths):
        padded = np.pad(rescaled, pad_widths, mode=pad_mode)
    else:
        padded = rescaled

    return padded.astype(np.float32, copy=False), norm, original_shape, pad_widths


def build_postprocess(
    reconstruction: np.ndarray,
    *,
    normalizer: PercentileNormalizer,
    original_shape: Sequence[int],
    pad_widths: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Crop a padded reconstruction and undo intensity rescaling.

    Parameters
    ----------
    reconstruction
        Padded reconstruction emitted by the decoder, shape ``(H', W', D')``.
    normalizer
        The same :class:`PercentileNormalizer` returned by :func:`build_preprocess`.
    original_shape
        Pre-pad spatial shape used to slice the reconstruction.
    pad_widths
        Optional pad widths; if absent, derived from the difference between
        ``reconstruction.shape`` and ``original_shape``.
    """
    if reconstruction.ndim != 3:
        raise ValueError(f"expected 3D reconstruction, got shape {reconstruction.shape}")

    if pad_widths is None:
        pad_widths = tuple(
            (0, max(int(reconstruction.shape[i] - original_shape[i]), 0)) for i in range(3)
        )
    sl = tuple(slice(p[0], p[0] + s) for p, s in zip(pad_widths, original_shape))
    cropped = reconstruction[sl]
    return normalizer.inverse(cropped)


def _ceil_to_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m
