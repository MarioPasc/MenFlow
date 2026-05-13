"""Thin reader around the MAISI-v2 latents H5 plus mask helpers.

The reader streams individual latents (a single `(M, C, H', W', D')` block) from
the provisional v0.2 schema (:mod:`menflow.maisi_autoencoder.latents_h5`). It
also exposes the per-scan intensity bounds that the decoder needs to undo the
percentile rescaling.

Mask helpers replicate the convention used by
``experiments/E2/compute_features``: the source-resolution tumour mask is
zero-padded to the encoder's padded shape and projected to the latent grid
via :func:`experiments.E2._lib.latent_features.mask.project_mask_to_latent`
(strided max-pool with the same downsampling factor as the encoder).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.E2._lib.latent_features.mask import project_mask_to_latent
from experiments.E2._lib.latent_features.pooling import mask_anchored_mean

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LatentMetadata:
    """Spatial/encoder provenance attrs needed to align masks & decode."""

    modalities: tuple[str, ...]
    n_modalities: int
    latent_channels: int
    latent_spatial_shape: tuple[int, int, int]
    padded_spatial_shape: tuple[int, int, int]
    working_spatial_shape: tuple[int, int, int]
    source_spatial_shape: tuple[int, int, int]
    spatial_op: str
    crop_offset: tuple[int, int, int]
    stride: int  # uniform downsampling factor (typically 4)


def read_latent_metadata(path: Path | str) -> LatentMetadata:
    """Read all spatial / encoder-related attrs into a frozen dataclass."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
    mods = [s if isinstance(s, str) else s.decode() for s in attrs["modalities"].tolist()]
    pad = tuple(int(x) for x in attrs["padded_spatial_shape"])
    latent = tuple(int(x) for x in attrs["latent_spatial_shape"])
    ratios = [p // l for p, l in zip(pad, latent)]
    if len(set(ratios)) != 1:
        raise ValueError(f"non-uniform latent stride: padded={pad}, latent={latent}")
    return LatentMetadata(
        modalities=tuple(mods),
        n_modalities=int(attrs["n_modalities"]),
        latent_channels=int(attrs["latent_channels"]),
        latent_spatial_shape=latent,
        padded_spatial_shape=pad,
        working_spatial_shape=tuple(int(x) for x in attrs["working_spatial_shape"]),
        source_spatial_shape=tuple(int(x) for x in attrs["source_spatial_shape"]),
        spatial_op=str(attrs.get("spatial_op", "none")),
        crop_offset=tuple(int(x) for x in attrs.get("crop_offset", (0, 0, 0))),
        stride=int(ratios[0]),
    )


def read_latent(path: Path | str, index: int) -> np.ndarray:
    """Return a single latent block, shape ``(M, C, H', W', D')`` float32."""
    with h5py.File(path, "r") as f:
        z = f["latents"][index]
    return np.asarray(z, dtype=np.float32)


def read_intensity_bounds(path: Path | str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` arrays of shape ``(M,)`` for one scan."""
    with h5py.File(path, "r") as f:
        lo = np.asarray(f["intensity_lower"][index], dtype=np.float32)
        hi = np.asarray(f["intensity_upper"][index], dtype=np.float32)
    return lo, hi


def prepare_mask_for_projection(
    mask: np.ndarray,
    *,
    spatial_op: str,
    working_shape: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
    crop_offset: tuple[int, int, int],
) -> np.ndarray:
    """Mirror the encoder's spatial preparation on a binary mask.

    Reused logic from ``compute_features._prepare_mask_for_projection`` but
    kept here so E2.5 has no import cycle on a private function. Encoder steps:

    1. ``spatial_op``: ``"none"`` (no-op), ``"crop"`` (slice by crop_offset),
       or ``"resize"`` (nearest-neighbour to working_shape).
    2. Zero-pad on the trailing edge of each axis to ``padded_shape``.
    """
    if spatial_op == "none":
        m = mask
    elif spatial_op == "crop":
        sl = tuple(slice(o, o + s) for o, s in zip(crop_offset, working_shape))
        m = mask[sl]
    elif spatial_op == "resize":
        import torch.nn.functional as F  # local import; keeps top-level light

        t = torch.from_numpy(mask.astype(np.uint8))[None, None].float()
        m = F.interpolate(t, size=working_shape, mode="nearest")[0, 0].numpy().astype(bool)
    else:
        raise ValueError(f"unknown spatial_op: {spatial_op!r}")

    pads = tuple((0, p - w) for w, p in zip(working_shape, padded_shape))
    if any(p[1] != 0 for p in pads):
        m = np.pad(m.astype(bool), pads, mode="constant", constant_values=False)
    return m.astype(bool)


def dilate_mask_3d(mask: np.ndarray, radius: int) -> np.ndarray:
    """3D binary dilation by ``radius`` voxels via a single max-pool."""
    if radius <= 0:
        return mask.astype(bool)
    t = torch.from_numpy(mask.astype(np.uint8))[None, None].float()
    k = 2 * radius + 1
    import torch.nn.functional as F

    pooled = F.max_pool3d(t, kernel_size=k, stride=1, padding=radius)
    return pooled[0, 0].numpy().astype(bool)


def project_to_latent_grid(
    mask_native: np.ndarray,
    *,
    meta: LatentMetadata,
    device: str = "cpu",
) -> np.ndarray:
    """Native-resolution boolean mask → boolean mask on the latent grid (H', W', D')."""
    padded = prepare_mask_for_projection(
        mask_native,
        spatial_op=meta.spatial_op,
        working_shape=meta.working_spatial_shape,
        padded_shape=meta.padded_spatial_shape,
        crop_offset=meta.crop_offset,
    )
    t = torch.from_numpy(padded.astype(np.uint8)).to(device)
    m_lat = project_mask_to_latent(t, stride=meta.stride)
    return m_lat[0, 0].cpu().numpy().astype(bool)


def mask_pool_per_modality(
    z: np.ndarray,
    mask_latent: np.ndarray,
) -> np.ndarray:
    """Mask-pool ``z`` of shape ``(M, C, H', W', D')`` over ``mask_latent``.

    Returns an ``(M, C)`` array of per-modality, per-channel means inside the
    mask. If the mask is empty, returns zeros (matching the E2.1 convention,
    where empty-mask scans are flagged separately via ``mask_lat_present``).
    """
    if not mask_latent.any():
        return np.zeros(z.shape[:2], dtype=np.float64)
    z_t = torch.from_numpy(z)  # (M, C, H', W', D')
    m_t = torch.from_numpy(mask_latent.astype(np.uint8))[None, None]  # (1, 1, ...)
    out = np.zeros(z.shape[:2], dtype=np.float64)
    for mi in range(z_t.shape[0]):
        pooled = mask_anchored_mean(z_t[mi : mi + 1].float(), m_t)  # (1, C)
        out[mi] = pooled.cpu().numpy()[0]
    return out
