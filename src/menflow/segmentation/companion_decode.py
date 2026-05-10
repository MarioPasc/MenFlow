"""Decode the non-t1c companion modalities for an E2.4 anchor.

E2.4 Phase A only steered + decoded the t1c channel. The BraTS multimodal
segmenter needs t1n/t2f/t2w as well. To keep the segmenter input domain-
consistent (all channels are MAISI-decoded, not a real/synthetic mix), this
module decodes the missing modalities at Δ=0 (no steering) once per anchor
and writes them with BraTS-style filenames so the segmenter can ingest them
verbatim.

The function is split off from
``routines.steer_decode.engine.steer_decode_engine`` because it only needs
the encoder-side latents and a much simpler iteration pattern.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

from menflow.maisi_autoencoder.config import MaisiV2Config
from menflow.maisi_autoencoder.model import MaisiAutoencoder
from menflow.maisi_autoencoder.transforms import PercentileNormalizer

logger = logging.getLogger(__name__)


def decode_companion_modalities(
    *,
    latents_h5: Path,
    anchor_index: int,
    scan_id: str,
    modality_indices: Sequence[int],
    modality_names: Sequence[str],
    output_dir: Path,
    checkpoint: Path,
    model_config: MaisiV2Config | None = None,
    dtype: str = "float16",
    device: str = "cuda",
    intensity_rescale: bool = True,
) -> dict[str, Path]:
    """Decode each requested modality of a single anchor at Δ=0.

    Parameters
    ----------
    latents_h5
        Self-describing latents H5 (``brats_men_maisi_latents.h5``).
    anchor_index
        Row index in the latents H5.
    scan_id
        BraTS-style scan identifier (used in output filenames).
    modality_indices
        Which columns of ``latents`` to decode (e.g. ``[1, 2, 3]`` for
        t1n/t2f/t2w when t1c is at column 0).
    modality_names
        Names matching ``modality_indices``; must be the same length.
    output_dir
        Per-anchor output directory. Created if absent. NIfTIs land at
        ``output_dir / f"{scan_id}-{modality}.nii.gz"``.
    checkpoint
        MAISI-v2 checkpoint path (``autoencoder_v2.pt``).
    model_config
        Optional :class:`MaisiV2Config` override.
    dtype, device
        Forwarded to :meth:`MaisiAutoencoder.from_checkpoint`.
    intensity_rescale
        If ``True`` (default), invert the per-scan percentile rescale stored
        in the latents H5 so the saved NIfTI is in source intensity units.

    Returns
    -------
    dict[str, pathlib.Path]
        ``{modality_name: nifti_path}`` for each successfully decoded modality.
    """
    if len(modality_indices) != len(modality_names):
        raise ValueError(
            f"modality_indices ({len(modality_indices)}) and modality_names "
            f"({len(modality_names)}) must have the same length"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MaisiAutoencoder.from_checkpoint(
        checkpoint, config=model_config, device=device, dtype=dtype
    )
    torch_device = torch.device(device)
    torch_dtype = next(model.parameters()).dtype

    out_paths: dict[str, Path] = {}
    with h5py.File(latents_h5, "r") as lat:
        modalities_in_lat = [_decode(m) for m in lat.attrs["modalities"]]
        source_shape = tuple(int(x) for x in lat.attrs["source_spatial_shape"])
        spacing = tuple(float(x) for x in lat.attrs.get("spacing_mm", (1.0, 1.0, 1.0)))
        affine = np.eye(4, dtype=np.float64)
        for axis, sp in enumerate(spacing):
            affine[axis, axis] = sp

        for m_idx, m_name in zip(modality_indices, modality_names, strict=True):
            if not 0 <= m_idx < len(modalities_in_lat):
                raise IndexError(
                    f"modality_index {m_idx} out of range for {len(modalities_in_lat)} modalities"
                )
            recorded = modalities_in_lat[m_idx]
            if recorded != m_name:
                logger.warning(
                    "modality name mismatch: latents H5 says %r at index %d, "
                    "caller passed %r — using caller's name in filename",
                    recorded,
                    m_idx,
                    m_name,
                )

            z_np = lat["latents"][anchor_index, m_idx].astype(np.float32)
            z_t = torch.from_numpy(z_np).to(device=torch_device, dtype=torch_dtype)
            x_pad = model.decode(z_t[None])[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
            sl = tuple(slice(0, s) for s in source_shape)
            x_cropped = x_pad[sl]

            if intensity_rescale:
                lower = float(lat["intensity_lower"][anchor_index, m_idx])
                upper = float(lat["intensity_upper"][anchor_index, m_idx])
                normalizer = PercentileNormalizer(
                    lower_value=lower, upper_value=upper, b_min=0.0, b_max=1.0
                )
                x_image = normalizer.inverse(x_cropped).astype(np.float32, copy=False)
            else:
                x_image = x_cropped.astype(np.float32, copy=False)

            nifti_path = output_dir / f"{scan_id}-{m_name}.nii.gz"
            nib.save(nib.Nifti1Image(x_image, affine), str(nifti_path))
            out_paths[m_name] = nifti_path

            del z_t, x_pad, x_cropped, x_image
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    logger.info(
        "Decoded %d companion modalities for %s -> %s",
        len(out_paths),
        scan_id,
        output_dir,
    )
    return out_paths


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U", "O"}:
        try:
            return _decode(value.item())
        except Exception:  # pragma: no cover
            return str(value)
    return str(value)
