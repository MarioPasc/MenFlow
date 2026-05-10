"""Encode a unified MenFlow HDF5 cohort into a self-describing latent HDF5.

Each scan is encoded one modality at a time as a single-channel volume; the
per-modality latents are stacked along the modality axis. Per-volume intensity
percentiles and the spatial preparation (resize / crop / none) are recorded so
:mod:`routines.decode` can reverse the operations exactly.

Spatial operation
-----------------

The engine supports three spatial preparation strategies, controlled by
``inference.spatial_op``:

* ``"none"`` (default for Picasso) — encode at the cohort's native shape; only
  pad to ``pad_multiple``.
* ``"crop"`` — center-crop the volume to ``target_spatial_shape`` before pad.
  Voxel spacing is preserved, so tumor-volume metrics remain valid (the cropped
  region simply contains less anatomy at the edges).
* ``"resize"`` — trilinear resize to ``target_spatial_shape``. Changes effective
  voxel volume; reserved for legacy smoke configs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from monai.transforms import Resize
from tqdm import tqdm

from menflow.maisi_autoencoder.config import InferenceConfig, MaisiV2Config
from menflow.maisi_autoencoder.latents_h5 import LatentH5Writer
from menflow.maisi_autoencoder.model import MaisiAutoencoder
from menflow.maisi_autoencoder.transforms import build_preprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routine config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodeRoutineConfig:
    """Top-level config for the encode routine. Built from a YAML file."""

    source_h5: Path
    output_h5: Path
    checkpoint: Path
    modalities: tuple[str, ...] | None = None
    max_scans: int | None = None
    latent_dtype: str = "float16"
    compression_level: int = 4
    log_level: str = "INFO"
    model: MaisiV2Config = field(default_factory=MaisiV2Config)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EncodeRoutineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        # The YAML may carry a `slurm:` block consumed by launcher.sh; ignore it
        # at the routine level.
        raw.pop("slurm", None)
        model = MaisiV2Config(**(raw.pop("model", {}) or {}))
        inf_raw = raw.pop("inference", {}) or {}
        if "target_spatial_shape" in inf_raw and inf_raw["target_spatial_shape"] is not None:
            inf_raw["target_spatial_shape"] = tuple(inf_raw["target_spatial_shape"])
        inference = InferenceConfig(**inf_raw)
        for path_key in ("source_h5", "output_h5", "checkpoint"):
            if path_key in raw:
                raw[path_key] = Path(raw[path_key]).expanduser()
        if raw.get("modalities") is not None:
            raw["modalities"] = tuple(raw["modalities"])
        return cls(model=model, inference=inference, **raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EncodeEngine:
    """Drive a full encode pass over a MenFlow H5 cohort."""

    def __init__(self, config: EncodeRoutineConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        if _latent_already_encoded(cfg.output_h5):
            size_mb = cfg.output_h5.stat().st_size / (1024**2)
            logger.info(
                "Skipping encode: output already exists at %s (%.1f MB). "
                "Delete the file to force re-encoding.",
                cfg.output_h5,
                size_mb,
            )
            return cfg.output_h5
        if not cfg.source_h5.is_file():
            raise FileNotFoundError(cfg.source_h5)
        if not cfg.checkpoint.is_file():
            raise FileNotFoundError(cfg.checkpoint)
        _prepare_output_path(cfg.output_h5)

        with h5py.File(cfg.source_h5, "r") as src:
            (
                modalities,
                modality_indices,
                source_shape,
                working_shape,
                crop_offset,
                padded_shape,
            ) = self._resolve_geometry(src)

            n_scans = self._resolve_n_scans(src)
            d = cfg.model.downsampling_factor
            latent_spatial = tuple(s // d for s in padded_shape)
            latent_channels = cfg.model.latent_channels

            scan_ids = [_decode(s) for s in src["scan_ids"][:n_scans]]
            patient_ids = [_decode(s) for s in src["patient_ids"][:n_scans]]
            timepoint_idx = src["timepoint_idx"][:n_scans].astype(np.int32)

            patient_list, patient_offsets = _truncated_csr(src, n_scans)
            split_arrays = _truncated_splits(src, n_patients_kept=len(patient_list))

            self._log_geometry(
                source_shape, working_shape, padded_shape, latent_spatial, latent_channels
            )

            model = MaisiAutoencoder.from_checkpoint(
                cfg.checkpoint,
                config=cfg.model,
                device=cfg.inference.device,
                dtype=cfg.inference.dtype,
            )
            torch_device = torch.device(cfg.inference.device)
            torch_dtype = next(model.parameters()).dtype

            resizer = (
                Resize(spatial_size=working_shape, mode="trilinear", align_corners=False)
                if cfg.inference.spatial_op == "resize"
                else None
            )

            with LatentH5Writer(
                cfg.output_h5,
                dataset_name=_decode(src.attrs["dataset_name"]),
                dataset_type=_decode(src.attrs["dataset_type"]),
                modalities=modalities,
                orientation=_decode(src.attrs["orientation"]),
                spacing_mm=tuple(float(x) for x in src.attrs["spacing_mm"]),
                patient_list=patient_list,
                patient_offsets=patient_offsets,
                n_scans=n_scans,
                source_spatial_shape=source_shape,
                working_spatial_shape=working_shape,
                padded_spatial_shape=padded_shape,
                latent_spatial_shape=latent_spatial,
                latent_channels=latent_channels,
                spatial_op=cfg.inference.spatial_op,
                crop_offset=crop_offset,
                source_h5=cfg.source_h5,
                encoder_checkpoint=cfg.checkpoint,
                encoder_config=cfg.model,
                inference_config=cfg.inference,
                deterministic=cfg.inference.deterministic,
                latent_dtype=cfg.latent_dtype,
                compression_level=cfg.compression_level,
            ) as writer:
                dataset_name = _decode(src.attrs["dataset_name"])
                for i in tqdm(range(n_scans), desc=f"Encoding {dataset_name}"):
                    image = src["images"][i]  # (M_src, H, W, D)
                    per_modality_latents: list[np.ndarray] = []
                    lower_vals: list[float] = []
                    upper_vals: list[float] = []

                    for m_idx in modality_indices:
                        vol = self._prepare_volume(
                            image[m_idx],
                            source_shape=source_shape,
                            working_shape=working_shape,
                            crop_offset=crop_offset,
                            resizer=resizer,
                        )

                        padded, normalizer, _orig, _pads = build_preprocess(
                            vol,
                            lower_percentile=cfg.inference.intensity_lower_percentile,
                            upper_percentile=cfg.inference.intensity_upper_percentile,
                            b_min=cfg.inference.intensity_b_min,
                            b_max=cfg.inference.intensity_b_max,
                            pad_multiple=cfg.inference.pad_multiple,
                        )
                        if padded.shape != tuple(padded_shape):
                            raise RuntimeError(
                                f"padded shape {padded.shape} != expected {padded_shape}; "
                                "check pad_multiple vs spatial_op interaction"
                            )

                        x = torch.from_numpy(padded).to(device=torch_device, dtype=torch_dtype)[
                            None, None
                        ]
                        z = model.encode(x, deterministic=cfg.inference.deterministic)
                        per_modality_latents.append(
                            z[0].detach().to("cpu", dtype=torch.float32).numpy()
                        )
                        lower_vals.append(normalizer.lower_value)
                        upper_vals.append(normalizer.upper_value)
                        del x, z

                    writer.write_scan(
                        i,
                        scan_id=scan_ids[i],
                        patient_id=patient_ids[i],
                        timepoint=int(timepoint_idx[i]),
                        latent=np.stack(per_modality_latents, axis=0),
                        intensity_lower=lower_vals,
                        intensity_upper=upper_vals,
                    )

                for split_name, split_indices in split_arrays.items():
                    writer.write_split(split_name, split_indices)

        size_mb = cfg.output_h5.stat().st_size / (1024**2)
        logger.info("Encoded %d scans -> %s (%.1f MB)", n_scans, cfg.output_h5, size_mb)
        return cfg.output_h5

    # ------------------------------------------------------------------
    # Geometry / source-shape resolution
    # ------------------------------------------------------------------

    def _resolve_geometry(
        self, src: h5py.File
    ) -> tuple[
        tuple[str, ...],
        list[int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ]:
        cfg = self.config
        source_modalities = tuple(_decode(m) for m in src.attrs["modalities"])
        modalities = cfg.modalities or source_modalities
        for m in modalities:
            if m not in source_modalities:
                raise ValueError(f"requested modality {m!r} not in source ({source_modalities})")
        modality_indices = [source_modalities.index(m) for m in modalities]
        source_shape = tuple(int(x) for x in src.attrs["spatial_shape"])

        op = cfg.inference.spatial_op
        target = cfg.inference.target_spatial_shape
        if op == "none":
            working_shape = source_shape
            crop_offset = (0, 0, 0)
        elif op == "resize":
            working_shape = tuple(int(x) for x in target)  # type: ignore[arg-type]
            crop_offset = (0, 0, 0)
        elif op == "crop":
            target_t = tuple(int(x) for x in target)  # type: ignore[arg-type]
            for axis, (s, t) in enumerate(zip(source_shape, target_t)):
                if t > s:
                    raise ValueError(f"crop target axis {axis} ({t}) exceeds source ({s})")
            crop_offset = tuple((s - t) // 2 for s, t in zip(source_shape, target_t))
            working_shape = target_t
        else:  # pragma: no cover — guarded in InferenceConfig
            raise ValueError(f"unknown spatial_op: {op}")

        padded_shape = tuple(
            _ceil_to_multiple(s, cfg.inference.pad_multiple) for s in working_shape
        )
        return modalities, modality_indices, source_shape, working_shape, crop_offset, padded_shape

    def _resolve_n_scans(self, src: h5py.File) -> int:
        total = int(src.attrs["n_scans"])
        return total if self.config.max_scans is None else min(self.config.max_scans, total)

    def _prepare_volume(
        self,
        vol: np.ndarray,
        *,
        source_shape: tuple[int, int, int],
        working_shape: tuple[int, int, int],
        crop_offset: tuple[int, int, int],
        resizer: Resize | None,
    ) -> np.ndarray:
        """Apply spatial_op (none / crop / resize) to one volume."""
        op = self.config.inference.spatial_op
        if op == "none":
            return vol
        if op == "crop":
            sl = tuple(slice(o, o + s) for o, s in zip(crop_offset, working_shape))
            return vol[sl]
        if op == "resize":
            assert resizer is not None
            resized = resizer(vol[np.newaxis])
            arr = resized.numpy() if hasattr(resized, "numpy") else np.asarray(resized)
            return arr[0]
        raise ValueError(op)  # pragma: no cover

    def _log_geometry(
        self,
        source_shape: tuple[int, int, int],
        working_shape: tuple[int, int, int],
        padded_shape: tuple[int, int, int],
        latent_spatial: tuple[int, int, int],
        latent_channels: int,
    ) -> None:
        cfg = self.config
        logger.info(
            "spatial_op=%s | source %s -> working %s -> padded %s -> latent (%d,%s)",
            cfg.inference.spatial_op,
            source_shape,
            working_shape,
            padded_shape,
            latent_channels,
            latent_spatial,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _prepare_output_path(path: Path) -> None:
    """Ensure the parent directory of *path* exists before opening for write."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _latent_already_encoded(path: Path) -> bool:
    """True iff the given path is a non-empty, openable HDF5 file.

    Used to short-circuit ``EncodeEngine.run`` on re-launch so an encode pass
    that already finished is not redone. The check opens the file in read mode
    so a half-written or truncated artifact is not mistaken for a finished one.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def _ceil_to_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _truncated_csr(src: h5py.File, n_scans: int) -> tuple[list[str], np.ndarray]:
    """Slice the source CSR layout to the first *n_scans* scans."""
    src_offsets = src["longitudinal/patient_offsets"][:].astype(int)
    src_patient_list = [_decode(s) for s in src["longitudinal/patient_list"][:]]
    if n_scans == int(src_offsets[-1]):
        return src_patient_list, src_offsets.astype(np.int32)
    n_patients_kept = int(np.searchsorted(src_offsets, n_scans, side="left"))
    if int(src_offsets[n_patients_kept]) != n_scans:
        n_patients_kept += 1
    new_offsets = src_offsets[: n_patients_kept + 1].copy()
    new_offsets[-1] = n_scans
    return src_patient_list[:n_patients_kept], new_offsets.astype(np.int32)


def _truncated_splits(src: h5py.File, *, n_patients_kept: int) -> dict[str, np.ndarray]:
    """Filter source splits to the kept-patient subset."""
    out: dict[str, np.ndarray] = {}
    if "splits" not in src:
        return out
    for name, ds in src["splits"].items():
        idx = ds[:]
        out[name] = idx[idx < n_patients_kept].astype(np.int32)
    return out
