"""Encode a unified MenFlow HDF5 cohort to a provisional latent HDF5.

The engine streams scans one at a time, encodes each modality independently as
a single-channel volume, and writes the per-modality latent stack. Intensity
percentiles are recorded so :mod:`routines.decode` can produce a reconstruction
in the original intensity scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
    modalities: tuple[str, ...] | None = None  # None ⇒ encode all source modalities
    max_scans: int | None = None
    latent_dtype: str = "float16"
    compression_level: int = 4
    log_level: str = "INFO"
    model: MaisiV2Config = field(default_factory=MaisiV2Config)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EncodeRoutineConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        model = MaisiV2Config(**(raw.pop("model", {}) or {}))
        inf_raw = raw.pop("inference", {}) or {}
        if "target_spatial_shape" in inf_raw and inf_raw["target_spatial_shape"] is not None:
            inf_raw["target_spatial_shape"] = tuple(inf_raw["target_spatial_shape"])
        inference = InferenceConfig(**inf_raw)
        for path_key in ("source_h5", "output_h5", "checkpoint"):
            if path_key in raw:
                raw[path_key] = Path(raw[path_key]).expanduser()
        if "modalities" in raw and raw["modalities"] is not None:
            raw["modalities"] = tuple(raw["modalities"])
        return cls(model=model, inference=inference, **raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EncodeEngine:
    """Drive a full encode pass over a MenFlow H5 cohort."""

    def __init__(self, config: EncodeRoutineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.config
        if not cfg.source_h5.is_file():
            raise FileNotFoundError(cfg.source_h5)
        if not cfg.checkpoint.is_file():
            raise FileNotFoundError(cfg.checkpoint)

        with h5py.File(cfg.source_h5, "r") as src:
            source_dataset_name = _decode(src.attrs["dataset_name"])
            source_modalities = tuple(_decode(m) for m in src.attrs["modalities"])
            source_shape = tuple(int(x) for x in src.attrs["spatial_shape"])
            n_scans_total = int(src.attrs["n_scans"])

            modalities = cfg.modalities or source_modalities
            for m in modalities:
                if m not in source_modalities:
                    raise ValueError(
                        f"requested modality {m!r} not in source ({source_modalities})"
                    )
            modality_indices = [source_modalities.index(m) for m in modalities]

            scan_ids = [_decode(s) for s in src["scan_ids"][:]]
            patient_ids = [_decode(s) for s in src["patient_ids"][:]]
            timepoint_idx = src["timepoint_idx"][:].astype(np.int32)

            n_scans = n_scans_total if cfg.max_scans is None else min(
                cfg.max_scans, n_scans_total
            )

            # Determine working spatial shape
            inf = cfg.inference
            working_shape: tuple[int, int, int]
            if inf.target_spatial_shape is not None:
                working_shape = tuple(int(x) for x in inf.target_spatial_shape)
            else:
                working_shape = source_shape

            padded_shape = tuple(
                _ceil_to_multiple(s, inf.pad_multiple) for s in working_shape
            )
            d = cfg.model.downsampling_factor
            latent_spatial = tuple(s // d for s in padded_shape)
            latent_shape = (cfg.model.latent_channels, *latent_spatial)

            logger.info("Source H5: %s (%s)", cfg.source_h5, source_dataset_name)
            logger.info("Modalities to encode: %s", modalities)
            logger.info(
                "Source shape %s -> working %s -> padded %s -> latent %s",
                source_shape, working_shape, padded_shape, latent_shape,
            )

            # Load model
            model = MaisiAutoencoder.from_checkpoint(
                cfg.checkpoint,
                config=cfg.model,
                device=inf.device,
                dtype=inf.dtype,
            )
            torch_device = torch.device(inf.device)
            torch_dtype = next(model.parameters()).dtype

            resizer = (
                Resize(spatial_size=working_shape, mode="trilinear", align_corners=False)
                if working_shape != source_shape
                else None
            )

            with LatentH5Writer(
                cfg.output_h5,
                n_scans=n_scans,
                modalities=modalities,
                latent_shape=latent_shape,
                source_h5=cfg.source_h5,
                source_dataset_name=source_dataset_name,
                source_spatial_shape=source_shape,
                encoder_checkpoint=cfg.checkpoint,
                encoder_config=cfg.model,
                inference_config=inf,
                deterministic=inf.deterministic,
                latent_dtype=cfg.latent_dtype,
                compression_level=cfg.compression_level,
            ) as writer:
                for i in tqdm(range(n_scans), desc=f"Encoding {source_dataset_name}"):
                    image = src["images"][i]  # (M_src, H, W, D)
                    per_modality_latents: list[np.ndarray] = []
                    lower_vals: list[float] = []
                    upper_vals: list[float] = []

                    for m_idx in modality_indices:
                        vol = image[m_idx]
                        if resizer is not None:
                            vol = (
                                resizer(vol[np.newaxis])[0].numpy()  # type: ignore[union-attr]
                                if hasattr(resizer(vol[np.newaxis]), "numpy")
                                else np.asarray(resizer(vol[np.newaxis]))[0]
                            )

                        padded, normalizer, _orig, _pads = build_preprocess(
                            vol,
                            lower_percentile=inf.intensity_lower_percentile,
                            upper_percentile=inf.intensity_upper_percentile,
                            b_min=inf.intensity_b_min,
                            b_max=inf.intensity_b_max,
                            pad_multiple=inf.pad_multiple,
                        )
                        x = torch.from_numpy(padded).to(
                            device=torch_device, dtype=torch_dtype
                        )[None, None]  # (1, 1, H', W', D')

                        z = model.encode(x, deterministic=inf.deterministic)
                        per_modality_latents.append(
                            z[0].detach().to("cpu", dtype=torch.float32).numpy()
                        )
                        lower_vals.append(normalizer.lower_value)
                        upper_vals.append(normalizer.upper_value)
                        del x, z

                    latent_stack = np.stack(per_modality_latents, axis=0)  # (M, C, H', W', D')
                    writer.write_scan(
                        i,
                        scan_id=scan_ids[i],
                        patient_id=patient_ids[i],
                        timepoint=int(timepoint_idx[i]),
                        latent=latent_stack,
                        intensity_lower=lower_vals,
                        intensity_upper=upper_vals,
                    )

        size_mb = cfg.output_h5.stat().st_size / (1024 ** 2)
        logger.info(
            "Encoded %d scans -> %s (%.1f MB)", n_scans, cfg.output_h5, size_mb
        )
        return cfg.output_h5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _ceil_to_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m
