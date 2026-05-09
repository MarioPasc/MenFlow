"""Decode a self-describing latent HDF5 back to a MenFlow-schema reconstruction.

The output file conforms to the unified MenFlow schema produced by the dataset
converters: every per-scan / per-modality reconstruction sits in the same
``/images`` slot as the original cohort, with the same scan ordering,
``/longitudinal`` CSR layout, and ``/metadata`` group. The schema validator runs
on close so a non-conformant file cannot be written.

Per-scan workflow
-----------------

1. Read latent ``(M, C, H', W', D')`` from the latent H5.
2. For each modality independently:
   * decode latent to a padded volume,
   * crop back to ``working_spatial_shape``,
   * inverse-percentile-rescale using the stored bounds,
   * if ``spatial_op == "resize"`` upsample back to ``source_spatial_shape``.
3. Stack modalities and write to ``/images[i]`` in the output H5.

Spatial-op semantics
--------------------

* ``"none"`` — output spatial shape == source.
* ``"resize"`` — output spatial shape == source (recon is upsampled back).
* ``"crop"`` — output spatial shape == working (cropped-out regions cannot be
  reconstructed). Segmentations and other source-shape datasets are *also*
  cropped to ``working_spatial_shape`` using the recorded ``crop_offset`` so the
  output remains schema-compliant.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from monai.transforms import Resize
from tqdm import tqdm

from menflow.data.h5_schema import SCHEMA_VERSION, assert_h5_valid
from menflow.maisi_autoencoder.config import MaisiV2Config
from menflow.maisi_autoencoder.model import MaisiAutoencoder
from menflow.maisi_autoencoder.transforms import (
    PercentileNormalizer,
    build_postprocess,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routine config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecodeRoutineConfig:
    """Top-level config for the decode routine."""

    latent_h5: Path
    output_h5: Path
    checkpoint: Path
    source_h5: Path | None = None  # optional override; else read from latent attrs
    max_scans: int | None = None
    compression_level: int = 4
    output_dtype: str = "float32"
    log_level: str = "INFO"
    model: MaisiV2Config = field(default_factory=MaisiV2Config)
    device: str = "cuda"
    dtype: str = "float32"

    @classmethod
    def from_yaml(cls, path: str | Path) -> DecodeRoutineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("slurm", None)
        model = MaisiV2Config(**(raw.pop("model", {}) or {}))
        for path_key in ("latent_h5", "output_h5", "checkpoint", "source_h5"):
            if path_key in raw and raw[path_key] is not None:
                raw[path_key] = Path(raw[path_key]).expanduser()
        return cls(model=model, **raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DecodeEngine:
    """Drive a full decode pass and emit a schema-compliant MenFlow H5."""

    def __init__(self, config: DecodeRoutineConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        if not cfg.latent_h5.is_file():
            raise FileNotFoundError(cfg.latent_h5)
        if not cfg.checkpoint.is_file():
            raise FileNotFoundError(cfg.checkpoint)

        with h5py.File(cfg.latent_h5, "r") as lat:
            geom = _LatentGeometry.from_attrs(lat)
            n_scans_total = int(lat.attrs["n_scans"])
            n_scans = n_scans_total if cfg.max_scans is None else min(cfg.max_scans, n_scans_total)

            source_h5_path = cfg.source_h5 or Path(_decode(lat.attrs["source_h5"]))
            if not source_h5_path.is_file():
                raise FileNotFoundError(
                    f"Source H5 not found: {source_h5_path}. Set source_h5 in the decode config."
                )

            output_spatial_shape = (
                geom.working_shape if geom.spatial_op == "crop" else geom.source_shape
            )

            logger.info(
                "spatial_op=%s | latent (%d,%s) -> padded %s -> working %s -> output %s",
                geom.spatial_op,
                geom.latent_channels,
                geom.latent_spatial,
                geom.padded_shape,
                geom.working_shape,
                output_spatial_shape,
            )

            model = MaisiAutoencoder.from_checkpoint(
                cfg.checkpoint,
                config=cfg.model,
                device=cfg.device,
                dtype=cfg.dtype,
            )
            torch_device = torch.device(cfg.device)
            torch_dtype = next(model.parameters()).dtype

            resizer_to_source = (
                Resize(
                    spatial_size=geom.source_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                if geom.spatial_op == "resize"
                else None
            )

            with (
                h5py.File(source_h5_path, "r") as src,
                h5py.File(cfg.output_h5, "w") as out,
            ):
                _initialise_output(
                    out,
                    src=src,
                    geom=geom,
                    n_scans=n_scans,
                    output_spatial_shape=output_spatial_shape,
                    output_dtype=cfg.output_dtype,
                    compression_level=cfg.compression_level,
                    decode_provenance={
                        "decoder_name": "MAISI-v2",
                        "decoder_checkpoint": str(cfg.checkpoint),
                        "latent_h5": str(cfg.latent_h5),
                        "decoded_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    },
                )

                pad_widths = tuple(
                    (0, geom.padded_shape[i] - geom.working_shape[i]) for i in range(3)
                )
                modalities = geom.modalities

                for i in tqdm(range(n_scans), desc="Decoding"):
                    latent = lat["latents"][i]
                    lower = lat["intensity_lower"][i]
                    upper = lat["intensity_upper"][i]

                    recon_modalities: list[np.ndarray] = []
                    for m_idx in range(len(modalities)):
                        z = torch.from_numpy(np.asarray(latent[m_idx])).to(
                            device=torch_device, dtype=torch_dtype
                        )[None]
                        x_hat = (
                            model.decode(z)[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
                        )

                        normalizer = PercentileNormalizer(
                            lower_value=float(lower[m_idx]),
                            upper_value=float(upper[m_idx]),
                            b_min=0.0,
                            b_max=1.0,
                        )
                        recon = build_postprocess(
                            x_hat,
                            normalizer=normalizer,
                            original_shape=geom.working_shape,
                            pad_widths=pad_widths,
                        )
                        if resizer_to_source is not None:
                            recon = np.asarray(resizer_to_source(recon[np.newaxis]))[0]
                        recon_modalities.append(recon)
                        del z, x_hat

                    out["images"][i] = np.stack(recon_modalities, axis=0).astype(
                        cfg.output_dtype, copy=False
                    )

        size_gb = cfg.output_h5.stat().st_size / (1024**3)
        logger.info("Wrote %s (%.2f GB)", cfg.output_h5, size_gb)
        assert_h5_valid(cfg.output_h5)
        logger.info("Schema validation passed.")
        return cfg.output_h5


# ---------------------------------------------------------------------------
# Latent geometry helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LatentGeometry:
    modalities: tuple[str, ...]
    source_shape: tuple[int, int, int]
    working_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    latent_spatial: tuple[int, int, int]
    latent_channels: int
    spatial_op: str
    crop_offset: tuple[int, int, int]

    @classmethod
    def from_attrs(cls, lat: h5py.File) -> _LatentGeometry:
        modalities = tuple(_decode(m) for m in lat.attrs["modalities"])
        source_shape = tuple(int(x) for x in lat.attrs["source_spatial_shape"])
        working_shape = tuple(int(x) for x in lat.attrs["working_spatial_shape"])
        padded_shape = tuple(int(x) for x in lat.attrs["padded_spatial_shape"])
        latent_spatial = tuple(int(x) for x in lat.attrs["latent_spatial_shape"])
        latent_channels = int(lat.attrs["latent_channels"])
        spatial_op = _decode(lat.attrs["spatial_op"])
        crop_offset = tuple(int(x) for x in lat.attrs["crop_offset"])
        return cls(
            modalities=modalities,
            source_shape=source_shape,
            working_shape=working_shape,
            padded_shape=padded_shape,
            latent_spatial=latent_spatial,
            latent_channels=latent_channels,
            spatial_op=spatial_op,
            crop_offset=crop_offset,
        )


# ---------------------------------------------------------------------------
# Output H5 initialisation
# ---------------------------------------------------------------------------


def _initialise_output(
    out: h5py.File,
    *,
    src: h5py.File,
    geom: _LatentGeometry,
    n_scans: int,
    output_spatial_shape: tuple[int, int, int],
    output_dtype: str,
    compression_level: int,
    decode_provenance: dict,
) -> None:
    """Create the unified-schema skeleton in *out*, copying from *src*.

    For ``spatial_op == 'crop'``, source-shape datasets (``segmentations``) are
    cropped using the recorded ``crop_offset`` so they match
    ``output_spatial_shape``.
    """
    src_n_scans = int(src.attrs["n_scans"])
    if n_scans > src_n_scans:
        raise ValueError(f"n_scans={n_scans} > source n_scans={src_n_scans}")

    H, W, D = output_spatial_shape
    M = len(geom.modalities)
    vlen = h5py.special_dtype(vlen=str)

    src_offsets = src["longitudinal/patient_offsets"][:].astype(int)
    src_patient_list = [_decode(s) for s in src["longitudinal/patient_list"][:]]
    if n_scans == int(src_offsets[-1]):
        new_offsets = src_offsets.astype(np.int32)
        new_patient_list = src_patient_list
    else:
        n_patients_kept = int(np.searchsorted(src_offsets, n_scans, side="left"))
        if int(src_offsets[n_patients_kept]) != n_scans:
            n_patients_kept += 1
        new_offsets = src_offsets[: n_patients_kept + 1].copy()
        new_offsets[-1] = n_scans
        new_offsets = new_offsets.astype(np.int32)
        new_patient_list = src_patient_list[:n_patients_kept]

    n_patients_kept = len(new_patient_list)

    # --- Root attrs ---
    out.attrs["schema_version"] = SCHEMA_VERSION
    out.attrs["dataset_name"] = _decode(src.attrs["dataset_name"]) + "-MAISIrecon"
    out.attrs["dataset_type"] = _decode(src.attrs["dataset_type"])
    out.attrs["n_scans"] = np.int64(n_scans)
    out.attrs["n_patients"] = np.int64(n_patients_kept)
    out.attrs["modalities"] = np.asarray(list(geom.modalities), dtype=object)
    out.attrs["n_modalities"] = np.int64(M)
    out.attrs["spatial_shape"] = np.asarray([H, W, D], dtype=np.int64)
    out.attrs["spacing_mm"] = np.asarray(src.attrs["spacing_mm"], dtype=np.float64)
    out.attrs["orientation"] = _decode(src.attrs["orientation"])
    out.attrs["label_map"] = _decode(src.attrs["label_map"])
    out.attrs["intensity_normalized"] = False
    out.attrs["created_at"] = _dt.datetime.now(_dt.UTC).isoformat()
    out.attrs["has_segmentation_any"] = bool(np.any(src["has_segmentation"][:n_scans]))
    out.attrs["spatial_op"] = geom.spatial_op
    out.attrs["crop_offset"] = np.asarray(geom.crop_offset, dtype=np.int64)
    for k, v in decode_provenance.items():
        out.attrs[k] = v

    # --- Datasets ---
    out.create_dataset(
        "images",
        shape=(n_scans, M, H, W, D),
        dtype=output_dtype,
        chunks=(1, M, H, W, D),
        compression="gzip",
        compression_opts=compression_level,
    )

    # Stream-copy segmentations one scan at a time. Loading
    # src["segmentations"][:n_scans] in one shot would peak at
    # n_scans*H*W*D bytes (≈10 GB for BraTS-MEN); h5py reads + the gzip
    # chunked write per row keeps peak RAM at a single segmentation (~9 MB).
    seg_ds = out.create_dataset(
        "segmentations",
        shape=(n_scans, H, W, D),
        dtype="int8",
        chunks=(1, H, W, D),
        compression="gzip",
        compression_opts=compression_level,
    )
    crop_slices = (
        tuple(slice(o, o + t) for o, t in zip(geom.crop_offset, output_spatial_shape))
        if geom.spatial_op == "crop"
        else None
    )
    for i in range(n_scans):
        s = src["segmentations"][i]
        if crop_slices is not None:
            s = s[crop_slices]
        seg_ds[i] = s

    out.create_dataset("has_segmentation", data=src["has_segmentation"][:n_scans])
    out.create_dataset("scan_ids", data=src["scan_ids"][:n_scans], dtype=vlen)
    out.create_dataset("patient_ids", data=src["patient_ids"][:n_scans], dtype=vlen)
    out.create_dataset("timepoint_idx", data=src["timepoint_idx"][:n_scans])

    long_grp = out.create_group("longitudinal")
    long_grp.create_dataset("patient_offsets", data=new_offsets)
    long_grp.create_dataset(
        "patient_list", data=np.asarray(new_patient_list, dtype=object), dtype=vlen
    )

    meta_grp = out.create_group("metadata")
    if "metadata" in src:
        for name, dataset in src["metadata"].items():
            meta_grp.create_dataset(
                name,
                data=dataset[:n_scans] if dataset.ndim >= 1 else dataset[()],
                dtype=dataset.dtype if dataset.dtype.kind != "O" else vlen,
            )

    if "splits" in src:
        splits_grp = out.create_group("splits")
        for name, dataset in src["splits"].items():
            mask = dataset[:] < n_patients_kept
            splits_grp.create_dataset(name, data=dataset[:][mask].astype(np.int32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
