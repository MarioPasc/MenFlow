"""Compute pooled MAISI-v2 latent features for the E2 chain.

Inputs
------
- ``latent_h5``: a self-describing latent HDF5 (``menflow.maisi_autoencoder.latents_h5``)
  with shape ``(N, M, C, H', W', D')``.
- ``source_h5``: the unified MenFlow HDF5 with ``/segmentations`` of shape
  ``(N, H, W, D)`` aligned to the same scan order as the latent file (both
  emitted from the same encode pass).

Outputs
-------
A small features H5 that the downstream E2.X experiments consume directly:

::

    /
    ├── attrs:
    │   ├── schema_version       "0.1"
    │   ├── source_latents_h5    str
    │   ├── source_h5            str
    │   ├── modality             str   (e.g. "t1c")
    │   ├── modality_index       int   (column of /latents along axis=1)
    │   ├── pooling_seed         int
    │   ├── compute_z_full       bool
    │   ├── created_at           ISO-8601
    │   ├── dataset_name         str
    │   ├── n_scans, n_patients, latent_channels, spacing_mm, latent_spatial_shape
    │   └── label_set_used       int_array (label values treated as "tumor")
    ├── z_tumor              (N, C) float32
    ├── z_global             (N, C) float32
    ├── z_random             (N, C) float32
    ├── log_volume           (N,)   float32   = log(prod(spacing) * n_voxels / 1000) cm³
    ├── n_voxels_tumor       (N,)   int32     voxel count *at source resolution*
    ├── n_voxels_lat_tumor   (N,)   int32     voxel count at latent resolution
    ├── mask_lat_present     (N,)   bool
    ├── scan_ids             (N,)   vlen str
    ├── patient_ids          (N,)   vlen str
    └── timepoint_idx        (N,)   int32

If ``compute_z_full=True`` (toggle for E2.3/E2.4) the file additionally contains
``z_full (N, C, H', W', D')`` and ``mask_lat (N, 1, H', W', D')``. Disabled by
default to keep the artifact small (~150 KB for N=1141 with the toggle off).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from menflow.latent_features.mask import project_mask_to_latent
from menflow.latent_features.pooling import (
    global_mean,
    mask_anchored_mean,
    random_region_mean,
)
from menflow.latent_features.volume import voxel_count_to_log_volume_cm3

logger = logging.getLogger(__name__)


FEATURES_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class ComputeFeaturesRoutineConfig:
    """Routine config; built from a YAML file."""

    latent_h5: Path
    source_h5: Path
    output_h5: Path
    modality: str = "t1c"
    mask_label_set: tuple[int, ...] = (1, 2, 3)
    random_region_seed: int = 0
    compute_z_full: bool = False
    max_scans: int | None = None
    log_level: str = "INFO"
    device: str = "cpu"
    compression_level: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> ComputeFeaturesRoutineConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        raw.pop("slurm", None)
        for path_key in ("latent_h5", "source_h5", "output_h5"):
            if path_key in raw:
                raw[path_key] = Path(raw[path_key]).expanduser()
        if "mask_label_set" in raw and raw["mask_label_set"] is not None:
            raw["mask_label_set"] = tuple(int(x) for x in raw["mask_label_set"])
        return cls(**raw)


class ComputeFeaturesEngine:
    """Drive a single feature-computation pass over a paired latent + source H5."""

    def __init__(self, config: ComputeFeaturesRoutineConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        if not cfg.latent_h5.is_file():
            raise FileNotFoundError(cfg.latent_h5)
        if not cfg.source_h5.is_file():
            raise FileNotFoundError(cfg.source_h5)
        cfg.output_h5.parent.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg.device)

        with h5py.File(cfg.latent_h5, "r") as lat, h5py.File(cfg.source_h5, "r") as src:
            lat_modalities = [_decode(m) for m in lat.attrs["modalities"]]
            if cfg.modality not in lat_modalities:
                raise ValueError(
                    f"requested modality {cfg.modality!r} not in latents H5 "
                    f"(available: {lat_modalities})"
                )
            modality_idx = lat_modalities.index(cfg.modality)

            n_scans_lat = int(lat.attrs["n_scans"])
            n_scans_src = int(src.attrs["n_scans"])
            if n_scans_lat != n_scans_src:
                raise ValueError(
                    f"latent N={n_scans_lat} != source N={n_scans_src}; both files "
                    "must come from the same encode pass."
                )

            n_scans = n_scans_lat if cfg.max_scans is None else min(cfg.max_scans, n_scans_lat)
            c = int(lat.attrs["latent_channels"])
            h_l, w_l, d_l = (int(x) for x in lat.attrs["latent_spatial_shape"])
            spacing_mm = tuple(float(x) for x in src.attrs["spacing_mm"])
            stride = self._infer_stride(src, lat)
            label_set = np.asarray(cfg.mask_label_set, dtype=np.int32)
            spatial_op = _decode(lat.attrs.get("spatial_op", "none"))
            working_shape = tuple(int(x) for x in lat.attrs["working_spatial_shape"])
            padded_shape = tuple(int(x) for x in lat.attrs["padded_spatial_shape"])
            crop_offset = tuple(int(x) for x in lat.attrs.get("crop_offset", (0, 0, 0)))

            self._check_scan_alignment(lat, src, n_scans)

            scan_ids = np.asarray([_decode(s) for s in lat["scan_ids"][:n_scans]], dtype=object)
            patient_ids = np.asarray(
                [_decode(s) for s in lat["patient_ids"][:n_scans]], dtype=object
            )
            timepoint_idx = lat["timepoint_idx"][:n_scans].astype(np.int32)
            has_seg = src["has_segmentation"][:n_scans].astype(bool)

            logger.info(
                "Computing features: n_scans=%d, modality=%s (idx=%d), C=%d, "
                "latent_shape=(%d,%d,%d), stride=%d, compute_z_full=%s",
                n_scans,
                cfg.modality,
                modality_idx,
                c,
                h_l,
                w_l,
                d_l,
                stride,
                cfg.compute_z_full,
            )

            # Allocate output arrays in memory; the artifact is small.
            z_tumor = np.zeros((n_scans, c), dtype=np.float32)
            z_global_arr = np.zeros((n_scans, c), dtype=np.float32)
            z_random_arr = np.zeros((n_scans, c), dtype=np.float32)
            log_volume = np.zeros((n_scans,), dtype=np.float32)
            n_vox_src = np.zeros((n_scans,), dtype=np.int32)
            n_vox_lat = np.zeros((n_scans,), dtype=np.int32)
            mask_lat_present = np.zeros((n_scans,), dtype=bool)

            z_full_arr: np.ndarray | None = None
            mask_lat_arr: np.ndarray | None = None
            if cfg.compute_z_full:
                z_full_arr = np.zeros((n_scans, c, h_l, w_l, d_l), dtype=np.float32)
                mask_lat_arr = np.zeros((n_scans, 1, h_l, w_l, d_l), dtype=bool)

            for i in tqdm(range(n_scans), desc="features"):
                z_i = lat["latents"][i, modality_idx].astype(np.float32)  # (C, H', W', D')
                seg_i = src["segmentations"][i].astype(np.int32)  # (H, W, D)
                tumor_mask_src = np.isin(seg_i, label_set)  # (H, W, D)
                n_src = int(tumor_mask_src.sum())

                # Apply the same spatial preparation the encoder used: optional
                # crop / resize, then zero-pad to padded_spatial_shape with
                # (0, pad) per axis (matches transforms.build_preprocess).
                tumor_mask_padded = _prepare_mask_for_projection(
                    tumor_mask_src,
                    spatial_op=spatial_op,
                    working_shape=working_shape,
                    padded_shape=padded_shape,
                    crop_offset=crop_offset,
                )

                z_t = torch.from_numpy(z_i).to(device).unsqueeze(0)  # (1, C, H', W', D')
                m_t = torch.from_numpy(tumor_mask_padded).to(device)  # (H_pad, W_pad, D_pad)
                m_lat = project_mask_to_latent(m_t, stride=stride)  # (1, 1, H', W', D')
                if m_lat.shape[2:] != z_t.shape[2:]:
                    raise RuntimeError(
                        f"latent shape mismatch at scan {i}: z={tuple(z_t.shape[2:])} "
                        f"vs mask_lat={tuple(m_lat.shape[2:])}"
                    )

                n_lat = int(m_lat.sum().item())
                n_vox_src[i] = n_src
                n_vox_lat[i] = n_lat
                mask_lat_present[i] = bool(n_lat > 0 and has_seg[i])
                log_volume[i] = float(voxel_count_to_log_volume_cm3(n_src, spacing_mm))

                z_tumor[i] = mask_anchored_mean(z_t, m_lat).cpu().numpy()[0]
                z_global_arr[i] = global_mean(z_t).cpu().numpy()[0]
                z_random_arr[i] = (
                    random_region_mean(
                        z_t,
                        torch.tensor([n_lat], dtype=torch.long, device=device),
                        seed=cfg.random_region_seed + i,
                    )
                    .cpu()
                    .numpy()[0]
                )
                if cfg.compute_z_full:
                    assert z_full_arr is not None and mask_lat_arr is not None
                    z_full_arr[i] = z_t.cpu().numpy()[0]
                    mask_lat_arr[i] = m_lat.cpu().numpy()[0]

            self._write(
                cfg=cfg,
                lat_attrs=dict(lat.attrs),
                src_attrs=dict(src.attrs),
                modality_idx=modality_idx,
                spacing_mm=spacing_mm,
                latent_channels=c,
                latent_spatial_shape=(h_l, w_l, d_l),
                label_set=label_set,
                scan_ids=scan_ids,
                patient_ids=patient_ids,
                timepoint_idx=timepoint_idx,
                z_tumor=z_tumor,
                z_global=z_global_arr,
                z_random=z_random_arr,
                log_volume=log_volume,
                n_voxels_tumor=n_vox_src,
                n_voxels_lat_tumor=n_vox_lat,
                mask_lat_present=mask_lat_present,
                z_full=z_full_arr,
                mask_lat=mask_lat_arr,
            )

        size_kb = cfg.output_h5.stat().st_size / 1024
        logger.info("Wrote features H5 -> %s (%.1f KB)", cfg.output_h5, size_kb)
        return cfg.output_h5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_stride(src: h5py.File, lat: h5py.File) -> int:
        src_shape = tuple(int(x) for x in src.attrs["spatial_shape"])
        lat_shape = tuple(int(x) for x in lat.attrs["latent_spatial_shape"])
        padded = tuple(int(x) for x in lat.attrs["padded_spatial_shape"])
        # The encoder downsamples padded_shape by a fixed factor. We project the
        # source mask through the same factor; padded == source for spatial_op=none.
        if padded != src_shape and lat.attrs.get("spatial_op", b"none") == b"none":
            logger.warning(
                "padded_spatial_shape %s differs from source %s with spatial_op=none; "
                "mask projection assumes a uniform stride.",
                padded,
                src_shape,
            )
        ratios = [p // l for p, l in zip(padded, lat_shape)]
        if len(set(ratios)) != 1:
            raise ValueError(
                f"non-uniform latent stride detected: padded={padded}, lat={lat_shape}"
            )
        return int(ratios[0])

    @staticmethod
    def _check_scan_alignment(lat: h5py.File, src: h5py.File, n: int) -> None:
        lat_ids = [_decode(s) for s in lat["scan_ids"][: min(n, 16)]]
        src_ids = [_decode(s) for s in src["scan_ids"][: min(n, 16)]]
        if lat_ids != src_ids:
            raise ValueError(
                "scan_ids in latent and source H5 do not match in the first 16 rows; "
                "the two files must come from the same encode pass.\n"
                f"  latent[:16]={lat_ids}\n  source[:16]={src_ids}"
            )

    def _write(
        self,
        *,
        cfg: ComputeFeaturesRoutineConfig,
        lat_attrs: dict,
        src_attrs: dict,
        modality_idx: int,
        spacing_mm: tuple[float, float, float],
        latent_channels: int,
        latent_spatial_shape: tuple[int, int, int],
        label_set: np.ndarray,
        scan_ids: np.ndarray,
        patient_ids: np.ndarray,
        timepoint_idx: np.ndarray,
        z_tumor: np.ndarray,
        z_global: np.ndarray,
        z_random: np.ndarray,
        log_volume: np.ndarray,
        n_voxels_tumor: np.ndarray,
        n_voxels_lat_tumor: np.ndarray,
        mask_lat_present: np.ndarray,
        z_full: np.ndarray | None,
        mask_lat: np.ndarray | None,
    ) -> None:
        n_scans = z_tumor.shape[0]
        vlen = h5py.special_dtype(vlen=str)
        with h5py.File(cfg.output_h5, "w") as f:
            f.attrs["schema_version"] = FEATURES_SCHEMA_VERSION
            f.attrs["source_latents_h5"] = str(cfg.latent_h5)
            f.attrs["source_h5"] = str(cfg.source_h5)
            f.attrs["modality"] = cfg.modality
            f.attrs["modality_index"] = np.int64(modality_idx)
            f.attrs["pooling_seed"] = np.int64(cfg.random_region_seed)
            f.attrs["compute_z_full"] = bool(cfg.compute_z_full)
            f.attrs["created_at"] = _dt.datetime.now(_dt.UTC).isoformat()
            f.attrs["dataset_name"] = _decode(lat_attrs.get("dataset_name", ""))
            f.attrs["n_scans"] = np.int64(n_scans)
            f.attrs["n_patients"] = np.int64(len(np.unique(patient_ids)))
            f.attrs["latent_channels"] = np.int64(latent_channels)
            f.attrs["latent_spatial_shape"] = np.asarray(latent_spatial_shape, dtype=np.int64)
            f.attrs["spacing_mm"] = np.asarray(spacing_mm, dtype=np.float64)
            f.attrs["label_set_used"] = label_set.astype(np.int32)
            f.attrs["routine_config"] = json.dumps(_jsonable(dataclasses.asdict(cfg)))

            f.create_dataset("z_tumor", data=z_tumor)
            f.create_dataset("z_global", data=z_global)
            f.create_dataset("z_random", data=z_random)
            f.create_dataset("log_volume", data=log_volume)
            f.create_dataset("n_voxels_tumor", data=n_voxels_tumor)
            f.create_dataset("n_voxels_lat_tumor", data=n_voxels_lat_tumor)
            f.create_dataset("mask_lat_present", data=mask_lat_present)
            f.create_dataset("scan_ids", shape=(n_scans,), dtype=vlen)
            f["scan_ids"][...] = scan_ids
            f.create_dataset("patient_ids", shape=(n_scans,), dtype=vlen)
            f["patient_ids"][...] = patient_ids
            f.create_dataset("timepoint_idx", data=timepoint_idx)

            if z_full is not None:
                c, h_l, w_l, d_l = z_full.shape[1:]
                f.create_dataset(
                    "z_full",
                    data=z_full,
                    chunks=(1, c, h_l, w_l, d_l),
                    compression="gzip",
                    compression_opts=cfg.compression_level,
                )
            if mask_lat is not None:
                _, _, h_l, w_l, d_l = mask_lat.shape
                f.create_dataset(
                    "mask_lat",
                    data=mask_lat,
                    chunks=(1, 1, h_l, w_l, d_l),
                    compression="gzip",
                    compression_opts=cfg.compression_level,
                )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U"}:
        return str(value)
    return str(value)


def _prepare_mask_for_projection(
    mask: np.ndarray,
    *,
    spatial_op: str,
    working_shape: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
    crop_offset: tuple[int, int, int],
) -> np.ndarray:
    """Replicate the encoder's spatial preparation on a binary mask.

    The encoder applies (i) crop or resize to ``working_shape``, (ii) zero-pad
    to ``padded_shape`` with ``(0, pad)`` per axis. We mirror that for the
    mask, except resize uses nearest-neighbor instead of trilinear so the
    binary semantics are preserved.
    """
    if spatial_op == "none":
        m = mask
    elif spatial_op == "crop":
        sl = tuple(slice(o, o + s) for o, s in zip(crop_offset, working_shape))
        m = mask[sl]
    elif spatial_op == "resize":
        # Nearest-neighbor resize via torch.nn.functional.interpolate.
        t = torch.from_numpy(mask.astype(np.uint8))[None, None].float()
        t = torch.nn.functional.interpolate(t, size=working_shape, mode="nearest")
        m = t[0, 0].numpy().astype(bool)
    else:
        raise ValueError(f"unknown spatial_op: {spatial_op!r}")

    pads = tuple((0, p - w) for w, p in zip(working_shape, padded_shape))
    if any(p[1] != 0 for p in pads):
        m = np.pad(m.astype(bool), pads, mode="constant", constant_values=False)
    return m.astype(bool)


def _jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj
