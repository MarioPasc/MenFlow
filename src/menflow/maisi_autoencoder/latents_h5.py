"""Provisional self-describing HDF5 schema for storing MAISI-v2 latents.

Schema version 0.2-provisional. Every latent file mirrors the cohort identity
fields of the unified MenFlow schema (`dataset_name`, `dataset_type`,
`n_patients`, `longitudinal/*`, optional `splits/*`) so the file can be consumed
without the source MenFlow H5 also being available — a hard requirement for
running flow training on Picasso when the source dataset lives on a different
machine.

Schema layout
-------------

::

    /
    ├── attrs:
    │   # provenance
    │   ├── schema_version           "0.2-provisional"
    │   ├── source_h5                (path of the encoded MenFlow H5)
    │   ├── encoded_at               ISO-8601 UTC
    │   ├── encoder_name             "MAISI-v2"
    │   ├── encoder_checkpoint       (path)
    │   ├── encoder_config           JSON of MaisiV2Config
    │   ├── inference_config         JSON of InferenceConfig
    │   ├── deterministic            bool
    │   # cohort identity (mirrors menflow.data.h5_schema)
    │   ├── dataset_name             string
    │   ├── dataset_type             "cross-sectional" | "longitudinal"
    │   ├── n_scans                  int64
    │   ├── n_patients               int64
    │   ├── modalities               string[M]
    │   ├── n_modalities             int64
    │   ├── orientation              string
    │   ├── spacing_mm               float64[3]
    │   # spatial provenance — every transformation between source voxels and
    │   # the encoder's input is recorded so decode is fully reversible.
    │   ├── source_spatial_shape     int64[3]   original cohort shape
    │   ├── working_spatial_shape    int64[3]   after spatial_op, before pad
    │   ├── padded_spatial_shape     int64[3]   what the encoder actually saw
    │   ├── spatial_op               "none" | "resize" | "crop"
    │   ├── crop_offset              int64[3]   only meaningful if op=="crop"
    │   ├── latent_spatial_shape     int64[3]   (H', W', D')
    │   └── latent_channels          int64
    ├── latents               [N, M, C, H', W', D']  fp16 (default) or fp32
    ├── scan_ids              [N] vlen str
    ├── patient_ids           [N] vlen str
    ├── timepoint_idx         [N] int32
    ├── intensity_lower       [N, M] float32   per-scan, per-modality lower percentile value
    ├── intensity_upper       [N, M] float32   per-scan, per-modality upper percentile value
    ├── longitudinal/
    │   ├── patient_offsets   [n_patients+1] int32   CSR offsets into the scan axis
    │   └── patient_list      [n_patients]   vlen str
    └── splits/   (optional, mirrored from source)
        └── <name> [k] int32

Future extension points
-----------------------

* ``mask_latents`` — auxiliary latent stream for the tumour mask, same trailing
  dims as ``latents`` minus the modality axis.
* ``residual_latents`` — lesion-residual encoder output for the §9 mitigation.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


LATENT_SCHEMA_VERSION = "0.2-provisional"


# Canonical training-time splits the latent H5 is expected to expose. Driven by
# the E3.1 spec (§4.2): patient-grouped, log-volume-stratified train/val/test.
# Any other split name on the source H5 is dropped by the encode engine, and
# legacy ``train``/``val`` (BraTS challenge cohort partition) trigger a
# deprecation warning if written here.
EXPECTED_E3_SPLITS: tuple[str, ...] = ("e3_train", "e3_val", "e3_test")
LEGACY_SPLIT_NAMES: tuple[str, ...] = ("train", "val")


class LatentH5Writer:
    """Streaming writer for the v0.2 self-describing latent HDF5.

    All datasets are pre-allocated, so :meth:`write_scan` is order-independent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        # cohort identity (mirrors source unified-schema attrs)
        dataset_name: str,
        dataset_type: str,
        modalities: Sequence[str],
        orientation: str,
        spacing_mm: Sequence[float],
        # patient grouping (CSR layout, length n_patients+1 for offsets)
        patient_list: Sequence[str],
        patient_offsets: Sequence[int],
        # cohort size
        n_scans: int,
        # spatial provenance
        source_spatial_shape: Sequence[int],
        working_spatial_shape: Sequence[int],
        padded_spatial_shape: Sequence[int],
        latent_spatial_shape: Sequence[int],
        latent_channels: int,
        spatial_op: str,
        crop_offset: Sequence[int] = (0, 0, 0),
        # encoder provenance
        source_h5: str | Path,
        encoder_checkpoint: str | Path,
        encoder_config: Mapping,
        inference_config: Mapping,
        deterministic: bool,
        # storage
        latent_dtype: str = "float16",
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        if spatial_op not in {"none", "resize", "crop"}:
            raise ValueError(f"spatial_op must be one of {{none,resize,crop}}; got {spatial_op!r}")
        if dataset_type not in {"cross-sectional", "longitudinal"}:
            raise ValueError(
                f"dataset_type must be cross-sectional or longitudinal; got {dataset_type!r}"
            )

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

        m = len(modalities)
        c = int(latent_channels)
        h_l, w_l, d_l = (int(x) for x in latent_spatial_shape)
        n_patients = len(patient_list)
        if len(patient_offsets) != n_patients + 1:
            raise ValueError(
                f"patient_offsets must have length n_patients+1={n_patients + 1}, "
                f"got {len(patient_offsets)}"
            )
        if int(patient_offsets[0]) != 0 or int(patient_offsets[-1]) != n_scans:
            raise ValueError(
                f"patient_offsets must start at 0 and end at n_scans={n_scans}; "
                f"got [{patient_offsets[0]}, ..., {patient_offsets[-1]}]"
            )

        self._file = h5py.File(path, "w")
        f = self._file
        vlen = h5py.special_dtype(vlen=str)

        # --- Provenance attrs ---
        f.attrs["schema_version"] = LATENT_SCHEMA_VERSION
        f.attrs["source_h5"] = str(source_h5)
        f.attrs["encoded_at"] = _dt.datetime.now(_dt.UTC).isoformat()
        f.attrs["encoder_name"] = "MAISI-v2"
        f.attrs["encoder_checkpoint"] = str(encoder_checkpoint)
        f.attrs["encoder_config"] = json.dumps(_jsonable(encoder_config))
        f.attrs["inference_config"] = json.dumps(_jsonable(inference_config))
        f.attrs["deterministic"] = bool(deterministic)

        # --- Cohort identity ---
        f.attrs["dataset_name"] = str(dataset_name)
        f.attrs["dataset_type"] = str(dataset_type)
        f.attrs["n_scans"] = np.int64(n_scans)
        f.attrs["n_patients"] = np.int64(n_patients)
        f.attrs["modalities"] = np.asarray(list(modalities), dtype=object)
        f.attrs["n_modalities"] = np.int64(m)
        f.attrs["orientation"] = str(orientation)
        f.attrs["spacing_mm"] = np.asarray(spacing_mm, dtype=np.float64)

        # --- Spatial provenance ---
        f.attrs["source_spatial_shape"] = np.asarray(source_spatial_shape, dtype=np.int64)
        f.attrs["working_spatial_shape"] = np.asarray(working_spatial_shape, dtype=np.int64)
        f.attrs["padded_spatial_shape"] = np.asarray(padded_spatial_shape, dtype=np.int64)
        f.attrs["spatial_op"] = str(spatial_op)
        f.attrs["crop_offset"] = np.asarray(crop_offset, dtype=np.int64)
        f.attrs["latent_spatial_shape"] = np.asarray([h_l, w_l, d_l], dtype=np.int64)
        f.attrs["latent_channels"] = np.int64(c)

        # --- Datasets ---
        self._latents = f.create_dataset(
            "latents",
            shape=(n_scans, m, c, h_l, w_l, d_l),
            dtype=latent_dtype,
            chunks=(1, 1, c, h_l, w_l, d_l),
            compression=compression,
            compression_opts=compression_level,
        )
        f.create_dataset("scan_ids", shape=(n_scans,), dtype=vlen)
        f.create_dataset("patient_ids", shape=(n_scans,), dtype=vlen)
        f.create_dataset("timepoint_idx", shape=(n_scans,), dtype="int32")
        f.create_dataset("intensity_lower", shape=(n_scans, m), dtype="float32")
        f.create_dataset("intensity_upper", shape=(n_scans, m), dtype="float32")

        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset(
            "patient_offsets",
            data=np.asarray(patient_offsets, dtype=np.int32),
        )
        long_grp.create_dataset(
            "patient_list",
            data=np.asarray(list(patient_list), dtype=object),
            dtype=vlen,
        )

        # /splits is created lazily by write_split
        self._splits_grp: h5py.Group | None = None

    # ----- Streaming API -----

    def write_scan(
        self,
        index: int,
        *,
        scan_id: str,
        patient_id: str,
        timepoint: int,
        latent: np.ndarray,  # shape (M, C, H', W', D')
        intensity_lower: Sequence[float],
        intensity_upper: Sequence[float],
    ) -> None:
        if latent.ndim != 5:
            raise ValueError(f"latent must be (M, C, H', W', D'); got {latent.shape}")
        f = self._file
        self._latents[index] = latent.astype(self._latents.dtype, copy=False)
        f["scan_ids"][index] = scan_id
        f["patient_ids"][index] = patient_id
        f["timepoint_idx"][index] = np.int32(timepoint)
        f["intensity_lower"][index] = np.asarray(intensity_lower, dtype=np.float32)
        f["intensity_upper"][index] = np.asarray(intensity_upper, dtype=np.float32)

    def write_split(self, name: str, indices: np.ndarray) -> None:
        """Persist a patient-level index split (mirrors source/splits/<name>).

        Names in :data:`LEGACY_SPLIT_NAMES` (``train``, ``val``) emit a
        :class:`DeprecationWarning` because they correspond to the BraTS
        challenge cohort partition rather than the patient-grouped, log-volume
        stratified training splits required by E3.1 §4.2 (see
        :data:`EXPECTED_E3_SPLITS`). They are still written for backwards
        compatibility, but new code must switch to the e3_* names.
        """
        if name in LEGACY_SPLIT_NAMES:
            warnings.warn(
                f"latent H5 split name {name!r} is deprecated; use one of "
                f"{EXPECTED_E3_SPLITS} instead. The legacy names refer to the "
                "BraTS challenge train/val cohorts, of which the val cohort "
                "is unannotated and unsuitable for FM finetuning.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self._splits_grp is None:
            self._splits_grp = self._file.create_group("splits")
        self._splits_grp.create_dataset(name, data=np.asarray(indices, dtype=np.int32))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> LatentH5Writer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_latent_h5(path: str | Path) -> dict:
    """Read a latent H5 into a flat dict for inspection/testing."""
    out: dict = {}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            out[f"attr/{k}"] = v.tolist() if isinstance(v, np.ndarray) else v
        for name in (
            "latents",
            "scan_ids",
            "patient_ids",
            "timepoint_idx",
            "intensity_lower",
            "intensity_upper",
        ):
            if name in f:
                out[name] = f[name][:]
        if "longitudinal/patient_offsets" in f:
            out["patient_offsets"] = f["longitudinal/patient_offsets"][:]
        if "longitudinal/patient_list" in f:
            out["patient_list"] = f["longitudinal/patient_list"][:]
        if "splits" in f:
            out["splits"] = {name: ds[:] for name, ds in f["splits"].items()}
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Mapping | object) -> object:
    """Recursively convert dataclass-likes / numpy types to JSON-safe primitives."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    elif hasattr(obj, "__dict__"):
        obj = vars(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj
