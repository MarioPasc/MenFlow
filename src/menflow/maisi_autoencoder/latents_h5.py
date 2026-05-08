"""Provisional HDF5 schema for storing MAISI-v2 latents.

Provisional notice
------------------

This schema is *provisional*: the future MeningiomaFlow architecture may add an
auxiliary latent stream for the tumour mask, residual encoders, or multi-scale
features. To accommodate this, every per-scan latent group is opened with room
for sibling datasets:

* ``latents`` — required, shape ``(N, M, C, H', W', D')`` float16/32.
* ``mask_latents`` — *future, optional*; same trailing dims, no penalty if absent.
* ``residual_latents`` — *future, optional*.

Schema layout
-------------

::

    /
    ├── attrs:
    │   ├── schema_version           "0.1-provisional"
    │   ├── source_h5                (absolute path of the encoded MenFlow H5)
    │   ├── source_dataset_name      (mirror of /attrs/dataset_name in source)
    │   ├── source_spatial_shape     int64[3]
    │   ├── encoder_name             "MAISI-v2"
    │   ├── encoder_checkpoint       (path)
    │   ├── encoder_config           JSON of MaisiV2Config
    │   ├── inference_config         JSON of InferenceConfig
    │   ├── modalities               string[M]
    │   ├── n_modalities             int64
    │   ├── n_scans                  int64
    │   ├── latent_shape             int64[4]   (C, H', W', D')
    │   ├── deterministic            bool
    │   └── encoded_at               ISO-8601
    ├── latents               [N, M, C, H', W', D']  fp16 (default) or fp32
    ├── scan_ids              [N] str
    ├── patient_ids           [N] str
    ├── timepoint_idx         [N] int32
    ├── intensity_lower       [N, M] float32   (per-scan, per-modality lower percentile value)
    ├── intensity_upper       [N, M] float32   (... upper percentile value)
    └── original_spatial_shape int32[3]   (echo of source for convenience)
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

logger = logging.getLogger(__name__)


LATENT_SCHEMA_VERSION = "0.1-provisional"


class LatentH5Writer:
    """Streaming writer for the provisional MAISI-v2 latent HDF5.

    The file is opened on construction; per-scan ``write_scan`` calls update a
    single row of each dataset; :meth:`close` (or context-manager exit) finalises.
    All datasets are pre-allocated, so write order is irrelevant.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        n_scans: int,
        modalities: Sequence[str],
        latent_shape: tuple[int, int, int, int],  # (C, H', W', D')
        source_h5: str | Path,
        source_dataset_name: str,
        source_spatial_shape: tuple[int, int, int],
        encoder_checkpoint: str | Path,
        encoder_config: Mapping,
        inference_config: Mapping,
        deterministic: bool,
        latent_dtype: str = "float16",
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

        m = len(modalities)
        c, h, w, d = latent_shape

        self._file = h5py.File(path, "w")
        f = self._file
        vlen = h5py.special_dtype(vlen=str)

        f.attrs["schema_version"] = LATENT_SCHEMA_VERSION
        f.attrs["source_h5"] = str(source_h5)
        f.attrs["source_dataset_name"] = source_dataset_name
        f.attrs["source_spatial_shape"] = np.asarray(source_spatial_shape, dtype=np.int64)
        f.attrs["encoder_name"] = "MAISI-v2"
        f.attrs["encoder_checkpoint"] = str(encoder_checkpoint)
        f.attrs["encoder_config"] = json.dumps(_jsonable(encoder_config))
        f.attrs["inference_config"] = json.dumps(_jsonable(inference_config))
        f.attrs["modalities"] = np.asarray(list(modalities), dtype=object)
        f.attrs["n_modalities"] = np.int64(m)
        f.attrs["n_scans"] = np.int64(n_scans)
        f.attrs["latent_shape"] = np.asarray([c, h, w, d], dtype=np.int64)
        f.attrs["deterministic"] = bool(deterministic)
        f.attrs["encoded_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

        self._latents = f.create_dataset(
            "latents",
            shape=(n_scans, m, c, h, w, d),
            dtype=latent_dtype,
            chunks=(1, 1, c, h, w, d),
            compression=compression,
            compression_opts=compression_level,
        )
        f.create_dataset("scan_ids", shape=(n_scans,), dtype=vlen)
        f.create_dataset("patient_ids", shape=(n_scans,), dtype=vlen)
        f.create_dataset("timepoint_idx", shape=(n_scans,), dtype="int32")
        f.create_dataset(
            "intensity_lower", shape=(n_scans, m), dtype="float32"
        )
        f.create_dataset(
            "intensity_upper", shape=(n_scans, m), dtype="float32"
        )
        f.create_dataset(
            "original_spatial_shape", data=np.asarray(source_spatial_shape, dtype=np.int32)
        )

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
            raise ValueError(
                f"latent must be (M, C, H', W', D'); got {latent.shape}"
            )
        f = self._file
        self._latents[index] = latent.astype(self._latents.dtype, copy=False)
        f["scan_ids"][index] = scan_id
        f["patient_ids"][index] = patient_id
        f["timepoint_idx"][index] = np.int32(timepoint)
        f["intensity_lower"][index] = np.asarray(intensity_lower, dtype=np.float32)
        f["intensity_upper"][index] = np.asarray(intensity_upper, dtype=np.float32)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "LatentH5Writer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_latent_h5(path: str | Path) -> dict:
    """Read a latent H5 into a flat dict for inspection/testing.

    Loads everything into memory; intended for small files and diagnostics, not
    for the production decode path (the decode engine should stream).
    """
    out: dict = {}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            out[f"attr/{k}"] = (
                v.tolist() if isinstance(v, np.ndarray) else v
            )
        for name in ("latents", "scan_ids", "patient_ids", "timepoint_idx",
                     "intensity_lower", "intensity_upper", "original_spatial_shape"):
            if name in f:
                out[name] = f[name][:]
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
