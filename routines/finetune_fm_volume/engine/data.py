"""Dataset that joins MAISI latents and per-modality `log_volume` features by scan_id.

Inputs (no re-encoding required)
--------------------------------
- Latent H5 (`brats_men_maisi_latents.h5`): `latents [N, M, C, H', W', D']` fp16,
  `scan_ids`, ``splits/kfold/k{N}/...`` patient-index arrays (see
  :mod:`menflow.data.kfold_splits`), ``longitudinal/patient_offsets``.
- Features H5 directory holding per-modality files
  ``brats_men_features_{t1c,t1n,t2f,t2w}.h5``. Each provides ``log_volume [N]``
  and ``scan_ids [N]``. Volume is segmentation-derived → identical across
  modalities.

The dataset returns ``(z * scale_factor, log_v, modality_idx, spacing_tensor,
scan_id)`` samples for the active modalities, filtered to the active split.
Splits are patient-level; the class expands them to scan rows via
``longitudinal/patient_offsets``. The ``split`` field accepts any HDF5 path
under ``splits/``, e.g.

* ``"kfold/k1/fold_0/train"`` — k=1 holdout train set (default).
* ``"kfold/k1/fold_0/val"`` — k=1 holdout val.
* ``"kfold/k1/test"``      — shared held-out test set.
* ``"kfold/k5/fold_2/val"`` — fold 2 of the 5-fold split.

Legacy flat names like ``"e3_train"`` continue to resolve directly under
``splits/`` for backwards compatibility.

The H5 file handle is opened lazily per worker to remain ``num_workers > 0`` safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from menflow.finetuning.unet_wrapper import MAISI_MODALITY_INDEX

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """File paths and selection knobs for :class:`JointLatentVolumeDataset`."""

    latent_h5: Path
    features_h5_dir: Path  # holds brats_men_features_<modality>.h5
    features_h5_template: str = "brats_men_features_{modality}.h5"
    modalities: tuple[str, ...] = ("t1c",)
    split: str = "kfold/k1/fold_0/train"  # any HDF5 path under "splits/", or "all"
    log_v_floor: float = -6.0  # drop scans with log_v <= floor (sentinel −6.9078)
    spacing_scale: float = 100.0  # MAISI convention: spacing_mm * 100
    pad_multiple: int = 16  # FM U-Net has 3 downsample steps; pad to /16 for safety
    modality_index_override: dict[str, int] = field(default_factory=dict)


class JointLatentVolumeDataset(Dataset):
    """Yield (latent, log_v, modality_idx, spacing) samples for FM finetuning.

    Multi-modality: `__len__ = n_active_scans * len(modalities)`. The same scan
    contributes one sample per modality, with its segmentation-derived `log_v`.

    Notes
    -----
    - Latents are returned in **float32** ready for autocast forward; the on-disk
      fp16 storage is upcast at read.
    - The ``scale_factor`` is **not** applied here; the engine multiplies it onto
      the batch (centralizes the value, which lives in the FM checkpoint).
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._latent_handle: h5py.File | None = None
        self._features_handles: dict[str, h5py.File] | None = None

        index_map = dict(MAISI_MODALITY_INDEX)
        index_map.update(cfg.modality_index_override)
        for m in cfg.modalities:
            if m not in index_map:
                raise KeyError(f"no MAISI class index registered for modality {m!r}")
        self._modality_index = {m: int(index_map[m]) for m in cfg.modalities}

        with h5py.File(cfg.latent_h5, "r") as f:
            scan_ids = np.asarray(
                [s.decode() if isinstance(s, bytes) else str(s) for s in f["scan_ids"][:]]
            )
            modalities_attr = list(f.attrs["modalities"])
            modalities_attr = [
                m.decode() if isinstance(m, bytes) else str(m) for m in modalities_attr
            ]
            for m in cfg.modalities:
                if m not in modalities_attr:
                    raise ValueError(
                        f"modality {m!r} not in latent H5 modalities {modalities_attr}"
                    )
            self._modality_axis = {m: modalities_attr.index(m) for m in cfg.modalities}
            self._latent_shape = tuple(int(x) for x in f.attrs["latent_spatial_shape"])
            self._latent_channels = int(f.attrs["latent_channels"])
            spacing_mm = np.asarray(f.attrs.get("spacing_mm", [1.0, 1.0, 1.0]), dtype=np.float32)
            self._spacing_mm = spacing_mm

            patient_offsets = np.asarray(f["longitudinal/patient_offsets"][:], dtype=np.int64)
            scan_rows_for_split = self._scan_rows_from_split(f, cfg.split, patient_offsets)

        # Per-modality log_volume (segmentation-derived, identical across mods).
        # Use the t1c file by default — alignment was checked at exploration.
        canonical_mod = cfg.modalities[0]
        feat_path = cfg.features_h5_dir / cfg.features_h5_template.format(modality=canonical_mod)
        with h5py.File(feat_path, "r") as f:
            feat_scan_ids = np.asarray(
                [s.decode() if isinstance(s, bytes) else str(s) for s in f["scan_ids"][:]]
            )
            log_v_all = np.asarray(f["log_volume"][:], dtype=np.float32)
        if not np.array_equal(scan_ids, feat_scan_ids):
            raise ValueError(f"scan_id ordering differs between latent H5 and {feat_path.name}")

        # Filter the chosen split rows by log_v floor.
        keep = log_v_all > cfg.log_v_floor
        usable_rows = np.asarray([r for r in scan_rows_for_split if keep[r]], dtype=np.int64)
        n_dropped = len(scan_rows_for_split) - len(usable_rows)
        if n_dropped > 0:
            logger.info(
                "dropped %d/%d scans below log_v floor %.3f (split=%s)",
                n_dropped,
                len(scan_rows_for_split),
                cfg.log_v_floor,
                cfg.split,
            )

        self._scan_rows = usable_rows
        self._scan_ids = scan_ids
        self._log_v = log_v_all
        self._modalities = tuple(cfg.modalities)

    # ------------------------------------------------------------------
    # Worker-safe lazy file handles
    # ------------------------------------------------------------------

    def _open(self) -> h5py.File:
        if self._latent_handle is None:
            self._latent_handle = h5py.File(self.cfg.latent_h5, "r", swmr=True)
        return self._latent_handle

    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        st["_latent_handle"] = None
        return st

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._scan_rows) * len(self._modalities)

    def __getitem__(self, idx: int) -> dict:
        n_rows = len(self._scan_rows)
        if idx < 0 or idx >= n_rows * len(self._modalities):
            raise IndexError(idx)
        modality_idx_in_list = idx // n_rows
        row_idx = idx % n_rows
        modality = self._modalities[modality_idx_in_list]
        scan_row = int(self._scan_rows[row_idx])

        f = self._open()
        # latents shape: (N, M, C, H', W', D')
        m_axis = self._modality_axis[modality]
        z = f["latents"][scan_row, m_axis]  # (C, H', W', D'), fp16 on disk
        z = np.asarray(z, dtype=np.float32)

        pad = self.cfg.pad_multiple
        if pad > 1:
            pad_amounts = [(0, (pad - dim % pad) % pad) for dim in z.shape[1:]]
            if any(p[1] > 0 for p in pad_amounts):
                z = np.pad(z, [(0, 0), *pad_amounts], mode="constant", constant_values=0.0)

        log_v = float(self._log_v[scan_row])
        scan_id = str(self._scan_ids[scan_row])
        modality_class = self._modality_index[modality]
        spacing = (self._spacing_mm * self.cfg.spacing_scale).astype(np.float32)

        return {
            "z": torch.from_numpy(z),  # (C, H', W', D') fp32
            "log_v": torch.tensor(log_v, dtype=torch.float32),
            "modality_idx": torch.tensor(modality_class, dtype=torch.long),
            "spacing": torch.from_numpy(spacing),  # (3,) fp32
            "scan_id": scan_id,
            "modality": modality,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_rows_from_split(
        f: h5py.File,
        split: str,
        patient_offsets: np.ndarray,
    ) -> np.ndarray:
        """Resolve a split path under ``splits/`` to scan-row indices.

        Accepts any HDF5 path inside the ``splits`` group (e.g.
        ``"kfold/k1/fold_0/train"`` or the legacy flat ``"e3_train"``). The
        special name ``"all"`` returns every scan row. Missing paths raise
        :class:`KeyError`.
        """
        n_scans = int(f.attrs["n_scans"])
        if split == "all":
            return np.arange(n_scans, dtype=np.int64)
        if "splits" not in f:
            raise KeyError(f"latent H5 has no /splits group; cannot resolve {split!r}")
        h5_path = f"splits/{split.lstrip('/')}"
        if h5_path not in f:
            raise KeyError(
                f"split path {h5_path!r} not present in latent H5; top-level splits keys: "
                f"{list(f['splits'].keys())}"
            )
        patient_indices = np.asarray(f[h5_path][:], dtype=np.int64)
        rows: list[int] = []
        for p_idx in patient_indices:
            start = int(patient_offsets[p_idx])
            end = int(patient_offsets[p_idx + 1])
            rows.extend(range(start, end))
        return np.asarray(rows, dtype=np.int64)


def collate_samples(batch: list[dict]) -> dict:
    """Default collate that stacks tensors and keeps `scan_id` as a list of strings."""
    out: dict = {}
    for k in ("z", "log_v", "modality_idx", "spacing"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["scan_id"] = [b["scan_id"] for b in batch]
    out["modality"] = [b["modality"] for b in batch]
    return out
