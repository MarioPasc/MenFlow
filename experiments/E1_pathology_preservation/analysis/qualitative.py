"""Qualitative case selection — dump best/worst/median scans as NPZ.

For each "anchor" metric the per-scan score is computed by averaging the
per-modality rows. Three scans are then selected:

* **best** — argmax (higher-is-better metrics) / argmin (lower-is-better).
* **worst** — opposite of best.
* **median** — scan whose aggregated score is closest to the cohort median.

For each selected scan we save a single ``.npz`` holding the full source
volume, the recon volume (both ``(M, H, W, D)``), the segmentation, and the
modality names. Loading three NPZ per metric is cheap, so this keeps the
qualitative-visualisation workflow off Picasso entirely.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


HIGHER_IS_BETTER_PREFIXES: tuple[str, ...] = ("psnr", "ssim", "msssim")
LOWER_IS_BETTER_PREFIXES: tuple[str, ...] = ("mae",)


def _metric_direction(metric: str) -> str:
    if metric.startswith(LOWER_IS_BETTER_PREFIXES):
        return "lower"
    if metric.startswith(HIGHER_IS_BETTER_PREFIXES):
        return "higher"
    raise ValueError(f"Unknown direction for metric '{metric}'")


def _select_cases(per_scan: pd.Series, *, direction: str) -> dict[str, str]:
    """Return {case_name: scan_id} for best/worst/median."""
    s = per_scan.dropna()
    if s.empty:
        return {}
    sorted_idx = s.sort_values(ascending=True).index  # low -> high
    if direction == "higher":
        best, worst = sorted_idx[-1], sorted_idx[0]
    else:
        best, worst = sorted_idx[0], sorted_idx[-1]
    median_value = s.median()
    median = (s - median_value).abs().idxmin()
    return {"best": best, "worst": worst, "median": median}


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _aligned_source_volume(
    src: h5py.File,
    *,
    idx: int,
    spatial_op: str,
    crop_offset: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    vol = src["images"][idx]  # (M, H, W, D)
    if spatial_op == "crop":
        sl = (slice(None),) + tuple(
            slice(o, o + s) for o, s in zip(crop_offset, target_shape)
        )
        return vol[sl]
    return vol


def _aligned_seg(
    *,
    src: h5py.File,
    rec: h5py.File,
    idx: int,
    spatial_op: str,
) -> np.ndarray:
    if spatial_op == "crop":
        return rec["segmentations"][idx]
    return src["segmentations"][idx]


def save_qualitative_cases(
    *,
    df: pd.DataFrame,
    source_h5: str | Path,
    recon_h5: str | Path,
    output_dir: str | Path,
    anchor_metrics: Sequence[str],
) -> dict[tuple[str, str], Path]:
    """Save best/worst/median scans per anchor metric as ``.npz`` files.

    Parameters
    ----------
    df : pd.DataFrame
        Per-scan/per-modality metrics (output of ``compute_metrics``).
    source_h5 : path
        Original cohort H5 (used to load full-resolution source volumes).
    recon_h5 : path
        Reconstruction H5 (provides recon + segmentations + spatial_op attr).
    output_dir : path
        Directory under which ``qualitative/<metric>_<case>.npz`` is written.
    anchor_metrics : sequence of str
        Metric column names from ``df`` driving case selection.

    Returns
    -------
    dict
        ``{(metric, case): npz_path}`` for every successful write.
    """
    if df.empty:
        logger.warning("Qualitative dump skipped: empty metrics DataFrame.")
        return {}

    out_root = Path(output_dir) / "qualitative"
    out_root.mkdir(parents=True, exist_ok=True)

    written: dict[tuple[str, str], Path] = {}
    with h5py.File(source_h5, "r") as src, h5py.File(recon_h5, "r") as rec:
        spatial_op = (
            _decode(rec.attrs["spatial_op"]) if "spatial_op" in rec.attrs else "none"
        )
        crop_offset = (
            tuple(int(x) for x in rec.attrs["crop_offset"])
            if "crop_offset" in rec.attrs
            else (0, 0, 0)
        )
        target_shape = tuple(int(x) for x in rec.attrs["spatial_shape"])
        recon_modalities = tuple(_decode(m) for m in rec.attrs["modalities"])
        source_modalities = tuple(_decode(m) for m in src.attrs["modalities"])
        modality_idx_in_source = [source_modalities.index(m) for m in recon_modalities]

        scan_to_row = {row.scan_id: row for row in rec_scan_index(rec)}

        for metric in anchor_metrics:
            if metric not in df.columns:
                logger.warning("Anchor metric '%s' missing from DataFrame", metric)
                continue
            direction = _metric_direction(metric)
            per_scan = (
                df.groupby("scan_id", sort=False)[metric].mean()
            )
            cases = _select_cases(per_scan, direction=direction)
            for case_name, scan_id in cases.items():
                if scan_id not in scan_to_row:
                    logger.warning(
                        "scan_id %s present in metrics but missing from recon H5; skipping",
                        scan_id,
                    )
                    continue
                meta = scan_to_row[scan_id]
                idx = meta.index
                source_aligned_full = _aligned_source_volume(
                    src,
                    idx=idx,
                    spatial_op=spatial_op,
                    crop_offset=crop_offset,
                    target_shape=target_shape,
                )
                source_vol = np.stack(
                    [source_aligned_full[m_src_idx] for m_src_idx in modality_idx_in_source],
                    axis=0,
                ).astype(np.float32, copy=False)
                recon_vol = np.asarray(rec["images"][idx], dtype=np.float32)
                seg = np.asarray(
                    _aligned_seg(src=src, rec=rec, idx=idx, spatial_op=spatial_op),
                    dtype=np.int8,
                )

                npz_path = out_root / f"{metric}_{case_name}.npz"
                np.savez_compressed(
                    npz_path,
                    source=source_vol,
                    recon=recon_vol,
                    segmentation=seg,
                    modalities=np.asarray(recon_modalities),
                    scan_id=np.asarray(scan_id),
                    patient_id=np.asarray(meta.patient_id),
                    metric_name=np.asarray(metric),
                    metric_value=np.float64(per_scan[scan_id]),
                    case=np.asarray(case_name),
                    spatial_op=np.asarray(spatial_op),
                )
                logger.info(
                    "Saved qualitative case %s/%s (scan=%s, %s=%.4f) -> %s",
                    metric,
                    case_name,
                    scan_id,
                    metric,
                    per_scan[scan_id],
                    npz_path,
                )
                written[(metric, case_name)] = npz_path
    return written


class _ScanMeta:
    __slots__ = ("scan_id", "patient_id", "index")

    def __init__(self, scan_id: str, patient_id: str, index: int) -> None:
        self.scan_id = scan_id
        self.patient_id = patient_id
        self.index = index


def rec_scan_index(rec: h5py.File) -> list[_ScanMeta]:
    """Build a list of (scan_id, patient_id, row_index) for the recon H5."""
    n = int(rec.attrs["n_scans"])
    scan_ids = [_decode(s) for s in rec["scan_ids"][:n]]
    patient_ids = [_decode(s) for s in rec["patient_ids"][:n]]
    return [
        _ScanMeta(scan_id=sid, patient_id=pid, index=i)
        for i, (sid, pid) in enumerate(zip(scan_ids, patient_ids))
    ]
