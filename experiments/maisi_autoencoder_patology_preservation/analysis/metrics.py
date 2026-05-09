"""Per-scan reconstruction metrics for E1 (§6.1 global, §6.2 tumor-region).

Two mask definitions are evaluated in parallel:

* ``whole_tumor``  — labels {1, 2, 3} merged (NETC + SNFH + ET).
* ``tumor_core``   — labels {1, 3} (NETC + ET, excludes peritumoral edema).

For each modality–scan pair we compute global PSNR/SSIM/MS-SSIM and, per
tumor-region definition, PSNR_tumor / SSIM_tumor / MAE_tumor plus the absolute
tumor volume in cm³ (used by §5 volume stratification).

LPIPS (§6.1) is currently disabled: it requires the ``lpips`` package and a
slice-wise loop with a 2-D net. Re-enable by passing ``compute_lpips=True``
once the package is added to a Picasso-compatible install.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import pandas as pd
import torch
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric, SSIMMetric

logger = logging.getLogger(__name__)


WHOLE_TUMOR_LABELS: tuple[int, ...] = (1, 2, 3)
TUMOR_CORE_LABELS: tuple[int, ...] = (1, 3)


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    """Per (scan_id, modality) reconstruction metrics."""

    scan_id: str
    patient_id: str
    modality: str
    # tumor volumes — same for every modality in a scan, but kept here for
    # convenient pandas joining
    volume_wt_cm3: float
    volume_tc_cm3: float
    # global metrics
    psnr_global: float
    ssim_global: float
    msssim_global: float
    # tumor-region — whole tumor
    psnr_wt: float
    ssim_wt: float
    mae_wt: float
    # tumor-region — tumor core
    psnr_tc: float
    ssim_tc: float
    mae_tc: float

    def to_row(self) -> dict[str, float | str]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Atomic metric helpers
# ---------------------------------------------------------------------------


def _mask_volume_cm3(mask: np.ndarray, voxel_volume_cm3: float) -> float:
    """Volume in cm^3 of a binary mask."""
    return float(mask.sum()) * voxel_volume_cm3


def _binary_mask(seg: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Boolean mask = union of voxels matching any label in *labels*."""
    out = np.zeros_like(seg, dtype=bool)
    for lbl in labels:
        out |= seg == lbl
    return out


def _global_psnr(x: torch.Tensor, x_hat: torch.Tensor, *, data_range: float) -> float:
    """Global PSNR with explicit data range (peak intensity)."""
    metric = PSNRMetric(max_val=data_range)
    return float(metric(x_hat.unsqueeze(0), x.unsqueeze(0)).item())


def _global_ssim(x: torch.Tensor, x_hat: torch.Tensor, *, data_range: float) -> float:
    metric = SSIMMetric(spatial_dims=3, data_range=data_range)
    return float(metric(x_hat.unsqueeze(0), x.unsqueeze(0)).item())


def _global_msssim(x: torch.Tensor, x_hat: torch.Tensor, *, data_range: float) -> float:
    """MS-SSIM. Falls back to NaN for volumes too small for the 5-scale pyramid."""
    spatial = tuple(int(s) for s in x.shape[1:])  # (1, H, W, D) -> (H,W,D)
    if min(spatial) < 32:
        return float("nan")
    try:
        metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=data_range)
        return float(metric(x_hat.unsqueeze(0), x.unsqueeze(0)).item())
    except Exception as exc:  # pragma: no cover — depends on input size
        logger.warning("MS-SSIM failed, returning NaN: %s", exc)
        return float("nan")


def _masked_psnr(x: np.ndarray, x_hat: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    diff = (x[mask] - x_hat[mask]) ** 2
    mse = float(diff.mean())
    if mse == 0.0:
        return float("inf")
    peak = float(x.max())
    if peak <= 0:
        return float("nan")
    return 10.0 * math.log10(peak**2 / mse)


def _masked_mae(x: np.ndarray, x_hat: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.abs(x[mask] - x_hat[mask]).mean())


def _masked_ssim_3d(
    x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor, *, data_range: float
) -> float:
    """Mask-weighted SSIM via the per-voxel SSIM map (no reduction)."""
    if mask.sum().item() == 0:
        return float("nan")
    metric = SSIMMetric(spatial_dims=3, data_range=data_range, reduction="none")
    ssim_map = metric(x_hat.unsqueeze(0), x.unsqueeze(0))  # (1, 1, H, W, D)
    weighted = (ssim_map * mask.unsqueeze(0)).sum() / mask.sum().clamp_min(1.0)
    return float(weighted.item())


# ---------------------------------------------------------------------------
# Per-scan / per-modality driver
# ---------------------------------------------------------------------------


def _per_scan_metrics(
    *,
    scan_id: str,
    patient_id: str,
    modality: str,
    source: np.ndarray,
    recon: np.ndarray,
    seg: np.ndarray,
    voxel_volume_cm3: float,
    device: torch.device,
) -> ReconstructionMetrics:
    """Compute every metric for one (scan, modality) pair."""
    if source.shape != recon.shape:
        raise ValueError(f"shape mismatch: source {source.shape} vs recon {recon.shape}")
    if seg.shape != source.shape:
        raise ValueError(f"seg {seg.shape} != source {source.shape}")

    data_range = float(max(source.max() - source.min(), 1e-6))

    src_t = torch.from_numpy(source.astype(np.float32, copy=False))[None].to(device)
    rec_t = torch.from_numpy(recon.astype(np.float32, copy=False))[None].to(device)

    psnr_g = _global_psnr(src_t, rec_t, data_range=data_range)
    ssim_g = _global_ssim(src_t, rec_t, data_range=data_range)
    msssim_g = _global_msssim(src_t, rec_t, data_range=data_range)

    wt = _binary_mask(seg, WHOLE_TUMOR_LABELS)
    tc = _binary_mask(seg, TUMOR_CORE_LABELS)

    wt_t = torch.from_numpy(wt.astype(np.float32))[None].to(device)
    tc_t = torch.from_numpy(tc.astype(np.float32))[None].to(device)

    return ReconstructionMetrics(
        scan_id=scan_id,
        patient_id=patient_id,
        modality=modality,
        volume_wt_cm3=_mask_volume_cm3(wt, voxel_volume_cm3),
        volume_tc_cm3=_mask_volume_cm3(tc, voxel_volume_cm3),
        psnr_global=psnr_g,
        ssim_global=ssim_g,
        msssim_global=msssim_g,
        psnr_wt=_masked_psnr(source, recon, wt),
        ssim_wt=_masked_ssim_3d(src_t, rec_t, wt_t, data_range=data_range),
        mae_wt=_masked_mae(source, recon, wt),
        psnr_tc=_masked_psnr(source, recon, tc),
        ssim_tc=_masked_ssim_3d(src_t, rec_t, tc_t, data_range=data_range),
        mae_tc=_masked_mae(source, recon, tc),
    )


# ---------------------------------------------------------------------------
# Cohort-level driver
# ---------------------------------------------------------------------------


def compute_metrics(
    *,
    source_h5: str,
    recon_h5: str,
    output_csv: str | None = None,
    max_scans: int | None = None,
    device: str = "cpu",
    require_segmentation: bool = True,
) -> pd.DataFrame:
    """Compute per-scan/per-modality metrics over a (source, recon) cohort pair.

    The recon H5's ``spatial_op`` attr drives whether segmentations are taken
    from the source (op = "none" or "resize") or already-cropped from the recon
    file itself (op = "crop"). In all cases the recon and the segmentation
    finally arrive at the same shape before metrics are evaluated.
    """
    torch_device = torch.device(device)
    rows: list[dict] = []

    with h5py.File(source_h5, "r") as src, h5py.File(recon_h5, "r") as rec:
        recon_shape = tuple(int(x) for x in rec.attrs["spatial_shape"])
        recon_modalities = tuple(_decode(m) for m in rec.attrs["modalities"])
        source_modalities = tuple(_decode(m) for m in src.attrs["modalities"])
        spacing_mm = tuple(float(x) for x in rec.attrs["spacing_mm"])
        voxel_volume_cm3 = float(np.prod(spacing_mm) / 1000.0)
        spatial_op = _decode(rec.attrs["spatial_op"]) if "spatial_op" in rec.attrs else "none"
        crop_offset = (
            tuple(int(x) for x in rec.attrs["crop_offset"])
            if "crop_offset" in rec.attrs
            else (0, 0, 0)
        )

        n_scans = int(rec.attrs["n_scans"])
        if max_scans is not None:
            n_scans = min(max_scans, n_scans)

        scan_ids = [_decode(s) for s in rec["scan_ids"][:n_scans]]
        patient_ids = [_decode(s) for s in rec["patient_ids"][:n_scans]]
        modality_idx_in_source = [source_modalities.index(m) for m in recon_modalities]
        has_seg = rec["has_segmentation"][:n_scans]
        recon_segs = rec["segmentations"]

        for i in range(n_scans):
            if require_segmentation and not bool(has_seg[i]):
                logger.info("Skipping scan %s (no segmentation)", scan_ids[i])
                continue

            seg = _aligned_seg(
                src=src,
                rec=rec,
                idx=i,
                spatial_op=spatial_op,
                crop_offset=crop_offset,
                target_shape=recon_shape,
            )

            for m_out_idx, m_src_idx in enumerate(modality_idx_in_source):
                source_vol = _aligned_source_volume(
                    src=src,
                    idx=i,
                    modality_idx=m_src_idx,
                    spatial_op=spatial_op,
                    crop_offset=crop_offset,
                    target_shape=recon_shape,
                )
                recon_vol = rec["images"][i, m_out_idx]

                m = _per_scan_metrics(
                    scan_id=scan_ids[i],
                    patient_id=patient_ids[i],
                    modality=recon_modalities[m_out_idx],
                    source=np.asarray(source_vol, dtype=np.float32),
                    recon=np.asarray(recon_vol, dtype=np.float32),
                    seg=np.asarray(seg, dtype=np.int8),
                    voxel_volume_cm3=voxel_volume_cm3,
                    device=torch_device,
                )
                rows.append(m.to_row())

    df = pd.DataFrame(rows)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        logger.info("Wrote per-scan metrics CSV to %s", output_csv)
    return df


# ---------------------------------------------------------------------------
# Spatial alignment helpers
# ---------------------------------------------------------------------------


def _aligned_seg(
    *,
    src: h5py.File,
    rec: h5py.File,
    idx: int,
    spatial_op: str,
    crop_offset: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a segmentation aligned to ``target_shape`` (the recon's shape)."""
    if spatial_op == "crop":
        # Recon already stores cropped segmentations.
        seg = rec["segmentations"][idx]
    else:
        seg = src["segmentations"][idx]
    if tuple(seg.shape) != target_shape:
        raise ValueError(f"seg shape {seg.shape} mismatched with target {target_shape}")
    return seg


def _aligned_source_volume(
    *,
    src: h5py.File,
    idx: int,
    modality_idx: int,
    spatial_op: str,
    crop_offset: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Pull source modality, applying the spatial_op so it aligns with recon."""
    vol = src["images"][idx, modality_idx]
    if spatial_op == "crop":
        sl = tuple(slice(o, o + s) for o, s in zip(crop_offset, target_shape))
        return vol[sl]
    if spatial_op == "resize":
        # Source is at original shape; recon was upsampled back. Compare at
        # source shape would require re-resizing recon — simpler: assume recon
        # was upsampled to source shape (which is what decode does for resize).
        return vol
    return vol


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
