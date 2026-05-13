"""E2.5 — orchestration engine.

Pipeline per spec §6.1:

1. Load latents + raw images, segs, log_volume from the unified H5.
2. Select cohort (Patience monotone-3, C4 ladder).
3. Reconstruct per-modality ridge probes from E2.1 ``direction.npz`` artifacts.
4. For each retained patient triple, compute:
     - ρ_lin (joint over modalities)
     - β_vol via probe sweep + β_time
     - decode at 6 β values + the three endpoint round-trip decodes
     - image SSIM at β_vol against x², non-tumour SSIM at each anatomy-β.
5. Cross-patient null baseline.
6. Bootstrap aggregation + decision.
7. Persist per-patient CSV, aggregate JSON, decision JSON, figures, report.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import gc
import json
import logging
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments.E2.E2_5_ae_longitudinal.analysis import latents as lat_mod
from experiments.E2.E2_5_ae_longitudinal.analysis.cohort import (
    PatientTriplet,
    select_cohort,
)
from experiments.E2.E2_5_ae_longitudinal.analysis.metrics import (
    DegenerateTripletError,
    linearity_residual,
    linearity_residual_pooled,
    ssim_3d,
    ssim_masked,
    volume_matched_beta,
)
from experiments.E2.E2_5_ae_longitudinal.analysis.null_baseline import compute_null_baseline
from experiments.E2.E2_5_ae_longitudinal.analysis.probe import RidgeProbe
from experiments.E2.E2_5_ae_longitudinal.analysis.subspace import (
    pca_longitudinal_direction,
    null_pca_alignment,
)
from experiments.E2.E2_5_ae_longitudinal.config import AELongitudinalConfig
from experiments.E2.E2_5_ae_longitudinal.decision import evaluate as evaluate_decision
from experiments.E2.E2_5_ae_longitudinal.reporting import (
    bootstrap_ci,
    bootstrap_paired_diff,
    cluster_bootstrap_ci,
    write_json,
    write_per_patient_csv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoder loader (lazy)
# ---------------------------------------------------------------------------


def _load_decoder(cfg: AELongitudinalConfig):
    """Construct the MAISI-v2 autoencoder for the decode-only path.

    Defensive cuDNN handling: the cu126 torch wheel does not ship cuDNN
    libraries, which leaves ``F.conv3d`` raising ``CUDNN_STATUS_NOT_INITIALIZED``
    at first use. Disable cuDNN globally so PyTorch uses the native
    implementation (slower but functional).
    """
    import torch  # local — keep top-level lighter for unit tests

    try:
        # Minimal probe: a tiny fp16 conv on the target device.
        if cfg.device.startswith("cuda"):
            x = torch.randn(1, 2, 4, 4, 4, device=cfg.device, dtype=torch.float16)
            torch.nn.Conv3d(2, 2, 3, padding=1).to(device=cfg.device, dtype=torch.float16)(x)
    except RuntimeError as err:
        if "CUDNN" in str(err).upper():
            logger.warning(
                "cuDNN unusable on this system (%s). Disabling cuDNN; expect ~2-4× slowdown.",
                err,
            )
            torch.backends.cudnn.enabled = False
        else:
            raise

    from menflow.maisi_autoencoder.config import MaisiV2Config
    from menflow.maisi_autoencoder.model import MaisiAutoencoder

    device = torch.device(cfg.device)
    maisi_cfg = MaisiV2Config()
    if cfg.num_splits != maisi_cfg.num_splits:
        maisi_cfg = dataclasses.replace(maisi_cfg, num_splits=cfg.num_splits)
        logger.info("MaisiV2Config num_splits override → %d", cfg.num_splits)
    model = MaisiAutoencoder.from_checkpoint(
        cfg.maisi_checkpoint,
        config=maisi_cfg,
        device=device,
        dtype=cfg.decode_dtype,
        strict=True,
    )
    return model, device


def _decode_one(model, z_block: np.ndarray, *, device, dtype) -> np.ndarray:
    """Decode all modalities of a single scan latent block.

    Parameters
    ----------
    z_block
        ``(M, C, H', W', D')`` numpy array.

    Returns
    -------
    np.ndarray
        ``(M, H_pad, W_pad, D_pad)`` float32 — the model's raw output in the
        [b_min, b_max] inference range. Inverse intensity rescaling is applied
        outside this function (it depends on per-scan percentile bounds).
    """
    import torch

    out_list: list[np.ndarray] = []
    for mi in range(z_block.shape[0]):
        z_t = torch.from_numpy(z_block[mi]).to(device=device, dtype=dtype)[None]
        x_hat = model.decode(z_t)  # (1, 1, H_pad, W_pad, D_pad)
        out_list.append(x_hat[0, 0].detach().cpu().float().numpy())
        del x_hat, z_t
    return np.stack(out_list, axis=0)


def _undo_pad_and_intensity(
    decoded: np.ndarray,
    *,
    source_shape: tuple[int, int, int],
    intensity_lower: np.ndarray,
    intensity_upper: np.ndarray,
    b_min: float = 0.0,
    b_max: float = 1.0,
) -> np.ndarray:
    """Crop ``(M, H_pad, W_pad, D_pad)`` back to source shape and undo rescaling."""
    sl = (slice(None),) + tuple(slice(0, s) for s in source_shape)
    cropped = decoded[sl]
    M = cropped.shape[0]
    out = np.empty_like(cropped, dtype=np.float32)
    for mi in range(M):
        lo = float(intensity_lower[mi])
        hi = float(intensity_upper[mi])
        scale = (hi - lo) / max(b_max - b_min, 1e-8)
        out[mi] = (cropped[mi] - b_min) * scale + lo
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AELongitudinalEngine:
    """Runs the full E2.5 diagnostic and persists artefacts."""

    def __init__(self, config: AELongitudinalConfig) -> None:
        self.config = config
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        cfg = self.config
        t0 = time.perf_counter()
        logger.info("E2.5 starting; output_dir=%s", cfg.output_dir)

        # -------- inputs --------
        for p in (cfg.unified_h5, cfg.latents_h5, cfg.maisi_checkpoint):
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        meta = lat_mod.read_latent_metadata(cfg.latents_h5)
        logger.info("Latent metadata: %s", meta)
        if set(cfg.modalities) - set(meta.modalities):
            missing = set(cfg.modalities) - set(meta.modalities)
            raise ValueError(f"requested modalities {missing} not present in latents H5")
        modality_indices = [meta.modalities.index(m) for m in cfg.modalities]

        # Probes (one per requested modality).
        probes: dict[str, RidgeProbe] = {}
        for m in cfg.modalities:
            if m not in cfg.e2_1_direction_npz:
                raise ValueError(f"no E2.1 direction.npz path supplied for modality {m!r}")
            probes[m] = RidgeProbe.load_npz(cfg.e2_1_direction_npz[m])
            logger.info(
                "probe[%s]: coef=%s intercept=%.4f",
                m,
                np.array_str(probes[m].coef, precision=3),
                probes[m].intercept,
            )

        # -------- cohort --------
        spread_ladder = (cfg.min_log_volume_spread,) + tuple(cfg.min_log_volume_spread_relaxed)
        triples, construction_log = select_cohort(
            unified_h5=Path(cfg.unified_h5),
            latents_h5=Path(cfg.latents_h5),
            min_timepoints=cfg.min_timepoints,
            min_spread_ladder=spread_ladder,
            min_effective_cohort=cfg.min_effective_cohort,
            max_patients=cfg.max_patients,
        )
        if not triples:
            raise RuntimeError("No patient triples after cohort filtering — cannot run E2.5.")

        # -------- decoder (lazy) --------
        model = None
        device = None
        if not cfg.skip_decode:
            try:
                model, device = _load_decoder(cfg)
            except Exception as e:
                logger.error("Decoder load failed (%s). Continuing without decode.", e)
                model = None

        # -------- main loop --------
        per_patient_records: list[dict] = []
        anatomy_curves: list[list[float]] = []
        rho_lin_joint_list: list[float] = []
        round_trip_pass_count = 0
        round_trip_attempt_count = 0
        qualitative_panels: list[dict] = []
        n_decodes_total = 0
        # Keyed by patient_id → (p1, p2, p3, log_volume_spread). The spread is
        # used to pick the most-informative triple per patient for the PCA.
        pooled_per_patient: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

        with h5py.File(cfg.unified_h5, "r") as src, h5py.File(cfg.latents_h5, "r") as lat:
            for ti, triple in enumerate(triples):
                t_start = time.perf_counter()
                record = self._process_patient(
                    triple=triple,
                    src=src,
                    lat=lat,
                    meta=meta,
                    modality_indices=modality_indices,
                    probes=probes,
                    model=model,
                    device=device,
                    qualitative_keep=ti < cfg.n_qualitative_panel_patients,
                )
                record["elapsed_seconds"] = float(time.perf_counter() - t_start)
                per_patient_records.append(record)
                if math.isfinite(record["rho_lin_joint"]):
                    rho_lin_joint_list.append(record["rho_lin_joint"])
                round_trip_attempt_count += 1
                if record.get("round_trip_pass", False):
                    round_trip_pass_count += 1
                if record.get("anatomy_ssim_curve") is not None:
                    anatomy_curves.append(record["anatomy_ssim_curve"])
                if record.get("qualitative_panel") is not None:
                    qualitative_panels.append(record["qualitative_panel"])
                n_decodes_total += int(record.get("n_decodes", 0))
                # Keep one pooled triple per patient for the PCA test
                # (largest spread, i.e. the first one encountered since cohort
                # ordering preserves per-patient grouping; we explicitly choose
                # the triple with the largest log_volume_spread instead).
                pid = triple.patient_id
                spread = float(triple.log_volume_spread)
                p1v = np.asarray(record["pooled_p1"], dtype=np.float64)
                p2v = np.asarray(record["pooled_p2"], dtype=np.float64)
                p3v = np.asarray(record["pooled_p3"], dtype=np.float64)
                if pid not in pooled_per_patient or spread > pooled_per_patient[pid][3]:
                    pooled_per_patient[pid] = (p1v, p2v, p3v, spread)
                if model is not None:
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                logger.info(
                    "[%d/%d] %s ρ_lin=%.3f β_vol=%.2f β_time=%.2f β_res=%.2f t=%.1fs",
                    ti + 1,
                    len(triples),
                    triple.patient_id,
                    record["rho_lin_joint"],
                    record["beta_vol"],
                    record["beta_time"],
                    record["beta_residual"],
                    record["elapsed_seconds"],
                )

        rho_lin_arr = np.asarray(rho_lin_joint_list, dtype=np.float64)
        clusters = np.asarray([r["patient_id"] for r in per_patient_records])

        # -------- null baseline (joint + pooled + masked) --------
        null = None
        delta_rho = {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        delta_rho_masked = dict(delta_rho)
        delta_rho_pooled = dict(delta_rho)
        if cfg.n_null_triples > 0:
            logger.info("Computing null baseline (n=%d) ...", cfg.n_null_triples)
            null = compute_null_baseline(
                latents_h5_path=str(cfg.latents_h5),
                unified_h5_path=str(cfg.unified_h5),
                n_triples=cfg.n_null_triples,
                seed=cfg.null_random_seed,
                use_joint=True,
            )
            if null.n_drawn >= 2 and rho_lin_arr.size >= 2:
                delta_rho = bootstrap_paired_diff(
                    rho_lin_arr,
                    null.rho_lin,
                    n_resamples=cfg.n_bootstrap_resamples,
                    alpha=cfg.bootstrap_ci_alpha,
                    seed=cfg.bootstrap_seed,
                )
        # Pooled null: cheap to compute alongside, just re-pool the same null
        # triples in the (M, C) feature space. Done in a small helper here to
        # avoid loading latents twice.
        pooled_null = None
        masked_null = None
        if null is not None and null.n_drawn > 0:
            pooled_null, masked_null = self._null_pooled_and_masked(
                latents_h5_path=str(cfg.latents_h5),
                unified_h5_path=str(cfg.unified_h5),
                n_triples=null.n_drawn,
                seed=cfg.null_random_seed,
            )
            obs_masked = _col_local(per_patient_records, "rho_lin_masked")
            obs_pooled = _col_local(per_patient_records, "rho_lin_pooled")
            if obs_masked[np.isfinite(obs_masked)].size >= 2 and masked_null.size >= 2:
                delta_rho_masked = bootstrap_paired_diff(
                    obs_masked,
                    masked_null,
                    n_resamples=cfg.n_bootstrap_resamples,
                    alpha=cfg.bootstrap_ci_alpha,
                    seed=cfg.bootstrap_seed + 10,
                )
            if obs_pooled[np.isfinite(obs_pooled)].size >= 2 and pooled_null.size >= 2:
                delta_rho_pooled = bootstrap_paired_diff(
                    obs_pooled,
                    pooled_null,
                    n_resamples=cfg.n_bootstrap_resamples,
                    alpha=cfg.bootstrap_ci_alpha,
                    seed=cfg.bootstrap_seed + 11,
                )

        # -------- PCA subspace test --------
        pca_result_dict: dict[str, Any] | None = None
        pca_obj = None
        if len(pooled_per_patient) >= 3:
            pca_input = [
                (v[0], v[1], v[2]) for v in pooled_per_patient.values()
            ]
            try:
                pca_obj = pca_longitudinal_direction(pca_input)
                pca_result_dict = {
                    "variance_explained_pc1": pca_obj.variance_explained,
                    "singular_values": pca_obj.singular_values.tolist(),
                    "cos_z2_minus_z1_mean": float(np.mean(pca_obj.cos_z2_minus_z1)),
                    "cos_z2_minus_z1_median": float(np.median(pca_obj.cos_z2_minus_z1)),
                    "cos_z2_minus_z1_per_patient": pca_obj.cos_z2_minus_z1.tolist(),
                    "cos_z3_minus_z1_mean": float(np.mean(pca_obj.cos_z3_minus_z1)),
                    "n_patients_pca": len(pca_input),
                }
            except Exception:
                logger.exception("PCA subspace fit failed; continuing without it.")

        # -------- aggregate metrics --------
        def _col(name: str) -> np.ndarray:
            return np.asarray([r.get(name, np.nan) for r in per_patient_records], dtype=np.float64)

        def _boot_cluster(name: str, seed_offset: int) -> dict[str, float]:
            return cluster_bootstrap_ci(
                _col(name),
                clusters,
                n_resamples=cfg.n_bootstrap_resamples,
                alpha=cfg.bootstrap_ci_alpha,
                seed=cfg.bootstrap_seed + seed_offset,
            )

        aggregate: dict[str, Any] = {
            "n_effective": len({r["patient_id"] for r in per_patient_records}),
            "n_triples": len(per_patient_records),
            "modalities_used": list(cfg.modalities),
            "rho_lin": _boot_cluster("rho_lin_joint", 0),
            "rho_lin_masked": _boot_cluster("rho_lin_masked", 20),
            "rho_lin_pooled": _boot_cluster("rho_lin_pooled", 21),
            "rho_lin_null": (
                bootstrap_ci(
                    null.rho_lin,
                    n_resamples=cfg.n_bootstrap_resamples,
                    alpha=cfg.bootstrap_ci_alpha,
                    seed=cfg.bootstrap_seed + 1,
                )
                if null is not None
                else None
            ),
            "rho_lin_null_masked": (
                bootstrap_ci(
                    masked_null,
                    n_resamples=cfg.n_bootstrap_resamples,
                    alpha=cfg.bootstrap_ci_alpha,
                    seed=cfg.bootstrap_seed + 22,
                )
                if masked_null is not None and masked_null.size > 0
                else None
            ),
            "rho_lin_null_pooled": (
                bootstrap_ci(
                    pooled_null,
                    n_resamples=cfg.n_bootstrap_resamples,
                    alpha=cfg.bootstrap_ci_alpha,
                    seed=cfg.bootstrap_seed + 23,
                )
                if pooled_null is not None and pooled_null.size > 0
                else None
            ),
            "delta_rho": delta_rho,
            "delta_rho_masked": delta_rho_masked,
            "delta_rho_pooled": delta_rho_pooled,
            "pca": pca_result_dict,
            "beta_residual": _boot_cluster("beta_residual", 2),
            "ssim_at_beta_vol": _boot_cluster("ssim_at_beta_vol", 3),
            "ssim_nontumour_mean": _boot_cluster("ssim_nontumour_mean", 4),
            "ssim_nontumour_min_across_beta": _boot_cluster("ssim_nontumour_min", 5),
            "round_trip_ssim_mean": _boot_cluster("round_trip_ssim_mean", 6),
            "round_trip_pass_fraction": (
                round_trip_pass_count / max(round_trip_attempt_count, 1)
            ),
            "dice_at_beta_vol": None,
            "cohort_construction_log": construction_log,
            "n_null_triples_drawn": null.n_drawn if null else 0,
            "n_decodes_total": int(n_decodes_total),
            "completed_at": _dt.datetime.now(_dt.UTC).isoformat(),
        }

        caveats: list[str] = []
        if construction_log.get("below_min_effective_cohort"):
            caveats.append(
                f"n_effective={aggregate['n_effective']}<min_effective_cohort={cfg.min_effective_cohort}"
            )
        relaxed_to = construction_log.get("min_log_volume_spread_used")
        if relaxed_to is not None and relaxed_to < cfg.min_log_volume_spread:
            caveats.append(f"c4_relaxed_to_{relaxed_to:.2f}")

        if not cfg.skip_decode:
            rt_frac = aggregate["round_trip_pass_fraction"]
            if rt_frac < cfg.round_trip_pass_fraction:
                caveats.append(
                    f"round_trip_pass_fraction={rt_frac:.2f}<threshold={cfg.round_trip_pass_fraction}"
                )
        else:
            caveats.append("decode_skipped")

        decision = evaluate_decision(aggregate, cfg, caveats_extra=caveats)

        # -------- persist --------
        out = cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)
        # Persist per-patient CSV without the qualitative_panel / anatomy_ssim_curve fields.
        csv_records = []
        for r in per_patient_records:
            row = {
                k: v
                for k, v in r.items()
                if k not in (
                    "qualitative_panel",
                    "anatomy_ssim_curve",
                    "pooled_p1",
                    "pooled_p2",
                    "pooled_p3",
                )
            }
            csv_records.append(row)
        write_per_patient_csv(csv_records, out / "per_patient.csv")
        write_json(
            {**aggregate, "decision": decision.to_dict(), "config": _config_snapshot(cfg)},
            out / "aggregate.json",
        )
        write_json(decision.to_dict(), out / "decision.json")

        # Figures
        try:
            self._write_figures(
                per_patient_records,
                anatomy_curves,
                qualitative_panels,
                null,
                out,
                pca_dict=pca_result_dict,
            )
        except Exception:
            logger.exception("Plotting failed (non-fatal); continuing.")

        # Local report markdown
        self._write_report(decision, aggregate, out)

        elapsed = time.perf_counter() - t0
        logger.info(
            "E2.5 done in %.1f s; regime=%s; outputs at %s",
            elapsed,
            decision.regime,
            out,
        )
        return out

    # ------------------------------------------------------------------
    # Per-patient processing
    # ------------------------------------------------------------------

    def _process_patient(
        self,
        *,
        triple: PatientTriplet,
        src: h5py.File,
        lat: h5py.File,
        meta: lat_mod.LatentMetadata,
        modality_indices: list[int],
        probes: dict[str, RidgeProbe],
        model,
        device,
        qualitative_keep: bool,
    ) -> dict[str, Any]:
        cfg = self.config
        ids = triple.scan_indices  # (i1, i2, i3) into src+lat

        # Latents — select only the requested modality indices to save RAM.
        z_full = lat["latents"][ids[0]].astype(np.float32)[modality_indices]  # (M, C, H', W', D')
        z2 = lat["latents"][ids[1]].astype(np.float32)[modality_indices]
        z3 = lat["latents"][ids[2]].astype(np.float32)[modality_indices]
        z1 = z_full
        del z_full

        # Intensity bounds.
        i_lo = np.asarray(lat["intensity_lower"][:], dtype=np.float32)[:, modality_indices]
        i_hi = np.asarray(lat["intensity_upper"][:], dtype=np.float32)[:, modality_indices]

        # Source images and segs.
        x1 = np.asarray(src["images"][ids[0]][modality_indices], dtype=np.float32)
        x2 = np.asarray(src["images"][ids[1]][modality_indices], dtype=np.float32)
        x3 = np.asarray(src["images"][ids[2]][modality_indices], dtype=np.float32)
        m1 = np.asarray(src["segmentations"][ids[0]], dtype=np.int32)
        m3 = np.asarray(src["segmentations"][ids[2]], dtype=np.int32)
        m2 = np.asarray(src["segmentations"][ids[1]], dtype=np.int32)
        label_set = np.asarray(cfg.mask_label_set, dtype=np.int32)
        tumour1 = np.isin(m1, label_set)
        tumour3 = np.isin(m3, label_set)
        tumour2 = np.isin(m2, label_set)

        # Union mask + dilation at native resolution, then project to latent.
        union_native = tumour1 | tumour3
        union_dilated = lat_mod.dilate_mask_3d(union_native, cfg.mask_dilation_voxels)
        union_lat = lat_mod.project_to_latent_grid(union_dilated, meta=meta)

        # ---- Test 1: three ρ_lin variants ----
        # (a) joint over the full spatial latent (576 k dim) — the spec metric.
        # (b) mask-localised: residual restricted to voxels inside union_lat.
        # (c) mask-pooled: feature-space ρ_lin on the (M, C) = 16-D vector.
        try:
            res_joint = linearity_residual(z1, z2, z3)
            rho_lin_joint = float(res_joint.rho_lin)
            beta_star_joint = float(res_joint.beta_star)
        except DegenerateTripletError:
            rho_lin_joint = float("nan")
            beta_star_joint = float("nan")

        if union_lat.any():
            try:
                res_masked = linearity_residual(z1, z2, z3, mask_latent=union_lat)
                rho_lin_masked = float(res_masked.rho_lin)
                beta_star_masked = float(res_masked.beta_star)
            except DegenerateTripletError:
                rho_lin_masked = float("nan")
                beta_star_masked = float("nan")
        else:
            rho_lin_masked = float("nan")
            beta_star_masked = float("nan")

        pooled1 = lat_mod.mask_pool_per_modality(z1, union_lat)  # (M, C)
        pooled2 = lat_mod.mask_pool_per_modality(z2, union_lat)
        pooled3 = lat_mod.mask_pool_per_modality(z3, union_lat)
        if union_lat.any():
            try:
                res_pooled = linearity_residual_pooled(pooled1, pooled2, pooled3)
                rho_lin_pooled = float(res_pooled.rho_lin)
                beta_star_pooled = float(res_pooled.beta_star)
            except DegenerateTripletError:
                rho_lin_pooled = float("nan")
                beta_star_pooled = float("nan")
        else:
            rho_lin_pooled = float("nan")
            beta_star_pooled = float("nan")

        rho_lin_per_mod: dict[str, float] = {}
        for mi, m in enumerate(cfg.modalities):
            try:
                rm = linearity_residual(z1[mi], z2[mi], z3[mi]).rho_lin
            except DegenerateTripletError:
                rm = float("nan")
            rho_lin_per_mod[m] = float(rm)

        # ---- Test 2: volume-matched β ----
        per_mod_predict = [probes[m].predict for m in cfg.modalities]
        log_v2_target = triple.log_volumes[1]
        beta_vol, betas, log_v_hat = volume_matched_beta(
            z1,
            z3,
            mask_latent=union_lat,
            log_v_target=log_v2_target,
            probe_predict_per_modality=per_mod_predict,
            n_grid=cfg.beta_grid_size,
        )
        # β_time
        t1, t2, t3 = triple.timepoint_idx
        if t3 - t1 == 0:
            beta_time = 0.5
        else:
            beta_time = float((t2 - t1) / (t3 - t1))
        beta_time = float(np.clip(beta_time, 0.0, 1.0))
        beta_residual = abs(beta_vol - beta_time)

        # ---- Test 3 + round-trip: decode at six β's + endpoints ----
        record: dict[str, Any] = {
            "patient_id": triple.patient_id,
            "scan_indices": list(triple.scan_indices),
            "timepoint_idx": list(triple.timepoint_idx),
            "log_v1": float(triple.log_volumes[0]),
            "log_v2": float(triple.log_volumes[1]),
            "log_v3": float(triple.log_volumes[2]),
            "log_volume_spread": float(triple.log_volume_spread),
            "rho_lin_joint": rho_lin_joint,
            "rho_lin_beta_star": beta_star_joint,
            "rho_lin_masked": rho_lin_masked,
            "rho_lin_masked_beta_star": beta_star_masked,
            "rho_lin_pooled": rho_lin_pooled,
            "rho_lin_pooled_beta_star": beta_star_pooled,
            "pooled_p1": pooled1.ravel().tolist(),
            "pooled_p2": pooled2.ravel().tolist(),
            "pooled_p3": pooled3.ravel().tolist(),
            "beta_vol": beta_vol,
            "beta_time": beta_time,
            "beta_residual": beta_residual,
            "log_v_hat_at_beta_vol": float(log_v_hat[int(np.nanargmin(np.abs(log_v_hat - log_v2_target)))])
            if np.any(np.isfinite(log_v_hat))
            else float("nan"),
            "dice_at_beta_vol": float("nan"),  # deferred
            "ssim_at_beta_vol": float("nan"),
            "ssim_nontumour_mean": float("nan"),
            "ssim_nontumour_min": float("nan"),
            "round_trip_ssim_mean": float("nan"),
            "round_trip_pass": False,
            "n_decodes": 0,
            "qualitative_panel": None,
            "anatomy_ssim_curve": None,
        }
        for m, v in rho_lin_per_mod.items():
            record[f"rho_lin_{m}"] = v
        if cfg.skip_decode or model is None:
            return record

        # ---- Decode budget per patient ----
        # Round-trip @ each endpoint (3) + anatomy β grid + β_vol if not already.
        anatomy_betas = list(cfg.anatomy_beta_grid)
        if not any(abs(b - beta_vol) < 1e-6 for b in anatomy_betas):
            anatomy_betas_with_vol = sorted(anatomy_betas + [beta_vol])
        else:
            anatomy_betas_with_vol = anatomy_betas

        # Round-trip on endpoints.
        rt_ssims: list[float] = []
        for ep_i, ep_z, ep_x, ep_idx in (
            (0, z1, x1, ids[0]),
            (2, z3, x3, ids[2]),
            (1, z2, x2, ids[1]),
        ):
            decoded_pad = _decode_one(model, ep_z, device=device, dtype=_torch_dtype(cfg.decode_dtype))
            decoded = _undo_pad_and_intensity(
                decoded_pad,
                source_shape=tuple(int(s) for s in src.attrs["spatial_shape"]),
                intensity_lower=i_lo[ep_idx],
                intensity_upper=i_hi[ep_idx],
            )
            # Mean SSIM across modalities at native resolution.
            ssim_modalities = [
                ssim_3d(decoded[mi], ep_x[mi]) for mi in range(decoded.shape[0])
            ]
            rt_ssims.append(float(np.mean(ssim_modalities)))
            record["n_decodes"] += int(decoded.shape[0])
            del decoded, decoded_pad
        record["round_trip_ssim_mean"] = float(np.mean(rt_ssims))
        record["round_trip_ssim_t1"] = rt_ssims[0]
        record["round_trip_ssim_t3"] = rt_ssims[1]
        record["round_trip_ssim_t2"] = rt_ssims[2]
        record["round_trip_pass"] = record["round_trip_ssim_mean"] >= cfg.round_trip_ssim_threshold

        # Anatomy + β_vol decodes.
        nontumour_mask_native = ~union_dilated  # bool (H, W, D)
        anatomy_ssims: list[float] = []
        ssim_at_beta_vol = float("nan")
        decoded_for_panel: list[tuple[float, np.ndarray]] = []
        per_scan_intensity = (i_lo[ids[0]] + i_hi[ids[0]]) * 0.0  # placeholder
        for beta in anatomy_betas_with_vol:
            z_beta = (1.0 - beta) * z1 + beta * z3
            # Interpolated intensity bounds (endpoint mean).
            i_lo_b = (1.0 - beta) * i_lo[ids[0]] + beta * i_lo[ids[2]]
            i_hi_b = (1.0 - beta) * i_hi[ids[0]] + beta * i_hi[ids[2]]
            decoded_pad = _decode_one(model, z_beta, device=device, dtype=_torch_dtype(cfg.decode_dtype))
            decoded = _undo_pad_and_intensity(
                decoded_pad,
                source_shape=tuple(int(s) for s in src.attrs["spatial_shape"]),
                intensity_lower=i_lo_b,
                intensity_upper=i_hi_b,
            )
            record["n_decodes"] += int(decoded.shape[0])
            # Choose reference endpoint by proximity.
            x_ref = x1 if beta < 0.5 else x3
            nt_per_mod = [
                ssim_masked(decoded[mi], x_ref[mi], nontumour_mask_native)
                for mi in range(decoded.shape[0])
            ]
            if beta in cfg.anatomy_beta_grid:
                anatomy_ssims.append(float(np.mean(nt_per_mod)))
            if abs(beta - beta_vol) < 1e-6:
                ssim_per_mod = [
                    ssim_3d(decoded[mi], x2[mi]) for mi in range(decoded.shape[0])
                ]
                ssim_at_beta_vol = float(np.mean(ssim_per_mod))
            if qualitative_keep:
                # Keep mid-axial slice of t1c (first modality) only.
                decoded_for_panel.append((float(beta), decoded[0].copy()))
            del decoded, decoded_pad

        record["ssim_at_beta_vol"] = ssim_at_beta_vol
        if anatomy_ssims:
            record["ssim_nontumour_mean"] = float(np.mean(anatomy_ssims))
            record["ssim_nontumour_min"] = float(np.min(anatomy_ssims))
            record["anatomy_ssim_curve"] = anatomy_ssims
        for bi, ssim_v in zip(cfg.anatomy_beta_grid, anatomy_ssims):
            record[f"ssim_nt_at_b{int(bi*100):03d}"] = float(ssim_v)

        if qualitative_keep:
            record["qualitative_panel"] = {
                "patient_id": triple.patient_id,
                "rho_lin": rho_lin_joint,
                "real_t1": x1[0],
                "real_t2": x2[0],
                "real_t3": x3[0],
                "decoded": decoded_for_panel,
            }
        return record

    # ------------------------------------------------------------------
    # Pooled & masked null baseline (re-uses the same volume-ordered triples
    # as the joint null so the per-rho comparison is paired by sample).
    # ------------------------------------------------------------------

    def _null_pooled_and_masked(
        self,
        *,
        latents_h5_path: str,
        unified_h5_path: str,
        n_triples: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        rng = np.random.default_rng(seed)
        meta = lat_mod.read_latent_metadata(latents_h5_path)
        mod_idx = [meta.modalities.index(m) for m in cfg.modalities]
        with h5py.File(unified_h5_path, "r") as f:
            log_vol = f["features/log_volume_cm3"][:].astype(np.float64)
            has_seg = f["has_segmentation"][:].astype(bool)
            offsets = f["longitudinal/patient_offsets"][:].astype(int)
            patient_list = [
                s.decode() if isinstance(s, bytes) else s
                for s in f["longitudinal/patient_list"][:]
            ]
            label_set = np.asarray(cfg.mask_label_set, dtype=np.int32)
            seg_ds = f["segmentations"]
            valid = has_seg & np.isfinite(log_vol)
            per_patient_valid: list[np.ndarray] = []
            for pi in range(len(patient_list)):
                rows = np.arange(int(offsets[pi]), int(offsets[pi + 1]))
                rows = rows[valid[rows]]
                per_patient_valid.append(rows)
            eligible = [pi for pi, rs in enumerate(per_patient_valid) if rs.size > 0]
            with h5py.File(latents_h5_path, "r") as lat:
                latents = lat["latents"][:].astype(np.float32)[:, mod_idx]
            n_draw = 0
            rho_pooled = np.empty(n_triples, dtype=np.float64)
            rho_masked = np.empty(n_triples, dtype=np.float64)
            max_attempts = 30 * n_triples
            attempts = 0
            while n_draw < n_triples and attempts < max_attempts:
                attempts += 1
                ps = rng.choice(eligible, size=3, replace=False)
                rs = [int(rng.choice(per_patient_valid[p])) for p in ps]
                vols = log_vol[rs]
                order = np.argsort(vols)
                if (
                    vols[order[0]] >= vols[order[1]]
                    or vols[order[1]] >= vols[order[2]]
                ):
                    continue
                a, b, c = (int(np.asarray(rs)[i]) for i in order)
                seg_a = np.isin(np.asarray(seg_ds[a], dtype=np.int32), label_set)
                seg_c = np.isin(np.asarray(seg_ds[c], dtype=np.int32), label_set)
                union = lat_mod.dilate_mask_3d(seg_a | seg_c, cfg.mask_dilation_voxels)
                union_lat = lat_mod.project_to_latent_grid(union, meta=meta)
                if not union_lat.any():
                    continue
                za, zb, zc = latents[a], latents[b], latents[c]
                # masked
                try:
                    r_m = linearity_residual(za, zb, zc, mask_latent=union_lat).rho_lin
                except DegenerateTripletError:
                    continue
                # pooled
                pa = lat_mod.mask_pool_per_modality(za, union_lat)
                pb = lat_mod.mask_pool_per_modality(zb, union_lat)
                pc = lat_mod.mask_pool_per_modality(zc, union_lat)
                try:
                    r_p = linearity_residual_pooled(pa, pb, pc).rho_lin
                except DegenerateTripletError:
                    continue
                rho_masked[n_draw] = r_m
                rho_pooled[n_draw] = r_p
                n_draw += 1
            return rho_pooled[:n_draw], rho_masked[:n_draw]

    # ------------------------------------------------------------------
    # Figures + report
    # ------------------------------------------------------------------

    def _write_figures(
        self,
        per_patient_records: list[dict],
        anatomy_curves: list[list[float]],
        qualitative_panels: list[dict],
        null,
        out: Path,
        *,
        pca_dict: dict | None = None,
    ) -> None:
        from experiments.E2.E2_5_ae_longitudinal.analysis import plotting

        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        rho_lin = np.asarray([r.get("rho_lin_joint", np.nan) for r in per_patient_records])
        rho_null = null.rho_lin if null is not None else np.array([])
        plotting.plot_rho_lin_distribution(
            rho_lin=rho_lin[np.isfinite(rho_lin)],
            rho_lin_null=rho_null[np.isfinite(rho_null)] if rho_null.size else rho_null,
            rho_pass=cfg.rho_lin_pass,
            rho_intermediate=cfg.rho_lin_intermediate,
            out_path=fig_dir / "rho_lin_distribution.png",
        )
        import pandas as pd

        df = pd.DataFrame.from_records(per_patient_records)
        plotting.plot_beta_residual_scatter(df, out_path=fig_dir / "interpolation_quality_scatter.png")

        if anatomy_curves:
            betas = np.asarray(self.config.anatomy_beta_grid, dtype=float)
            arr = np.full((len(anatomy_curves), len(betas)), np.nan, dtype=float)
            for i, curve in enumerate(anatomy_curves):
                cl = min(len(curve), len(betas))
                arr[i, :cl] = curve[:cl]
            plotting.plot_anatomy_ssim_curves(arr, betas=betas, out_path=fig_dir / "anatomy_ssim_curves.png")

        if qualitative_panels:
            plotting.plot_qualitative_panel(panels=qualitative_panels, out_path=fig_dir / "qualitative_panel.png")

        plotting.plot_rho_variants(
            df,
            out_path=fig_dir / "rho_variants.png",
            rho_pass=cfg.rho_lin_pass,
            rho_intermediate=cfg.rho_lin_intermediate,
        )

        if pca_dict is not None:
            plotting.plot_pca_alignment(pca_dict, out_path=fig_dir / "pca_alignment.png")

    def _write_report(self, decision, aggregate: dict, out: Path) -> None:
        cfg = self.config
        cohort_log = aggregate.get("cohort_construction_log", {})

        def fmt(d: dict | None) -> str:
            if not d or d.get("mean") is None:
                return "n/a"
            return f"{d['mean']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"

        pca = aggregate.get("pca") or {}
        body = f"""# E2.5 — local-run report

**Regime:** `{decision.regime}` → `{decision.downstream_action}`

**Rationale:**

> {decision.rationale}

**Caveats:** {', '.join(decision.caveats) if decision.caveats else 'none'}

## Headline metrics (cluster bootstrap, 95 % CI)

| Metric | Same-patient | Null cross-patient | Δρ |
|---|---|---|---|
| ρ_lin pooled (16-D) | {fmt(aggregate.get('rho_lin_pooled'))} | {fmt(aggregate.get('rho_lin_null_pooled'))} | {fmt(aggregate.get('delta_rho_pooled'))} |
| ρ_lin masked | {fmt(aggregate.get('rho_lin_masked'))} | {fmt(aggregate.get('rho_lin_null_masked'))} | {fmt(aggregate.get('delta_rho_masked'))} |
| ρ_lin full latent | {fmt(aggregate.get('rho_lin'))} | {fmt(aggregate.get('rho_lin_null'))} | {fmt(aggregate.get('delta_rho'))} |

| Metric | Value |
|---|---|
| PCA PC1 variance explained | {pca.get('variance_explained_pc1', float('nan')):.3f} |
| Mean cos(z²−z¹, PC1) | {pca.get('cos_z2_minus_z1_mean', float('nan')):.3f} |
| Mean cos(z³−z¹, PC1) | {pca.get('cos_z3_minus_z1_mean', float('nan')):.3f} |
| n_patients (PCA) | {pca.get('n_patients_pca', 'n/a')} |
| β_residual | {fmt(aggregate.get('beta_residual'))} |
| SSIM at β_vol | {fmt(aggregate.get('ssim_at_beta_vol'))} |
| Non-tumour SSIM | {fmt(aggregate.get('ssim_nontumour_mean'))} |
| Round-trip SSIM mean | {fmt(aggregate.get('round_trip_ssim_mean'))} |
| Round-trip pass fraction | {aggregate.get('round_trip_pass_fraction', 0.0):.2%} |

## Cohort

- n_effective patients = {aggregate['n_effective']} (target ≥ {cfg.min_effective_cohort})
- n_triples = {aggregate.get('n_triples', 'n/a')}
- C4 spread used = {cohort_log.get('min_log_volume_spread_used')}
- patients with monotone-3 triple = {cohort_log.get('n_patients_with_monotone_triple')}
- enumerate_all_triples = {cohort_log.get('enumerate_all_triples', True)}
- ladder attempts = {cohort_log.get('attempts')}

## Downstream impact

- **R1** → E3.2 trains standard volume-conditional FM with anchor propagation; the inference sampler treats the latent space as approximately Euclidean for linear interpolation (Mechanism 1).
- **R2** → add Option-γ anchor-similarity loss on non-tumour latent voxels to E3.2.
- **R3** → halt E3.2 until MAISI decoder is LoRA-adapted with Δ-LFM ArcRank loss.

## Caveats

The Dice@β_vol gate is **deferred**: no Python-callable nnU-Net is wired into the local pipeline; closing the gate requires re-running on Picasso with the Docker BraTS segmenter. The regime call above uses ρ_lin + non-tumour SSIM + Δρ only.
"""
        (out / "E2_5_local_report.md").write_text(body)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _col_local(records: list[dict], name: str) -> np.ndarray:
    return np.asarray([r.get(name, np.nan) for r in records], dtype=np.float64)


def _torch_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _config_snapshot(cfg: AELongitudinalConfig) -> dict:
    """Serialise the config (paths -> str) for embedding in aggregate.json."""
    d = dataclasses.asdict(cfg)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, dict) and all(isinstance(p, Path) for p in v.values()):
            d[k] = {kk: str(vv) for kk, vv in v.items()}
    return d
