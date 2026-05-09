"""E2.3 — Spatial localization: tumor-region vs global-pooling probe comparison.

Establishes whether tumor-masked latent features (z_tumor) carry substantially
more volume information than global-average pooled features (z_global) and a
matched random-region baseline (z_random). FWL residualization rules out
demographic-confound explanations for any advantage.

Decision operator: "global" / "local" / "strongly_local" (consumed by E2.4).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.E2_3_spatial_localization.analysis.plotting import make_all
from menflow.directions.fwl import fwl_residualize
from menflow.probes.linear import fit_linear_probe
from menflow.probes.metrics import r2_score
from menflow.stats.bootstrap import patient_level_bootstrap_r2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpatialRoutineConfig:
    """All parameters governing a single E2.3 run.

    Attributes
    ----------
    features_h5:
        Path to the per-modality features H5 produced by routines/compute_features.
        Must contain z_tumor, z_global, z_random, log_volume, scan_ids,
        patient_ids, mask_lat_present.
    source_h5:
        Unified BraTS-MEN H5; used for laterality computation when cache is absent.
    clinical_xlsx:
        BraTS-MEN supplementary clinical xlsx for FWL covariates.
    output_dir:
        Where result.json, bootstrap_scores.npz, and figures are written.
    n_boot:
        Number of patient-level bootstrap replicates (default 1000).
    seed:
        RNG seed for reproducibility.
    log_level:
        Python logging level string.
    laterality_cache_csv:
        Optional CSV path caching laterality labels indexed by scan_id.
        Reuses E2.2's cache if provided.
    synthetic:
        If True, run entirely in memory with planted synthetic data.
    synthetic_n:
        Number of synthetic scans (ignored when synthetic=False).
    synthetic_c:
        Latent dimensionality for synthetic data.
    synthetic_global_strength:
        Coupling coefficient from tumor features to global features (planted
        localization signal strength).
    """

    features_h5: Path
    source_h5: Path
    clinical_xlsx: Path
    output_dir: Path
    n_boot: int = 1000
    seed: int = 0
    log_level: str = "INFO"
    laterality_cache_csv: Path | None = None
    synthetic: bool = False
    synthetic_n: int = 200
    synthetic_c: int = 4
    synthetic_global_strength: float = 0.125

    @classmethod
    def from_yaml(cls, path: Path | str) -> SpatialRoutineConfig:
        """Load config from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML config file.

        Returns
        -------
        SpatialRoutineConfig
        """
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        raw.pop("slurm", None)

        path_keys = ("features_h5", "source_h5", "clinical_xlsx", "output_dir")
        for k in path_keys:
            if k in raw and raw[k] is not None:
                raw[k] = Path(str(raw[k])).expanduser()

        if "laterality_cache_csv" in raw and raw["laterality_cache_csv"] is not None:
            raw["laterality_cache_csv"] = Path(raw["laterality_cache_csv"]).expanduser()

        return cls(**raw)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpatialResult:
    """All metrics from one E2.3 run.

    Attributes
    ----------
    r2_global:
        In-sample R² of global-pooling probe (raw).
    r2_tumor:
        In-sample R² of tumor-masked probe (raw).
    r2_random:
        In-sample R² of random-region probe (raw; implementation guard).
    r2_global_ci:
        95% bootstrap CI for r2_global as (lo, hi).
    r2_tumor_ci:
        95% bootstrap CI for r2_tumor as (lo, hi).
    r2_random_ci:
        95% bootstrap CI for r2_random as (lo, hi).
    r2_global_post_fwl:
        In-sample R² of global-pooling probe after FWL residualization.
    r2_tumor_post_fwl:
        In-sample R² of tumor-masked probe after FWL residualization.
    r2_random_post_fwl:
        In-sample R² of random-region probe after FWL residualization.
    r2_global_post_fwl_ci:
        95% bootstrap CI for r2_global_post_fwl as (lo, hi).
    r2_tumor_post_fwl_ci:
        95% bootstrap CI for r2_tumor_post_fwl as (lo, hi).
    r2_random_post_fwl_ci:
        95% bootstrap CI for r2_random_post_fwl as (lo, hi).
    delta_r2_spatial:
        r2_tumor - r2_global (raw gap).
    delta_r2_post_fwl:
        r2_tumor_post_fwl - r2_global_post_fwl (confounder-corrected gap).
    operator:
        One of "global", "local", "strongly_local". Decided on delta_r2_post_fwl.
    n_scans_used:
        Number of scans after filtering.
    n_imputed_for_fwl:
        Count of scans where at least one covariate was imputed.
    impl_guard_pass:
        True iff r2_random < r2_global + 0.05 AND r2_random < r2_tumor - 0.30.
    """

    r2_global: float
    r2_tumor: float
    r2_random: float
    r2_global_ci: tuple[float, float]
    r2_tumor_ci: tuple[float, float]
    r2_random_ci: tuple[float, float]
    r2_global_post_fwl: float
    r2_tumor_post_fwl: float
    r2_random_post_fwl: float
    r2_global_post_fwl_ci: tuple[float, float]
    r2_tumor_post_fwl_ci: tuple[float, float]
    r2_random_post_fwl_ci: tuple[float, float]
    delta_r2_spatial: float
    delta_r2_post_fwl: float
    operator: str
    n_scans_used: int
    n_imputed_for_fwl: int
    impl_guard_pass: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_operator(delta: float) -> str:
    """Map a ΔR² value to an operator string.

    Parameters
    ----------
    delta:
        r2_tumor_post_fwl - r2_global_post_fwl.

    Returns
    -------
    str
        One of "strongly_local", "local", or "global".
    """
    if delta >= 0.15:
        return "strongly_local"
    if delta >= 0.05:
        return "local"
    return "global"


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _compute_r2_and_bootstrap(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_fn: Any,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    """Fit probe, compute point R², and bootstrap CI.

    Parameters
    ----------
    z:
        Feature matrix (N, C).
    y:
        Target vector (N,).
    groups:
        Patient IDs (N,) for grouped CV.
    fit_fn:
        Callable (z, y, g) -> model with .predict().
    n_boot:
        Bootstrap replicates.
    seed:
        RNG seed.

    Returns
    -------
    r2 : float
        In-sample R² on full data.
    ci_lo : float
        Lower 95% CI bound.
    ci_hi : float
        Upper 95% CI bound.
    scores : np.ndarray
        All bootstrap R² values.
    """
    model = fit_fn(z, y, groups)
    r2 = float(r2_score(model.predict(z), y))
    ci_lo, ci_hi, scores = patient_level_bootstrap_r2(
        z, y, groups, fit_fn, n_boot=n_boot, seed=seed
    )
    return r2, ci_lo, ci_hi, scores


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_spatial(cfg: SpatialRoutineConfig) -> SpatialResult:
    """Execute the full E2.3 spatial localization pipeline.

    Parameters
    ----------
    cfg:
        Fully validated experiment configuration.

    Returns
    -------
    SpatialResult
        All scalar metrics and operator decision.
    """
    # Step 1: logging and output dir
    logging.getLogger().setLevel(cfg.log_level)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("E2.3 spatial localization starting. output_dir=%s", cfg.output_dir)

    # Step 2: data loading
    if cfg.synthetic:
        logger.info(
            "Synthetic mode: N=%d, C=%d, global_strength=%.3f",
            cfg.synthetic_n,
            cfg.synthetic_c,
            cfg.synthetic_global_strength,
        )
        (
            z_tumor,
            z_global,
            z_random,
            log_v,
            patient_ids,
            scan_ids,
            covariates_df,
            n_imputed_for_fwl,
        ) = _make_synthetic_data(cfg)
    else:
        (
            z_tumor,
            z_global,
            z_random,
            log_v,
            patient_ids,
            scan_ids,
            covariates_df,
            n_imputed_for_fwl,
        ) = _load_real_data(cfg)

    groups = patient_ids
    y = log_v
    n_scans_used = len(y)
    logger.info("Scans used: N=%d, n_imputed_for_fwl=%d", n_scans_used, n_imputed_for_fwl)

    # Step 5: fit_fn closure
    def _fit(z: np.ndarray, y: np.ndarray, g: np.ndarray) -> object:
        model, _, _ = fit_linear_probe(z, y, g)
        return model

    # Step 6: raw R² + bootstrap for each pooling
    logger.info("Computing raw R² and bootstrap CIs...")
    r2_global, ci_lo_g, ci_hi_g, scores_global = _compute_r2_and_bootstrap(
        z_global, y, groups, _fit, cfg.n_boot, cfg.seed
    )
    r2_tumor, ci_lo_t, ci_hi_t, scores_tumor = _compute_r2_and_bootstrap(
        z_tumor, y, groups, _fit, cfg.n_boot, cfg.seed
    )
    r2_random, ci_lo_r, ci_hi_r, scores_random = _compute_r2_and_bootstrap(
        z_random, y, groups, _fit, cfg.n_boot, cfg.seed
    )
    logger.info(
        "Raw R²: global=%.4f, tumor=%.4f, random=%.4f",
        r2_global,
        r2_tumor,
        r2_random,
    )

    # Step 7: FWL residualization (once, reuse y_resid)
    logger.info("Running FWL residualization...")
    z_global_resid, y_resid = fwl_residualize(z_global, y, covariates_df, groups)
    z_tumor_resid, _ = fwl_residualize(z_tumor, y, covariates_df, groups)
    z_random_resid, _ = fwl_residualize(z_random, y, covariates_df, groups)
    logger.info("FWL residualization complete.")

    # Step 8: post-FWL R² + bootstrap
    logger.info("Computing post-FWL R² and bootstrap CIs...")
    r2_global_post, ci_lo_gp, ci_hi_gp, scores_global_post = _compute_r2_and_bootstrap(
        z_global_resid, y_resid, groups, _fit, cfg.n_boot, cfg.seed
    )
    r2_tumor_post, ci_lo_tp, ci_hi_tp, scores_tumor_post = _compute_r2_and_bootstrap(
        z_tumor_resid, y_resid, groups, _fit, cfg.n_boot, cfg.seed
    )
    r2_random_post, ci_lo_rp, ci_hi_rp, scores_random_post = _compute_r2_and_bootstrap(
        z_random_resid, y_resid, groups, _fit, cfg.n_boot, cfg.seed
    )
    logger.info(
        "Post-FWL R²: global=%.4f, tumor=%.4f, random=%.4f",
        r2_global_post,
        r2_tumor_post,
        r2_random_post,
    )

    # Step 9: operator decision
    delta_r2_spatial = r2_tumor - r2_global
    delta_r2_post_fwl = r2_tumor_post - r2_global_post
    operator = _classify_operator(delta_r2_post_fwl)
    logger.info(
        "Delta raw=%.4f, post-FWL=%.4f -> operator=%s",
        delta_r2_spatial,
        delta_r2_post_fwl,
        operator,
    )

    # Step 10: implementation guard
    impl_guard_pass = (r2_random < r2_global + 0.05) and (r2_random < r2_tumor - 0.30)
    if not impl_guard_pass:
        msg = (
            f"IMPLEMENTATION GUARD FAILED.\n"
            f"  r2_random={r2_random:.4f}\n"
            f"  r2_global={r2_global:.4f} (threshold: r2_random < r2_global + 0.05)\n"
            f"  r2_tumor={r2_tumor:.4f} (threshold: r2_random < r2_tumor - 0.30)\n"
            f"  check_1 (r2_random < r2_global+0.05): {r2_random < r2_global + 0.05}\n"
            f"  check_2 (r2_random < r2_tumor-0.30): {r2_random < r2_tumor - 0.30}\n"
            "This indicates spatially-uniform encoding or broken pooling."
        )
        logger.error(msg)
        (cfg.output_dir / "BROKEN.txt").write_text(msg)

    # Assemble result
    result = SpatialResult(
        r2_global=r2_global,
        r2_tumor=r2_tumor,
        r2_random=r2_random,
        r2_global_ci=(ci_lo_g, ci_hi_g),
        r2_tumor_ci=(ci_lo_t, ci_hi_t),
        r2_random_ci=(ci_lo_r, ci_hi_r),
        r2_global_post_fwl=r2_global_post,
        r2_tumor_post_fwl=r2_tumor_post,
        r2_random_post_fwl=r2_random_post,
        r2_global_post_fwl_ci=(ci_lo_gp, ci_hi_gp),
        r2_tumor_post_fwl_ci=(ci_lo_tp, ci_hi_tp),
        r2_random_post_fwl_ci=(ci_lo_rp, ci_hi_rp),
        delta_r2_spatial=delta_r2_spatial,
        delta_r2_post_fwl=delta_r2_post_fwl,
        operator=operator,
        n_scans_used=n_scans_used,
        n_imputed_for_fwl=n_imputed_for_fwl,
        impl_guard_pass=impl_guard_pass,
    )

    # Step 11: save artifacts
    payload = dataclasses.asdict(result)
    payload["_fwl_bootstrap_strategy"] = "fit_once_residual_bootstrap"
    payload["_synthetic"] = cfg.synthetic
    payload["_n_boot"] = cfg.n_boot
    result_path = cfg.output_dir / "result.json"
    with open(result_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    logger.info("result.json written to %s", result_path)

    npz_path = cfg.output_dir / "bootstrap_scores.npz"
    np.savez(
        npz_path,
        **{
            "global": scores_global,
            "tumor": scores_tumor,
            "random": scores_random,
            "global_post": scores_global_post,
            "tumor_post": scores_tumor_post,
            "random_post": scores_random_post,
        },
    )
    logger.info("bootstrap_scores.npz written to %s", npz_path)

    # Step 12: plots
    make_all(cfg.output_dir, result)
    logger.info("E2.3 complete. Artifacts in %s", cfg.output_dir)

    return result


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    cfg: SpatialRoutineConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    int,
]:
    """Generate synthetic latent features with a planted spatial localization signal.

    Plants a volume signal along a random unit direction visible in z_tumor.
    z_global has a weak coupling (cfg.synthetic_global_strength) to z_tumor,
    simulating partial spillover. z_random is pure noise at the same scale as
    z_global noise. A sex-correlated confound axis tests FWL robustness.

    Parameters
    ----------
    cfg:
        Routine config (uses seed, synthetic_n, synthetic_c, synthetic_global_strength).

    Returns
    -------
    z_tumor, z_global, z_random, log_v, patient_ids, scan_ids,
    covariates_df, n_imputed_for_fwl
    """
    rng = np.random.default_rng(cfg.seed)
    N, C = cfg.synthetic_n, cfg.synthetic_c

    # Random planted directions
    u_vol = rng.standard_normal(C)
    u_vol /= np.linalg.norm(u_vol)
    u_conf = rng.standard_normal(C)
    u_conf /= np.linalg.norm(u_conf)

    log_v = rng.standard_normal(N)
    sex = rng.choice(["Male", "Female"], size=N)
    sex_eff = np.where(sex == "Male", 1.0, -1.0)

    # Small confound coupling in log_v (sex-correlated)
    log_v = log_v + 0.5 * (sex == "Male")

    # z_tumor: strong volume signal + sex confound + small noise
    z_tumor = (
        log_v[:, None] * u_vol + 0.3 * sex_eff[:, None] * u_conf + 0.1 * rng.standard_normal((N, C))
    )

    # z_global: weak coupling to tumor (planted spatial localization)
    z_global = cfg.synthetic_global_strength * z_tumor + 0.1 * rng.standard_normal((N, C))

    # z_random: pure noise at same scale as z_global noise
    z_random = 0.1 * rng.standard_normal((N, C))

    patient_ids = np.array([f"PID{i:04d}" for i in range(N)], dtype=object)
    scan_ids = np.array([f"SID{i:04d}" for i in range(N)], dtype=object)

    ages = rng.uniform(40, 80, size=N)
    covariates_df = pd.DataFrame(
        {
            "sex": sex,
            "grade_bin": np.full(N, "1"),
            "laterality": np.full(N, "L"),
            "age": ages,
        }
    )

    return z_tumor, z_global, z_random, log_v, patient_ids, scan_ids, covariates_df, 0


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------


def _load_real_data(
    cfg: SpatialRoutineConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    int,
]:
    """Load features, clinical, and laterality from disk.

    Parameters
    ----------
    cfg:
        Routine config with valid file paths.

    Returns
    -------
    z_tumor, z_global, z_random, log_v, patient_ids, scan_ids,
    covariates_df, n_imputed_for_fwl
    """
    import h5py

    from experiments.E2_2_identifiability.analysis.identifiability import (
        _load_or_compute_laterality,
    )
    from menflow.data.clinical import join_clinical_to_scans, load_brats_men_clinical

    logger.info("Loading features from %s", cfg.features_h5)
    with h5py.File(cfg.features_h5, "r") as f:
        z_tumor = f["z_tumor"][:].astype(np.float64)
        z_global = f["z_global"][:].astype(np.float64)
        z_random = f["z_random"][:].astype(np.float64)
        log_v = f["log_volume"][:].astype(np.float64)
        scan_ids = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]],
            dtype=object,
        )
        patient_ids = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]],
            dtype=object,
        )
        mask_lat_present = f["mask_lat_present"][:].astype(bool)

    keep = mask_lat_present & np.isfinite(log_v)
    n_drop = int((~keep).sum())
    if n_drop:
        logger.warning(
            "Dropping %d / %d scans with empty latent mask or non-finite log V.",
            n_drop,
            len(keep),
        )
    z_tumor = z_tumor[keep]
    z_global = z_global[keep]
    z_random = z_random[keep]
    log_v = log_v[keep]
    scan_ids = scan_ids[keep]
    patient_ids = patient_ids[keep]
    logger.info("Features loaded: N=%d scans after filtering.", len(log_v))

    # Clinical covariates
    logger.info("Loading clinical covariates from %s", cfg.clinical_xlsx)
    clinical = load_brats_men_clinical(cfg.clinical_xlsx)
    covariates_df = join_clinical_to_scans(scan_ids, clinical)

    # Laterality
    laterality_arr = _load_or_compute_laterality(scan_ids, cfg.source_h5, cfg.laterality_cache_csv)
    covariates_df["laterality"] = laterality_arr

    # Impute for FWL
    n_imputed_sex = int(covariates_df["sex"].isna().sum())
    n_imputed_age = int(covariates_df["age"].isna().sum())
    n_imputed_for_fwl = int((covariates_df["sex"].isna() | covariates_df["age"].isna()).sum())

    if n_imputed_age > 0:
        age_mean = float(covariates_df["age"].mean())
        covariates_df["age"] = covariates_df["age"].fillna(age_mean)

    if n_imputed_sex > 0:
        sex_mode = covariates_df["sex"].dropna().mode()
        sex_fill = sex_mode.iloc[0] if len(sex_mode) > 0 else "Female"
        covariates_df["sex"] = covariates_df["sex"].fillna(sex_fill)

    covariates_df["laterality"] = covariates_df["laterality"].fillna("midline")

    logger.info(
        "FWL imputed: age=%d, sex=%d (total distinct rows=%d).",
        n_imputed_age,
        n_imputed_sex,
        n_imputed_for_fwl,
    )

    return (
        z_tumor,
        z_global,
        z_random,
        log_v,
        patient_ids,
        scan_ids,
        covariates_df,
        n_imputed_for_fwl,
    )
