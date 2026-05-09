"""E2.2 — direction identifiability: bootstrap stability + confounder audit.

Establishes that the naive volume direction u_naive (carried forward from E2.1)
is (i) statistically stable across patient resamples and (ii) not dominated by
demographic confounders (sex, age, histological grade, laterality).

Three analysis families:
- Bootstrap: 200 patient-level resamples; pairwise cosine mean and p5.
- TCAV: matched-control logistic direction; cosine to u_naive (rho_conf_tcav).
- FWL: Frisch-Waugh-Lovell residualized Ridge direction; cosine to u_naive
  (rho_conf_fwl).

Decision logic produces one of four outcomes per modality:
    "use_naive"               — bootstrap stable, unconfounded (rho_conf >= 0.90)
    "use_tcav"                — stable, mild confounding (0.60 <= rho_conf < 0.90)
    "investigate_disagreement" — TCAV and FWL disagree by >= 0.15; manual review
    "defer_e2_6"              — bootstrap weak OR rho_conf < 0.60

Deviation from spec: covariates are sourced from the supplementary clinical xlsx
(not from the unified H5 metadata which is currently empty). ``location`` and
``site`` are absent and excluded from matching; ``laterality`` is derived from
segmentation centroid.
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

from experiments.E2_2_identifiability.analysis.plotting import make_all
from menflow.data.clinical import (
    ClinicalCovariates,
    join_clinical_to_scans,
    load_brats_men_clinical,
)
from menflow.directions import (
    InsufficientMatchedControlsError,
    fwl_direction,
    fwl_residualize,
    matched_control_tcav,
)
from menflow.latent_features.laterality import compute_laterality_from_mask
from menflow.stats.bootstrap import bootstrap_directions, pairwise_cosine_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentifiabilityRoutineConfig:
    """All parameters that govern a single E2.2 run.

    Attributes
    ----------
    features_h5:
        Path to the per-modality features H5 produced by
        ``routines/compute_features``. Set to the sentinel string
        ``"synthetic"`` to run without external data (smoke / unit-test mode).
    source_h5:
        Unified BraTS-MEN H5; used only to read segmentation masks for
        laterality computation. May be ``"synthetic"`` in smoke mode.
    clinical_xlsx:
        Path to the BraTS-MEN supplementary clinical xlsx. May be
        ``"synthetic"`` in smoke mode.
    naive_direction_npz:
        NPZ produced by E2.1 containing the key ``"direction"`` (unit vector).
        May be ``"synthetic"`` in smoke mode.
    output_dir:
        Where result.json, NPZ artefacts, and figures are written.
    n_boot:
        Number of patient-level bootstrap replicates (default 200).
    large_quantile:
        Upper log-volume quantile for TCAV large group (default 0.8).
    small_quantile:
        Lower log-volume quantile for TCAV small group (default 0.2).
    age_window:
        Max absolute age difference (years) for matched-control pairing
        (default 5.0).
    min_pairs:
        Minimum matched pairs required for TCAV (default 30).
    seed:
        RNG seed for reproducibility.
    log_level:
        Python logging level string (e.g. ``"INFO"``).
    laterality_cache_parquet:
        Optional path to a parquet file caching laterality labels indexed by
        scan_id.  If the file exists and covers all scan_ids, the segmentation
        read is skipped.  Written on first run and reused for other modalities.
    """

    features_h5: Path
    source_h5: Path
    clinical_xlsx: Path
    naive_direction_npz: Path
    output_dir: Path
    n_boot: int = 200
    large_quantile: float = 0.8
    small_quantile: float = 0.2
    age_window: float = 5.0
    min_pairs: int = 30
    seed: int = 0
    log_level: str = "INFO"
    laterality_cache_parquet: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> IdentifiabilityRoutineConfig:
        """Load config from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML config file.

        Returns
        -------
        IdentifiabilityRoutineConfig
        """
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        raw.pop("slurm", None)

        path_keys = (
            "features_h5",
            "source_h5",
            "clinical_xlsx",
            "naive_direction_npz",
            "output_dir",
        )
        for k in path_keys:
            if k in raw and raw[k] is not None and str(raw[k]) != "synthetic":
                raw[k] = Path(raw[k]).expanduser()
            elif k in raw and raw[k] is not None:
                raw[k] = Path(str(raw[k]))  # keep "synthetic" as Path("synthetic")

        if "laterality_cache_parquet" in raw and raw["laterality_cache_parquet"] is not None:
            raw["laterality_cache_parquet"] = Path(raw["laterality_cache_parquet"]).expanduser()

        return cls(**raw)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentifiabilityResult:
    """Summary statistics and decision for one modality E2.2 run.

    Attributes
    ----------
    rho_boot_mean:
        Mean pairwise cosine across all bootstrap-direction pairs.
    rho_boot_p5:
        5th percentile of pairwise cosines (lower stability bound).
    rho_conf_tcav:
        Absolute cosine between u_naive and u_TCAV.
    rho_conf_fwl:
        Absolute cosine between u_naive and u_FWL.
    tcav_norm:
        Euclidean norm of the raw TCAV logistic coefficient.
    tcav_roc_auc:
        In-sample ROC AUC of the TCAV logistic classifier.
    n_matched_pairs:
        Number of large–small matched pairs used in TCAV.
    n_scans_used:
        Number of scans after dropping those with missing sex/age
        (used for TCAV quantile binning).
    n_imputed_for_fwl:
        Number of scans where at least one covariate was imputed for FWL.
    fraction_unknown_grade_pairs:
        Fraction of matched TCAV pairs where both members have
        ``grade_bin == "unknown"``.
    regime:
        One of ``"stable_unconfounded"``, ``"stable_confounded"``,
        ``"subspace"``, ``"path_d"``.
    decision:
        One of ``"use_naive"``, ``"use_tcav"``,
        ``"investigate_disagreement"``, ``"defer_e2_6"``.
    tcav_fwl_disagreement:
        ``|rho_conf_tcav - rho_conf_fwl|``.
    blocked_carry_forward:
        True when E2.3+ must NOT consume the TCAV direction without manual
        review (disagreement gate or bootstrap/confounding failures).
    """

    rho_boot_mean: float
    rho_boot_p5: float
    rho_conf_tcav: float
    rho_conf_fwl: float
    tcav_norm: float
    tcav_roc_auc: float
    n_matched_pairs: int
    n_scans_used: int
    n_imputed_for_fwl: int
    fraction_unknown_grade_pairs: float
    regime: str
    decision: str
    tcav_fwl_disagreement: float
    blocked_carry_forward: bool


# ---------------------------------------------------------------------------
# Synthetic data generator (smoke / unit-test path)
# ---------------------------------------------------------------------------


def _make_synthetic_features(
    cfg: IdentifiabilityRoutineConfig,
) -> tuple[
    np.ndarray,  # z_tumor (N, C)
    np.ndarray,  # log_v  (N,)
    np.ndarray,  # scan_ids (N,)
    np.ndarray,  # patient_ids (N,)
    pd.DataFrame,  # covariates_df
    np.ndarray,  # laterality_arr (N,) of str
    np.ndarray,  # u_naive (C,) planted direction
]:
    """Generate synthetic latent features with a planted volume direction.

    Plants a volume signal along the first latent dimension plus a sex-
    correlated confound along the second dimension.  This lets the smoke test
    verify that TCAV recovers the planted volume direction rather than the
    confound axis.

    Parameters
    ----------
    cfg:
        Routine config (only ``seed`` and ``n_boot`` are used here; the rest
        are ignored in synthetic mode).

    Returns
    -------
    z_tumor, log_v, scan_ids, patient_ids, covariates_df, laterality_arr, u_naive
    """
    rng = np.random.default_rng(cfg.seed)
    N = 200
    C = 4

    # Planted volume direction: unit vector along first latent dim
    u_vol = np.zeros(C, dtype=np.float64)
    u_vol[0] = 1.0

    # Planted confound direction: along second latent dim (sex-correlated)
    u_conf = np.zeros(C, dtype=np.float64)
    u_conf[1] = 1.0

    # 50/50 sex split: 0 = Female, 1 = Male
    sex_arr = rng.integers(0, 2, size=N)

    # log-volume: signal + small noise
    log_v = rng.normal(loc=0.0, scale=1.0, size=N).astype(np.float64)

    # Latent features = volume signal + confound (sex) + noise
    z_tumor = np.zeros((N, C), dtype=np.float64)
    z_tumor[:, 0] = log_v * 1.0 + rng.normal(0, 0.3, N)  # volume direction
    z_tumor[:, 1] = sex_arr * 0.5 + rng.normal(0, 0.3, N)  # sex confound
    z_tumor[:, 2] = rng.normal(0, 0.5, N)
    z_tumor[:, 3] = rng.normal(0, 0.5, N)

    # Patient IDs: one scan per patient for simplicity
    patient_ids = np.array([f"PT{i:04d}" for i in range(N)], dtype=object)
    scan_ids = np.array([f"BraTS-MEN-SYN-{i:04d}-000" for i in range(N)], dtype=object)

    # Covariates
    ages = rng.uniform(40, 75, size=N)
    grades = rng.choice(["1", "2", "3", "unknown"], size=N, p=[0.25, 0.35, 0.10, 0.30])
    lateralities = rng.choice(["L", "R", "midline"], size=N, p=[0.45, 0.45, 0.10])
    sex_labels = np.where(sex_arr == 0, "Female", "Male")
    # Introduce ~5% NaN sex and age to mimic real data
    nan_idx_sex = rng.choice(N, size=int(0.05 * N), replace=False)
    nan_idx_age = rng.choice(N, size=int(0.05 * N), replace=False)
    sex_labels = sex_labels.astype(object)
    sex_labels[nan_idx_sex] = None
    ages_obj = ages.astype(object)
    ages_obj[nan_idx_age] = None

    covariates_df = pd.DataFrame(
        {
            "sex": sex_labels,
            "grade_bin": grades,
            "laterality": lateralities,
            "age": ages_obj,
        }
    )

    # Naive direction from Ridge probe on the synthetic data (planted = u_vol)
    # We plant u_naive directly as the true volume direction so we can test cosine logic
    u_naive = u_vol.copy()

    return z_tumor, log_v, scan_ids, patient_ids, covariates_df, lateralities, u_naive


# ---------------------------------------------------------------------------
# Laterality helpers
# ---------------------------------------------------------------------------


def _load_or_compute_laterality(
    scan_ids: np.ndarray,
    source_h5_path: Path,
    cache_path: Path | None,
) -> np.ndarray:
    """Return laterality labels (str array) for each scan_id.

    Tries to load from the parquet cache first. If the cache does not cover
    all scan_ids, recomputes from segmentation masks and writes the cache.

    Parameters
    ----------
    scan_ids:
        1-D str array, shape ``(N,)``.
    source_h5_path:
        Path to the unified BraTS-MEN H5 containing ``/segmentations`` and
        ``/scan_ids``.
    cache_path:
        Optional path to a parquet file keyed by ``scan_id``.

    Returns
    -------
    np.ndarray
        String array ``(N,)`` with values in ``{"L", "R", "midline"}``.
    """
    import h5py

    if cache_path is not None and cache_path.exists():
        cache_df = pd.read_csv(cache_path, dtype={"scan_id": str, "laterality": str})
        if set(scan_ids).issubset(set(cache_df["scan_id"].values)):
            logger.info("Laterality loaded from cache: %s", cache_path)
            mapping = dict(zip(cache_df["scan_id"], cache_df["laterality"]))
            return np.array([mapping[sid] for sid in scan_ids], dtype=object)
        logger.info("Laterality cache incomplete; recomputing from segmentations.")

    logger.info("Computing laterality from segmentation masks in %s", source_h5_path)
    with h5py.File(source_h5_path, "r") as f:
        h5_scan_ids = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]],
            dtype=object,
        )
        scan_id_to_idx = {sid: i for i, sid in enumerate(h5_scan_ids)}
        lateralities = []
        for sid in scan_ids:
            idx = scan_id_to_idx.get(sid)
            if idx is None:
                logger.warning(
                    "scan_id '%s' not found in source H5; laterality set to 'midline'", sid
                )
                lateralities.append("midline")
                continue
            seg = f["segmentations"][idx]  # (H, W, D) int8
            binary_mask = (seg > 0).astype(np.float64)
            lat = compute_laterality_from_mask(binary_mask, axis=0)
            lateralities.append(lat)

    lat_arr = np.array(lateralities, dtype=object)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"scan_id": scan_ids, "laterality": lat_arr}).to_csv(cache_path, index=False)
        logger.info("Laterality cache written to %s", cache_path)

    return lat_arr


# ---------------------------------------------------------------------------
# Drop-one sensitivity
# ---------------------------------------------------------------------------


def _drop_one_sensitivity(
    z_tumor: np.ndarray,
    log_v: np.ndarray,
    covariates_df: pd.DataFrame,
    patient_ids: np.ndarray,
    tcav_full_direction: np.ndarray,
    cfg: IdentifiabilityRoutineConfig,
) -> dict[str, float | None]:
    """Refit TCAV dropping one covariate at a time; report cosine to full TCAV.

    Parameters
    ----------
    z_tumor, log_v, covariates_df, patient_ids:
        Same arrays used for the full TCAV fit.
    tcav_full_direction:
        Unit direction from the full (all-covariate) TCAV.
    cfg:
        Routine config (for quantile/age_window/min_pairs/seed).

    Returns
    -------
    dict
        Mapping ``dropped_covariate -> |cos|`` or ``None`` if TCAV failed.
    """
    results: dict[str, float | None] = {}
    drop_specs: list[tuple[str, list[str] | None, float]] = [
        # (name, cols_to_drop_from_key, age_window_override)
        ("sex", ["sex"], cfg.age_window),
        ("grade_bin", ["grade_bin"], cfg.age_window),
        ("laterality", ["laterality"], cfg.age_window),
        ("age", None, np.inf),  # drop age by setting age_window to inf
    ]

    for drop_name, drop_cols, age_window_override in drop_specs:
        # Build a temporary covariates df that has the dropped column set to a
        # constant so matching ignores it (TCAV matcher checks equality on all
        # columns in covariates; laterality/sex/grade become "any" by setting
        # a shared constant value).
        cov_tmp = covariates_df.copy()

        if drop_cols is not None:
            for col in drop_cols:
                cov_tmp[col] = "any"  # all rows match on this key

        # Drop NaN sex/age before quantile binning (same rule as main TCAV)
        if drop_name == "sex":
            keep = cov_tmp["age"].notna()
        elif drop_name == "age":
            keep = cov_tmp["sex"].notna()
        else:
            keep = cov_tmp["sex"].notna() & cov_tmp["age"].notna()

        # age column: if we zeroed age_window, set age to constant so the
        # window check always passes (any - any = 0 <= inf).
        if age_window_override == np.inf:
            cov_tmp["age"] = 0.0
            keep = cov_tmp["sex"].notna()  # drop only NaN-sex rows

        z_k = z_tumor[keep]
        lv_k = log_v[keep]
        cov_k = cov_tmp[keep].reset_index(drop=True)

        try:
            tcav_drop = matched_control_tcav(
                z_k,
                lv_k,
                cov_k,
                large_quantile=cfg.large_quantile,
                small_quantile=cfg.small_quantile,
                age_window=age_window_override,
                min_pairs=cfg.min_pairs,
                seed=cfg.seed,
            )
            cos_val = abs(float(np.dot(tcav_full_direction, tcav_drop.direction)))
            results[drop_name] = cos_val
            logger.info("Drop-one [%s]: cos=%.4f (%d pairs)", drop_name, cos_val, tcav_drop.n_pairs)
        except InsufficientMatchedControlsError as exc:
            logger.warning("Drop-one [%s]: insufficient pairs (%s)", drop_name, exc)
            results[drop_name] = None

    return results


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def _classify(
    rho_boot_mean: float,
    rho_conf_tcav: float,
    rho_conf_fwl: float,
) -> tuple[str, str, bool]:
    """Apply the three-gate decision logic.

    Returns
    -------
    regime, decision, blocked_carry_forward
    """
    disagree = abs(rho_conf_tcav - rho_conf_fwl)

    # Gate 1: bootstrap stability
    if rho_boot_mean < 0.70:
        return "subspace", "defer_e2_6", True

    # Gate 2: TCAV-FWL agreement
    if disagree >= 0.15:
        regime = "stable_unconfounded" if rho_conf_tcav >= 0.90 else "stable_confounded"
        return regime, "investigate_disagreement", True

    # Gate 3: confounding level
    if rho_conf_tcav < 0.60:
        return "stable_confounded", "defer_e2_6", True
    if rho_conf_tcav < 0.90:
        return "stable_confounded", "use_tcav", False
    return "stable_unconfounded", "use_naive", False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_identifiability(cfg: IdentifiabilityRoutineConfig) -> IdentifiabilityResult:
    """Execute the full E2.2 direction identifiability pipeline.

    Parameters
    ----------
    cfg:
        Fully validated experiment configuration.

    Returns
    -------
    IdentifiabilityResult
        All scalar metrics, regime classification, and decision string.
        Arrays (bootstrap direction matrix, TCAV direction, FWL direction) are
        saved to ``cfg.output_dir`` as separate NPZ files.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine data path: synthetic vs real
    # ------------------------------------------------------------------
    is_synthetic = str(cfg.features_h5) == "synthetic"

    if is_synthetic:
        logger.info("Synthetic mode: generating planted-direction data (N=200, C=4).")
        (
            z_tumor,
            log_v,
            scan_ids,
            patient_ids,
            covariates_df,
            laterality_arr,
            u_naive,
        ) = _make_synthetic_features(cfg)
    else:
        # ------------------------------------------------------------------
        # 2. Load features H5
        # ------------------------------------------------------------------
        import h5py

        logger.info("Loading features from %s", cfg.features_h5)
        with h5py.File(cfg.features_h5, "r") as f:
            z_tumor = f["z_tumor"][:].astype(np.float64)
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
        z_tumor, log_v, scan_ids, patient_ids = (
            z_tumor[keep],
            log_v[keep],
            scan_ids[keep],
            patient_ids[keep],
        )
        logger.info("Features loaded: N=%d scans after filtering.", len(log_v))

        # ------------------------------------------------------------------
        # 3. Load clinical xlsx and join covariates
        # ------------------------------------------------------------------
        logger.info("Loading clinical covariates from %s", cfg.clinical_xlsx)
        clinical: ClinicalCovariates = load_brats_men_clinical(cfg.clinical_xlsx)
        covariates_df = join_clinical_to_scans(scan_ids, clinical)

        # ------------------------------------------------------------------
        # 4. Compute / load laterality
        # ------------------------------------------------------------------
        laterality_arr = _load_or_compute_laterality(
            scan_ids, cfg.source_h5, cfg.laterality_cache_parquet
        )
        covariates_df["laterality"] = laterality_arr

        # ------------------------------------------------------------------
        # 5. Load naive direction from E2.1
        # ------------------------------------------------------------------
        logger.info("Loading naive direction from %s", cfg.naive_direction_npz)
        npz = np.load(cfg.naive_direction_npz)
        if "direction" in npz:
            u_naive = npz["direction"].astype(np.float64).ravel()
        else:
            # Fallback: first array in the npz
            first_key = list(npz.keys())[0]
            u_naive = npz[first_key].astype(np.float64).ravel()
        u_naive = u_naive / (np.linalg.norm(u_naive) + 1e-12)

    n_total = len(log_v)
    logger.info("Total scans for analysis: N=%d", n_total)

    # ------------------------------------------------------------------
    # 6. Bootstrap direction stability
    # ------------------------------------------------------------------
    logger.info("Running bootstrap_directions (n_boot=%d).", cfg.n_boot)
    W = bootstrap_directions(
        z_tumor,
        log_v,
        patient_ids,
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )
    np.savez(cfg.output_dir / "bootstrap_directions.npz", W=W)

    rho_boot_mean, rho_boot_p5 = pairwise_cosine_stats(W)
    logger.info("Bootstrap pairwise cosines: mean=%.4f, p5=%.4f", rho_boot_mean, rho_boot_p5)

    # ------------------------------------------------------------------
    # 7. TCAV — drop NaN sex/age first
    # ------------------------------------------------------------------
    keep_tcav = covariates_df["sex"].notna() & covariates_df["age"].notna()
    n_scans_used = int(keep_tcav.sum())
    logger.info("TCAV: %d / %d scans with non-NaN sex and age.", n_scans_used, n_total)

    z_tcav = z_tumor[keep_tcav]
    lv_tcav = log_v[keep_tcav]
    cov_tcav = covariates_df[keep_tcav].reset_index(drop=True)

    logger.info("Running matched_control_tcav.")
    tcav = matched_control_tcav(
        z_tcav,
        lv_tcav,
        cov_tcav,
        large_quantile=cfg.large_quantile,
        small_quantile=cfg.small_quantile,
        age_window=cfg.age_window,
        min_pairs=cfg.min_pairs,
        seed=cfg.seed,
    )
    np.savez(cfg.output_dir / "tcav_direction.npz", direction=tcav.direction.astype(np.float64))
    logger.info("TCAV: %d pairs, ROC AUC=%.4f", tcav.n_pairs, tcav.roc_auc)

    # Fraction of matched pairs where both members have unknown grade
    pairs_large = tcav.pairs_index[0::2]  # indices into z_tcav
    pairs_small = tcav.pairs_index[1::2]
    grade_large = cov_tcav["grade_bin"].iloc[pairs_large].values
    grade_small = cov_tcav["grade_bin"].iloc[pairs_small].values
    n_unknown_pairs = int(np.sum((grade_large == "unknown") & (grade_small == "unknown")))
    fraction_unknown_grade_pairs = n_unknown_pairs / max(tcav.n_pairs, 1)
    logger.info(
        "TCAV: %.1f%% matched pairs are both unknown-grade.",
        fraction_unknown_grade_pairs * 100,
    )

    # ------------------------------------------------------------------
    # 8. FWL — impute missing covariates
    # ------------------------------------------------------------------
    cov_fwl = covariates_df.copy()

    # Impute: age → mean, sex → mode, grade_bin already has "unknown" for NaN
    n_imputed_age = int(cov_fwl["age"].isna().sum())
    n_imputed_sex = int(cov_fwl["sex"].isna().sum())
    n_imputed_for_fwl = int((cov_fwl["sex"].isna() | cov_fwl["age"].isna()).sum())

    if n_imputed_age > 0:
        age_mean = float(cov_fwl["age"].mean())
        cov_fwl["age"] = cov_fwl["age"].fillna(age_mean)

    if n_imputed_sex > 0:
        sex_mode = cov_fwl["sex"].dropna().mode()
        sex_fill = sex_mode.iloc[0] if len(sex_mode) > 0 else "Female"
        cov_fwl["sex"] = cov_fwl["sex"].fillna(sex_fill)

    # Ensure laterality is present (may be NaN for edge-case scans)
    cov_fwl["laterality"] = cov_fwl["laterality"].fillna("midline")

    logger.info(
        "FWL imputed: age=%d, sex=%d (total distinct rows=%d).",
        n_imputed_age,
        n_imputed_sex,
        n_imputed_for_fwl,
    )

    logger.info("Running fwl_residualize.")
    z_resid, y_resid = fwl_residualize(z_tumor, log_v, cov_fwl, patient_ids)

    logger.info("Running fwl_direction.")
    u_fwl = fwl_direction(z_resid, y_resid, patient_ids)
    np.savez(cfg.output_dir / "fwl_direction.npz", direction=u_fwl.astype(np.float64))

    # ------------------------------------------------------------------
    # 9. Cosines (absolute: direction sign is identifiable only up to flip)
    # ------------------------------------------------------------------
    rho_conf_tcav = abs(float(np.dot(u_naive, tcav.direction)))
    rho_conf_fwl = abs(float(np.dot(u_naive, u_fwl)))
    tcav_fwl_disagreement = abs(rho_conf_tcav - rho_conf_fwl)
    logger.info(
        "rho_conf_tcav=%.4f, rho_conf_fwl=%.4f, disagreement=%.4f",
        rho_conf_tcav,
        rho_conf_fwl,
        tcav_fwl_disagreement,
    )

    # ------------------------------------------------------------------
    # 10. Decision logic
    # ------------------------------------------------------------------
    regime, decision, blocked = _classify(rho_boot_mean, rho_conf_tcav, rho_conf_fwl)
    logger.info("Regime=%s, decision=%s, blocked=%s", regime, decision, blocked)

    if blocked:
        blocked_path = cfg.output_dir / "BLOCKED.txt"
        reason = (
            f"E2.2 carry-forward blocked.\n"
            f"  regime={regime}\n"
            f"  decision={decision}\n"
            f"  rho_boot_mean={rho_boot_mean:.4f}\n"
            f"  rho_conf_tcav={rho_conf_tcav:.4f}\n"
            f"  rho_conf_fwl={rho_conf_fwl:.4f}\n"
            f"  disagreement={tcav_fwl_disagreement:.4f}\n"
        )
        blocked_path.write_text(reason)
        logger.warning("BLOCKED. Wrote %s.", blocked_path)

    # ------------------------------------------------------------------
    # 11. Drop-one sensitivity
    # ------------------------------------------------------------------
    logger.info("Running drop-one covariate sensitivity analysis.")
    sensitivity = _drop_one_sensitivity(
        z_tcav, lv_tcav, cov_tcav, patient_ids[keep_tcav], tcav.direction, cfg
    )

    sens_path = cfg.output_dir / "sensitivity.json"
    with open(sens_path, "w") as fh:
        json.dump(sensitivity, fh, indent=2)
    logger.info("Sensitivity written to %s", sens_path)

    # ------------------------------------------------------------------
    # 12. Assemble result
    # ------------------------------------------------------------------
    result = IdentifiabilityResult(
        rho_boot_mean=float(rho_boot_mean),
        rho_boot_p5=float(rho_boot_p5),
        rho_conf_tcav=float(rho_conf_tcav),
        rho_conf_fwl=float(rho_conf_fwl),
        tcav_norm=float(tcav.norm),
        tcav_roc_auc=float(tcav.roc_auc),
        n_matched_pairs=int(tcav.n_pairs),
        n_scans_used=int(n_scans_used),
        n_imputed_for_fwl=int(n_imputed_for_fwl),
        fraction_unknown_grade_pairs=float(fraction_unknown_grade_pairs),
        regime=regime,
        decision=decision,
        tcav_fwl_disagreement=float(tcav_fwl_disagreement),
        blocked_carry_forward=bool(blocked),
    )

    # ------------------------------------------------------------------
    # 13. Persist result.json
    # ------------------------------------------------------------------
    payload = dataclasses.asdict(result)
    payload["features_h5"] = str(cfg.features_h5)
    payload["naive_direction_npz"] = str(cfg.naive_direction_npz)
    payload["_deviations"] = (
        "Covariates sourced from xlsx (H5 metadata empty). "
        "location/site absent; replaced by mask-derived laterality. "
        "grade missing in ~30% scans (encoded as 'unknown' stratum). "
        "Absolute cosines used (sign unidentifiable up to flip)."
    )
    result_path = cfg.output_dir / "result.json"
    with open(result_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    logger.info("result.json written to %s", result_path)

    # ------------------------------------------------------------------
    # 14. Plots
    # ------------------------------------------------------------------
    make_all(
        cfg.output_dir,
        W=W,
        rho_boot_mean=rho_boot_mean,
        rho_boot_p5=rho_boot_p5,
        rho_conf_tcav=rho_conf_tcav,
        rho_conf_fwl=rho_conf_fwl,
        regime=regime,
        sensitivity=sensitivity,
    )

    logger.info("E2.2 complete. Artifacts in %s", cfg.output_dir)
    return result


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
