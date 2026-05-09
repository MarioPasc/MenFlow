"""Follow-up diagnostic for the t1c TCAV-FWL disagreement (E2.2).

Tests whether the disagreement between matched-control TCAV and the
naive / FWL directions is small-n / low-C underdetermination — the
working hypothesis after the base run — or substantive confounding.

Four sub-tests:

A. Matched-pair bootstrap. Resample matched pairs with replacement,
   refit the logistic classifier, record cosine to u_naive and u_FWL.
   A wide CI that overlaps u_FWL supports the small-n hypothesis.

B. Quantile sweep. Re-run TCAV with successively wider large/small
   bins (q in {0.10, 0.20, 0.30, 0.40, 0.50}). Record n_pairs and the
   TCAV-vs-naive / TCAV-vs-FWL cosines. If TCAV converges to FWL as
   n_pairs grows, the base disagreement is finite-sample noise.

C. Logistic-regularization sweep. Refit logistic on the base matched
   set with C in {1e-3, 1e-2, ..., 1e2}. If u_TCAV drifts toward
   u_naive / u_FWL as C decreases (stronger L2 shrinkage, less
   freedom to overfit), the base direction is exploiting capacity
   beyond what the volume signal needs.

D. Leave-one-pair-out cross-validated AUC of the matched-set logistic.
   Held-out AUC much below 1.0 confirms the in-sample perfect
   separation is overfit, not a stable boundary.

Output: ``per_modality/t1c/tcav_diagnostic/{summary.json, plots/*.png}``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from experiments.E2_2_identifiability.analysis.identifiability import (
    _load_or_compute_laterality,
)
from menflow.data.clinical import join_clinical_to_scans, load_brats_men_clinical
from menflow.directions.fwl import fwl_direction, fwl_residualize
from menflow.directions.tcav import matched_control_tcav

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-modality path templates (mirroring local_brats_men.yaml).
# ---------------------------------------------------------------------------

VALID_MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")

SOURCE_H5 = Path("/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/h5/brats_men.h5")
CLINICAL_XLSX = Path(
    "/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/"
    "Meningioma_supplementary_clinical_data_and_imaging_parameters_for_"
    "training_and_validation_sets.xlsx"
)
LATERALITY_CACHE = Path(
    "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
    "E2_2_identifiability/laterality_cache.csv"
)


@dataclass(frozen=True, slots=True)
class _Paths:
    modality: str
    features_h5: Path
    naive_direction_npz: Path
    output_dir: Path


def _paths_for(modality: str) -> _Paths:
    if modality not in VALID_MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {VALID_MODALITIES}")
    return _Paths(
        modality=modality,
        features_h5=Path(
            "/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/per_modality_features/"
            f"brats_men_features_{modality}.h5"
        ),
        naive_direction_npz=Path(
            "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
            f"E2_1_recoverability/per_modality/{modality}/direction.npz"
        ),
        output_dir=Path(
            "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
            f"E2_2_identifiability/per_modality/{modality}/tcav_diagnostic"
        ),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LoadedData:
    z_tumor: np.ndarray  # (N, C)
    log_v: np.ndarray  # (N,)
    patient_ids: np.ndarray  # (N,)
    scan_ids: np.ndarray  # (N,)
    covariates: pd.DataFrame  # rows aligned to z_tumor


def _decode(arr: np.ndarray) -> np.ndarray:
    return np.array(
        [s.decode() if isinstance(s, (bytes, np.bytes_)) else s for s in arr],
        dtype=object,
    )


def _load(features_h5: Path) -> _LoadedData:
    """Load features + clinical + laterality, drop unusable rows."""
    with h5py.File(features_h5, "r") as f:
        z = np.asarray(f["z_tumor"][:], dtype=np.float64)
        log_v = np.asarray(f["log_volume"][:], dtype=np.float64)
        scan_ids = _decode(f["scan_ids"][:])
        patient_ids = _decode(f["patient_ids"][:])
        keep_mask = np.asarray(f["mask_lat_present"][:]).astype(bool) & np.isfinite(log_v)
    z = z[keep_mask]
    log_v = log_v[keep_mask]
    scan_ids = scan_ids[keep_mask]
    patient_ids = patient_ids[keep_mask]

    clinical = load_brats_men_clinical(CLINICAL_XLSX)
    cov = join_clinical_to_scans(scan_ids, clinical).reset_index(drop=True)

    laterality = _load_or_compute_laterality(scan_ids, SOURCE_H5, LATERALITY_CACHE)
    cov["laterality"] = laterality

    # Drop NaN sex/age (matches main analysis pre-TCAV filter).
    full_mask = cov["sex"].notna() & cov["age"].notna()
    z = z[full_mask.values]
    log_v = log_v[full_mask.values]
    patient_ids = patient_ids[full_mask.values]
    scan_ids = scan_ids[full_mask.values]
    cov = cov.loc[full_mask].reset_index(drop=True)

    logger.info("Loaded N=%d scans for diagnostic.", len(z))
    return _LoadedData(z, log_v, patient_ids, scan_ids, cov)


# ---------------------------------------------------------------------------
# Reference directions
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-norm direction")
    return v / n


def _fit_naive(z: np.ndarray, log_v: np.ndarray) -> np.ndarray:
    """Best-alpha-by-CV ridge already lives in ``menflow.probes.linear``;
    here we replicate the simple α=0.1 choice (E2.1 found 0.1 optimal for
    every modality) to avoid re-running CV."""
    return _unit(Ridge(alpha=0.1).fit(z, log_v).coef_)


def _fit_fwl(loaded: _LoadedData) -> np.ndarray:
    cov = loaded.covariates.copy()
    cov["sex"] = cov["sex"].fillna(cov["sex"].mode().iloc[0])
    cov["age"] = cov["age"].fillna(cov["age"].mean())
    cov["grade_bin"] = cov["grade_bin"].fillna("unknown")
    cov["laterality"] = cov["laterality"].fillna("midline")
    z_res, y_res = fwl_residualize(loaded.z_tumor, loaded.log_v, cov, loaded.patient_ids)
    return fwl_direction(z_res, y_res, loaded.patient_ids)


# ---------------------------------------------------------------------------
# Phase A — matched-pair bootstrap
# ---------------------------------------------------------------------------


def phase_a_bootstrap(
    z: np.ndarray,
    pairs_index: np.ndarray,  # (2*n_pairs,) interleaved large/small
    u_naive: np.ndarray,
    u_fwl: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    n_pairs = len(pairs_index) // 2
    rng = np.random.default_rng(seed)
    z_pairs_L = z[pairs_index[0::2]]  # (n_pairs, C)
    z_pairs_S = z[pairs_index[1::2]]

    cos_naive: list[float] = []
    cos_fwl: list[float] = []
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        X_L = z_pairs_L[idx]
        X_S = z_pairs_S[idx]
        X = np.empty((2 * n_pairs, z.shape[1]))
        X[0::2] = X_L
        X[1::2] = X_S
        y = np.tile([1, 0], n_pairs)
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
        except Exception:  # noqa: BLE001
            continue
        w = _unit(clf.coef_.flatten())
        cos_naive.append(abs(float(np.dot(w, u_naive))))
        cos_fwl.append(abs(float(np.dot(w, u_fwl))))
        aucs.append(float(roc_auc_score(y, clf.decision_function(X))))

    arr_n = np.array(cos_naive)
    arr_f = np.array(cos_fwl)
    arr_a = np.array(aucs)
    return {
        "n_boot": int(len(arr_n)),
        "cos_naive_mean": float(arr_n.mean()),
        "cos_naive_p2_5": float(np.percentile(arr_n, 2.5)),
        "cos_naive_p97_5": float(np.percentile(arr_n, 97.5)),
        "cos_fwl_mean": float(arr_f.mean()),
        "cos_fwl_p2_5": float(np.percentile(arr_f, 2.5)),
        "cos_fwl_p97_5": float(np.percentile(arr_f, 97.5)),
        "auc_mean": float(arr_a.mean()),
        "auc_p2_5": float(np.percentile(arr_a, 2.5)),
        "auc_p97_5": float(np.percentile(arr_a, 97.5)),
        "_cos_naive_samples": arr_n.tolist(),
        "_cos_fwl_samples": arr_f.tolist(),
    }


# ---------------------------------------------------------------------------
# Phase B — quantile sweep
# ---------------------------------------------------------------------------


def phase_b_quantile_sweep(
    loaded: _LoadedData,
    u_naive: np.ndarray,
    u_fwl: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50),
    age_window: float = 5.0,
    seed: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    for q in quantiles:
        try:
            tcav = matched_control_tcav(
                loaded.z_tumor,
                loaded.log_v,
                loaded.covariates,
                large_quantile=1.0 - q,
                small_quantile=q,
                age_window=age_window,
                min_pairs=10,
                seed=seed,
            )
            rows.append(
                {
                    "q_lower": float(q),
                    "q_upper": float(1.0 - q),
                    "n_pairs": int(tcav.n_pairs),
                    "roc_auc": float(tcav.roc_auc),
                    "norm": float(tcav.norm),
                    "cos_naive": abs(float(np.dot(tcav.direction, u_naive))),
                    "cos_fwl": abs(float(np.dot(tcav.direction, u_fwl))),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "q_lower": float(q),
                    "q_upper": float(1.0 - q),
                    "status": f"failed: {exc}",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Phase C — logistic regularization sweep on the base matched set
# ---------------------------------------------------------------------------


def phase_c_regularization_sweep(
    z_subset: np.ndarray,
    y_subset: np.ndarray,
    u_naive: np.ndarray,
    u_fwl: np.ndarray,
    *,
    cs: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2),
) -> list[dict]:
    rows: list[dict] = []
    for c_inv in cs:
        clf = LogisticRegression(max_iter=5000, C=c_inv).fit(z_subset, y_subset)
        w_raw = clf.coef_.flatten()
        w = _unit(w_raw)
        rows.append(
            {
                "C": float(c_inv),
                "norm": float(np.linalg.norm(w_raw)),
                "auc_in_sample": float(roc_auc_score(y_subset, clf.decision_function(z_subset))),
                "cos_naive": abs(float(np.dot(w, u_naive))),
                "cos_fwl": abs(float(np.dot(w, u_fwl))),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Phase D — leave-one-pair-out CV AUC
# ---------------------------------------------------------------------------


def phase_d_loo_pair_auc(
    z_subset: np.ndarray,
    y_subset: np.ndarray,
) -> dict:
    """Hold out each (large, small) pair in turn; train on the rest."""
    assert len(z_subset) % 2 == 0
    n_pairs = len(z_subset) // 2

    held_y: list[int] = []
    held_p: list[float] = []
    pair_correct = 0
    for k in range(n_pairs):
        train_mask = np.ones(2 * n_pairs, dtype=bool)
        train_mask[2 * k] = False
        train_mask[2 * k + 1] = False
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(
            z_subset[train_mask], y_subset[train_mask]
        )
        scores = clf.decision_function(z_subset[~train_mask])
        # Index 0 is the large (label 1), 1 is the small (label 0).
        held_y.extend([1, 0])
        held_p.extend([float(scores[0]), float(scores[1])])
        if scores[0] > scores[1]:
            pair_correct += 1
    held_auc = float(roc_auc_score(held_y, held_p))
    return {
        "loo_pair_auc": held_auc,
        "pair_ranking_accuracy": pair_correct / n_pairs,
        "n_pairs": int(n_pairs),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_bootstrap(boot: dict, out_path: Path, modality: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(
        boot["_cos_naive_samples"],
        bins=40,
        alpha=0.6,
        label=r"$|\cos(u_\text{TCAV}^b, u_\text{naive})|$",
    )
    ax.hist(
        boot["_cos_fwl_samples"],
        bins=40,
        alpha=0.6,
        label=r"$|\cos(u_\text{TCAV}^b, u_\text{FWL})|$",
    )
    ax.axvline(boot["cos_naive_mean"], color="C0", linestyle="--", label="mean (vs naive)")
    ax.axvline(boot["cos_fwl_mean"], color="C1", linestyle="--", label="mean (vs FWL)")
    ax.axvline(0.90, color="grey", linestyle=":", label="spec threshold 0.90")
    ax.set_xlabel("absolute cosine")
    ax.set_ylabel("count")
    ax.set_title(f"Phase A — matched-pair bootstrap (B={boot['n_boot']}, {modality})")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_quantile_sweep(rows: list[dict], out_path: Path, modality: str) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    qs = [r["q_lower"] for r in ok]
    n_pairs = [r["n_pairs"] for r in ok]
    cos_n = [r["cos_naive"] for r in ok]
    cos_f = [r["cos_fwl"] for r in ok]
    aucs = [r["roc_auc"] for r in ok]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(qs, cos_n, "o-", label=r"$|\cos(u_\text{TCAV}, u_\text{naive})|$")
    axes[0].plot(qs, cos_f, "s-", label=r"$|\cos(u_\text{TCAV}, u_\text{FWL})|$")
    axes[0].plot(qs, aucs, "^-", color="C2", alpha=0.6, label="in-sample ROC AUC")
    axes[0].axhline(0.90, color="grey", linestyle=":", label="spec 0.90")
    axes[0].set_ylabel("metric")
    axes[0].set_title(f"Phase B — quantile sweep ({modality}). q=0.5 is median split.")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(qs, n_pairs, "kD-")
    axes[1].set_xlabel(r"small-bin quantile $q$ (large bin = $1-q$)")
    axes[1].set_ylabel("matched pairs")
    axes[1].grid(alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_regularization_sweep(rows: list[dict], out_path: Path, modality: str) -> None:
    cs = [r["C"] for r in rows]
    cos_n = [r["cos_naive"] for r in rows]
    cos_f = [r["cos_fwl"] for r in rows]
    aucs = [r["auc_in_sample"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.semilogx(cs, cos_n, "o-", label=r"$|\cos(u_\text{TCAV}, u_\text{naive})|$")
    ax.semilogx(cs, cos_f, "s-", label=r"$|\cos(u_\text{TCAV}, u_\text{FWL})|$")
    ax.semilogx(cs, aucs, "^-", color="C2", alpha=0.6, label="in-sample ROC AUC")
    ax.axhline(0.90, color="grey", linestyle=":", label="spec 0.90")
    ax.set_xlabel(r"sklearn LogisticRegression $C$ (lower = stronger L2)")
    ax.set_ylabel("metric")
    ax.set_title(f"Phase C — regularization sweep ({modality})")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_one(modality: str, *, n_boot: int = 500, seed: int = 0) -> dict:
    """Run the four-phase diagnostic for one modality. Returns the summary dict."""
    paths = _paths_for(modality)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = paths.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info("=== %s ===", modality)
    loaded = _load(paths.features_h5)

    # Reference directions.
    u_naive = _fit_naive(loaded.z_tumor, loaded.log_v)
    # Sanity-check against the saved E2.1 direction.
    if paths.naive_direction_npz.exists():
        npz = np.load(paths.naive_direction_npz)
        u_naive_e2_1 = _unit(
            npz[npz.files[0]] if "direction" not in npz.files else npz["direction"]
        )
        cos_check = abs(float(np.dot(u_naive, u_naive_e2_1)))
        logger.info("Sanity check: cos(α=0.1 ridge, E2.1 saved direction) = %.4f", cos_check)
    u_fwl = _fit_fwl(loaded)
    cos_naive_fwl = abs(float(np.dot(u_naive, u_fwl)))
    logger.info("|cos(u_naive, u_FWL)| = %.4f", cos_naive_fwl)

    # Base TCAV (default 0.2/0.8).
    base_tcav = matched_control_tcav(
        loaded.z_tumor,
        loaded.log_v,
        loaded.covariates,
        large_quantile=0.8,
        small_quantile=0.2,
        age_window=5.0,
        min_pairs=30,
        seed=seed,
    )
    base_dir = base_tcav.direction
    cos_base_naive = abs(float(np.dot(base_dir, u_naive)))
    cos_base_fwl = abs(float(np.dot(base_dir, u_fwl)))
    logger.info(
        "Base TCAV: n_pairs=%d, AUC=%.3f, |cos(naive)|=%.4f, |cos(FWL)|=%.4f",
        base_tcav.n_pairs,
        base_tcav.roc_auc,
        cos_base_naive,
        cos_base_fwl,
    )

    # Build the matched-set arrays once (used by Phases A, C, D).
    pairs_idx = base_tcav.pairs_index
    z_subset = loaded.z_tumor[pairs_idx]
    y_subset = np.tile([1, 0], base_tcav.n_pairs)

    # Phase A.
    logger.info("Phase A — bootstrap (B=%d) ...", n_boot)
    boot = phase_a_bootstrap(loaded.z_tumor, pairs_idx, u_naive, u_fwl, n_boot=n_boot, seed=seed)
    logger.info(
        "  cos(TCAV_b, naive): %.3f [%.3f, %.3f]",
        boot["cos_naive_mean"],
        boot["cos_naive_p2_5"],
        boot["cos_naive_p97_5"],
    )
    logger.info(
        "  cos(TCAV_b, FWL  ): %.3f [%.3f, %.3f]",
        boot["cos_fwl_mean"],
        boot["cos_fwl_p2_5"],
        boot["cos_fwl_p97_5"],
    )

    # Phase B.
    logger.info("Phase B — quantile sweep ...")
    qsweep = phase_b_quantile_sweep(loaded, u_naive, u_fwl)
    for r in qsweep:
        if r["status"] == "ok":
            logger.info(
                "  q=%.2f n_pairs=%d AUC=%.3f cos_naive=%.3f cos_fwl=%.3f",
                r["q_lower"],
                r["n_pairs"],
                r["roc_auc"],
                r["cos_naive"],
                r["cos_fwl"],
            )
        else:
            logger.warning("  q=%.2f %s", r["q_lower"], r["status"])

    # Phase C.
    logger.info("Phase C — regularization sweep ...")
    csweep = phase_c_regularization_sweep(z_subset, y_subset, u_naive, u_fwl)
    for r in csweep:
        logger.info(
            "  C=%.0e norm=%.3f AUC=%.3f cos_naive=%.3f cos_fwl=%.3f",
            r["C"],
            r["norm"],
            r["auc_in_sample"],
            r["cos_naive"],
            r["cos_fwl"],
        )

    # Phase D.
    logger.info("Phase D — leave-one-pair-out AUC ...")
    loo = phase_d_loo_pair_auc(z_subset, y_subset)
    logger.info(
        "  loo_pair_auc=%.3f pair_ranking_acc=%.3f (n=%d)",
        loo["loo_pair_auc"],
        loo["pair_ranking_accuracy"],
        loo["n_pairs"],
    )

    # Plots.
    _plot_bootstrap(boot, plots_dir / "phase_a_bootstrap_hist.png", modality)
    _plot_quantile_sweep(qsweep, plots_dir / "phase_b_quantile_sweep.png", modality)
    _plot_regularization_sweep(csweep, plots_dir / "phase_c_regularization_sweep.png", modality)

    summary = {
        "modality": modality,
        "n_scans_used": int(len(loaded.z_tumor)),
        "u_naive_vs_u_fwl_cos": cos_naive_fwl,
        "base_tcav": {
            "n_pairs": int(base_tcav.n_pairs),
            "roc_auc_in_sample": float(base_tcav.roc_auc),
            "norm": float(base_tcav.norm),
            "cos_naive": cos_base_naive,
            "cos_fwl": cos_base_fwl,
        },
        "phase_a_bootstrap": {k: v for k, v in boot.items() if not k.startswith("_")},
        "phase_b_quantile_sweep": qsweep,
        "phase_c_regularization_sweep": csweep,
        "phase_d_held_out": loo,
    }
    summary_path = paths.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Summary written to %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        default="all",
        choices=("all", *VALID_MODALITIES),
        help="Run for one modality or all four (default: all).",
    )
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    targets = list(VALID_MODALITIES) if args.modality == "all" else [args.modality]
    summaries: list[dict] = []
    for mod in targets:
        summaries.append(run_one(mod, n_boot=args.n_boot, seed=args.seed))

    if len(summaries) > 1:
        comparison_path = Path(
            "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
            "E2_2_identifiability/per_modality/tcav_diagnostic_comparison.json"
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(summaries, indent=2, default=float))
        logger.info("Cross-modality comparison written to %s", comparison_path)


if __name__ == "__main__":
    main()
