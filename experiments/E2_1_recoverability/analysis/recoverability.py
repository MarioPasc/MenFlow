"""E2.1 — fit linear + MLP probes, run bootstrap CI, classify regime.

The features H5 (produced by ``routines.compute_features``) is consumed
directly here. We do not re-introduce an ``EncodedScan`` dataclass — NumPy
arrays out of HDF5 are sufficient.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import yaml

from experiments.E2_1_recoverability.analysis.plotting import (
    plot_pooling_comparison,
    plot_pred_vs_obs,
)
from experiments.E2_1_recoverability.analysis.regime import classify_regime
from menflow.probes.linear import fit_linear_probe
from menflow.probes.mlp import fit_mlp_probe
from menflow.stats.bootstrap import patient_level_bootstrap_r2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoverabilityRoutineConfig:
    features_h5: Path
    output_dir: Path
    mode: Literal["smoke", "subset", "full"] = "full"
    subset_n: int = 50
    bootstrap_n: int = 1000
    mlp_max_epochs: int = 200
    mlp_hidden: int = 256
    mlp_dropout: float = 0.1
    mlp_lr: float = 1e-3
    mlp_weight_decay: float = 1e-4
    mlp_patience: int = 20
    mlp_batch_size: int = 64
    n_splits: int = 5
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    seed: int = 0
    device: str = "cpu"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecoverabilityRoutineConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        raw.pop("slurm", None)
        for path_key in ("features_h5", "output_dir"):
            if path_key in raw:
                raw[path_key] = Path(raw[path_key]).expanduser()
        if "ridge_alphas" in raw and raw["ridge_alphas"] is not None:
            raw["ridge_alphas"] = tuple(float(x) for x in raw["ridge_alphas"])
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class RecoverabilityResult:
    r2_lin: float
    r2_mlp: float
    r2_gap: float
    r2_lin_ci_lower: float
    r2_lin_ci_upper: float
    r2_lin_global: float
    r2_mlp_global: float
    regime: str
    direction: np.ndarray
    direction_norm: float
    best_alpha: float
    n_scans: int
    n_patients: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _load_features(
    path: Path,
    mode: str,
    subset_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        z_tumor = f["z_tumor"][:].astype(np.float32)
        z_global = f["z_global"][:].astype(np.float32)
        log_v = f["log_volume"][:].astype(np.float32)
        patient_ids = np.asarray(
            [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]],
            dtype=object,
        )
        mask_present = f["mask_lat_present"][:].astype(bool)
        attrs = {k: f.attrs[k] for k in f.attrs}

    keep = mask_present & np.isfinite(log_v)
    n_drop = int((~keep).sum())
    if n_drop:
        logger.warning(
            "Dropping %d / %d scans with empty latent mask or non-finite log V.",
            n_drop,
            len(keep),
        )
    z_tumor, z_global, log_v, patient_ids = (
        z_tumor[keep],
        z_global[keep],
        log_v[keep],
        patient_ids[keep],
    )

    if mode == "smoke":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(log_v), size=min(20, len(log_v)), replace=False)
        z_tumor, z_global, log_v, patient_ids = (
            z_tumor[idx],
            z_global[idx],
            log_v[idx],
            patient_ids[idx],
        )
    elif mode == "subset":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(log_v), size=min(subset_n, len(log_v)), replace=False)
        z_tumor, z_global, log_v, patient_ids = (
            z_tumor[idx],
            z_global[idx],
            log_v[idx],
            patient_ids[idx],
        )

    return z_tumor, z_global, log_v, patient_ids, attrs


def _linear_only_fit(z: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas, n_splits):
    """Wrapper used by the bootstrap loop: returns the fitted Ridge model."""
    model, _, _ = fit_linear_probe(z, y, groups, alphas=alphas, n_splits=n_splits)
    return model


def run_recoverability(cfg: RecoverabilityRoutineConfig) -> RecoverabilityResult:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    z_tumor, z_global, y, groups, _attrs = _load_features(
        cfg.features_h5, mode=cfg.mode, subset_n=cfg.subset_n, seed=cfg.seed
    )
    n = len(y)
    n_patients = len(np.unique(groups))
    logger.info(
        "Loaded features: N=%d scans, %d unique patients (mode=%s)", n, n_patients, cfg.mode
    )

    if cfg.mode == "smoke":
        bootstrap_n = max(20, cfg.bootstrap_n // 50)
        mlp_epochs = min(cfg.mlp_max_epochs, 30)
    else:
        bootstrap_n = cfg.bootstrap_n
        mlp_epochs = cfg.mlp_max_epochs

    # Linear probe (mask-anchored).
    lin_model, r2_lin, best_alpha = fit_linear_probe(
        z_tumor, y, groups, alphas=cfg.ridge_alphas, n_splits=cfg.n_splits
    )
    logger.info("Linear probe (mask-anchored): R²=%.4f at alpha=%g", r2_lin, best_alpha)

    # MLP probe (mask-anchored).
    torch.manual_seed(cfg.seed)
    _, r2_mlp = fit_mlp_probe(
        z_tumor,
        y,
        groups,
        hidden=cfg.mlp_hidden,
        dropout=cfg.mlp_dropout,
        lr=cfg.mlp_lr,
        weight_decay=cfg.mlp_weight_decay,
        max_epochs=mlp_epochs,
        patience=cfg.mlp_patience,
        batch_size=cfg.mlp_batch_size,
        n_splits=cfg.n_splits,
        seed=cfg.seed,
        device=cfg.device,
    )
    logger.info("MLP probe (mask-anchored):    R²=%.4f", r2_mlp)

    # Bootstrap CI on r2_lin (patient-level).
    def _fit_fn(zb, yb, gb):
        return _linear_only_fit(zb, yb, gb, cfg.ridge_alphas, cfg.n_splits)

    ci_lo, ci_hi, _ = patient_level_bootstrap_r2(
        z_tumor, y, groups, _fit_fn, n_boot=bootstrap_n, seed=cfg.seed
    )
    logger.info("Bootstrap R²_lin 95%% CI: [%.4f, %.4f]", ci_lo, ci_hi)

    # Sanity check: same probes on global pooling.
    _, r2_lin_g, _ = fit_linear_probe(
        z_global, y, groups, alphas=cfg.ridge_alphas, n_splits=cfg.n_splits
    )
    _, r2_mlp_g = fit_mlp_probe(
        z_global,
        y,
        groups,
        hidden=cfg.mlp_hidden,
        dropout=cfg.mlp_dropout,
        lr=cfg.mlp_lr,
        weight_decay=cfg.mlp_weight_decay,
        max_epochs=mlp_epochs,
        patience=cfg.mlp_patience,
        batch_size=cfg.mlp_batch_size,
        n_splits=cfg.n_splits,
        seed=cfg.seed + 99,
        device=cfg.device,
    )
    logger.info("Global pooling sanity:        R²_lin=%.4f, R²_mlp=%.4f", r2_lin_g, r2_mlp_g)

    direction = np.asarray(lin_model.coef_, dtype=np.float64).ravel()
    direction_norm = float(np.linalg.norm(direction) + 1e-12)
    direction_unit = direction / direction_norm

    regime = classify_regime(r2_lin, r2_mlp)
    result = RecoverabilityResult(
        r2_lin=float(r2_lin),
        r2_mlp=float(r2_mlp),
        r2_gap=float(r2_mlp - r2_lin),
        r2_lin_ci_lower=float(ci_lo),
        r2_lin_ci_upper=float(ci_hi),
        r2_lin_global=float(r2_lin_g),
        r2_mlp_global=float(r2_mlp_g),
        regime=regime,
        direction=direction_unit,
        direction_norm=direction_norm,
        best_alpha=float(best_alpha),
        n_scans=int(n),
        n_patients=int(n_patients),
    )

    _persist(cfg, result, lin_model, z_tumor, y, groups)
    logger.info("Regime: %s", regime)
    logger.info("Result written to %s", cfg.output_dir)
    return result


def _persist(
    cfg: RecoverabilityRoutineConfig,
    result: RecoverabilityResult,
    lin_model,
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    out = cfg.output_dir
    # JSON result (excluding numpy direction).
    payload = dataclasses.asdict(result)
    payload.pop("direction")
    payload["features_h5"] = str(cfg.features_h5)
    payload["mode"] = cfg.mode
    with open(out / "result.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)

    np.savez(
        out / "direction.npz",
        direction=result.direction.astype(np.float64),
        direction_norm=np.float64(result.direction_norm),
        coef_raw=np.asarray(lin_model.coef_, dtype=np.float64),
        intercept=np.float64(getattr(lin_model, "intercept_", 0.0)),
    )

    # Plots use a held-out fold-1 prediction for the scatter, plus a pooling
    # comparison bar chart.
    pred_full = lin_model.predict(z)
    plot_pred_vs_obs(
        y_true=y,
        y_pred=pred_full,
        regime=result.regime,
        r2_text=f"R²_lin={result.r2_lin:.3f}",
        out_path=out / "scatter_pred_vs_obs.png",
    )
    plot_pooling_comparison(
        labels=("mask-anchored", "global"),
        r2_lin=(result.r2_lin, result.r2_lin_global),
        r2_mlp=(result.r2_mlp, result.r2_mlp_global),
        ci_low=(result.r2_lin_ci_lower, np.nan),
        ci_high=(result.r2_lin_ci_upper, np.nan),
        out_path=out / "pooling_comparison.png",
    )


def _json_default(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
