"""Driver: run E2.1 separately for each MAISI-v2 modality slice (t1c, t1n, t2f, t2w).

For each modality:
  1. Build a per-modality features H5 via :class:`ComputeFeaturesEngine`.
  2. Run :func:`run_recoverability` on it.
  3. Aggregate the per-modality result into a comparison JSON + PNG bar.

The aim is to test whether volume decoding from MAISI-v2 latents is
sequence-agnostic. If t1c dominates the four modalities, the chain has to
either restrict to t1c or weight modalities asymmetrically. If R² is similar
across modalities, the linear direction generalizes and the downstream
flow-matching project can use whichever channel is convenient.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.E2.compute_features.engine.compute_features_engine import (
    ComputeFeaturesEngine,
    ComputeFeaturesRoutineConfig,
)
from experiments.E2.E2_1_recoverability.analysis.recoverability import (
    RecoverabilityResult,
    RecoverabilityRoutineConfig,
    run_recoverability,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Paths:
    latent_h5: Path
    source_h5: Path
    features_dir: Path  # where per-modality features H5 land
    results_dir: Path  # where per-modality E2.1 outputs land


_DEFAULTS = _Paths(
    latent_h5=Path("/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/brats_men_maisi_latents.h5"),
    source_h5=Path("/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/h5/brats_men.h5"),
    features_dir=Path("/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/per_modality_features"),
    results_dir=Path(
        "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
        "E2_1_recoverability/per_modality"
    ),
)


def _run_one(modality: str, paths: _Paths) -> tuple[Path, RecoverabilityResult]:
    feats_path = paths.features_dir / f"brats_men_features_{modality}.h5"
    feats_cfg = ComputeFeaturesRoutineConfig(
        latent_h5=paths.latent_h5,
        source_h5=paths.source_h5,
        output_h5=feats_path,
        modality=modality,
        mask_label_set=(1, 2, 3),
        random_region_seed=0,
        compute_z_full=False,
        log_level="WARNING",
    )
    if feats_path.is_file():
        logger.info("[%s] features H5 already exists at %s — reusing.", modality, feats_path)
    else:
        logger.info("[%s] computing features → %s", modality, feats_path)
        ComputeFeaturesEngine(feats_cfg).run()

    out_dir = paths.results_dir / modality
    rec_cfg = RecoverabilityRoutineConfig(
        features_h5=feats_path,
        output_dir=out_dir,
        mode="full",
        bootstrap_n=1000,
        n_splits=5,
        mlp_max_epochs=200,
        mlp_hidden=256,
        mlp_dropout=0.1,
        mlp_lr=1e-3,
        mlp_weight_decay=1e-4,
        mlp_patience=20,
        mlp_batch_size=64,
        ridge_alphas=(0.1, 1.0, 10.0, 100.0),
        seed=0,
        device="cpu",
        log_level="WARNING",
    )
    logger.info("[%s] running E2.1 → %s", modality, out_dir)
    result = run_recoverability(rec_cfg)
    return feats_path, result


def _comparison_plot(rows: list[dict], out_path: Path) -> None:
    mods = [r["modality"] for r in rows]
    r2_lin = np.array([r["r2_lin"] for r in rows])
    r2_mlp = np.array([r["r2_mlp"] for r in rows])
    ci_lo = np.array([r["r2_lin_ci_lower"] for r in rows])
    ci_hi = np.array([r["r2_lin_ci_upper"] for r in rows])
    r2_lin_global = np.array([r["r2_lin_global"] for r in rows])

    x = np.arange(len(mods))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - width, r2_lin, width, label="ridge (mask)", color="#1f77b4")
    ax.bar(x, r2_mlp, width, label="MLP (mask)", color="#ff7f0e")
    ax.bar(x + width, r2_lin_global, width, label="ridge (global)", color="#bbbbbb")

    err_lo = r2_lin - ci_lo
    err_hi = ci_hi - r2_lin
    ax.errorbar(
        x - width,
        r2_lin,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel(r"$R^2$ on $\log V$ (cm³)")
    ax.set_ylim(min(0, min(r2_lin_global.min(), 0) - 0.05), 1.0)
    ax.axhline(0.6, color="grey", lw=0.8, ls=":")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_title("E2.1 — recoverability by MAISI-v2 modality slice")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1c", "t1n", "t2f", "t2w"],
        help="Modalities to evaluate, in this order.",
    )
    parser.add_argument(
        "--latent-h5",
        type=Path,
        default=_DEFAULTS.latent_h5,
    )
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=_DEFAULTS.source_h5,
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=_DEFAULTS.features_dir,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULTS.results_dir,
    )
    args = parser.parse_args()
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    paths = _Paths(
        latent_h5=args.latent_h5,
        source_h5=args.source_h5,
        features_dir=args.features_dir,
        results_dir=args.results_dir,
    )
    paths.features_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for modality in args.modalities:
        feats_path, result = _run_one(modality, paths)
        d = dataclasses.asdict(result)
        d.pop("direction", None)  # keep comparison.json compact
        d["modality"] = modality
        d["features_h5"] = str(feats_path)
        rows.append(d)
        logger.info(
            "[%s] R²_lin=%.4f (CI [%.4f, %.4f]) R²_mlp=%.4f gap=%.4f regime=%s",
            modality,
            d["r2_lin"],
            d["r2_lin_ci_lower"],
            d["r2_lin_ci_upper"],
            d["r2_mlp"],
            d["r2_gap"],
            d["regime"],
        )

    comp_json = paths.results_dir / "comparison.json"
    with open(comp_json, "w") as fh:
        json.dump(rows, fh, indent=2, default=_json_default)
    _comparison_plot(rows, paths.results_dir / "comparison.png")
    logger.info("Wrote %s and comparison.png", comp_json)


def _json_default(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
