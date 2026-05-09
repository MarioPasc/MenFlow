"""Driver: run E2.3 spatial localization for each MAISI-v2 modality slice.

For each modality (t1c, t1n, t2f, t2w):
  1. Build an in-memory SpatialRoutineConfig pointing at the per-modality features H5.
  2. Run run_spatial.
  3. After all modalities, write per_modality/comparison.json and comparison.png.

NOTE: Not executed in the initial iteration (t1c only). Wire is complete for a
future single-command sweep.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.E2_3_spatial_localization.analysis.spatial import (
    SpatialResult,
    SpatialRoutineConfig,
    run_spatial,
)

logger = logging.getLogger(__name__)

MODALITIES = ("t1c", "t1n", "t2f", "t2w")

FEATURES_DIR = Path("/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/per_modality_features")
SOURCE_H5 = Path("/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/h5/brats_men.h5")
CLINICAL_XLSX = Path(
    "/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/"
    "Meningioma_supplementary_clinical_data_and_imaging_parameters_for_training_and_validation_sets.xlsx"
)
RESULTS_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
    "E2_3_spatial_localization/per_modality"
)
LATERALITY_CACHE = Path(
    "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
    "E2_2_identifiability/laterality_cache.csv"
)


@dataclass(frozen=True, slots=True)
class _RunRow:
    modality: str
    status: str
    result: SpatialResult | None


def _run_one(modality: str, n_boot: int = 1000, seed: int = 0) -> _RunRow:
    """Run E2.3 for one modality.

    Parameters
    ----------
    modality:
        One of "t1c", "t1n", "t2f", "t2w".
    n_boot:
        Bootstrap replicates.
    seed:
        RNG seed.

    Returns
    -------
    _RunRow
        Status and result (or None on failure).
    """
    cfg = SpatialRoutineConfig(
        features_h5=FEATURES_DIR / f"brats_men_features_{modality}.h5",
        source_h5=SOURCE_H5,
        clinical_xlsx=CLINICAL_XLSX,
        output_dir=RESULTS_DIR / modality,
        n_boot=n_boot,
        seed=seed,
        log_level="WARNING",
        laterality_cache_csv=LATERALITY_CACHE,
        synthetic=False,
    )
    logger.info("[%s] Starting E2.3 -> %s", modality, cfg.output_dir)
    try:
        result = run_spatial(cfg)
        return _RunRow(modality=modality, status="ok", result=result)
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Failed: %s", modality, exc)
        return _RunRow(modality=modality, status=f"failed: {exc}", result=None)


def _comparison_plot(rows: list[_RunRow], out_path: Path) -> None:
    """4-row table-style figure: r2_tumor / r2_global / delta_r2_post_fwl / operator."""
    mods = [r.modality for r in rows]
    x = np.arange(len(mods))
    width = 0.28

    r2_tumor = np.array([r.result.r2_tumor if r.result is not None else 0.0 for r in rows])
    r2_global = np.array([r.result.r2_global if r.result is not None else 0.0 for r in rows])
    delta_post = np.array(
        [r.result.delta_r2_post_fwl if r.result is not None else 0.0 for r in rows]
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(x - width, r2_tumor, width, label="r2_tumor", color="#2ca02c", alpha=0.85)
    ax.bar(x, r2_global, width, label="r2_global", color="#1f77b4", alpha=0.85)
    ax.bar(
        x + width,
        delta_post,
        width,
        label="delta_r2_post_fwl",
        color="#ff7f0e",
        alpha=0.85,
    )

    ax.axhline(0.15, color="#2ca02c", lw=0.8, ls="--", label="strongly_local 0.15")
    ax.axhline(0.05, color="#ff7f0e", lw=0.8, ls=":", label="local 0.05")

    ax.set_xticks(list(x))
    ax.set_xticklabels(mods)
    ax.set_ylabel("R² / ΔR²")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("E2.3 — Spatial localization by modality")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate operator below bars
    for i, row in enumerate(rows):
        label = row.result.operator if row.result is not None else row.status
        ax.text(x[i], -0.08, label, ha="center", va="top", fontsize=7, rotation=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved comparison plot to %s", out_path)


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(MODALITIES),
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[_RunRow] = []
    for modality in args.modalities:
        rows.append(_run_one(modality, n_boot=args.n_boot, seed=args.seed))
        r = rows[-1]
        if r.result is not None:
            logger.info(
                "[%s] status=%s r2_tumor=%.4f r2_global=%.4f delta_post=%.4f operator=%s",
                modality,
                r.status,
                r.result.r2_tumor,
                r.result.r2_global,
                r.result.delta_r2_post_fwl,
                r.result.operator,
            )
        else:
            logger.warning("[%s] status=%s", modality, r.status)

    # Comparison artifacts
    comp_data = []
    for row in rows:
        d: dict = {"modality": row.modality, "status": row.status}
        if row.result is not None:
            d.update(dataclasses.asdict(row.result))
        comp_data.append(d)

    comp_json = RESULTS_DIR / "comparison.json"
    with open(comp_json, "w") as fh:
        json.dump(comp_data, fh, indent=2, default=_json_default)
    logger.info("Wrote %s", comp_json)

    _comparison_plot(rows, RESULTS_DIR / "comparison.png")


if __name__ == "__main__":
    main()
