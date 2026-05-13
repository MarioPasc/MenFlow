"""Driver: run E2.2 direction identifiability for each MAISI-v2 modality slice.

For each modality (t1c, t1n, t2f, t2w):
  1. Build an in-memory IdentifiabilityRoutineConfig pointing at the
     per-modality features H5 and the corresponding E2.1 naive direction.
  2. Run run_identifiability.
  3. Catch InsufficientMatchedControlsError and record status="failed_matching".
  4. After all modalities, write comparison.json and comparison.png.

The laterality parquet cache is shared across all modalities (one I/O pass over
the segmentation array rather than four).
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

from experiments.E2._lib.directions import InsufficientMatchedControlsError
from experiments.E2.E2_2_identifiability.analysis.identifiability import (
    IdentifiabilityResult,
    IdentifiabilityRoutineConfig,
    run_identifiability,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Paths:
    source_h5: Path
    clinical_xlsx: Path
    features_dir: Path  # per_modality_features/{mod}.h5 live here
    e21_results_dir: Path  # E2.1 per_modality/{mod}/direction.npz
    results_dir: Path  # E2.2 per_modality/{mod}/ outputs
    laterality_cache: Path  # shared parquet; written once, reused per modality


_DEFAULTS = _Paths(
    source_h5=Path("/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/h5/brats_men.h5"),
    clinical_xlsx=Path(
        "/media/mpascual/MeningD2/MENINGIOMAS/BRATS_MEN/"
        "Meningioma_supplementary_clinical_data_and_imaging_parameters_for_training_and_validation_sets.xlsx"
    ),
    features_dir=Path("/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/per_modality_features"),
    e21_results_dir=Path(
        "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
        "E2_1_recoverability/per_modality"
    ),
    results_dir=Path(
        "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
        "E2_2_identifiability/per_modality"
    ),
    laterality_cache=Path(
        "/media/mpascual/Sandisk2TB/research/menflow/experiments/E2/"
        "E2_2_identifiability/laterality_cache.csv"
    ),
)


def _run_one(
    modality: str,
    paths: _Paths,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[IdentifiabilityResult | None, str]:
    """Run E2.2 for one modality.

    Returns
    -------
    result
        IdentifiabilityResult on success; None on InsufficientMatchedControlsError.
    status
        ``"ok"`` or ``"failed_matching"``.
    """
    feats_path = paths.features_dir / f"brats_men_features_{modality}.h5"
    naive_npz = paths.e21_results_dir / modality / "direction.npz"
    out_dir = paths.results_dir / modality

    cfg = IdentifiabilityRoutineConfig(
        features_h5=feats_path,
        source_h5=paths.source_h5,
        clinical_xlsx=paths.clinical_xlsx,
        naive_direction_npz=naive_npz,
        output_dir=out_dir,
        n_boot=n_boot,
        large_quantile=0.8,
        small_quantile=0.2,
        age_window=5.0,
        min_pairs=30,
        seed=seed,
        log_level="WARNING",
        laterality_cache_parquet=paths.laterality_cache,
    )
    logger.info("[%s] Starting E2.2 → %s", modality, out_dir)
    try:
        result = run_identifiability(cfg)
        return result, "ok"
    except InsufficientMatchedControlsError as exc:
        logger.warning("[%s] InsufficientMatchedControlsError: %s", modality, exc)
        return None, "failed_matching"


def _comparison_plot(rows: list[dict], out_path: Path) -> None:
    """Table-style bar figure showing key metrics per modality."""
    mods = [r["modality"] for r in rows]
    x = np.arange(len(mods))
    width = 0.25

    rho_boot = np.array(
        [r.get("rho_boot_mean", 0.0) if r.get("status") == "ok" else 0.0 for r in rows]
    )
    rho_tcav = np.array(
        [r.get("rho_conf_tcav", 0.0) if r.get("status") == "ok" else 0.0 for r in rows]
    )
    rho_fwl = np.array(
        [r.get("rho_conf_fwl", 0.0) if r.get("status") == "ok" else 0.0 for r in rows]
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - width, rho_boot, width, label="rho_boot_mean", color="#1f77b4")
    ax.bar(x, rho_tcav, width, label="rho_conf_tcav", color="#ff7f0e")
    ax.bar(x + width, rho_fwl, width, label="rho_conf_fwl", color="#2ca02c")

    ax.axhline(0.90, color="grey", lw=0.8, ls="--", label="0.90")
    ax.axhline(0.70, color="grey", lw=0.8, ls=":", label="0.70")
    ax.axhline(0.60, color="grey", lw=0.6, ls="-.", label="0.60")

    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel("Cosine / stability")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("E2.2 — Direction identifiability by modality")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate regime + decision below bars
    for i, r in enumerate(rows):
        label = r.get("regime", r.get("status", "?"))
        ax.text(x[i], -0.12, label, ha="center", va="top", fontsize=6, rotation=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1c", "t1n", "t2f", "t2w"],
    )
    parser.add_argument("--source-h5", type=Path, default=_DEFAULTS.source_h5)
    parser.add_argument("--clinical-xlsx", type=Path, default=_DEFAULTS.clinical_xlsx)
    parser.add_argument("--features-dir", type=Path, default=_DEFAULTS.features_dir)
    parser.add_argument("--e21-results-dir", type=Path, default=_DEFAULTS.e21_results_dir)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULTS.results_dir)
    parser.add_argument("--laterality-cache", type=Path, default=_DEFAULTS.laterality_cache)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    paths = _Paths(
        source_h5=args.source_h5,
        clinical_xlsx=args.clinical_xlsx,
        features_dir=args.features_dir,
        e21_results_dir=args.e21_results_dir,
        results_dir=args.results_dir,
        laterality_cache=args.laterality_cache,
    )
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for modality in args.modalities:
        result, status = _run_one(modality, paths, n_boot=args.n_boot, seed=args.seed)
        if result is not None:
            d = dataclasses.asdict(result)
            d["modality"] = modality
            d["status"] = status
        else:
            d = {"modality": modality, "status": status}
        rows.append(d)
        logger.info(
            "[%s] status=%s%s",
            modality,
            status,
            (
                f" regime={d.get('regime')} decision={d.get('decision')}"
                f" rho_boot={d.get('rho_boot_mean', 'n/a'):.4f}"
                f" rho_tcav={d.get('rho_conf_tcav', 'n/a'):.4f}"
                if status == "ok"
                else ""
            ),
        )

    comp_json = paths.results_dir / "comparison.json"
    with open(comp_json, "w") as fh:
        json.dump(rows, fh, indent=2, default=_json_default)
    _comparison_plot(rows, paths.results_dir / "comparison.png")
    logger.info("Wrote %s and comparison.png", comp_json)


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
