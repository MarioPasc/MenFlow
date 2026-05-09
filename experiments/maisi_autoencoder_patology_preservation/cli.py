"""CLI for the E1 analysis step.

Inputs (from a YAML config):
    source_h5      original MenFlow cohort H5
    recon_h5       MAISI-v2 reconstruction H5 (output of decode routine)
    output_dir     where CSVs and figures land
    max_scans      optional truncation
    device         "cpu" or "cuda" — only matters for the few SSIM forward passes

Outputs:
    output_dir/per_scan_metrics.csv
    output_dir/by_bin_summary.csv
    output_dir/huber_regression.csv
    output_dir/breakpoint.csv
    output_dir/figures/{box_<metric>.png, loglog_<metric>.png}

Usage:
    python -m experiments.maisi_autoencoder_patology_preservation.cli \
        experiments/maisi_autoencoder_patology_preservation/configs/local_3060.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from experiments.maisi_autoencoder_patology_preservation.analysis.metrics import (
    compute_metrics,
)
from experiments.maisi_autoencoder_patology_preservation.analysis.plotting import (
    render_all,
)
from experiments.maisi_autoencoder_patology_preservation.analysis.regression import (
    fit_all,
)
from experiments.maisi_autoencoder_patology_preservation.analysis.volume_stratification import (
    stratify_by_volume,
    summarize_by_bin,
)

logger = logging.getLogger(__name__)


# Metric columns reported by ``compute_metrics`` and consumed downstream.
DEFAULT_METRICS = (
    "psnr_global",
    "ssim_global",
    "msssim_global",
    "psnr_wt",
    "ssim_wt",
    "mae_wt",
    "psnr_tc",
    "ssim_tc",
    "mae_tc",
)


@dataclass(frozen=True, slots=True)
class AnalysisRoutineConfig:
    source_h5: Path
    recon_h5: Path
    output_dir: Path
    max_scans: int | None = None
    device: str = "cpu"
    n_bootstrap: int = 500
    seed: int = 0
    require_segmentation: bool = True
    log_level: str = "INFO"
    metrics: tuple[str, ...] = field(default_factory=lambda: DEFAULT_METRICS)
    # which volume mask drives the §5 stratification
    stratification_volume: str = "volume_wt_cm3"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalysisRoutineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("slurm", None)
        for path_key in ("source_h5", "recon_h5", "output_dir"):
            if path_key in raw and raw[path_key] is not None:
                raw[path_key] = Path(raw[path_key]).expanduser()
        if "metrics" in raw and raw["metrics"] is not None:
            raw["metrics"] = tuple(raw["metrics"])
        return cls(**raw)


def run(cfg: AnalysisRoutineConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    per_scan_csv = cfg.output_dir / "per_scan_metrics.csv"
    by_bin_csv = cfg.output_dir / "by_bin_summary.csv"
    huber_csv = cfg.output_dir / "huber_regression.csv"
    bp_csv = cfg.output_dir / "breakpoint.csv"
    fig_dir = cfg.output_dir / "figures"

    logger.info("Computing per-scan metrics...")
    df = compute_metrics(
        source_h5=str(cfg.source_h5),
        recon_h5=str(cfg.recon_h5),
        output_csv=str(per_scan_csv),
        max_scans=cfg.max_scans,
        device=cfg.device,
        require_segmentation=cfg.require_segmentation,
    )
    if df.empty:
        logger.warning("No metrics computed — every scan was skipped (no segmentations?)")
        return cfg.output_dir

    df = stratify_by_volume(df, volume_field=cfg.stratification_volume)
    df.to_csv(per_scan_csv, index=False)

    logger.info("Summarising by volume bin...")
    by_bin = summarize_by_bin(df, metric_columns=cfg.metrics)
    by_bin.to_csv(by_bin_csv, index=False)

    logger.info("Fitting Huber regression and breakpoint detection...")
    huber_df, bp_df = fit_all(
        df,
        metrics=cfg.metrics,
        volume_field=cfg.stratification_volume,
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.seed,
    )
    huber_df.to_csv(huber_csv, index=False)
    bp_df.to_csv(bp_csv, index=False)

    logger.info("Rendering figures...")
    render_all(
        metrics_df=df,
        huber_df=huber_df,
        breakpoint_df=bp_df,
        metrics=cfg.metrics,
        volume_field=cfg.stratification_volume,
        output_dir=fig_dir,
    )

    logger.info("E1 analysis complete: %s", cfg.output_dir)
    return cfg.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E1 reconstruction-sufficiency analysis.")
    parser.add_argument("config", type=Path, help="Path to the analysis YAML config.")
    args = parser.parse_args()

    cfg = AnalysisRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    run(cfg)


if __name__ == "__main__":
    main()
