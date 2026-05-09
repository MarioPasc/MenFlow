"""CLI entry point for E2.3 — spatial localization.

Usage::

    python -m experiments.E2_3_spatial_localization.cli experiments/E2_3_spatial_localization/configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiments.E2_3_spatial_localization.analysis.spatial import (
    SpatialRoutineConfig,
    run_spatial,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2.3 spatial localization.")
    parser.add_argument("config", type=Path, help="Path to the experiment YAML config.")
    args = parser.parse_args()

    cfg = SpatialRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    run_spatial(cfg)


if __name__ == "__main__":
    main()
