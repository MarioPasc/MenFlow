"""CLI entry point for the compute-features routine.

Usage::

    python -m routines.compute_features.cli routines/compute_features/configs/local_brats_men.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from routines.compute_features.engine.compute_features_engine import (
    ComputeFeaturesEngine,
    ComputeFeaturesRoutineConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pooled MAISI-v2 latent features (z_tumor, z_global, z_random) "
            "and log V per scan from a latents H5 + source MenFlow H5."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to the routine YAML config.")
    args = parser.parse_args()

    cfg = ComputeFeaturesRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    ComputeFeaturesEngine(cfg).run()


if __name__ == "__main__":
    main()
