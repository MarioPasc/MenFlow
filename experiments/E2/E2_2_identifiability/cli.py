"""CLI entry point for E2.2 — direction identifiability.

Usage::

    python -m experiments.E2.E2_2_identifiability.cli experiments/E2_2_identifiability/configs/local_brats_men.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiments.E2.E2_2_identifiability.analysis.identifiability import (
    IdentifiabilityRoutineConfig,
    run_identifiability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2.2 direction identifiability.")
    parser.add_argument("config", type=Path, help="Path to the experiment YAML config.")
    args = parser.parse_args()

    cfg = IdentifiabilityRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    run_identifiability(cfg)


if __name__ == "__main__":
    main()
