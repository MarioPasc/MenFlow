"""CLI entry point for E2.1 — volume recoverability.

Usage::

    python -m experiments.E2.E2_1_recoverability.cli experiments/E2_1_recoverability/configs/local_brats_men.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiments.E2.E2_1_recoverability.analysis.recoverability import (
    RecoverabilityRoutineConfig,
    run_recoverability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2.1 volume recoverability probe.")
    parser.add_argument("config", type=Path, help="Path to the experiment YAML config.")
    args = parser.parse_args()

    cfg = RecoverabilityRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    run_recoverability(cfg)


if __name__ == "__main__":
    main()
