"""E2.5 CLI: ``python -m experiments.E2.E2_5_ae_longitudinal.cli <yaml>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiments.E2.E2_5_ae_longitudinal.config import AELongitudinalConfig
from experiments.E2.E2_5_ae_longitudinal.engine import AELongitudinalEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2.5 — MAISI AE longitudinal traversal diagnostic."
    )
    parser.add_argument("config", type=Path, help="YAML config path.")
    args = parser.parse_args()

    cfg = AELongitudinalConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    AELongitudinalEngine(cfg).run()


if __name__ == "__main__":
    main()
