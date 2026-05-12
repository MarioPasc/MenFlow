"""CLI entry for the steer-decode routine.

Usage::

    python -m experiments.E2.E2_4_causal_steering.steer_decode.cli routines/steer_decode/configs/local_3060_t1c.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiments.E2.E2_4_causal_steering.steer_decode.engine.steer_decode_engine import (
    SteerDecodeEngine,
    SteerDecodeRoutineConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Steer MAISI-v2 latents along a learned volume direction, decode the "
            "perturbed latents to NIfTI, and persist sweep metadata for downstream "
            "segmentation (Phase B of E2.4)."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the routine YAML config.")
    args = parser.parse_args()

    cfg = SteerDecodeRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    SteerDecodeEngine(cfg).run()


if __name__ == "__main__":
    main()
