"""CLI entry point for the decode routine.

Usage::

    python -m routines.decode.cli routines/decode/configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from routines.decode.engine.decode_engine import DecodeEngine, DecodeRoutineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode MAISI-v2 latents to a MenFlow H5.")
    parser.add_argument("config", type=Path, help="Path to the routine YAML config.")
    args = parser.parse_args()

    cfg = DecodeRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    DecodeEngine(cfg).run()


if __name__ == "__main__":
    main()
