"""CLI entry point for the encode routine.

Usage::

    python -m routines.encode.cli routines/encode/config.yaml

The single positional argument is the YAML config file. Everything else is
controlled from there: which H5 to encode, which checkpoint, target spatial
shape, dtype, etc.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from routines.encode.engine.encode_engine import EncodeEngine, EncodeRoutineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a MenFlow H5 cohort with MAISI-v2.")
    parser.add_argument("config", type=Path, help="Path to the routine YAML config.")
    args = parser.parse_args()

    cfg = EncodeRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    EncodeEngine(cfg).run()


if __name__ == "__main__":
    main()
