"""CLI entry for the segment-brats routine.

Usage::

    python -m routines.segment_brats.cli routines/segment_brats/configs/brats_men_val_pseudolabel.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from routines.segment_brats.engine.segment_brats_engine import (
    SegmentBratsEngine,
    SegmentBratsRoutineConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a BraTS-MEN winner Docker container over a directory of "
            "per-subject 4-modality NIfTIs (or a unified-schema H5)."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the routine YAML config.")
    args = parser.parse_args()

    cfg = SegmentBratsRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    SegmentBratsEngine(cfg).run()


if __name__ == "__main__":
    main()
