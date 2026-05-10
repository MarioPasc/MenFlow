"""CLI entry point for the finetune_fm_volume routine.

Usage:
    python -m routines.finetune_fm_volume.cli <yaml>
    menflow-finetune-fm-volume <yaml>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from routines.finetune_fm_volume.engine.finetune_fm_volume_engine import (
    FinetuneFMVolumeEngine,
    FinetuneFMVolumeRoutineConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Volume-conditional MAISI-v2 FM finetune.")
    parser.add_argument("config", type=Path, help="Path to the routine YAML config.")
    args = parser.parse_args()
    cfg = FinetuneFMVolumeRoutineConfig.from_yaml(args.config)
    # rich handler gives us level coloring and rendered tracebacks; engine
    # logger.* calls remain unchanged.
    logging.basicConfig(
        level=cfg.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    FinetuneFMVolumeEngine(cfg).run()


if __name__ == "__main__":
    main()
