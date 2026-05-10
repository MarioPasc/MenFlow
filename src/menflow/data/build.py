"""Single entry point for building MenFlow HDF5 datasets.

Replaces the old per-cohort console scripts (``menflow-convert-brats-men``,
``menflow-convert-mengrowth``) with one ``menflow-build`` command that
dispatches on ``--cohort``. The legacy scripts continue to exist as thin
shims that delegate here, so previous shell scripts keep working.

The unified CLI hard-wires the canonical k-fold layout (k ∈ {1, 3, 5, 10})
shared across cohorts, so every output H5 carries identical split semantics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================


def build_brats_men(
    *,
    train_root: Path | None,
    val_root: Path | None,
    output: Path,
    metadata_xlsx: Path | None = None,
    max_scans: int | None = None,
    compression_level: int = 4,
    kfold_k_values: tuple[int, ...] = (1, 3, 5, 10),
    kfold_test_pct: float = 0.1,
    kfold_seed: int = 42,
) -> Path:
    """Convert BraTS-MEN-2023 NIfTI cohort to the unified H5 schema."""
    from menflow.data.conversors.brats_men import BraTSMENConverter

    if train_root is None and val_root is None:
        raise ValueError("at least one of --train-root / --val-root is required")

    roots: dict[str, Path] = {}
    if train_root is not None:
        roots["train"] = train_root
    if val_root is not None:
        roots["val"] = val_root

    conv = BraTSMENConverter(
        metadata_xlsx=metadata_xlsx,
        kfold_k_values=kfold_k_values,
        kfold_test_pct=kfold_test_pct,
        kfold_seed=kfold_seed,
    )
    return conv.convert(
        roots,
        output,
        max_scans=max_scans,
        compression_level=compression_level,
    )


def build_mengrowth(
    *,
    root: Path,
    output: Path,
    max_scans: int | None = None,
    compression_level: int = 4,
    kfold_k_values: tuple[int, ...] = (1, 3, 5),
    kfold_test_pct: float = 0.1,
    kfold_seed: int = 42,
) -> Path:
    """Convert MenGrowth NIfTI cohort to the unified H5 schema."""
    from menflow.data.conversors.mengrowth import MenGrowthConverter

    conv = MenGrowthConverter(
        kfold_k_values=kfold_k_values,
        kfold_test_pct=kfold_test_pct,
        kfold_seed=kfold_seed,
    )
    return conv.convert(
        {"root": root},
        output,
        max_scans=max_scans,
        compression_level=compression_level,
    )


# ============================================================================
# CLI
# ============================================================================


def _add_common_kfold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--kfold-k-values",
        type=int,
        nargs="+",
        default=None,
        help="K-fold k values to pre-compute. Default depends on cohort size.",
    )
    p.add_argument(
        "--kfold-test-pct",
        type=float,
        default=0.1,
        help="Held-out test fraction (shared across every k). Default 0.1.",
    )
    p.add_argument(
        "--kfold-seed",
        type=int,
        default=42,
        help="Random seed driving the test holdout and per-fold shuffles.",
    )


def cli() -> None:
    """``menflow-build`` console entry point."""
    parser = argparse.ArgumentParser(
        description="Build a MenFlow HDF5 dataset (one entry point for all cohorts)."
    )
    parser.add_argument(
        "--cohort",
        required=True,
        choices=("brats_men", "mengrowth"),
        help="Which cohort to build.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .h5 file.")
    parser.add_argument(
        "--max-scans", type=int, default=None, help="Limit to first N scans (dry run)."
    )
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    # BraTS-MEN-specific paths
    parser.add_argument(
        "--train-root", type=Path, default=None, help="(brats_men) Annotated cohort root directory."
    )
    parser.add_argument(
        "--val-root", type=Path, default=None, help="(brats_men) Unannotated cohort root directory."
    )
    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=None,
        help="(brats_men) Optional clinical metadata spreadsheet.",
    )

    # MenGrowth-specific paths
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="(mengrowth) Single root containing patient subdirs.",
    )

    _add_common_kfold_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.cohort == "brats_men":
        kvals = tuple(args.kfold_k_values) if args.kfold_k_values else (1, 3, 5, 10)
        out = build_brats_men(
            train_root=args.train_root,
            val_root=args.val_root,
            output=args.output,
            metadata_xlsx=args.metadata_xlsx,
            max_scans=args.max_scans,
            compression_level=args.compression_level,
            kfold_k_values=kvals,
            kfold_test_pct=args.kfold_test_pct,
            kfold_seed=args.kfold_seed,
        )
    elif args.cohort == "mengrowth":
        if args.root is None:
            parser.error("--root is required for cohort=mengrowth")
        kvals = tuple(args.kfold_k_values) if args.kfold_k_values else (1, 3, 5)
        out = build_mengrowth(
            root=args.root,
            output=args.output,
            max_scans=args.max_scans,
            compression_level=args.compression_level,
            kfold_k_values=kvals,
            kfold_test_pct=args.kfold_test_pct,
            kfold_seed=args.kfold_seed,
        )
    else:  # pragma: no cover — argparse choices guards this
        parser.error(f"unknown cohort {args.cohort!r}")
        return
    logger.info("Built %s", out)


if __name__ == "__main__":
    cli()
