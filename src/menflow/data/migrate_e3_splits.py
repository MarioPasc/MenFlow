"""Migrate existing source/latent H5 files to the canonical kfold splits layout.

Replaces the legacy flat ``splits/{train, val}`` (BraTS challenge cohort
partition, val cohort unannotated) and the intermediate
``splits/{e3_train, e3_val, e3_test}`` (single-holdout E3.1 layout) with the
self-describing ``splits/kfold/`` group emitted natively by current converters.

Inputs
------
* Source unified H5 (``brats_men.h5``) — provides per-scan
  ``metadata/subset`` and the patient-list / scan-list ordering.
* Features H5 (e.g. ``brats_men_features_t1c.h5``) — provides ``log_volume``
  per scan. Volume is segmentation-derived and modality-agnostic, so any
  per-modality features H5 of the same cohort yields identical splits.
* Latent H5 (``brats_men_maisi_latents.h5``) — patched in place with the same
  layouts (mirrors the source).

Idempotence
-----------
The migration recomputes the splits from scratch. If ``splits/kfold`` already
exists in a target file, it is overwritten unless ``--check`` is set, in
which case the script verifies the existing layout matches the freshly
computed one (sha256 of every per-fold index array) and exits non-zero on
mismatch.

References
----------
LaBella et al., 2023 (arXiv:2305.07642).
E3.1 §4.2 — patient-grouped, log-volume stratified splits.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from menflow.data.kfold_splits import (
    KFoldLayout,
    build_kfold_splits,
    write_kfold_to_h5,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================


def migrate_kfold_splits(
    *,
    source_h5: Path,
    features_h5: Path,
    latent_h5: Path | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    test_pct: float = 0.1,
    seed: int = 42,
    n_strata: int = 5,
    log_v_floor: float = -6.0,
    drop_legacy: bool = True,
    check_only: bool = False,
) -> dict[str, Any]:
    """Compute kfold splits and write them in place into source + latent H5s.

    Returns a manifest describing the splits, the per-fold log-volume
    distribution, and the sha256 hash of every emitted index array. The
    manifest is also written to ``<h5>.splits_provenance.json`` next to each
    target file.
    """
    source_h5 = Path(source_h5).expanduser().resolve()
    features_h5 = Path(features_h5).expanduser().resolve()
    latent_h5 = Path(latent_h5).expanduser().resolve() if latent_h5 is not None else None
    for p in (source_h5, features_h5):
        if not p.is_file():
            raise FileNotFoundError(p)
    if latent_h5 is not None and not latent_h5.is_file():
        raise FileNotFoundError(latent_h5)

    with h5py.File(source_h5, "r") as src:
        n_scans = int(src.attrs["n_scans"])
        scan_ids = _decode_array(src["scan_ids"][:])
        patient_ids = _decode_array(src["patient_ids"][:])
        patient_list = _decode_array(src["longitudinal/patient_list"][:])
        if "metadata/subset" in src:
            subset_per_scan = _decode_array(src["metadata/subset"][:])
        else:
            subset_per_scan = ["train"] * n_scans

    with h5py.File(features_h5, "r") as feat:
        feat_scan_ids = _decode_array(feat["scan_ids"][:])
        log_v = np.asarray(feat["log_volume"][:], dtype=np.float64)
    if list(feat_scan_ids) != list(scan_ids):
        feat_lookup = dict(zip(feat_scan_ids, log_v))
        if not all(s in feat_lookup for s in scan_ids):
            missing = [s for s in scan_ids if s not in feat_lookup][:5]
            raise ValueError(
                f"features H5 does not cover every source scan; first missing: {missing}"
            )
        log_v = np.asarray([feat_lookup[s] for s in scan_ids], dtype=np.float64)

    layouts = build_kfold_splits(
        patient_list=list(patient_list),
        scan_to_patient=list(zip(scan_ids, patient_ids)),
        scan_log_v={s: float(v) for s, v in zip(scan_ids, log_v)},
        scan_subset={s: str(t) for s, t in zip(scan_ids, subset_per_scan)},
        k_values=k_values,
        test_pct=test_pct,
        seed=seed,
        n_strata=n_strata,
        log_v_floor=log_v_floor,
    )
    sha = _layouts_sha256(layouts)

    if check_only:
        for path in (source_h5, latent_h5):
            if path is None:
                continue
            existing = _existing_layouts_sha(path)
            if existing != sha:
                raise RuntimeError(
                    f"existing kfold splits in {path} do not match the recomputed layouts"
                )
        return _build_manifest(
            layouts,
            sha,
            source_h5,
            latent_h5,
            features_h5,
            seed,
            n_strata,
            k_values,
            test_pct,
            status="checked_ok",
        )

    _apply_layouts(source_h5, layouts, drop_legacy=drop_legacy)
    if latent_h5 is not None:
        _apply_layouts(latent_h5, layouts, drop_legacy=drop_legacy)

    manifest = _build_manifest(
        layouts,
        sha,
        source_h5,
        latent_h5,
        features_h5,
        seed,
        n_strata,
        k_values,
        test_pct,
        status="written",
    )
    _write_provenance(source_h5, manifest)
    if latent_h5 is not None:
        _write_provenance(latent_h5, manifest)
    return manifest


# ============================================================================
# Backwards-compat alias (legacy callers)
# ============================================================================


def migrate_e3_splits(**kwargs):
    """Alias for :func:`migrate_kfold_splits` — preserved for old callers.

    Old callers passed ``test_fold`` / ``val_fold``; both are silently dropped
    because the new layout is k-fold, not single-holdout-with-named-fold.
    """
    for legacy_kw in ("test_fold", "val_fold", "n_folds", "force"):
        kwargs.pop(legacy_kw, None)
    return migrate_kfold_splits(**kwargs)


# ============================================================================
# CLI
# ============================================================================


def cli() -> None:
    """Console entry for ``menflow-migrate-e3-splits``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--features-h5", type=Path, required=True)
    parser.add_argument("--latent-h5", type=Path, default=None)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K-fold k values to pre-compute. Default: 1 3 5 10.",
    )
    parser.add_argument("--test-pct", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-strata", type=int, default=5)
    parser.add_argument("--log-v-floor", type=float, default=-6.0)
    parser.add_argument(
        "--keep-legacy", action="store_true", help="Do not delete legacy splits/{train, val, e3_*}."
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify existing splits without writing."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    manifest = migrate_kfold_splits(
        source_h5=args.source_h5,
        latent_h5=args.latent_h5,
        features_h5=args.features_h5,
        k_values=tuple(args.k_values),
        test_pct=args.test_pct,
        seed=args.seed,
        n_strata=args.n_strata,
        log_v_floor=args.log_v_floor,
        drop_legacy=not args.keep_legacy,
        check_only=args.check,
    )
    summary = {k: v for k, v in manifest.items() if k != "layouts"}
    print(json.dumps(summary, indent=2))


# ============================================================================
# Helpers
# ============================================================================


def _decode_array(arr: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr]


def _sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.astype(np.int32, copy=False).tobytes())
    return h.hexdigest()


def _layouts_sha256(layouts: dict[int, KFoldLayout]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, layout in layouts.items():
        out[f"k{k}"] = {
            "test": _sha256_of_array(layout.test),
            "folds": [
                {
                    "train": _sha256_of_array(f.train),
                    "val": _sha256_of_array(f.val),
                }
                for f in layout.folds
            ],
        }
    return out


def _existing_layouts_sha(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        with h5py.File(path, "r") as f:
            if "splits/kfold" not in f:
                return {}
            for key in f["splits/kfold"]:
                kg = f["splits/kfold"][key]
                rec: dict[str, Any] = {
                    "test": _sha256_of_array(np.asarray(kg["test"][:])),
                    "folds": [],
                }
                i = 0
                while f"fold_{i}" in kg:
                    fg = kg[f"fold_{i}"]
                    rec["folds"].append(
                        {
                            "train": _sha256_of_array(np.asarray(fg["train"][:])),
                            "val": _sha256_of_array(np.asarray(fg["val"][:])),
                        }
                    )
                    i += 1
                out[key] = rec
    except OSError:
        return {}
    return out


def _apply_layouts(path: Path, layouts: dict[int, KFoldLayout], *, drop_legacy: bool) -> None:
    logger.info("patching %s", path)
    with h5py.File(path, "r+") as f:
        if "splits" not in f:
            f.create_group("splits")
        splits_grp = f["splits"]
        if drop_legacy:
            for legacy in ("train", "val", "e3_train", "e3_val", "e3_test"):
                if legacy in splits_grp:
                    del splits_grp[legacy]
                    logger.info("  removed legacy splits/%s from %s", legacy, path.name)
        if "kfold" in splits_grp:
            del splits_grp["kfold"]
        kfold_grp = splits_grp.create_group("kfold")
        write_kfold_to_h5(kfold_grp, layouts)


def _build_manifest(
    layouts: dict[int, KFoldLayout],
    sha: dict[str, dict[str, Any]],
    source_h5: Path,
    latent_h5: Path | None,
    features_h5: Path,
    seed: int,
    n_strata: int,
    k_values: tuple[int, ...],
    test_pct: float,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "source_h5": str(source_h5),
        "latent_h5": str(latent_h5) if latent_h5 is not None else None,
        "features_h5": str(features_h5),
        "algorithm": "StratifiedKFold(patient-level, log_v-quantile) with shared test holdout",
        "seed": int(seed),
        "n_strata": int(n_strata),
        "k_values": list(k_values),
        "test_pct": float(test_pct),
        "split_sha256": sha,
        "log_v_distribution": {f"k{k}": l.log_v_distribution for k, l in layouts.items()},
        "layouts": {
            f"k{k}": {
                "test": l.test.tolist(),
                "folds": [{"train": f.train.tolist(), "val": f.val.tolist()} for f in l.folds],
            }
            for k, l in layouts.items()
        },
    }


def _write_provenance(h5_path: Path, manifest: dict[str, Any]) -> None:
    out = h5_path.with_suffix(h5_path.suffix + ".splits_provenance.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("provenance written: %s", out)


if __name__ == "__main__":
    cli()
