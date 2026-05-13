"""End-to-end test of the E2.1 pipeline on a synthetic features H5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.E2.E2_1_recoverability.analysis.recoverability import (
    RecoverabilityRoutineConfig,
    run_recoverability,
)
from experiments.E2.E2_1_recoverability.analysis.regime import classify_regime


def _write_synthetic_features(
    path: Path, n: int = 240, c: int = 4, seed: int = 0, signal_scale: float = 1.0
) -> None:
    rng = np.random.default_rng(seed)
    y = rng.uniform(-1.0, 4.0, size=n).astype(np.float32)
    direction = np.ones(c, dtype=np.float32) / np.sqrt(c)
    z_tumor = (
        signal_scale * y[:, None] * direction[None] + 0.05 * rng.standard_normal((n, c))
    ).astype(np.float32)
    z_global = (0.1 * y[:, None] * direction[None] + 0.5 * rng.standard_normal((n, c))).astype(
        np.float32
    )
    z_random = rng.standard_normal((n, c)).astype(np.float32)
    patient_ids = np.array([f"P{i:04d}" for i in range(n)], dtype=object)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "0.1"
        f.attrs["modality"] = "t1c"
        f.attrs["n_scans"] = np.int64(n)
        f.attrs["latent_channels"] = np.int64(c)
        f.create_dataset("z_tumor", data=z_tumor)
        f.create_dataset("z_global", data=z_global)
        f.create_dataset("z_random", data=z_random)
        f.create_dataset("log_volume", data=y)
        f.create_dataset("mask_lat_present", data=np.ones(n, dtype=bool))
        vlen = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("patient_ids", shape=(n,), dtype=vlen)
        ds[...] = patient_ids
        f.create_dataset("scan_ids", shape=(n,), dtype=vlen)
        f["scan_ids"][...] = patient_ids
        f.create_dataset("timepoint_idx", data=np.zeros(n, dtype=np.int32))


def test_recoverability_full_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    feats = tmp_path / "features.h5"
    _write_synthetic_features(feats, n=240, signal_scale=1.0)
    cfg = RecoverabilityRoutineConfig(
        features_h5=feats,
        output_dir=tmp_path / "out",
        mode="full",
        bootstrap_n=20,
        n_splits=3,
        mlp_max_epochs=30,
        mlp_hidden=64,
        mlp_patience=5,
        mlp_batch_size=32,
        seed=0,
        device="cpu",
        log_level="WARNING",
    )
    result = run_recoverability(cfg)

    assert result.regime == "R1", f"expected R1 on clean linear data; got {result.regime}"
    assert result.r2_lin > 0.95
    assert result.r2_mlp > 0.85
    assert result.r2_lin_ci_lower > 0.5
    assert result.direction.shape == (4,)
    cos = float(np.dot(result.direction, np.ones(4) / np.sqrt(4)))
    assert abs(cos) > 0.99

    # Files written
    assert (cfg.output_dir / "result.json").is_file()
    assert (cfg.output_dir / "direction.npz").is_file()
    assert (cfg.output_dir / "scatter_pred_vs_obs.png").is_file()
    assert (cfg.output_dir / "pooling_comparison.png").is_file()


def test_recoverability_smoke_mode_runs(tmp_path: Path) -> None:
    feats = tmp_path / "features.h5"
    _write_synthetic_features(feats, n=80, seed=1)
    cfg = RecoverabilityRoutineConfig(
        features_h5=feats,
        output_dir=tmp_path / "out",
        mode="smoke",
        bootstrap_n=20,
        n_splits=3,
        mlp_max_epochs=10,
        mlp_hidden=32,
        mlp_patience=3,
        mlp_batch_size=8,
        seed=0,
        device="cpu",
        log_level="WARNING",
    )
    result = run_recoverability(cfg)
    assert result.n_scans <= 20
    assert (cfg.output_dir / "result.json").is_file()


@pytest.mark.parametrize(
    "r2_lin,r2_mlp,expected",
    [
        (0.7, 0.72, "R1"),
        (0.55, 0.58, "borderline_R1"),
        (0.55, 0.7, "R2"),  # gap > 0.10
        (0.4, 0.6, "R2"),  # MLP rescues
        (0.2, 0.3, "R3"),
    ],
)
def test_classify_regime_decision_table(r2_lin, r2_mlp, expected) -> None:
    assert classify_regime(r2_lin, r2_mlp) == expected
