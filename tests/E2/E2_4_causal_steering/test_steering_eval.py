"""Unit tests for steering_eval — Phase A placeholder + perfect-linear PASS."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.E2.E2_4_causal_steering.analysis.steering_eval import (
    evaluate_sweep,
    write_result_json,
)


def _write_sweep(
    path: Path,
    *,
    predicted: np.ndarray,
    decoded: np.ndarray,
    drift: np.ndarray,
    completed: bool,
) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["segmenter_completed"] = completed
        f.create_dataset("predicted_log_v", data=predicted.astype(np.float32))
        f.create_dataset("decoded_log_v", data=decoded.astype(np.float32))
        f.create_dataset("drift", data=drift.astype(np.float32))


def test_phase_a_placeholder(tmp_path: Path):
    pred = np.tile(np.array([-1, 0, 1], dtype=np.float32), (4, 1))
    dec = np.full_like(pred, np.nan)
    drift = np.full_like(pred, 0.05)
    sweep = tmp_path / "sw.h5"
    _write_sweep(sweep, predicted=pred, decoded=dec, drift=drift, completed=False)
    res = evaluate_sweep(sweep)
    assert res.segmenter_completed is False
    assert res.decision == "PHASE_A_ONLY"
    assert np.isnan(res.slope)
    assert res.drift_max == pytest.approx(0.05)


def test_perfect_linear_pass(tmp_path: Path):
    # 6 anchors, 7 deltas; decoded == predicted exactly => slope=1, R²=1.
    deltas = np.linspace(-1.5, 1.5, 7)
    log_v0 = np.linspace(-1.0, 3.0, 6)
    pred = log_v0[:, None] + deltas[None, :]
    dec = pred.copy()
    drift = np.full(pred.shape, 0.05, dtype=np.float32)
    sweep = tmp_path / "sw.h5"
    _write_sweep(sweep, predicted=pred, decoded=dec, drift=drift, completed=True)
    res = evaluate_sweep(sweep, n_boot=50, seed=0)
    assert res.decision == "PASS"
    assert res.slope == pytest.approx(1.0, abs=1e-4)
    assert res.r2 == pytest.approx(1.0, abs=1e-6)
    assert res.pct_monotone == pytest.approx(1.0)


def test_failing_slope(tmp_path: Path):
    deltas = np.linspace(-1.5, 1.5, 7)
    log_v0 = np.linspace(-1.0, 3.0, 5)
    pred = log_v0[:, None] + deltas[None, :]
    dec = 0.3 * pred  # slope ~ 0.3 — outside [0.7, 1.3]
    drift = np.full(pred.shape, 0.02)
    sweep = tmp_path / "sw.h5"
    _write_sweep(sweep, predicted=pred, decoded=dec, drift=drift, completed=True)
    res = evaluate_sweep(sweep, n_boot=50, seed=0)
    assert res.decision == "FAIL_SLOPE"


def test_write_result_json_handles_nan(tmp_path: Path):
    pred = np.tile(np.array([-1, 0, 1], dtype=np.float32), (3, 1))
    dec = np.full_like(pred, np.nan)
    drift = np.full_like(pred, 0.1)
    sweep = tmp_path / "sw.h5"
    _write_sweep(sweep, predicted=pred, decoded=dec, drift=drift, completed=False)
    res = evaluate_sweep(sweep)
    out = tmp_path / "result.json"
    write_result_json(res, out)
    assert out.is_file()
    text = out.read_text()
    # Must be valid JSON, NaN encoded as null.
    import json

    payload = json.loads(text)
    assert payload["decision"] == "PHASE_A_ONLY"
    assert payload["slope"] is None
