"""Smoke tests for E2.3 spatial localization engine.

All tests use synthetic mode (synthetic=True) so no external data is required.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from experiments.E2_3_spatial_localization.analysis.spatial import (
    SpatialRoutineConfig,
    _classify_operator,
    run_spatial,
)


def _make_cfg(tmp_path: Path, **overrides) -> SpatialRoutineConfig:
    """Build a minimal synthetic SpatialRoutineConfig for testing."""
    defaults = dict(
        features_h5=Path("/tmp/_unused.h5"),
        source_h5=Path("/tmp/_unused.h5"),
        clinical_xlsx=Path("/tmp/_unused.xlsx"),
        output_dir=tmp_path / "out",
        n_boot=100,
        seed=0,
        log_level="WARNING",
        laterality_cache_csv=None,
        synthetic=True,
        synthetic_n=200,
        synthetic_c=4,
        synthetic_global_strength=0.125,
    )
    defaults.update(overrides)
    return SpatialRoutineConfig(**defaults)


class TestSmokeLocalized:
    """End-to-end synthetic run: planted localized signal must be recovered."""

    def test_smoke_synthetic_localized(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        result = run_spatial(cfg)

        # Spatial ordering: tumor >> global >> random
        assert result.r2_tumor > result.r2_global + 0.30, (
            f"r2_tumor={result.r2_tumor:.4f} not > r2_global+0.30={result.r2_global + 0.30:.4f}"
        )
        assert result.r2_tumor > result.r2_random + 0.30, (
            f"r2_tumor={result.r2_tumor:.4f} not > r2_random+0.30={result.r2_random + 0.30:.4f}"
        )
        assert result.r2_random < result.r2_global + 0.10, (
            f"r2_random={result.r2_random:.4f} not < r2_global+0.10={result.r2_global + 0.10:.4f}"
        )

        # Decision
        assert result.operator == "strongly_local", (
            f"Expected strongly_local, got {result.operator}"
        )
        assert result.impl_guard_pass is True, (
            "Implementation guard failed in synthetic localized scenario"
        )

        # Artifact existence
        out = cfg.output_dir
        assert (out / "result.json").exists(), "result.json missing"
        assert (out / "bootstrap_scores.npz").exists(), "bootstrap_scores.npz missing"
        assert (out / "r2_three_bars.png").exists(), "r2_three_bars.png missing"
        assert (out / "delta_r2_summary.png").exists(), "delta_r2_summary.png missing"

    def test_smoke_post_fwl_preserves_ordering(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        result = run_spatial(cfg)

        # Post-FWL ordering must survive confounder removal
        assert result.r2_tumor_post_fwl > result.r2_global_post_fwl + 0.20, (
            f"r2_tumor_post_fwl={result.r2_tumor_post_fwl:.4f} not > "
            f"r2_global_post_fwl+0.20={result.r2_global_post_fwl + 0.20:.4f}"
        )
        assert result.delta_r2_post_fwl >= 0.30, (
            f"delta_r2_post_fwl={result.delta_r2_post_fwl:.4f} < 0.30"
        )


@pytest.mark.parametrize(
    "delta_r2, expected_operator",
    [
        (0.01, "global"),
        (0.10, "local"),
        (0.20, "strongly_local"),
    ],
)
def test_decision_thresholds(delta_r2: float, expected_operator: str) -> None:
    """_classify_operator maps delta values to the correct operator string."""
    result = _classify_operator(delta_r2)
    assert result == expected_operator, (
        f"delta={delta_r2}: expected {expected_operator}, got {result}"
    )


def test_spatial_result_serializable(tmp_path: Path) -> None:
    """SpatialResult round-trips through json.dumps(asdict(...)) without error."""
    cfg = _make_cfg(tmp_path)
    result = run_spatial(cfg)

    d = dataclasses.asdict(result)

    def _default(o: object) -> object:
        import numpy as np

        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    serialized = json.dumps(d, default=_default)
    assert len(serialized) > 0
    # Round-trip parse
    parsed = json.loads(serialized)
    assert "operator" in parsed
    assert "r2_tumor" in parsed
    assert "impl_guard_pass" in parsed
