"""CLI entry for E2.4 — analysis + plotting on top of an existing sweep_results.h5.

Three modes, picked from the YAML config:

* ``mode: full`` — invoke the steer-decode routine (re-using the same YAML),
  then run analysis. Useful when re-driving the whole experiment.
* ``mode: phase_b`` — assume Phase A's ``sweep_results.h5`` already exists;
  run the Phase B Docker-backed segmenter to fill ``decoded_log_v``, then
  the standard analysis + headline-grid figures.
* ``mode: analyze_only`` (default) — re-run analysis + plots only.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.E2_4_causal_steering.analysis.manual_scoring import build_montage
from experiments.E2_4_causal_steering.analysis.plotting import (
    make_all,
)
from experiments.E2_4_causal_steering.analysis.steering_eval import (
    evaluate_sweep,
    write_result_json,
)
from routines.steer_decode.engine.steer_decode_engine import (
    SteerDecodeEngine,
    SteerDecodeRoutineConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class E24ExperimentConfig:
    """Thin wrapper around the routine config + analysis options."""

    routine: SteerDecodeRoutineConfig
    mode: str = "analyze_only"
    write_phase_a_report: bool = True
    log_level: str = "INFO"
    # Phase-B-only options (ignored in other modes).
    phase_b_model_id: str = "BraTS25_1"
    phase_b_gpu: bool = True
    phase_b_timeout_s: float = 1800.0
    phase_b_direction_sign: int = 1

    @classmethod
    def from_yaml(cls, path: Path | str) -> E24ExperimentConfig:
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        mode = raw.pop("mode", "analyze_only")
        report = raw.pop("write_phase_a_report", True)
        phase_b_model_id = raw.pop("phase_b_model_id", "BraTS25_1")
        phase_b_gpu = raw.pop("phase_b_gpu", True)
        phase_b_timeout_s = raw.pop("phase_b_timeout_s", 1800.0)
        phase_b_direction_sign = raw.pop("phase_b_direction_sign", 1)
        # Drop top-level keys that don't belong to the routine.
        raw.pop("slurm", None)
        return cls(
            routine=SteerDecodeRoutineConfig.from_yaml(path),
            mode=mode,
            write_phase_a_report=report,
            log_level=raw.get("log_level", "INFO"),
            phase_b_model_id=phase_b_model_id,
            phase_b_gpu=phase_b_gpu,
            phase_b_timeout_s=phase_b_timeout_s,
            phase_b_direction_sign=phase_b_direction_sign,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2.4 — causal steering: decode + drift + montage + slope eval."
    )
    parser.add_argument("config", type=Path, help="Experiment YAML config.")
    args = parser.parse_args()

    cfg = E24ExperimentConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    output_dir = cfg.routine.output_dir
    sweep_h5 = output_dir / "sweep_results.h5"
    decoded_root = output_dir / "decoded"

    if cfg.mode == "full" or (cfg.mode != "phase_b" and not sweep_h5.is_file()):
        logger.info("Running steer-decode routine (mode=%s)", cfg.mode)
        SteerDecodeEngine(cfg.routine).run()

    if not sweep_h5.is_file():
        raise FileNotFoundError(f"sweep_results.h5 not produced at {sweep_h5}")

    if cfg.mode == "phase_b":
        logger.info("Running Phase B segmentation (model=%s)", cfg.phase_b_model_id)
        run_phase_b(
            PhaseBConfig(
                sweep_h5=sweep_h5,
                decoded_root=decoded_root,
                latents_h5=cfg.routine.latents_h5,
                checkpoint=cfg.routine.checkpoint,
                model_id=cfg.phase_b_model_id,
                gpu=cfg.phase_b_gpu,
                timeout_s=cfg.phase_b_timeout_s,
                dtype=cfg.routine.dtype,
                device=cfg.routine.device,
                direction_sign=cfg.phase_b_direction_sign,
            )
        )

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_sweep(sweep_h5)
    write_result_json(result, analysis_dir / "result.json")
    logger.info("evaluate_sweep result: %s", dataclasses.asdict(result))

    if result.segmenter_completed:
        artefacts = make_all_with_grids(
            sweep_h5,
            cfg.routine.source_h5,
            analysis_dir,
            decoded_root=decoded_root,
        )
        _write_phase_b_report(analysis_dir / "PHASE_B_REPORT.md", sweep_h5, result, cfg)
    else:
        artefacts = make_all(sweep_h5, analysis_dir)
    logger.info("Plot artefacts: %s", {k: str(v) for k, v in artefacts.items()})

    montage_path = build_montage(
        sweep_h5,
        analysis_dir / "anatomical_montage.png",
        source_h5_path=cfg.routine.source_h5,
    )
    logger.info("Montage written -> %s", montage_path)

    if cfg.write_phase_a_report and not result.segmenter_completed:
        _write_phase_a_report(analysis_dir / "PHASE_A_REPORT.md", sweep_h5, result)
        logger.info("PHASE_A_REPORT.md written")


def _write_phase_a_report(path: Path, sweep_h5: Path, result) -> None:
    """Emit a short markdown report describing what's done and what's pending."""
    body = f"""# E2.4 — Phase A report

This run produced decoded NIfTIs + a provisional `sweep_results.h5` whose
volume metrics (`decoded_log_v`, `tumor_voxels_decoded`) are sentinel-filled
because no segmenter was available. The figures in this directory describe
only the decoder-side observables (drift, intensity proxy).

**What is finalised**

- {result.n_anchors} anchors × {result.n_deltas} deltas decoded.
- Off-manifold drift: max = {result.drift_max:.4f}, mean = {result.drift_mean:.4f}.
- Anatomical montage at ±max delta + zero step.
- Provisional sweep H5 at `{sweep_h5}`.

**What Phase B must do**

1. Implement a concrete `Segmenter` (subclass / Protocol impl) under
   `experiments/E2_4_causal_steering/analysis/segmenter_interface.py` (or
   adjacent module). Required attributes: `modality`, `expected_shape`,
   `predict(image: np.ndarray) -> np.ndarray`.
2. Call `fill_sweep_results(sweep_h5_path, segmenter, decoded_root)` once. This
   walks every NIfTI listed in the H5, runs the segmenter, and writes
   `decoded_log_v` + `tumor_voxels_decoded` back into the file.
3. Re-run `menflow-e2-4 <config.yaml>` (analysis-only mode is the default —
   it skips the decode + uses the now-filled H5).

**Decision gate (E2.4 §5)**

After Phase B, the headline metrics are:

- slope ∈ [0.7, 1.3]
- R² ≥ 0.6
- per-anchor Spearman ρ ≥ 0.9 for ≥ 80 % of anchors
- drift max < 0.15

Any failure → Path D (project reframe). All-pass → proceed to E2.5 / E2.6.
"""
    with open(path, "w") as fh:
        fh.write(body)


def _write_phase_b_report(path: Path, sweep_h5: Path, result, cfg: E24ExperimentConfig) -> None:
    """Emit a markdown report with the slope, R², decision, and figures."""
    body = f"""# E2.4 — Phase B report

Phase B segmentation completed. Headline metrics (E2.4 §4.4 / §5):

- **decision**: `{result.decision}`
- slope = {result.slope:.4f}, 95 % CI [{result.slope_ci_lo:.4f}, {result.slope_ci_hi:.4f}]
- R² = {result.r2:.4f}
- per-anchor monotonicity (Spearman ρ ≥ 0.9): {result.pct_monotone:.1%}
- off-manifold drift: max = {result.drift_max:.4f}, mean = {result.drift_mean:.4f}
- saturation |Δ log V|: {result.saturation_log_v}

**Configuration**

- segmenter: `{cfg.phase_b_model_id}`
- companion-modality strategy: decoded at Δ=0 (purely MAISI inputs)
- direction_sign: {cfg.phase_b_direction_sign}

**Artefacts**

- sweep H5 (filled): `{sweep_h5}`
- result JSON: `{sweep_h5.parent / "analysis" / "result.json"}`
- headline grids: `{sweep_h5.parent / "analysis" / "headline_grid_*.png"}`
- scatter pred-vs-decoded, drift histogram, intensity proxy: in the same dir.

**Per §5 thresholds**

- slope ∈ [0.7, 1.3]: {0.7 <= result.slope <= 1.3}
- R² ≥ 0.6: {result.r2 >= 0.6}
- monotonicity ≥ 80 %: {result.pct_monotone >= 0.8}
- drift max < 0.15: {result.drift_max < 0.15}

If decision is `FAIL_SLOPE` and the slope is near −1, re-run with
`phase_b_direction_sign: -1` (this re-segments only; no re-decode needed).
"""
    with open(path, "w") as fh:
        fh.write(body)


if __name__ == "__main__":
    main()
