"""Gate evaluator: turn aggregate metrics into a regime decision.

Two complementary diagnostics now feed the regime call:

1. ρ_lin (mask-pooled) — the headline metric. Computed on the 16-D mask-pooled
   feature vector, which is exactly what the E2.1 ridge probe consumes. The
   spec's thresholds (R1 ≤ 0.30, R3 > 0.50) were calibrated against a
   "feature-scale" residual, so this version is the closest match to the
   spec's intent. The original full-latent ρ_lin is reported alongside but
   used only for context.
2. PCA subspace alignment — variance captured by PC1 of `{z³ − z¹}` and the
   mean cosine of `(z² − z¹)` against PC1. The roadmap's Mechanism 1 (anchor
   propagation) depends on this *population* direction more than on per-patient
   linearity, so this metric tightens or loosens the regime independently.

When both gates pass (low ρ_lin + high PC1 variance + high cosine) → R1.
When neither passes → R3. Mixed signals → R2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from experiments.E2.E2_5_ae_longitudinal.config import AELongitudinalConfig

Regime = Literal["R1", "R2", "R3", "INDETERMINATE"]


@dataclass(frozen=True, slots=True)
class Decision:
    regime: Regime
    downstream_action: str
    rationale: str
    metrics: dict[str, float]
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "downstream_action": self.downstream_action,
            "rationale": self.rationale,
            "metrics": self.metrics,
            "caveats": list(self.caveats),
        }


def _act(regime: str) -> str:
    return {
        "R1": "proceed_e3_2_standard",
        "R2": "proceed_e3_2_with_option_gamma",
        "R3": "trigger_ae_finetune_delta_lfm",
        "INDETERMINATE": "rerun_with_segmenter",
    }[regime]


def _interval_excludes_zero(ci_lo: float, ci_hi: float) -> bool:
    import math

    if math.isnan(ci_lo) or math.isnan(ci_hi):
        return False
    return (ci_lo > 0.0 and ci_hi > 0.0) or (ci_lo < 0.0 and ci_hi < 0.0)


def evaluate(
    aggregate: dict,
    config: AELongitudinalConfig,
    *,
    caveats_extra: list[str] | None = None,
) -> Decision:
    """Apply the regime gates to the new (pooled) headline metric + PCA test.

    Decision tree:

    * **R1** requires (a) mean mask-pooled ρ_lin ≤ ``rho_lin_pass``,
      (b) Δρ_pooled CI excludes zero, (c) PCA PC1 variance ≥ 0.70,
      (d) mean cosine of (z²−z¹) with PC1 ≥ 0.70, (e) non-tumour SSIM ≥
      pass threshold (if decode ran).
    * **R3** if ρ_lin > intermediate OR Δρ CI includes zero AND PCA PC1
      variance < 0.40 (no shared longitudinal direction).
    * **R2** otherwise: locally smooth but globally curved; add Option-γ.
    """
    caveats: list[str] = list(caveats_extra or [])
    metrics: dict[str, float] = {}

    # Headline ρ_lin: mask-pooled if available, else fall back to mask-localised,
    # else the full-latent original. Record which variant was used.
    pooled = aggregate.get("rho_lin_pooled") or {}
    masked = aggregate.get("rho_lin_masked") or {}
    joint = aggregate.get("rho_lin") or {}
    delta_pooled = aggregate.get("delta_rho_pooled") or {}
    delta_masked = aggregate.get("delta_rho_masked") or {}
    delta_joint = aggregate.get("delta_rho") or {}

    rho_pooled = float(pooled.get("mean", float("nan")))
    rho_masked = float(masked.get("mean", float("nan")))
    rho_joint = float(joint.get("mean", float("nan")))
    metrics["rho_lin_pooled"] = rho_pooled
    metrics["rho_lin_masked"] = rho_masked
    metrics["rho_lin_joint"] = rho_joint

    headline = "pooled"
    rho_headline = rho_pooled
    delta_headline = delta_pooled
    if not (rho_headline == rho_headline):  # NaN check
        headline = "masked"
        rho_headline = rho_masked
        delta_headline = delta_masked
    if not (rho_headline == rho_headline):
        headline = "joint"
        rho_headline = rho_joint
        delta_headline = delta_joint
    metrics["rho_lin_headline_variant"] = headline  # type: ignore[assignment]
    metrics["rho_lin_headline"] = rho_headline

    delta_lo = float(delta_headline.get("ci_lo", float("nan")))
    delta_hi = float(delta_headline.get("ci_hi", float("nan")))
    delta_mean = float(delta_headline.get("mean", float("nan")))
    metrics["delta_rho_headline_mean"] = delta_mean
    metrics["delta_rho_headline_ci_lo"] = delta_lo
    metrics["delta_rho_headline_ci_hi"] = delta_hi

    # SSIM (may be NaN if decode skipped).
    ssim_nt = float((aggregate.get("ssim_nontumour_mean") or {}).get("mean", float("nan")))
    metrics["ssim_nontumour_mean"] = ssim_nt

    # PCA.
    pca = aggregate.get("pca") or {}
    pca_var = float(pca.get("variance_explained_pc1", float("nan")))
    pca_cos_mean = float(pca.get("cos_z2_minus_z1_mean", float("nan")))
    pca_cos13_mean = float(pca.get("cos_z3_minus_z1_mean", float("nan")))
    metrics["pca_variance_explained_pc1"] = pca_var
    metrics["pca_cos_z2_minus_z1_mean"] = pca_cos_mean
    metrics["pca_cos_z3_minus_z1_mean"] = pca_cos13_mean

    n_eff = int(aggregate.get("n_effective", 0))
    metrics["n_effective"] = float(n_eff)

    if config.use_segmenter_dice:
        dice = (aggregate.get("dice_at_beta_vol") or {}).get("mean")
        if dice is None:
            return Decision(
                regime="INDETERMINATE",
                downstream_action=_act("INDETERMINATE"),
                rationale="use_segmenter_dice=True but no Dice value supplied.",
                metrics=metrics,
                caveats=caveats + ["dice_required_but_missing"],
            )
    else:
        caveats.append("dice_at_beta_vol_deferred")

    if n_eff < config.min_effective_cohort:
        caveats.append("n_effective_below_min")

    delta_excludes_zero = _interval_excludes_zero(delta_lo, delta_hi)

    pca_strong = (pca_var >= 0.70) and (pca_cos_mean >= 0.70)
    pca_weak = (pca_var < 0.40)

    rho_in_pass = rho_headline <= config.rho_lin_pass
    rho_in_intermediate = config.rho_lin_pass < rho_headline <= config.rho_lin_intermediate
    rho_fail = rho_headline > config.rho_lin_intermediate
    nt_pass = (ssim_nt >= config.ssim_nontumour_pass) if ssim_nt == ssim_nt else True
    nt_intermediate = (ssim_nt >= config.ssim_nontumour_intermediate) if ssim_nt == ssim_nt else True

    if (rho_fail or not delta_excludes_zero) and pca_weak:
        regime: Regime = "R3"
        rationale = (
            f"headline ρ_lin ({headline}) = {rho_headline:.3f} > {config.rho_lin_intermediate}"
            if rho_fail
            else f"Δρ_{headline} CI [{delta_lo:.3f}, {delta_hi:.3f}] includes zero "
            f"and PCA PC1 variance {pca_var:.2f} < 0.40 — no shared longitudinal direction."
        )
    elif rho_in_pass and delta_excludes_zero and pca_strong and nt_pass:
        regime = "R1"
        rationale = (
            f"ρ_lin_{headline} = {rho_headline:.3f} ≤ {config.rho_lin_pass}; "
            f"Δρ CI [{delta_lo:.3f}, {delta_hi:.3f}] excludes zero; "
            f"PCA PC1 variance = {pca_var:.2f}, mean cos(Δz¹², PC1) = {pca_cos_mean:.2f}."
        )
    elif (rho_in_pass or rho_in_intermediate) and delta_excludes_zero and not pca_weak and nt_intermediate:
        regime = "R2"
        rationale = (
            f"ρ_lin_{headline} = {rho_headline:.3f} in pass-or-intermediate band; "
            f"Δρ excludes zero; PCA variance {pca_var:.2f} (mean cos {pca_cos_mean:.2f}) — "
            f"locally smooth but globally curved. Add Option-γ anchor-similarity loss."
        )
    else:
        regime = "R3"
        rationale = (
            f"ρ_lin_{headline} = {rho_headline:.3f}, Δρ CI [{delta_lo:.3f}, {delta_hi:.3f}], "
            f"PCA variance {pca_var:.2f}, cos {pca_cos_mean:.2f}, non-tumour SSIM {ssim_nt:.3f} — "
            "fails R1 and R2 gates."
        )

    return Decision(
        regime=regime,
        downstream_action=_act(regime),
        rationale=rationale,
        metrics=metrics,
        caveats=caveats,
    )
