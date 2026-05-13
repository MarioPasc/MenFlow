"""Cohort construction for E2.5 — longitudinal triple selection.

Each retained patient yields one triple ``(t1, t2, t3)`` of timepoint indices
into the source HDF5, satisfying §5.1 criteria C1–C4:

* C1: ≥3 timepoints,
* C2: successful segmentation at every timepoint of the triple
  (here, ``has_segmentation==True`` and ``log_volume_cm3`` finite),
* C3: monotone volume order ``V₁<V₂<V₃`` — we pick the **longest strictly
  monotone increasing subsequence** by volume via Patience-sort, then choose
  the triple ``(first, middle, last)`` from that subsequence where ``middle``
  maximises end-to-end spread.
* C4: ``log V₃ − log V₁ ≥ min_spread``. If not enough triples survive at the
  default 0.3 spread, the C4 threshold is relaxed in steps from the config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PatientTriplet:
    """One ``(z¹, z², z³)`` candidate, identified only by row indices into the H5s."""

    patient_id: str
    scan_indices: tuple[int, int, int]  # rows into /images, /segmentations, /latents
    timepoint_idx: tuple[int, int, int]  # values from /timepoint_idx
    log_volumes: tuple[float, float, float]

    @property
    def log_volume_spread(self) -> float:
        return float(self.log_volumes[2] - self.log_volumes[0])


def _longest_monotone_indices(values: np.ndarray) -> list[int]:
    """Return indices of the longest strictly increasing subsequence of ``values``.

    Classic Patience-sort with parent pointers. Time: O(n log n).
    Ties (``values[i] == values[j]``) are excluded — the requirement is
    *strict* monotonicity for E2.5's volume order.
    """
    n = len(values)
    if n == 0:
        return []
    tails: list[int] = []  # index of smallest tail value for each LIS length
    parents = [-1] * n
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        # find leftmost tail with value >= v (strict monotonicity)
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[tails[mid]] >= v:
                hi = mid
            else:
                lo = mid + 1
        if lo > 0:
            parents[i] = tails[lo - 1]
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i
    # reconstruct
    if not tails:
        return []
    k = tails[-1]
    out: list[int] = []
    while k != -1:
        out.append(k)
        k = parents[k]
    out.reverse()
    return out


def _pick_triple_from_lis(lis_indices: list[int], log_vol: np.ndarray) -> tuple[int, int, int] | None:
    """Pick the triple maximising end-to-end spread, with a fixed middle.

    With an LIS of length ≥ 3, take ``first`` = LIS[0], ``last`` = LIS[-1],
    and ``middle`` = the interior LIS index whose volume is closest to the
    midpoint between V_first and V_last (so β_time and β_vol are comparable).
    """
    if len(lis_indices) < 3:
        return None
    first, last = lis_indices[0], lis_indices[-1]
    interior = lis_indices[1:-1]
    target = 0.5 * (log_vol[first] + log_vol[last])
    middle = min(interior, key=lambda j: abs(float(log_vol[j]) - target))
    return first, middle, last


def _all_monotone_triples(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Enumerate every strictly volume-monotone (i < j < k, V_i < V_j < V_k) triple.

    O(n³) but n ≤ 6 in MenGrowth, so cost is negligible. Indices are returned
    in *time* order (i < j < k); strict monotonicity in volume is also
    enforced. NaN volumes are skipped implicitly because the inequality
    ``V_i < V_j`` is False for NaN.
    """
    n = len(values)
    out: list[tuple[int, int, int]] = []
    finite = np.isfinite(values)
    for i in range(n):
        if not finite[i]:
            continue
        vi = values[i]
        for j in range(i + 1, n):
            if not finite[j] or values[j] <= vi:
                continue
            vj = values[j]
            for k in range(j + 1, n):
                if not finite[k] or values[k] <= vj:
                    continue
                out.append((i, j, k))
    return out


def select_cohort(
    *,
    unified_h5: Path,
    latents_h5: Path,
    min_timepoints: int,
    min_spread_ladder: tuple[float, ...],
    min_effective_cohort: int,
    max_patients: int | None = None,
    enumerate_all_triples: bool = True,
) -> tuple[list[PatientTriplet], dict]:
    """Build the cohort with automatic C4 relaxation.

    Returns the list of patient triples and a construction log dict suitable
    for embedding in aggregate.json.
    """
    with h5py.File(unified_h5, "r") as f, h5py.File(latents_h5, "r") as lat:
        # Align scan order: latents H5 must be produced from the same source.
        u_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
        l_ids = [s.decode() if isinstance(s, bytes) else s for s in lat["scan_ids"][:]]
        if u_ids != l_ids:
            # Try to recover via lookup; many encode runs preserve order, so this
            # is mainly a guard against stale latents.
            raise RuntimeError(
                f"scan_ids mismatch between unified H5 ({unified_h5}) and "
                f"latents H5 ({latents_h5}). The latents must come from the same "
                "encode pass on the unified file."
            )

        patient_offsets = f["longitudinal/patient_offsets"][:].astype(int)
        patient_list = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["longitudinal/patient_list"][:]
        ]
        log_vol = f["features/log_volume_cm3"][:].astype(np.float64)
        has_seg = f["has_segmentation"][:].astype(bool)
        timepoint_idx = f["timepoint_idx"][:].astype(int)

    # All patients with >= min_timepoints, sorted by patient index.
    candidates: list[tuple[str, list[int]]] = []
    for pi, pid in enumerate(patient_list):
        rows = list(range(int(patient_offsets[pi]), int(patient_offsets[pi + 1])))
        # Drop rows without segmentation or with non-finite log_volume.
        rows = [r for r in rows if has_seg[r] and np.isfinite(log_vol[r])]
        if len(rows) >= min_timepoints:
            candidates.append((pid, rows))

    construction_log: dict = {
        "n_patients_total": len(patient_list),
        "n_patients_passing_C1_C2": len(candidates),
        "min_timepoints": min_timepoints,
        "min_spread_ladder": list(min_spread_ladder),
        "min_effective_cohort": min_effective_cohort,
        "attempts": [],
    }

    # Two enumeration modes:
    # - ``enumerate_all_triples=True``: every strict-monotone (i,j,k) triple
    #   per patient (in time order). Boosts n at the cost of within-patient
    #   correlation, which we account for via patient-cluster bootstrap.
    # - ``False``: one canonical triple per patient (longest LIS, middle by
    #   log-volume midpoint). Matches the original spec.
    base_triples: list[PatientTriplet] = []
    n_patients_with_triple = 0
    for pid, rows in candidates:
        rows_arr = np.asarray(rows, dtype=int)
        vals = log_vol[rows_arr]
        if enumerate_all_triples:
            triples_local = _all_monotone_triples(vals)
        else:
            lis_in_local = _longest_monotone_indices(vals)
            t = _pick_triple_from_lis(lis_in_local, vals)
            triples_local = [t] if t is not None else []
        if triples_local:
            n_patients_with_triple += 1
        for triple_local in triples_local:
            gi = tuple(int(rows_arr[i]) for i in triple_local)
            ti = tuple(int(timepoint_idx[g]) for g in gi)
            lv = tuple(float(log_vol[g]) for g in gi)
            base_triples.append(
                PatientTriplet(
                    patient_id=pid,
                    scan_indices=gi,
                    timepoint_idx=ti,
                    log_volumes=lv,
                )
            )

    construction_log["n_patients_with_monotone_triple"] = n_patients_with_triple
    construction_log["n_triples_generated"] = len(base_triples)
    construction_log["enumerate_all_triples"] = bool(enumerate_all_triples)

    # Apply C4 ladder. With multi-triples per patient the ladder decision is
    # gated on **unique-patient count**, not triple count — otherwise loosening
    # C4 trivially passes because each patient can contribute many triples.
    selected: list[PatientTriplet] = []
    final_spread = min_spread_ladder[-1] if min_spread_ladder else 0.0
    for spread in min_spread_ladder:
        keep = [t for t in base_triples if t.log_volume_spread >= spread]
        n_patients_keep = len({t.patient_id for t in keep})
        construction_log["attempts"].append(
            {
                "min_log_volume_spread": float(spread),
                "n_triples": len(keep),
                "n_patients": n_patients_keep,
            }
        )
        if n_patients_keep >= max(min_effective_cohort, 15):
            selected = keep
            final_spread = spread
            break
    else:
        # No threshold satisfied; use the loosest (last) one regardless.
        selected = [t for t in base_triples if t.log_volume_spread >= final_spread]
        construction_log["attempts"][-1]["used"] = True

    construction_log["min_log_volume_spread_used"] = float(final_spread)
    construction_log["n_effective"] = len({t.patient_id for t in selected})
    construction_log["n_triples_selected"] = len(selected)
    construction_log["below_min_effective_cohort"] = (
        construction_log["n_effective"] < min_effective_cohort
    )

    if max_patients is not None:
        # Cap by unique patients (not by triples) so smoke ↔ full behave the same.
        keep_pids: list[str] = []
        capped: list[PatientTriplet] = []
        for t in selected:
            if t.patient_id not in keep_pids:
                if len(keep_pids) >= max_patients:
                    continue
                keep_pids.append(t.patient_id)
            capped.append(t)
        if len(capped) < len(selected):
            logger.info(
                "Truncating cohort to max_patients=%d unique patients (was %d triples; now %d)",
                max_patients,
                len(selected),
                len(capped),
            )
            selected = capped
            construction_log["truncated_to_max_patients"] = max_patients

    construction_log["n_triples_used"] = len(selected)
    construction_log["n_unique_patients_used"] = len({t.patient_id for t in selected})

    logger.info(
        "Cohort selection complete: n_triples=%d across %d patients (C4=%.2f, min target=%d, %s)",
        len(selected),
        construction_log["n_unique_patients_used"],
        final_spread,
        min_effective_cohort,
        "OK" if not construction_log["below_min_effective_cohort"] else "SUGGESTIVE",
    )
    return selected, construction_log
