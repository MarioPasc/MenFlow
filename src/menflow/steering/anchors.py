"""Stratified anchor sampling for the causal-steering sweep.

E2.4 §4.1 calls for 50 anchors stratified by E1 volume bins (10 per B1-B5). At
local-3060 capacity we use 1 anchor per bin (5 total). The sampler is
deterministic given a seed, restricted to a candidate index set (the
``val`` split, with ``mask_lat_present`` filter already applied), and raises
when a bin is empty so silent omissions are impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StratifiedAnchor:
    """One sampled anchor.

    Attributes
    ----------
    index
        Row index into the latents H5 / source H5 (both share ordering).
    log_v
        Source-mask ``log V`` in cm³ for this anchor.
    bin_id
        1-based volume bin (B1..B5) the anchor belongs to.
    """

    index: int
    log_v: float
    bin_id: int


def stratified_anchor_indices(
    log_v: np.ndarray,
    candidate_indices: np.ndarray,
    bin_edges: np.ndarray,
    n_per_bin: int,
    seed: int = 0,
) -> list[StratifiedAnchor]:
    """Sample ``n_per_bin`` anchors from each ``log V`` bin among the candidates.

    Parameters
    ----------
    log_v
        Length-N array of ``log V`` per scan (full cohort).
    candidate_indices
        1-D int array of row indices that are eligible (e.g. val split intersected
        with ``mask_lat_present``).
    bin_edges
        Monotone-increasing 1-D array of bin edges. ``B`` bins requires
        ``B+1`` edges. Bin ``b`` spans ``[edges[b-1], edges[b])`` (right-open),
        with the last bin right-closed.
    n_per_bin
        Number of anchors to draw per bin. If a bin has fewer eligible scans
        than requested, all of them are taken (logged via the missing count in
        the returned list).
    seed
        RNG seed for reproducibility.

    Returns
    -------
    list[StratifiedAnchor]
        Anchors in (bin_id, then-stable-sample) order.

    Raises
    ------
    ValueError
        If ``bin_edges`` is not monotone, or any bin is empty in ``candidate_indices``.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"bin_edges must be 1-D with >= 2 entries; got shape {edges.shape}")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")
    cand = np.asarray(candidate_indices, dtype=np.int64)
    log_v = np.asarray(log_v, dtype=np.float64)
    if cand.ndim != 1:
        raise ValueError("candidate_indices must be 1-D")
    rng = np.random.default_rng(seed)
    n_bins = edges.size - 1

    out: list[StratifiedAnchor] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (log_v[cand] >= lo) & (log_v[cand] <= hi if b == n_bins - 1 else log_v[cand] < hi)
        bin_cands = cand[in_bin]
        if bin_cands.size == 0:
            raise ValueError(
                f"bin B{b + 1} ({lo:.3f} <= log V < {hi:.3f}) has no eligible candidates"
            )
        take = min(int(n_per_bin), bin_cands.size)
        chosen = rng.choice(bin_cands, size=take, replace=False)
        chosen = np.sort(chosen)  # stable, comparable across runs
        for idx in chosen:
            out.append(
                StratifiedAnchor(
                    index=int(idx),
                    log_v=float(log_v[int(idx)]),
                    bin_id=int(b + 1),
                )
            )
    return out
