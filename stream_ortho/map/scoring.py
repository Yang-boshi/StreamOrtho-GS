"""Budget scoring and eviction policies (design doc §7.6 item 1/2).

All strategies take the committed map arrays and return a boolean keep-mask of
size ``N_max``.  They are deliberately simple and comparable:

==========  ==================================================================
strategy    score / rule
==========  ==================================================================
``random``  uniform random subset
``conf``    ``conf * opacity * min(n_obs, 8)``
``voxel``   one point per coarse cell first (highest conf), then by conf
``ortho``   ``conf * opacity * min(n_obs,8) * coverage_gap / footprint`` where the
            coverage gap comes from a low-res occupancy grid of the map
``edge``    **(user's idea)** keep 100% of the edge-tagged points (by conf), then
            fill the rest with area points by conf -- the kept fraction of the
            area points is whatever fits the budget
==========  ==================================================================
"""

from __future__ import annotations

import numpy as np


def _basic_score(arr: dict, jitter: float = 1e-6) -> np.ndarray:
    """conf * opacity * min(n_obs,8), plus a tiny deterministic jitter.

    Without the jitter, ties (uniform plane-splat confidences) are broken by array
    order, which collapses the selection onto the first keyframes and leaves the
    evaluation tile empty -- observed in the first week-5 run.
    """
    base = (arr["conf"].astype(np.float64)
            * arr["opacity"].astype(np.float64)
            * np.minimum(arr["n_obs"].astype(np.float64), 8.0))
    m = arr["means"].astype(np.int64)          # quantise to 1 m before hashing
    h = (m[:, 0] * 73856093) ^ (m[:, 1] * 19349663) ^ (m[:, 2] * 83492791)
    j = np.abs(np.sin(h.astype(np.float64) * 1e-3)) * jitter
    return base + j


def coverage_gap(arr: dict, grid: int = 96) -> np.ndarray:
    """Per-point 1/(1+occupancy) of its 2-D cell -> a cheap "unique coverage" term."""
    m = arr["means"]
    lo = m[:, :2].min(axis=0)
    hi = m[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    ij = np.floor((m[:, :2] - lo) / span * (grid - 1)).astype(int)
    ij = np.clip(ij, 0, grid - 1)
    cell = ij[:, 0] * grid + ij[:, 1]
    counts = np.bincount(cell, minlength=grid * grid).astype(np.float64)
    return 1.0 / (1.0 + counts[cell])


def footprint(arr: dict) -> np.ndarray:
    s = arr["scales"].astype(np.float64)
    return np.maximum(s[:, 0] * s[:, 1], 1e-12)


def ortho_score(arr: dict) -> np.ndarray:
    return _basic_score(arr) * coverage_gap(arr) / footprint(arr)


def cfg_cell(arr: dict, factor: float = 3.0, floor: float = 1.0) -> float:
    """Coverage cell size: a few times the median point spacing, at least 1 m."""
    if len(arr["means"]) < 32:
        return floor
    from stream_ortho.map.gaussian import voxel_keys

    for cell in np.geomspace(0.1, 20.0, 40):
        k = voxel_keys(arr["means"], cell)
        uniq = len(np.unique(k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791))
        occ = uniq / len(k)                      # 1.0 = every point in its own cell
        if occ < 0.55:
            return float(max(cell * factor, floor))
    return floor


def choose_quota_cell(arr: dict, n_max: int, target_fill: float = 0.8) -> float:
    """Cell size such that the budget can give **every** occupied cell >= 1 point.

    If the cell is finer than ``n_max / n_cells`` the quota drops below 1 and the
    selection starts leaving cells empty -- that was the 17-30% no-data in the
    first S7a runs.
    """
    from stream_ortho.map.gaussian import voxel_keys

    n = len(arr["means"])
    if n <= n_max:
        return 1.0
    for cell in np.geomspace(0.2, 60.0, 60):
        k = voxel_keys(arr["means"], cell)
        n_cells = int(np.unique(k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791).size)
        if n_cells <= target_fill * n_max:
            return float(cell)
    return 60.0


def quota_keep(arr: dict, n_max: int, cell_m: float = None,
               detail_power: float = 1.0, edge_boost: float = 1.0) -> np.ndarray:
    """Equal budget per ground cell + detail-weighted within each cell.

    This is the week-8 "ortho-aware" policy: every ground cell contributes the
    same number of Gaussians, so the ortho cannot lose a whole area to a smarter
    but spatially biased score.  ``n_obs`` is deliberately **not** used as a
    primary key here: in a streaming map it is bimodal (1 or 2) and correlated
    with *where* the keyframes overlap, which collapses the selection onto the
    overlap bands (observed: 30% no-data with the plain ``ortho2`` ranking).
    """
    from stream_ortho.map.gaussian import voxel_keys

    n = len(arr["means"])
    if n == 0 or n_max >= n:
        return np.ones(n, bool)
    if cell_m is None:
        cell_m = choose_quota_cell(arr, n_max)
    keys = voxel_keys(arr["means"], cell_m)
    cell = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    # per-cell detail: conf * opacity * detail weight (edges/colour structure)
    score = (arr["conf"].astype(np.float64) * arr["opacity"].astype(np.float64)
             * (detail_weight(arr) ** detail_power))
    if edge_boost != 1.0 and "kind" in arr and len(arr["kind"]) == n:
        score = score * np.where(arr["kind"].astype(bool), edge_boost, 1.0)
    order = np.lexsort((-score, cell))
    sc = cell[order]
    rank = np.empty(n, np.int64)
    starts = np.searchsorted(sc, sc, side="left")
    rank[order] = np.arange(n, dtype=np.int64) - starts
    n_cells = int(np.unique(cell).size)
    q = n_max / max(n_cells, 1)
    keep = rank < q
    # fractional quota overshoots by at most one point per cell -> trim the extras
    if int(keep.sum()) > n_max:
        over = int(keep.sum()) - n_max
        extra = np.where(keep & (rank >= np.floor(q)))[0]
        if extra.size == 0:
            extra = np.where(keep)[0]
        drop = extra[np.argsort(score[extra])[:min(over, extra.size)]]
        keep[drop] = False
    # if cells were too small to absorb their quota, fill the rest globally
    need = n_max - int(keep.sum())
    if need > 0:
        rest = np.where(~keep)[0]
        take = rest[np.argpartition(-score[rest], min(need, len(rest)) - 1)[:need]]
        keep[take] = True
    return keep


def keep_mask(arr: dict, n_max: int, strategy: str = "conf",
              rng: np.random.Generator = None) -> np.ndarray:
    """Return a boolean mask with exactly ``min(n_max, N)`` True entries."""
    rng = rng or np.random.default_rng(0)
    n = len(arr["means"])
    if n_max <= 0 or n <= n_max:
        return np.ones(n, bool)

    keep = np.zeros(n, bool)
    if strategy == "random":
        idx = rng.choice(n, size=n_max, replace=False)
        keep[idx] = True
        return keep

    if strategy in ("ortho3", "ortho3_edge"):
        return quota_keep(arr, n_max, edge_boost=2.5 if strategy == "ortho3_edge" else 1.0)

    if strategy in ("conf", "ortho", "voxel", "ortho2", "prune", "ortho2_edge"):
        if strategy == "conf":
            score = _basic_score(arr)
        elif strategy == "ortho":
            score = ortho_score(arr)
        elif strategy in ("ortho2", "ortho2_edge", "prune"):
            score = ortho_score_v2(arr)
            if strategy == "ortho2_edge" and "kind" in arr:
                score = score * np.where(arr["kind"].astype(bool), 2.5, 1.0)
            if strategy == "prune":
                from stream_ortho.map.prune import prune_mask

                pm = prune_mask(arr)["mask"]
                score = np.where(pm, 0.0, score)
        else:
            score = _basic_score(arr)
        if strategy == "voxel":
            # coarse dedup first: highest score per 2x-voxel cell
            cell = np.floor(arr["means"] / 0.8).astype(np.int64)
            key = cell[:, 0] * 73856093 ^ cell[:, 1] * 19349663 ^ cell[:, 2] * 83492791
            order = np.lexsort((-score, key))
            _, first = np.unique(key[order], return_index=True)
            rep = order[np.sort(first)]
            if len(rep) > n_max:                      # too many representative cells
                rep = rep[np.argpartition(-score[rep], n_max - 1)[:n_max]]
            keep[rep] = True
            # fill the rest by score
        rest = np.where(~keep)[0]
        need = n_max - int(keep.sum())
        if need > 0 and len(rest):
            take = rest[np.argpartition(-score[rest], need - 1)[:need]]
            keep[take] = True
        if strategy in ("ortho2", "ortho2_edge", "prune"):
            res = protect_unique_coverage(arr, keep, cell_m=cfg_cell(arr))
            keep = trim_to_budget(arr, res["keep"], n_max, res["reps"], score)
        return keep

    if strategy == "edge":
        kind = arr["kind"].astype(bool) if "kind" in arr else np.zeros(n, bool)
        score = _basic_score(arr)
        edge_idx = np.where(kind)[0]
        if len(edge_idx) >= n_max:
            take = edge_idx[np.argpartition(-score[edge_idx], n_max - 1)[:n_max]]
            keep[take] = True
            return keep
        keep[edge_idx] = True
        rest = np.where(~keep)[0]
        need = n_max - int(keep.sum())
        if need > 0 and len(rest):
            take = rest[np.argpartition(-score[rest], need - 1)[:need]]
            keep[take] = True
        res = protect_unique_coverage(arr, keep, cell_m=cfg_cell(arr))
        keep = trim_to_budget(arr, res["keep"], n_max, res["reps"], score)
        return keep

    raise ValueError(f"unknown budget strategy {strategy}")


def composition(arr: dict, keep: np.ndarray) -> dict:
    kind = arr["kind"].astype(bool) if "kind" in arr else np.zeros(len(keep), bool)
    kept = int(keep.sum())
    return {
        "n_total": int(len(keep)),
        "n_kept": kept,
        "edge_total": int(kind.sum()),
        "edge_kept": int((kind & keep).sum()),
        "edge_keep_ratio": float((kind & keep).sum() / max(kind.sum(), 1)),
        "area_total": int((~kind).sum()),
        "area_kept": int((~kind & keep).sum()),
        "area_keep_ratio": float((~kind & keep).sum() / max((~kind).sum(), 1)),
    }


# ---------------------------------------------------------------------------
# Week 8 (S7a): ortho-aware scoring, v2
# ---------------------------------------------------------------------------
#
# Week 5's first attempt (``ortho_score`` above) rewarded *isolated* points with
# an unbounded ``1/(1+count)`` bonus and divided by the footprint: on a uniform
# plane-splat map that sent the whole budget to the map frontier and left the
# evaluated tile 100% empty.  v2 keeps the useful idea but **bounds** every
# factor so it can only modulate, never dominate:

def coverage_cells(arr: dict, cell_m: float = None, grid: int = 192):
    """Occupancy of a 2-D cell grid over the map's plane bbox.

    Returns ``(cell_id, counts, expected, shape, bbox)`` where ``expected`` is the
    median occupancy of the *occupied* cells (the "healthy density" baseline).
    """
    from stream_ortho.map.gaussian import voxel_keys

    m = arr["means"]
    if len(m) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float64), 1.0, (grid, grid), None
    lo = m[:, :2].min(axis=0) - 1e-6
    hi = m[:, :2].max(axis=0) + 1e-6
    span = np.maximum(hi - lo, 1e-6)
    ij = np.floor((m[:, :2] - lo) / span * (grid - 1)).astype(np.int64)
    ij = np.clip(ij, 0, grid - 1)
    cid = ij[:, 0] * grid + ij[:, 1]
    counts = np.bincount(cid, minlength=grid * grid).astype(np.float64)
    occ = counts[counts > 0]
    expected = float(np.median(occ)) if occ.size else 1.0
    return cid, counts, expected, (grid, grid), (lo, hi)


def detail_weight(arr: dict, kind_boost: float = 1.6, cell_colour_std: bool = True) -> np.ndarray:
    """Protect structural detail: tagged edges, and colourful (non-uniform) cells."""
    n = len(arr["means"])
    w = np.ones(n, np.float64)
    if "kind" in arr and len(arr["kind"]) == n:
        w *= np.where(arr["kind"].astype(bool), kind_boost, 1.0)
    if cell_colour_std and "colors" in arr and len(arr["colors"]) == n:
        cid, counts, expected, shape, bbox = coverage_cells(arr)
        # per-cell colour std (structure proxy): high variance -> textured -> protect
        col = arr["colors"].astype(np.float64)
        gsum = np.bincount(cid, weights=col[:, 0], minlength=counts.size)
        gsum2 = np.bincount(cid, weights=col[:, 1] ** 2 + col[:, 0] ** 2, minlength=counts.size)
        mean = gsum / np.maximum(counts, 1)
        var = np.maximum(gsum2 / np.maximum(counts, 1) - mean ** 2, 0)
        std = np.sqrt(var[cid])
        hi = np.percentile(std[counts[cid] > 1], 90) if (counts[cid] > 1).any() else 1.0
        w *= 1.0 + 0.4 * np.clip(std / max(hi, 1e-6), 0, 1)
    return w


def ortho_score_v2(arr: dict, gap_max: float = 1.5, gap_min: float = 0.35,
                   cost_power: float = 0.5, detail_power: float = 1.0) -> np.ndarray:
    """Bounded ortho-aware score (week 8).

    ``score = conf * opacity * min(n_obs,8)
              * gap_term        in [gap_min, gap_max]   (dense cells penalised, sparse capped)
              * detail_term**dp in [1, 1.6] * colour structure
              * cost_term       = (median_fp / fp)**cost_power

    Every modifier is clamped, so the confidence term stays dominant -- this is
    what prevents the week-5 collapse (budget fleeing to the frontier).
    """
    base = _basic_score(arr)
    n = len(arr["means"])
    if n == 0:
        return base
    cid, counts, expected, _, _ = coverage_cells(arr)
    ratio = expected / np.maximum(counts[cid], 1.0)          # >1 = sparser than typical
    gap = np.clip(ratio, gap_min, gap_max)
    det = detail_weight(arr) ** detail_power
    fp = footprint(arr)
    med = float(np.median(fp)) if n else 1.0
    cost = np.power(np.clip(med / np.maximum(fp, 1e-12), 0.2, 5.0), cost_power)
    return base * gap * det * cost


def protect_unique_coverage(arr: dict, keep: np.ndarray, cell_m: float = 2.0) -> dict:
    """Guarantee >= 1 kept point per occupied ``cell_m`` cell (no new holes).

    ``cell_m`` must be **coarser** than the point spacing (default 2 m ~ 10x GSD,
    while the plane splat samples every ~0.8 m), otherwise every point is its own
    cell and the protection degenerates to "keep everything" (observed in the
    first S7a run).
    """
    from stream_ortho.map.gaussian import voxel_keys

    n = len(arr["means"])
    if n == 0:
        return {"rescued": 0, "cells": 0, "keep": keep, "reps": np.zeros(0, bool)}
    keys = voxel_keys(arr["means"], cell_m)
    key = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    order = np.argsort(key, kind="stable")
    sk = key[order]
    score = _basic_score(arr)
    keep2 = keep.copy()
    reps = np.zeros(n, bool)
    rescued = 0
    start = 0
    while start < n:
        end = int(np.searchsorted(sk, sk[start], side="right"))
        idx = order[start:end]
        kept_here = idx[keep2[idx]]
        if kept_here.size:
            best = kept_here[np.argmax(score[kept_here])]
        else:
            best = idx[np.argmax(score[idx])]
            keep2[best] = True
            rescued += 1
        reps[best] = True
        start = end
    return {"rescued": rescued, "cells": int(np.unique(sk).size), "keep": keep2, "reps": reps}


def trim_to_budget(arr: dict, keep: np.ndarray, n_max: int, reps: np.ndarray,
                   score: np.ndarray = None, drop_first: np.ndarray = None) -> np.ndarray:
    """Drop the lowest-score *non-representative* points until ``N <= n_max``.

    ``drop_first`` marks points that may be dropped before the rest (e.g. area
    points before structural edges), so a strategy's own priority survives the
    coverage guarantee.
    """
    over = int(keep.sum()) - int(n_max)
    if over <= 0:
        return keep
    sc = _basic_score(arr) if score is None else score
    keep = keep.copy()
    non_rep = keep & ~reps
    if drop_first is not None:
        pref = np.where(non_rep & drop_first)[0]
        take = pref[np.argsort(sc[pref])[:min(over, pref.size)]]
        keep[take] = False
        over = int(keep.sum()) - int(n_max)
        if over <= 0:
            return keep
    cand = np.where(keep & ~reps)[0]
    if cand.size == 0:
        return keep                      # cannot trim without breaking coverage
    drop = cand[np.argsort(sc[cand])[:min(over, cand.size)]]
    keep[drop] = False
    return keep
