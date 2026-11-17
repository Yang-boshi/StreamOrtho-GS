"""Pruning rules for the global map (design doc §7.9 item 2).

Four independent reasons, each with its own counter so the week-8 visualisation
can colour the pruned Gaussians by *why* they were dropped:

* ``opacity``   -- nearly transparent splats contribute nothing;
* ``scale``     -- a splat larger than ``scale_max_m`` is a floater/blur blob;
* ``curtain``   -- a splat stretched **along the plane normal** (vertical) is the
  classic "wall of fog" artefact of aerial 3DGS; real buildings are vertical
  *structures*, not vertical *splats*, so we drop those that are thin in-plane
  and long along z;
* ``isolated``  -- no neighbour within ``iso_radius_m`` **and** low confidence
  (a lone point in empty space cannot be verified by multi-view).

Protections (never pruned):

* ``kind == 1`` (structural edge) with ``conf >= edge_conf_min``;
* the last point of a 0.4 m cell (``protect_unique_coverage``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PruneConfig:
    opacity_min: float = 0.05
    scale_max_m: float = 3.0
    curtain_ratio: float = 4.0        # sigma_z / max(sigma_xy)
    curtain_xy_max_m: float = 0.6
    iso_radius_m: float = 1.5
    iso_conf_max: float = 1.2
    edge_conf_min: float = 1.5
    max_prune_frac: float = 0.35      # safety cap


def _iso_mask(means: np.ndarray, radius: float) -> np.ndarray:
    """True where a point has *no* other point within ``radius`` (coarse voxel test)."""
    if len(means) < 2:
        return np.zeros(len(means), bool)
    from stream_ortho.map.gaussian import voxel_keys

    v = max(radius, 1e-3)
    keys = voxel_keys(means, v)
    key = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return cnt[inv] <= 1


def prune_mask(arr: dict, cfg: PruneConfig = None) -> dict:
    """Return ``{mask, reasons, n_pruned}``; ``mask`` True = *drop this point*."""
    cfg = cfg or PruneConfig()
    n = len(arr["means"])
    if n == 0:
        return {"mask": np.zeros(0, bool), "reasons": {}, "n_pruned": 0, "n": 0}

    kind = arr["kind"].astype(bool) if "kind" in arr else np.zeros(n, bool)
    conf = arr["conf"].astype(np.float64)
    s = arr["scales"].astype(np.float64)
    sxy = s[:, :2].max(axis=1)
    sz = s[:, 2]

    reasons = {
        "opacity": arr["opacity"].astype(np.float64) < cfg.opacity_min,
        "scale": s.max(axis=1) > cfg.scale_max_m,
        "curtain": (sz > cfg.curtain_ratio * np.maximum(sxy, 1e-9)) & (sxy < cfg.curtain_xy_max_m),
        "isolated": _iso_mask(arr["means"], cfg.iso_radius_m) & (conf < cfg.iso_conf_max),
    }
    prunable = np.zeros(n, bool)
    for v in reasons.values():
        prunable |= v

    protected = kind & (conf >= cfg.edge_conf_min)
    prunable &= ~protected

    # safety: never prune more than ``max_prune_frac`` of the map
    if prunable.sum() > cfg.max_prune_frac * n:
        score = conf * arr["opacity"].astype(np.float64)
        idx = np.where(prunable)[0]
        keep_n = int(np.ceil(cfg.max_prune_frac * n))
        drop = idx[np.argsort(score[idx])[:keep_n]]     # drop the least informative
        prunable = np.zeros(n, bool)
        prunable[drop] = True

    return {
        "mask": prunable,
        "reasons": {k: int(v.sum()) for k, v in reasons.items()},
        "n_pruned": int(prunable.sum()),
        "n": int(n),
        "n_protected_edges": int(protected.sum()),
    }


def floater_ratio(arr: dict, cfg: PruneConfig = None) -> float:
    """Fraction of points that *look* like floaters (for the week-8 metric)."""
    cfg = cfg or PruneConfig()
    n = len(arr["means"])
    if n == 0:
        return 0.0
    s = arr["scales"].astype(np.float64)
    sxy = s[:, :2].max(axis=1)
    sz = s[:, 2]
    is_curtain = (sz > cfg.curtain_ratio * np.maximum(sxy, 1e-9)) & (sxy < cfg.curtain_xy_max_m)
    is_big = s.max(axis=1) > cfg.scale_max_m
    is_iso = _iso_mask(arr["means"], cfg.iso_radius_m) & (arr["conf"].astype(np.float64) < cfg.iso_conf_max)
    return float((is_curtain | is_big | is_iso).mean())
