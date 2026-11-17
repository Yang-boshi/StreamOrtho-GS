"""Module A: filter a frame's pointmap *before* it enters the product.

Three stages, all vectorised numpy and O(points):

``gate``    confidence / depth / height bounds -- obvious junk
``nfp``     nadir free-space prior: a point well *below* the running per-column surface is
            occluded by it, so it must not be fused (this is what keeps hidden facades out)
``prune``   isolated floaters (low local support and far above the local surface) and
            curtains (steep vertical strips), with an **edge guard**: columns next to a
            height discontinuity are protected, so roof edges are not eaten

The filter reports how many points each stage removed; the pipeline writes those counts to
``filter_stats.json`` so an ablation (`filter.enabled: false`) is directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilterConfig:
    enabled: bool = True
    gate: bool = True
    nfp: bool = True
    prune: bool = True
    gate_only: bool = False         # stop after the gate: no grid/NFP state
    min_conf: float = 0.5
    depth_min_m: float = 5.0
    depth_max_m: float = 1500.0
    height_min_m: float = -3.0
    height_max_m: float = 120.0
    iso_radius_m: float = 1.5
    iso_conf_max: float = 1.2
    curtain_ratio: float = 4.0
    max_prune_frac: float = 0.35
    cell: float = 0.45
    origin: tuple = (0.0, 0.0)      # plane-frame origin of the product grid
    #: ``above`` (default, the original NFP: veto points *above* the known column top),
    #: ``below`` (hidden facades -- only meaningful on layered columns) or ``both``.
    nfp_mode: str = "above"
    nfp_top_pct: float = 90.0       # top surface percentile (running max until enough samples)
    nfp_bottom_pct: float = 10.0
    nfp_below_min_gap_m: float = 1.5
    nfp_below_tau_m: float = 0.6
    protect_unique: bool = True     # a cell's only point in this frame is never vetoed
    protect_low_obs: bool = True    # a column seen < nfp_min_obs times vetoes nothing
    #: how far above the column top a point must sit before it is called a floater.  It has
    #: to be larger than a legitimate step between a resolved column and a nearby roof,
    #: otherwise the first roof frame after ground frames is vetoed.
    nfp_tau_m: float = 4.0
    nfp_min_obs: int = 2            # a column must be seen this often before it vetoes
    prune_cell_margin: int = 2      # ROI margin (cells) for the cached 3-D support volume
    prune_vz_m: float = 1.5         # vertical voxel size
    prune_max_layers: int = 128
    iso_min_support: int = 2        # fewer neighbours within iso_radius -> "isolated"
    iso_height_gap_m: float = 1.2   # ... and this far above the local surface -> floater
    edge_protect_cells: int = 2


class PointmapFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self._col_hi = None             # running per-column max (top surface proxy)
        self._col_lo = None             # running per-column min
        self._col_cnt = None
        self._shape = None
        self._vol = None                # cached 3-D support volume (persistent)
        self._vol_z0 = 0.0
        self._vol_nz = 0

    # ------------------------------------------------------------------
    def _grid(self, x, y):
        c = self.cfg.cell
        ox, oy = self.cfg.origin
        # the filter must work in the *product's* index space: plane coordinates can be
        # negative, and a raw floor() would produce a negative grid shape
        ix = np.floor((np.asarray(x) - ox) / c).astype(np.int64)
        iy = np.floor((np.asarray(y) - oy) / c).astype(np.int64)
        if self._shape is None:
            self._shape = (int(iy.max()) + 2, int(ix.max()) + 2)
        h, w = self._shape
        ix = np.clip(ix, 0, w - 1)
        iy = np.clip(iy, 0, h - 1)
        return iy * w + ix, iy, ix

    def _ensure(self, size):
        if self._col_hi is None or self._col_hi.size != size:
            self._col_hi = np.full(size, -np.inf, np.float64)
            self._col_lo = np.full(size, np.inf, np.float64)
            self._col_cnt = np.zeros(size, np.int32)

    # ------------------------------------------------------------------
    def apply(self, x, y, z, conf, depth=None, height=None):
        """Return ``(keep, reasons)`` for one frame; updates the internal NFP state."""
        cfg = self.cfg
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
        z = np.asarray(z, np.float64); conf = np.asarray(conf, np.float64)
        n = z.size
        keep = np.ones(n, bool)
        reasons = {"gate": 0, "nfp": 0, "nfp_above": 0, "nfp_below": 0, "prune": 0,
                   "protected_unique": 0, "protected_low_obs": 0, "total": n}
        if not cfg.enabled or n == 0:
            reasons.update({"kept": int(keep.sum()), "removed": 0, "removed_frac": 0.0})
            return keep, reasons

        if cfg.gate:
            bad = ~np.isfinite(z) | ~np.isfinite(conf) | (conf < cfg.min_conf)
            if depth is not None:
                d = np.asarray(depth, np.float64)
                bad |= (d < cfg.depth_min_m) | (d > cfg.depth_max_m)
            if height is not None:
                h = np.asarray(height, np.float64)
                bad |= (h < cfg.height_min_m) | (h > cfg.height_max_m)
            reasons["gate"] = int(bad.sum())
            keep &= ~bad

        if getattr(cfg, "gate_only", False):
            # P5 3DGS final render: deterministic isfinite+conf gate, no NFP grid
            # bookkeeping (the caller has no product grid to anchor `origin` to)
            reasons.update({"kept": int(keep.sum()), "removed": int((~keep).sum()),
                            "removed_frac": float((~keep).mean()) if n else 0.0})
            return keep, reasons

        flat, iy, ix = self._grid(x, y)
        w = self._shape[1]
        self._ensure(self._shape[0] * self._shape[1])

        # ---------------- NFP: only "above the known top" is free space ----------------
        if cfg.nfp:
            cnt = self._col_cnt[flat]
            top = self._col_hi[flat]
            bot = self._col_lo[flat]
            known = (cnt >= cfg.nfp_min_obs) & np.isfinite(top)
            above = known & (z > top + cfg.nfp_tau_m)
            below = np.zeros(n, bool)
            if cfg.nfp_mode in ("below", "both"):
                layered = known & np.isfinite(bot) & ((top - bot) > cfg.nfp_below_min_gap_m)
                below = layered & (z < bot - cfg.nfp_below_tau_m)
            bad = above | below
            if cfg.protect_low_obs:
                bad &= (cnt >= cfg.nfp_min_obs)
            if cfg.protect_unique:
                # a cell's only point in this frame is the only chance the product has to
                # cover it -- never veto it
                uniq = np.zeros(n, bool)
                counts = np.bincount(flat, minlength=self._col_cnt.size)
                uniq = counts[flat] <= 1
                bad &= ~uniq
                reasons["protected_unique"] = int((above | below)[uniq].sum())
            reasons["nfp_above"] = int((above & keep).sum())
            reasons["nfp_below"] = int((below & keep).sum())
            reasons["nfp"] = int((bad & keep).sum())
            keep &= ~bad

        # ---------------- prune: 3-D support on a cached ROI ----------------
        if cfg.prune and keep.any():
            reasons.update(self._prune_roi(x, y, z, keep, flat, iy, ix))
            keep &= ~reasons.pop("_bad", np.zeros(n, bool))

        # update the column state with the kept points only
        if keep.any():
            np.maximum.at(self._col_hi, flat[keep], z[keep])
            np.minimum.at(self._col_lo, flat[keep], z[keep])
            np.add.at(self._col_cnt, flat[keep], 1)
        reasons["kept"] = int(keep.sum())
        reasons["removed"] = int(n - keep.sum())
        reasons["removed_frac"] = float((n - keep.sum()) / n) if n else 0.0
        return keep, reasons

    def _prune_roi(self, x, y, z, keep, flat, iy, ix):
        """Remove isolated floaters using a cached 3-D support volume, filtered on an ROI."""
        from scipy.ndimage import uniform_filter
        cfg = self.cfg
        n = z.size
        out = {"prune": 0, "_bad": np.zeros(n, bool)}
        m = cfg.prune_cell_margin
        iy0, iy1 = max(int(iy.min()) - m, 0), min(int(iy.max()) + m + 1, self._shape[0])
        ix0, ix1 = max(int(ix.min()) - m, 0), min(int(ix.max()) + m + 1, self._shape[1])
        if self._vol is None:
            fin = np.isfinite(z)
            self._vol_z0 = float(np.floor(np.min(z[fin]) / cfg.prune_vz_m)) if fin.any() else 0.0
            zspan = float(np.max(z[fin]) - np.min(z[fin])) if fin.any() else 0.0
            self._vol_nz = int(min(max(zspan / cfg.prune_vz_m + 1, 1), cfg.prune_max_layers))
            self._vol = np.zeros((self._shape[0], self._shape[1], self._vol_nz), np.float32)
        nz = self._vol_nz
        viz = np.clip(np.floor(z / cfg.prune_vz_m - self._vol_z0).astype(np.int64), 0, nz - 1)
        # accumulate this frame's kept points (whole frame; cheap scatter)
        np.add.at(self._vol, (iy[keep], ix[keep], viz[keep]), 1.0)
        # filter only the ROI (padded by one voxel so the border sees its neighbours)
        sub = self._vol[max(iy0 - 1, 0):iy1 + 1, max(ix0 - 1, 0):ix1 + 1, :]
        sup = uniform_filter(sub, size=(3, 3, 3), mode="constant") * 27.0
        off_y, off_x = max(iy0 - 1, 0), max(ix0 - 1, 0)
        sy = iy - off_y
        sx = ix - off_x
        ok = (sy >= 0) & (sy < sup.shape[0]) & (sx >= 0) & (sx < sup.shape[1])
        support = np.zeros(n, np.float32)
        support[ok] = sup[sy[ok], sx[ok], viz[ok]]
        support = np.maximum(support - 1.0, 0.0)          # exclude the point itself
        top = self._col_hi[flat]
        lvl = np.where(np.isfinite(top), top, np.nan)
        floating = np.isfinite(lvl) & (z > lvl + cfg.iso_height_gap_m) & (support < cfg.iso_min_support)
        bad = floating & keep
        cap = int(cfg.max_prune_frac * max(int(keep.sum()), 1))
        if bad.sum() > cap:
            idx = np.nonzero(bad)[0]
            order = np.argsort(-(z[idx] - lvl[idx]))
            keep_bad = np.zeros(n, bool)
            keep_bad[idx[order[:cap]]] = True
            bad = keep_bad
        out["prune"] = int(bad.sum())
        out["_bad"] = bad
        return out

    def reset(self):
        self._col_hi = None
        self._col_lo = None
        self._col_cnt = None
        self._shape = None
        self._vol = None


def conf_stat_value(conf, n_upd, stat: str = "avg"):
    """Per-point confidence in the requested convention.

    ``avg`` reproduces upstream's ``get_average_conf() = C / N``; ``max`` returns the raw
    accumulator, which is what this repository used before P3.3.
    """
    import numpy as _np
    c = _np.asarray(conf, _np.float64)
    if stat == "avg":
        n = _np.asarray(n_upd, _np.float64)
        n = _np.where(_np.isfinite(n) & (n > 0), n, 1.0)
        return c / n
    return c
