"""P3.2-A (light): multi-view support carving at the layer level.

Per-frame pointmaps splash around building edges.  A per-frame height threshold cannot remove
that splash -- it sits only a metre or two above the roof and is often a cell's only
observation.  What separates it from a real surface is *support*: a real roof layer is seen by
several frames, a splash usually by one.

For every ``(cell, layer)`` this module counts the distinct frames that support it and vetoes a
layer when it is

  * supported by fewer than ``min_frames`` frames, and
  * floating above a *supported* lower layer of the same cell.

A single-layer cell is never touched (that would eat coverage).  Full ray casting through the
depth maps is the heavier version of the same idea.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CarveConfig:
    min_frames: int = 2          # a layer needs this many distinct frames to be trusted
    min_gap_m: float = 1.0       # only veto layers this far above the supported lower layer
    enabled: bool = True


def carve_rows(d, cfg: CarveConfig | None = None):
    """Return ``(keep_mask, stats)`` for the observation table ``d``."""
    cfg = cfg or CarveConfig()
    kf = np.asarray(d["kf"])
    key = np.asarray(d["key"], np.int64)
    layer = np.asarray(d["layer"], np.int32)
    z = np.asarray(d["z"], np.float64)
    n = int(kf.size)
    keep = np.ones(n, bool)
    stats = {"free_space": 0, "no_support": 0, "kept": n, "total": n,
             "removed_frac": 0.0, "layers": 0}
    if not cfg.enabled or n == 0:
        return keep, stats

    pair = key * 8 + layer.astype(np.int64)
    # distinct frames per (cell, layer)
    order = np.lexsort((kf, pair))
    ps, ks = pair[order], kf[order]
    grp = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]
    new_kf = np.r_[True, ks[1:] != ks[:-1]]
    counts = np.bincount(grp[new_kf], minlength=int(grp[-1]) + 1)
    uniq = np.r_[ps[np.r_[True, ps[1:] != ps[:-1]]]]
    cnt_of = dict(zip(uniq.tolist(), counts.tolist()))

    supported = np.array([cnt_of[int(p)] >= cfg.min_frames for p in pair], bool)
    # the lowest *supported* layer of each cell is the reference surface
    low = {}
    for cell, zz in zip(key[supported], z[supported]):
        c = int(cell)
        if c not in low or zz < low[c]:
            low[c] = float(zz)

    rem = np.zeros(n, bool)
    idx = np.nonzero(~supported)[0]
    for i in idx:
        base = low.get(int(key[i]))
        if base is None:
            continue                       # no supported layer here: keep it (coverage)
        if z[i] > base + cfg.min_gap_m:
            rem[i] = True
    keep &= ~rem
    stats.update({"no_support": int(rem.sum()), "kept": int(keep.sum()),
                  "removed_frac": float(rem.sum() / n), "layers": int(uniq.size)})
    return keep, stats


def suppress_top_rows(d, tau: float = 2.0, min_support: int = 2):
    """Drop frame-rows that sit more than ``tau`` above their cell's median height.

    A cell's multiple frames normally agree within centimetres; a row that is metres above the
    median is the per-frame splash, and if it survives it *becomes* the top surface.  Cells with
    fewer than ``min_support`` rows are left alone (there is no median to trust).
    """
    z = np.asarray(d["z"], np.float64)
    key = np.asarray(d["key"], np.int64)
    n = z.size
    keep = np.ones(n, bool)
    stats = {"total": n, "suppressed": 0, "suppressed_frac": 0.0}
    if n == 0 or tau <= 0:
        return keep, stats
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[start, n])
    med = np.array([np.median(zs[s:s + c]) for s, c in zip(start, cnt)])
    med_row = np.repeat(med, cnt)
    bad_sorted = (zs > med_row + tau) & (np.repeat(cnt, cnt) >= min_support)
    bad = np.zeros(n, bool)
    bad[order] = bad_sorted
    keep &= ~bad
    stats["suppressed"] = int(bad.sum())
    stats["suppressed_frac"] = float(bad.sum() / n)
    return keep, stats
