"""Roof-placement accuracy (P0 acceptance metric).

**How do we know what is a roof without labels?**

We define it *geometrically, per frame, from the point map only* -- i.e. from
``X_canon`` (MASt3R-SLAM's per-pixel 3-D point, which is **independent of the Gaussian
head and of the DSM** we are testing):

1. lift ``X_canon`` to the plane frame with the Sim(3): ``P_pm``;
2. rasterise ``P_pm`` on a coarse grid (2 m) and take the **5th percentile of z** per
   cell (robust: buildings rarely fill a cell, and we want the ground, not the roof);
3. take a **minimum filter over a 25 m window** on that percentile grid, bilinearly
   sampled back per pixel -> ``z_ground_local``;
4. a pixel is a **"roof pixel"** if ``z_pm - z_ground_local >= h_min`` (default 3 m).

The metric is then: *where did **our** placement put those roof pixels?*

* ``acc_z``   : fraction with ``|z_placed - z_pm| <= tol_z``   (height right?)
* ``acc_xy``  : fraction with ``|xy_placed - xy_pm| <= tol_xy`` (planimetry right?)
* ``acc_both``: both (the headline number)

Interpretation caveat: the point map is itself noisy (per-frame relief errors of a few
metres), so the number is only meaningful **relative to baselines** that this module also
computes -- ``ground`` (place every pixel at the plane ``z=0``: cannot reach a roof) and
``identity`` (the point-map positions themselves: the upper bound of the metric).
"""

from __future__ import annotations

import numpy as np


def _grid(xy, z, cell: float, pct: float = 5.0):
    """Per-cell percentile of ``z`` on a ``cell``-metre grid."""
    ij = np.floor(xy / cell).astype(np.int64)
    ij = ij - ij.min(0, keepdims=True)
    nx, ny = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    flat = ij[:, 1] * nx + ij[:, 0]
    order = np.lexsort((z, flat))
    uniq, start, counts = np.unique(flat[order], return_index=True, return_counts=True)
    zs = z[order]
    out = np.full(ny * nx, np.nan, np.float32)
    for u, s0, c in zip(uniq, start, counts):
        out[u] = np.percentile(zs[s0:s0 + c], pct)
    return out.reshape(ny, nx), ij.min(0), cell


def local_ground(xy: np.ndarray, z: np.ndarray, cell: float = 2.0,
                 window: float = 60.0, pct: float = 5.0) -> np.ndarray:
    """Local ground height per sample: per-cell low percentile + window minimum.

    The window must be **larger than the biggest building** -- with a 25 m window a
    60 m roof fills the window completely and the "ground" becomes the roof (measured:
    roof recall collapsed to 2%).  A global low-percentile floor is also applied, which
    is valid here because the plane frame is ground-aligned.
    """
    from scipy.ndimage import minimum_filter

    g, base, c = _grid(xy, z, cell, pct)
    filled = np.where(np.isfinite(g), g, np.nan)
    if np.isnan(filled).any():
        med = np.nanmedian(filled) if np.isfinite(filled).any() else 0.0
        filled = np.where(np.isfinite(filled), filled, med)
    k = max(1, int(round(window / cell)))
    lo = minimum_filter(filled, size=k, mode="nearest")
    floor = float(np.percentile(np.asarray(z, np.float64), 2.0)) + 2.0
    lo = np.minimum(lo, floor)
    ij = np.floor(xy / cell).astype(np.int64) - base
    ij[:, 0] = np.clip(ij[:, 0], 0, lo.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, lo.shape[0] - 1)
    return lo[ij[:, 1], ij[:, 0]].astype(np.float32)


def roof_mask(P_pm: np.ndarray, cell: float = 2.0, window: float = 60.0,
              h_min: float = 3.0) -> np.ndarray:
    """Pixels whose point-map point is at least ``h_min`` above the local ground."""
    z = np.asarray(P_pm[:, 2], np.float32)
    zg = local_ground(np.asarray(P_pm[:, :2], np.float64), z, cell, window)
    return (z - zg) >= h_min


def roof_placement_accuracy(P_pm: np.ndarray, P_placed: np.ndarray,
                            cell: float = 2.0, window: float = 60.0, h_min: float = 3.0,
                            tol_xy: float = 3.0, tol_z: float = 3.0) -> dict:
    """Fraction of roof pixels that our placement puts (a) at the right height and
    (b) at the right planimetric position, relative to the point map.

    ``P_pm`` and ``P_placed`` must be aligned row-wise (same pixel order).
    """
    P_pm = np.asarray(P_pm, np.float64)
    P_placed = np.asarray(P_placed, np.float64)
    ok = np.isfinite(P_pm).all(1) & np.isfinite(P_placed).all(1)
    P_pm, P_placed = P_pm[ok], P_placed[ok]
    if len(P_pm) < 1000:
        return {"n": int(len(P_pm)), "n_roof": 0, "acc_z": float("nan"),
                "acc_xy": float("nan"), "acc_both": float("nan")}
    roof = roof_mask(P_pm, cell, window, h_min)
    dz = np.abs(P_placed[:, 2] - P_pm[:, 2])
    dxy = np.linalg.norm(P_placed[:, :2] - P_pm[:, :2], axis=1)
    r = roof
    out = {
        "n": int(len(P_pm)), "n_roof": int(r.sum()),
        "roof_frac": float(r.mean()),
        "acc_z": float((dz[r] <= tol_z).mean()) if r.any() else float("nan"),
        "acc_xy": float((dxy[r] <= tol_xy).mean()) if r.any() else float("nan"),
        "acc_both": float(((dz[r] <= tol_z) & (dxy[r] <= tol_xy)).mean()) if r.any() else float("nan"),
        "med_dz_roof": float(np.median(dz[r])) if r.any() else float("nan"),
        "med_dxy_roof": float(np.median(dxy[r])) if r.any() else float("nan"),
        "tol_xy": tol_xy, "tol_z": tol_z, "h_min": h_min,
    }
    return out
