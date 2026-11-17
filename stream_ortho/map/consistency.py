"""S1/S2 -- cross-view depth consistency gating and per-frame depth calibration.

Both act **before** a new keyframe is committed, and both target the same failure mode:
a new frame's point map is displaced by a few metres, so its points disagree with the
surface that earlier (well-observed) frames already established.  Without a check the
cell selection simply hands the cell to the new frame (its ``conf*opacity`` is often
higher) and the structure visibly drifts.

* **S1** ``cross_view_mask``   : drop pixels whose 3D point disagrees with the depth the
  *neighbouring keyframes* measured for the same physical point (occlusion / bad point).
* **S2** ``fit_depth_calibration`` : fit a 1-DoF-per-frame ``z' = a*z + b`` that aligns
  the new frame to the committed surface; reject implausible fits instead of applying
  them (the earlier ``z_align`` applied them blindly and made things worse).
* ``cell_surface``           : the per-cell top surface, used both by S2 and by the
  drift metric.
"""

from __future__ import annotations

import numpy as np


def head_grid_rays(K, H, W, h, w, R):
    """Ray directions (h, w, 3) in the **world** frame for the head-resolution grid."""
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * (W / float(w)),
                         (np.arange(h) + 0.5) * (H / float(h)), indexing="xy")
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    return rays @ R.T


def cross_view_mask(npz, npz_intr, sim3_fn, k: int, neighbours, tau: float = 2.0,
                    min_valid: float = 0.05, max_views: int = 3):
    """S1: pixels whose point-map point disagrees with the neighbours' depth.

    ``sim3_fn(k)`` must return ``(R, t, s)`` of the camera-to-world Sim(3) for keyframe
    ``k``.  ``npz_intr`` is ``(K, H, W)`` of the original (full-resolution) image, used
    only to build the sampling grid; the comparison itself is in **depth along the
    neighbour's z**.

    Returns ``(keep_mask (h,w) bool, info dict)`` with ``h,w = X_canon.shape``.
    """
    X = np.asarray(npz["X_canon"][k], np.float32)
    h, w = X.shape[:2]
    K, H, W = npz_intr
    R, t, s = sim3_fn(k)
    d = X[..., 2] * float(s)
    rays = head_grid_rays(K, H, W, h, w, R)
    Pw = (d[..., None] * rays) + t                    # plane-frame/world points
    votes = np.zeros((h, w), np.float32)
    used = np.zeros((h, w), np.float32)
    for j in neighbours[:max_views]:
        Rj, tj, sj = sim3_fn(j)
        Xj = np.asarray(npz["X_canon"][j], np.float32)
        hj, wj = Xj.shape[:2]
        # world -> camera j
        Pc = (Pw - tj) @ Rj                          # R^T (P - t) with R orthonormal
        z = Pc[..., 2]
        ok = z > 1e-6
        u = np.full_like(z, -1.0); v = np.full_like(z, -1.0)
        u[ok] = K[0, 0] * Pc[..., 0][ok] / z[ok] + K[0, 2]
        v[ok] = K[1, 1] * Pc[..., 1][ok] / z[ok] + K[1, 2]
        iu = np.clip((u / (W / float(wj))).astype(np.int64), 0, wj - 1)
        iv = np.clip((v / (H / float(hj))).astype(np.int64), 0, hj - 1)
        inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dj = Xj[..., 2] * float(sj)
        diff = np.abs(z - dj[iv, iu]) / np.maximum(dj[iv, iu], 1e-3)
        valid = inside & np.isfinite(diff)
        # metric tolerance: tau metres on a ~250 m depth -> convert to relative
        rel = tau / np.maximum(dj[iv, iu], 1.0)
        agree = valid & (diff <= np.maximum(rel, min_valid))
        votes += agree.astype(np.float32)
        used += valid.astype(np.float32)
    keep = (votes >= 1.0) | (used < 1.0)             # no coverage -> keep (can't judge)
    info = {"frac_dropped": float((~keep).mean()),
            "n_views": int(min(len(neighbours), max_views)),
            "frac_judged": float((used >= 1.0).mean())}
    return keep, info


def cell_surface(means, T_WP, cell: float = 0.5, q=None, mode: str = "top"):
    """Per-cell surface height (top or best-quality) -> ``(keys, z, idx)``."""
    means = np.asarray(means, np.float64)
    P = (means - T_WP[:3, 3]) @ T_WP[:3, :3]
    from stream_ortho.map.selection import pack_cell

    ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
    # **absolute** cell key: keys must be comparable across two snapshots taken at
    # different times.  ``pack_cell`` is an exact bijection including negative indices
    # (the naive ``ix*100003+iy`` wrapped for negative y and broke the dense grid).
    key = pack_cell(ij[:, 0], ij[:, 1])
    if mode == "top":
        order = np.lexsort((-P[:, 2], key))
    else:
        qq = np.ones(len(means), np.float32) if q is None else np.asarray(q, np.float32)
        order = np.lexsort((-qq, key))
    ks = key[order]
    _, first = np.unique(ks, return_index=True)
    idx = order[first]
    return key[idx], P[idx, 2].astype(np.float32), idx


def fit_depth_calibration(means_new, committed_means, T_WP, cell: float = 0.5,
                          z_range: float = 40.0, min_pairs: int = 2000,
                          max_scale_dev: float = 0.05, max_bias: float = 8.0,
                          min_gain: float = 0.15):
    """S2: robust per-frame ``z' = a*z + b`` against the committed surface.

    Uses the **ground and the roofs** (|z| < ``z_range``) and a median/quantile fit so
    a few outliers cannot move the frame.  Returns ``dict(ok, a, b, n, resid)`` and
    **refuses** implausible corrections instead of applying them.
    """
    Pn = (np.asarray(means_new, np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
    key_c, z_c, _ = cell_surface(committed_means, T_WP, cell=cell, mode="top")
    if len(key_c) < 100:
        return {"ok": False, "a": 1.0, "b": 0.0, "n": 0, "resid": 0.0, "reason": "empty-map"}
    ij = np.floor(Pn[:, :2] / float(cell)).astype(np.int64)
    from stream_ortho.map.selection import pack_cell
    key_n = pack_cell(ij[:, 0], ij[:, 1])     # same absolute packing as ``cell_surface``
    # index the committed cells by key
    order = np.argsort(key_c)
    kc, zc = key_c[order], z_c[order]
    pos = np.searchsorted(kc, key_n)
    pos = np.clip(pos, 0, len(kc) - 1)
    hit = kc[pos] == key_n
    sel = hit & (np.abs(Pn[:, 2]) < z_range)
    n = int(sel.sum())
    if n < min_pairs:
        return {"ok": False, "a": 1.0, "b": 0.0, "n": n, "resid": 0.0, "reason": "too-few-pairs"}
    zn, zc_ = Pn[sel, 2], zc[pos[sel]].astype(np.float64)
    dz = zc_ - zn
    # robust: median offset + Theil-Sen style slope from quantile pairs
    b0 = float(np.median(dz))
    resid0 = float(np.median(np.abs(dz - b0)))
    a0 = 1.0
    try:
        qs = np.quantile(zn, [0.1, 0.9])
        m1 = np.abs(zn - qs[0]) < 2.0
        m2 = np.abs(zn - qs[1]) < 2.0
        if m1.sum() > 50 and m2.sum() > 50:
            slope = (np.median(zc_[m2]) - np.median(zc_[m1])) / max(qs[1] - qs[0], 1e-3)
            a0 = float(np.clip(slope, 0.9, 1.1))
    except Exception:
        a0 = 1.0
    b1 = float(np.median(zc_ - a0 * zn))
    resid1 = float(np.median(np.abs(zc_ - (a0 * zn + b1))))
    # accept only if the correction is plausible **and** really helps: a large shift
    # that does not reduce the residual is exactly the failure mode of the old z_align
    ok = (abs(a0 - 1.0) <= max_scale_dev and abs(b1) <= max_bias
          and resid1 <= resid0 * (1.0 - min_gain))
    return {"ok": bool(ok), "a": a0, "b": b1, "n": n,
            "resid": resid1, "resid_before": resid0,
            "reason": "ok" if ok else "rejected"}


def drift_metric(z_before: dict, z_after: dict, tol: float = 0.3) -> dict:
    """How much a *stable* surface moved when new frames were added.

    ``z_before`` / ``z_after`` map cell-key -> top z.  Only cells present in both (i.e.
    already well observed before) are considered.
    """
    common = np.intersect1d(np.array(list(z_before.keys())), np.array(list(z_after.keys())))
    if len(common) == 0:
        return {"n": 0}
    a = np.array([z_before[int(k)] for k in common], np.float32)
    b = np.array([z_after[int(k)] for k in common], np.float32)
    d = np.abs(b - a)
    return {"n": int(len(common)), "median_dz": float(np.median(d)),
            "p95_dz": float(np.percentile(d, 95)),
            "frac_moved": float((d > tol).mean())}


def _local_surface(keys, zs, cell: float, rad: int = 1):
    """Sparse grid of the established surface; returns a lookup ``(z, count)`` per cell."""
    from stream_ortho.map.selection import unpack_cell
    ix, iy = unpack_cell(keys)
    x0, y0 = int(ix.min()), int(iy.min())
    nx, ny = int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1
    grid = np.full((ny, nx), np.nan, np.float32)
    grid[iy - y0, ix - x0] = zs.astype(np.float32)
    from scipy.ndimage import median_filter, uniform_filter
    cnt = np.isfinite(grid).astype(np.float32)
    fill = float(np.nanmedian(grid)) if np.isfinite(grid).any() else 0.0
    sm = median_filter(np.where(np.isfinite(grid), grid, fill), size=2 * rad + 1)
    # a *median* over a binary occupancy grid is almost always 0 (with >=50% empty
    # neighbours); the count must be averaged -> uniform_filter
    cm = uniform_filter(cnt, size=2 * rad + 1)

    def lookup(k):
        ixx, iyy = unpack_cell(k)
        ok = (ixx >= x0) & (ixx < x0 + nx) & (iyy >= y0) & (iyy < y0 + ny)
        zz = np.full(len(k), np.nan, np.float32)
        cc = np.zeros(len(k), np.float32)
        if ok.any():
            zz[ok] = sm[iyy[ok] - y0, ixx[ok] - x0]
            cc[ok] = cm[iyy[ok] - y0, ixx[ok] - x0]
        return zz, cc

    return lookup


def admission_mask(means_new, committed_means, T_WP, cell: float = 0.5,
                   tau: float = 1.5, min_support_cells: float = 0.2, rad: int = 1,
                   pending=None, support: int = 2, agree: float = 0.5, frame: int = 0):
    """S3 admission control: reject new points that float above the established surface.

    A candidate is **rejected** when the neighbourhood already has a surface
    (``count >= min_support_cells``) and ``|z_new - z_local| > tau``.  Rejected
    candidates are recorded in ``pending`` (dict cell-key -> [z, count]); when
    ``support`` independent **frames** (``frame`` id, not points) propose the *same*
    height (within ``agree``) the cell
    is admitted -- so a genuinely new structure (a roof appearing for the first time)
    still enters, but a single frame cannot inject an off-surface point next to an
    established building.

    Returns ``(keep_mask, info)``.
    """
    from stream_ortho.map.selection import quality_score

    means_new = np.asarray(means_new, np.float64)
    if support <= 1:
        return np.ones(len(np.asarray(means_new)), bool), {"rejected": 0, "judged": 0,
                                                           "admitted_late": 0, "pending_cells": 0}
    if len(means_new) == 0 or committed_means is None or len(committed_means) < 1000:
        return np.ones(len(means_new), bool), {"rejected": 0, "judged": 0, "admitted_late": 0}
    if pending is None:
        pending = {}
    q = None
    keys_c, zs_c, _ = cell_surface(committed_means, T_WP, cell=cell, q=q, mode="quality")
    if len(keys_c) < 50:
        return np.ones(len(means_new), bool), {"rejected": 0, "judged": 0, "admitted_late": 0}
    lookup = _local_surface(keys_c, zs_c, cell, rad=rad)
    Pn = (means_new - T_WP[:3, 3]) @ T_WP[:3, :3]
    from stream_ortho.map.selection import pack_cell
    ij = np.floor(Pn[:, :2] / float(cell)).astype(np.int64)
    key = pack_cell(ij[:, 0], ij[:, 1])
    zl, cc = lookup(key)
    judged = np.isfinite(zl) & (cc >= min_support_cells)
    off = judged & (np.abs(Pn[:, 2] - zl) > tau)
    keep = ~off
    late = 0
    if off.any():
        for i in np.where(off)[0]:
            k = int(key[i])
            z = float(Pn[i, 2])
            e = pending.get(k)
            if e is None:
                pending[k] = [z, {int(frame)}]
            elif abs(e[0] - z) <= agree:
                e[1].add(int(frame))                   # count **frames**, not points
                if len(e[1]) >= support:
                    keep[i] = True                      # independent frames agree
                    late += 1
            else:
                pending[k] = [z, {int(frame)}]           # conflicting: restart the vote
    info = {"rejected": int(off.sum()), "judged": int(judged.sum()),
            "admitted_late": late, "pending_cells": len(pending)}
    return keep, info


def freeze_update(prev: dict, cur: dict, tol: float = 0.5) -> dict:
    """S3 freeze: cells that kept their height between two snapshots are locked.

    Returns ``{cell_key: locked_z}`` (a dict, because ``collapse_cells`` needs the
    height to keep), not a bare set.
    """
    if not prev or not cur:
        return {}
    common = prev.keys() & cur.keys()
    return {int(k): float(prev[k]) for k in common if abs(prev[k] - cur[k]) <= tol}


def floating_new_cells(snap: dict, after: dict, cell: float = 0.5) -> dict:
    """Fraction of *new* cells that sit away from the already-established surface."""
    from scipy.ndimage import median_filter, uniform_filter

    if not snap or not after:
        return {"n_new": 0}
    new = np.array([k for k in after if k not in snap], np.int64)
    if len(new) == 0:
        return {"n_new": 0}
    from stream_ortho.map.selection import unpack_cell
    ix, iy = unpack_cell(np.array(list(snap.keys()), np.int64))
    x0, y0 = int(ix.min()), int(iy.min())
    nx, ny = int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1
    grid = np.full((ny, nx), np.nan, np.float32)
    grid[iy - y0, ix - x0] = np.array(list(snap.values()), np.float32)
    cnt = np.isfinite(grid).astype(np.float32)
    fill = float(np.nanmedian(grid)) if np.isfinite(grid).any() else 0.0
    sm = median_filter(np.where(np.isfinite(grid), grid, fill), size=3)
    cm = uniform_filter(cnt, size=3)
    nix, niy = unpack_cell(new)
    ok = (nix >= x0) & (nix < x0 + nx) & (niy >= y0) & (niy < y0 + ny)
    zz = np.array([after[int(k)] for k in new], np.float32)
    dz = np.full(len(new), np.nan, np.float32)
    cc = np.zeros(len(new), np.float32)
    dz[ok] = np.abs(zz[ok] - sm[niy[ok] - y0, nix[ok] - x0])
    cc[ok] = cm[niy[ok] - y0, nix[ok] - x0]
    j = np.isfinite(dz) & (cc > 0.2)
    if j.sum() == 0:
        return {"n_new": int(len(new)), "n_judged": 0}
    d = dz[j]
    return {"n_new": int(len(new)), "n_judged": int(j.sum()),
            "median_dz": float(np.median(d)), "p90_dz": float(np.percentile(d, 90)),
            "frac_gt1m": float((d > 1.0).mean()), "frac_gt3m": float((d > 3.0).mean())}


def fsc_heldout(points_world, kf, npz, npz_intr, sim3_fn, T_WP=None, tau: float = 1.5,
                stride_px: int = 2) -> dict:
    """Free-Space Contamination on a **held-out** keyframe (no GT needed).

    ``points_world`` are the map's Gaussians in the **world** frame (that is what
    ``GaussianMap.committed["means"]`` stores).  Earlier versions took plane
    coordinates, which silently produced 0 visible points / a nonsense 91%.

    Every map point is projected into a keyframe that was *not* used to build the map.
    If it lands **in front of** the surface that keyframe observed
    (``z < d_obs - tau``), the map is claiming geometry in space that this view saw
    through -> contamination ("floating point").  This is view-dependent by design: a
    floating point only shows up in views that see through it, which is exactly how a
    human notices it.
    """
    K, H, W = npz_intr
    Pw = np.asarray(points_world, np.float64)
    R, t, s = sim3_fn(kf)
    Pc = (Pw - t) @ R
    z = Pc[..., 2]
    ok = z > 1e-6
    u = np.full(len(Pw), -1.0)
    v = np.full(len(Pw), -1.0)
    u[ok] = K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 2]
    v[ok] = K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]
    X = np.asarray(npz["X_canon"][kf], np.float32)
    hj, wj = X.shape[:2]
    iu = np.clip((u / (W / float(wj))).astype(np.int64), 0, wj - 1)
    iv = np.clip((v / (H / float(hj))).astype(np.int64), 0, hj - 1)
    inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d_obs = X[..., 2] * float(s)
    dz = z - d_obs[iv, iu]
    front = inside & (dz < -tau)
    n_in = int(inside.sum())
    return {"n_points": int(len(Pw)), "n_visible": n_in,
            "fsc": float(front.sum() / max(n_in, 1)),
            "med_depth_offset": float(np.median(dz[inside])) if n_in else float("nan")}
