"""P1a: 3-DOF (vertical offset + two tilts) gauge calibration per keyframe.

Model, in the plane frame::

    z_obs = z_ref(cell) + b + tx * (x - cx) + ty * (y - cy) + r

Fitted **only on cells that are single-layer, flat and supported**
(``n_layers == 1 & sigma_z < 0.3 & support >= 2``), so roofs/ground drive the fit and
edges, facades and mixed layers cannot.  Huber IRLS (4 iterations), weights
``conf * |cos incidence|``.  The first frame defines the gauge (b = tx = ty = 0); later
frames are aligned to the current product.

Only ``frame.z`` changes -- cell keys, colours and confidences are untouched, and the
observations are never discarded.
"""

from __future__ import annotations

import numpy as np


def _plane_xy(frame, store, cell: float):
    from stream_ortho.map.selection import unpack_cell

    ix, iy = unpack_cell(np.asarray(frame.cell_key, np.int64))
    x = (ix.astype(np.float64) + 0.5) * cell
    y = (iy.astype(np.float64) + 0.5) * cell
    return ix, iy, x, y


def _cell_lookup(store, ix, iy):
    H, W = store.H, store.W
    j = iy - store.oy
    i = ix - store.ox
    ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    return i, j, ok


def fit_gauge(frame, store, cell: float, max_dz: float = 3.0, iters: int = 4,
              max_pts: int = 200_000):
    """Return ``(b, tx, ty)`` for one frame (see the module docstring)."""
    ix, iy, x, y = _plane_xy(frame, store, cell)
    # subsample first: the per-cell work below is O(N) and would cost ~250 ms/frame
    if len(ix) > max_pts:
        sel = np.random.default_rng(0).choice(len(ix), max_pts, replace=False)
        ix, iy, x, y = ix[sel], iy[sel], x[sel], y[sel]
        sub = sel
    else:
        sub = None
    i, j, ok = _cell_lookup(store, ix, iy)
    if ok.sum() < 500:
        return np.zeros(3)
    ii, jj = i[ok], j[ok]
    occ = store.n[jj, ii] > 0                              # (N, L)
    n_layers = occ.sum(-1)
    sw = np.maximum(store.sum_w[jj, ii], 1e-6)
    mz = store.sum_wz[jj, ii] / sw
    var = np.maximum(store.sum_wz2[jj, ii] / sw - mz * mz, 0.0)
    top = np.where(occ, store.z_ref[jj, ii], -np.inf).argmax(-1)
    ar = np.arange(len(top))
    zref = store.z_ref[jj, ii][ar, top]
    sup = store.n[jj, ii][ar, top]
    sig = np.sqrt(var[ar, top])

    m = (n_layers == 1) & (sig < 0.3) & (sup >= 2)
    if m.sum() < 500:
        return np.zeros(3)
    idx = np.nonzero(ok)[0][m]
    zr = zref[m]
    if len(idx) > max_pts:
        sel = np.random.default_rng(0).choice(len(idx), max_pts, replace=False)
        idx, zr = idx[sel], zr[sel]
    z_all = np.asarray(frame.z, np.float64)[sub] if sub is not None else \
        np.asarray(frame.z, np.float64)
    z = z_all[idx]
    dz = z - zr
    keep = np.abs(dz) <= max_dz
    if keep.sum() < 500:
        return np.zeros(3)
    idx, dz = idx[keep], dz[keep]
    conf_full = np.asarray(frame.conf, np.float32)[sub] if sub is not None else \
        np.asarray(frame.conf, np.float32)
    cos_full = np.asarray(frame.cos_inc, np.float32)[sub] if sub is not None else \
        np.asarray(frame.cos_inc, np.float32)
    conf = conf_full[idx].astype(np.float64)
    cos = cos_full[idx].astype(np.float64)
    w0 = np.clip(conf, 1e-3, None) * np.clip(np.abs(cos), 0.2, 1.0)
    cx, cy = x[idx].mean(), y[idx].mean()
    A = np.stack([np.ones(len(idx)), x[idx] - cx, y[idx] - cy], 1)
    p = np.zeros(3)
    for _ in range(int(iters)):
        r = A @ p - dz
        d = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        huber = np.minimum(1.0, 1.345 * d / np.maximum(np.abs(r), 1e-9))
        w = w0 * huber
        Aw = A * w[:, None]
        p = np.linalg.lstsq(Aw.T @ A, Aw.T @ dz, rcond=None)[0]
    return p


def apply_gauge(frame, params, cell: float, store):
    """Return a new ``FrameObs`` whose ``z`` is corrected by the 3-DOF model."""
    from stream_ortho.map.obs import FrameObs

    b, tx, ty = (float(v) for v in params)
    if abs(b) < 1e-9 and abs(tx) < 1e-12 and abs(ty) < 1e-12:
        return frame
    ix, iy, x, y = _plane_xy(frame, store, cell)
    cx, cy = x.mean(), y.mean()
    z = np.asarray(frame.z, np.float32) - (b + tx * (x - cx) + ty * (y - cy)).astype(np.float32)
    return FrameObs(kf=frame.kf, cell_key=frame.cell_key, z=z.astype(np.float32),
                    rgb=frame.rgb, conf=frame.conf, cos_inc=frame.cos_inc, owner=frame.owner)
