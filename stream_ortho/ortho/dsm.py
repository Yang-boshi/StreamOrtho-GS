"""2.5D DSM from the SLAM point maps (week8-end-4 task 4, v0).

The plane-splat path puts **every** pixel on ``z = 0``, so roofs (height ``h``)
and facades are displaced by ``h * tan(theta)`` away from the block centre --
measured 9-12 m for a 15 m building at 30-40 deg.  That is what makes building
edges look "smeared" in the ortho.

This module builds a coarse height field from the *actual* 3D points that
MASt3R-SLAM produced (``X_canon`` in ``keyframes.npz``), lifted to the GT world
with the est->GT Sim3 and rasterised into the plane grid taking the **top**
surface per cell.  The plane patch then uses ``z = DSM(x, y)`` instead of 0.

It is a v0: no hole filling beyond a small morphological closing, no roof/facade
separation, and the point maps themselves have a few metres of error.  It is
enough to test whether lifting the surface helps -- which it should, most of all
on `factory` (a wall-dominated block).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DSM:
    x0: float
    y0: float
    gsd: float
    z: np.ndarray                 # (ny, nx) float32, NaN = no data

    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.z.shape

    def lookup(self, x, y, fill: float = 0.0):
        """Nearest-cell height at plane coordinates (metres)."""
        ij = np.stack([(np.asarray(x) - self.x0) / self.gsd,
                       (np.asarray(y) - self.y0) / self.gsd], -1)
        i = np.clip(np.round(ij[..., 0]).astype(np.int64), 0, self.z.shape[1] - 1)
        j = np.clip(np.round(ij[..., 1]).astype(np.int64), 0, self.z.shape[0] - 1)
        out = self.z[j, i]
        return np.where(np.isfinite(out), out, fill).astype(np.float32)

    def stats(self) -> dict:
        v = self.z[np.isfinite(self.z)]
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "min": float(v.min()), "p5": float(np.percentile(v, 5)),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max()), "valid_frac": float(np.isfinite(self.z).mean())}

    def save_preview(self, path, zmin=None, zmax=None):
        import cv2

        z = self.z.copy()
        v = z[np.isfinite(z)]
        if v.size == 0:
            return
        zmin = float(np.percentile(v, 2)) if zmin is None else zmin
        zmax = float(np.percentile(v, 98)) if zmax is None else zmax
        img = np.clip((z - zmin) / max(zmax - zmin, 1e-6), 0, 1)
        img = np.nan_to_num(img, nan=0.0)
        col = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(path), col)


# ---------------------------------------------------------------------------
def _rasterise(x, y, z, gsd: float, percentile: float = 95.0, close_px: int = 3,
               height_max: float = 80.0) -> Optional[DSM]:
    """Points (plane frame, metres) -> top-surface height field."""
    import cv2

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z < height_max) & (z > -50.0)
    x, y, z = x[ok], y[ok], z[ok]
    if x.size < 100:
        return None
    x0, x1 = np.percentile(x, [0.5, 99.5])
    y0, y1 = np.percentile(y, [0.5, 99.5])
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    x, y, z = x[keep], y[keep], z[keep]
    nx = int(min(np.ceil((x1 - x0) / gsd) + 1, 20000))
    ny = int(min(np.ceil((y1 - y0) / gsd) + 1, 20000))
    i = np.clip(((x - x0) / gsd).astype(np.int64), 0, nx - 1)
    j = np.clip(((y - y0) / gsd).astype(np.int64), 0, ny - 1)

    order = np.lexsort((z, j, i))
    cell = (j[order] * nx + i[order]).astype(np.int64)
    zs_sorted = z[order]
    uniq, start, counts = np.unique(cell, return_index=True, return_counts=True)
    q = np.clip(percentile / 100.0, 0.0, 1.0)
    pick = start + np.minimum(((counts - 1) * q).astype(np.int64), counts - 1)
    ztop = zs_sorted[pick].astype(np.float32)
    zimg = np.full((ny, nx), np.nan, np.float32)
    zimg.ravel()[uniq] = ztop

    valid = np.isfinite(zimg).astype(np.uint8)
    if valid.sum() > 100:
        filled = zimg.copy()
        nan = ~np.isfinite(zimg)
        if nan.any() and (~nan).any():
            filled[nan] = float(np.nanmedian(zimg))
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
        filled = cv2.medianBlur(filled, 3)
        filled[valid == 0] = np.nan
        zimg = filled
    return DSM(x0=float(x0), y0=float(y0), gsd=gsd, z=zimg)


def build_from_da3_depth(ds, npz, sim3_vec_gt, kfs, da3_npz, gsd: float = 0.4,
                         percentile: float = 95.0, stride: int = 4,
                         max_relief: float = 60.0) -> Optional[DSM]:
    """DSM from Depth-Anything-3 depth, **anchored per frame to the SLAM depth**.

    DA3's absolute scale is unreliable here (measured 2-4.5x frame-to-frame
    drift), but its *relative* structure is all we need.  For every keyframe we
    fit one scalar ``s_k`` so that the median DA3 depth equals the median
    MASt3R point-map depth (which is metrically consistent), then lift the
    pixels with the metric camera pose.
    """
    from stream_ortho.geometry import compose_sim3, sim3_from_vec8

    d = np.load(da3_npz)
    depth = d["depth"]
    fids = np.asarray(d["frame_id"]).astype(np.int64)
    idx_of = {int(f): n for n, f in enumerate(fids)}
    Hd, Wd = depth.shape[1:]
    H, W = ds.cfg.height, ds.cfg.width
    K = ds.K_orig
    Kinv = np.linalg.inv(K)
    uu, vv = np.meshgrid(np.arange(Wd, dtype=np.float32), np.arange(Hd, dtype=np.float32))
    u_img = (uu + 0.5) * (W / Wd)
    v_img = (vv + 0.5) * (H / Hd)
    rays = np.stack([u_img, v_img, np.ones_like(u_img)], -1) @ Kinv.T   # (Hd,Wd,3)
    rays = rays[stride // 2::stride, stride // 2::stride].reshape(-1, 3)
    N = ds.T_WP[:3, 2]

    xs, ys, zs = [], [], []
    for k in kfs:
        fid = int(npz["frame_id"][k])
        if fid not in idx_of:
            continue
        dk = np.asarray(depth[idx_of[fid]], np.float32)[stride // 2::stride, stride // 2::stride].reshape(-1)
        ok = np.isfinite(dk) & (dk > 1e-6)
        # anchor: median depth of the SLAM point map for this keyframe (metric)
        X = np.asarray(npz["X_canon"][k], np.float32).reshape(-1, 3)
        R, t, sc = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
        d_slam = float(np.median(X[:, 2])) * float(sc)      # metres along the optical axis
        s_k = d_slam / float(np.median(dk[ok])) if ok.any() else None
        if s_k is None or not np.isfinite(s_k):
            continue
        depth_metric = dk * s_k
        Pc = rays * depth_metric[:, None]                   # (N,3) camera frame, metres
        Pw = Pc @ R.T + t
        Pp = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        zrel = Pp[:, 2] - np.median(Pp[:, 2])
        good = np.abs(zrel) < max_relief
        xs.append(Pp[good, 0]); ys.append(Pp[good, 1]); zs.append(Pp[good, 2])
    if not xs:
        return None
    return _rasterise(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                      gsd, percentile)


def build_from_keyframes(ds, npz, sim3_vec_gt, kfs, gsd: float = 0.4,
                         percentile: float = 95.0, close_px: int = 3,
                         height_max: float = 80.0, stride: int = 2) -> Optional[DSM]:
    """Rasterise the lifted SLAM point maps into a top-surface height field."""
    import cv2

    from stream_ortho.geometry import compose_sim3, sim3_from_vec8, sim3_matrix

    xs, ys, zs = [], [], []
    n = ds.T_WP[:3, 2]
    for k in kfs:
        X = np.asarray(npz["X_canon"][k], np.float32)[::stride, ::stride].reshape(-1, 3)
        if not np.isfinite(X).all():
            X = np.nan_to_num(X)
        # NB: sim3_from_vec8 expects the 8-vector, not the 4x4 matrix
        R, t, s = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
        # X_canon is in the keyframe's camera frame (SLAM units); the est->GT Sim3
        # already carries the metric scale, so use it globally (per-keyframe
        # renormalisation was tried and *spread* the cloud to kilometres).
        Pw = float(s) * (X @ R.T) + t
        Pp = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        xs.append(Pp[:, 0]); ys.append(Pp[:, 1]); zs.append(Pp[:, 2])
    if not xs:
        return None
    return _rasterise(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                      gsd, percentile, close_px, height_max)


# ---------------------------------------------------------------------------
def fill_dsm_holes(h: np.ndarray, valid: np.ndarray, ground: float,
                   iters: int = 12, close_px: int = 5) -> np.ndarray:
    """Fill the DSM's NaN holes (P0-3).

    ``build_piecewise_dsm`` writes only the ground median and roof components >= 40
    cells and never fills the gaps, so **73%** of the raster is NaN.  The old
    ``_rasterise`` filled them (``nan->median`` + ``MORPH_CLOSE`` + ``medianBlur``); the
    piecewise rewrite dropped that, and ``lookup(fill=0.0)`` then treated every hole as
    ``z=0`` while the real ground sits at ``-2.39 m`` -- which split the ground into two
    layers and made the "chessboard" lifting.
    """
    import cv2

    out = np.asarray(h, np.float32).copy()
    ok = np.asarray(valid, bool) & np.isfinite(out)
    if not ok.any():
        return np.full_like(out, ground)
    hole = ~ok
    for _ in range(iters):
        if not hole.any():
            break
        k = np.ones((3, 3), np.float32)
        s = cv2.filter2D(np.where(ok, out, 0.0), -1, k, borderType=cv2.BORDER_REPLICATE)
        c = cv2.filter2D(ok.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
        fillable = hole & (c > 0.5)
        out[fillable] = s[fillable] / c[fillable]
        ok = ok | fillable
        hole = ~ok
    if hole.any():
        out[hole] = float(ground)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    out = cv2.medianBlur(out.astype(np.float32), 3)
    return out.astype(np.float32)


def build_piecewise_dsm(ds, npz, sim3_vec_gt, kfs, gsd: float = 0.5,
                        roof_min: float = 4.0, min_cells: int = 40,
                        stride: int = 2, max_components: int = 400) -> Optional[DSM]:
    """**Piecewise-planar** DSM: every roof (and the ground) is fitted with a plane.

    The per-cell-quantile DSM keeps the buildings but its roof boundary is fuzzy and
    its heights are stripy, which is why the ortho still smears facades.  Here we

    1. rasterise the lifted point map into a median height per cell,
    2. split cells into ground (``h < roof_min``) and roofs (``h >= roof_min``),
    3. take connected components of the roof mask and fit **one plane per component**
       (plus one robust plane for the ground),
    4. write the fitted heights back -> flat roofs, crisp boundaries, smooth ground.

    Heights are in the plane frame (metres above the ground plane).
    """
    import cv2

    from stream_ortho.geometry import compose_sim3, sim3_from_vec8

    xs, ys, zs = [], [], []
    for k in kfs:
        X = np.asarray(npz["X_canon"][k], np.float32)[::stride, ::stride].reshape(-1, 3)
        R, t, sc = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
        Pw = float(sc) * (X @ R.T) + t
        Pp = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        ok = np.isfinite(Pp).all(1) & (Pp[:, 2] > -20.0) & (Pp[:, 2] < 80.0)
        xs.append(Pp[ok, 0]); ys.append(Pp[ok, 1]); zs.append(Pp[ok, 2])
    if not xs:
        return None
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    x0, x1 = np.percentile(x, [0.5, 99.5])
    y0, y1 = np.percentile(y, [0.5, 99.5])
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    x, y, z = x[keep], y[keep], z[keep]
    nx = int(np.ceil((x1 - x0) / gsd)) + 1
    ny = int(np.ceil((y1 - y0) / gsd)) + 1
    i = np.clip(((x - x0) / gsd).astype(np.int64), 0, nx - 1)
    j = np.clip(((y - y0) / gsd).astype(np.int64), 0, ny - 1)
    cell = (j * nx + i).astype(np.int64)
    hmed = np.full(ny * nx, np.nan, np.float32)
    order = np.lexsort((z, cell))
    uniq, start, counts = np.unique(cell[order], return_index=True, return_counts=True)
    zs_sorted = z[order]
    pick = start + (counts - 1) // 2                       # median
    hmed[uniq] = zs_sorted[pick]
    hmed = hmed.reshape(ny, nx)

    valid = np.isfinite(hmed)
    if valid.sum() < 200:
        return None
    h = hmed.copy()
    h[~valid] = float(np.nanmedian(hmed))

    # ---- ground: one robust plane (the plane frame is already ~ground aligned) ----
    gm = valid & (h < roof_min)
    if gm.sum() > 200:
        # the ground is *horizontal by definition* in the plane frame; a free plane fit
        # absorbs the residual tilt of the est->GT Sim3 and makes the whole DSM slope
        plane_g = np.full_like(h, float(np.median(h[gm])))
    else:
        plane_g = np.zeros_like(h)

    # ---- roofs: connected components, one fitted plane each ----
    out = np.full((ny, nx), np.nan, np.float32)
    out[gm] = plane_g[gm].astype(np.float32)
    rm = (valid & (h >= roof_min)).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(rm, connectivity=8)
    n_roof = 0
    if n_lab - 1 > max_components:      # keep the largest components, stay bounded
        areas = np.bincount(lab.ravel())[1:]
        order_lb = np.argsort(areas)[::-1][:max_components] + 1
    else:
        order_lb = np.arange(1, n_lab)
    for lb in order_lb:
        m = lab == lb
        if m.sum() < min_cells:
            continue
        jj, ii = np.nonzero(m)
        A = np.stack([ii, jj, np.ones(m.sum())], 1).astype(np.float64)
        try:
            coef, *_ = np.linalg.lstsq(A, h[m].astype(np.float64), rcond=None)
        except np.linalg.LinAlgError:
            continue
        fit = coef[0] * ii + coef[1] * jj + coef[2]
        if np.abs(fit - h[m]).mean() > 3.0:               # tilted/vegetation: keep median
            fit = np.full(m.sum(), float(np.median(h[m])))
        out[m] = fit.astype(np.float32)
        n_roof += 1
    # P0-3: fill the remaining holes with a neighbourhood median and a light close, then
    # a median blur; anything still empty becomes the **ground** height (not z=0)
    ground_level = float(np.median(plane_g[gm])) if gm.any() else -2.4
    filled = fill_dsm_holes(out, np.isfinite(out), ground_level)
    return DSM(x0=float(x0), y0=float(y0), gsd=gsd, z=filled)
