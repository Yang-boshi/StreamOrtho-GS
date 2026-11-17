"""Texture-preserving orthophoto: backward warp of full-resolution images onto the product.

The product stores one colour per 0.45 m cell, so rendering it at 0.21 m can only up-sample a
blurred image.  This module keeps the **calibrated frozen geometry** (the product's top
surface) and takes the colour from the **original frames**: for every output pixel it finds
the surface height, projects that 3-D point into a few candidate frames, rejects frames where
the point is occluded (a per-frame depth buffer) and samples the image bilinearly.

Nothing here changes the geometry or the calibration: it is a product renderer.

``FrameIO`` owns the per-frame geometry (Sim3 + intrinsics), the depth buffers built from the
observation table, and the image cache.
"""
from __future__ import annotations

import pathlib

import numpy as np


def _sim3_from_vec8(v):
    from stream_ortho.geometry import sim3_from_vec8
    return sim3_from_vec8(v)


class FrameIO:
    """Per-frame geometry, depth buffer and image access.

    Parameters
    ----------
    kf_npz : the keyframes archive (``T_WC``, ``K``, ``timestamp``, ``frame_id``)
    obs    : the observation archive (``T_WP``, ``kf``, ``uv``, ``D``)
    image_dir : directory of ``<timestamp>.jpg``
    sim3   : the Sim3 reference vector used when the observation table was built
    depth_hw : resolution of the per-frame depth buffer (small on purpose)
    """

    def __init__(self, kf_npz, obs, image_dir, sim3, depth_hw=(270, 480), scale_img=None):
        from stream_ortho.geometry import compose_sim3

        self.npz = kf_npz
        self.T_WP = np.asarray(obs["T_WP"], np.float64)
        self.sim3 = np.asarray(sim3, np.float64)
        self.dir = pathlib.Path(image_dir)
        self.depth_hw = tuple(depth_hw)
        self.scale_img = scale_img
        self.kfs = sorted(set(int(k) for k in np.unique(obs["kf"])))
        self._img = {}

        # per-frame camera (R, t, s) in the SLAM frame and the plane-frame centre
        self.R, self.t, self.s, self.centre = {}, {}, {}, {}
        for k in self.kfs:
            R, t, s = _sim3_from_vec8(compose_sim3(self.npz["T_WC"][k], self.sim3))
            self.R[k] = np.asarray(R, np.float64)
            self.t[k] = np.asarray(t, np.float64)
            self.s[k] = float(s)
            self.centre[k] = np.asarray((self.t[k] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3],
                                        np.float64)

        # depth buffers: min metric distance per buffer pixel, from the observation table
        H, W = self.depth_hw
        self.depth = {k: np.full((H, W), np.inf, np.float32) for k in self.kfs}
        kf = np.asarray(obs["kf"], np.int32)
        uv = np.asarray(obs["uv"], np.float32)
        D = np.asarray(obs["D"], np.float32)
        px = np.clip(((uv[:, 0] + 1.0) * 0.5 * (W - 1)).astype(np.int64), 0, W - 1)
        py = np.clip(((uv[:, 1] + 1.0) * 0.5 * (H - 1)).astype(np.int64), 0, H - 1)
        flat = py * W + px
        for k in self.kfs:
            m = kf == k
            if not m.any():
                continue
            buf = np.full(H * W, np.inf, np.float32)
            np.minimum.at(buf, flat[m], D[m])
            self.depth[k] = buf.reshape(H, W)

        self.K = np.asarray(self.npz["K"], np.float64)
        pw, ph = self.K[0, 2] * 2, self.K[1, 2] * 2      # pointmap width/height from the K cx,cy
        self.img_scale = (1920.0 / pw) if self.scale_img is None else float(self.scale_img)

    # ------------------------------------------------------------------
    def image(self, k):
        if k not in self._img:
            from PIL import Image
            ts = float(self.npz["timestamp"][k])
            p = self.dir / f"{ts:.6f}.jpg"
            im = Image.open(p).convert("RGB")
            self._img[k] = np.asarray(im, np.float32) / 255.0
        return self._img[k]

    def project(self, k, P):
        """Plane points ``P`` (N,3) -> image pixels (u,v) at the *original* resolution, distance d."""
        Pw = P @ self.T_WP[:3, :3].T + self.T_WP[:3, 3]
        X = (Pw - self.t[k]) @ self.R[k]
        d = np.linalg.norm(X, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Xs = X * self.s[k]
            u = self.K[0, 0] * Xs[:, 0] / Xs[:, 2] + self.K[0, 2]
            v = self.K[1, 1] * Xs[:, 1] / Xs[:, 2] + self.K[1, 2]
        return u * self.img_scale, v * self.img_scale, d

    def depth_at(self, k, u, v):
        """Depth-buffer value at the *original-resolution* pixel (u,v)."""
        H, W = self.depth_hw
        im = self.image(k)
        iw, ih = im.shape[1], im.shape[0]
        px = np.clip((u / max(iw - 1, 1) * (W - 1)).astype(np.int64), 0, W - 1)
        py = np.clip((v / max(ih - 1, 1) * (H - 1)).astype(np.int64), 0, H - 1)
        return self.depth[k][py, px]


def _bilinear(img, u, v):
    """Bilinear sample; returns (rgb, valid) with valid=False outside the image."""
    h, w = img.shape[:2]
    ok = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    c00 = img[v0, u0]
    c01 = img[v0, np.minimum(u0 + 1, w - 1)]
    c10 = img[np.minimum(v0 + 1, h - 1), u0]
    c11 = img[np.minimum(v0 + 1, h - 1), np.minimum(u0 + 1, w - 1)]
    return (c00 * (1 - du) * (1 - dv) + c01 * du * (1 - dv) +
            c10 * (1 - du) * dv + c11 * du * dv), ok


def top_surface_at(product, x, y):
    """Nearest-cell top height of the frozen product."""
    ix = np.clip(np.floor(x / product["cell"]).astype(np.int64) - product["ox"], 0,
                 product["W"] - 1)
    iy = np.clip(np.floor(y / product["cell"]).astype(np.int64) - product["oy"], 0,
                 product["H"] - 1)
    occ = product["n"][iy, ix] > 0
    z = np.where(occ, product["z_ref"][iy, ix], -np.inf).max(-1)
    return z, occ.any(-1)


def render_texture_tile(product, fio, grid, i, j, gsd=0.21, tau=0.5, max_frames=3,
                        owner_only=False):
    """Render one output tile by backward-warping the original frames.

    Returns ``(rgb, conf, hole, info)``: ``rgb`` float32 ``(h,w,3)`` in [0,1], ``conf`` the
    number of frames that passed the depth test (normalised), ``hole`` where nothing was
    visible.
    """
    h, w = int(grid["h"]), int(grid["w"])
    xs = grid["x0"] + (np.arange(w) + 0.5) * gsd
    ys = grid["y0"] + (np.arange(h) + 0.5) * gsd
    X, Y = np.meshgrid(xs, ys)
    xf, yf = X.ravel(), Y.ravel()

    # --- product top surface + owner at those pixels ---
    cx = np.clip((xf // product["cell"]).astype(np.int64) - product["ox"], 0, product["W"] - 1)
    cy = np.clip((yf // product["cell"]).astype(np.int64) - product["oy"], 0, product["H"] - 1)
    n_l = product["n"][cy, cx]                      # (N, L)
    z_l = product["z_ref"][cy, cx]
    occ = n_l > 0
    live = occ.any(-1)
    idx_top = np.where(occ, z_l, -np.inf).argmax(-1)
    z = z_l[np.arange(live.size), idx_top]
    owner_top = product["owner"][cy, cx][np.arange(live.size), idx_top]

    P = np.stack([xf, yf, np.where(live, z, 0.0)], 1)

    # --- candidate frames: the cell owner, plus the nearest cameras ---
    centres = np.stack([fio.centre[k] for k in fio.kfs])
    d_cam = np.linalg.norm(centres[:, None, :] - P[None, :, :], axis=2)   # (F, N)
    order = np.argsort(d_cam, axis=0)
    rank = np.empty_like(order)
    rank[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[0])[:, None]

    acc = np.zeros((live.size, 3), np.float64)
    wsum = np.zeros(live.size, np.float64)
    for pos, k in enumerate(fio.kfs):
        if owner_only:
            m = live & (wsum == 0) & (owner_top == k)
        else:
            m = live & (wsum < max_frames) & ((owner_top == k) | (rank[pos] < max_frames))
        if not m.any():
            continue
        u, v, d = fio.project(k, P[m])
        img = fio.image(k)
        inb = (u >= 0) & (v >= 0) & (u <= img.shape[1] - 1) & (v <= img.shape[0] - 1)
        good = np.zeros(int(m.sum()), bool)
        if inb.any():
            dbuf = fio.depth_at(k, u[inb], v[inb])
            vis = np.abs(dbuf - d[inb]) <= tau
            good[np.nonzero(inb)[0][vis]] = True
        if good.any():
            rgb, _ = _bilinear(img, u[good], v[good])
            sel = np.nonzero(m)[0][good]
            acc[sel] += rgb
            wsum[sel] += 1.0

    hole = (wsum == 0) | ~live
    conf = np.minimum(wsum / max(max_frames, 1), 1.0)
    rgb = np.zeros((live.size, 3), np.float32)
    g = wsum > 0
    rgb[g] = (acc[g] / wsum[g][:, None]).astype(np.float32)
    info = {"tile": [int(i), int(j)], "n_pixels": int(live.size),
            "frac_hole": float(hole.mean()),
            "frac_multi_frame": float((wsum > 1).mean())}
    return rgb.reshape(h, w, 3), conf.reshape(h, w), hole.reshape(h, w), info


def edge_width_10_90(img, min_grad_pct=99.0, max_px=7, mask=None):
    """Median 10-90% transition width (in pixels) over the strongest vertical edges.

    A cheap, reproducible proxy for the effective GSD of an orthophoto: take the pixels whose
    horizontal luminance gradient is in the top 1%, walk left/right along the row to the 10%
    and 90% levels of the local step and report the distance.
    """
    g = img.mean(-1) if img.ndim == 3 else img
    dx = np.abs(np.diff(g, axis=1))
    thr = np.percentile(dx, min_grad_pct)
    sel = dx >= thr
    if mask is not None:
        sel &= mask[:, :-1] & mask[:, 1:]
    rows, cols = np.nonzero(sel)
    widths = []
    for r, c in zip(rows[:: max(len(rows) // 4000, 1)], cols[:: max(len(rows) // 4000, 1)]):
        lo, hi = c, c + 1
        a, b = g[r, max(c - max_px, 0)], g[r, min(c + 1 + max_px, g.shape[1] - 1)]
        if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-3:
            continue
        step_lo, step_hi = min(a, b), max(a, b)
        t10 = step_lo + 0.1 * (step_hi - step_lo)
        t90 = step_lo + 0.9 * (step_hi - step_lo)
        seg = g[r, max(c - max_px, 0): min(c + 1 + max_px, g.shape[1])]
        if seg.size < 3:
            continue
        try:
            i10 = int(np.argmax(seg >= t10)) if b > a else int(np.argmax(seg <= t90))
            i90 = int(np.argmax(seg >= t90)) if b > a else int(np.argmax(seg <= t10))
        except ValueError:
            continue
        # a perfect step sampled on the output grid is one pixel wide: max(...,1) not +1
        wid = max(abs(i90 - i10), 1)
        if 0 < wid <= 2 * max_px:
            widths.append(wid)
    return float(np.median(widths)) if widths else float("nan"), len(widths)
