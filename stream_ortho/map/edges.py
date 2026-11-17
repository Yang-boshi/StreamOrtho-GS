"""Edge extraction and edge-aware Gaussian shapes (week-5 experiment).

Motivation (user's idea, 2026-10-25): ortho rendering blurs **building / object
edges** because a fixed budget and an isotropic, flat splat cannot represent a
thin line.  Proposal:

1. detect the ground "line elements" in each keyframe;
2. **keep 100% of the Gaussians on edges**, keep only a ratio of the "area"
   (face) Gaussians -- so the budget follows the scene content instead of a
   fixed 300k or a fixed 25/m^2;
3. tag each Gaussian as *edge* or *area* and give it the matching **shape
   prior**: elongated along the edge for line elements, flat/disc-like for the
   area ones (the tag is stored in the map so a later fine-tune can enforce it).

This module only does (1), (3) and the tagging; the budget logic lives in
``map/scoring.py`` / ``GaussianMap.enforce_budget``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeConfig:
    blur: int = 3
    canny_low: int = 40
    canny_high: int = 120
    grad_percentile: float = 85.0   # fallback: keep the strongest gradient pixels
    dilate: int = 0                 # >0 widens the edge band


def edge_map(img_rgb: np.ndarray, cfg: EdgeConfig = None):
    """Return ``(kind, theta)`` for one image.

    ``kind``: 1 = edge (line element), 0 = area; ``theta``: edge orientation in
    the image plane [rad] (direction *along* the edge), NaN for area points.
    """
    import cv2

    cfg = cfg or EdgeConfig()
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if cfg.blur > 1:
        g = cv2.GaussianBlur(g, (cfg.blur | 1, cfg.blur | 1), 0)
    edges = cv2.Canny(g, cfg.canny_low, cfg.canny_high)
    if cfg.dilate > 0:
        edges = cv2.dilate(edges, np.ones((cfg.dilate, cfg.dilate), np.uint8))

    # gradient orientation via Scharr -> structure tensor
    gx = cv2.Scharr(g.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g.astype(np.float32), cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # edge direction = perpendicular to the gradient
    theta = np.arctan2(gx, -gy)

    if edges.sum() < 0.002 * edges.size:      # too few -> use a gradient threshold
        thr = np.percentile(mag, cfg.grad_percentile)
        edges = (mag >= thr).astype(np.uint8) * 255

    kind = (edges > 0).astype(np.int8)
    theta = theta.astype(np.float32)
    theta[kind == 0] = np.nan
    return kind, theta, mag


def edge_stats(kind: np.ndarray, theta: np.ndarray) -> dict:
    kind = np.asarray(kind).reshape(-1)
    n = kind.size
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "edge_count": int(kind.sum()),
        "edge_ratio": float(kind.mean()),
        "theta_valid_ratio": float(np.isfinite(theta).mean()),
    }


# ---------------------------------------------------------------------------
# Shape prior
# ---------------------------------------------------------------------------

def shape_from_kind(kind, theta, gsd: float, R_wc: np.ndarray,
                    plane_normal_world: np.ndarray, area_iso: float = 1.1,
                    edge_along: float = 1.8, edge_across: float = 0.55,
                    height_sigma: float = 0.35):
    """Per-point ``(scales, quats_wxyz)`` in the **world** frame.

    * area: disc  (``area_iso*gsd`` in-plane, ``height_sigma*gsd`` along the normal)
    * edge: rod   (``edge_along*gsd`` along the edge, ``edge_across*gsd`` across,
      ``height_sigma*gsd`` along the normal) -- the elongated prior for line elements.

    ``theta`` is the edge orientation in the *image*; ``R_wc`` is that keyframe's
    camera->world rotation, used to rotate the image direction onto the plane.
    """
    from stream_ortho.geometry import R_to_quat_wxyz_batch

    kind = np.asarray(kind).reshape(-1).astype(bool)
    th = np.asarray(theta, np.float64).reshape(-1)
    n_pts = kind.size
    gsd = float(gsd)
    R_wc = np.asarray(R_wc, np.float64)

    nz = np.asarray(plane_normal_world, np.float64)
    nz = nz / max(np.linalg.norm(nz), 1e-9)

    scales = np.zeros((n_pts, 3), np.float32)
    quats = np.zeros((n_pts, 4), np.float32)

    def basis(d_world):
        d_world = np.atleast_2d(d_world)
        x = d_world - (d_world @ nz)[:, None] * nz[None]
        nx = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.maximum(nx, 1e-9)
        y = np.cross(nz[None], x)
        y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)
        return np.stack([x, y, np.tile(nz, (len(x), 1))], axis=2)   # columns

    # area: any in-plane axis (use world +x projected) -- a disc has no preferred direction
    base = np.tile(np.array([1.0, 0.0, 0.0]), (n_pts, 1))
    R_area = basis(base)
    scales[~kind] = np.array([area_iso * gsd, area_iso * gsd, height_sigma * gsd], np.float32)
    if (~kind).any():
        quats[~kind] = R_to_quat_wxyz_batch(R_area[~kind]).astype(np.float32)

    if kind.any():
        a = np.nan_to_num(th[kind], nan=0.0)
        d_img = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)
        d_w = d_img @ R_wc.T
        R_edge = basis(d_w)
        scales[kind] = np.stack([np.full(len(a), edge_along * gsd),
                                 np.full(len(a), edge_across * gsd),
                                 np.full(len(a), height_sigma * gsd)], axis=1).astype(np.float32)
        quats[kind] = R_to_quat_wxyz_batch(R_edge).astype(np.float32)
    return scales, quats
