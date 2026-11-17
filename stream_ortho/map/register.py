"""Inter-frame ground registration (week8-end-1 task 2).

The streaming map places every keyframe's ground points with the est->GT Sim3,
but the per-keyframe residual is 0.5-2 m (measured in weeks 3/5/7: 0.8-18 px at
0.2 m/px).  When several keyframes are merged into one voxel map these residuals
show up as ghosting/doubling in the ortho.

This module estimates a **per-keyframe 2-D offset in the ground plane** by

1. warping each keyframe's image onto the plane at a coarse GSD (a plane
   homography, so a single ``cv2.warpPerspective`` per frame);
2. phase-correlating every canvas against an anchor canvas;
3. returning the median-consistent offsets in metres.

The offsets are then applied when the patch is built (``plane_patch`` gets a
``world_offset``), i.e. the geometry is corrected *before* it enters the map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RegisterConfig:
    gsd: float = 0.5            # coarse canvas resolution [m/px]
    min_response: float = 0.05  # phase-correlation confidence
    max_offset_m: float = 8.0   # reject implausible shifts
    anchor: int = -1            # kf index used as the anchor (-1 = middle)


def _plane_homography(T_gt_cam: np.ndarray, K: np.ndarray, W: int, H: int,
                      T_WP: np.ndarray, z_ground: float, gsd: float,
                      origin_xy: np.ndarray) -> Optional[np.ndarray]:
    """Image pixel -> canvas pixel homography (or None if degenerate)."""
    n = T_WP[:3, 2]
    p0 = T_WP[:3, 3] + n * float(z_ground)
    uv = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    dst = []
    for u, v in uv:
        d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        d_w = T_gt_cam[:3, :3] @ d_cam
        den = float(d_w @ n)
        if abs(den) < 1e-9:
            return None
        t = float((p0 - T_gt_cam[:3, 3]) @ n) / den
        if t <= 0:
            return None
        pW = T_gt_cam[:3, 3] + t * d_w
        pP = (pW - T_WP[:3, 3]) @ T_WP[:3, :3]
        dst.append([(pP[0] - origin_xy[0]) / gsd, (pP[1] - origin_xy[1]) / gsd])
    import cv2

    Hm = cv2.getPerspectiveTransform(uv.astype(np.float32), np.asarray(dst, np.float32))
    return Hm


def estimate_offsets(ds, npz, sim3_vec_gt, kfs: List[int], cfg: Optional[RegisterConfig] = None,
                     verbose: bool = True) -> Dict[int, Tuple[float, float]]:
    """Per-keyframe 2-D ground offset (metres, plane frame) w.r.t. the anchor."""
    import cv2

    from stream_ortho.geometry import compose_sim3, sim3_from_vec8

    cfg = cfg or RegisterConfig()
    K = ds.K_orig
    W, H = int(ds.cfg.width), int(ds.cfg.height)
    T_WP = ds.T_WP

    # common canvas: union of the footprints (plane frame)
    boxes = []
    for k in kfs:
        T = compose_sim3(npz["T_WC"][k], sim3_vec_gt)
        Tm = np.eye(4)
        R, t, s = sim3_from_vec8(T)
        Tm[:3, :3] = s * R
        Tm[:3, 3] = t
        pts = []
        for u in (0, W):
            for v in (0, H):
                d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
                dw = Tm[:3, :3] @ d
                n = T_WP[:3, 2]
                den = float(dw @ n)
                tt = float((T_WP[:3, 3] - Tm[:3, 3]) @ n) / den
                pW = Tm[:3, 3] + tt * dw
                pts.append((pW - T_WP[:3, 3]) @ T_WP[:3, :3])
        P = np.asarray(pts)
        boxes.append((P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()))
    boxes = np.asarray(boxes)
    lo = boxes[:, :2].min(axis=0) - 5.0
    hi = boxes[:, 2:].max(axis=0) + 5.0
    cw = int(np.ceil((hi[0] - lo[0]) / cfg.gsd))
    ch = int(np.ceil((hi[1] - lo[1]) / cfg.gsd))

    canv = {}
    for k in kfs:
        T = compose_sim3(npz["T_WC"][k], sim3_vec_gt)
        R, t, s = sim3_from_vec8(T)
        Tm = np.eye(4)
        Tm[:3, :3] = s * R
        Tm[:3, 3] = t
        Hm = _plane_homography(Tm, K, W, H, T_WP, 0.0, cfg.gsd, lo)
        if Hm is None:
            continue
        img = ds.read_img(int(npz["frame_id"][k]))
        warp = cv2.warpPerspective(img, Hm, (cw, ch), flags=cv2.INTER_AREA)
        g = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        canv[k] = g

    if not canv:
        return {}
    order = [k for k in kfs if k in canv]
    ref_any = next(iter(canv.values()))

    def shift(a, b):
        """(dx,dy) metres such that b must move by d to match a (or None)."""
        win = ((a > 0.02) & (b > 0.02)).astype(np.float32)
        if win.sum() < 20000:
            return None
        am = a[win > 0].mean(); bm = b[win > 0].mean()
        aa = np.where(win > 0, a - am, 0.0).astype(np.float32)
        bb = np.where(win > 0, b - bm, 0.0).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(aa, bb)
        if resp < cfg.min_response:
            return None
        dx_m, dy_m = dx * cfg.gsd, dy * cfg.gsd
        if max(abs(dx_m), abs(dy_m)) > cfg.max_offset_m:
            return None
        return (float(dx_m), float(dy_m), float(resp))

    # consecutive pairwise shifts (robust: adjacent keyframes overlap a lot)
    pair = {}
    for i in range(len(order) - 1):
        d = shift(canv[order[i]], canv[order[i + 1]])
        if d is not None:
            pair[(order[i], order[i + 1])] = d
            if verbose:
                print(f"    [reg] kf{order[i]:3d}->kf{order[i+1]:3d} shift=({d[0]:+.2f},{d[1]:+.2f}) m resp={d[2]:.3f}")

    anchor = cfg.anchor if cfg.anchor >= 0 and cfg.anchor in order else order[len(order) // 2]
    offsets: Dict[int, Tuple[float, float]] = {anchor: (0.0, 0.0)}
    # chain forward
    for i in range(order.index(anchor), len(order) - 1):
        a, b = order[i], order[i + 1]
        d = pair.get((a, b))
        if d is None:
            offsets[b] = offsets[a]
        else:
            offsets[b] = (offsets[a][0] + d[0], offsets[a][1] + d[1])
    # chain backward
    for i in range(order.index(anchor), 0, -1):
        a, b = order[i - 1], order[i]
        d = pair.get((a, b))
        if d is None:
            offsets[a] = offsets[b]
        else:
            offsets[a] = (offsets[b][0] - d[0], offsets[b][1] - d[1])

    if verbose:
        for k in order:
            ox, oy = offsets[k]
            print(f"    [reg] kf{k:3d} accumulated offset ({ox:+.2f}, {oy:+.2f}) m")
    return offsets
