"""Ground-plane reference orthomosaic from **GT poses** (design doc S4 / §7.4 item 4).

`phantom3-centralPark-kfs/result.png` (the only real orthomosaic reference) is
no longer available, and there is no GCP-accurate ortho for these sequences, so
the P1 gate uses this **plane-warp mosaic**: every image is warped onto the
estimated ground plane with its GT pose and blended with a border-feather
weight.  On flat ground this is exact; buildings are smeared by construction
(true ortho needs a DSM).

Why it is a fair reference: it uses exactly the same plane, GSD and tile grid
as the Gaussian renderer, so a mismatch isolates *our* geometry, not the
reference's convention.

Weighting: ``w = clip(min(u, v, W-1-u, H-1-v) / (0.12*min(W,H)), 0, 1)`` -- the
image centre wins, the borders fade out (less parallax / lens softness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.data.plane import world_to_plane
from stream_ortho.ortho.grid import OrthoGrid, footprint_bbox


@dataclass
class FrameFootprint:
    idx: int
    T_WC: np.ndarray                 # 4x4, camera -> GT world
    bbox: Tuple[float, float, float, float]   # plane coords
    nadir: Tuple[float, float] = (0.0, 0.0)   # camera centre projected to the plane


class PlaneReferenceMosaic:
    """Warp GT-posed images onto the plane and average them per tile."""

    def __init__(self, dataset: NPUDroneMapDataset, gsd: float, z_ground: float,
                 stride: int = 1, feather_frac: float = 0.12,
                 frame_indices: Optional[List[int]] = None,
                 poses: str = "gt", slam_poses: Optional[dict] = None):
        self.ds = dataset
        self.gsd = float(gsd)
        self.z_ground = float(z_ground)
        self.feather_frac = float(feather_frac)
        self.poses = poses
        self.slam_poses = slam_poses or {}   # frame_id -> 4x4 camera->GT world
        self.K = dataset.K_orig
        self.W = int(dataset.cfg.width)
        self.H = int(dataset.cfg.height)

        self.frames: List[FrameFootprint] = []
        n = len(dataset)
        idxs = frame_indices if frame_indices is not None else range(0, n, max(1, stride))
        for idx in idxs:
            if self.poses == "slam":
                T = self.slam_poses.get(int(idx))
            else:
                T = dataset.gt_pose(int(idx))
            if T is None:
                continue
            bbox = footprint_bbox(T, self.K, self.W, self.H, dataset.T_WP, self.z_ground)
            if bbox is None:
                continue
            c = (T[:3, 3] - dataset.T_WP[:3, 3]) @ dataset.T_WP[:3, :3]
            self.frames.append(FrameFootprint(int(idx), T, bbox, (float(c[0]), float(c[1]))))
        if not self.frames:
            raise RuntimeError("没有可用的 GT 位姿帧")

    # ------------------------------------------------------------------
    def frames_for_tile(self, grid: OrthoGrid, i: int, j: int,
                        max_frames: int = None, select: str = "nadir") -> List[FrameFootprint]:
        """Frames whose footprint intersects the tile.

        ``max_frames`` keeps the ``select`` best ones: ``nadir`` = camera centre
        closest to the tile centre (sharpest single-view warp, least parallax),
        ``id`` = closest in time to the tile centre frame.
        """
        x0, y0, x1, y1 = grid.tile_window(i, j)
        cand = [f for f in self.frames
                if f.bbox[0] <= x1 and f.bbox[2] >= x0 and f.bbox[1] <= y1 and f.bbox[3] >= y0]
        if max_frames is None or len(cand) <= max_frames:
            return cand
        cx, cy = grid.tile_xy(i, j)
        if select == "nadir":
            cand.sort(key=lambda f: (f.nadir[0] - cx) ** 2 + (f.nadir[1] - cy) ** 2)
        else:
            cand.sort(key=lambda f: abs(f.idx - self.frames[len(self.frames) // 2].idx))
        return cand[:max_frames]

    def render_tile(self, grid: OrthoGrid, i: int, j: int, read_scale: int = 1,
                    max_frames: int = 3):
        """Return ``(rgb, coverage, n_frames, weight)`` for tile ``(i, j)``."""
        import cv2

        S = grid.tile
        frames = self.frames_for_tile(grid, i, j, max_frames=max_frames)
        acc = np.zeros((S, S, 3), np.float32)
        wsum = np.zeros((S, S, 1), np.float32)
        n_used = 0

        # tile pixels -> plane points (one shared grid)
        u, v = np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64))
        pP = grid.px_to_plane(u, v, i, j, z=self.z_ground).reshape(-1, 3)
        pW = pP @ grid.T_WP[:3, :3].T + grid.T_WP[:3, 3]

        for f in frames:
            T_CW = np.linalg.inv(f.T_WC)
            pC = pW @ T_CW[:3, :3].T + T_CW[:3, 3]
            z = pC[:, 2].reshape(S, S)
            ok = z > 1e-6
            if ok.sum() < 100:
                continue
            uv = pC.reshape(-1, 3) @ self.K.T
            uu = (uv[:, 0] / np.maximum(uv[:, 2], 1e-9)).reshape(S, S)
            vv = (uv[:, 1] / np.maximum(uv[:, 2], 1e-9)).reshape(S, S)
            inside = ok & (uu >= 0) & (uu < self.W - 1) & (vv >= 0) & (vv < self.H - 1)
            if inside.sum() < 100:
                continue
            img = cv2.imread(str(self.ds.rgb_files[f.idx]))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if read_scale > 1:
                img = img[::read_scale, ::read_scale]
                uu = uu / read_scale
                vv = vv / read_scale
                h, w = img.shape[:2]
            else:
                h, w = self.H, self.W
            samp = cv2.remap(img, uu.astype(np.float32), vv.astype(np.float32),
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # border feather in image space
            d = np.minimum(np.minimum(uu, vv), np.minimum(w - 1 - uu, h - 1 - vv))
            lim = max(1.0, self.feather_frac * min(w, h))
            wgt = np.clip(d / lim, 0.0, 1.0) * inside
            acc += samp * wgt[..., None]
            wsum += wgt[..., None]
            n_used += 1

        cov = (wsum[..., 0] > 1e-3).astype(np.float32)
        rgb = acc / np.maximum(wsum, 1e-6)
        return rgb.astype(np.float32), cov, n_used, wsum[..., 0]

    def render_block(self, grid: OrthoGrid, i0: int, j0: int, rows: int, cols: int,
                     max_frames: int = 3):
        """Render a ``rows x cols`` tile block; returns stitched ``(rgb, cov, info)``."""
        S = grid.tile
        out = np.zeros((rows * S, cols * S, 3), np.float32)
        cov = np.zeros((rows * S, cols * S), np.float32)
        info = {}
        for dj in range(rows):
            for di in range(cols):
                i, j = i0 + di, j0 + dj
                rgb, c, n, _ = self.render_tile(grid, i, j, max_frames=max_frames)
                r = (rows - 1 - dj) * S
                cc = di * S
                out[r:r + S, cc:cc + S] = rgb
                cov[r:r + S, cc:cc + S] = c
                info[f"{i},{j}"] = n
        return out, cov, info


# ---------------------------------------------------------------------------
# Ground level estimation
# ---------------------------------------------------------------------------

def estimate_ground_level(points_P: List[np.ndarray], percentile: float = 5.0) -> float:
    """Ground z in the plane frame from point-cloud z quantiles (design doc S1).

    ``points_P`` are point clouds already expressed in the plane frame; the
    ground is the low quantile of the pooled distribution.
    """
    if not points_P:
        raise ValueError("no clouds")
    z = np.concatenate([np.asarray(p)[:, 2].reshape(-1) for p in points_P])
    z = z[np.isfinite(z)]
    return float(np.percentile(z, percentile))


def cloud_in_plane(T_WP: np.ndarray, pts_W: np.ndarray) -> np.ndarray:
    return world_to_plane(T_WP, pts_W)
