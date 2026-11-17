"""Ortho compositor: geometry proxy + image orthorectification + verify/fallback.

This is the v3 main path (design doc §4.6.1).  Instead of splatting the Gaussian
map, each ortho tile is produced by

1. selecting the best K frames covering the tile,
2. warping every frame's **full-resolution image** through a geometry proxy
   (plane ``z=0`` by default, or a DSM),
3. fusing the warps with view-quality weights,
4. **verifying** the result against the individual warps and **falling back** to
   the best single frame when fusion does not help.

Guarantee: a tile is never worse than the best single-frame warp, because step 4
only accepts the fused image when its sharpness is within ``sharpen_tol`` of the
best single warp AND its cross-frame agreement is not worse than the best single
warp by more than ``agree_tol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CompositorConfig:
    max_frames: int = 3
    sharpen_tol: float = 0.02       # fused sharpness >= (1-tol) * best single
    agree_tol: float = 0.05         # fused disagreement <= (1+tol) * best single
    feather_frac: float = 0.15      # weight falls to 0 at the image border
    min_valid_px: int = 20000


def _grad(img: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = g.astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))


class OrthoCompositor:
    def __init__(self, ds, gsd: float, z_ground: float = 0.0, dsm=None,
                 cfg: Optional[CompositorConfig] = None, poses: str = "gt"):
        self.ds = ds
        self.gsd = float(gsd)
        self.z_ground = float(z_ground)
        self.dsm = dsm
        self.cfg = cfg or CompositorConfig()
        self.K = ds.K_orig
        self.W = int(ds.cfg.width)
        self.H = int(ds.cfg.height)
        self.T_WP = ds.T_WP
        self.poses = poses
        self._frame_cache: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    def _frame_pose(self, fid: int, T_gt: Optional[Dict[int, np.ndarray]] = None):
        if T_gt is not None and fid in T_gt:
            return T_gt[fid]
        return self.ds.gt_pose(fid) if self.poses == "gt" else None

    def _warp_frame(self, fid: int, T_wc: np.ndarray, grid, i: int, j: int,
                    z_plane: Optional[np.ndarray] = None):
        """Orthorectify one frame into the tile grid (with halo, then cropped)."""
        import cv2

        S = grid.render_size
        h = grid.halo
        uu, vv = np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64))
        # pixel (in the *rendered* tile incl. halo) -> plane coords
        s = grid.tile * grid.gsd
        x = (i * grid.tile - h) * grid.gsd + (uu + 0.5) * grid.gsd
        y = (j * grid.tile + grid.tile + h) * grid.gsd - (vv + 0.5) * grid.gsd
        z = np.full_like(x, self.z_ground)
        if self.dsm is not None:
            z = self.dsm.lookup(x, y, fill=self.z_ground).astype(np.float64)
        Pw = np.stack([x, y, z], -1) @ self.T_WP[:3, :3].T + self.T_WP[:3, 3]
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        Pc = (Pw - t) @ R                      # world -> camera
        zc = Pc[..., 2]
        ok = zc > 1e-6
        zc_safe = np.where(ok, zc, 1.0)
        u = self.K[0, 0] * Pc[..., 0] / zc_safe + self.K[0, 2]
        v = self.K[1, 1] * Pc[..., 1] / zc_safe + self.K[1, 2]
        valid = ok & (u >= 0) & (u < self.W - 1) & (v >= 0) & (v < self.H - 1)
        img = self.ds.read_img(fid)
        sample = cv2.remap(img, u.astype(np.float32), v.astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        rgb = sample.astype(np.float32) / 255.0
        # feather: weight 1 in the interior, ->0 near the image border
        du = np.minimum(u, self.W - 1 - u) / self.W
        dv = np.minimum(v, self.H - 1 - v) / self.H
        r = np.minimum(du, dv) / max(self.cfg.feather_frac, 1e-6)
        wgt = np.clip(r, 0.0, 1.0)
        wgt = np.where(valid, wgt, 0.0)
        # crop the halo
        sl = (slice(h, h + grid.tile), slice(h, h + grid.tile))
        return rgb[sl], wgt[sl], valid[sl]

    # ------------------------------------------------------------------
    def select_frames(self, grid, i: int, j: int, frame_ids: List[int],
                      T_gt: Optional[Dict[int, np.ndarray]] = None) -> List[int]:
        """Frames whose footprint covers the tile, ordered by nadir distance."""
        from stream_ortho.frontend.keyframe_policy import bbox_from_pose

        x0, y0, x1, y1 = grid.tile_window(i, j)
        cx, cy = grid.tile_xy(i, j)
        scored = []
        for fid in frame_ids:
            T = self._frame_pose(fid, T_gt)
            if T is None:
                continue
            bb = bbox_from_pose(T, self.K, self.W, self.H, self.T_WP, self.z_ground)
            if bb is None:
                continue
            if bb[2] < x0 or bb[0] > x1 or bb[3] < y0 or bb[1] > y1:
                continue
            P = (T[:3, 3] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
            scored.append((float(np.hypot(P[0] - cx, P[1] - cy)), int(fid)))
        scored.sort()
        return [f for _, f in scored[: self.cfg.max_frames]]

    def candidates(self, grid, i: int, j: int, frame_ids: List[int],
                   T_gt: Optional[Dict[int, np.ndarray]] = None) -> List[dict]:
        out = []
        sel = self.select_frames(grid, i, j, frame_ids, T_gt)
        for fid in sel:
            T = self._frame_pose(fid, T_gt)
            if T is None:
                continue
            rgb, wgt, valid = self._warp_frame(fid, T, grid, i, j)
            if valid.sum() < self.cfg.min_valid_px:
                continue
            out.append({"fid": int(fid), "rgb": rgb, "w": wgt, "valid": valid,
                        "sharp": float(_grad(rgb)[valid].mean())})
        return out

    # ------------------------------------------------------------------
    def compose(self, grid, i: int, j: int, frame_ids: List[int],
                T_gt: Optional[Dict[int, np.ndarray]] = None) -> dict:
        cands = self.candidates(grid, i, j, frame_ids, T_gt)
        if not cands:
            return {"ok": False, "reason": "no-candidate"}
        stack = np.stack([c["rgb"] for c in cands], 0)
        vm = np.stack([c["valid"] for c in cands], 0)
        wm = np.stack([c["w"] for c in cands], 0)

        # exposure normalisation: per-candidate (gain, offset) fitted to the reference
        # candidate (the sharpest one) over their common valid area.  Aerial frames
        # differ by a few % in exposure; without this the fused image is a blend of
        # different brightness levels and the cross-frame agreement is meaningless.
        if len(cands) > 1:
            k_ref = int(np.argmax([c["sharp"] for c in cands]))
            ref = stack[k_ref]
            mref = vm[k_ref]
            for k in range(stack.shape[0]):
                if k == k_ref:
                    continue
                sel = mref & vm[k]
                if sel.sum() < 5000:
                    continue
                x = stack[k][sel].reshape(-1)
                y = ref[sel].reshape(-1)
                A = np.stack([x, np.ones_like(x)], 1)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                a_, b_ = float(coef[0]), float(coef[1])
                a_ = float(np.clip(a_, 0.7, 1.4))
                b_ = float(np.clip(b_, -0.15, 0.15))
                stack[k] = np.clip(a_ * stack[k] + b_, 0, 1) * vm[k][..., None]
        wsum = wm.sum(0)                       # (H, W)
        fused = (stack * wm[..., None]).sum(0) / np.maximum(wsum, 1e-6)[..., None]
        any_valid = vm.any(0)

        # cross-frame agreement: |X - median(others)|
        med = np.median(stack, axis=0)
        def agree(img, k=None):
            """Mean |img - median(other candidates)| over the valid area."""
            if stack.shape[0] > 1:
                others = stack if k is None else np.delete(stack, int(k), axis=0)
                m = np.median(others, axis=0)
            else:
                m = med
            sel = any_valid
            return float(np.abs(img[sel] - m[sel]).mean()) if sel.any() else 0.0

        ag_fused = agree(fused, None) if stack.shape[0] > 1 else 0.0
        k_best = int(np.argmax([c["sharp"] for c in cands]))
        ag_best = agree(stack[k_best], k_best) if stack.shape[0] > 1 else 0.0
        sh_fused = float(_grad(fused)[any_valid].mean())
        sh_best = float(cands[k_best]["sharp"])

        better = ((sh_fused >= (1.0 - self.cfg.sharpen_tol) * sh_best)
                  and (ag_fused <= (1.0 + self.cfg.agree_tol) * max(ag_best, 1e-9)))
        chosen = "fused" if (len(cands) > 1 and better) else "single"
        rgb = fused if chosen == "fused" else stack[k_best]
        return {"ok": True, "rgb": rgb, "chosen": chosen, "k": len(cands),
                "sharp_fused": sh_fused, "sharp_best": sh_best,
                "agree_fused": ag_fused, "agree_best": ag_best,
                "fid_best": cands[k_best]["fid"],
                "fids": [c["fid"] for c in cands],
                "valid": any_valid}
