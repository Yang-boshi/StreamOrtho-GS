"""Keyframe policy skeleton (design doc §7.4 item 6): overlap + baseline + ortho gain.

MASt3R-SLAM decides keyframes from *tracking* (match fraction / pose change).
For orthomosaicking that is not enough: what matters is whether the new frame
adds **new ground area** and whether the overlap is still good enough for the
Gaussian patch to be merged into the map (week 4's commit gate).

This module is intentionally small and dependency-free so it can be imported by
the streaming loop, the offline evaluation and the week-4 commit gate.

Terms (all in the *plane* frame, metres):

* ``footprint``  : bbox of the image's ground footprint (from the pose + plane)
* ``baseline``   : horizontal distance between the new and the last keyframe
* ``overlap``    : footprint IoU between the new frame and the last keyframe
* ``new_area``   : fraction of the new footprint that is *not* covered by the
                   already accepted keyframes (a coarse ortho-coverage proxy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


def bbox_from_pose(T_WC: np.ndarray, K: np.ndarray, W: int, H: int,
                   T_WP: np.ndarray, z_ground: float = 0.0,
                   margin: float = 1.0) -> Optional[BBox]:
    """Ground footprint bbox of a camera pose (plane frame), or ``None``."""
    n = T_WP[:3, 2]
    p0 = T_WP[:3, 3] + n * z_ground
    pts = []
    for u in (0, W * 0.5, W):
        for v in (0, H * 0.5, H):
            d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
            d_w = T_WC[:3, :3] @ d_cam
            denom = float(d_w @ n)
            if abs(denom) < 1e-9:
                continue
            t = float((p0 - T_WC[:3, 3]) @ n) / denom
            if not np.isfinite(t) or t <= 0:
                continue
            pts.append(T_WC[:3, 3] + t * d_w)
    if len(pts) < 4:
        return None
    P = np.asarray(pts)
    P = (P - T_WP[:3, 3]) @ T_WP[:3, :3]
    x0, y0 = P[:, 0].min() - margin, P[:, 1].min() - margin
    x1, y1 = P[:, 0].max() + margin, P[:, 1].max() + margin
    return (float(x0), float(y0), float(x1), float(y1))


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PolicyConfig:
    # tracking-quality gates (mirror MASt3R-SLAM's thinking)
    min_overlap: float = 0.30          # below this, merging patches gets unreliable
    max_overlap: float = 0.92          # above this, the frame adds nothing
    min_baseline_m: float = 2.0
    max_baseline_m: float = 200.0
    max_rotation_deg: float = 30.0
    # ortho-coverage gate
    min_new_area: float = 0.05         # at least 5% of the footprint must be new
    # integration
    ortho_gain_weight: float = 0.5     # score = w*new_area + (1-w)*overlap_quality


@dataclass
class KeyframeDecision:
    add: bool
    reason: str
    overlap: float
    new_area: float
    baseline_m: float
    rotation_deg: float
    score: float
    footprint: Optional[BBox] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "add": self.add, "reason": self.reason,
            "overlap": round(self.overlap, 4), "new_area": round(self.new_area, 4),
            "baseline_m": round(self.baseline_m, 3),
            "rotation_deg": round(self.rotation_deg, 2),
            "score": round(self.score, 4),
        }


class KeyframePolicy:
    """Skeleton: decide whether a tracked frame becomes a keyframe.

    Usage in the streaming loop (week 4+)::

        policy = KeyframePolicy(PolicyConfig(), K, W, H, T_WP, z_ground)
        d = policy.decide(T_WC_new, tracking_ok=..., tracking_score=...)
        if d.add:  map.ingest(patch); policy.commit(T_WC_new, d.footprint)
    """

    def __init__(self, cfg: PolicyConfig, K: np.ndarray, W: int, H: int,
                 T_WP: np.ndarray, z_ground: float = 0.0):
        self.cfg = cfg
        self.K = K
        self.W, self.H = int(W), int(H)
        self.T_WP = T_WP
        self.z_ground = float(z_ground)
        self.footprints: List[BBox] = []
        self.n_decisions = 0
        self.n_added = 0

    # ------------------------------------------------------------------
    def _new_area(self, footprint: BBox) -> float:
        """Fraction of ``footprint`` not covered by any accepted footprint.

        A coarse but fast proxy: rasterize the accepted bboxes into a small
        occupancy grid restricted to the new footprint (32x32), no Python loops
        over pixels.
        """
        if not self.footprints:
            return 1.0
        x0, y0, x1, y1 = footprint
        if x1 <= x0 or y1 <= y0:
            return 0.0
        G = 32
        cov = np.zeros((G, G), bool)
        xs = np.linspace(x0, x1, G + 1)
        ys = np.linspace(y0, y1, G + 1)
        for b in self.footprints:
            if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            i0 = max(0, int(math.floor((b[0] - x0) / (x1 - x0) * G)))
            i1 = min(G, int(math.ceil((b[2] - x0) / (x1 - x0) * G)))
            j0 = max(0, int(math.floor((b[1] - y0) / (y1 - y0) * G)))
            j1 = min(G, int(math.ceil((b[3] - y0) / (y1 - y0) * G)))
            cov[j0:j1, i0:i1] = True
        return float(1.0 - cov.mean())

    # ------------------------------------------------------------------
    def decide(self, T_WC: np.ndarray, tracking_ok: bool = True,
               tracking_score: float = 1.0) -> KeyframeDecision:
        self.n_decisions += 1
        footprint = bbox_from_pose(T_WC, self.K, self.W, self.H, self.T_WP, self.z_ground)
        if footprint is None:
            return KeyframeDecision(False, "no-footprint", 0, 0, 0, 0, 0)

        if not self.footprints:
            self._commit(T_WC, footprint)
            return KeyframeDecision(True, "first", 0.0, 1.0, 0.0, 0.0, 1.0, footprint)

        last_pose = self.last_pose
        baseline = float(np.linalg.norm(T_WC[:3, 3] - last_pose[:3, 3]))
        R_rel = T_WC[:3, :3].T @ last_pose[:3, :3]
        cos = (np.trace(R_rel) - 1.0) / 2.0
        rot = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        overlap = bbox_iou(self.footprints[-1], footprint)
        new_area = self._new_area(footprint)

        if not tracking_ok:
            reason, add = "tracking-lost", False
        elif overlap > self.cfg.max_overlap:
            reason, add = "too-similar", False
        elif overlap < self.cfg.min_overlap and baseline > self.cfg.max_baseline_m:
            reason, add = "lost-overlap", False
        elif new_area < self.cfg.min_new_area:
            reason, add = "no-new-area", False
        elif baseline < self.cfg.min_baseline_m and rot < self.cfg.max_rotation_deg:
            reason, add = "too-close", False
        else:
            reason, add = "add", True

        w = self.cfg.ortho_gain_weight
        score = w * new_area + (1 - w) * overlap * tracking_score
        if add:
            self._commit(T_WC, footprint)
        return KeyframeDecision(add, reason, overlap, new_area, baseline, rot, score, footprint)

    def _commit(self, T_WC, footprint):
        self.footprints.append(footprint)
        self.last_pose = T_WC
        self.n_added += 1

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        area = sum(bbox_area(b) for b in self.footprints)
        return {
            "n_decisions": self.n_decisions,
            "n_added": self.n_added,
            "accept_ratio": self.n_added / max(1, self.n_decisions),
            "n_footprints": len(self.footprints),
            "sum_footprint_area_m2": round(area, 1),
            "mean_footprint_area_m2": round(area / max(1, len(self.footprints)), 1),
        }
