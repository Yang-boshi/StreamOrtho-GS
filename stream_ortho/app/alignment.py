"""Align the live SLAM cloud to the plane frame the ortho grid works in.

The replay path gets this for free: its observation table already carries plane-frame x/y and
a ``T_WP``.  The live path only has ``X_canon`` (camera frame) and ``T_WC``, so the points sit
in the SLAM world -- a different scale, a different tilt, a different origin.  Without this
step the ortho grid sees a degenerate footprint (P4 smoke: 0 tiles).

``estimate_ground_plane`` fits the dominant low surface (RANSAC over the lower part of the
cloud) and returns a rigid transform that puts that plane on z = 0 with +Z up.
``PlaneAligner`` bootstraps: it buffers the first keyframes, fits once, then maps every
subsequent cloud into the plane frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def _fit_plane_ransac(P, iters=200, thresh=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    n = P.shape[0]
    best_inl, best = None, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = P[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -float(nrm @ p0)
        res = np.abs(P @ nrm + d)
        inl = res < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best = inl, (nrm, d)
    if best is None:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n, bool)
    nrm, d = best
    # refit on the inliers by least squares
    Q = P[best_inl]
    if Q.shape[0] >= 3:
        c = Q.mean(0)
        u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = vt[-1]
        d = -float(nrm @ c)
    inl = np.abs(P @ nrm + d) < thresh
    return nrm, d, inl


def estimate_ground_plane(Pw, ransac_iters=200, thresh=0.5, low_pct=(5, 60)):
    """Fit the ground plane of a world-frame cloud; returns ``(T_WP, z_ground, info)``."""
    P = np.asarray(Pw, np.float64).reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    if P.shape[0] < 50:
        raise ValueError("not enough points to estimate a plane")
    # the ground is the lower part of the cloud: keep it so a roof cannot win the RANSAC
    z = P[:, 2]
    lo, hi = np.percentile(z, low_pct)
    cand = P[z <= hi]
    if cand.shape[0] < 50:
        cand = P
    nrm, d, inl = _fit_plane_ransac(cand, ransac_iters, thresh)
    # make +Z point up (the plane normal should face the camera side, z positive)
    if nrm[2] < 0:
        nrm, d = -nrm, -d
    # rotation taking nrm to +Z
    zaxis = np.array([0.0, 0.0, 1.0])
    v = np.cross(nrm, zaxis)
    s = np.linalg.norm(v)
    if s < 1e-9:
        R = np.eye(3) if nrm[2] > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - np.cos(np.arccos(np.clip(nrm @ zaxis, -1, 1)))) /
                                       max(s ** 2, 1e-12))
    T = np.eye(4)
    T[:3, :3] = R
    Pp = P @ R.T
    z_ground = float(np.percentile(Pp[:, 2], 5))
    T[:3, 3] = np.array([0.0, 0.0, -z_ground])
    info = {"normal": nrm.tolist(), "d": float(d), "inlier_frac": float(inl.mean()),
            "z_ground": z_ground, "n_points": int(P.shape[0])}
    return T, z_ground, info


@dataclass
class PlaneAligner:
    mode: str = "auto"            # auto | gt | plane | identity
    T_WP: np.ndarray | None = None
    bootstrap_kfs: int = 5
    thresh: float = 0.5

    def __post_init__(self):
        self._buffer = []
        self._T = None
        self._info = {}
        if self.mode in ("gt", "identity") or self.T_WP is not None:
            if self.mode == "identity" and self.T_WP is None:
                self._T = np.eye(4)
            elif self.T_WP is not None:
                self._T = np.asarray(self.T_WP, np.float64)
            self._info = {"mode": self.mode,
                          "source": "config" if self.T_WP is not None else "identity"}

    @property
    def ready(self):
        return self._T is not None

    def add(self, Pw) -> bool:
        """Buffer a cloud; returns True once the transform is available."""
        if self._T is not None:
            return True
        self._buffer.append(np.asarray(Pw, np.float64).reshape(-1, 3))
        if len(self._buffer) >= max(int(self.bootstrap_kfs), 1):
            P = np.vstack(self._buffer)
            self._T, z_g, info = estimate_ground_plane(P, thresh=self.thresh)
            self._info = {"mode": self.mode, "source": "estimated",
                          "bootstrap_frames": len(self._buffer), **info}
            self._buffer = []
            return True
        return False

    def to_plane(self, Pw):
        if self._T is None:
            raise RuntimeError("alignment not ready")
        P = np.asarray(Pw, np.float64).reshape(-1, 3)
        return P @ self._T[:3, :3].T + self._T[:3, 3]

    def record(self) -> dict:
        return dict(self._info)

    def save(self, path):
        import pathlib
        d = {"mode": self.mode, **self.record()}
        if self._T is not None:
            d["T_WP"] = self._T.tolist()
        pathlib.Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return d
