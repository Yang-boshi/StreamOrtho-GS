"""Trajectory I/O and Sim3 (Umeyama) alignment + ATE metrics.

Everything here is pure numpy so it can run without CUDA / torch.

Conventions
-----------
* trajectory files: ``timestamp x y z qx qy qz qw`` (scalar-last), camera-to-world.
* alignment: similarity transform ``dst ~= s * R @ src + t`` with ``src=estimate``,
  ``dst=reference``; scale is estimated (MASt3R-SLAM's Sim3 scale is not metric).
* ATE is reported both in metres and as a percentage of the GT path length.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from stream_ortho.data.plane import quat_xyzw_to_matrix


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

@dataclass
class PoseTrajectory:
    timestamps: np.ndarray      # (N,)
    positions: np.ndarray       # (N,3)
    quaternions: np.ndarray     # (N,4) scalar-last

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def path_length(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    def rotations(self) -> np.ndarray:
        return np.stack([quat_xyzw_to_matrix(q) for q in self.quaternions]) if len(self) else np.zeros((0, 3, 3))


def load_tum(path) -> PoseTrajectory:
    """Load ``timestamp tx ty tz qx qy qz qw`` (extra columns ignored)."""
    path = pathlib.Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            rows.append([float(x) for x in parts[:8]])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"轨迹为空或格式不符: {path}")
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    return PoseTrajectory(arr[:, 0], arr[:, 1:4], arr[:, 4:8])


def save_tum(path, traj: PoseTrajectory, header: Optional[str] = None):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n")
        for t, p, q in zip(traj.timestamps, traj.positions, traj.quaternions):
            f.write(
                f"{t:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------

def associate_by_time(
    est: PoseTrajectory, ref: PoseTrajectory, max_dt: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-timestamp association; returns index arrays ``(i_est, i_ref)``."""
    if len(est) == 0 or len(ref) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(ref.timestamps)
    ref_ts = ref.timestamps[order]
    idx_est, idx_ref = [], []
    for i, t in enumerate(est.timestamps):
        j = int(np.searchsorted(ref_ts, t))
        cands = [c for c in (j - 1, j) if 0 <= c < len(ref_ts)]
        if not cands:
            continue
        best = min(cands, key=lambda c: abs(ref_ts[c] - t))
        if abs(ref_ts[best] - t) <= max_dt:
            idx_est.append(i)
            idx_ref.append(int(order[best]))
    return np.asarray(idx_est, np.int64), np.asarray(idx_ref, np.int64)


# ---------------------------------------------------------------------------
# Umeyama
# ---------------------------------------------------------------------------

@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray    # (3,3)
    translation: np.ndarray # (3,)

    def apply(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return self.scale * (pts @ self.rotation.T) + self.translation

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.scale * self.rotation
        T[:3, 3] = self.translation
        return T


def umeyama(src, dst, with_scale: bool = True) -> Sim3:
    """Least-squares similarity transform ``dst ~= s R src + t``.

    Standard Umeyama (1991) with the reflection guard.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama 需要成对的 (N,3) 点且 N>=3")
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = dst_c.T @ src_c / n                     # (3,3)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    var_src = (src_c ** 2).sum(axis=1).mean()
    if var_src <= 0:
        raise ValueError("源点退化（方差为 0），无法估计相似变换")
    scale = float(np.trace(np.diag(D) @ S) / var_src) if with_scale else 1.0
    t = mu_dst - scale * (R @ mu_src)
    return Sim3(scale=scale, rotation=R, translation=t)


# ---------------------------------------------------------------------------
# ATE
# ---------------------------------------------------------------------------

@dataclass
class AteResult:
    n_pairs: int
    n_est: int
    n_ref: int
    scale: float
    rmse: float
    mean: float
    median: float
    p95: float
    max: float
    path_length_ref: float
    gt_start_offset: float          # distance between the first frames (no align trick)
    errors: np.ndarray
    rot_rmse_deg: float = float("nan")

    @property
    def ate_percent(self) -> float:
        return 100.0 * self.rmse / self.path_length_ref if self.path_length_ref > 0 else float("nan")

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "n_est": self.n_est,
            "n_ref": self.n_ref,
            "scale": round(self.scale, 6),
            "rmse": round(self.rmse, 6),
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "p95": round(self.p95, 6),
            "max": round(self.max, 6),
            "path_length_ref": round(self.path_length_ref, 4),
            "ate_percent": round(self.ate_percent, 4),
            "rot_rmse_deg": round(float(self.rot_rmse_deg), 4),
        }


def _wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def evaluate_ate(
    est: PoseTrajectory,
    ref: PoseTrajectory,
    max_dt: float = 0.05,
    with_scale: bool = True,
) -> AteResult:
    """Sim3-aligned ATE of ``est`` against ``ref`` (nearest-timestamp pairs)."""
    i_est, i_ref = associate_by_time(est, ref, max_dt=max_dt)
    if len(i_est) < 3:
        raise ValueError(f"有效关联帧太少 ({len(i_est)})，无法评估 ATE")
    src = est.positions[i_est]
    dst = ref.positions[i_ref]
    sim3 = umeyama(src, dst, with_scale=with_scale)
    aligned = sim3.apply(src)
    err = np.linalg.norm(aligned - dst, axis=1)

    rot_err = float("nan")
    if len(est.quaternions) and len(ref.quaternions):
        R_est = est.rotations()[i_est]
        R_ref = ref.rotations()[i_ref]
        R_delta = np.einsum("nij,nkj->nik", sim3.rotation[None] @ R_est, R_ref)
        # angle of each relative rotation
        cos = (np.trace(R_delta, axis1=1, axis2=2) - 1.0) / 2.0
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        rot_err = float(np.sqrt((ang ** 2).mean()))

    # GT path length over the *associated* part (comparable denominator)
    gt_assoc = ref.positions[np.sort(i_ref)]
    path_len = float(np.linalg.norm(np.diff(gt_assoc, axis=0), axis=1).sum()) if len(gt_assoc) > 1 else 0.0

    return AteResult(
        n_pairs=len(i_est),
        n_est=len(est),
        n_ref=len(ref),
        scale=sim3.scale,
        rmse=float(np.sqrt((err ** 2).mean())),
        mean=float(err.mean()),
        median=float(np.median(err)),
        p95=float(np.percentile(err, 95)),
        max=float(err.max()),
        path_length_ref=path_len,
        gt_start_offset=float(np.linalg.norm(src[0] - dst[0])),
        errors=err,
        rot_rmse_deg=rot_err,
    )
