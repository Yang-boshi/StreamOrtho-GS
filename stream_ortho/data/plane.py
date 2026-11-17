"""Parsing of NPU/DroneMap ``config.cfg`` and the ground-plane pose ``T_WP``.

Data facts (verified on ``/home/zxy/syb/NPU``, 2026-10-24):

``config.cfg`` (example ``phantom3-centralPark-kfs``)::

    Plane=-39.930891 6.111688 -78.090839 -0.006565 -0.000468 0.000003 0.999978
    GPS.Origin=114.071121 22.536340 0.000000
    Camera.CameraType=PinHole
    Camera.Paraments=[1920,1080,1220.000000,1220.000000,960.000000,540.000000]
    TrajectoryFile=$(Svar.ParsingPath)/trajectory.txt

* ``Plane`` is **7 numbers**: ``x y z qx qy qz qw`` -- the plane frame origin
  in world coordinates plus a scalar-last quaternion.  The quaternion is very
  close to identity (``|q_xyz| ~ 1e-3``), i.e. the plane frame is a
  gravity-aligned ENU-like frame.  It is *not* ``a b c d`` plane coefficients;
  interpreting it that way gives a unit normal of norm 40, which is absurd.
* ``Camera.Paraments`` keeps the upstream typo; order is ``W H fx fy cx cy``.
* ``GPS.Origin`` is ``lon lat alt`` (degrees / metres).

Conventions used project-wide (see ``results/week1/NOTES.md``):

* world frame ``W``: the GT trajectory frame (local ENU, metres).
* plane frame ``P``: ``T_WP`` maps plane -> world; plane ``+Z`` is "up",
  ground is approximately ``z_P = 0`` (verified against trajectory heights).
* camera frame ``C``: OpenCV, ``+x`` right, ``+y`` down, ``+z`` forward.
* world points of a MASt3R pointmap: ``X_W = T_WC.act(X_canon)``.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_KV_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=\s*(.*?)\s*$")
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# config.cfg
# ---------------------------------------------------------------------------

@dataclass
class NpuConfig:
    """Parsed ``config.cfg`` of one NPU sequence."""

    path: pathlib.Path
    raw: Dict[str, str] = field(default_factory=dict)

    # camera
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    camera_type: str = "PinHole"
    distortion: List[float] = field(default_factory=list)

    # ground plane: translation + scalar-last quaternion
    plane_xyz: Optional[np.ndarray] = None
    plane_quat: Optional[np.ndarray] = None        # (qx, qy, qz, qw)

    # georeference
    gps_origin: Optional[np.ndarray] = None        # (lon, lat, alt)
    trajectory_file: Optional[pathlib.Path] = None

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix in *original* image pixels."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def calib(self) -> List[float]:
        return [self.fx, self.fy, self.cx, self.cy]

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "camera_type": self.camera_type,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": self.distortion,
            "plane_xyz": None if self.plane_xyz is None else self.plane_xyz.tolist(),
            "plane_quat": None if self.plane_quat is None else self.plane_quat.tolist(),
            "gps_origin": None if self.gps_origin is None else self.gps_origin.tolist(),
            "trajectory_file": None if self.trajectory_file is None else str(self.trajectory_file),
        }


def parse_config_cfg(path) -> NpuConfig:
    """Parse a DroneMap style ``config.cfg``.

    ``$(Svar.ParsingPath)`` is replaced by the file's directory.  Unknown keys
    are kept in ``raw`` so nothing is silently lost.
    """
    path = pathlib.Path(path)
    cfg = NpuConfig(path=path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _KV_RE.match(line)
        if m:
            cfg.raw[m.group(1)] = m.group(2)

    def _numbers(text: str) -> List[float]:
        return [float(x) for x in _FLOAT_RE.findall(text)]

    if "Camera.Paraments" in cfg.raw:
        v = _numbers(cfg.raw["Camera.Paraments"])
        if len(v) < 6:
            raise ValueError(f"Camera.Paraments 少于 6 个数: {path}")
        cfg.width, cfg.height = int(round(v[0])), int(round(v[1]))
        cfg.fx, cfg.fy, cfg.cx, cfg.cy = v[2:6]
        if len(v) > 6:
            cfg.distortion = v[6:]
    else:
        raise ValueError(f"config.cfg 缺少 Camera.Paraments: {path}")

    if "Camera.CameraType" in cfg.raw:
        cfg.camera_type = cfg.raw["Camera.CameraType"].strip()

    if "Plane" in cfg.raw:
        v = _numbers(cfg.raw["Plane"])
        if len(v) == 7:
            cfg.plane_xyz = np.asarray(v[:3], dtype=np.float64)
            cfg.plane_quat = np.asarray(v[3:7], dtype=np.float64)
        elif len(v) >= 7:
            # Defensive: take the first 7 numbers, keep the rest for diagnosis.
            cfg.plane_xyz = np.asarray(v[:3], dtype=np.float64)
            cfg.plane_quat = np.asarray(v[3:7], dtype=np.float64)
            cfg.raw["_plane_extra"] = str(v[7:])

    if "GPS.Origin" in cfg.raw:
        v = _numbers(cfg.raw["GPS.Origin"])
        if len(v) >= 2:
            cfg.gps_origin = np.asarray(v[:3] if len(v) >= 3 else v[:2] + [0.0],
                                        dtype=np.float64)

    if "TrajectoryFile" in cfg.raw:
        tf = cfg.raw["TrajectoryFile"].replace("$(Svar.ParsingPath)", str(path.parent))
        cfg.trajectory_file = pathlib.Path(tf)

    return cfg


# ---------------------------------------------------------------------------
# Plane pose
# ---------------------------------------------------------------------------

def quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix from a scalar-first quaternion (lietorch / TUM order)."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrix from a scalar-last quaternion (NPU ``Plane`` order)."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return quat_wxyz_to_matrix(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))


def plane_to_world(plane_xyz, plane_quat_xyzw) -> np.ndarray:
    """``T_WP``: plane -> world, as a 4x4 SE3 matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix(plane_quat_xyzw)
    T[:3, 3] = np.asarray(plane_xyz, dtype=np.float64).reshape(3)
    return T


def T_WP_from_cfg(cfg: NpuConfig) -> np.ndarray:
    if cfg.plane_xyz is None or cfg.plane_quat is None:
        raise ValueError(f"{cfg.path} 没有可解析的 Plane 行")
    return plane_to_world(cfg.plane_xyz, cfg.plane_quat)


def world_to_plane(T_WP: np.ndarray, pts_W) -> np.ndarray:
    """Transform ``(N,3)`` world points into the plane frame."""
    pts = np.asarray(pts_W, dtype=np.float64).reshape(-1, 3)
    T_PW = np.linalg.inv(T_WP)
    return pts @ T_PW[:3, :3].T + T_PW[:3, 3]


def plane_normal_W(T_WP: np.ndarray) -> np.ndarray:
    """Plane ``+Z`` axis expressed in world coordinates (unit)."""
    return T_WP[:3, 2] / np.linalg.norm(T_WP[:3, 2])


def camera_heights(T_WP: np.ndarray, positions_W) -> np.ndarray:
    """Height of camera centres above the ground plane (metres)."""
    return world_to_plane(T_WP, positions_W)[:, 2]


# ---------------------------------------------------------------------------
# trajectory.txt
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """``trajectory.txt``: ``timestamp x y z qx qy qz qw`` (camera-to-world)."""

    timestamps: np.ndarray
    positions: np.ndarray       # (N,3)
    quaternions: np.ndarray     # (N,4) scalar-last
    dt: np.ndarray

    def __len__(self) -> int:
        return len(self.timestamps)

    def __post_init__(self):
        if not (len(self.timestamps) == len(self.positions) == len(self.quaternions)):
            raise ValueError("trajectory 字段长度不一致")

    @property
    def path_length(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    def matrix(self, i: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = quat_xyzw_to_matrix(self.quaternions[i])
        T[:3, 3] = self.positions[i]
        return T

    def index_of(self, timestamp: float, tol: float = 5e-4) -> Optional[int]:
        idx = np.searchsorted(self.timestamps, timestamp)
        for j in (idx - 1, idx):
            if 0 <= j < len(self):
                if abs(float(self.timestamps[j]) - float(timestamp)) <= tol:
                    return int(j)
        return None

    def interpolate(self, timestamps) -> np.ndarray:
        """Linear interpolation of positions at arbitrary timestamps (clipped)."""
        ts = np.asarray(timestamps, dtype=np.float64)
        out = np.empty((len(ts), 3), dtype=np.float64)
        for k in range(3):
            out[:, k] = np.interp(ts, self.timestamps, self.positions[:, k])
        return out


def parse_trajectory(path) -> Trajectory:
    path = pathlib.Path(path)
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
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
        raise ValueError(f"trajectory 为空或格式不符: {path}")
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    dt = np.zeros(len(arr), dtype=np.float64)
    if len(arr) > 1:
        dt[1:] = np.diff(arr[:, 0])
    return Trajectory(arr[:, 0], arr[:, 1:4], arr[:, 4:8], dt)
