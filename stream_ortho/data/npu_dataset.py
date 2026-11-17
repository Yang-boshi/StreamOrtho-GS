"""``NPUDroneMapDataset`` -- the S1 deliverable.

Compatible with both on-disk layouts found in the NPU DroneMap release::

    <root>/rgb/<timestamp>.jpg          (single level)
    <root>/<root.name>/rgb/*.jpg        (double level)

It subclasses the upstream ``MonocularDataset`` so it can be dropped into the
MASt3R-SLAM loop unchanged, but it never writes ``.ply`` (``save_results`` is
forced to ``False``).

Verified data facts (2026-10-24, ``/home/zxy/syb/NPU``):

=====================  ======  ======  ==========
sequence               images  gt      note
=====================  ======  ======  ==========
phantom3-centralPark    835    --      reference ``result.png`` only
phantom3-ieu              0    467     RGB images missing
others                 285-648   same  complete
=====================  ======  ======  ==========
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from stream_ortho import compat
from stream_ortho.data.plane import (
    NpuConfig,
    Trajectory,
    camera_heights,
    parse_config_cfg,
    parse_trajectory,
    T_WP_from_cfg,
)

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
_TS_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)")

# ``MonocularDataset`` import is lazy: importing torch is fine but slow-ish and
# not needed for pure parsing tests.
try:  # pragma: no cover - exercised implicitly everywhere
    compat.ensure_upstream_on_path()
    from mast3r_slam.dataloader import Intrinsics, MonocularDataset
except Exception:  # pragma: no cover - fallback for doc-only environments
    Intrinsics = None  # type: ignore
    MonocularDataset = object  # type: ignore


def list_timestamped_images(rgb_dir) -> List[Tuple[float, pathlib.Path]]:
    """List ``rgb/`` images sorted by *numeric* timestamp (never by name)."""
    rgb_dir = pathlib.Path(rgb_dir)
    items: List[Tuple[float, pathlib.Path]] = []
    for p in rgb_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in _IMG_SUFFIXES:
            continue
        m = _TS_RE.match(p.name)
        if not m:
            continue
        items.append((float(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return items


def resolve_sequence_root(path) -> pathlib.Path:
    """Find the directory that actually holds ``config.cfg``."""
    root = pathlib.Path(path)
    if (root / "config.cfg").exists():
        return root
    sub = root / root.name
    if (sub / "config.cfg").exists():
        return sub
    cands = sorted(root.glob("*/config.cfg"))
    if cands:
        return cands[0].parent
    raise FileNotFoundError(f"在 {root} 下找不到 config.cfg")


@dataclass
class GtCoverage:
    n_images: int
    n_matched: int
    n_missing: int
    n_extra_traj: int

    @property
    def ratio(self) -> float:
        return self.n_matched / self.n_images if self.n_images else 0.0

    def as_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "n_matched": self.n_matched,
            "n_missing": self.n_missing,
            "n_extra_traj": self.n_extra_traj,
            "ratio": round(self.ratio, 4),
        }


class NPUDroneMapDataset(MonocularDataset):  # type: ignore[misc]
    """Streaming view over one NPU DroneMap sequence.

    Parameters
    ----------
    root:
        Sequence directory (either nesting level).
    img_size:
        MASt3R input size (512).  The frame is ``512x288`` for 1920x1080 input.
    stride:
        Keep every ``stride``-th image (after timestamp sorting).
    use_calib:
        If True, expose the upstream ``Intrinsics`` object (needed for the
        ``--calib`` code path).  Defaults to the upstream global config.
    max_frames:
        Optional cap (smoke tests / bounded latency experiments).
    """

    def __init__(
        self,
        root,
        img_size: int = 512,
        stride: int = 1,
        use_calib: Optional[bool] = None,
        max_frames: Optional[int] = None,
        dtype=np.float32,
    ):
        # upstream ``MonocularDataset.__init__`` reads ``config["use_calib"]``,
        # so the global config must be loaded *before* calling super().
        compat.load_upstream_config()
        from mast3r_slam.config import config as up_cfg

        super().__init__(dtype=dtype)

        root = pathlib.Path(root).expanduser()
        self.root = resolve_sequence_root(root)
        self.layout = "single" if root == self.root else "nested"
        self.dataset_path = self.root          # upstream expects .dataset_path
        self.name = self.root.name
        self.img_size = img_size
        self.save_results = False              # main path never writes .ply

        # --- config.cfg -------------------------------------------------
        self.cfg: NpuConfig = parse_config_cfg(self.root / "config.cfg")
        self.T_WP = T_WP_from_cfg(self.cfg)

        # --- images -----------------------------------------------------
        rgb_dir = self.root / "rgb"
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"缺少 rgb/ 目录: {rgb_dir}")
        items = list_timestamped_images(rgb_dir)[::stride]
        if not items:
            raise FileNotFoundError(f"rgb/ 下没有可用影像: {rgb_dir}")
        self.timestamps = np.asarray([t for t, _ in items], dtype=np.float64)
        self.rgb_files: List[pathlib.Path] = [p for _, p in items]

        # --- optional GT trajectory ------------------------------------
        self.traj: Optional[Trajectory] = None
        traj_path = None
        if self.cfg.trajectory_file is not None and self.cfg.trajectory_file.exists():
            traj_path = self.cfg.trajectory_file
        elif (self.root / "trajectory.txt").exists():
            traj_path = self.root / "trajectory.txt"
        if traj_path is not None:
            try:
                self.traj = parse_trajectory(traj_path)
            except Exception as exc:  # pragma: no cover
                print(f"[NPUDroneMapDataset] 读取 trajectory 失败: {exc}")
        self.traj_path = traj_path

        self._gt_index = np.full(len(self), -1, dtype=np.int64)
        self.gt_positions = np.full((len(self), 3), np.nan, dtype=np.float64)
        self._build_gt_index()
        self.gt_coverage = self._compute_coverage()

        # --- intrinsics -------------------------------------------------
        self.use_calibration = False
        want_calib = up_cfg.get("use_calib", False) if use_calib is None else use_calib
        self._want_calib = bool(want_calib)
        if want_calib:
            self.enable_calibration()

        if max_frames is not None:
            self._truncate(max_frames)

        self._check_timestamps()

    # ------------------------------------------------------------------
    # GT association
    # ------------------------------------------------------------------
    def _build_gt_index(self):
        if self.traj is None:
            return
        ts = self.traj.timestamps
        # exact match first (tolerance 0.5 ms), then nearest inside 25 ms
        for i, t in enumerate(self.timestamps):
            j = self.traj.index_of(t)
            if j is None:
                k = int(np.argmin(np.abs(ts - t)))
                if abs(ts[k] - t) <= 2.5e-2:
                    j = k
            if j is not None:
                self._gt_index[i] = j
                self.gt_positions[i] = self.traj.positions[j]
        # interpolate the rest (still useful for ATE on subsequences)
        miss = self._gt_index < 0
        if miss.any():
            self.gt_positions[miss] = self.traj.interpolate(self.timestamps[miss])

    def _compute_coverage(self) -> GtCoverage:
        if self.traj is None:
            return GtCoverage(len(self), 0, len(self), 0)
        matched = int((self._gt_index >= 0).sum())
        used = set(self._gt_index[self._gt_index >= 0].tolist())
        return GtCoverage(
            n_images=len(self),
            n_matched=matched,
            n_missing=len(self) - matched,
            n_extra_traj=len(self.traj) - len(used),
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def enable_calibration(self):
        """Build the upstream ``Intrinsics`` (undistort + resize to 512).

        ``Intrinsics.from_calib`` consults the *global* upstream config, so we
        flip it for the duration of the call and restore it afterwards.
        """
        if Intrinsics is None:
            raise RuntimeError("无法导入 mast3r_slam.dataloader.Intrinsics")
        from mast3r_slam.config import config as up_cfg

        calib = list(self.cfg.calib) + list(self.cfg.distortion)
        previous_use_calib = up_cfg.get("use_calib", False)
        previous_instance = self.use_calibration
        up_cfg["use_calib"] = True
        self.use_calibration = True
        try:
            self.camera_intrinsics = Intrinsics.from_calib(
                self.img_size, self.cfg.width, self.cfg.height, calib
            )
        finally:
            up_cfg["use_calib"] = previous_use_calib
            self.use_calibration = previous_instance
        if self.camera_intrinsics is None:
            raise RuntimeError("Intrinsics.from_calib 返回 None")

    # ------------------------------------------------------------------
    # Intrinsics in the 512 frame (always available, also without upstream)
    # ------------------------------------------------------------------
    @property
    def K_orig(self) -> np.ndarray:
        return self.cfg.K

    def frame_shape(self) -> Tuple[int, int]:
        """``(h, w)`` of the MASt3R input frame."""
        W, H = self.cfg.width, self.cfg.height
        if self.img_size == 512:
            return (288, 512) if W / H >= 1.5 else (512, round(H * 512 / W))
        return (self.img_size, self.img_size)

    def K_frame_analytic(self) -> np.ndarray:
        """Analytic 512-frame intrinsics (long side resized, centre crop).

        Equals the upstream ``Intrinsics.K_frame`` up to the
        ``cv2.getOptimalNewCameraMatrix`` rounding (~0.3 px for NPU data, whose
        distortion is identically zero).
        """
        W, H = self.cfg.width, self.cfg.height
        h, w = self.frame_shape()
        sx, sy = w / W, h / H
        K = self.cfg.K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    def K_frame(self) -> np.ndarray:
        """Intrinsics of the 512 input frame (upstream one when available)."""
        if getattr(self, "camera_intrinsics", None) is not None:
            return np.asarray(self.camera_intrinsics.K_frame, dtype=np.float64)
        return self.K_frame_analytic()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def subsample(self, subsample: int):
        if subsample is None or subsample <= 1:
            return
        keep = slice(None, None, subsample)
        self.rgb_files = self.rgb_files[keep]
        self.timestamps = self.timestamps[keep]
        self._gt_index = self._gt_index[keep]
        self.gt_positions = self.gt_positions[keep]

    def _truncate(self, n: int):
        if n >= len(self):
            return
        self.rgb_files = self.rgb_files[:n]
        self.timestamps = self.timestamps[:n]
        self._gt_index = self._gt_index[:n]
        self.gt_positions = self.gt_positions[:n]

    def _check_timestamps(self):
        if len(self.timestamps) > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("时间戳不是严格递增（数据排序有误）")

    def read_img(self, idx):
        import cv2

        img = cv2.imread(str(self.rgb_files[idx]))
        if img is None:
            raise IOError(f"读取失败: {self.rgb_files[idx]}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_timestamp(self, idx):
        return float(self.timestamps[idx])

    # ------------------------------------------------------------------
    # GT / plane helpers
    # ------------------------------------------------------------------
    def gt_index(self, idx: int) -> int:
        return int(self._gt_index[idx])

    def gt_pose(self, idx: int) -> Optional[np.ndarray]:
        j = self._gt_index[idx]
        if self.traj is None or j < 0:
            return None
        return self.traj.matrix(int(j))

    def gt_positions_matched(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(timestamps, positions)`` for frames with a GT match."""
        m = self._gt_index >= 0
        return self.timestamps[m], self.gt_positions[m]

    def camera_heights(self) -> np.ndarray:
        """Camera heights above the plane frame's ``z=0`` (metres)."""
        return camera_heights(self.T_WP, self.gt_positions)

    def plane_normal_world(self) -> np.ndarray:
        n = self.T_WP[:3, 2]
        return n / np.linalg.norm(n)

    def height_diagnostics(self) -> dict:
        """Statistics of camera height above the plane (plane-origin sanity).

        The ``Plane`` translation is *not* guaranteed to sit on the ground
        (design doc S1: "Plane 的平移不是平面中心"), so a negative median here
        means "the plane origin is above the camera", not "the drone flew
        underground".  Only the **spread** is a real quality signal.
        """
        h = self.camera_heights()
        h = h[np.isfinite(h)]
        n_W = self.plane_normal_world()
        base = {
            "n": int(len(h)),
            "plane_normal_dot_world_z": float(n_W[2]),
            "plane_is_horizontal": bool(abs(n_W[2]) > 0.99),
        }
        if not len(h):
            base.update({"median": float("nan"), "median_positive": False,
                         "spread_p5_p95": float("nan"), "relative_spread": float("nan")})
            return base
        q = np.percentile(h, [0, 5, 25, 50, 75, 95, 100])
        median = float(q[3])
        spread = float(q[5] - q[1])
        base.update({
            "min": float(q[0]),
            "p5": float(q[1]),
            "median": median,
            "p95": float(q[5]),
            "max": float(q[6]),
            "spread_p5_p95": spread,
            "relative_spread": float(spread / abs(median)) if median != 0 else float("inf"),
            "median_positive": bool(median > 0),
        })
        return base

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        pos = self.gt_positions
        valid = np.isfinite(pos).all(axis=1)
        ts_valid = self.timestamps[valid]
        pos_valid = pos[valid]
        order = np.argsort(ts_valid)
        pos_valid = pos_valid[order]

        out = {
            "name": self.name,
            "root": str(self.root),
            "layout": self.layout,
            "n_frames": len(self),
            "n_frames_on_disk": len(list_timestamped_images(self.root / "rgb")),
            "image_shape": [self.cfg.height, self.cfg.width],
            "frame_shape": list(self.frame_shape()),
            "camera_type": self.cfg.camera_type,
            "fx": self.cfg.fx,
            "fy": self.cfg.fy,
            "cx": self.cfg.cx,
            "cy": self.cfg.cy,
            "K_frame": self.K_frame().tolist(),
            "K_frame_analytic": self.K_frame_analytic().tolist(),
            "has_traj": self.traj is not None,
            "traj_path": None if self.traj_path is None else str(self.traj_path),
            "n_traj": 0 if self.traj is None else len(self.traj),
            "gt_coverage": self.gt_coverage.as_dict(),
            "timestamps_monotonic": bool(
                len(self.timestamps) < 2 or np.all(np.diff(self.timestamps) > 0)
            ),
            "dt_median": float(np.median(np.diff(self.timestamps)))
            if len(self.timestamps) > 1
            else 0.0,
            "plane_xyz": self.cfg.plane_xyz.tolist() if self.cfg.plane_xyz is not None else None,
            "plane_quat": self.cfg.plane_quat.tolist() if self.cfg.plane_quat is not None else None,
            "plane_normal_W": self.plane_normal_world().tolist(),
            "T_WP": self.T_WP.tolist(),
            "T_WP_roundtrip_error": float(
                np.abs(self.T_WP @ np.linalg.inv(self.T_WP) - np.eye(4)).max()
            ),
            "gps_origin": None if self.cfg.gps_origin is None else self.cfg.gps_origin.tolist(),
            "height": self.height_diagnostics(),
        }
        if len(pos_valid) > 1:
            out["gt_span"] = (pos_valid.max(axis=0) - pos_valid.min(axis=0)).tolist()
            out["gt_path_length"] = float(
                np.linalg.norm(np.diff(pos_valid, axis=0), axis=1).sum()
            )
        else:
            out["gt_span"] = None
            out["gt_path_length"] = 0.0
        return out


def discover_sequences(root) -> List[pathlib.Path]:
    """All sequence directories under ``root`` (handles both nesting levels)."""
    root = pathlib.Path(root)
    seqs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.is_symlink():
            continue
        try:
            seqs.append(resolve_sequence_root(p))
        except FileNotFoundError:
            continue
    return seqs


def load_npu_dataset(root, **kwargs) -> NPUDroneMapDataset:
    return NPUDroneMapDataset(root, **kwargs)
