"""Orthographic tile grid in the ground-plane frame (design doc S4 / §7.4 item 1).

Conventions
-----------
* plane frame ``P``: ``T_WP`` maps ``P -> W`` (GT world).  ``+Z_P`` is up; the
  ground is at ``z_P = z_ground`` (estimated per dataset, **not** assumed 0).
* ortho camera: looks along ``-Z_P``, ``+x_P`` right, ``+y_P`` up
  (``R_PO = diag(1,-1,-1)``), pixel ``(0,0)`` at the tile's top-left.
* tile ``(i, j)`` covers
  ``x_P in [i*S*g, (i+1)*S*g]``, ``y_P in [j*S*g, (j+1)*S*g]``
  with ``g`` the GSD in metres/pixel and ``S`` the tile size.
* each tile is rendered at ``S + 2*halo`` and the centre ``S`` is kept, so
  neighbouring tiles blend over a guard band instead of cutting Gaussians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from stream_ortho.geometry import ortho_K, ortho_viewmat, R_PO


@dataclass
class OrthoGrid:
    T_WP: np.ndarray
    gsd: float                 # metres per pixel
    tile: int = 512
    halo: int = 32
    z_ground: float = 0.0
    z_top: float = 200.0       # ortho camera height above z_ground

    # -- indexing -------------------------------------------------------
    @property
    def render_size(self) -> int:
        return self.tile + 2 * self.halo

    def tile_xy(self, i: int, j: int) -> Tuple[float, float]:
        """Plane coordinates of tile ``(i, j)``'s **centre**."""
        s = self.tile * self.gsd
        return (i + 0.5) * s, (j + 0.5) * s

    def tile_window(self, i: int, j: int) -> Tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of tile ``(i, j)`` in plane coords."""
        s = self.tile * self.gsd
        return i * s, j * s, (i + 1) * s, (j + 1) * s

    def tile_of(self, x: float, y: float) -> Tuple[int, int]:
        s = self.tile * self.gsd
        return int(math.floor(x / s)), int(math.floor(y / s))

    def tiles_of_bbox(self, x0, y0, x1, y1) -> Iterable[Tuple[int, int]]:
        i0, j0 = self.tile_of(x0, y0)
        i1, j1 = self.tile_of(x1, y1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                yield i, j

    # -- camera ---------------------------------------------------------
    def viewmat(self, i: int, j: int) -> np.ndarray:
        """World->camera for tile ``(i, j)`` (4x4, float32-ready)."""
        cx, cy = self.tile_xy(i, j)
        return ortho_viewmat(self.T_WP, cx, cy, self.z_ground + self.z_top)

    def K(self) -> np.ndarray:
        """Orthographic intrinsics for the **render** size (``S + 2*halo``)."""
        n = self.render_size
        return ortho_K(self.gsd, n, n)

    # -- diagnostics ----------------------------------------------------
    def plane_to_px(self, pts_P, i: int, j: int) -> np.ndarray:
        """Plane points -> pixel coordinates **inside the cropped tile**.

        ``(0, 0)`` is the tile's top-left corner; the halo is *outside* and is
        removed by the crop, so it must not shift this mapping.
        """
        pts = np.asarray(pts_P, dtype=np.float64).reshape(-1, 3)
        s = self.tile * self.gsd
        u = (pts[:, 0] - i * s) / self.gsd
        v = (j * s + s - pts[:, 1]) / self.gsd
        return np.stack([u, v], axis=-1)

    def px_to_plane(self, u, v, i: int, j: int, z: float = None) -> np.ndarray:
        """Pixel coordinates of the cropped tile -> plane points."""
        z = self.z_ground if z is None else z
        s = self.tile * self.gsd
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        x = i * s + (u + 0.5) * self.gsd
        y = j * s + s - (v + 0.5) * self.gsd
        return np.stack([x, y, np.full_like(x, z)], axis=-1)

    def describe(self) -> dict:
        return {
            "gsd_m_per_px": self.gsd,
            "tile": self.tile,
            "halo": self.halo,
            "z_ground": self.z_ground,
            "z_top": self.z_top,
            "tile_extent_m": self.tile * self.gsd,
            "render_size": self.render_size,
            "T_WP": np.asarray(self.T_WP).tolist(),
        }


def gsd_from_altitude(altitude_m: float, fx_px: float) -> float:
    """Native GSD of a nadir image: ``altitude / focal`` (metres per pixel)."""
    return float(altitude_m) / float(fx_px)


def native_gsd_from_run(dataset, run_npz, kf: int) -> float:
    """Native GSD [m/px] of a keyframe, using the SLAM Sim3 scale and GT-free depth.

    ``depth_world = s * median(z_camera)`` is the camera-to-ground distance in
    the estimator's world units; the est->GT scale then converts it to metres.
    """
    T = run_npz["T_WC"]
    s = float(T[kf, 7]) if T.shape[1] >= 8 else 1.0
    X = run_npz["X_canon"][kf]
    depth = float(np.median(X[..., 2])) * s
    return depth / float(dataset.K_frame()[0, 0])


def footprint_bbox(T_WC, K, W: int, H: int, T_WP, z_plane: float,
                   margin_px: int = 0):
    """Plane-frame bbox of an image footprint (all corners + edge midpoints)."""
    pts = []
    xs = [0, W // 2, W] if margin_px == 0 else [-margin_px, 0, W // 2, W, W + margin_px]
    ys = [0, H // 2, H] if margin_px == 0 else [-margin_px, 0, H // 2, H, H + margin_px]
    for u in xs:
        for v in ys:
            pts.append((u, v))
    pts = np.asarray(pts, dtype=np.float64)
    # pixel -> camera ray -> plane intersection
    d = np.stack([(pts[:, 0] - K[0, 2]) / K[0, 0],
                  (pts[:, 1] - K[1, 2]) / K[1, 1],
                  np.ones(len(pts))], axis=-1)
    # world direction of each ray (camera frame = OpenCV, +z forward)
    Rw = T_WC[:3, :3]
    tw = T_WC[:3, 3]
    dirs = d @ Rw.T
    # plane: n . (p - p0) = 0
    n = T_WP[:3, 2]
    p0 = T_WP[:3, 3] + n * z_plane
    denom = dirs @ n
    ok = np.abs(denom) > 1e-9
    t = ((p0 - tw) @ n) / np.where(ok, denom, np.nan)
    pw = tw + t[:, None] * dirs
    ok &= np.isfinite(t) & (t > 0)
    if not ok.any():
        return None
    pw = pw[ok]
    pP = (pw - T_WP[:3, 3]) @ T_WP[:3, :3]      # world -> plane (rotation only)
    return (float(pP[:, 0].min()), float(pP[:, 1].min()),
            float(pP[:, 0].max()), float(pP[:, 1].max()))
