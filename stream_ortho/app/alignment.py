"""Align the live SLAM cloud to the leveled ground-plane frame.

Conventions:
    * ``T_PW`` maps World -> Plane (``P_P = T_PW @ P_W``, ground at ``z_P ≈ 0``, ``+Z_P`` up).
    * ``T_WP = inv(T_PW)`` maps Plane -> World.
    * ``PlaneAligner`` bootstraps once; ``ProgressivePlaneAligner`` additionally
      performs per-keyframe local ground-normal leveling and overlapping-surface
      elevation tie-in to eliminate inter-strip pitch/roll "bowl" drift and
      Z-undulation in monocular aerial SLAM.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np


def _rodrigues_align(n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    """Minimal 3x3 rotation matrix taking unit vector ``n_from`` to ``n_to``."""
    a = np.asarray(n_from, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(n_to, np.float64)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-deg flip around any orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, ortho)
        u /= max(np.linalg.norm(u), 1e-12)
        return 2.0 * np.outer(u, u) - np.eye(3, dtype=np.float64)
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / max(s * s, 1e-12))


def _fit_plane_ransac(P: np.ndarray, iters: int = 200, thresh: float = 0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    n = P.shape[0]
    if n < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n, bool)
    best_inl, best = None, None
    best_cnt = -1
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
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl, best = inl, (nrm, d)
    if best is None:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n, bool)
    nrm, d = best
    Q = P[best_inl]
    if Q.shape[0] >= 3:
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = vt[-1]
        d = -float(nrm @ c)
    inl = np.abs(P @ nrm + d) < thresh
    return nrm, d, inl


def _auto_ransac_thresh(P: np.ndarray, explicit: float | None = None) -> float:
    if explicit is not None and explicit > 0:
        return float(explicit)
    span = float(max(P[:, 0].ptp(), P[:, 1].ptp(), 1e-3))
    return float(min(0.08, max(0.015 * span, 1e-3)))


def _spatial_ground_cell_anchors(
    Pp: np.ndarray,
    grid_div: int = 28,
    min_pts_per_cell: int = 2,
    pct: float = 15.0,
) -> np.ndarray:
    """Return one low-percentile anchor per coarse 2D cell to resist rooftop dominance."""
    if Pp.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if Pp.shape[0] > 32768:
        Pp = Pp[np.linspace(0, Pp.shape[0] - 1, 32768, dtype=np.int64)]
    span = float(max(Pp[:, 0].ptp(), Pp[:, 1].ptp(), 1e-3))
    cell = max(span / float(max(grid_div, 4)), 0.05)
    ix = np.floor(Pp[:, 0] / cell).astype(np.int64)
    iy = np.floor(Pp[:, 1] / cell).astype(np.int64)
    # Bias away from signed-overflow collisions cleanly
    keys = (ix + 10_000_000) * 20_000_000 + (iy + 10_000_000)
    order = np.argsort(keys, kind="mergesort")
    ks = keys[order]
    ps = Pp[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], ks.size]
    anchors = []
    for s0, s1 in zip(starts, ends):
        if s1 - s0 < min_pts_per_cell:
            continue
        sub = ps[s0:s1]
        z_a = float(np.percentile(sub[:, 2], pct))
        x_a = float(np.median(sub[:, 0]))
        y_a = float(np.median(sub[:, 1]))
        anchors.append((x_a, y_a, z_a))
    if not anchors:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(anchors, dtype=np.float64)


def _fit_residual_plane_ransac(
    xy: np.ndarray,
    r: np.ndarray,
    iters: int = 120,
    thresh: float = 0.05,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``r(x, y) = a*x + b*y + c`` by RANSAC over 2D correspondences."""
    rng = rng or np.random.default_rng(0)
    n = xy.shape[0]
    if n < 3:
        return np.array([0.0, 0.0, float(np.median(r)) if n else 0.0]), np.ones(n, dtype=bool)
    span_x = float(max(xy[:, 0].ptp(), 1e-3))
    span_y = float(max(xy[:, 1].ptp(), 1e-3))
    cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    x = (xy[:, 0] - cx) / span_x
    y = (xy[:, 1] - cy) / span_y

    best_cnt = -1
    best_inl = None
    best_coef = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        M = np.array(
            [
                [x[idx[0]], y[idx[0]], 1.0],
                [x[idx[1]], y[idx[1]], 1.0],
                [x[idx[2]], y[idx[2]], 1.0],
            ],
            dtype=np.float64,
        )
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        coef_n = np.linalg.solve(M, r[idx])
        pred = coef_n[0] * x + coef_n[1] * y + coef_n[2]
        inl = np.abs(r - pred) < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
            best_coef = coef_n

    if best_inl is None or best_cnt < 6:
        med = float(np.median(r))
        return np.array([0.0, 0.0, med], dtype=np.float64), (np.abs(r - med) < thresh)

    # Unscale back to raw (xy[:, 0], xy[:, 1]) via least-squares on inliers
    A = np.column_stack([xy[best_inl, 0], xy[best_inl, 1], np.ones(best_cnt, dtype=np.float64)])
    coef, _, _, _ = np.linalg.lstsq(A, r[best_inl], rcond=None)
    pred_full = coef[0] * xy[:, 0] + coef[1] * xy[:, 1] + coef[2]
    inl = np.abs(r - pred_full) < thresh
    return coef.astype(np.float64), inl


def estimate_ground_plane(Pw, ransac_iters: int = 200, thresh: float | None = None,
                          low_pct=(5, 60), max_sample: int = 8192,
                          cam_pos=None):
    """Fit the dominant ground plane of a world-frame cloud.

    Returns ``(T_PW, z_ground, info)`` where ``T_PW`` maps world points to the
    leveled plane frame (``+Z`` up toward the camera, ground near ``z = 0``).
    When ``cam_pos`` is supplied, the approximate "sky/up" direction is
    inferred from ``cam_pos - cloud_center`` so camera-frame point clouds
    (where ``+Z_C`` points down into the ground) are oriented right-side up.
    """
    P = np.asarray(Pw, np.float64).reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    if P.shape[0] < 50:
        raise ValueError("not enough points to estimate a plane")

    if cam_pos is not None:
        cp = np.asarray(cam_pos, np.float64).reshape(-1, 3)
        v_up = cp.mean(0) - P.mean(0)
        vn = float(np.linalg.norm(v_up))
        v_up = v_up / vn if vn > 1e-9 else np.array([0.0, 0.0, 1.0])
    else:
        v_up = np.array([0.0, 0.0, 1.0])

    # Project onto approximate up-vector so ground is always the lower band
    h_approx = P @ v_up
    _, hi = np.percentile(h_approx, low_pct)
    cand = P[h_approx <= hi]
    if cand.shape[0] < 50:
        cand = P
    if cand.shape[0] > max_sample:
        cand_sub = cand[np.linspace(0, cand.shape[0] - 1, max_sample, dtype=np.int64)]
    else:
        cand_sub = cand

    th = _auto_ransac_thresh(cand_sub, thresh)
    nrm, d, inl = _fit_plane_ransac(cand_sub, ransac_iters, th)
    # Orient plane normal toward the camera (+Z_P up toward sky)
    if float(np.dot(nrm, v_up)) < 0.0:
        nrm, d = -nrm, -d
    R_init = _rodrigues_align(nrm, np.array([0.0, 0.0, 1.0]))

    # Spatial-binned refinement prevents dense rooftop clusters in bootstrap frames
    # from biasing the ground normal.
    P_sub_full = P if P.shape[0] <= 32768 else P[np.linspace(0, P.shape[0] - 1, 32768, dtype=np.int64)]
    anc_p = _spatial_ground_cell_anchors(P_sub_full @ R_init.T, grid_div=24, min_pts_per_cell=3, pct=15.0)
    if anc_p.shape[0] >= 24:
        anc_w = anc_p @ R_init
        th_a = _auto_ransac_thresh(anc_w, thresh)
        nrm_a, d_a, inl_a = _fit_plane_ransac(anc_w, ransac_iters, th_a)
        if float(np.dot(nrm_a, v_up)) < 0.0:
            nrm_a, d_a = -nrm_a, -d_a
        if inl_a.mean() >= 0.45:
            nrm, d, inl, th = nrm_a, d_a, inl_a, th_a

    R = _rodrigues_align(nrm, np.array([0.0, 0.0, 1.0]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    Pp = P @ R.T
    z_ground = float(np.percentile(Pp[:, 2], 5))
    T[:3, 3] = np.array([0.0, 0.0, -z_ground])
    info = {
        "normal": nrm.tolist(),
        "d": float(d),
        "thresh": float(th),
        "inlier_frac": float(inl.mean()),
        "z_ground": z_ground,
        "n_points": int(P.shape[0]),
    }
    return T, z_ground, info


@dataclass
class PlaneAligner:
    """Rigid World->Plane aligner (bootstraps once, then maps subsequent frames)."""

    mode: str = "auto"            # auto | gt | plane | identity
    T_WP: np.ndarray | None = None
    bootstrap_kfs: int = 5
    thresh: float | None = None

    def __post_init__(self):
        self._buffer = []
        self._cam_buffer = []
        self._T = None            # T_PW: World -> Plane
        self._info = {}
        if self.mode in ("gt", "identity") or self.T_WP is not None:
            if self.mode == "identity" and self.T_WP is None:
                self._T = np.eye(4, dtype=np.float64)
            elif self.T_WP is not None:
                # Input T_WP may be Plane->World (project convention); invert for T_PW
                self._T = np.linalg.inv(np.asarray(self.T_WP, np.float64))
            self._info = {
                "mode": self.mode,
                "source": "config" if self.T_WP is not None else "identity",
            }

    @property
    def ready(self) -> bool:
        return self._T is not None

    @property
    def T_PW(self) -> np.ndarray:
        if self._T is None:
            raise RuntimeError("alignment not ready")
        return self._T

    @property
    def T_WP_mat(self) -> np.ndarray:
        return np.linalg.inv(self.T_PW)

    def add(self, Pw, cam_pos=None) -> bool:
        """Buffer a cloud; returns True once the transform is available."""
        if self._T is not None:
            return True
        self._buffer.append(np.asarray(Pw, np.float64).reshape(-1, 3))
        if cam_pos is not None:
            self._cam_buffer.append(np.asarray(cam_pos, np.float64).reshape(3))
        if len(self._buffer) >= max(int(self.bootstrap_kfs), 1):
            P = np.vstack(self._buffer)
            cp = np.vstack(self._cam_buffer) if self._cam_buffer else None
            self._T, _, info = estimate_ground_plane(P, thresh=self.thresh, cam_pos=cp)
            self._info = {
                "mode": self.mode,
                "source": "estimated",
                "bootstrap_frames": len(self._buffer),
                **info,
            }
            self._buffer = []
            self._cam_buffer = []
            return True
        return False

    def to_plane(self, Pw) -> np.ndarray:
        if self._T is None:
            raise RuntimeError("alignment not ready")
        P = np.asarray(Pw, np.float64).reshape(-1, 3)
        return P @ self._T[:3, :3].T + self._T[:3, 3]

    def record(self) -> dict:
        return dict(self._info)

    def save(self, path):
        d = {"mode": self.mode, **self.record()}
        if self._T is not None:
            # Keep "T_WP" for legacy readers AND explicitly record both matrices
            d["T_PW"] = self._T.tolist()
            d["T_WP"] = np.linalg.inv(self._T).tolist()
        pathlib.Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return d


@dataclass
class ProgressivePlaneAligner(PlaneAligner):
    """Per-keyframe ground-plane leveling & elevation tie-in on top of global alignment.

    Fixes multi-strip pitch/roll drift and vertical bow in monocular aerial SLAM:
    for each keyframe, estimates the local ground plane normal in the global plane
    frame, levels any residual tilt (> ``min_tilt_deg``) around the local ground
    centroid, and aligns residual vertical drift against a dedicated coarse
    ground-inlier elevation grid so adjacent strips never stack or separate in Z.
    """

    min_tilt_deg: float = 0.5
    max_tilt_deg: float = 35.0
    min_inlier_frac: float = 0.35
    ransac_iters: int = 80
    rect_history: dict = field(default_factory=dict)
    rect_info_history: dict = field(default_factory=dict)
    _kf_gnd_w: dict = field(default_factory=dict)
    _kf_T_WC_init: dict = field(default_factory=dict)
    _gnd_map: dict = field(default_factory=dict)
    _gnd_cell: float = 0.0

    def save(self, path):
        d = {"mode": self.mode, **self.record()}
        if self._T is not None:
            d["T_PW"] = self._T.tolist()
            d["T_WP"] = np.linalg.inv(self._T).tolist()
        if self.rect_info_history:
            d["per_kf"] = [self.rect_info_history[k] for k in sorted(self.rect_info_history)]
        pathlib.Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return d

    def estimate_global_relevel(
        self,
        map_pts_p: np.ndarray,
        min_eig_ratio: float = 0.08,
        min_tilt_deg: float = 0.5,
        cam_pos_p: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Compute a rigid Plane-frame correction leveling global ground onto ``z=0``.

        Requires wide 2D multi-strip coverage (checking trajectory camera centers when
        supplied so single strips never trigger cross-roll) before global plane fitting.
        """
        if not self.ready:
            return None
        if cam_pos_p is not None:
            cp = np.asarray(cam_pos_p, dtype=np.float64).reshape(-1, 3)
            if cp.shape[0] < 12:
                return None
            c_xy = cp[:, :2] - cp[:, :2].mean(0, keepdims=True)
            c_cov = (c_xy.T @ c_xy) / max(cp.shape[0] - 1, 1)
            c_eigs = np.linalg.eigvalsh(c_cov)
            if float(c_eigs[0]) / max(float(c_eigs[1]), 1e-9) < 0.05:
                return None

        P = np.asarray(map_pts_p, dtype=np.float64).reshape(-1, 3)
        P = P[np.isfinite(P).all(1)]
        if P.shape[0] < 500:
            return None

        anc = _spatial_ground_cell_anchors(P, grid_div=32, min_pts_per_cell=4, pct=15.0)
        if anc.shape[0] < 48:
            return None

        xy = anc[:, :2] - anc[:, :2].mean(0, keepdims=True)
        cov = (xy.T @ xy) / max(anc.shape[0] - 1, 1)
        eigs = np.linalg.eigvalsh(cov)
        if float(eigs[0]) / max(float(eigs[1]), 1e-9) < float(min_eig_ratio):
            return None

        span = float(max(anc[:, 0].ptp(), anc[:, 1].ptp(), 1e-3))
        th = max(0.05, 0.012 * span)
        nrm, _, inl = _fit_plane_ransac(anc, iters=max(self.ransac_iters, 160), thresh=th)
        if nrm[2] < 0:
            nrm = -nrm
        inl_frac = float(inl.mean()) if inl.size else 0.0
        tilt_deg = float(np.degrees(np.arccos(np.clip(nrm[2], -1.0, 1.0))))
        if inl_frac < 0.55 or tilt_deg > 25.0:
            return None

        inl_anc = anc[inl] if inl.any() else anc
        pivot = inl_anc.mean(0)
        R_lvl = _rodrigues_align(nrm, np.array([0.0, 0.0, 1.0]))
        rot_anc = (inl_anc - pivot) @ R_lvl.T + pivot
        dz = -float(np.median(rot_anc[:, 2]))
        if tilt_deg < float(min_tilt_deg) and abs(dz) < 0.02:
            return None

        dT = np.eye(4, dtype=np.float64)
        if tilt_deg >= float(min_tilt_deg):
            dT[:3, :3] = R_lvl
            dT[:3, 3] = pivot - R_lvl @ pivot
        else:
            rot_anc = inl_anc
            dz = -float(np.median(rot_anc[:, 2]))
        dT[2, 3] += dz
        return dT

    def apply_global_plane_transform(
        self,
        dT_P: np.ndarray,
        map_pts_p: np.ndarray | None = None,
    ) -> None:
        """Apply rigid Plane-frame re-leveling to base alignment, per-kf cache, and _gnd_map."""
        dT = np.asarray(dT_P, dtype=np.float64)
        if self._T is not None:
            self._T = dT @ self._T
        for kf_k, T_old in list(self.rect_history.items()):
            self.rect_history[kf_k] = dT @ T_old
        self._gnd_map.clear()
        if map_pts_p is not None and self._gnd_cell > 0.0:
            anc = _spatial_ground_cell_anchors(
                np.asarray(map_pts_p, dtype=np.float64),
                grid_div=32,
                min_pts_per_cell=3,
                pct=15.0,
            )
            if anc.shape[0] > 0:
                gx = np.floor(anc[:, 0] / self._gnd_cell).astype(np.int64)
                gy = np.floor(anc[:, 1] / self._gnd_cell).astype(np.int64)
                for px_i, py_i, pz_v in zip(gx.tolist(), gy.tolist(), anc[:, 2].tolist()):
                    self._gnd_map[(px_i, py_i)] = float(pz_v)

    def _query_smooth_map_matches(
        self,
        Pp_sub: np.ndarray,
        surface_ref,
        exclude_kf: int | None = None,
        edge_tol: float = 0.12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query committed map heights on locally flat interior regions (exclude walls/cliffs)."""
        xs = Pp_sub[:, 0]
        ys = Pp_sub[:, 1]
        if exclude_kf is not None and hasattr(surface_ref, "query_elevation_excluding"):
            z_map, hit = surface_ref.query_elevation_excluding(xs, ys, int(exclude_kf))
        else:
            z_map, hit = surface_ref.query_elevation(xs, ys)
        if not np.any(hit):
            return np.zeros(Pp_sub.shape[0], dtype=bool), np.zeros(0, dtype=np.float64)

        step = max(float(getattr(surface_ref, "cell_2d", 0.01)) * 2.0, 1e-3)
        smooth = hit.copy()
        for dx_n, dy_n in ((-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)):
            if exclude_kf is not None and hasattr(surface_ref, "query_elevation_excluding"):
                zn, hn = surface_ref.query_elevation_excluding(xs + dx_n, ys + dy_n, int(exclude_kf))
            else:
                zn, hn = surface_ref.query_elevation(xs + dx_n, ys + dy_n)
            # Only reject populated cliff edges so sparse synthetic maps still match cleanly
            smooth &= (~hn) | (np.abs(zn - z_map) <= float(edge_tol))
        return smooth, z_map.astype(np.float64)

    def _fit_rect_against_map(
        self,
        Pp_sub: np.ndarray,
        surface_ref,
        exclude_kf: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
        """Align ``Pp_sub`` directly to overlapping smooth committed map surfaces."""
        if surface_ref is None or getattr(surface_ref, "n_points", 0) <= 0:
            return None
        span = float(max(Pp_sub[:, 0].ptp(), Pp_sub[:, 1].ptp(), 1e-3))
        if self._gnd_cell <= 0.0:
            self._gnd_cell = max(span / 40.0, 1e-3)

        smooth, z_map_all = self._query_smooth_map_matches(
            Pp_sub, surface_ref, exclude_kf=exclude_kf, edge_tol=0.12
        )
        n_ov = int(smooth.sum())
        if n_ov < 48:
            return None

        ov_pts = Pp_sub[smooth]
        z_map = z_map_all[smooth]
        ov_span = float(max(ov_pts[:, 0].ptp(), ov_pts[:, 1].ptp(), 1e-3))
        if ov_span < 0.18 * span:
            return None

        r = z_map - ov_pts[:, 2]
        z_band = float(getattr(surface_ref, "z_band", 0.04))
        th_r = max(0.05, 0.015 * ov_span, min(1.5 * z_band, 0.05 * span))
        coef, inl_ov = _fit_residual_plane_ransac(
            ov_pts[:, :2],
            r,
            iters=max(int(self.ransac_iters), 120),
            thresh=th_r,
        )
        inl_frac = float(inl_ov.mean()) if inl_ov.size else 0.0
        a, b = float(coef[0]), float(coef[1])
        res_tilt_deg = float(np.degrees(np.arctan(np.hypot(a, b))))

        med_all = float(np.median(r))
        mad_dz_only = float(np.median(np.abs(r - med_all)))

        inl_pts = ov_pts[inl_ov] if inl_ov.any() else ov_pts
        z_map_inl = z_map[inl_ov] if inl_ov.any() else z_map
        eig_ratio = 0.0
        if inl_pts.shape[0] >= 12:
            xy_c = inl_pts[:, :2] - inl_pts[:, :2].mean(0, keepdims=True)
            cov = (xy_c.T @ xy_c) / max(inl_pts.shape[0] - 1, 1)
            eigs = np.linalg.eigvalsh(cov)
            eig_ratio = float(eigs[0]) / max(float(eigs[1]), 1e-9)

        T_rect = np.eye(4, dtype=np.float64)
        rectified = False
        max_tilt_cap = min(float(self.max_tilt_deg), 25.0)
        if (
            inl_frac >= max(self.min_inlier_frac, 0.45)
            and max(float(self.min_tilt_deg), 2.5) <= res_tilt_deg <= max_tilt_cap
            and eig_ratio >= 0.04
            and mad_dz_only > max(0.05, 0.02 * span)
        ):
            n_rel = np.array([a, b, 1.0], dtype=np.float64)
            n_rel /= max(float(np.linalg.norm(n_rel)), 1e-12)
            R_lvl = _rodrigues_align(n_rel, np.array([0.0, 0.0, 1.0]))
            pivot = inl_pts.mean(0)
            ov_rot = (ov_pts - pivot) @ R_lvl.T + pivot
            r_rot_inl = z_map_inl - ((inl_pts - pivot) @ R_lvl.T + pivot)[:, 2]
            dz_cand = float(np.median(r_rot_inl))
            mad_after_rot = float(np.median(np.abs((z_map - ov_rot[:, 2]) - dz_cand)))
            flat_ok = True
            if res_tilt_deg > 4.0 and Pp_sub.shape[0] >= 32:
                lo_cut = float(np.percentile(Pp_sub[:, 2], 45))
                lo_m = Pp_sub[:, 2] <= lo_cut
                if int(lo_m.sum()) >= 16:
                    std_before = float(np.std(Pp_sub[lo_m, 2]))
                    std_after = float(
                        np.std(((Pp_sub[lo_m] - pivot) @ R_lvl.T + pivot)[:, 2])
                    )
                    flat_ok = (std_before > 0.06) and (std_after < 0.70 * std_before)
            if (
                flat_ok
                and mad_after_rot < 0.65 * mad_dz_only
                and mad_after_rot < max(0.06, 0.04 * span)
            ):
                T_rect[:3, :3] = R_lvl
                T_rect[:3, 3] = pivot - R_lvl @ pivot
                inl_pts = (inl_pts - pivot) @ R_lvl.T + pivot
                rectified = True

        r_after = z_map_inl - inl_pts[:, 2]
        med_after = float(np.median(r_after))
        mad_after = float(np.median(np.abs(r_after - med_after)))
        dz = 0.0
        tie_rejected = False
        max_dz = max(1.5 * span, 0.5) if (mad_after < 0.06 * span and inl_frac >= 0.55) else 0.25 * span
        dz_deadband = max(0.06, min(0.02 * span, 0.8 * z_band))
        if (
            inl_frac >= max(self.min_inlier_frac, 0.40)
            and abs(med_after) < max_dz
            and mad_after < 0.12 * span
        ):
            if abs(med_after) >= dz_deadband:
                dz = med_after
                inl_pts = inl_pts.copy()
                inl_pts[:, 2] += dz
                T_rect[2, 3] += dz
        elif abs(med_after) >= max_dz:
            tie_rejected = True

        # Keep full-length mask for _kf_gnd_w caching
        inl_full = np.zeros(Pp_sub.shape[0], dtype=bool)
        ov_idx = np.nonzero(smooth)[0]
        inl_full[ov_idx[inl_ov]] = True

        # Update coarse _gnd_map from verified low-band ground matches only
        if inl_pts.shape[0] >= 16 and self._gnd_cell > 0.0:
            z_lo = np.percentile(inl_pts[:, 2], 35)
            low_pts = inl_pts[inl_pts[:, 2] <= z_lo + 0.05]
            if low_pts.shape[0] > 0:
                gx = np.floor(low_pts[:, 0] / self._gnd_cell).astype(np.int64)
                gy = np.floor(low_pts[:, 1] / self._gnd_cell).astype(np.int64)
                buckets: dict = {}
                for kx_i, ky_i, kz_v in zip(gx.tolist(), gy.tolist(), low_pts[:, 2].tolist()):
                    buckets.setdefault((kx_i, ky_i), []).append(kz_v)
                for k_xy, vals in buckets.items():
                    z_m = float(np.median(vals))
                    if k_xy not in self._gnd_map:
                        self._gnd_map[k_xy] = z_m
                    else:
                        self._gnd_map[k_xy] = 0.7 * self._gnd_map[k_xy] + 0.3 * z_m

        info = {
            "tilt_deg": res_tilt_deg,
            "dz": dz,
            "inlier_frac": inl_frac,
            "rectified": rectified,
            "tie_rejected": tie_rejected,
        }
        return T_rect, inl_full, inl_pts, info

    def estimate_residual_rect(self, pts_p: np.ndarray, conf: np.ndarray | None = None) -> dict:
        """Read-only residual ground tilt & vertical offset vs verified _gnd_map."""
        Pp = np.asarray(pts_p, dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(Pp).all(1)
        if conf is not None:
            c = np.asarray(conf, np.float64).ravel()
            if c.size == Pp.shape[0] and valid.any():
                valid &= np.isfinite(c) & (c >= np.percentile(c[valid], 25))
        Pv = Pp[valid]
        if Pv.shape[0] < 100:
            return {"res_tilt_deg": 0.0, "res_dz": 0.0, "inlier_frac": 0.0}
        z_hi = np.percentile(Pv[:, 2], 35)
        cand = Pv[Pv[:, 2] <= z_hi]
        if cand.shape[0] < 60:
            cand = Pv
        if cand.shape[0] > 4096:
            idx_sub = np.linspace(0, cand.shape[0] - 1, 4096, dtype=np.int64)
            cand_sub = cand[idx_sub]
        else:
            cand_sub = cand
        span = float(max(cand_sub[:, 0].ptp(), cand_sub[:, 1].ptp(), 1e-3))
        th = _auto_ransac_thresh(cand_sub, self.thresh)
        nrm, _, inl = _fit_plane_ransac(cand_sub, iters=self.ransac_iters, thresh=th)
        if nrm[2] < 0:
            nrm = -nrm
        inl_frac = float(inl.mean()) if inl.size else 0.0
        tilt_deg = float(np.degrees(np.arccos(np.clip(nrm[2], -1.0, 1.0))))
        inl_pts = cand_sub[inl] if inl.any() else cand_sub
        res_dz = 0.0
        if self._gnd_map and self._gnd_cell > 0.0:
            g_cell = self._gnd_cell
            gx = np.floor(inl_pts[:, 0] / g_cell).astype(np.int64)
            gy = np.floor(inl_pts[:, 1] / g_cell).astype(np.int64)
            diffs = [
                self._gnd_map[(int(px_i), int(py_i))] - float(pz_i)
                for px_i, py_i, pz_i in zip(gx, gy, inl_pts[:, 2])
                if (int(px_i), int(py_i)) in self._gnd_map
            ]
            if len(diffs) >= 16:
                arr = np.asarray(diffs, dtype=np.float64)
                med_diff = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med_diff)))
                if abs(med_diff) < 0.25 * span and mad < 0.12 * span:
                    res_dz = med_diff
        return {"res_tilt_deg": tilt_deg, "res_dz": res_dz, "inlier_frac": inl_frac}

    def _fit_rect_on_plane_ground(
        self,
        cand_sub: np.ndarray,
        update_gnd_map: bool = True,
        max_tilt_cap: float = 12.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Fallback local ground leveling & pure-Z tie-in when no surface_ref overlap exists."""
        span = float(max(cand_sub[:, 0].ptp(), cand_sub[:, 1].ptp(), 1e-3))
        if self._gnd_cell <= 0.0:
            self._gnd_cell = max(span / 40.0, 1e-3)

        th = _auto_ransac_thresh(cand_sub, self.thresh)
        nrm, _, inl = _fit_plane_ransac(cand_sub, iters=self.ransac_iters, thresh=th)
        if nrm[2] < 0:
            nrm = -nrm
        inl_frac = float(inl.mean()) if inl.size else 0.0
        tilt_rad = float(np.arccos(np.clip(nrm[2], -1.0, 1.0)))
        tilt_deg = float(np.degrees(tilt_rad))

        T_rect = np.eye(4, dtype=np.float64)
        inl_pts = cand_sub[inl] if inl.any() else cand_sub
        rectified = False
        if (inl_frac >= max(self.min_inlier_frac, 0.45)
                and max(float(self.min_tilt_deg), 2.0) <= tilt_deg <= float(max_tilt_cap)):
            pivot = inl_pts.mean(0)
            R_lvl = _rodrigues_align(nrm, np.array([0.0, 0.0, 1.0]))
            T_rect[:3, :3] = R_lvl
            T_rect[:3, 3] = pivot - R_lvl @ pivot
            inl_pts = (inl_pts - pivot) @ R_lvl.T + pivot
            rectified = True

        dz = 0.0
        g_cell = self._gnd_cell
        gx = np.floor(inl_pts[:, 0] / g_cell).astype(np.int64)
        gy = np.floor(inl_pts[:, 1] / g_cell).astype(np.int64)
        if self._gnd_map:
            diffs = []
            for px_i, py_i, pz_i in zip(gx, gy, inl_pts[:, 2]):
                ref_z = self._gnd_map.get((int(px_i), int(py_i)))
                if ref_z is not None:
                    diffs.append(ref_z - float(pz_i))
            if len(diffs) >= 16:
                arr = np.asarray(diffs, dtype=np.float64)
                med_diff = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med_diff)))
                if abs(med_diff) < 0.25 * span and mad < 0.12 * span:
                    dz = med_diff
        elif inl_frac >= 0.45:
            dz = -float(np.median(inl_pts[:, 2]))

        if dz != 0.0:
            inl_pts = inl_pts.copy()
            inl_pts[:, 2] += dz
            T_rect[2, 3] += dz

        if update_gnd_map and inl_frac >= max(self.min_inlier_frac, 0.40):
            keys = list(zip(gx.tolist(), gy.tolist()))
            buckets: dict = {}
            for k_xy, z_val in zip(keys, inl_pts[:, 2].tolist()):
                buckets.setdefault(k_xy, []).append(z_val)
            for k_xy, vals in buckets.items():
                z_m = float(np.median(vals))
                if k_xy not in self._gnd_map:
                    self._gnd_map[k_xy] = z_m
                else:
                    self._gnd_map[k_xy] = 0.7 * self._gnd_map[k_xy] + 0.3 * z_m

        info = {
            "tilt_deg": tilt_deg,
            "dz": dz,
            "inlier_frac": inl_frac,
            "rectified": rectified,
        }
        return T_rect, inl, inl_pts, info

    def realign_after_ba(
        self,
        kf_id: int,
        T_WC_old: np.ndarray,
        T_WC_new: np.ndarray,
        enabled: bool = True,
        surface_ref=None,
    ) -> np.ndarray | None:
        """Recompute T_rect from scratch after a BA pose update (replaces, never stacks)."""
        if not enabled or not self.ready:
            return None
        from stream_ortho.geometry import sim3_matrix

        kf_i = int(kf_id)
        gnd_w_init = self._kf_gnd_w.get(kf_i)
        if gnd_w_init is None or gnd_w_init.shape[0] < 60:
            return None

        T_init = self._kf_T_WC_init.setdefault(kf_i, np.asarray(T_WC_old, dtype=np.float64).copy())
        M_init = sim3_matrix(T_init)
        M_new = sim3_matrix(np.asarray(T_WC_new, dtype=np.float64))
        dM_W = M_new @ np.linalg.inv(M_init)
        gnd_w_new = gnd_w_init @ dM_W[:3, :3].T + dM_W[:3, 3]

        cand_p = self.to_plane(gnd_w_new)
        fit = None
        if surface_ref is not None and getattr(surface_ref, "n_points", 0) > 0:
            fit = self._fit_rect_against_map(cand_p, surface_ref, exclude_kf=kf_i)
        if fit is not None:
            T_rect_new, _, _, fit_info = fit
        else:
            T_rect_new, _, _, fit_info = self._fit_rect_on_plane_ground(
                cand_p, update_gnd_map=False, max_tilt_cap=12.0
            )
        T_PW_new = T_rect_new @ self.T_PW
        self.rect_history[kf_i] = T_PW_new
        prev_info = self.rect_info_history.get(kf_i, {})
        self.rect_info_history[kf_i] = {
            **prev_info,
            "kf": kf_i,
            **fit_info,
            "t_rect": T_rect_new.tolist(),
            "realigned_ba": True,
        }
        return T_PW_new

    def rectify_keyframe(
        self,
        kf_id: int,
        Pw: np.ndarray,
        conf: np.ndarray | None = None,
        surface_ref=None,
        enabled: bool = True,
        return_points: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray, dict]:
        """Map ``Pw`` into the leveled global plane frame with map-relative rectification."""
        Pw_arr = np.asarray(Pw, dtype=np.float64).reshape(-1, 3)
        Pp = self.to_plane(Pw_arr)
        T_PW_base = self.T_PW
        info = {
            "kf": int(kf_id),
            "tilt_deg": 0.0,
            "dz": 0.0,
            "inlier_frac": 0.0,
            "rectified": False,
        }
        if not enabled:
            self.rect_history[int(kf_id)] = T_PW_base.copy()
            self.rect_info_history[int(kf_id)] = dict(info)
            return (Pp if return_points else None), T_PW_base, info

        valid = np.isfinite(Pp).all(1)
        if conf is not None:
            c = np.asarray(conf, np.float64).ravel()
            if c.size == Pp.shape[0] and valid.any():
                valid &= np.isfinite(c) & (c >= np.percentile(c[valid], 25))
        Pv = Pp[valid]
        Pw_v = Pw_arr[valid]
        if Pv.shape[0] < 100:
            self.rect_history[int(kf_id)] = T_PW_base.copy()
            return (Pp if return_points else None), T_PW_base, info

        if surface_ref is not None and getattr(surface_ref, "n_points", 0) > 0:
            if Pv.shape[0] > 4096:
                idx_s = np.linspace(0, Pv.shape[0] - 1, 4096, dtype=np.int64)
                Pv_sub = Pv[idx_s]
                Pw_sub = Pw_v[idx_s]
            else:
                Pv_sub = Pv
                Pw_sub = Pw_v
            fit = self._fit_rect_against_map(Pv_sub, surface_ref)
            if fit is not None:
                T_rect, inl_full, _, fit_info = fit
                info.update(fit_info)
                inl_w = Pw_sub[inl_full] if inl_full.any() else Pw_sub
                if inl_w.shape[0] > 2048:
                    inl_w = inl_w[np.linspace(0, inl_w.shape[0] - 1, 2048, dtype=np.int64)]
                self._kf_gnd_w[int(kf_id)] = np.asarray(inl_w, dtype=np.float64).copy()
            else:
                # Sparse unexplored frontier when map exists: keep pure base transform without
                # risky blind p35 tilting onto roofs/trees.
                T_rect = np.eye(4, dtype=np.float64)
                z_hi = np.percentile(Pv[:, 2], 35)
                cand_w = Pw_v[Pv[:, 2] <= z_hi]
                if cand_w.shape[0] < 60:
                    cand_w = Pw_v
                if cand_w.shape[0] > 2048:
                    cand_w = cand_w[np.linspace(0, cand_w.shape[0] - 1, 2048, dtype=np.int64)]
                self._kf_gnd_w[int(kf_id)] = np.asarray(cand_w, dtype=np.float64).copy()
        else:
            z_hi = np.percentile(Pv[:, 2], 35)
            low_mask = Pv[:, 2] <= z_hi
            cand = Pv[low_mask]
            cand_w = Pw_v[low_mask]
            if cand.shape[0] < 60:
                cand = Pv
                cand_w = Pw_v
            if cand.shape[0] > 4096:
                idx_sub = np.linspace(0, cand.shape[0] - 1, 4096, dtype=np.int64)
                cand_sub = cand[idx_sub]
                cand_w_sub = cand_w[idx_sub]
            else:
                cand_sub = cand
                cand_w_sub = cand_w

            T_rect, inl, _, fit_info = self._fit_rect_on_plane_ground(
                cand_sub, update_gnd_map=True, max_tilt_cap=12.0
            )
            info.update(fit_info)

            inl_w = cand_w_sub[inl] if inl.any() else cand_w_sub
            if inl_w.shape[0] > 2048:
                inl_w = inl_w[np.linspace(0, inl_w.shape[0] - 1, 2048, dtype=np.int64)]
            self._kf_gnd_w[int(kf_id)] = np.asarray(inl_w, dtype=np.float64).copy()

        T_PW_kf = T_rect @ T_PW_base
        self.rect_history[int(kf_id)] = T_PW_kf
        info["t_rect"] = T_rect.tolist()
        self.rect_info_history[int(kf_id)] = dict(info)
        if not return_points:
            return None, T_PW_kf, info
        Pp = Pp @ T_rect[:3, :3].T + T_rect[:3, 3]
        return Pp, T_PW_kf, info
