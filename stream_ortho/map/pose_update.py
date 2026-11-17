"""Loop-closure / pose-update support (design doc §7.8 items 1-3).

Model: every Gaussian belongs to one **owner** keyframe (``owner == kf_id``).  A
pose update moves the owner's Gaussians rigidly with ``Δ = T_new ⊗ T_old⁻¹``:

* means:   ``p' = s·R·p + t``
* quats:   ``q' = q_Δ ⊗ q``   (wxyz, so the splat orientation follows the camera)
* scales:  multiplied by ``s`` (a Sim3 update also rescales)

Cross-group seams are handled by the tile scheduler: the affected tiles are
re-rendered; Gaussians outside the group are **bit-identical**, which is what
makes the incremental update tear-free (verified in ``test_pose_update.py``).

Deliberate limitation: the map stores a single anchor per Gaussian, so a
Gaussian observed by two keyframes follows the one it was born from.  The
"secondary anchor deviation -> re-fuse" rule of §7.8 needs per-observation
anchors and is left to the week-8 fusion step; the interface below returns the
per-group ``max_deviation`` so that rule can be plugged in.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from stream_ortho.geometry import quat_mul_wxyz, sim3_from_vec8, sim3_matrix


def sim3_vec_from_matrix(T4: np.ndarray) -> np.ndarray:
    """8-vector (lietorch layout) from a 4x4 with the scale folded in."""
    T = np.asarray(T4, np.float64)
    A = T[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(A), 1e-12)))
    R = A / s
    from stream_ortho.geometry import R_to_quat_xyzw

    q = R_to_quat_xyzw(R)
    return np.concatenate([T[:3, 3], q, [s]])


def world_delta(T_old_wc: np.ndarray, T_new_wc: np.ndarray) -> np.ndarray:
    """``Δ = T_new ⊗ T_old⁻¹`` as an 8-vector (world-frame camera poses)."""
    d = np.asarray(T_new_wc, np.float64) @ np.linalg.inv(np.asarray(T_old_wc, np.float64))
    return sim3_vec_from_matrix(d)


def invert_delta(delta: np.ndarray) -> np.ndarray:
    """Inverse of an 8-vector Sim3 delta."""
    return sim3_vec_from_matrix(np.linalg.inv(sim3_matrix(delta)))


def apply_pose_update(gmap, kf_ids: Iterable[int], delta: np.ndarray,
                      rescale: bool = True) -> dict:
    """Move every Gaussian owned by ``kf_ids`` by ``delta`` (in place)."""
    ids = np.asarray(sorted(set(int(k) for k in kf_ids)), np.int32)
    res = {"kf_ids": ids.tolist(), "n_moved": 0, "max_displacement_m": 0.0,
           "max_rotation_deg": 0.0}
    if gmap.n_committed == 0 or len(ids) == 0:
        return res

    owner = gmap.committed["owner"]
    sel = np.isin(owner, ids)
    if not sel.any():
        return res
    idx = np.where(sel)[0]

    R, t, s = sim3_from_vec8(delta)
    from stream_ortho.geometry import R_to_quat_wxyz

    qd = R_to_quat_wxyz(R)

    means = gmap.committed["means"]
    old = means[idx].copy()
    moved = float(s) * (old @ R.T) + t
    means[idx] = moved.astype(np.float32)

    quats = gmap.committed["quats"]
    q = quats[idx].astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    q_new = quat_mul_wxyz(np.broadcast_to(qd, q.shape), q)
    q_new /= np.maximum(np.linalg.norm(q_new, axis=1, keepdims=True), 1e-9)
    quats[idx] = q_new.astype(np.float32)

    if rescale and abs(float(s) - 1.0) > 1e-9:
        gmap.committed["scales"][idx] = (gmap.committed["scales"][idx] * float(s)).astype(np.float32)

    disp = np.linalg.norm(moved - old, axis=1)
    ang = 2.0 * np.degrees(np.arccos(np.clip(np.abs(q_new[:, 0]), 0.0, 1.0)))
    res.update(n_moved=int(len(idx)),
               max_displacement_m=float(disp.max()),
               mean_displacement_m=float(disp.mean()),
               max_rotation_deg=float(ang.max()))
    return res


def group_footprint(gmap, kf_ids: Iterable[int], T_WP: np.ndarray):
    """Plane-frame bbox of the Gaussians owned by ``kf_ids`` (for dirty marking)."""
    ids = np.asarray(sorted(set(int(k) for k in kf_ids)), np.int32)
    if gmap.n_committed == 0:
        return None
    sel = np.isin(gmap.committed["owner"], ids)
    if not sel.any():
        return None
    P = (gmap.committed["means"][sel].astype(np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
    pad = 2.0
    return (float(P[:, 0].min()) - pad, float(P[:, 1].min()) - pad,
            float(P[:, 0].max()) + pad, float(P[:, 1].max()) + pad)
