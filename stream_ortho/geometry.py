"""Small geometry/Sim3 toolbox shared by the S2/S3/S4 stages (numpy + torch).

Conventions (see ``results/week1/NOTES.md``):

* lietorch ``Sim3`` data layout: ``[tx, ty, tz, qx, qy, qz, qw, s]`` and
  ``p_W = s · R(q) · p_C + t``.
* quaternions are **scalar-last** on the numpy side (same as ``trajectory.txt``
  and ``Plane``), **wxyz** inside gsplat tensors.
* the plane frame ``P`` comes from ``T_WP``; orthographic cameras look down
  ``-Z_P`` with ``+x_P`` right and ``+y_P`` **up** (``R_po = diag(1,-1,-1)``).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------

def quat_xyzw_to_R(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_xyzw(R) -> np.ndarray:
    """Rotation matrix -> scalar-last unit quaternion (robust branch-free pick)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_mul_wxyz(a, b):
    """Hamilton product for wxyz quaternions (numpy or torch)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


# ---------------------------------------------------------------------------
# Sim3
# ---------------------------------------------------------------------------

def sim3_from_vec8(vec) -> tuple[np.ndarray, np.ndarray, float]:
    """(R, t, s) from lietorch ``Sim3.data`` (7 numbers -> s = 1)."""
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 7:
        t, q, s = v[:3], v[3:7], 1.0
    elif v.size >= 8:
        t, q, s = v[:3], v[3:7], float(v[7])
    else:
        raise ValueError(f"expected 7 or 8 Sim3 numbers, got {v.size}")
    return quat_xyzw_to_R(q), t, s


def compose_sim3(vec_ab, vec_bc) -> np.ndarray:
    """Compose ``T_ac = T_bc ∘ T_ab`` and return an 8-vector (lietorch layout).

    Both inputs are ``[t(3), q(4), s]`` with ``p_child = s R p_parent + t``.
    Used to fold ``camera->SLAM world`` and ``SLAM world->GT world`` into one.
    """
    Ra, ta, sa = sim3_from_vec8(vec_ab)
    Rb, tb, sb = sim3_from_vec8(vec_bc)
    R = Rb @ Ra
    t = sb * (Rb @ ta) + tb
    s = sb * sa
    q = R_to_quat_xyzw(R)
    return np.concatenate([t, q, [s]])


def sim3_matrix(vec) -> np.ndarray:
    """4x4 with the Sim3 scale folded into the rotation block."""
    R, t, s = sim3_from_vec8(vec)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def sim3_act(vec, pts) -> np.ndarray:
    R, t, s = sim3_from_vec8(vec)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return s * (pts @ R.T) + t


def sim3_act_inv(vec, pts) -> np.ndarray:
    R, t, s = sim3_from_vec8(vec)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return ((pts - t) / s) @ R


# ---------------------------------------------------------------------------
# Gaussian world transform (torch)
# ---------------------------------------------------------------------------

def quat_rotate_batch_torch(quats_wxyz, R):
    """Rotate a batch of ``(N, 4)`` wxyz torch quaternions by ``(3, 3)`` matrix ``R``."""
    import torch

    qt = quats_wxyz.reshape(-1, 4)
    if qt.numel() == 0:
        return qt
    q_R = torch.as_tensor(R_to_quat_wxyz(R), dtype=qt.dtype, device=qt.device).expand_as(qt)
    qw = torch.stack([
        q_R[:, 0] * qt[:, 0] - q_R[:, 1] * qt[:, 1] - q_R[:, 2] * qt[:, 2] - q_R[:, 3] * qt[:, 3],
        q_R[:, 0] * qt[:, 1] + q_R[:, 1] * qt[:, 0] + q_R[:, 2] * qt[:, 3] - q_R[:, 3] * qt[:, 2],
        q_R[:, 0] * qt[:, 2] - q_R[:, 1] * qt[:, 3] + q_R[:, 2] * qt[:, 0] + q_R[:, 3] * qt[:, 1],
        q_R[:, 0] * qt[:, 3] + q_R[:, 1] * qt[:, 2] - q_R[:, 2] * qt[:, 1] + q_R[:, 3] * qt[:, 0],
    ], dim=-1)
    return qw / qw.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def transform_gaussians_to_world(mu_cam, quats_wxyz, scales, sim3_vec, device="cuda"):
    """Camera-frame Gaussians -> world frame (full Sim3, scale-aware).

    Returns ``(mu_W, quats_W, scales_W)`` as torch tensors, ``quats_W`` still
    normalised and in the same (wxyz) convention gsplat expects.
    """
    import torch

    R, t, s = sim3_from_vec8(sim3_vec)
    Rt = torch.as_tensor(R, dtype=torch.float32, device=device)
    tt = torch.as_tensor(t, dtype=torch.float32, device=device)

    mu = mu_cam.reshape(-1, 3).to(device, torch.float32)
    sc = scales.reshape(-1, 3).to(device, torch.float32)
    qt = quats_wxyz.reshape(-1, 4).to(device, torch.float32)
    qt = qt / qt.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    mu_w = float(s) * (mu @ Rt.T) + tt
    sc_w = sc * float(s)
    qw = quat_rotate_batch_torch(qt, R)
    return mu_w, qw, sc_w


# ---------------------------------------------------------------------------
# Orthographic camera in the plane frame
# ---------------------------------------------------------------------------

# plane (+x right, +y up, +z up) -> orthographic camera (+x right, +y DOWN, +z down)
R_PO = np.array([[1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0]])


def ortho_viewmat(T_WP: np.ndarray, cx: float, cy: float, z_top: float) -> np.ndarray:
    """World->ortho-camera matrix for a tile centred at plane ``(cx, cy)``."""
    T_PO = np.eye(4)
    T_PO[:3, :3] = R_PO
    T_PO[:3, 3] = [cx, cy, z_top]
    T_WO = T_WP @ T_PO
    return np.linalg.inv(T_WO)


def ortho_K(gsd: float, W: int, H: int) -> np.ndarray:
    """Orthographic intrinsics: ``fx = fy = 1/gsd`` (pixels per metre)."""
    return np.array([[1.0 / gsd, 0.0, W / 2.0],
                     [0.0, 1.0 / gsd, H / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def plane_point_to_pixel(T_WP, cx, cy, gsd, W, H, pts_P):
    """Plane-frame points -> tile pixel coordinates (for reference warps)."""
    from stream_ortho.data.plane import world_to_plane  # noqa: F401  (doc)

    pts = np.asarray(pts_P, dtype=np.float64).reshape(-1, 3)
    u = (pts[:, 0] - (cx - 0.5 * W * gsd)) / gsd
    v = (cy + 0.5 * H * gsd - pts[:, 1]) / gsd
    return np.stack([u, v], axis=-1)


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> **wxyz** unit quaternion (the gsplat/3DGS convention)."""
    q = R_to_quat_xyzw(R)
    return q[[3, 0, 1, 2]]


def R_to_quat_wxyz_batch(R: np.ndarray) -> np.ndarray:
    """Vectorised ``(N,3,3) -> (N,4) wxyz``."""
    q = R_to_quat_xyzw_batch(R)
    return q[:, [3, 0, 1, 2]]


def R_to_quat_xyzw_batch(R: np.ndarray) -> np.ndarray:
    """Vectorised rotation matrices (N,3,3) -> quaternions xyzw (N,4).

    Same convention as :func:`R_to_quat_xyzw`; used by the shape prior where a
    per-point Python loop over ~1e4 edge Gaussians cost ~300 ms per keyframe.
    """
    R = np.asarray(R, np.float64)
    n = R.shape[0]
    q = np.zeros((n, 4), np.float64)
    if n == 0:
        return q
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    # four candidate branches, choose the numerically best one per element
    i = np.argmax(np.stack([tr, m00, m11, m22], axis=1), axis=1)
    for k in range(4):
        sel = i == k
        if not sel.any():
            continue
        r = R[sel]
        if k == 0:
            s2 = np.sqrt(np.maximum(tr[sel] + 1.0, 1e-12)) * 2.0
            q[sel, 3] = 0.25 * s2
            q[sel, 0] = (r[:, 2, 1] - r[:, 1, 2]) / s2
            q[sel, 1] = (r[:, 0, 2] - r[:, 2, 0]) / s2
            q[sel, 2] = (r[:, 1, 0] - r[:, 0, 1]) / s2
        elif k == 1:
            s2 = np.sqrt(np.maximum(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-12)) * 2.0
            q[sel, 3] = (r[:, 2, 1] - r[:, 1, 2]) / s2
            q[sel, 0] = 0.25 * s2
            q[sel, 1] = (r[:, 0, 1] + r[:, 1, 0]) / s2
            q[sel, 2] = (r[:, 0, 2] + r[:, 2, 0]) / s2
        elif k == 2:
            s2 = np.sqrt(np.maximum(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], 1e-12)) * 2.0
            q[sel, 3] = (r[:, 0, 2] - r[:, 2, 0]) / s2
            q[sel, 0] = (r[:, 0, 1] + r[:, 1, 0]) / s2
            q[sel, 1] = 0.25 * s2
            q[sel, 2] = (r[:, 1, 2] + r[:, 2, 1]) / s2
        else:
            s2 = np.sqrt(np.maximum(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], 1e-12)) * 2.0
            q[sel, 3] = (r[:, 1, 0] - r[:, 0, 1]) / s2
            q[sel, 0] = (r[:, 0, 2] + r[:, 2, 0]) / s2
            q[sel, 1] = (r[:, 1, 2] + r[:, 2, 1]) / s2
            q[sel, 2] = 0.25 * s2
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1e-12)
