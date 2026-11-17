"""P5.1c Task 8: standard 3DGS PLY export format tests."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from stream_ortho.scripts.export_3dgs_ply import (_DTYPE, read_gaussian_ply,
                                                  quat_rotate_batch,
                                                  write_gaussian_ply)


def _fake_gaussians(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.random((n, 3)).astype(np.float32) * 10.0
    sh = (rng.random((n, 3)).astype(np.float32) - 0.5) / 0.28209479177387814
    opacity = rng.random((n, 1)).astype(np.float32)
    scales = (rng.random((n, 3)).astype(np.float32) * 0.01 + 0.002)
    q = rng.random((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    kf = rng.integers(0, 22, n).astype(np.int32)
    return means, sh, opacity, scales, q, kf


def test_ply_format_roundtrip(tmp_path):
    means, sh, opacity, scales, q, kf = _fake_gaussians()
    p = tmp_path / "ground_3dgs.ply"
    info = write_gaussian_ply(p, means, sh, opacity, scales, q, kf,
                              meta={"gsd_m_per_px": "0.0075", "n_keyframes": 22})
    assert info["n"] == len(means)
    assert info["bytes"] == p.stat().st_size
    assert info["bytes"] < 500e6

    r = read_gaussian_ply(p)
    assert r["n"] == len(means)
    # field names and order per the standard 3DGS schema (+ keyframe_id)
    assert r["props"] == ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                          "scale_0", "scale_1", "scale_2",
                          "rot_0", "rot_1", "rot_2", "rot_3", "keyframe_id"]
    d = r["data"]
    np.testing.assert_allclose(d["x"], means[:, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(d["f_dc_0"], sh[:, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d["opacity"], opacity[:, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(d["scale_1"], scales[:, 1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(d["rot_3"], q[:, 3], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(d["keyframe_id"], kf)
    # binary little endian + 15 properties
    assert "format binary_little_endian 1.0" in r["header"]
    assert "comment gsd_m_per_px: 0.0075" in r["header"]


def test_ply_rejects_mismatched_or_nonfinite():
    means, sh, opacity, scales, q, kf = _fake_gaussians(50)
    with pytest.raises(AssertionError):
        write_gaussian_ply("/tmp/x.ply", means[:10], sh, opacity, scales, q, kf)
    bad = means.copy(); bad[3, 1] = np.inf
    with pytest.raises(AssertionError):
        write_gaussian_ply("/tmp/x.ply", bad, sh, opacity, scales, q, kf)


def _qrot(q, v):
    """Rotate unit vectors v by unit wxyz quaternions q (Hamilton sandwich)."""
    w, x, y, z = (q[:, i] for i in range(4))
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # v' = q * (0,v) * q*  expanded
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    t = np.stack([tx, ty, tz], 1)
    return v + w[:, None] * t + np.cross(np.stack([x, y, z], 1), t)


def test_quat_rotation_matches_matrix():
    """quat_rotate_batch(q, R) == compose: R(q_R) applied AFTER R(q)."""
    rng = np.random.default_rng(1)
    q = rng.random((200, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    # a random rotation matrix
    A = rng.random((3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    qp = quat_rotate_batch(q, Q)
    assert np.allclose(np.linalg.norm(qp, axis=1), 1.0, atol=1e-9)
    # rotating the identity quaternion must give Q's own quaternion
    ident = np.tile(np.array([1.0, 0, 0, 0]), (5, 1))
    qr = quat_rotate_batch(ident, Q)
    from scipy.spatial.transform import Rotation as _R
    ref = _R.from_matrix(Q).as_quat()   # scipy: xyzw
    ref_wxyz = np.array([ref[3], ref[0], ref[1], ref[2]])  # -> wxyz
    assert np.allclose(np.abs(qr[0]), np.abs(ref_wxyz), atol=1e-9)
    # composition: R(qp) @ v == Q @ (R(q) @ v)  -- q_R carries q's frame
    v = rng.random((200, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    expected = _qrot(q, v) @ Q.T      # R(q)·v, then Q·
    np.testing.assert_allclose(_qrot(qp, v), expected, atol=1e-9)


def test_dtype_layout():
    names = list(_DTYPE.names)
    assert names[:3] == ["x", "y", "z"]
    assert names[-1] == "keyframe_id"
    assert _DTYPE["keyframe_id"] == np.int32
    # 14 float32 + 1 int32, tightly packed (no numpy padding): 60 bytes
    assert _DTYPE.itemsize == 15 * 4
    assert _DTYPE.fields["keyframe_id"][1] == 14 * 4
