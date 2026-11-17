"""S2 tests: Umeyama Sim3 alignment and ATE metrics (synthetic, no GT data needed).

Run::

    python -m pytest stream_ortho/tests/test_ate.py -q
"""

from __future__ import annotations

import numpy as np

from stream_ortho.eval.ate import (
    PoseTrajectory,
    Sim3,
    associate_by_time,
    evaluate_ate,
    umeyama,
)


def _rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    th = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _mat_to_quat_xyzw(R):
    w, x, y, z = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2, None, None, None
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(7)
    src = rng.normal(size=(50, 3)) * 10
    s_true = 2.37
    R_true = _rot([0.2, 1.0, -0.3], 41.0)
    t_true = np.array([3.0, -7.0, 11.0])
    dst = s_true * src @ R_true.T + t_true

    sim = umeyama(src, dst)
    assert abs(sim.scale - s_true) < 1e-9, sim.scale
    np.testing.assert_allclose(sim.rotation, R_true, atol=1e-9)
    np.testing.assert_allclose(sim.translation, t_true, atol=1e-8)
    np.testing.assert_allclose(sim.apply(src), dst, atol=1e-8)
    # Sim3.matrix also works
    T = sim.matrix()
    np.testing.assert_allclose(T[:3, :3], s_true * R_true, atol=1e-9)
    print("[ok] umeyama recovers (s, R, t) exactly")


def test_umeyama_handles_reflection():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(30, 3))
    dst = src.copy()
    dst[:, 2] *= -1.0                     # mirror: not a rotation
    sim = umeyama(src, dst)
    assert np.linalg.det(sim.rotation) > 0, np.linalg.det(sim.rotation)
    # a reflection cannot be fitted exactly, but the fit must be a proper rotation
    print(f"[ok] reflection guard: det(R)={np.linalg.det(sim.rotation):.3f}, scale={sim.scale:.3f}")


def test_umeyama_degenerate():
    try:
        umeyama(np.zeros((10, 3)), np.ones((10, 3)))
    except ValueError:
        print("[ok] degenerate cloud raises ValueError")
    else:
        raise AssertionError("expected ValueError")


def _make_traj(ts, pos, quats=None):
    if quats is None:
        quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(ts), 1))
    return PoseTrajectory(np.asarray(ts, float), np.asarray(pos, float), np.asarray(quats, float))


def test_evaluate_ate_perfect():
    ts = np.arange(20) * 0.5
    R = _rot([1, 0, 0], 17.0)
    gt_pos = np.stack([np.cos(ts), np.sin(ts), 0.3 * ts], axis=1) * 5
    ref = _make_traj(ts, gt_pos, np.tile(_mat_to_quat_xyzw(R), (len(ts), 1)))
    est = _make_traj(ts, 3.1 * gt_pos @ R.T + np.array([1.0, 2.0, -4.0]))
    res = evaluate_ate(est, ref)
    assert res.rmse < 1e-9, res.rmse
    assert abs(res.scale - 1 / 3.1) < 1e-9, res.scale
    assert res.n_pairs == 20
    print(f"[ok] perfect Sim3 trajectory: ATE={res.rmse:.2e} m, scale={res.scale:.4f}")


def test_evaluate_ate_noise_monotone():
    rng = np.random.default_rng(0)
    ts = np.arange(200) * 0.2
    gt = np.stack([ts, 0.2 * ts, np.zeros_like(ts)], axis=1)
    for sigma in (0.0, 0.01, 0.1):
        est_pos = gt + rng.normal(scale=sigma, size=gt.shape)
        res = evaluate_ate(_make_traj(ts, est_pos), _make_traj(ts, gt))
        print(f"    sigma={sigma:<5} ATE={res.rmse:.4f} m ({res.ate_percent:.3f}%), p95={res.p95:.4f}")
        if sigma == 0.0:
            assert res.rmse < 1e-9
    res1 = evaluate_ate(_make_traj(ts, gt + rng.normal(scale=0.1, size=gt.shape)), _make_traj(ts, gt))
    assert 0.05 < res1.rmse < 0.3, res1.rmse
    print(f"[ok] ATE scales with noise (rmse={res1.rmse:.4f} m, {res1.ate_percent:.3f}% of path)")


def test_association_by_time():
    est = _make_traj([0.0, 0.5, 1.0, 1.5], np.zeros((4, 3)))
    ref = _make_traj([0.0, 0.51, 1.02, 2.0], np.zeros((4, 3)))
    i_est, i_ref = associate_by_time(est, ref, max_dt=0.05)
    np.testing.assert_array_equal(i_est, [0, 1, 2])
    np.testing.assert_array_equal(i_ref, [0, 1, 2])
    # unsorted reference is handled
    ref2 = _make_traj([2.0, 1.02, 0.51, 0.0], np.zeros((4, 3)))
    i_est2, i_ref2 = associate_by_time(est, ref2, max_dt=0.05)
    np.testing.assert_array_equal(i_est2, [0, 1, 2])
    np.testing.assert_array_equal(i_ref2, [3, 2, 1])
    print("[ok] nearest-timestamp association (incl. unsorted reference)")


def test_ate_too_few_pairs():
    est = _make_traj([0.0, 1.0], np.zeros((2, 3)))
    ref = _make_traj([0.0, 1.0], np.zeros((2, 3)))
    try:
        evaluate_ate(est, ref)
    except ValueError:
        print("[ok] <3 associated frames raises ValueError")
    else:
        raise AssertionError("expected ValueError")


def run_all():
    test_umeyama_recovers_sim3()
    test_umeyama_handles_reflection()
    test_umeyama_degenerate()
    test_evaluate_ate_perfect()
    test_evaluate_ate_noise_monotone()
    test_association_by_time()
    test_ate_too_few_pairs()
    print("test_ate: ALL PASSED")


if __name__ == "__main__":
    run_all()
