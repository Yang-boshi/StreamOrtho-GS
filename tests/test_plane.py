"""S1 tests: ``config.cfg`` parsing, quaternion math, ``T_WP``.

Run::

    python -m pytest stream_ortho/tests/test_plane.py -q
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from stream_ortho.data.plane import (
    NpuConfig,
    camera_heights,
    parse_config_cfg,
    parse_trajectory,
    plane_normal_W,
    plane_to_world,
    quat_wxyz_to_matrix,
    quat_xyzw_to_matrix,
    world_to_plane,
)

SYNTH_CFG = """\
Plane=1.5 -2.0 3.0 0.0 0.0 0.0 1.0
GPS.Origin=114.071121 22.536340 0.000000
Camera.CameraType=PinHole
Camera.Paraments=[1920,1080,1220.000000,1220.000000,960.000000,540.000000]
TrajectoryFile=$(Svar.ParsingPath)/trajectory.txt
PrepareFrameNum=200
"""


def test_parse_synthetic():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "config.cfg"
        p.write_text(SYNTH_CFG)
        cfg = parse_config_cfg(p)
    assert (cfg.width, cfg.height) == (1920, 1080)
    assert (cfg.fx, cfg.fy, cfg.cx, cfg.cy) == (1220.0, 1220.0, 960.0, 540.0)
    np.testing.assert_allclose(cfg.plane_xyz, [1.5, -2.0, 3.0])
    np.testing.assert_allclose(cfg.plane_quat, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(cfg.gps_origin, [114.071121, 22.536340, 0.0])
    assert cfg.trajectory_file.name == "trajectory.txt"
    assert cfg.trajectory_file.parent.name  # $(Svar.ParsingPath) substituted
    assert cfg.raw["PrepareFrameNum"] == "200"
    assert cfg.K.shape == (3, 3) and cfg.K[0, 2] == 960.0
    print("[ok] config.cfg parsing (W H fx fy cx cy / Plane / GPS.Origin / $() expansion)")


def test_quaternion_identity_and_axes():
    np.testing.assert_allclose(quat_xyzw_to_matrix([0, 0, 0, 1]), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(quat_wxyz_to_matrix([1, 0, 0, 0]), np.eye(3), atol=1e-12)
    # 90 deg about x (scalar-last q = [sin(45),0,0,cos(45)])
    s = np.sin(np.pi / 4)
    R = quat_xyzw_to_matrix([s, 0, 0, s])
    np.testing.assert_allclose(R @ np.array([0, 1.0, 0]), [0, 0, 1], atol=1e-12)
    # both conventions of the same rotation agree
    q_xyzw = [0.1, -0.2, 0.3, 0.927]
    q_wxyz = [q_xyzw[3], *q_xyzw[:3]]
    np.testing.assert_allclose(
        quat_xyzw_to_matrix(q_xyzw), quat_wxyz_to_matrix(q_wxyz), atol=1e-12
    )
    print("[ok] quaternion convention (scalar-last for Plane / trajectory)")


def test_plane_roundtrip_and_normal():
    T_WP = plane_to_world([10.0, -5.0, 2.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(T_WP @ np.linalg.inv(T_WP), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(plane_normal_W(T_WP), [0, 0, 1], atol=1e-12)

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 3))
    p_plane = world_to_plane(T_WP, pts)
    back = p_plane @ T_WP[:3, :3].T + T_WP[:3, 3]
    np.testing.assert_allclose(back, pts, atol=1e-12)
    # plane origin maps to (0,0,0) in the plane frame
    np.testing.assert_allclose(world_to_plane(T_WP, T_WP[:3, 3][None])[0], 0, atol=1e-12)

    # heights are measured relative to the plane origin (z=2 here)
    h = camera_heights(T_WP, [[0, 0, 30.0], [0, 0, 2.0]])
    np.testing.assert_allclose(h, [28.0, 0.0], atol=1e-12)
    print("[ok] T_WP roundtrip / normal / heights")


def test_trajectory_parse(tmp=None):
    txt = "1.0 0 0 10 0 0 0 1\n0.5 1 2 3 0 0 0.7071068 0.7071068\n"
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "trajectory.txt"
        p.write_text(txt)
        traj = parse_trajectory(p)
    assert len(traj) == 2
    np.testing.assert_allclose(traj.timestamps, [0.5, 1.0])   # sorted
    np.testing.assert_allclose(traj.positions[0], [1, 2, 3])
    assert abs(traj.dt[1] - 0.5) < 1e-12
    T = traj.matrix(0)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    assert traj.index_of(1.0) == 1 and traj.index_of(0.5) == 0
    assert traj.index_of(0.75) is None
    np.testing.assert_allclose(traj.interpolate([0.75])[0], [0.5, 1.0, 6.5])
    print("[ok] trajectory parsing / sorting / lookup / interpolation")


def test_real_configs():
    """Every sequence under ``PROJ/data``: sane intrinsics + horizontal plane + exact T_WP."""
    from stream_ortho import config as zcfg
    from stream_ortho.data.npu_dataset import discover_sequences

    root = zcfg.dataset_root()
    if not root.exists():
        print("[skip] data root missing")
        return
    seqs = discover_sequences(root)
    assert len(seqs) >= 3, [s_.name for s_ in seqs]
    for seq in seqs:
        cfg = parse_config_cfg(seq / "config.cfg")
        assert (cfg.width, cfg.height) == (1920, 1080), (seq.name, cfg.width, cfg.height)
        assert 900 < cfg.fx < 1500, (seq.name, cfg.fx)
        assert abs(cfg.fx - cfg.fy) < 1e-6, seq.name
        assert cfg.plane_xyz is not None and cfg.plane_quat is not None, seq.name
        q = np.asarray(cfg.plane_quat)
        assert abs(np.linalg.norm(q) - 1) < 1e-3, (seq.name, np.linalg.norm(q))
        T_WP = plane_to_world(cfg.plane_xyz, cfg.plane_quat)
        assert np.abs(T_WP @ np.linalg.inv(T_WP) - np.eye(4)).max() < 1e-9, seq.name
        n = plane_normal_W(T_WP)
        assert abs(n[2]) > 0.99, (seq.name, n)   # plane must be horizontal
    print(f"[ok] {len(seqs)}/{len(seqs)} real config.cfg: intrinsics, unit quaternion, horizontal plane")


def run_all():
    test_parse_synthetic()
    test_quaternion_identity_and_axes()
    test_plane_roundtrip_and_normal()
    test_trajectory_parse()
    test_real_configs()
    print("test_plane: ALL PASSED")


if __name__ == "__main__":
    run_all()
