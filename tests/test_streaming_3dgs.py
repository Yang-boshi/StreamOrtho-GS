"""Unit tests for streaming 3DGS alignment, voxel consensus, pose sync, and PLY encoding."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from stream_ortho.app.alignment import ProgressivePlaneAligner
from stream_ortho.app.config import load_config
from stream_ortho.online.gs_map import StreamingGaussianMap
from stream_ortho.scripts.export_3dgs_ply import read_gaussian_ply


def _make_tilted_strip(n=2000, tilt_x_deg=0.0, tilt_y_deg=0.0, dz=0.0, cx=0.0, cy=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-5.0, 5.0, size=(n, 2))
    z = rng.normal(0.0, 0.02, size=(n, 1))
    # Add a 2.0m tall building roof on 15% of points
    bldg = (xy[:, 0] > 1.0) & (xy[:, 0] < 3.0) & (xy[:, 1] > -1.0) & (xy[:, 1] < 1.0)
    z[bldg, 0] += 2.0
    pts = np.hstack([xy, z])
    ax, ay = np.radians(tilt_x_deg), np.radians(tilt_y_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R = Ry @ Rx
    pts = pts @ R.T + np.array([cx, cy, dz])
    return pts, ~bldg


def test_progressive_plane_aligner_levels_drifted_strips():
    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    p0, g0 = _make_tilted_strip(tilt_x_deg=4.0, dz=1.5, cx=0.0, cy=0.0, seed=0)
    assert aligner.add(p0) is True
    pp0, T0, info0 = aligner.rectify_keyframe(0, p0)
    assert abs(np.median(pp0[g0, 2])) < 0.08

    # Simulate a second overlapping strip that drifted by 18 deg pitch and +3.0 units in Z
    p1, g1 = _make_tilted_strip(tilt_x_deg=22.0, tilt_y_deg=-8.0, dz=4.5, cx=3.0, cy=0.0, seed=1)
    # Use a tiny elevation surface proxy from strip 0
    gs_map = StreamingGaussianMap(voxel_size=0.1, tile_world_size=10.0)
    n0 = pp0.shape[0]
    ident_q = np.tile([1.0, 0.0, 0.0, 0.0], (n0, 1)).astype(np.float32)
    gs_map.ingest_keyframe(
        0, pp0, ident_q, np.full((n0, 3), 0.05), np.full(n0, 0.9),
        np.zeros((n0, 3)), np.ones(n0), np.ones(n0),
        np.array([0, 0, 0, 0, 0, 0, 1, 1], float), T0,
    )
    pp1, T1, info1 = aligner.rectify_keyframe(1, p1, surface_ref=gs_map)
    assert info1["rectified"] is True
    # Ground plane of drifted strip 1 must now lie flat near z = 0
    assert np.std(pp1[g1, 2]) < 0.08
    assert abs(np.median(pp1[g1, 2]) - np.median(pp0[g0, 2])) < 0.15


def test_streaming_gaussian_map_nadir_consensus_and_nfp():
    gs_map = StreamingGaussianMap(voxel_size=0.2, tile_world_size=10.0, nadir_replace_margin=0.08)
    n = 100
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(0.0, 10.0, n)
    pts[:, 2] = 5.0  # roof at z=5
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    sc = np.full((n, 3), 0.1, dtype=np.float32)
    op = np.full(n, 0.9, dtype=np.float32)
    sh_oblique = np.full((n, 3), -1.0, dtype=np.float32)
    sh_nadir = np.full((n, 3), +1.0, dtype=np.float32)
    rays_oblique = np.tile([0.6, 0.0, -0.8], (n, 1)).astype(np.float32)
    rays_nadir = np.tile([0.0, 0.0, -1.0], (n, 1)).astype(np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)

    r0 = gs_map.ingest_keyframe(0, pts, q, sc, op, sh_oblique, np.ones(n), np.ones(n),
                                T_id, np.eye(4), cam_rays_p=rays_oblique)
    assert r0["added"] > 0
    n_init = gs_map.n_points

    # Same coordinates observed by a straight-down nadir view -> replaces in-place, no bloat
    r1 = gs_map.ingest_keyframe(1, pts, q, sc, op, sh_nadir, np.full(n, 2.0), np.ones(n),
                                T_id, np.eye(4), cam_rays_p=rays_nadir)
    assert r1["replaced"] == n_init
    assert r1["added"] == 0
    assert gs_map.n_points == n_init
    assert np.allclose(gs_map.sh_dc, 1.0)
    assert np.all(gs_map.owner_kf == 1)

    # Oblique point underneath the roof at z=1.0 must be vetoed by NFP
    pts_below = pts.copy()
    pts_below[:, 2] = 1.0
    r2 = gs_map.ingest_keyframe(2, pts_below, q, sc, op, sh_oblique, np.full(n, 0.5), np.ones(n),
                                T_id, np.eye(4), cam_rays_p=rays_oblique)
    assert r2["added"] == 0
    assert gs_map.n_points == n_init


def test_streaming_gaussian_map_pose_update_and_ply_encoding(tmp_path):
    gs_map = StreamingGaussianMap(voxel_size=0.2, tile_world_size=5.0)
    pts = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    q = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sc = np.array([[0.05, 0.05, 0.01], [0.08, 0.04, 0.01]], dtype=np.float32)
    op = np.array([0.9, 0.2], dtype=np.float32)
    sh = np.zeros((2, 3), dtype=np.float32)
    T_old = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    T_new = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64)

    gs_map.ingest_keyframe(0, pts, q, sc, op, sh, np.ones(2), np.ones(2), T_old, np.eye(4))
    dirty = gs_map.update_poses({0: T_new})
    assert len(dirty) == 1
    # Scaled by 2 and translated by +10 in X
    np.testing.assert_allclose(gs_map.means[:, 0], [12.0, 14.0], atol=1e-4)
    np.testing.assert_allclose(gs_map.scales[0], [0.10, 0.10, 0.02], atol=1e-5)

    # Spatial query hits new location and misses old location
    assert gs_map.query_bbox(11.5, 1.5, 14.5, 2.5).size == 2
    assert gs_map.query_bbox(0.0, 0.0, 3.0, 2.0).size == 0

    # PLY export encodes log(scales) and logit(opacity) for SuperSplat compatibility
    ply_p = tmp_path / "ground_3dgs.ply"
    gs_map.export_ply(ply_p)
    r = read_gaussian_ply(ply_p)
    d = r["data"]
    np.testing.assert_allclose(np.exp(d["scale_0"]), [0.10, 0.16], rtol=1e-4)
    sig_op = 1.0 / (1.0 + np.exp(-d["opacity"]))
    np.testing.assert_allclose(sig_op, [0.9, 0.2], atol=1e-4)


def test_base_config_loads_and_validates():
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    cfg = load_config(cfg_path)
    assert cfg.gaussian.streaming_3dgs is True
    assert cfg.frontend.use_calib == "auto"
    assert cfg.map.bootstrap_kfs == 3
    assert cfg.map.progressive_align is True
    assert cfg.ortho.max_tiles == 24
    assert cfg.ortho.dump_format == "jpg"
    assert cfg.ortho.dump_scale == pytest.approx(0.5)
    assert cfg.ortho.dump_queue == 8


def test_deduplicate_columns_after_ba_pose_overlap():
    gs_map = StreamingGaussianMap(voxel_size=0.2, tile_world_size=5.0)
    pts0 = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    pts1 = np.array([[10.0, 1.0, 0.2], [11.0, 1.0, 0.2]], dtype=np.float32)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32)
    sc = np.full((2, 3), 0.05, dtype=np.float32)
    op = np.full(2, 0.9, dtype=np.float32)
    sh = np.zeros((2, 3), dtype=np.float32)
    T0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    gs_map.ingest_keyframe(0, pts0, q, sc, op, sh, np.ones(2), np.ones(2), T0, np.eye(4))
    gs_map.ingest_keyframe(1, pts1, q, sc, op, sh, np.full(2, 2.0), np.ones(2), T0, np.eye(4))
    assert gs_map.n_points == 4

    # BA shifts kf=1 by -9.0 in X so its points land directly on kf=0's columns
    T1_shifted = np.array([-9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    gs_map.update_poses({1: T1_shifted})
    assert gs_map.n_points == 2
    assert np.all(gs_map.owner_kf == 1)


def test_ground_only_alignment_ignores_tall_roof_and_levels_strip():
    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    p0, _ = _make_tilted_strip(tilt_x_deg=0.0, dz=0.0, cx=0.0, cy=0.0, seed=3)
    assert aligner.add(p0) is True
    pp0, T0, _ = aligner.rectify_keyframe(0, p0)

    # Inject a tall +2.0m building roof in gs_map (overlapping kf=1) to verify
    # rectify_keyframe never tilts onto roofs in the surface map.
    gs_map = StreamingGaussianMap(voxel_size=0.1, tile_world_size=20.0)
    n0 = pp0.shape[0]
    gs_map.ingest_keyframe(
        0, pp0, np.tile([1, 0, 0, 0], (n0, 1)).astype(np.float32),
        np.full((n0, 3), 0.05, np.float32), np.full(n0, 0.9, np.float32),
        np.zeros((n0, 3), np.float32), np.ones(n0, np.float32), np.ones(n0, np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1, 1], float), T0,
    )

    p1, g1 = _make_tilted_strip(tilt_x_deg=10.0, dz=0.4, cx=1.0, cy=0.0, seed=4)
    pp1, _, info1 = aligner.rectify_keyframe(1, p1, surface_ref=gs_map)
    assert info1["rectified"] is True
    assert abs(np.median(pp1[g1, 2])) < 0.08
    assert np.std(pp1[g1, 2]) < 0.08
    # Roof in kf=1 must remain cleanly ~+2.0m above ground without layering or tilting
    roof1 = ~g1
    assert np.median(pp1[roof1, 2]) > 1.8


def test_calib_txt_auto_detection_and_none_override(tmp_path):
    from PIL import Image
    from stream_ortho.data.rgb_folder import RGBFolderDataset

    Image.new("RGB", (32, 24), color=(120, 130, 140)).save(tmp_path / "0001.png")
    (tmp_path / "calib.txt").write_text(
        "1679.71,1679.71,775.465,521.95,-0.0498,0.0684\n", encoding="utf-8"
    )
    ds_auto = RGBFolderDataset(tmp_path, intrinsics="auto", load_images=False)
    assert ds_auto.has_calib() is True
    assert ds_auto.intrinsics_source == "calib.txt"
    assert ds_auto.K_orig[0, 0] == pytest.approx(1679.71)

    ds_none = RGBFolderDataset(tmp_path, intrinsics="none", load_images=False)
    assert ds_none.has_calib() is False
    assert ds_none.intrinsics_source == "none"


def test_show_cli_flag_and_headless_viewer(monkeypatch):
    from stream_ortho.app.config import Config, override
    from stream_ortho.app.run import build_parser
    from stream_ortho.app.viewer import LiveOrthoViewer

    args = build_parser().parse_args(["--show"])
    assert args.show is True
    cfg = Config()
    assert cfg.ortho.show is False
    override(cfg, ortho__show=True)
    assert cfg.ortho.show is True

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    viewer = LiveOrthoViewer()
    assert viewer.active is False
    viewer.submit([], "empty")
    viewer.close()


def test_stage0_gs_quality_metrics_and_residual_alignment():
    from stream_ortho.eval.gs_metrics import compute_gs_quality_metrics

    rng = np.random.default_rng(11)
    # 500 ground pts around z=0, plus 200 layered overhang ghost pts at z=0.5 above the same XY
    gnd_xy = rng.uniform(0.0, 0.4, size=(500, 2))
    gnd_z = rng.normal(0.0, 0.005, size=(500, 1))
    ovh_xy = gnd_xy[:200]
    ovh_z = rng.normal(0.50, 0.005, size=(200, 1))
    pts = np.vstack([np.hstack([gnd_xy, gnd_z]), np.hstack([ovh_xy, ovh_z])])
    owners = np.concatenate([np.zeros(500, dtype=np.int32), np.ones(200, dtype=np.int32)])

    m = compute_gs_quality_metrics(pts, owner_kf=owners, cell_size=0.05, band_gap=0.04)
    assert m["n_points"] == 700
    assert m["fine"]["multi_band_frac"] > 0.3
    assert m["fine"]["overhang_elev_pt_frac"] > 0.5
    assert m["inter_kf_ground"]["n_kfs"] == 2

    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    p0, _ = _make_tilted_strip(tilt_x_deg=0.0, dz=0.0, seed=12)
    assert aligner.add(p0) is True
    pp0, _, _ = aligner.rectify_keyframe(0, p0)
    rr_clean = aligner.estimate_residual_rect(pp0)
    assert abs(rr_clean["res_dz"]) < 0.05

    pp_shifted = pp0.copy()
    pp_shifted[:, 2] += 0.25
    rr_shift = aligner.estimate_residual_rect(pp_shifted)
    assert rr_shift["res_dz"] == pytest.approx(-0.25, abs=0.05)


def test_stage1_realign_after_ba_replaces_stale_rect_without_accumulation():
    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    p0, _ = _make_tilted_strip(tilt_x_deg=0.0, dz=0.0, cx=0.0, cy=0.0, seed=21)
    assert aligner.add(p0) is True
    pp0, T0, _ = aligner.rectify_keyframe(0, p0)

    gs_map = StreamingGaussianMap(voxel_size=0.1, tile_world_size=20.0)
    n0 = pp0.shape[0]
    ident_q = np.tile([1.0, 0.0, 0.0, 0.0], (n0, 1)).astype(np.float32)
    T_id = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    gs_map.ingest_keyframe(0, pp0, ident_q, np.full((n0, 3), 0.05), np.full(n0, 0.9),
                           np.zeros((n0, 3)), np.ones(n0), np.ones(n0), T_id, T0)

    # kf=1 arrives with pre-BA drift (-0.40 in Z, 6 deg tilt); rectify levels it near z=0
    p1, g1 = _make_tilted_strip(tilt_x_deg=6.0, dz=-0.40, cx=2.0, cy=0.0, seed=22)
    pp1, T1_old, _ = aligner.rectify_keyframe(1, p1)
    assert abs(np.median(pp1[g1, 2])) < 0.06
    n1 = pp1.shape[0]
    ident_q1 = np.tile([1.0, 0.0, 0.0, 0.0], (n1, 1)).astype(np.float32)
    gs_map.ingest_keyframe(1, pp1, ident_q1, np.full((n1, 3), 0.05), np.full(n1, 0.9),
                           np.zeros((n1, 3)), np.ones(n1), np.ones(n1), T_id, T1_old)

    # Later BA shifts kf=1 up by +0.55 in World Z (fixing pre-BA drift); recompute T_rect must NOT stack
    T1_ba = np.array([0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    T_pw_new = aligner.realign_after_ba(1, T_id, T1_ba, enabled=True)
    assert T_pw_new is not None
    gs_map.update_poses({1: T1_ba}, new_T_PW={1: T_pw_new})

    sel1 = gs_map.owner_kf == 1
    # Check low-band ground of kf=1 remains flat near z=0 (no +0.55 or +0.95 double-shift)
    z1 = gs_map.means[sel1, 2]
    lo1 = z1[z1 <= np.percentile(z1, 60)]
    assert abs(np.median(lo1)) < 0.06
    assert np.std(lo1) < 0.06

    # Second incremental BA update must also remain invariant (no accumulation)
    T1_ba2 = np.array([0.0, 0.0, 0.70, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    T_pw_new2 = aligner.realign_after_ba(1, T1_ba, T1_ba2, enabled=True)
    gs_map.update_poses({1: T1_ba2}, new_T_PW={1: T_pw_new2})
    z1_2 = gs_map.means[gs_map.owner_kf == 1, 2]
    lo1_2 = z1_2[z1_2 <= np.percentile(z1_2, 60)]
    assert abs(np.median(lo1_2)) < 0.06


def test_stage2_decoupled_fusion_weighted_median_roof_and_best_color():
    gs_map = StreamingGaussianMap(voxel_size=0.05, tile_world_size=10.0, z_band=0.20, nadir_replace_margin=0.15)
    # 10 roof columns observed by 4 frames with differing predicted heights & alternating winners
    xs = np.arange(10, dtype=np.float32) * 0.05 + 0.01
    ys = np.zeros(10, dtype=np.float32) + 0.01
    q = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)).astype(np.float32)
    sc = np.full((10, 3), 0.02, dtype=np.float32)
    op = np.full(10, 0.9, dtype=np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    rays = np.tile([0.0, 0.0, -1.0], (10, 1)).astype(np.float32)

    heights = [0.48, 0.55, 0.57, 0.59]
    # Frame 1 wins even columns, Frame 3 wins odd columns for color (strongest nadir_score)
    for k, h in enumerate(heights):
        pts = np.stack([xs, ys, np.full(10, h, dtype=np.float32)], axis=1)
        ns = np.ones(10, dtype=np.float32)
        if k == 1:
            ns[0::2] = 3.0
        elif k == 3:
            ns[1::2] = 3.0
        sh = np.full((10, 3), float(k), dtype=np.float32)
        gs_map.ingest_keyframe(k, pts, q, sc, op, sh, ns, np.ones(10, np.float32), T_id, np.eye(4), cam_rays_p=rays)

    # Color comes from the best nadir frame per column (k=1 on even, k=3 on odd)
    assert np.allclose(gs_map.sh_dc[0::2, 0], 1.0)
    assert np.allclose(gs_map.sh_dc[1::2, 0], 3.0)
    # Geometry height is smoothed via weighted median consensus across all 4 frames (~0.55-0.57)
    assert np.max(np.abs(gs_map.means[0::2, 2] - gs_map.means[1::2, 2])) < 0.025
    assert np.all((gs_map.means[:, 2] >= 0.54) & (gs_map.means[:, 2] <= 0.58))


def test_stage2_zbuffer_carving_two_votes_removes_ghost_and_keeps_roof():
    # Camera at Plane (0, 0, 2.0) looking straight down (-Z_P == +Z_C): R_WC = diag(1, -1, -1), t_WC = (0, 0, 2)
    T_WC = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    T_PW = np.eye(4, dtype=np.float64)
    K_d = (100.0, 100.0, 32.0, 32.0)
    H_d, W_d = 64, 64

    # Depth map: ground at depth 2.0 everywhere, except a roof at pixel u in [34, 48], v in [24, 40] at depth 1.5
    depth_map = np.full((H_d, W_d), 2.0, dtype=np.float32)
    depth_map[24:41, 34:49] = 1.5

    gs_map = StreamingGaussianMap(voxel_size=0.05, tile_world_size=10.0)
    # Insert 2 points from kf=0:
    # pt 0: ghost floater at (x=-0.30, y=0.0, z=0.40) -> u = 32 + 100*(-0.30)/1.6 = 13.2 (open ground!)
    # pt 1: real roof edge at (x=+0.05, y=0.0, z=0.50) -> u = 32 + 100*(0.05)/1.5 = 35.3 (on roof 1.5!)
    pts = np.array([[-0.30, 0.0, 0.40], [0.05, 0.0, 0.50]], dtype=np.float32)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32)
    sc = np.full((2, 3), 0.02, dtype=np.float32)
    op = np.full(2, 0.9, dtype=np.float32)
    sh = np.zeros((2, 3), dtype=np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    gs_map.ingest_keyframe(0, pts, q, sc, op, sh, np.ones(2), np.ones(2), T_id, T_PW)

    bbox = (-0.6, -0.6, 0.6, 0.6)
    # Vote 1 from kf=1: ghost gets 1st vote (NOT carved yet), roof gets 0 votes
    r1 = gs_map.carve_from_depth(1, depth_map, K_d, T_WC, T_PW, bbox, carve_tol=0.08, min_votes=2, device="cpu")
    assert r1["voted"] == 1
    assert r1["carved"] == 0
    assert gs_map.n_points == 2

    # Vote 2 from kf=2: ghost hits 2 votes and IS carved; real roof survives cleanly
    r2 = gs_map.carve_from_depth(2, depth_map, K_d, T_WC, T_PW, bbox, carve_tol=0.08, min_votes=2, device="cpu")
    assert r2["voted"] == 1
    assert r2["carved"] == 1
    assert gs_map.n_points == 1
    np.testing.assert_allclose(gs_map.means[0], [0.05, 0.0, 0.50], atol=1e-5)
    # O(N) _prune_keep_mask must preserve column lookup & spatial bucket lookup cleanly
    assert gs_map.query_bbox(-0.5, -0.5, 0.5, 0.5).tolist() == [0]
    z_q, hit_q = gs_map.query_elevation(np.array([0.05, -0.30]), np.array([0.0, 0.0]))
    assert bool(hit_q[0]) is True and bool(hit_q[1]) is False
    assert z_q[0] == pytest.approx(0.50, abs=1e-5)


def test_lossless_2d_column_hash_zero_collisions_on_million_grid():
    from stream_ortho.online.gs_map import _hash_2d

    ix, iy = np.meshgrid(np.arange(-500, 500), np.arange(-500, 500))
    h = _hash_2d(ix.ravel(), iy.ravel())
    assert np.unique(h).size == 1_000_000


def test_opencv_camera_bootstrap_orients_z_up_toward_camera():
    """Camera at origin looking +Z at ground (z=+1.8) and roof (z=+1.5) must flip +Z_P up."""
    rng = np.random.default_rng(7)
    n = 2000
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(1.8, 0.01, size=(n, 1))
    roof = (xy[:, 0] > 0.2) & (xy[:, 0] < 0.6) & (xy[:, 1] > -0.2) & (xy[:, 1] < 0.2)
    z[roof, 0] = 1.5  # closer to camera -> higher building roof!
    Pw = np.hstack([xy, z])
    cam_pos = np.array([0.0, 0.0, 0.0])

    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    assert aligner.add(Pw, cam_pos=cam_pos) is True
    Pp, T_PW, _ = aligner.rectify_keyframe(0, Pw)

    # Ground near 0, roof strictly ABOVE ground
    assert abs(np.median(Pp[~roof, 2])) < 0.05
    assert np.median(Pp[roof, 2]) > 0.2

    # Camera rays (pointing from cam=0 -> Pw) in Plane frame must have -Z_P > 0.7
    rays_w = Pw / np.linalg.norm(Pw, axis=1, keepdims=True)
    rays_p = rays_w @ T_PW[:3, :3].T
    cos_nadir = -rays_p[:, 2]
    assert np.all(cos_nadir > 0.7)


def test_ba_pose_updates_preserve_dense_roof_lattice_and_block_floaters():
    """Dense roof columns must survive repeated BA shifts (>98%) without Moiré floater holes."""
    cell = 0.01
    gs_map = StreamingGaussianMap(voxel_size=cell, tile_world_size=1.0, z_band=0.08, nadir_replace_margin=0.15)
    # Generate a dense sub-cell cloud (4 samples per cell) over a 20x20 roof patch at z=0.15
    gx, gy = np.meshgrid(np.linspace(0.001, 0.199, 40), np.linspace(0.001, 0.199, 40))
    n = gx.size
    pts0 = np.column_stack([gx.ravel(), gy.ravel(), np.full(n, 0.15, dtype=np.float32)])
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    sc = np.full((n, 3), 0.009, dtype=np.float32)
    op = np.full(n, 0.95, dtype=np.float32)
    sh_roof = np.full((n, 3), 0.8, dtype=np.float32)
    T0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    rays_nadir = np.tile([0.0, 0.0, -1.0], (n, 1)).astype(np.float32)

    gs_map.ingest_keyframe(0, pts0, q, sc, op, sh_roof, np.full(n, 100.0, np.float32),
                           np.ones(n, np.float32), T0, np.eye(4), cam_rays_p=rays_nadir)
    assert gs_map.n_points == 400

    # Apply 6 fractional BA translations; snapped lattice + 4-neighbor rescue keeps 100% of roof cells
    for step in range(1, 7):
        T_step = np.array([0.0037 * step, -0.0029 * step, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        gs_map.update_poses({0: T_step})
    assert gs_map.n_points >= 395

    # Now an oblique frame tries to insert high floaters (z=0.28) with lower score (75 < 100): must be rejected!
    pts_flt = gs_map.means.copy()
    pts_flt[:, 2] = 0.28
    m_f = pts_flt.shape[0]
    gs_map.ingest_keyframe(
        1, pts_flt, q[:m_f], sc[:m_f], op[:m_f], np.full((m_f, 3), -1.0, np.float32),
        np.full(m_f, 75.0, np.float32), np.ones(m_f, np.float32), T_step, np.eye(4),
        cam_rays_p=rays_nadir[:m_f],
    )
    assert np.allclose(gs_map.means[:, 2], 0.15, atol=1e-4)
    assert np.all(gs_map.owner_kf == 0)


def test_interior_courtyard_clears_attenuated_wall_veil():
    """High-scoring interior ground view must clear an earlier 0.01x-attenuated mid-air wall veil."""
    gs_map = StreamingGaussianMap(voxel_size=0.01, tile_world_size=1.0, z_band=0.08)
    pts_veil = np.array([[0.055, 0.055, 0.14]], dtype=np.float32)
    pts_gnd = np.array([[0.055, 0.055, 0.02]], dtype=np.float32)
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sc = np.array([[0.009, 0.009, 0.002]], dtype=np.float32)
    op = np.array([0.95], dtype=np.float32)
    T0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    # Moderate viewing angle (cos_nadir = 0.78 < 0.85)
    ray_mod = np.array([[0.625, 0.0, -0.78]], dtype=np.float32)

    # kf=0 inserts an edge-attenuated red veil point at z=0.14 (score = 1.0)
    gs_map.ingest_keyframe(0, pts_veil, q, sc, op, np.array([[1.0, -1.0, -1.0]], np.float32),
                           np.array([1.0], np.float32), np.array([1.0], np.float32),
                           T0, np.eye(4), cam_rays_p=ray_mod)
    assert gs_map.means[0, 2] == pytest.approx(0.14)

    # kf=1 observes clean grey courtyard at z=0.02 with interior score = 100.0 (> 5x veil score)
    gs_map.ingest_keyframe(1, pts_gnd, q, sc, op, np.array([[0.0, 0.0, 0.0]], np.float32),
                           np.array([100.0], np.float32), np.array([1.0], np.float32),
                           T0, np.eye(4), cam_rays_p=ray_mod)
    assert gs_map.means[0, 2] == pytest.approx(0.02)
    assert gs_map.owner_kf[0] == 1


def test_carve_from_depth_protects_elevated_roof_from_lower_roof_disagreement():
    """Cross-frame roof height disagreement (0.33m vs 0.46m) must not carve a committed roof,
    while a mid-air floater (0.40m) over open ground (0.0m) still gets carved."""
    T_WC = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    T_PW = np.eye(4, dtype=np.float64)
    K_d = (100.0, 100.0, 32.0, 32.0)
    H_d, W_d = 64, 64

    # Ground depth = 2.0 (z=0.0); right patch predicts lower roof at depth 1.67 (z=0.33m)
    depth_map = np.full((H_d, W_d), 2.0, dtype=np.float32)
    depth_map[24:41, 34:49] = 1.67

    gs_map = StreamingGaussianMap(voxel_size=0.05, tile_world_size=10.0)
    # pt 0: ghost floater at (x=-0.30, y=0.0, z=0.40) over open ground (z_bg=0.0)
    # pt 1: committed nadir roof at (x=+0.06, y=0.0, z=0.46) projecting onto lower roof (z_bg=0.33)
    pts = np.array([[-0.30, 0.0, 0.40], [0.06, 0.0, 0.46]], dtype=np.float32)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32)
    sc = np.full((2, 3), 0.02, dtype=np.float32)
    op = np.full(2, 0.9, dtype=np.float32)
    sh = np.zeros((2, 3), dtype=np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    gs_map.ingest_keyframe(0, pts, q, sc, op, sh, np.ones(2), np.ones(2), T_id, T_PW)

    bbox = (-0.6, -0.6, 0.6, 0.6)
    r1 = gs_map.carve_from_depth(1, depth_map, K_d, T_WC, T_PW, bbox, carve_tol=0.08, min_votes=2, device="cpu")
    r2 = gs_map.carve_from_depth(2, depth_map, K_d, T_WC, T_PW, bbox, carve_tol=0.08, min_votes=2, device="cpu")
    assert r1["voted"] == 1
    assert r2["voted"] == 1 and r2["carved"] == 1
    assert gs_map.n_points == 1
    np.testing.assert_allclose(gs_map.means[0], [0.06, 0.0, 0.46], atol=1e-5)


def test_stage3_gpu_ingest_matches_cpu_and_avoids_spurious_bucket_rebuild():
    """Torch CUDA (or CPU tensor) ingest produces identical sorted columns and keeps buckets clean."""
    torch = pytest.importorskip("torch")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(19)
    n = 1200
    xy = rng.uniform(0.0, 0.4, size=(n, 2)).astype(np.float32)
    z = rng.normal(0.0, 0.01, size=(n, 1)).astype(np.float32)
    pts = np.hstack([xy, z])
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    sc = np.full((n, 3), 0.01, dtype=np.float32)
    op = np.full(n, 0.9, dtype=np.float32)
    sh = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    ns = rng.uniform(1.0, 10.0, size=n).astype(np.float32)
    cf = np.ones(n, dtype=np.float32)
    rays = np.tile([0.0, 0.0, -1.0], (n, 1)).astype(np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)

    g_cpu = StreamingGaussianMap(voxel_size=0.02, tile_world_size=1.0)
    g_gpu = StreamingGaussianMap(voxel_size=0.02, tile_world_size=1.0)
    r_c = g_cpu.ingest_keyframe(0, pts, q, sc, op, sh, ns, cf, T_id, np.eye(4), cam_rays_p=rays)
    r_g = g_gpu.ingest_keyframe(
        0,
        torch.from_numpy(pts).to(dev),
        torch.from_numpy(q).to(dev),
        torch.from_numpy(sc).to(dev),
        torch.from_numpy(op).to(dev),
        torch.from_numpy(sh).to(dev),
        torch.from_numpy(ns).to(dev),
        torch.from_numpy(cf).to(dev),
        T_id,
        np.eye(4),
        cam_rays_p=torch.from_numpy(rays).to(dev),
    )
    assert r_c["added"] == r_g["added"]
    assert np.all(np.diff(g_cpu._col_h) > 0)
    assert np.array_equal(g_cpu._col_h, g_gpu._col_h)
    np.testing.assert_allclose(g_cpu.means[g_cpu._col_i], g_gpu.means[g_gpu._col_i], atol=1e-5)

    # Trigger bucket build, then replace columns in-cell from a better nadir view: buckets stay clean!
    _ = g_gpu.query_bbox(0.0, 0.0, 0.5, 0.5)
    assert g_gpu._buckets_dirty is False
    r_g2 = g_gpu.ingest_keyframe(
        1,
        torch.from_numpy(pts).to(dev),
        torch.from_numpy(q).to(dev),
        torch.from_numpy(sc).to(dev),
        torch.from_numpy(op).to(dev),
        torch.from_numpy(sh).to(dev),
        torch.from_numpy(ns * 2.0).to(dev),
        torch.from_numpy(cf).to(dev),
        T_id,
        np.eye(4),
        cam_rays_p=torch.from_numpy(rays).to(dev),
    )
    assert r_g2["replaced"] > 0
    assert g_gpu._buckets_dirty is False


def test_stage3_incremental_moved_subset_dedup_matches_full_dedup():
    """Incremental _deduplicate_moved_subset must match full deduplicate_columns cleanly."""
    rng = np.random.default_rng(23)
    T0 = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    g_inc = StreamingGaussianMap(voxel_size=0.05, tile_world_size=2.0)
    g_full = StreamingGaussianMap(voxel_size=0.05, tile_world_size=2.0)

    for kf_id, x_off in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
        gx, gy = np.meshgrid(np.linspace(x_off, x_off + 0.8, 15), np.linspace(0.0, 0.8, 15))
        n = gx.size
        pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n)]).astype(np.float32)
        q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
        sc = np.full((n, 3), 0.02, dtype=np.float32)
        op = np.full(n, 0.9, dtype=np.float32)
        sh = np.full((n, 3), float(kf_id), dtype=np.float32)
        ns = rng.uniform(1.0, 5.0, size=n).astype(np.float32)
        cf = np.ones(n, dtype=np.float32)
        g_inc.ingest_keyframe(kf_id, pts, q, sc, op, sh, ns, cf, T0, np.eye(4))
        g_full.ingest_keyframe(kf_id, pts, q, sc, op, sh, ns, cf, T0, np.eye(4))

    # Move kf=1 right on top of kf=0 (1/5 of points), causing cross-kf competition
    for g in (g_inc, g_full):
        sel1 = g.owner_kf == 1
        g.means[sel1, 0] -= 0.98
    moved_mask = g_inc.owner_kf == 1
    g_inc._deduplicate_moved_subset(moved_mask)
    g_full.deduplicate_columns()

    assert g_inc.n_points == g_full.n_points
    assert np.array_equal(g_inc._col_h, g_full._col_h)
    np.testing.assert_allclose(g_inc.means[g_inc._col_i], g_full.means[g_full._col_i], atol=1e-5)
    assert np.array_equal(g_inc.owner_kf[g_inc._col_i], g_full.owner_kf[g_full._col_i])


def test_stage3_async_ortho_worker_order_coalescing_and_errors():
    """_AsyncOrthoWorker drains keyframes before pending poses, coalesces poses, and raises errors."""
    import threading
    import time
    from types import SimpleNamespace
    from stream_ortho.app.pipeline import _AsyncOrthoWorker

    events = []
    gate = threading.Event()

    def on_kf(pkt):
        if pkt.kf == 0:
            gate.wait(timeout=2.0)
        events.append(("kf", pkt.kf, tuple(pkt.T_WC.tolist())))

    def on_pose(poses):
        events.append(("pose", {k: v[0] for k, v in sorted(poses.items())}))

    w = _AsyncOrthoWorker(on_kf, on_pose, max_queue=8)
    w.submit_keyframe(SimpleNamespace(kf=0, T_WC=np.zeros(8)))
    time.sleep(0.02)
    # While worker blocks on kf=0, submit kf=1 and multiple rapid BA pose updates
    w.submit_keyframe(SimpleNamespace(kf=1, T_WC=np.zeros(8)))
    w.submit_pose_updates({0: np.array([1.0] + [0.0] * 7), 1: np.array([2.0] + [0.0] * 7)})
    w.submit_pose_updates({0: np.array([3.0] + [0.0] * 7), 1: np.array([4.0] + [0.0] * 7)})
    gate.set()
    w.drain()

    # kf=1 got the latest coalesced pose (4.0) directly; pose batch only ran for kf=0 (3.0) afterwards
    assert events[0][0] == "kf" and events[0][1] == 0
    assert events[1][0] == "kf" and events[1][1] == 1 and events[1][2][0] == pytest.approx(4.0)
    assert events[2] == ("pose", {0: 3.0})
    assert w.coalesced_pose_batches >= 1

    # Exception propagation check
    def bad_kf(_pkt):
        raise ValueError("boom-worker")

    w_err = _AsyncOrthoWorker(bad_kf, on_pose, max_queue=4)
    w_err.submit_keyframe(SimpleNamespace(kf=0, T_WC=np.zeros(8)))
    with pytest.raises(ValueError, match="boom-worker"):
        w_err.drain()


def test_stage3_fuse_z_slots_torch_matches_numpy_and_uniform_4096_subsample():
    """GPU _fuse_z_slots_torch matches CPU NumPy and aligner 4096 subsample covers full cloud."""
    from stream_ortho.online.gs_map import _fuse_z_slots, _fuse_z_slots_torch

    rng = np.random.default_rng(31)
    m = 512
    z4 = rng.uniform(0.0, 0.5, size=(m, 4)).astype(np.float32)
    w4 = rng.uniform(0.0, 5.0, size=(m, 4)).astype(np.float32)
    w4[:, 3] = 0.0  # mix of empty and full slots
    kf4 = rng.integers(0, 4, size=(m, 4), dtype=np.int32)
    kf4[:, 3] = -1
    zn = rng.uniform(0.0, 0.5, size=m).astype(np.float32)
    wn = rng.uniform(0.5, 6.0, size=m).astype(np.float32)

    # Exercise NumPy reference on a slice (bypass >=50_000 GPU gate) vs GPU helper when CUDA exists
    z4_c, w4_c, kf4_c, zm_c = _fuse_z_slots(z4.copy(), w4.copy(), kf4.copy(), zn, wn, 2)
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        z4_g, w4_g, kf4_g, zm_g = _fuse_z_slots_torch(z4.copy(), w4.copy(), kf4.copy(), zn, wn, 2)
        np.testing.assert_allclose(zm_c, zm_g, atol=1e-5)
        np.testing.assert_allclose(z4_c, z4_g, atol=1e-5)
        np.testing.assert_allclose(w4_c, w4_g, atol=1e-5)
        assert np.array_equal(kf4_c, kf4_g)

    # Cloud of 7372 ground points sorted top-to-bottom where bottom half has a distinct Z slope:
    # uniform 4096 subsampling must span the full [0, 7371] range rather than slicing [:4096].
    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    n_p = 16384
    xy = np.column_stack([
        np.linspace(-5.0, 5.0, n_p),
        rng.uniform(-5.0, 5.0, size=n_p),
    ])
    pts_flat = np.column_stack([xy, np.zeros(n_p)])
    assert aligner.add(pts_flat) is True
    # Tilt by 8 deg in Y on a 16384-pt cloud sorted in X: uniform 4096 + 2048 subsampling
    # must span the full [-5, +5] X range rather than truncating via [:4096] or [:2048].
    ang = np.radians(8.0)
    pts_tilt = np.column_stack([xy[:, 0], xy[:, 1], xy[:, 1] * np.sin(ang)])
    pp, _, info = aligner.rectify_keyframe(1, pts_tilt)
    assert info["rectified"] is True
    gnd_saved = aligner._kf_gnd_w[1]
    assert gnd_saved[:, 0].min() < -4.5 and gnd_saved[:, 0].max() > 4.5
    assert np.std(pp[:, 2]) < 0.05


def test_carve_from_depth_ignores_downward_drifted_frame_and_tracks_ba_shift_px():
    """A frame whose ground plane drifted <-0.15m must not carve valid map points,
    and update_poses records last_max_shift_px for BA seam crack healing."""
    T_WC = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    T_PW = np.eye(4, dtype=np.float64)
    K_d = (100.0, 100.0, 32.0, 32.0)
    # Drifted depth map places ground at z = 2.0 - 2.4 = -0.40 (< -0.15)
    depth_drifted = np.full((64, 64), 2.4, dtype=np.float32)

    gs_map = StreamingGaussianMap(voxel_size=0.05, tile_world_size=10.0)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sc = np.full((1, 3), 0.02, dtype=np.float32)
    op = np.array([0.9], dtype=np.float32)
    sh = np.zeros((1, 3), dtype=np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    gs_map.ingest_keyframe(0, pts, q, sc, op, sh, np.ones(1), np.ones(1), T_id, T_PW)

    r1 = gs_map.carve_from_depth(1, depth_drifted, K_d, T_WC, T_PW, (-0.5, -0.5, 0.5, 0.5), device="cpu")
    assert r1["voted"] == 0 and gs_map.n_points == 1

    # Shift kf=0 by +0.25m (5 cells at voxel_size=0.05) -> last_max_shift_px == 5.0
    T_shift = np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    gs_map.update_poses({0: T_shift})
    assert gs_map.last_max_shift_px == pytest.approx(5.0, abs=1e-4)


def test_map_relative_aligner_resists_dense_rooftop_canyon_and_global_relevels():
    """Map-relative alignment must not fit phantom p35 tilts over dense rooftops/trees,
    while 2D global re-leveling removes bootstrap tilt once multi-strip span develops."""
    rng = np.random.default_rng(123)
    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1)
    # Build initial frame tilted globally by 4 deg around X so global relevel is exercised
    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, 40), np.linspace(-2.0, 2.0, 40))
    pts0 = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=np.float64)])
    # 75% tall rooftop (+1.2m) and stepped canopy (+0.4m) leaving < 25% bare ground
    roof_mask = (gx.ravel() > -1.0) & (gy.ravel() > -1.0)
    pts0[roof_mask, 2] = 1.2
    ax = np.radians(4.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    pts_init_gnd = np.column_stack([np.linspace(-1, 1, 200), np.linspace(-1, 1, 200), np.zeros(200)])
    pts_init_gnd = pts_init_gnd @ Rx.T
    assert aligner.add(np.vstack([pts_init_gnd, pts_init_gnd + [0.1, -0.1, 0.0]])) is True

    # Commit tilted map strip at 4 deg tilt relative to bootstrap
    pts_tilted = pts0 @ np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]).T
    pts_p0 = aligner.to_plane(pts_tilted)
    gs_map = StreamingGaussianMap(voxel_size=0.08, tile_world_size=10.0, z_band=0.06)
    n0 = pts_p0.shape[0]
    gs_map.ingest_keyframe(
        0, pts_p0, np.tile([1, 0, 0, 0], (n0, 1)).astype(np.float32),
        np.full((n0, 3), 0.04, np.float32), np.full(n0, 0.9, np.float32),
        np.zeros((n0, 3), np.float32), np.ones(n0, np.float32), np.ones(n0, np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1, 1], float), aligner.T_PW,
    )

    # New overlapping keyframe dominated (80%) by tall +1.2m rooftop in the same frame:
    # must align smoothly to gs_map WITHOUT hallucinating a bogus p35 tilt!
    gx1, gy1 = np.meshgrid(np.linspace(-1.5, 2.0, 35), np.linspace(-1.5, 2.0, 35))
    pts1 = np.column_stack([gx1.ravel(), gy1.ravel(), np.zeros(gx1.size, dtype=np.float64)])
    r1 = (gx1.ravel() > -1.0) & (gy1.ravel() > -1.0)
    pts1[r1, 2] = 1.2 + rng.normal(0.0, 0.005, size=int(r1.sum()))
    pts1_w = pts1 @ Rx.T
    pp1, _, info1 = aligner.rectify_keyframe(1, pts1_w, surface_ref=gs_map)
    assert info1["rectified"] is False
    assert abs(info1["dz"]) < 0.08

    # Inject a tilted second strip to give full 2D spatial leverage and test global releveling
    gx2, gy2 = np.meshgrid(np.linspace(-4.0, 4.0, 60), np.linspace(-4.0, 4.0, 60))
    g2_w = np.column_stack([gx2.ravel(), gy2.ravel(), np.zeros(gx2.size)])
    tilted_full_p = aligner.to_plane(g2_w @ Rx.T)
    # Apply explicit 5-deg global tilt onto a clean gs_map2 so 2D global relevel is unambiguously needed
    ay = np.radians(5.0)
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    g_skew = np.column_stack([gx2.ravel(), gy2.ravel(), np.zeros(gx2.size)])
    g_skew[(gx2.ravel() > 0) & (gy2.ravel() > 0), 2] = 1.5
    g_skew = g_skew @ Ry.T + np.array([0.0, 0.0, 0.35])
    gs_map2 = StreamingGaussianMap(voxel_size=0.08, tile_world_size=20.0)
    n2 = g_skew.shape[0]
    gs_map2.ingest_keyframe(
        0, g_skew, np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        np.full((n2, 3), 0.04, np.float32), np.full(n2, 0.9, np.float32),
        np.zeros((n2, 3), np.float32), np.ones(n2, np.float32), np.ones(n2, np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1, 1], float), np.eye(4),
    )
    dT = aligner.estimate_global_relevel(gs_map2.means)
    assert dT is not None
    gs_map2.apply_plane_transform(dT)
    aligner.apply_global_plane_transform(dT, map_pts_p=gs_map2.means)
    gnd_after = gs_map2.means[gs_map2.means[:, 2] < 0.8, 2]
    assert abs(np.median(gnd_after)) < 0.05
    assert np.std(gnd_after) < 0.05


def test_carve_from_depth_edge_guard_and_sloped_terrain_safety():
    """5x5 boundary guard protects roof edges and relative ground floor supports sloped terrain."""
    T_WC = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    T_PW = np.eye(4, dtype=np.float64)
    K_d = (100.0, 100.0, 32.0, 32.0)
    H_d, W_d = 64, 64

    # Ground slopes/sits at z=-0.45m (depth=2.45m), roof patch sits at z=0.05m (depth=1.95m)
    depth_map = np.full((H_d, W_d), 2.45, dtype=np.float32)
    depth_map[24:41, 34:49] = 1.95

    gs_map = StreamingGaussianMap(voxel_size=0.05, tile_world_size=10.0)
    # pt 0: valid sloped ground point at z=-0.45 (must NOT self-carve)
    # pt 1: roof corner projecting within 1px of ground-to-roof discontinuity at u=33 (u=34 is edge)
    # pt 2: free-air ghost at (-0.30, 0, +0.10) over flat interior ground (-0.45 -> gap 0.55m)
    pts = np.array([[-0.10, 0.0, -0.45], [0.02, 0.0, 0.05], [-0.30, 0.0, 0.10]], dtype=np.float32)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)).astype(np.float32)
    sc = np.full((3, 3), 0.02, dtype=np.float32)
    op = np.full(3, 0.9, dtype=np.float32)
    sh = np.zeros((3, 3), dtype=np.float32)
    T_id = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    gs_map.ingest_keyframe(0, pts, q, sc, op, sh, np.ones(3), np.ones(3), T_id, T_PW)

    bbox = (-0.6, -0.6, 0.6, 0.6)
    r1 = gs_map.carve_from_depth(1, depth_map, K_d, T_WC, T_PW, bbox, carve_tol=0.10, min_votes=2, device="cpu")
    r2 = gs_map.carve_from_depth(2, depth_map, K_d, T_WC, T_PW, bbox, carve_tol=0.10, min_votes=2, device="cpu")
    assert r1["voted"] == 1
    assert r2["voted"] == 1 and r2["carved"] == 1
    assert gs_map.n_points == 2
    assert np.min(gs_map.means[:, 2]) == pytest.approx(-0.45, abs=1e-4)


def test_fit_rect_rejects_flat_frame_edge_rotation_and_small_dz_chaining():
    """Already-flat frames cannot rotate >4 deg on cliff mismatches, and <deadband dz is skipped."""
    rng = np.random.default_rng(7)
    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=1, max_tilt_deg=35.0, min_tilt_deg=0.5, min_inlier_frac=0.25)

    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 80), np.linspace(-2.5, 2.5, 80))
    pts_map = np.column_stack([gx.ravel(), gy.ravel(), rng.normal(0.0, 0.005, gx.size)])
    assert aligner.add(pts_map) is True
    pts_map, _, _ = aligner.rectify_keyframe(0, pts_map)

    gs_map = StreamingGaussianMap(voxel_size=0.05, tile_world_size=12.0)
    # Build a cliff/building step at x > 0.35 in the committed map (x <= 0.35 is flat ground)
    pts_map[pts_map[:, 0] > 0.35, 2] += 1.5
    n0 = pts_map.shape[0]
    gs_map.ingest_keyframe(
        0,
        pts_map,
        np.tile([1, 0, 0, 0], (n0, 1)).astype(np.float32),
        np.full((n0, 3), 0.03, np.float32),
        np.full(n0, 0.9, np.float32),
        np.zeros((n0, 3), np.float32),
        np.ones(n0, np.float32),
        np.ones(n0, np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1, 1], float),
        np.eye(4),
    )

    # Case 1: Flat frame with minor 0.03m residual offset stays inside deadband (dz == 0.0)
    gx1, gy1 = np.meshgrid(np.linspace(-2.0, 0.2, 40), np.linspace(-1.5, 1.5, 40))
    pts_small = np.column_stack([gx1.ravel(), gy1.ravel(), np.full(gx1.size, 0.03) + rng.normal(0.0, 0.004, gx1.size)])
    _, _, info_small = aligner.rectify_keyframe(1, pts_small, surface_ref=gs_map)
    assert info_small["rectified"] is False
    assert info_small["dz"] == 0.0

    # Case 2: Flat frame overlapping flat ground while only its edge ramp disagrees (>4 deg residual)
    # must NOT rotate the flat low surface into the air.
    gx3, gy3 = np.meshgrid(np.linspace(-3.5, 0.35, 55), np.linspace(-1.5, 1.5, 40))
    xr = gx3.ravel()
    z3 = np.where(xr > -0.3, -0.35 * (xr + 0.3), rng.normal(0.0, 0.004, xr.size))
    pts_edge = np.column_stack([xr, gy3.ravel(), z3])
    pp3, _, info_edge = aligner.rectify_keyframe(2, pts_edge, surface_ref=gs_map)
    assert info_edge["rectified"] is False
    assert float(np.std(pp3[xr <= -0.5, 2])) < 0.02

