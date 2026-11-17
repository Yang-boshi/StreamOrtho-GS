"""S6b tests: pose update + async worker + tear-free incremental re-render.

Covers the design-doc §7.8 acceptance:

* a pose update moves **only** the owner group (other Gaussians bit-identical);
* means move by ``s*R*p + t`` and quats rotate by ``q_delta ⊗ q`` (wxyz),
  verified against a known 90-degree rotation;
* an identity delta is a no-op;
* the inverse delta restores the original means/quats;
* the worker de-duplicates, respects priority and the per-pump budget;
* **tearing**: after a pose update + re-rendering the dirty tiles, the mosaic
  equals a full re-render of the same map (bit-exact within float noise).
"""

from __future__ import annotations

import numpy as np

from stream_ortho.backend.workers import TileWorker, WorkerConfig
from stream_ortho.geometry import R_to_quat_wxyz, quat_xyzw_to_R, sim3_matrix
from stream_ortho.map.gaussian import GaussianPatch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.pose_update import (apply_pose_update, invert_delta,
                                          sim3_vec_from_matrix, world_delta)
from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig

T = 64
GSD = 0.1


def _map(n_per_group=40, groups=(0, 1), spread=6.0, seed=0):
    rng = np.random.default_rng(seed)
    cfg = MapConfig(voxel_size=0.2, dedup=False, gate=False,
                    commit_min_obs=1, commit_max_age=0)
    gmap = GaussianMap(cfg, np.eye(4))
    for g in groups:
        n = n_per_group
        pts = (rng.random((n, 3)) * spread + np.array([g * 20.0, 0, 0])).astype(np.float32)
        p = GaussianPatch(
            means=pts, quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            scales=np.full((n, 3), 0.2, np.float32), opacity=np.full(n, 0.9, np.float32),
            colors=rng.random((n, 3)).astype(np.float32),
            conf=np.full(n, 2.0, np.float32), owner=g, kf_id=g)
        gmap.ingest(p, g)
        gmap.commit(g)
    return gmap


def _delta(t=(0, 0, 0), yaw_deg=0.0):
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    Tm = np.eye(4)
    Tm[:3, :3] = R
    Tm[:3, 3] = np.asarray(t, float)
    return sim3_vec_from_matrix(Tm)


def test_only_owner_group_moves():
    gmap = _map()
    before = gmap.committed["means"].copy()
    owner = gmap.committed["owner"].copy()
    res = apply_pose_update(gmap, [1], _delta(t=(2.0, -1.0, 0.5), yaw_deg=5.0))
    after = gmap.committed["means"]
    other = owner == 0
    assert res["n_moved"] == int((owner == 1).sum()), res
    assert np.array_equal(before[other], after[other]), "non-owner Gaussians must not move"
    assert not np.allclose(before[~other], after[~other]), "owner Gaussians must move"
    print(f"[ok] pose update moved {res['n_moved']} Gaussians "
          f"(max {res['max_displacement_m']:.2f} m), {int(other.sum())} untouched")


def test_quaternion_rotation():
    gmap = _map(n_per_group=4)
    # a rod whose local +x axis is world +x
    gmap.committed["quats"][:] = np.array([1.0, 0, 0, 0], np.float32)   # identity wxyz
    apply_pose_update(gmap, [0], _delta(yaw_deg=90.0))
    q = gmap.committed["quats"][0].astype(np.float64)
    R = quat_xyzw_to_R(np.array([q[1], q[2], q[3], q[0]]))
    x_axis = R[:, 0]
    assert np.allclose(x_axis, [0, 1, 0], atol=1e-5), x_axis
    print(f"[ok] quats rotated by 90 deg: local +x -> {np.round(x_axis, 3)}")


def test_identity_and_inverse():
    gmap = _map()
    m0 = gmap.committed["means"].copy()
    q0 = gmap.committed["quats"].copy()
    apply_pose_update(gmap, [0], _delta())
    assert np.allclose(m0, gmap.committed["means"], atol=1e-6)
    d = _delta(t=(1.5, 0.5, 0.2), yaw_deg=3.0)
    apply_pose_update(gmap, [0], d)
    apply_pose_update(gmap, [0], invert_delta(d))
    assert np.allclose(m0, gmap.committed["means"], atol=1e-4), "inverse must restore means"
    assert np.allclose(q0, gmap.committed["quats"], atol=1e-4), "inverse must restore quats"
    print("[ok] identity is a no-op, inverse delta restores means and quats")


def test_world_delta_roundtrip():
    T_old = np.eye(4)
    T_new = np.eye(4)
    T_new[:3, 3] = [1.0, 2.0, -0.5]
    d = world_delta(T_old, T_new)
    assert np.allclose(d[:3], [1.0, 2.0, -0.5], atol=1e-9)
    back = world_delta(T_new, T_old)
    assert np.allclose(back[:3], [-1.0, -2.0, 0.5], atol=1e-9)
    print("[ok] world_delta = T_new @ T_old^-1")


def _engine(gmap, margin=1, cap=16):
    grid = OrthoGrid(np.eye(4), GSD, tile=T, halo=8, z_ground=0.0, z_top=50.0)
    cache = TileCache()

    def render(g, i, j):
        from stream_ortho.ortho.rasterizer import render_tile

        rgb, a, d, _ = render_tile(gmap.to_cloud(), g, i, j)
        return rgb, a, d, {}

    eng = OrthoEngine(grid, cache, render,
                      SchedulerConfig(margin_tiles=margin, max_tiles_per_flush=cap, tile=T))
    return grid, cache, eng


def test_tearing_free_after_pose_update():
    gmap = _map(n_per_group=60, spread=4.0)
    grid, cache, eng = _engine(gmap)
    cap = 16
    # initial full render
    bbox = (-1.0, -1.0, 25.0, 6.0)
    eng.mark_footprint(bbox, kf_id=-1)
    eng.flush_all()
    before = cache.mosaic(sorted(eng.rendered), T).copy()

    # pose update on group 1 + re-render only its dirty tiles
    apply_pose_update(gmap, [1], _delta(t=(0.6, 0.3, 0.0), yaw_deg=1.0))
    eng.mark_pose_changed([1], gmap)
    eng.flush_all()
    after_incremental = cache.mosaic(sorted(eng.rendered), T).copy()

    # oracle: full re-render of the same map
    cache2 = TileCache()
    eng2 = OrthoEngine(grid, cache2, eng.render_fn,
                       SchedulerConfig(margin_tiles=1, max_tiles_per_flush=cap, tile=T))
    eng2.mark_footprint(bbox, kf_id=-1)
    eng2.flush_all()
    oracle = cache2.mosaic(sorted(eng2.rendered), T)

    diff = np.abs(after_incremental - oracle)
    assert diff.max() < 1e-5, f"incremental vs full re-render differs by {diff.max():.2e}"
    changed = np.abs(after_incremental - before).max(axis=2) > 1e-6
    print(f"[ok] tear-free: incremental == full re-render (max diff {diff.max():.2e}), "
          f"{100*changed.mean():.1f}% of the mosaic changed")


def test_worker_dedup_priority_budget():
    grid = OrthoGrid(np.eye(4), GSD, tile=T, halo=8, z_ground=0.0, z_top=50.0)
    cache = TileCache()
    calls = []

    def render(g, i, j):
        calls.append((i, j))
        return (np.zeros((T, T, 3), np.float32), np.ones((T, T), np.float32),
                np.ones((T, T), np.float32), {})

    w = TileWorker(cache, render, grid, WorkerConfig(budget_ms=1e9, max_tiles=2))
    n1 = w.submit([(0, 0), (0, 1)], priority=0.0)
    n2 = w.submit([(0, 1), (0, 2)], priority=5.0)     # (0,1) is a duplicate
    assert n1 == 2 and n2 == 1, (n1, n2)
    assert w.pending() == 3
    r = w.pump()
    assert r["rendered"] == 2 and w.pending() == 1, r
    # high priority tile must be rendered first
    assert calls[0] == (0, 2), calls
    w.drain()
    assert w.pending() == 0
    print(f"[ok] worker: dedup({w.deduped}), priority honoured, budget respected, "
          f"pending drained ({w.rendered} renders)")


def run_all():
    test_only_owner_group_moves()
    test_quaternion_rotation()
    test_identity_and_inverse()
    test_world_delta_roundtrip()
    test_tearing_free_after_pose_update()
    test_worker_dedup_priority_budget()
    print("test_pose_update: ALL PASSED")


if __name__ == "__main__":
    run_all()
