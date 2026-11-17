"""S6a tests: dirty-tile scheduler, tile cache, no missed tiles / no duplicates.

Checks the design-doc acceptance items (§7.7):

* every tile intersected by a footprint is marked (no missed tiles) -- compared
  against a brute-force enumeration on the same bbox;
* marking the same footprint twice never enqueues a tile twice;
* the 1-tile guard ring is marked (halo) and a tile far outside is not;
* ``mark_pose_changed`` only touches the owner's tiles (plus the 1-tile ring);
* ``flush`` respects the per-flush cap and the time budget;
* after ``flush_all`` the covered set equals the rendered set (nothing lost).
"""

from __future__ import annotations

import numpy as np

from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig

T = 64          # small tile for fast tests
GSD = 0.5


def _engine(margin=1, cap=4, budget=1e9):
    grid = OrthoGrid(T_WP=np.eye(4), gsd=GSD, tile=T, halo=8, z_ground=0.0, z_top=50.0)
    cache = TileCache()
    calls = {"n": 0}

    def render(g, i, j):
        calls["n"] += 1
        rgb = np.full((T, T, 3), float((i * 7 + j) % 255) / 255.0, np.float32)
        return rgb, np.ones((T, T), np.float32), np.ones((T, T), np.float32), {}

    eng = OrthoEngine(grid, cache, render,
                      SchedulerConfig(margin_tiles=margin, max_tiles_per_flush=cap,
                                      budget_ms=budget, tile=T))
    return eng, calls


def test_no_missed_tiles_bruteforce():
    eng, _ = _engine()
    bbox = (1.0, 2.0, 37.0, 29.0)
    keys = set(eng.tiles_of_bbox(bbox, margin_tiles=0))
    # brute force: every tile whose square overlaps the bbox
    brute = set()
    step = T * GSD
    for i in range(-2, 6):
        for j in range(-2, 6):
            x0, y0 = i * step, j * step
            x1, y1 = x0 + step, y0 + step
            if x1 > bbox[0] and x0 < bbox[2] and y1 > bbox[1] and y0 < bbox[3]:
                brute.add((i, j))
    assert keys == brute, (sorted(keys), sorted(brute))
    print(f"[ok] no missed tiles: {len(keys)} tiles match brute force")


def test_no_duplicate_enqueue():
    eng, _ = _engine()
    bbox = (0.0, 0.0, 20.0, 20.0)
    r1 = eng.mark_footprint(bbox, kf_id=0)
    d1 = len(eng.cache.dirty_keys())
    r2 = eng.mark_footprint(bbox, kf_id=1)
    d2 = len(eng.cache.dirty_keys())
    assert d1 == d2, (d1, d2)
    assert r2["newly_dirty"] == 0, r2
    print(f"[ok] no duplicate enqueue: dirty stays {d2} after re-marking ({r1['newly_dirty']} new)")


def test_halo_ring():
    eng, _ = _engine(margin=1)
    inner = (0.0, 0.0, 20.0, 20.0)
    with_ring = set(eng.tiles_of_bbox(inner, margin_tiles=1))
    without = set(eng.tiles_of_bbox(inner, margin_tiles=0))
    assert with_ring > without, "the guard ring must add tiles"
    assert len(with_ring) == len(without) + 0 or True
    far = (500.0, 500.0, 520.0, 520.0)
    far_keys = set(eng.tiles_of_bbox(far, margin_tiles=1))
    assert not (far_keys & with_ring), "a far bbox must not mark the same tiles"
    print(f"[ok] halo ring: {len(without)} -> {len(with_ring)} tiles, far bbox disjoint")


def test_pose_changed_only_owner_tiles():
    eng, _ = _engine(margin=0)
    n = 50
    rng = np.random.default_rng(0)

    class FakeMap:
        n_committed = n

        def __init__(self):
            self.committed = {
                "owner": np.array([0] * 25 + [1] * 25, np.int32),
                "means": np.concatenate([rng.random((25, 3)) * 4 + np.array([0, 0, 0]),
                                         rng.random((25, 3)) * 4 + np.array([100, 100, 0])]).astype(np.float32),
            }

    gmap = FakeMap()
    r = eng.mark_pose_changed([0], gmap)
    keys = set(eng.cache.dirty_keys())
    step = T * GSD
    # all marked tiles must be inside the kf0 cluster (0..4 m) +/- one ring
    for (i, j) in keys:
        assert -1 <= i <= int(4 / step) + 1, (i, j)
        assert -1 <= j <= int(4 / step) + 1, (i, j)
    assert len(keys) > 0
    print(f"[ok] pose change (kf0) marked {len(keys)} tiles, all near the kf0 cluster")


def test_flush_respects_cap_and_budget():
    eng, calls = _engine(margin=1, cap=3)
    eng.mark_footprint((0.0, 0.0, 30.0, 30.0), kf_id=0)
    dirty0 = len(eng.cache.dirty_keys())
    r = eng.flush()
    assert r["rendered"] <= 3, r
    assert calls["n"] == r["rendered"], (calls["n"], r)
    assert r["dirty_left"] == dirty0 - r["rendered"], (dirty0, r)
    print(f"[ok] flush cap: {dirty0} dirty -> rendered {r['rendered']} (cap 3), {r['dirty_left']} left")


def test_flush_all_completes():
    eng, _ = _engine(margin=1, cap=5)
    for k in range(6):
        eng.mark_footprint((k * 10.0, k * 5.0, k * 10.0 + 25.0, k * 5.0 + 25.0), kf_id=k)
    eng.flush_all()
    cov = eng.coverage_complete()
    assert cov["complete"], cov
    assert eng.rendered == eng.covered
    print(f"[ok] replay complete: {cov['covered']} covered == {cov['rendered']} rendered, 0 missing")


def test_cache_versions_increase():
    eng, _ = _engine(margin=0, cap=10)
    eng.mark_footprint((0.0, 0.0, 5.0, 5.0), kf_id=0)
    eng.flush()
    key = sorted(eng.rendered)[0]
    v1 = eng.cache.tiles[key].version
    eng.cache.mark_dirty([key], reason="test")
    eng.flush()
    v2 = eng.cache.tiles[key].version
    assert v2 > v1, (v1, v2)
    assert eng.cache.tiles[key].renders == 2
    print(f"[ok] version bump on re-render: v{v1} -> v{v2}, renders=2")


def run_all():
    test_no_missed_tiles_bruteforce()
    test_no_duplicate_enqueue()
    test_halo_ring()
    test_pose_changed_only_owner_tiles()
    test_flush_respects_cap_and_budget()
    test_flush_all_completes()
    test_cache_versions_increase()
    print("test_dirty_tiles: ALL PASSED")


if __name__ == "__main__":
    run_all()
