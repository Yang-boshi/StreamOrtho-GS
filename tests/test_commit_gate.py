"""S5a tests: commit gate + voxel dedup/merge + unique-coverage protection.

Design-doc acceptance (§7.5, "合成用例"):

* duplicate points are merged (and do not double the map);
* outliers are rejected by the gate;
* unique coverage is **not** deleted;
* commit waits for multi-view confirmation (or age).

Run::

    python -m pytest stream_ortho/tests/test_commit_gate.py -q
"""

from __future__ import annotations

import numpy as np

from stream_ortho.map.commit import GateConfig, commit_gate
from stream_ortho.map.gaussian import GaussianPatch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig

T_WP = np.eye(4)


def _patch(pts, colors=None, opacity=0.9, conf=2.0, scale=0.3, owner=0):
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    n = len(pts)
    colors = np.zeros((n, 3), np.float32) if colors is None else np.asarray(colors, np.float32)
    q = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    return GaussianPatch(means=pts, quats=q, scales=np.full((n, 3), scale, np.float32),
                         opacity=np.full(n, opacity, np.float32), colors=colors,
                         conf=np.full(n, conf, np.float32), owner=owner)


def test_gate_rejects_outliers():
    cfg = GateConfig()
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 500], [0, 0, -50], [0, 0, 1.0]], np.float32)
    p = _patch(pts)
    p.meta["depth_m"] = 200.0
    res = commit_gate(p, cfg, T_WP, z_ground=0.0)
    assert res["n_valid"] == 3, res          # (0,0,0), (1,0,0), (0,0,1)
    assert res["reasons"]["height_high"] == 1
    assert res["reasons"]["height_low"] == 1
    print(f"[ok] gate: {res['n_valid']}/{res['n']} valid, reasons={res['reasons']}")


def test_gate_rejects_bad_conf_opacity_scale():
    cfg = GateConfig()
    p = _patch(np.zeros((4, 3)), opacity=np.array([0.9, 0.01, 0.9, 0.9]),
               conf=np.array([2.0, 2.0, 0.1, 2.0]))
    p.scales[3] = 50.0                       # too large -> floater
    res = commit_gate(p, cfg, T_WP, 0.0)
    assert res["n_valid"] == 1, res
    assert res["reasons"]["opacity"] >= 1 and res["reasons"]["conf"] >= 1
    assert res["reasons"]["scale_max"] >= 1
    print(f"[ok] gate thresholds: valid={res['n_valid']}, reasons={res['reasons']}")


def test_dedup_merges_duplicates():
    cfg = MapConfig(voxel_size=0.5, gate=False, dedup=True,
                    commit_min_obs=1, commit_max_age=0)
    m = GaussianMap(cfg, T_WP)
    pts = np.array([[0.01, 0.01, 0], [0.02, 0.0, 0.01]], np.float32)   # same voxel
    m.ingest(_patch(pts), 0)
    m.commit(0)
    n1 = m.n_committed
    assert n1 == 1, n1
    m.ingest(_patch(pts + 0.01), 1)          # re-observed
    m.commit(1)
    assert m.n_committed == 1, m.n_committed
    assert int(m.committed["n_obs"][0]) >= 2
    assert m.duplicate_rate() == 0.0
    print(f"[ok] duplicates merged: N=1, n_obs={int(m.committed['n_obs'][0])}, dup=0")


def test_unique_coverage_preserved():
    cfg = MapConfig(voxel_size=0.5, gate=False, dedup=True,
                    commit_min_obs=1, commit_max_age=0)
    m = GaussianMap(cfg, T_WP)
    pts = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0]], np.float32)
    m.ingest(_patch(pts), 0)
    m.commit(0)
    assert m.n_committed == 4, m.n_committed
    m.ingest(_patch(pts[:1] + 0.01), 1)      # re-observe only one
    m.commit(1)
    assert m.n_committed == 4, m.n_committed
    print(f"[ok] unique coverage kept: N={m.n_committed} (re-observing one point does not delete others)")


def test_commit_needs_confirmation_or_age():
    cfg = MapConfig(voxel_size=0.5, gate=False, dedup=True,
                    commit_min_obs=2, commit_max_age=3)
    m = GaussianMap(cfg, T_WP)
    p = _patch(np.random.default_rng(0).random((10, 3)) * 10)
    m.ingest(p, 0)
    r = m.commit(0)
    assert m.n_committed == 0, "nothing should be committed on the first look"
    m.commit(1)
    m.commit(2)
    assert m.n_committed == 0, "still not confirmed"
    m.commit(3)                              # age = 3 -> commit
    assert m.n_committed > 0, "aged points must commit"
    print(f"[ok] commit rule: staged at kf0, committed after age 3 (N={m.n_committed})")


def test_budget_evicts_to_limit():
    cfg = MapConfig(voxel_size=0.5, gate=False, dedup=False,
                    commit_min_obs=1, commit_max_age=0)
    m = GaussianMap(cfg, T_WP)
    rng = np.random.default_rng(1)
    m.ingest(_patch(rng.random((500, 3)) * 50), 0)
    m.commit(0)
    n0 = m.n_committed
    assert n0 > 100
    res = m.enforce_budget(100, strategy="conf")
    assert m.n_committed == 100, (res, m.n_committed)
    print(f"[ok] budget: {n0} -> {m.n_committed} (evicted {res['evicted']})")


def run_all():
    test_gate_rejects_outliers()
    test_gate_rejects_bad_conf_opacity_scale()
    test_dedup_merges_duplicates()
    test_unique_coverage_preserved()
    test_commit_needs_confirmation_or_age()
    test_budget_evicts_to_limit()
    print("test_commit_gate: ALL PASSED")


if __name__ == "__main__":
    run_all()
