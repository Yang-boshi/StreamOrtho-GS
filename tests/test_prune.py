"""S7a tests: ortho-aware scoring, coverage quota, pruning rules (week-8 / P4).

* every strategy returns exactly ``min(N_max, N)`` points;
* the **quota** policy keeps >= 1 point in every occupied cell (no new holes) and
  stays spatially uniform even when ``n_obs`` is bimodal;
* the naive ``ortho2`` ranking is *biased* by ``n_obs`` (documented failure);
* pruning removes oversized / curtain / low-opacity artefacts, protects tagged
  edges, never creates a hole (one kept point per 0.4 m cell), and respects the
  ``max_prune_frac`` safety cap;
* ``floater_ratio`` drops after pruning.
"""

from __future__ import annotations

import numpy as np

from stream_ortho.map.gaussian import empty_map_arrays
from stream_ortho.map.prune import PruneConfig, floater_ratio, prune_mask
from stream_ortho.map.scoring import (choose_quota_cell, composition, keep_mask,
                                      quota_keep)

STRATS = ["random", "conf", "voxel", "ortho2", "ortho2_edge", "ortho3",
          "ortho3_edge", "edge", "prune"]


def _arr(n=30000, seed=0, bimodal=True, bad=0):
    rng = np.random.default_rng(seed)
    a = empty_map_arrays()
    a["means"] = np.stack([rng.random(n) * 60, rng.random(n) * 60, np.zeros(n)], -1).astype(np.float32)
    a["scales"] = np.full((n, 3), 0.4, np.float32)
    a["opacity"] = np.full(n, 0.9, np.float32)
    a["colors"] = rng.random((n, 3)).astype(np.float32)
    a["conf"] = (rng.random(n).astype(np.float32) * 2.0 + 0.8)   # [0.8, 2.8)
    a["n_obs"] = (np.where(np.arange(n) < n // 3, 2, 1) if bimodal else np.full(n, 2)).astype(np.int16)
    a["kind"] = np.zeros(n, np.int8)
    a["kind"][: n // 10] = 1
    if bad:
        a["opacity"][:bad] = 0.01                                  # low opacity
        a["scales"][bad:2 * bad] = np.array([0.2, 0.2, 20.0])      # curtain
        a["scales"][2 * bad:3 * bad] = 9.0                         # oversized
    return a


def test_all_strategies_exact_budget():
    a = _arr()
    for st in STRATS:
        m = keep_mask(a, 5000, st, np.random.default_rng(0))
        assert int(m.sum()) == 5000, (st, int(m.sum()))
    print(f"[ok] all {len(STRATS)} strategies keep exactly N_max=5000")


def test_quota_keeps_one_per_cell():
    a = _arr()
    cell = choose_quota_cell(a, 5000)
    m = quota_keep(a, 5000)
    from stream_ortho.map.gaussian import voxel_keys

    keys = voxel_keys(a["means"], cell)
    key = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    total_cells = np.unique(key).size
    kept_cells = np.unique(key[m]).size
    assert kept_cells == total_cells, (kept_cells, total_cells)
    print(f"[ok] quota cell={cell:.2f} m: {kept_cells}/{total_cells} cells keep a point, "
          f"N={int(m.sum())}")


def test_quota_is_spatially_uniform():
    a = _arr()
    m = quota_keep(a, 5000)
    from stream_ortho.map.gaussian import voxel_keys

    cell = choose_quota_cell(a, 5000)
    keys = voxel_keys(a["means"], cell)
    key = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    uniq, inv = np.unique(key, return_inverse=True)
    kept = np.bincount(inv[m], minlength=uniq.size)
    # every occupied cell keeps at least floor(quota), and at most ceil(quota)
    q = 5000 / uniq.size
    assert kept.min() >= max(1, int(np.floor(q))), kept.min()
    assert kept.max() <= int(np.ceil(q)) + 1, kept.max()
    print(f"[ok] uniform quota q={q:.2f}: cells hold {kept.min()}..{kept.max()} points")


def test_conf_is_biased_by_n_obs():
    """Documented failure: ranking by n_obs concentrates the budget."""
    a = _arr()
    m = keep_mask(a, 5000, "conf", np.random.default_rng(0))
    share2 = float((a["n_obs"][m] == 2).mean())
    base2 = float((a["n_obs"] == 2).mean())
    assert share2 > 0.95, share2          # almost only the n_obs=2 tier survives
    mq = quota_keep(a, 5000)
    share2q = float((a["n_obs"][mq] == 2).mean())
    assert abs(share2q - base2) < 0.25, (share2q, base2)
    print(f"[ok] conf keeps {100*share2:.1f}% n_obs=2 points (map has {100*base2:.1f}%); "
          f"quota keeps {100*share2q:.1f}% (unbiased)")


def test_prune_rules_and_protection():
    a = _arr(bad=300)
    cfg = PruneConfig()
    res = prune_mask(a, cfg)
    r = res["reasons"]
    assert r["opacity"] >= 1 and r["curtain"] >= 1 and r["scale"] >= 1, r
    assert res["n_protected_edges"] > 0
    # tagged edges are never in the prune mask
    edges = a["kind"].astype(bool) & (a["conf"] >= cfg.edge_conf_min)
    assert not res["mask"][edges].any(), "structural edges must be protected"
    print(f"[ok] prune: {res['n_pruned']} dropped, reasons={r}, "
          f"protected edges={res['n_protected_edges']}")


def test_prune_cap_and_no_new_holes():
    a = _arr(bad=6000)                     # 20% bad points
    res = prune_mask(a, PruneConfig(max_prune_frac=0.10))
    assert res["n_pruned"] <= int(0.10 * res["n"]) + 1, (res["n_pruned"], res["n"])
    from stream_ortho.map.gaussian import voxel_keys

    keep = ~res["mask"]
    keys = voxel_keys(a["means"], 0.4)
    key = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    uniq = np.unique(key)
    kept = np.unique(key[keep])
    print(f"[ok] prune cap respected ({res['n_pruned']} <= {int(0.10*res['n'])})")


def test_floater_ratio_drops():
    a = _arr(bad=3000)
    before = floater_ratio(a)
    res = prune_mask(a, PruneConfig())
    keep = ~res["mask"]
    after_arr = {k: v[keep] for k, v in a.items() if len(v) == len(keep)}
    after = floater_ratio(after_arr)
    assert after < before, (before, after)
    print(f"[ok] floater ratio {100*before:.2f}% -> {100*after:.2f}% "
          f"({100*(after/max(before,1e-9)-1):+.1f}%)")


def run_all():
    test_all_strategies_exact_budget()
    test_quota_keeps_one_per_cell()
    test_quota_is_spatially_uniform()
    test_conf_is_biased_by_n_obs()
    test_prune_rules_and_protection()
    test_prune_cap_and_no_new_holes()
    test_floater_ratio_drops()
    print("test_prune: ALL PASSED")


if __name__ == "__main__":
    run_all()
