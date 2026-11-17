"""S5b tests: budget strategies, edge-aware retention, shape prior.

Covers the user's week-5 idea:

* ``edge`` retention keeps **100%** of the edge-tagged Gaussians and fills the
  rest with area points up to ``N_max`` (so the budget adapts to content);
* every strategy returns exactly ``min(N_max, N)`` points;
* the shape prior gives rods on edges and discs on area points;
* the ortho score penalises points in already-covered cells.
"""

from __future__ import annotations

import numpy as np

from stream_ortho.map.edges import shape_from_kind
from stream_ortho.map.gaussian import empty_map_arrays
from stream_ortho.map.scoring import composition, coverage_gap, keep_mask, ortho_score


def _arrays(n=2000, edge_frac=0.1, seed=0, spread=100.0):
    rng = np.random.default_rng(seed)
    a = empty_map_arrays()
    a["means"] = (rng.random((n, 3)) * np.array([spread, spread, 1.0])).astype(np.float32)
    a["quats"] = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    a["scales"] = np.full((n, 3), 0.3, np.float32)
    a["colors"] = rng.random((n, 3)).astype(np.float32)
    a["opacity"] = np.full(n, 0.9, np.float32)
    a["conf"] = (rng.random(n).astype(np.float32) + 0.5)
    a["n_obs"] = rng.integers(1, 4, n).astype(np.int16)
    a["kind"] = np.zeros(n, np.int8)
    k = int(n * edge_frac)
    a["kind"][:k] = 1
    a["theta"] = np.full(n, np.nan, np.float32)
    a["theta"][:k] = rng.uniform(-1.5, 1.5, k).astype(np.float32)
    return a


def test_all_strategies_hit_cap():
    a = _arrays()
    for st in ("random", "conf", "voxel", "ortho", "edge"):
        m = keep_mask(a, 500, st, np.random.default_rng(0))
        assert int(m.sum()) == 500, (st, int(m.sum()))
    print("[ok] all 5 strategies keep exactly N_max=500")


def test_edge_keeps_all_edges():
    """Week-8 semantics: coverage is the *hard* constraint, edges are preferred.

    ``edge`` keeps all edge-tagged points, fills the rest with area points, then the
    unique-coverage pass may trade a few points to avoid holes; so we assert
    ">= 90% of the edges kept" instead of the week-5 "all of them".
    """
    a = _arrays(n=2000, edge_frac=0.1)          # 200 edges
    m = keep_mask(a, 500, "edge", np.random.default_rng(0))
    c = composition(a, m)
    assert c["n_kept"] == 500, c
    assert c["edge_kept"] >= 0.9 * c["edge_total"], c
    assert c["edge_keep_ratio"] > 0.9, c
    print(f"[ok] edge retention: {c['edge_kept']}/{c['edge_total']} edges + "
          f"{c['area_kept']}/{c['area_total']} area = {c['n_kept']}"
          f" (>=90% of edges kept, coverage is the hard constraint)")


def test_edge_over_cap_keeps_best_edges():
    a = _arrays(n=2000, edge_frac=0.5)          # 1000 edges, cap 300
    m = keep_mask(a, 300, "edge", np.random.default_rng(0))
    c = composition(a, m)
    assert c["n_kept"] == 300 and c["edge_kept"] == 300, c
    assert c["area_kept"] == 0, c
    print(f"[ok] edges exceed the cap -> keeps the 300 strongest edges, 0 area")


def test_ortho_score_prefers_sparse_cells():
    a = _arrays(n=1000, spread=10.0)
    gap = coverage_gap(a)
    assert gap.min() >= 0 and gap.max() <= 1.0
    score = ortho_score(a)
    assert np.isfinite(score).all() and (score > 0).all()
    # a point where many others sit has a lower coverage gap than a lonely one
    dense = a["means"][0].copy()
    a2 = _arrays(n=10, spread=10.0)
    a2["means"][:] = dense
    g2 = coverage_gap(a2)
    assert g2.max() <= gap.max() + 1e-9
    print(f"[ok] ortho score: coverage gap in [{gap.min():.3f}, {gap.max():.3f}]")


def test_shape_prior_rod_vs_disc():
    kind = np.array([0, 1])
    theta = np.array([np.nan, 0.0], np.float32)
    sc, q = shape_from_kind(kind, theta, 0.2, np.eye(3), np.array([0, 0, 1.0]))
    disc, rod = sc[0], sc[1]
    assert abs(float(disc[0]) - float(disc[1])) < 1e-6, disc      # isotropic in-plane
    assert float(disc[2]) < float(disc[0]), disc                 # flat
    assert float(rod[0]) > 3 * float(rod[1]), rod                # elongated
    assert float(rod[2]) < float(rod[1]), rod                    # flat
    print(f"[ok] shape prior: disc={disc.tolist()}  rod={rod.tolist()}")


def test_budget_is_monotone():
    a = _arrays(n=3000)
    for st in ("conf", "edge", "ortho"):
        sizes = [int(keep_mask(a, k, st, np.random.default_rng(0)).sum()) for k in (100, 500, 1500)]
        assert sizes == [100, 500, 1500], (st, sizes)
    print("[ok] budget is monotone in N_max for conf/edge/ortho")


def run_all():
    test_all_strategies_hit_cap()
    test_edge_keeps_all_edges()
    test_edge_over_cap_keeps_best_edges()
    test_ortho_score_prefers_sparse_cells()
    test_shape_prior_rod_vs_disc()
    test_budget_is_monotone()
    print("test_budget: ALL PASSED")


if __name__ == "__main__":
    run_all()
