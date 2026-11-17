"""Tests for the P0 fixes: ray grid, nearest-hit raycast, roof-placement metric."""

import numpy as np
import pytest

from stream_ortho.map.placement import local_ground, roof_mask, roof_placement_accuracy


class _BoxDSM:
    """Flat ground at z=-2 with a 30x30 m box of height 15 m on it."""

    def __init__(self, gsd=0.5, half=400.0, box=(-150.0, -150.0, 150.0, 150.0), h=15.0,
                 ground=-2.0):
        self.gsd, self.half, self.h = gsd, half, h
        self.x0, self.y0 = -half, -half
        n = int(2 * half / gsd) + 1
        self.z = np.full((n, n), ground, np.float32)
        bx0, by0, bx1, by1 = box
        i0 = int((bx0 - self.x0) / gsd); i1 = int((bx1 - self.x0) / gsd)
        j0 = int((by0 - self.y0) / gsd); j1 = int((by1 - self.y0) / gsd)
        self.z[j0:j1, i0:i1] = h

    def lookup(self, x, y, fill=np.nan):
        ij = np.stack([(np.asarray(x) - self.x0) / self.gsd,
                       (np.asarray(y) - self.y0) / self.gsd], -1)
        i = np.clip(np.round(ij[..., 0]).astype(np.int64), 0, self.z.shape[1] - 1)
        j = np.clip(np.round(ij[..., 1]).astype(np.int64), 0, self.z.shape[0] - 1)
        out = self.z[j, i]
        if np.isfinite(fill):
            out = np.where(np.isfinite(out), out, fill)
        return out


def _cam(T_WP, height=250.0):
    """Camera looking straight down from ``height`` above the plane origin."""
    t = np.asarray([0.0, 0.0, height], np.float64) @ T_WP[:3, :3].T + T_WP[:3, 3]
    return t


def test_ray_grid_shape_and_range():
    """P0-1: the grid must be (h, w) and u/v must cover the image, not 56% of it."""
    h, w, W, H = 288, 512, 1920, 1080
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * (W / w), (np.arange(h) + 0.5) * (H / h),
                         indexing="xy")
    assert uu.shape == (h, w) and vv.shape == (h, w)
    assert 0 <= uu.min() and uu.max() < W
    assert 0 <= vv.min() and vv.max() < H
    # the old (buggy) version, for the record
    bu, bv = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="xy")
    assert bu.shape == (w, h) and bu.max() * (W / w) < W


def test_raycast_hits_roof_and_ground():
    """P0-2: the nearest hit must find the box top, not the ground behind it."""
    from stream_ortho.map.build import raycast_dsm

    T_WP = np.eye(4)
    dsm = _BoxDSM()
    t_cam = _cam(T_WP, 250.0)
    K = np.array([[1220.0, 0, 960.0], [0, 1220.0, 540.0], [0, 0, 1.0]])
    H, W = 1080, 1920
    u, v = np.meshgrid(np.arange(0, W, 32) + 0.5, np.arange(0, H, 32) + 0.5, indexing="xy")
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    # a nadir camera: camera z points **down**, so the world-frame ray has z ~ -1
    R_c2w = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    dirs = rays @ R_c2w.T
    depth, hit = raycast_dsm(dsm, T_WP, t_cam, dirs)
    rays = dirs
    P = t_cam[None, None, :] + depth[..., None] * rays
    # a ray pointing at the box centre must land on the box top (z=15), not z=-2
    inside = (P[..., 0] > -100) & (P[..., 0] < 100) & (P[..., 1] > -100) & (P[..., 1] < 100)
    assert inside.sum() > 10, "test rays did not hit the box"
    assert np.allclose(P[..., 2][inside], 15.0, atol=0.5), P[..., 2][inside].mean()
    outside = (np.abs(P[..., 0]) > 170) | (np.abs(P[..., 1]) > 170)
    assert outside.sum() > 10
    assert np.allclose(P[..., 2][outside], -2.0, atol=0.5)


def test_roof_metric_identity_and_ground_baseline():
    """The metric must give ~1.0 for the truth and ~0 for a ground-only placement."""
    rng = np.random.default_rng(0)
    n = 20000
    P = np.zeros((n, 3), np.float64)
    P[:, 0] = rng.uniform(-100, 100, n)
    P[:, 1] = rng.uniform(-100, 100, n)
    roof = (np.abs(P[:, 0]) < 30) & (np.abs(P[:, 1]) < 30)
    P[:, 2] = np.where(roof, 15.0, 0.0)
    acc = roof_placement_accuracy(P, P.copy(), cell=2.0, window=100.0, h_min=3.0)
    assert acc["n_roof"] > 500
    assert acc["acc_both"] > 0.99
    ground = P.copy(); ground[:, 2] = 0.0
    bad = roof_placement_accuracy(P, ground, cell=2.0, window=100.0, h_min=3.0)
    assert bad["acc_both"] < 0.05
    assert bad["med_dz_roof"] > 10.0


def test_local_ground_is_below_roof():
    rng = np.random.default_rng(1)
    n = 8000
    xy = rng.uniform(-200, 200, (n, 2))
    roof = (np.abs(xy[:, 0]) < 40) & (np.abs(xy[:, 1]) < 40)
    z = np.where(roof, 12.0, 0.0) + rng.normal(0, 0.2, n)
    zg = local_ground(xy, z.astype(np.float32), cell=2.0, window=100.0)
    inside_roof = roof & (np.abs(xy[:, 0]) > 10) & (np.abs(xy[:, 1]) > 10)
    assert np.median(zg[inside_roof]) < 3.0
    m = roof_mask(np.c_[xy, z], window=100.0, h_min=3.0)
    assert m[inside_roof].mean() > 0.8


def test_cell_selection_keeps_best_and_preserves_coverage():
    """P1'': one observation per cell, best quality wins, no cell is lost."""
    from stream_ortho.map.selection import quality_score, select_by_cell

    T = np.eye(4)
    # two layers over the same ground, plus an isolated point
    M = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.35],
                  [1.0, 1.0, 0.0], [1.0, 1.0, 0.40],
                  [5.0, 5.0, 0.0]], np.float64)
    q = np.array([0.9, 0.2, 0.5, 0.8, 0.3], np.float32)
    sel = select_by_cell(M, T, cell=0.5, q=q)
    assert len(sel) == 3                      # one per occupied cell
    assert M[sel[0], 2] == 0.0 and M[sel[1], 2] == 0.40   # best quality kept
    # coverage: every occupied cell survives
    P = (M - T[:3, 3]) @ T[:3, :3]
    cells_all = {(int(np.floor(x / 0.5)), int(np.floor(y / 0.5))) for x, y in P[:, :2]}
    cells_sel = {(int(np.floor(x / 0.5)), int(np.floor(y / 0.5))) for x, y in P[sel, :2]}
    assert cells_all == cells_sel
    # keep_n=2 keeps both layers of the doubled cells
    sel2 = select_by_cell(M, T, cell=0.5, q=q, keep_n=2)
    assert len(sel2) == 5


def test_quality_score_is_bounded():
    from stream_ortho.map.selection import quality_score

    q = quality_score(conf=np.array([0.0, 0.5, 1.0]), opacity=np.array([1.0, 1.0, 1.0]),
                      wsum=np.array([1.0, 1.0, 1.0]), n_obs=np.array([1, 1, 3]))
    assert q.min() >= 0.0 and q.max() <= 1.0
    assert q[2] > q[1] > q[0]


def test_drift_metric_known_values():
    from stream_ortho.map.consistency import drift_metric

    before = {1: 0.0, 2: 5.0, 3: 10.0}
    after = {1: 0.0, 2: 5.5, 3: 10.0, 4: 99.0}
    d = drift_metric(before, after)
    assert d["n"] == 3                      # only cells present before *and* after
    assert abs(d["median_dz"] - 0.0) < 1e-6 or abs(d["median_dz"] - 0.0) < 0.3
    assert abs(d["frac_moved"] - 1 / 3) < 1e-6


def test_cell_surface_keys_are_absolute():
    """Keys must not depend on the current extent (that broke the drift metric)."""
    from stream_ortho.map.consistency import cell_surface

    T = np.eye(4)
    A = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 2.0]])
    B = np.array([[-50.0, -50.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 2.0]])
    ka, za, _ = cell_surface(A, T, cell=0.5)
    kb, zb, _ = cell_surface(B, T, cell=0.5)
    da, db = dict(zip(ka.tolist(), za.tolist())), dict(zip(kb.tolist(), zb.tolist()))
    assert da.keys() <= db.keys()            # the same physical cell keeps its key


def test_fit_depth_calibration_recovers_offset():
    from stream_ortho.map.consistency import fit_depth_calibration

    rng = np.random.default_rng(0)
    T = np.eye(4)
    xy = rng.uniform(-100, 100, (40000, 2))
    z = np.where(np.abs(xy[:, 0]) < 30, 8.0, 0.0)
    committed = np.c_[xy, z].astype(np.float64)
    shifted = committed.copy()
    shifted[:, 2] = shifted[:, 2] - 2.0      # frame sits 2 m low
    shifted[:, 0] += rng.normal(0, 0.05, len(shifted))
    r = fit_depth_calibration(shifted, committed, T, cell=0.5)
    assert r["n"] > 5000
    assert abs(r["b"] - 2.0) < 0.6, r
    assert r["ok"]


def test_nfp_vetoes_floaters_but_keeps_roof_and_ground():
    """NFP: ground + a 40 m roof; points 20 m up over either must be vetoed."""
    from stream_ortho.map.freespace import NFPConfig, NadirHeightField

    rng = np.random.default_rng(0)
    T = np.eye(4)
    gx, gy = np.meshgrid(np.arange(-150, 150, 1.0), np.arange(-150, 150, 1.0))
    ground = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, -2.0)]
    roof = np.c_[rng.uniform(-20, 20, (8000, 2)), np.full(8000, 15.0)]
    nfp = NadirHeightField(NFPConfig(cell=1.0, bootstrap=3), T)
    for kf in range(3):
        nfp.update(ground, kf)
        nfp.update(roof, kf)
    cov = nfp.coverage()
    assert cov["resolved"] > 0.9
    floaters = np.c_[rng.uniform(-150, 150, (20000, 2)), np.full(20000, 20.0)]
    keep, info = nfp.veto(floaters)
    assert info["vetoed"] / len(floaters) > 0.95
    # the roof is kept, except for its *border* cells where the 5x5 median smoothing
    # pulls the top down towards the neighbouring ground (measured: 99%)
    assert nfp.veto(roof)[0].mean() > 0.95
    assert nfp.veto(ground)[0].all()
    # bootstrap: before 3 frames nothing is vetoed
    nfp2 = NadirHeightField(NFPConfig(cell=1.0, bootstrap=3), T)
    nfp2.update(ground, 0)
    keep2, _ = nfp2.veto(floaters)
    assert keep2.all()


def test_cell_key_packing_handles_negative_indices():
    """The naive ix*100003+iy packing wrapped for negative iy (broke the dense grid)."""
    from stream_ortho.map.selection import pack_cell, unpack_cell

    ix = np.array([0, 0, -1, -1, 5, -300], np.int64)
    iy = np.array([0, -5, -5, 5, -7, -400], np.int64)
    k = pack_cell(ix, iy)
    assert len(np.unique(k)) == len(k)
    rx, ry = unpack_cell(k)
    assert (rx == ix).all() and (ry == iy).all()


def test_column_top_picks_top_layer_and_median_colour():
    """P0-2: ground + roof in one column -> keep the roof layer, median height/colour."""
    from stream_ortho.map.selection import select_column_top

    T = np.eye(4)
    M = np.array([[0, 0, 0.0], [0.05, 0, 0.1], [0.1, 0, 0.2],
                  [0.05, 0.05, 8.0], [0.1, 0.1, 8.2]], float)
    C = np.array([[1, 0, 0]] * 3 + [[0, 0, 1]] * 2, float)
    idx, col = select_column_top(M, C, T, cell=0.45, z_gap=2.0)
    assert len(idx) == 1
    assert M[idx[0], 2] > 7.9                       # the roof layer, not the ground
    assert abs(col[0, 2] - 1.0) < 1e-6              # median colour of that layer
    # no layers: a single observation is kept unchanged
    idx2, col2 = select_column_top(M[:1], C[:1], T, cell=0.45)
    assert len(idx2) == 1 and np.allclose(col2[0], C[0])


def test_avg_color_is_really_applied():
    """P0-3: collapse_cells(avg_color=True) must change the survivors' colours."""
    import numpy as np

    from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
    from stream_ortho.map.gaussian import GaussianPatch

    gm = GaussianMap(MapConfig(voxel_size=0.25, dedup=False, gate=False,
                               commit_min_obs=1, commit_max_age=1), np.eye(4))
    # collapse_cells deliberately no-ops below 1000 live points, so build 300 copies
    base_m = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.10, 0, 0], [5.0, 5, 0]], np.float32)
    base_c = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], np.float32)
    means = np.tile(base_m, (300, 1))
    cols = np.tile(base_c, (300, 1))
    n = len(means)
    patch = GaussianPatch(means=means, quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                          scales=np.full((n, 3), 0.2, np.float32),
                          opacity=np.ones(n, np.float32), colors=cols,
                          owner=0, conf=np.ones(n, np.float32), weight=np.ones(n, np.float32))
    gm.ingest(patch, 0)
    gm.commit(0)

    class _G:
        tile = 1
        gsd = 1.0
    gm.collapse_cells(_G(), cell=0.45, prefer="quality", avg_color=True)
    live = gm.committed["dead"] == 0
    c = gm.committed["colors"][live]
    assert len(c) == 2
    cell0 = c[np.abs(gm.committed["means"][live][:, 0]) < 0.2]
    assert cell0.shape[0] == 1
    # mean of 0, 1 and 0.5 -> 0.5, not the 0/1 of a single sample
    assert abs(float(cell0[0, 0]) - 0.5) < 1e-5, cell0


def test_column_top_colours_follow_the_points():
    """Regression: colours must stay aligned with the returned indices.

    The first version sorted the indices but left the colours in cell order, which made
    the whole ortho change colour at every collapse ("the map suddenly turns to a smear").
    """
    from stream_ortho.map.selection import select_column_top

    T = np.eye(4)
    # 3 cells along x, each with a red point on top and a blue point below
    M, C = [], []
    for i in range(3):
        M += [[i * 0.5, 0.0, 10.0], [i * 0.5, 0.0, 0.0]]
        C += [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    M, C = np.array(M, float), np.array(C, float)
    idx, col = select_column_top(M, C, T, cell=0.45, z_gap=2.0)
    assert len(idx) == 3
    # every kept point must carry the colour of ITS OWN cell (red), not a neighbour's
    for j, i in enumerate(idx):
        assert M[i, 2] > 9.0
        assert np.allclose(col[j], C[i]), (j, i, col[j], C[i])


def test_column_filter_drops_facade_points():
    """P1-4: points metres below their column's top surface are dropped."""
    from stream_ortho.map.selection import column_top_filter

    T = np.eye(4)
    # one column: roof at z=12, wall points at 6 and 2 (facade), ground at 0 nearby
    M = np.array([[0, 0, 12.0], [0, 0, 6.0], [0, 0, 2.0], [50, 50, 0.0]], float)
    keep = column_top_filter(M, T, cell=0.45, h_tol=2.0)
    assert keep[0] and not keep[1] and not keep[2]   # roof kept, facade dropped
    assert keep[3]                                   # unrelated ground point kept
