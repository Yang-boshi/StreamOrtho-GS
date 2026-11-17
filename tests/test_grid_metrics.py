"""P0-1 frozen metric tests (synthetic, deterministic).

The nine tests below are the acceptance set of ``02-P0-1-指标冻结.md`` §3.  They are
written against *synthetic* grids only: no data, no GPU, no annotation files.  Each test
also encodes a **counter-example**: the bug the metric must not have (truncation, sorting
based pairing, per-pair local origin, penalising a true roof edge, ...).

Conventions shared by every test (see ``refs/06-任务卡-P0.md`` §2.0):

* ``store`` is any object exposing ``cell/ox/oy/W/H/L/z_ref/n/sum_w/sum_wz/sum_wz2/rgb``
  -- the real ``map.columns.ColumnStore`` or the lightweight ``Store`` built below.
* ``sel`` is ``{"z","rgb","n","live","label"}`` with ``(H,W)`` arrays.
* empty layers are ``n == 0`` and their ``z_ref`` is meaningless (never matched).
"""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.eval.grid_metrics import (
    double_edge_rate,
    edge_width,
    eval_cells_from_raw,
    flat_color_discontinuity,
    fit_pair_t3,
    floater_rate,
    layer_consistency,
    pairwise_consistency,
    ripple_amplitude,
)
from stream_ortho.eval.hfr import hfr, surface_guards


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
class Store:
    """Minimal stand-in for ``map.columns.ColumnStore`` (only the fields metrics read)."""

    def __init__(self, z, n=None, sigma=None, rgb=None, cell=0.3, ox=0, oy=0):
        z = np.asarray(z, np.float64)
        assert z.ndim == 3, "z must be (H, W, L)"
        H, W, L = z.shape
        self.cell, self.ox, self.oy, self.W, self.H, self.L = cell, ox, oy, W, H, L
        self.n = np.zeros((H, W, L), np.uint16) if n is None else np.asarray(n, np.uint16)
        occ = self.n > 0
        self.z_ref = np.where(occ, np.nan_to_num(z), 0.0).astype(np.float64)
        # weighted statistics consistent with z_ref / n
        w = np.maximum(self.n, 1).astype(np.float64)
        s = np.zeros((H, W, L), np.float64) if sigma is None else np.asarray(sigma, np.float64)
        self.sum_w = np.where(occ, w, 0.0)
        self.sum_wz = self.z_ref * self.sum_w
        self.sum_wz2 = (s ** 2 + self.z_ref ** 2) * self.sum_w
        self.rgb = np.zeros((H, W, L, 3), np.float64) if rgb is None else np.asarray(rgb, np.float64)

    @property
    def shape(self):
        return self.H, self.W, self.L


def make_sel(store, z, rgb=None, n=None, live=None, label=None):
    H, W = store.H, store.W
    live = np.ones((H, W), bool) if live is None else np.asarray(live, bool)
    return {
        "z": np.asarray(z, np.float64),
        "rgb": np.zeros((H, W, 3), np.float64) if rgb is None else np.asarray(rgb, np.float64),
        "n": np.zeros((H, W)) if n is None else np.asarray(n, np.float64),
        "live": live,
        "label": np.zeros((H, W), np.int16) if label is None else np.asarray(label, np.int16),
    }


def product_1x1(z0=0.0, n0=5):
    return {"n": np.array([[[n0]]], np.uint16), "z_ref": np.array([[[z0]]], np.float64),
            "x0": 0.0, "y0": 0.0, "cell": 0.3, "W": 1, "H": 1, "L": 1}


def held_rows(zs, kf=0, x=0.15, y=0.15):
    zs = np.asarray(zs, np.float64)
    return [{"kf": kf, "x": np.full(zs.size, x), "y": np.full(zs.size, y), "z": zs}]


# --------------------------------------------------------------------------------------
# 1. HFR must not truncate |dz| > 3
# --------------------------------------------------------------------------------------
def test_hfr_no_drop():
    prod = product_1x1(0.0)
    raw = hfr(held_rows([0.1, 0.1, 0.1, 10.0]), prod)
    fixed = hfr(held_rows([0.2, 0.2, 0.2, 2.9]), prod)

    # the median is *not* dragged to the tail, but the tail itself stays in the numbers
    assert raw["near_pooled"] == pytest.approx(0.1, abs=1e-9)
    assert fixed["near_pooled"] == pytest.approx(0.2, abs=1e-9)

    # 10.0 is still there: p90 far above the 3 m tail threshold, and tail_frac == 0.25
    assert raw["near_p90"] > 3.0
    assert raw["tail_frac"] == pytest.approx(0.25)
    # the "repaired" run moved the outlier to 2.9 => no tail left, p90 <= 3
    assert fixed["near_p90"] <= 3.0
    assert fixed["tail_frac"] == pytest.approx(0.0)

    # counter-example: truncating at 3 m would report median 0.1 with tail_frac 0
    truncated = hfr(held_rows([0.1, 0.1, 0.1, 2.999]), prod)
    assert truncated["tail_frac"] == pytest.approx(0.0)
    assert raw["tail_frac"] != truncated["tail_frac"]


def test_hfr_top_vs_near_and_outside():
    """(not counted) top/near differ on a two-layer product; outside/empty are reported."""
    prod = {"n": np.array([[[5, 5]]], np.uint16),                # (H=1, W=1, L=2)
            "z_ref": np.array([[[0.0, 10.0]]], np.float64),
            "x0": 0.0, "y0": 0.0, "cell": 0.3, "W": 1, "H": 1, "L": 2}
    r = hfr(held_rows([0.2]), prod)
    assert r["top_pooled"] == pytest.approx(9.8)      # top layer = 10
    assert r["near_pooled"] == pytest.approx(0.2)     # nearest layer = 0
    outside = hfr([{"kf": 0, "x": np.array([99.0]), "y": np.array([0.0]),
                    "z": np.array([0.0])}], prod)
    assert outside["outside_frac"] == pytest.approx(1.0)
    assert outside["n_obs"] == 0
    assert np.isnan(outside["near_pooled"])


# --------------------------------------------------------------------------------------
# 2. PC must pair by cell key (never by rank)
# --------------------------------------------------------------------------------------
def _frame(kf, keys, z, conf=1.0):
    z = np.asarray(z, np.float64)
    return {"kf": kf, "key": np.asarray(keys, np.int64), "z": z,
            "conf": np.full(z.size, conf, np.float32),
            "cos_inc": np.ones(z.size, np.float32),
            "x": (np.asarray(keys) + 0.5) * 0.3, "y": np.zeros(z.size)}


def test_pc_key_match():
    keys = np.arange(100)
    # the two frames disagree at EVERY cell (roof/ground swap), by 2 m -- under the 3 m
    # pair-quality threshold, so the pair is "ok" and must not be discarded
    za = np.where(keys % 2 == 0, 0.0, 2.0)
    zb = np.where(keys % 2 == 0, 2.0, 0.0)
    a, b = _frame(0, keys, za), _frame(1, keys, zb)
    r = pairwise_consistency([a, b], eval_cells=keys, min_shared=10)
    assert r["pc"] == pytest.approx(2.0, abs=1e-9)

    # counter-example: sorting each frame's z before subtracting pairs the *ranks*, not the
    # cells.  Both frames hold the same value multiset, so rank pairing reports a perfect
    # 0.0 for two frames that disagree everywhere -- the week9-17 bug.
    sorted_pc = float(np.median(np.abs(np.sort(a["z"]) - np.sort(b["z"]))))
    assert sorted_pc == pytest.approx(0.0)
    assert r["pc"] > sorted_pc

    # a cell that is two-layered in either frame is excluded (z is ambiguous there)
    b2 = _frame(1, np.r_[keys, 100], np.r_[np.full(100, 0.5), 5.0])
    b2["key"][-1] = 50                       # duplicate key 50 -> not single-layer any more
    r2 = pairwise_consistency([a, b2], eval_cells=keys, min_shared=10)
    assert r2["pairs"][0]["n"] == 99


# --------------------------------------------------------------------------------------
# 3. every pair fit must use ONE global origin
# --------------------------------------------------------------------------------------
def _synthetic_pair(theta, x, y, origin, h=0.0):
    u, v = x - origin[0], y - origin[1]
    return h + theta[0] + theta[1] * u + theta[2] * v


def test_pc_global_origin():
    rng = np.random.default_rng(0)
    x = rng.uniform(-40, 40, 4000)
    y = rng.uniform(-40, 40, 4000)
    origin = (0.0, 0.0)
    th = [np.array([0.50, 0.010, -0.020]),
          np.array([0.20, -0.005, 0.010]),
          np.array([-0.10, 0.002, 0.003])]
    z = [_synthetic_pair(t, x, y, origin) for t in th]

    d01 = fit_pair_t3(z[0], z[1], x, y, origin)["p"]
    d12 = fit_pair_t3(z[1], z[2], x, y, origin)["p"]
    assert np.allclose(d01, th[0] - th[1], atol=1e-6)
    assert np.allclose(d12, th[1] - th[2], atol=1e-6)

    # composing the two pairwise estimates must reproduce theta_0 - theta_2 exactly
    composed = d01 + d12
    assert np.allclose(composed, th[0] - th[2], atol=1e-6)

    # counter-example: centring each pair at its own mean makes b absorb the tilt, so the
    # two estimates can no longer be composed
    local01 = fit_pair_t3(z[0], z[1], x, y, (x.mean(), y.mean()))["p"]
    local12 = fit_pair_t3(z[1], z[2], x, y, (x.mean(), y.mean()))["p"]
    err_global = np.abs(composed - (th[0] - th[2])).max()
    err_local = np.abs(local01 + local12 - (th[0] - th[2])).max()
    assert err_global < 1e-6
    assert err_local > 1e-3

    # counter-example 2: feeding a fabricated coordinate pair (the week9-14/17 ``linspace``
    # bug, where x and y were the same vector) makes the 3-DOF design rank-deficient, so the
    # fit degenerates even though the arithmetic "succeeds"
    A_bad = np.stack([np.ones_like(x), x - origin[0], x - origin[0]], 1)
    assert np.linalg.matrix_rank(A_bad) < 3
    p_bad = np.linalg.lstsq(A_bad, z[0] - z[1], rcond=None)[0]
    assert not np.allclose(p_bad, th[0] - th[1], atol=1e-3)


# --------------------------------------------------------------------------------------
# 4/5. LCR: a true roof/facade edge is not a double wall; a layer flip is
# --------------------------------------------------------------------------------------
def _edge_grid():
    """7 cells, 1 row: ground x3 | facade | roof x3  (layers are z-valued slots)."""
    z = np.full((1, 7, 3), np.nan)
    n = np.zeros((1, 7, 3), np.uint16)
    for i in range(7):
        if i < 3:                            # ground: one layer at 0
            z[0, i, 0], n[0, i, 0] = 0.0, 2
        elif i == 3:                         # facade: 0 and 10
            z[0, i, 0], n[0, i, 0] = 0.0, 2
            z[0, i, 1], n[0, i, 1] = 10.0, 2
        else:                                # roof: one layer at 10
            z[0, i, 0], n[0, i, 0] = 10.0, 2
    sel_z = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]])
    return Store(z, n), make_sel(Store(z, n), sel_z)


def test_lcr_real_edge_not_penalized():
    store, sel = _edge_grid()
    r = layer_consistency(store, sel, z_gap=2.0, h_tol=1.0)
    # ground<->facade shares only layer 0; facade<->roof shares only layer 10
    assert r["n_shared_pairs"] == 0
    assert r["n_inconsistent"] == 0
    assert r["lcr"] == pytest.approx(1.0)


def test_lcr_double_wall_penalized():
    z = np.full((1, 2, 3), np.nan)
    n = np.zeros((1, 2, 3), np.uint16)
    for i in range(2):                       # both cells see ground, mid and roof
        for k, v in enumerate([0.0, 10.0, 20.0]):
            z[0, i, k], n[0, i, k] = v, 2
    # the selector picks 10 on the left and 20 on the right -> a 10 m seam inside a
    # surface that both cells observe twice => double wall
    sel_z = np.array([[10.0, 20.0]])
    store = Store(z, n)
    r = layer_consistency(store, make_sel(store, sel_z), z_gap=2.0, h_tol=1.0)
    assert r["n_shared_pairs"] == 1
    assert r["n_inconsistent"] == 1
    assert r["lcr"] == pytest.approx(0.0)

    # choosing the same common layer removes the penalty
    r2 = layer_consistency(store, make_sel(store, np.array([[10.0, 10.0]])), z_gap=2.0, h_tol=1.0)
    assert r2["lcr"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# 6. FCD only looks at flat, same-surface neighbours
# --------------------------------------------------------------------------------------
def test_fcd_flat_same_surface():
    z = np.zeros((1, 8, 1))
    n = np.full((1, 8, 1), 2, np.uint16)
    rgb = np.zeros((1, 8, 1, 3))
    rgb[:] = 100.0
    store = Store(z, n, rgb=rgb)
    sel = make_sel(store, z[..., 0], rgb=rgb[:, :, 0, :])
    r = flat_color_discontinuity(store, sel)
    assert r["fcd"] == pytest.approx(0.0, abs=1e-9)

    rgb2 = rgb.copy()
    rgb2[0, 0, 0] = [200.0, 50.0, 50.0]
    store2 = Store(z, n, rgb=rgb2)
    sel2 = make_sel(store2, z[..., 0], rgb=rgb2[:, :, 0, :])
    r2 = flat_color_discontinuity(store2, sel2)
    assert r2["fcd"] > 0.0
    assert r2["n_pairs"] == 7

    # a genuine edge (|dz| > h_tol) is excluded, so recolouring across it is invisible
    z3 = np.zeros((1, 8, 1)); z3[0, 4:, 0] = 10.0
    n3 = np.full((1, 8, 1), 2, np.uint16)
    rgb3 = np.zeros((1, 8, 1, 3)); rgb3[:] = 100.0; rgb3[0, 4:, 0] = [200.0, 50.0, 50.0]
    store3 = Store(z3, n3, rgb=rgb3)
    sel3 = make_sel(store3, z3[..., 0], rgb=rgb3[:, :, 0, :])
    r3 = flat_color_discontinuity(store3, sel3)
    assert r3["fcd"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# 7. DER_h: two opposing steps that return to the start height
# --------------------------------------------------------------------------------------
def _profile_sel(zs):
    zs = np.asarray(zs, np.float64)
    store = Store(zs[None, :, None], np.full((1, zs.size, 1), 2, np.uint16))
    return make_sel(store, zs[None, :])


def test_der_h_double_edge():
    dbl = double_edge_rate(_profile_sel([0, 0, 10, 10, 0, 0]), h_tol=1.0, d_max=2)
    assert dbl["n_steps"] == 2
    assert dbl["n_double"] == 1
    assert dbl["der_h"] == pytest.approx(0.5)

    # a real step never comes back down -> no double edge
    single = double_edge_rate(_profile_sel([0, 0, 10, 10, 10, 10]), h_tol=1.0, d_max=2)
    assert single["n_steps"] == 1
    assert single["der_h"] == pytest.approx(0.0)

    # flat ground has no steps at all (and must not divide by zero)
    flat = double_edge_rate(_profile_sel([0, 0, 0, 0, 0, 0]), h_tol=1.0, d_max=2)
    assert flat["n_steps"] == 0
    assert flat["der_h"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# 8. RA: analytic amplitude response of the high-pass
# --------------------------------------------------------------------------------------
def test_ra_ripple():
    H, W = 64, 96
    xx = np.arange(W)[None, :].repeat(H, 0)
    lam, amp, sigma = 12.0, 1.0, 3.0
    z = amp * np.sin(2 * np.pi * xx / lam)
    store = Store(z[..., None], np.full((H, W, 1), 2, np.uint16))
    sel = make_sel(store, z)
    r = ripple_amplitude(store, sel, sigma_cells=sigma)

    # gaussian_filter amplitude response exp(-2 pi^2 sigma^2 / lam^2); the high-pass keeps
    # 1 - that, and the RMS of a sine of amplitude A is A / sqrt(2)
    resp = np.exp(-2 * np.pi ** 2 * sigma ** 2 / lam ** 2)
    expect = amp * (1.0 - resp) / np.sqrt(2)
    assert r["ra"] == pytest.approx(expect, abs=0.05)

    flat_store = Store(np.zeros((H, W, 1)), np.full((H, W, 1), 2, np.uint16))
    r_flat = ripple_amplitude(flat_store, make_sel(flat_store, np.zeros((H, W))), sigma_cells=sigma)
    assert r_flat["ra"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# 9. anti-cheat guards must reject a flattened product
# --------------------------------------------------------------------------------------
def test_guards_flatten():
    raw = np.array([[[0.0], [2.0], [4.0], [6.0]]], np.float64)
    ones = np.ones_like(raw, np.uint16)
    prod_raw = {"z_ref": raw, "n": ones}
    same = surface_guards({"z_ref": raw.copy(), "n": ones}, prod_raw)
    assert same["z_iqr_ratio"] == pytest.approx(1.0)
    assert same["median_z_shift"] == pytest.approx(0.0)
    assert same["pass"] is True

    flattened = np.full_like(raw, np.median(raw) + 1.0)     # crush the relief and shift it
    bad = surface_guards({"z_ref": flattened, "n": ones}, prod_raw)
    assert bad["z_iqr_ratio"] < 0.95
    assert bad["median_z_shift"] > 0.5
    assert bad["pass"] is False


# --------------------------------------------------------------------------------------
# helpers used by later tasks (not counted in the nine)
# --------------------------------------------------------------------------------------
def test_eval_cells_and_floater_rate():
    z = np.zeros((5, 5, 2))
    n = np.zeros((5, 5, 2), np.uint16)
    n[..., 0] = 3                                     # single-layer flat everywhere
    z[2, 2, 0] = 10.0
    n[2, 2, 0] = 1                                    # one isolated, unsupported floater
    store = Store(z, n)
    cells = eval_cells_from_raw(store)
    assert cells.size == 24                            # the under-supported cell is excluded

    sel = make_sel(store, z[..., 0], n=n[..., 0].astype(float))
    fr = floater_rate(store, sel, h_tol=1.5)
    assert fr["n_live"] == 25
    assert fr["fr"] > 0.0

    ew = edge_width(store, make_sel(store, np.zeros((5, 5))), cell=0.3)
    assert np.isnan(ew["ew_m"]) or ew["ew_m"] >= 0.0
