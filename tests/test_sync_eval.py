"""Tests for the E1 machinery (``eval/sync_eval.py``) -- synthetic, no data, no GPU.

The five assertions of ``04-P0-3`` §6:

* ``fit_pair`` recovers a known 3-DOF relative deformation (and survives 10% outliers);
* ``solve_sync`` pins the gauge and reproduces relative offsets on a chain;
* every pair uses **one global origin** (per-pair centring breaks composability);
* applying the synchronised theta reduces pairwise inconsistency;
* no fabricated coordinates: ``linspace`` may not be used to build x/y.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from stream_ortho.eval.sync_eval import SyncConfig, fit_pair, solve_sync


def _frame(k, keys, theta, origin=(0.0, 0.0), noise=0.0, seed=0):
    rng = np.random.default_rng(seed + k)
    keys = np.asarray(keys, np.int64)
    x = (keys % 500 - 250) * 0.45
    y = (keys // 500 % 500 - 250) * 0.45
    z = 3.0 + theta[0] + theta[1] * (x - origin[0]) + theta[2] * (y - origin[1])
    z = z + rng.normal(0, noise, z.size)
    o = np.argsort(keys, kind="stable")
    return {"rows": np.arange(keys.size), "key": keys[o], "z": z[o], "x": x[o], "y": y[o],
            "conf": np.ones(keys.size), "cos": np.ones(keys.size),
            "single": np.unique(keys)}


def test_fit_pair_recovers_theta():
    keys = np.arange(20_000)
    th0, th1 = np.array([0.5, 0.01, -0.02]), np.zeros(3)
    fi, fj = _frame(0, keys, th0), _frame(1, keys, th1)
    cfg = SyncConfig(min_shared=500)
    e = fit_pair(fi, fj, None, (0.0, 0.0), cfg)
    assert e["ok"] and e["n"] > 10_000
    assert np.allclose(e["p"], th0 - th1, atol=1e-6)

    # 10% gross outliers must not move the estimate by more than 1 cm
    fi2 = dict(fi)
    z = fi["z"].copy()
    idx = np.random.default_rng(7).choice(z.size, z.size // 10, replace=False)
    z[idx] += 50.0
    fi2["z"] = z
    e2 = fit_pair(fi2, fj, None, (0.0, 0.0), cfg)
    assert np.abs(e2["p"] - (th0 - th1)).max() < 0.01


def test_solve_sync_gauge():
    keys = np.arange(5_000)
    # small tilts: the pair-quality gate rejects a pair whose |dz| exceeds 3 m over half of
    # its cells, and that gate is part of the method (it is what flags roof/ground mismatch)
    th = {0: np.zeros(3), 1: np.array([0.4, 0.003, -0.002]), 2: np.array([-0.3, -0.004, 0.003])}
    frames = {k: _frame(k, keys, th[k]) for k in th}
    cfg = SyncConfig(min_shared=500, ridge=1e-9)     # no conditioning prior on clean data
    edges = [dict(fit_pair(frames[i], frames[j], None, (0.0, 0.0), cfg), i=i, j=j)
             for i in range(3) for j in range(i + 1, 3)]
    theta, n_ok, _ = solve_sync(edges, [0, 1, 2], cfg)
    assert n_ok == 3
    assert np.allclose(theta[0], 0.0, atol=1e-3)              # gauge frame
    for i in range(3):
        for j in range(3):
            assert np.allclose(theta[i] - theta[j], th[i] - th[j], atol=1e-2)


def test_global_origin():
    keys = np.arange(20_000)
    th = {0: np.array([0.5, 0.01, -0.02]), 1: np.array([0.2, -0.005, 0.01])}
    cfg = SyncConfig(min_shared=500, ridge=1e-9)
    o = (12.5, -30.0)
    # data expressed in the SAME origin used for the fit -> theta is recovered exactly
    fi, fj = _frame(0, keys, th[0], origin=o), _frame(1, keys, th[1], origin=o)
    p_global = fit_pair(fi, fj, None, o, cfg)["p"]
    assert np.allclose(p_global, th[0] - th[1], atol=1e-6)

    # reverse test: fitting the same pair with a *different* origin silently turns the
    # offset into a combination of the tilt (b' = b + tx*dx + ty*dy) -- which is exactly why
    # every pair of an experiment must share one (x0, y0)
    p_local = fit_pair(fi, fj, None, (0.0, 0.0), cfg)["p"]
    assert not np.allclose(p_local, p_global)
    assert np.allclose(p_local[1:], p_global[1:], atol=1e-9)   # only the constant moves


def test_apply_sync_reduces_pc():
    """Removing the true per-frame theta must drop the pairwise |dz| by >= 50%."""
    keys = np.arange(20_000)
    rng = np.random.default_rng(3)
    th = {0: np.array([1.5, 0.02, -0.01]), 1: np.array([-1.2, -0.015, 0.02]),
          2: np.array([0.4, 0.005, 0.01])}
    base = rng.normal(0, 3.0, keys.size)
    frames = {}
    for k, t in th.items():
        f = _frame(k, keys, t)
        f["z"] = base[np.argsort(keys, kind="stable")] + t[0] + t[1] * (f["x"]) + t[2] * (f["y"])
        frames[k] = f
    i0 = np.searchsorted(frames[0]["key"], keys)
    i1 = np.searchsorted(frames[1]["key"], keys)

    def pc(a, b):
        return float(np.median(np.abs(a - b)))

    before = pc(frames[0]["z"][i0], frames[1]["z"][i1])
    corr = lambda f, t: f["z"] - (t[0] + t[1] * f["x"] + t[2] * f["y"])
    after = pc(corr(frames[0], th[0]), corr(frames[1], th[1]))
    assert after < 0.5 * before


def test_no_linspace():
    """The module must not fabricate x/y with ``linspace`` (the week9-14/17 bug)."""
    src = pathlib.Path(__file__).resolve().parents[1] / "stream_ortho" / "eval" / "sync_eval.py"
    text = src.read_text()
    for line in text.splitlines():
        if "linspace" in line:
            raise AssertionError(f"linspace used in sync_eval.py: {line.strip()}")




def _weighted_residual(edges, theta):
    """sum_e ||A_e (theta_i - theta_j - p_e)||^2 with A^T A = inv(cov)."""
    tot = 0.0
    for e in edges:
        A = np.linalg.cholesky(np.linalg.inv(e["cov"] + 1e-6 * np.eye(3))).T
        r = A @ (theta[e["i"]] - theta[e["j"]] - e["p"])
        tot += float(r @ r)
    return tot


def test_solve_sync_whitens_all_three_components():
    """The whitening must couple b, tx and ty, not just the diagonal.

    Reference: the same normal equations assembled densely with the full 3x3 ``A`` blocks.
    The buggy diagonal-only solver lands elsewhere; the sparse solver must match the dense
    reference to machine precision.
    """
    p01 = np.array([1.00, 0.010, -0.004])
    p12 = np.array([0.60, -0.006, 0.002])
    p02 = np.array([2.00, 0.050, 0.020])          # far from p01 + p12: strong inconsistency
    cov = np.array([[1.0, 0.99, 0.98],
                    [0.99, 1.0, 0.99],
                    [0.98, 0.99, 1.0]]) * 0.01    # strongly correlated -> off-diagonals matter
    edges = [{"i": 0, "j": 1, "p": p01, "cov": cov, "ok": True},
             {"i": 1, "j": 2, "p": p12, "cov": cov, "ok": True},
             {"i": 0, "j": 2, "p": p02, "cov": cov, "ok": True}]

    theta, n_ok, _ = solve_sync(edges, [0, 1, 2], SyncConfig(ridge=1e-12))
    ours = _weighted_residual(edges, theta)
    assert n_ok == 3

    # dense reference with full 3x3 blocks + the same gauge rows
    blocks, rhs = [], []
    for e in edges:
        A = np.linalg.cholesky(np.linalg.inv(e["cov"] + 1e-6 * np.eye(3))).T
        B = np.zeros((3, 9))
        B[:, 3 * e["i"]:3 * e["i"] + 3] += A
        B[:, 3 * e["j"]:3 * e["j"] + 3] -= A
        blocks.append(B); rhs.append(A @ e["p"])
    G = np.zeros((3, 9)); G[0, 0] = G[1, 1] = G[2, 2] = 1e3
    M = np.vstack(blocks + [G])
    y = np.concatenate(rhs + [np.zeros(3)])
    ref = np.linalg.lstsq(M, y, rcond=None)[0].reshape(3, 3)
    assert np.allclose(theta, ref, atol=1e-8), "sparse solver does not match the dense reference"

    # the diagonal-only variant is measurably worse on the same objective
    from scipy.sparse import coo_matrix
    rows, cols, data, rhsb = [], [], [], []
    r = 0
    for e in edges:
        A = np.linalg.cholesky(np.linalg.inv(e["cov"] + 1e-6 * np.eye(3))).T
        for k in range(3):
            rows += [r, r]; cols += [3 * e["i"] + k, 3 * e["j"] + k]
            data += [A[k, k], -A[k, k]]
            rhsb.append(A[k] @ e["p"]); r += 1
    for k in range(3):
        rows.append(r); cols.append(k); data.append(1e3); rhsb.append(0.0); r += 1
    Mb = coo_matrix((data, (rows, cols)), shape=(r, 9)).tocsr()
    bad = np.linalg.lstsq((Mb.T @ Mb).toarray(), Mb.T @ np.asarray(rhsb), rcond=None)[0].reshape(3, 3)
    assert ours <= _weighted_residual(edges, bad) * 0.999
