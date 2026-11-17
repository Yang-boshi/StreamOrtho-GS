"""Tests for ``map/calib.py`` -- the PIVOT-B calibration core.

Three things must be true before anything else is believed:

* ``fit_t3`` recovers a known per-frame 3-DOF deformation and ignores 10% gross outliers;
* ``apply_t3`` subtracts exactly ``b + tx u + ty v`` and leaves other fields untouched;
* ``FrozenProduct.rebuild`` updates only the heights: the layer slots, the owner and the
  support stay exactly as they were (that is what removes the LCR regression).
"""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.map.calib import CalibConfig, FrozenProduct, apply_t3, fit_t3


def test_fit_t3_recovers_theta_and_survives_outliers():
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.uniform(-300, 300, n)
    y = rng.uniform(-300, 300, n)
    theta = np.array([0.8, 0.012, -0.007])
    z_near = rng.normal(0, 5, n)
    z = z_near + theta[0] + theta[1] * x + theta[2] * y

    cfg = CalibConfig()
    p = fit_t3(z, z_near, x, y, np.ones(n), cfg)
    assert np.allclose(p, theta, atol=1e-8)

    z2 = z.copy()
    idx = rng.choice(n, n // 10, replace=False)
    z2[idx] += 50.0                                    # 10% gross outliers
    p2 = fit_t3(z2, z_near, x, y, np.ones(n), cfg)
    assert np.abs(p2 - theta).max() < 0.01


def test_apply_t3_is_exact_and_keeps_other_fields():
    x = np.array([-10.0, 0.0, 25.0])
    y = np.array([5.0, -5.0, 0.0])
    z = np.array([1.0, 2.0, 3.0])
    theta = np.array([0.5, 0.01, -0.02])
    origin = (3.0, 4.0)
    zc = apply_t3(z, x, y, theta, origin)
    expect = z - (theta[0] + theta[1] * (x - origin[0]) + theta[2] * (y - origin[1]))
    assert np.allclose(zc, expect)
    assert np.allclose(z, [1.0, 2.0, 3.0])             # input untouched
    assert apply_t3(z, x, y, np.zeros(3), origin).tolist() == z.tolist()


def _toy_product():
    """3 cells x 2 layers; cell 0 has two observations in slot 0."""
    n = np.zeros((1, 3, 2), np.float64)
    z = np.zeros((1, 3, 2), np.float64)
    owner = np.full((1, 3, 2), -1, np.int64)
    # rows: (cell, slot, z, w)
    rows = [(0, 0, 1.0, 2.0), (0, 0, 3.0, 1.0), (1, 0, 5.0, 1.0), (2, 1, 7.0, 1.0)]
    assign = np.array([0 * 2 + 0, 0 * 2 + 0, 1 * 2 + 0, 2 * 2 + 1], np.int64)
    for c, s, zz, ww in rows:
        n[0, c, s] += 1
        z[0, c, s] += ww * zz
    sw = np.zeros_like(n)
    for c, s, zz, ww in rows:
        sw[0, c, s] += ww
    z = np.where(sw > 0, z / np.maximum(sw, 1e-12), 0.0)
    for c, s, _, _ in rows:
        owner[0, c, s] = 7 if c < 2 else 9
    prod = {"n": n, "z_ref": z, "owner": owner, "sum_w": sw,
            "sum_wz": z * sw, "sum_wz2": (z ** 2) * sw,
            "cell": 0.45, "W": 3, "H": 1, "L": 2, "ox": 0, "oy": 0, "x0": 0.0, "y0": 0.0,
            "assign": assign}
    return prod, np.array([1.0, 3.0, 5.0, 7.0]), np.array([2.0, 1.0, 1.0, 1.0])


def test_frozen_rebuild_updates_only_heights():
    prod, z_rows, w_rows = _toy_product()
    fp = FrozenProduct.from_product(prod)
    assert fp.owner.tolist() == prod["owner"].tolist()
    n_before = prod["n"].copy()
    owner_before = prod["owner"].copy()

    z_new = z_rows + 10.0                              # shift every observation
    out = fp.rebuild(z_new, w_rows)
    # owner and support are untouched
    assert np.array_equal(out["owner"], owner_before)
    assert np.array_equal(out["n"], n_before)
    # heights moved by exactly +10 where occupied
    occ = n_before > 0
    assert np.allclose(out["z_ref"][occ], prod["z_ref"][occ] + 10.0)
    # and an untouched cell stays at zero
    assert out["z_ref"][0, 0, 1] == 0.0


def test_frozen_rebuild_keeps_the_weighted_mean():
    prod, z_rows, w_rows = _toy_product()
    fp = FrozenProduct.from_product(prod)
    z_new = np.array([0.0, 4.0, 5.0, 7.0])             # cell0 slot0: weights 2 and 1
    out = fp.rebuild(z_new, w_rows)
    assert out["z_ref"][0, 0, 0] == pytest.approx((2 * 0 + 1 * 4) / 3)


def test_solver_precision_at_300m():
    """At ±300 m the normal equations lose the intercept; the QR path must not.

    This is the reason ``_solve_ls`` uses ``lstsq`` and why ``CalibConfig.ridge`` defaults to
    zero: the documented 1.6e-5 bias comes from the ridge acting on a badly conditioned
    ``A^T A``, not from the fit itself.
    """
    from stream_ortho.map.calib import _solve_ls

    rng = np.random.default_rng(11)
    n = 60_000
    x = rng.uniform(-300, 300, n)
    y = rng.uniform(-300, 300, n)
    theta = np.array([0.8, 0.012, -0.007])
    d = theta[0] + theta[1] * x + theta[2] * y
    A = np.stack([np.ones(n), x, y], 1)

    p_qr = _solve_ls(A, d)
    assert np.abs(p_qr - theta).max() < 1e-6

    # the normal-equation path with a 1e-9 ridge is measurably worse on the intercept
    ATA = A.T @ A
    scale = float(np.mean(np.diag(ATA)))
    p_ne = np.linalg.solve(ATA + 1e-9 * scale * np.eye(3), A.T @ d)
    assert abs(p_ne[0] - theta[0]) > 1e-6


def test_hybrid_objective_prior_weight():
    """``lam_p`` is the *coefficient* of the prior term (rows scale by sqrt), and the Huber
    scale can make a weak prior inert -- both behaviours are pinned here."""
    from stream_ortho.scripts.diag.e6_neighbor_fit import fit_objective

    rng = np.random.default_rng(7)
    n = 20_000
    x = rng.uniform(-200, 200, n)
    y = rng.uniform(-200, 200, n)
    A = np.stack([np.ones(n), x, y], 1)
    theta_true = np.array([0.4, 0.01, -0.02])
    z = A @ theta_true
    z_ref = np.zeros(n)
    # a *spread* prior: a constant offset would make the Huber scale collapse onto the prior
    # block and silently discard the consensus rows
    z_prior = z + 3.0 * (x / 200.0)
    w = np.ones(n)
    cfg = CalibConfig()

    pure = fit_objective(z, z_ref, z_prior, A, w, 0.0, cfg)
    assert np.allclose(pure, theta_true, atol=1e-8)          # lam_p = 0 ignores the prior

    half = fit_objective(z, z_ref, z_prior, A, w, 1.0, cfg)  # equal weights -> midpoint
    prior_only = fit_objective(z, z_ref, z_prior, A, w, 1e6, cfg)
    prior_fit = np.linalg.lstsq(A, z - z_prior, rcond=None)[0]
    assert np.allclose(half, 0.5 * (theta_true + prior_fit), atol=1e-6)
    assert np.allclose(prior_only, prior_fit, atol=1e-5)
