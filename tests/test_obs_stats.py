"""Tests for the observation-table statistics (``eval/obs_stats.py``).

Synthetic tables only: the point is to pin the *definitions* of ``frames_per_cell``,
``points_per_cell``, the 2-observation ``|dz|``, the rank-1 / rank-4 deformation models and
the flat-column relief (FCR), before any 114 M-point table exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.eval.obs_stats import (
    cross_frame_dz,
    fcr,
    fit_rank1,
    fit_rank3,
    fit_rank4,
    frames_per_cell,
    points_per_cell,
)


def test_frames_per_cell_counts_distinct_kf_per_cell():
    # cell 0 seen by kf {0,1}; cell 1 by {1,2}; cell 2 by {0,1,2}
    key = np.array([0, 0, 1, 1, 2, 2, 2], np.int64)
    kf = np.array([0, 1, 1, 2, 0, 1, 2], np.int32)
    r = frames_per_cell(key, kf)
    assert r["n_cells"] == 3
    assert r["frames_per_cell"] == pytest.approx((2 + 2 + 3) / 3)
    assert r["hist"]["1"] == pytest.approx(0.0)
    assert r["hist"]["2"] == pytest.approx(2 / 3)
    assert r["hist"][">=3"] == pytest.approx(1 / 3)

    # a cell that reports several layers in the SAME frame must still count as one frame
    key2 = np.array([0, 0, 0], np.int64)
    kf2 = np.array([7, 7, 7], np.int32)
    assert frames_per_cell(key2, kf2)["frames_per_cell"] == pytest.approx(1.0)


def test_points_per_cell_is_sum_n_over_cells():
    key = np.array([0, 0, 1], np.int64)
    n = np.array([5, 5, 10], np.int16)
    r = points_per_cell(key, n)
    assert r["n_cells"] == 2
    assert r["n_points"] == 20
    assert r["points_per_cell"] == pytest.approx(10.0)


def test_cross_frame_dz_only_two_observation_columns():
    # cell 0: two observations 0.0 / 4.0 -> |dz| 4
    # cell 1: three observations -> excluded
    # cell 2: two observations 1.0 / 1.5 -> |dz| 0.5
    key = np.array([0, 0, 1, 1, 1, 2, 2], np.int64)
    kf = np.array([0, 1, 0, 1, 2, 0, 1], np.int32)
    z = np.array([0.0, 4.0, 0.0, 1.0, 2.0, 1.0, 1.5], np.float64)
    r = cross_frame_dz(key, kf, z)
    assert r["n_columns"] == 2
    assert r["median_abs_dz"] == pytest.approx(2.25)      # median(4.0, 0.5)
    assert r["p90_abs_dz"] == pytest.approx(3.65, abs=1e-6)


def test_rank1_recovers_per_frame_offset():
    rng = np.random.default_rng(0)
    cells, kfs, z = [], [], []
    b = {0: 0.0, 1: 0.5, 2: -0.7}
    h = rng.uniform(-5, 5, 40)
    for c in range(40):
        for k in (0, 1, 2):
            cells.append(c); kfs.append(k); z.append(h[c] + b[k])
    key, kf, zv = map(np.asarray, (cells, kfs, z))
    r = fit_rank1(key, kf, zv)
    # offsets are only determined up to a constant: check the *differences*
    est = r["b"]
    assert est[1] - est[0] == pytest.approx(0.5, abs=1e-6)
    assert est[2] - est[0] == pytest.approx(-0.7, abs=1e-6)
    assert r["median_abs_dz_after"] < 1e-6


def test_rank4_recovers_affine_per_frame_and_beats_rank1():
    rng = np.random.default_rng(1)
    h = rng.uniform(-5, 5, 60)
    x = rng.uniform(-30, 30, 60)
    y = rng.uniform(-30, 30, 60)
    th = {0: (1.0, 0.0, 0.0, 0.0), 1: (1.05, 0.4, 0.02, -0.01), 2: (0.95, -0.3, -0.01, 0.02)}
    cells, kfs, zv, xs, ys = [], [], [], [], []
    for c in range(60):
        for k in (0, 1, 2):
            a, bb, cx, cy = th[k]
            cells.append(c); kfs.append(k); xs.append(x[c]); ys.append(y[c])
            zv.append(a * h[c] + bb + cx * x[c] + cy * y[c])
    key, kf = np.asarray(cells), np.asarray(kfs)
    zv, xs, ys = np.asarray(zv), np.asarray(xs), np.asarray(ys)

    r4 = fit_rank4(key, kf, zv, xs, ys)
    r1 = fit_rank1(key, kf, zv)
    assert r4["median_abs_dz_after"] < 1e-6
    assert r1["median_abs_dz_after"] > 0.2
    assert r4["median_abs_dz_after"] < r1["median_abs_dz_after"]


def test_fcr_is_relief_of_flat_columns():
    # one cell, 3 frames: flat within each frame, but 3 m apart across frames
    key = np.array([0, 0, 0], np.int64)
    kf = np.array([0, 1, 2], np.int32)
    z = np.array([0.0, 2.0, 3.0])
    z_std = np.array([0.05, 0.05, 0.05])
    n = np.array([10, 10, 10], np.int16)
    r = fcr(key, kf, z, z_std, n, flat_sigma=0.12, min_n=3)
    assert r["n_columns"] == 1
    assert r["fcr"] == pytest.approx(3.0)

    # a column whose layers are not flat is not a "flat column"
    r2 = fcr(key, kf, z, np.array([0.5, 0.5, 0.5]), n, flat_sigma=0.12, min_n=3)
    assert r2["n_columns"] == 0


def test_rank_fits_accept_a_flat_column_mask():
    """The task card fits the rank models only on flat columns: a mask must be honoured."""
    rng = np.random.default_rng(2)
    h = rng.uniform(-5, 5, 30)
    cells, kfs, z, flat = [], [], [], []
    for c in range(30):
        for k in (0, 1):
            cells.append(c); kfs.append(k)
            # every 3rd column is a rugged one: 20 m apart across frames
            rough = (c % 3 == 0)
            z.append(h[c] + (20.0 if rough and k == 1 else 0.0) + 0.5 * k)
            flat.append(not rough)
    key, kf = np.asarray(cells), np.asarray(kfs)
    zv, fm = np.asarray(z), np.asarray(flat, bool)

    r_all = fit_rank1(key, kf, zv)
    r_flat = fit_rank1(key, kf, zv, mask=fm)
    assert r_flat["n_columns"] == 20                    # only the flat columns are used
    assert r_flat["median_abs_dz_after"] < r_all["median_abs_dz_after"]


def test_rank3_recovers_tilt_and_beats_rank1():
    """rank-3 (b + tx u + ty v) must remove a pure tilt that rank-1 cannot."""
    rng = np.random.default_rng(3)
    n_cell = 80
    u = rng.uniform(-50, 50, n_cell)
    v = rng.uniform(-50, 50, n_cell)
    h = rng.uniform(-5, 5, n_cell)
    th = {0: (0.0, 0.0, 0.0), 1: (0.6, 0.02, -0.01), 2: (-0.4, -0.015, 0.02)}
    cells, kfs, z, xs, ys = [], [], [], [], []
    for c in range(n_cell):
        for k in (0, 1, 2):
            b, tx, ty = th[k]
            cells.append(c); kfs.append(k); xs.append(u[c]); ys.append(v[c])
            z.append(h[c] + b + tx * u[c] + ty * v[c])
    key, kf = np.asarray(cells), np.asarray(kfs)
    zv, xs, ys = np.asarray(z), np.asarray(xs), np.asarray(ys)

    r3 = fit_rank3(key, kf, zv, xs, ys)
    r1 = fit_rank1(key, kf, zv)
    assert r3["median_abs_dz_after"] < 1e-6
    assert r1["median_abs_dz_after"] > 0.5
    assert r3["median_abs_dz_after"] < 0.05 * r1["median_abs_dz_after"]


def test_rank4_guard_keeps_scale_in_range():
    """An unconstrained 4-DOF can flatten z; with a_range the scale must stay near 1."""
    rng = np.random.default_rng(4)
    n_cell = 120
    u = rng.uniform(-50, 50, n_cell)
    v = rng.uniform(-50, 50, n_cell)
    h = rng.uniform(-8, 8, n_cell)                  # real relief, worth preserving
    # per-frame: scale 1, pure offset -> the honest answer is a = 1
    cells, kfs, z, xs, ys = [], [], [], [], []
    for c in range(n_cell):
        for k in (0, 1, 2):
            cells.append(c); kfs.append(k); xs.append(u[c]); ys.append(v[c])
            z.append(h[c] + 0.5 * k)
    key, kf = np.asarray(cells), np.asarray(kfs)
    zv, xs, ys = np.asarray(z), np.asarray(xs), np.asarray(ys)

    free = fit_rank4(key, kf, zv, xs, ys)
    guarded = fit_rank4(key, kf, zv, xs, ys, a_range=(0.95, 1.05))
    a_free = np.array([free["params"][k][0] for k in free["params"]])
    a_guard = np.array([guarded["params"][k][0] for k in guarded["params"]])
    assert a_guard.min() >= 0.95 - 1e-9 and a_guard.max() <= 1.05 + 1e-9
    # with an honest scale the guard must not be worse than the free fit on this data
    assert guarded["median_abs_dz_after"] < 1e-6
    assert guarded["z_iqr_ratio"] > 0.95
    # the per-frame offsets are large by construction, so the *raw* shift is ~0.5 m; the
    # gate uses the shift AFTER the post-hoc constant alignment the pipeline mandates
    assert guarded["median_z_shift_raw"] > 0.4
    assert guarded["median_z_shift"] < 0.5
    assert a_free.max() <= 1.0 + 1e-6


def test_two_obs_only_restricts_fitting_set():
    """The rank ladder is defined on the 2-observation columns; mask them explicitly."""
    cells, kfs, z = [], [], []
    for c in range(30):
        n_obs = 2 if c % 2 == 0 else 3
        for k in range(n_obs):
            cells.append(c); kfs.append(k); z.append(float(c) + 0.5 * k)
    key, kf, zv = np.asarray(cells), np.asarray(kfs), np.asarray(z)
    r3 = fit_rank3(key, kf, zv, np.asarray(cells, float), np.zeros(len(cells)),
                   two_obs_only=True)
    assert r3["n_columns"] == 15                      # only the 2-obs columns
    r3_all = fit_rank3(key, kf, zv, np.asarray(cells, float), np.zeros(len(cells)))
    assert r3_all["n_columns"] > 15
