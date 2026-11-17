"""Module A: the per-frame filter.

Semantics under test (P2a): the free-space prior vetoes points **above** the resolved column
top (floaters), never points below it -- a first-surface product still needs each cell's lower
or first observation.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from stream_ortho.filter.pointmap import FilterConfig, PointmapFilter


def _plane(rng, n=20000, z0=10.0, lo=0.0, hi=50.0, noise=0.05):
    x = rng.uniform(lo, hi, n); y = rng.uniform(lo, hi, n)
    return x, y, np.full(n, z0) + rng.normal(0, noise, n), np.ones(n)


def _warm(f, x, y, z, c, passes=2):
    for _ in range(passes):
        f.apply(x, y, z, c)


def test_gate_removes_low_conf_and_out_of_range():
    rng = np.random.default_rng(0)
    x, y, z, c = _plane(rng, 500)
    c[:100] = 0.01
    z[100:150] = 1e4
    f = PointmapFilter(FilterConfig(cell=0.45, prune=False, nfp=False))
    keep, r = f.apply(x, y, z, c, height=z - 10.0)
    assert r["gate"] >= 150
    assert keep[100] is np.bool_(False)


def test_floater_above_the_column_is_pruned():
    """A point far above a resolved column is in free space -> vetoed (above mode)."""
    rng = np.random.default_rng(0)
    x, y, z, c = _plane(rng)
    f = PointmapFilter(FilterConfig(cell=0.45))
    _warm(f, x, y, z, c)
    keep, r = f.apply(np.r_[x, 25.0], np.r_[y, 25.0], np.r_[z, 40.0], np.r_[c, 1.0])
    assert keep[-1] == np.bool_(False)
    assert r["nfp_above"] >= 1


def test_roof_survives_the_filter():
    """A dense 3 m roof over the ground must be kept: it is not 'free space above'."""
    rng = np.random.default_rng(0)
    x, y, z, c = _plane(rng)
    n = x.size
    m = 2000
    xr = rng.uniform(10, 20, m); yr = rng.uniform(10, 20, m)
    zr = np.full(m, 13.0) + rng.normal(0, 0.05, m)
    f = PointmapFilter(FilterConfig(cell=0.45))
    _warm(f, x, y, z, c)
    keep, r = f.apply(np.r_[x, xr], np.r_[y, yr], np.r_[z, zr], np.r_[c, np.ones(m)])
    assert keep[n:].mean() >= 0.95
    assert keep[:n].mean() >= 0.95


def test_below_mode_is_opt_in_and_needs_a_layered_column():
    """below only vetoes under a *layered* column (gap > min_gap), never on a plain ground."""
    rng = np.random.default_rng(2)
    x, y, z, c = _plane(rng, 8000, z0=10.0)
    roof = 4000
    xr = rng.uniform(0, 50, roof); yr = rng.uniform(0, 50, roof)
    zr = np.full(roof, 24.0) + rng.normal(0, 0.05, roof)
    f = PointmapFilter(FilterConfig(cell=0.45, nfp_mode="above", prune=False))
    _warm(f, np.r_[x, xr], np.r_[y, yr], np.r_[z, zr], np.r_[c, np.ones(roof)])
    # with above mode the ground under the roof stays
    keep_a, ra = f.apply(x[:500], y[:500], z[:500], c[:500])
    assert ra["nfp_below"] == 0
    f2 = PointmapFilter(FilterConfig(cell=0.45, nfp_mode="both", prune=False))
    _warm(f2, np.r_[x, xr], np.r_[y, yr], np.r_[z, zr], np.r_[c, np.ones(roof)])
    # a point *below* the layered column's bottom (9.4 m) is a hidden surface
    keep_b, rb = f2.apply(x[:500], y[:500], np.full(500, 5.0), c[:500])
    assert rb["nfp_below"] > 0, "a layered column should veto its hidden lower surface"


def test_unique_point_is_protected():
    """A cell observed only once in this frame must never be vetoed."""
    x = np.array([1.0]); y = np.array([1.0]); z = np.array([100.0]); c = np.array([1.0])
    f = PointmapFilter(FilterConfig(cell=0.45, protect_unique=True))
    f._ensure(16)                              # pretend the column is resolved elsewhere
    f._col_cnt[:] = 0
    keep, r = f.apply(x, y, z, c)
    assert keep.all()


def test_low_obs_column_does_not_veto():
    # one point per column: cnt stays 1, so the column is never "resolved"
    x = np.array([1.0, 3.0, 5.0]); y = np.array([1.0, 3.0, 5.0])
    z = np.array([10.0, 10.0, 10.0]); c = np.ones(3)
    f = PointmapFilter(FilterConfig(cell=0.45, protect_low_obs=True, prune=False))
    f.apply(x, y, z, c)                        # one observation each
    keep, r = f.apply(x, y, z + 30.0, c)       # floaters, but the columns are unresolved
    assert r["nfp_above"] == 0
    assert keep.all()


def test_roi_cache_matches_a_full_frame():
    """Two frames with different bboxes: the ROI path must equal the plain computation."""
    rng = np.random.default_rng(4)
    x, y, z, c = _plane(rng, 20000)
    f = PointmapFilter(FilterConfig(cell=0.45))
    _warm(f, x, y, z, c)
    x2 = np.r_[x, 25.0]; y2 = np.r_[y, 25.0]; z2 = np.r_[z, 60.0]
    keep, r = f.apply(x2, y2, z2, np.r_[c, 1.0])
    assert keep[-1] == np.bool_(False)          # the floater is gone
    assert keep[:x.size].mean() > 0.95          # and the plane is intact


def test_filter_latency_on_a_pointmap_sized_frame():
    rng = np.random.default_rng(5)
    n = 147_000                                 # 288x512 pointmap
    x = rng.uniform(0, 300, n); y = rng.uniform(0, 300, n)
    z = np.full(n, 10.0) + rng.normal(0, 0.1, n)
    f = PointmapFilter(FilterConfig(cell=0.45))
    f.apply(x, y, z, np.ones(n))
    t0 = time.perf_counter()
    f.apply(x, y, z, np.ones(n))
    ms = (time.perf_counter() - t0) * 1000
    assert ms < 400, f"filter took {ms:.0f} ms"
