"""P4.5: ``StreamingProduct.insert`` takes ABSOLUTE cell indices.

The live pipeline subtracted the product origin itself before calling ``insert``, and ``_flat``
subtracts it again.  With a 2048-cell preallocated grid the origin is a large negative number,
so the second subtraction pushes every point past the last row and the mosaic collapses into
grey stripes.  These tests pin the contract and fail on the old caller.
"""
from __future__ import annotations

import numpy as np

from stream_ortho.online.product import StreamingProduct


def _layers(p):
    z, n, owner = p.layers()
    return z, n, owner


def test_absolute_indices_are_the_contract():
    cell, ox, oy = 0.5, -938, -1354
    p = StreamingProduct(cell=cell, ox=ox, oy=oy, H=2048, W=2048, L=3, z_gap=2.0)
    x, y = -400.0, -600.0
    ix, iy = int(np.floor(x / cell)), int(np.floor(y / cell))
    r = p.insert(0, np.array([ix]), np.array([iy]), np.array([1.0]), np.array([1.0]))
    assert r["inserted"] == 1
    _, n, _ = _layers(p)
    assert int(n.sum()) == 1
    assert float(n[iy - oy, ix - ox, 0]) == 1.0


def test_relative_indices_do_not_land_in_the_intended_cell():
    """The buggy call: subtract the origin first, then ``_flat`` subtracts it again."""
    cell, ox, oy = 0.5, -938, -1354
    p = StreamingProduct(cell=cell, ox=ox, oy=oy, H=2048, W=2048, L=3, z_gap=2.0)
    ix_abs, iy_abs = int(np.floor(-400.0 / cell)), int(np.floor(-600.0 / cell))
    p.insert(0, np.array([ix_abs - ox]), np.array([iy_abs - oy]),
             np.array([1.0]), np.array([1.0]))
    _, n, _ = _layers(p)
    assert float(n[iy_abs - oy, ix_abs - ox, 0]) == 0.0, "the intended cell must stay empty"
    live = np.argwhere(n[..., 0] > 0)
    assert len(live) == 1 and tuple(live[0]) != (iy_abs - oy, ix_abs - ox)


def test_a_plane_survives_with_a_large_negative_origin():
    """A 60x60 m patch must stay a patch, not collapse onto one row."""
    cell, ox, oy = 0.5, -938, -1354
    p = StreamingProduct(cell=cell, ox=ox, oy=oy, H=2048, W=2048, L=3, z_gap=2.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-420, -360, 5000)
    y = rng.uniform(-620, -560, 5000)
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    p.insert(0, ix, iy, np.full(5000, 5.0), np.ones(5000))
    _, n, _ = _layers(p)
    rows = np.unique(iy - oy)
    cols = np.unique(ix - ox)
    assert len(rows) > 50 and len(cols) > 50, (len(rows), len(cols))
    assert int((n > 0).sum()) > 0.5 * 5000
