"""Invariants of the streaming product: frozen owners, new layers only on a miss."""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.online.product import StreamingProduct


def _cells(n=3):
    ix, iy = np.meshgrid(np.arange(n), np.arange(n))
    return ix.ravel(), iy.ravel()


def test_owner_is_frozen_under_repeat_observations():
    p = StreamingProduct(cell=0.5, ox=0, oy=0, H=4, W=4, L=3, z_gap=2.0)
    ix, iy = _cells()
    p.insert(0, ix, iy, np.zeros(ix.size) + 10.0, np.ones(ix.size))
    p.insert(1, ix, iy, np.zeros(ix.size) + 10.2, np.ones(ix.size))   # same layer
    p.insert(2, ix, iy, np.zeros(ix.size) + 9.9, np.ones(ix.size))
    z, n, own = p.layers()
    live = n[..., 0] > 0
    assert live.any()
    assert np.all(own[..., 0][live] == 0)          # never re-owned
    assert np.all(n[..., 1] == 0)                  # never split


def test_new_layer_only_when_no_match():
    p = StreamingProduct(cell=0.5, ox=0, oy=0, H=4, W=4, L=3, z_gap=1.0)
    ix, iy = _cells()
    p.insert(0, ix, iy, np.zeros(ix.size) + 5.0, np.ones(ix.size))
    r = p.insert(1, ix, iy, np.zeros(ix.size) + 20.0, np.ones(ix.size))   # 15 m away
    assert r["new_layers"] > 0
    z, n, own = p.layers()
    assert (n[..., 1] > 0).any()
    assert np.all(own[..., 1][n[..., 1] > 0] == 1)


def test_top_and_near_agree_with_the_layers():
    p = StreamingProduct(cell=0.5, ox=0, oy=0, H=4, W=4, L=3, z_gap=1.0)
    ix, iy = _cells()
    p.insert(0, ix, iy, np.zeros(ix.size) + 2.0, np.ones(ix.size))
    p.insert(1, ix, iy, np.zeros(ix.size) + 12.0, np.ones(ix.size))
    top, owner = p.top()
    assert np.allclose(top[:3, :3], 12.0)
    assert np.all(owner[:3, :3] == 1)
    near = p.near(ix[:4], iy[:4], np.full(4, 2.1))
    assert np.allclose(near, 2.0)


def test_weights_are_a_weighted_mean():
    p = StreamingProduct(cell=0.5, ox=0, oy=0, H=2, W=2, L=2, z_gap=5.0)
    p.insert(0, np.array([0]), np.array([0]), np.array([10.0]), np.array([1.0]))
    p.insert(1, np.array([0]), np.array([0]), np.array([12.0]), np.array([3.0]))
    z, n, _ = p.layers()
    assert n[0, 0, 0] == 2
    assert z[0, 0, 0] == pytest.approx((1 * 10 + 3 * 12) / 4)
