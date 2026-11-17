"""P4.6: feather must never turn a covered pixel black.

Streaming tiles have no halo overlap, so the classic cosine ramp gives a border pixel weight 0
and ``0 / 1e-6`` becomes black -- the "black cross" seen wherever a tile border crossed content.
"""
from __future__ import annotations

import numpy as np

from stream_ortho.ortho.tiles import Tile, feather_blend, seam_gradient_ratio, stitch


def _flat_tiles(colour=(0.2, 0.5, 0.8), s=32, n=2):
    return [Tile(i=i, j=j, rgb=np.ones((s, s, 3), np.float32) * np.asarray(colour, np.float32),
                 alpha=np.ones((s, s), np.float32))
            for i in range(n) for j in range(n)]


def test_flat_tiles_stay_flat_with_feather():
    tiles = _flat_tiles(s=32)
    base, _ = stitch(tiles)
    out = feather_blend(tiles, 32, feather=8)
    assert out.min() > 0.19, "a covered pixel must never be black"
    assert np.allclose(out.mean(axis=(0, 1)), base.mean(axis=(0, 1)), atol=1e-3)


def test_feather_equals_stitch_on_flat_tiles():
    tiles = [Tile(i=0, j=0, rgb=np.full((16, 16, 3), 0.4, np.float32),
                  alpha=np.ones((16, 16), np.float32))]
    assert np.allclose(feather_blend(tiles, 16, feather=4), stitch(tiles)[0])


def test_zero_weight_falls_back_to_the_stitch():
    """Even when the ramp is floored, the guard must cover any uncovered weight."""
    tiles = _flat_tiles(colour=(0.3, 0.3, 0.3), s=24)
    out = feather_blend(tiles, 24, feather=12)
    assert np.isfinite(out).all()
    assert out.min() > 0.0


def test_seam_gradient_stays_low():
    tiles = _flat_tiles(colour=(0.35, 0.5, 0.65), s=64)
    out = feather_blend(tiles, 64, feather=8)
    m = seam_gradient_ratio(out, tile=64)
    assert m["ratio"] < 1.5, m


def test_feather_zero_is_the_plain_stitch():
    tiles = _flat_tiles(s=16)
    a, _ = stitch(tiles)
    assert np.allclose(feather_blend(tiles, 16, feather=0), a)
