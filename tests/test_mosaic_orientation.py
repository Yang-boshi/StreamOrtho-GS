"""P3.1b: the mosaic must have the same orientation as the product (no transpose/flip)."""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.app.mosaic_writer import snapshot_tiles, stitch_arrays
from stream_ortho.ortho.tiles import Tile, stitch


class _T:
    def __init__(self, rgb):
        self.rgb = rgb
        self.alpha = np.ones(rgb.shape[:2], np.float32)
        self.renders = 1


class _C:
    def __init__(self, d):
        self.tiles = d


def _grid_tiles(s=4):
    """2x2 tiles with a distinct colour each: i = column, j = row."""
    colours = {(0, 0): (1.0, 0.0, 0.0), (1, 0): (0.0, 1.0, 0.0),
               (0, 1): (0.0, 0.0, 1.0), (1, 1): (1.0, 1.0, 0.0)}
    return {k: _T(np.full((s, s, 3), v, np.float32)) for k, v in colours.items()}


def test_stitch_arrays_matches_the_reference_stitch():
    tiles = _grid_tiles()
    mine, a_mine = stitch_arrays(snapshot_tiles(_C(dict(tiles))))
    ref_tiles = [Tile(i=i, j=j, rgb=t.rgb, alpha=t.alpha) for (i, j), t in tiles.items()]
    ref, a_ref = stitch(ref_tiles)
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref), "the two stitchers must agree pixel for pixel"
    assert np.allclose(a_mine, a_ref)


def test_orientation_is_not_transposed():
    s = 4
    tiles = _grid_tiles(s)
    rgb, _ = stitch_arrays(snapshot_tiles(_C(dict(tiles))))
    # j = 1 is the "upper" row -> after the flip it lands on the top of the image
    assert np.allclose(rgb[0, 0], [0.0, 0.0, 1.0])      # (i=0, j=1)
    assert np.allclose(rgb[0, s], [1.0, 1.0, 0.0])      # (i=1, j=1)
    assert np.allclose(rgb[s, 0], [1.0, 0.0, 0.0])      # (i=0, j=0)
    assert np.allclose(rgb[s, s], [0.0, 1.0, 0.0])      # (i=1, j=0)


def test_single_tile_roundtrip():
    t = {(3, 7): _T(np.full((3, 3, 3), 0.5, np.float32))}
    rgb, a = stitch_arrays(snapshot_tiles(_C(dict(t))))
    assert rgb.shape == (3, 3, 3) and np.allclose(rgb, 0.5)


def test_tile_render_is_not_flipped():
    """A single marked cell at the tile's top-left must land at pixel (0, 0)."""
    import numpy as np
    from stream_ortho.online.product import StreamingProduct
    from stream_ortho.ortho.grid import OrthoGrid
    from stream_ortho.scripts.run_stream import tile_render_fn_factory

    cell = 0.45
    gsd = 0.30
    p = StreamingProduct(cell, 0, 0, 200, 200, 3, 2.0)
    # a red cell at the highest row of tile (0,0): x small (left), y large (top)
    x_tl = 0.5 * cell            # left edge
    y_tl = 150.0                 # top of the tile (tile spans 0..153.6 m)
    p.insert(0, np.array([1]), np.array([int(y_tl / cell)]),
             np.array([10.0]), np.array([1.0]), rgb=np.array([[255.0, 0.0, 0.0]], np.float32))
    g = OrthoGrid(np.eye(4), gsd, 512)
    r, a, d, _ = tile_render_fn_factory(p, gsd, 512, mode="rgb")(g, 0, 0)
    # row 0 must be the TOP of the tile; the marked cell therefore lands near row 0
    red_row = np.argmax((r[..., 0] > 0.4) & (r[..., 1] < 0.2))
    red_rows = np.nonzero((r[..., 0] > 0.4) & (r[..., 1] < 0.2))[0]
    red_cols = np.nonzero((r[..., 0] > 0.4) & (r[..., 1] < 0.2))[1]
    assert red_rows.size > 0, "the marked cell should be visible"
    assert red_rows.min() < 64, f"marked cell rendered at row {red_rows.min()}, should be near the top"
    assert red_cols.min() < 64, "marked cell should be near the left edge"
