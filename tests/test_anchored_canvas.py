"""Regression tests for the fixed-canvas (anchored) stills + video path.

The P4.7 canvas was anchored at tile (0,0): the plane origin sits inside the flight, most
tiles carry negative i/j, and stitch silently skipped them -- the first video frames were
black.  These tests pin the anchored behaviour end to end.
"""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.app.artifacts import write_update_video
from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.mosaic import save_mosaic
from stream_ortho.ortho.tiles import Tile, stitch


def _tile(i, j, S=8, colour=0.5):
    rgb = np.full((S, S, 3), colour, np.float32)
    return Tile(i=i, j=j, rgb=rgb, alpha=np.ones((S, S), np.float32))


def test_stitch_anchored_keeps_negative_tiles():
    """Tiles with negative i/j must land inside the canvas, never be skipped."""
    tiles = [_tile(-1, -1), _tile(0, 0)]
    rgb, a = stitch(tiles, shape=(3, 3), origin=(-1, -1), jmax_anchor=1)
    assert rgb.shape == (24, 24, 3)
    # tile (-1,-1): r = (jmax - j)*S = (1+1)*8 = 16, c = (i - imin)*S = 0
    assert a[16:24, 0:8].all(), "negative-index tile was skipped"
    # tile (0,0): r = (1-0)*8 = 8, c = (0+1)*8 = 8
    assert a[8:16, 8:16].all()


def test_save_mosaic_crop_box_roundtrip(tmp_path):
    """crop=True returns a box whose size equals the saved PNG (after scaling)."""
    cache = TileCache()
    S = 8
    for (i, j) in ((-1, -1), (0, 0)):
        t = _tile(i, j, S=S)
        cache.put((i, j), t.rgb, t.alpha, np.zeros((S, S), np.float32), ms=0.0)
    info = save_mosaic(cache, tmp_path / "m.png", tile_size=S, crop=True,
                       scale=0.5, origin=(-1, -1), shape=(3, 3), jmax_anchor=1)
    box = info["crop_box"]
    assert box is not None
    r0, r1, c0, c1 = box
    assert (r1 - r0, c1 - c0) == tuple(info["shape"])


def test_write_update_video_union_alignment(tmp_path):
    """Frames cropped to different content boxes land at the right offsets in the clip."""
    # frame A: 4x4 content at canvas rows 0-3, cols 0-3; frame B: 4x4 at rows 8-11, cols 4-7
    a = np.zeros((4, 4, 3), np.uint8); a[:] = 200
    b = np.zeros((4, 4, 3), np.uint8); b[:] = 100
    pa = tmp_path / "mosaic_00000.png"; pb = tmp_path / "mosaic_00001.png"
    from PIL import Image
    Image.fromarray(a).save(pa); Image.fromarray(b).save(pb)
    boxes = [(0, 4, 0, 4), (8, 12, 4, 8)]
    info = write_update_video(tmp_path / "v.mp4", [str(pa), str(pb)], None, 8,
                              fps=5, boxes=boxes)
    assert "error" not in info
    import cv2
    cap = cv2.VideoCapture(str(tmp_path / "v.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert n == 2 and (w, h) == (8, 12)
    ok, f0 = cap.read(); ok2, f1 = cap.read()
    cap.release()
    assert ok and ok2
    g0 = f0[..., 0]  # BGR -- all channels equal (grey)
    g1 = f1[..., 0]
    assert g0[0:4, 0:4].mean() > 150, "frame A content not at its box offset"
    assert g1[8:12, 4:8].mean() > 50, "frame B content not at its box offset"
    assert g0[8:12, :].max() == 0 and g1[0:4, :].max() == 0


def test_pipeline_canvas_anchor_covers_product():
    """The canvas formula must cover every tile index the product can touch."""
    cell, gsd, tile, G = 0.0158465, 0.0158465, 512, 2048
    ox, oy = -938, -1355                     # the live P4.5/P4.7 grid
    tile_m = gsd * tile
    i0 = int(np.floor((ox * cell) / tile_m)); i1 = int(np.floor((ox + G) * cell / tile_m - 1e-9))
    j0 = int(np.floor((oy * cell) / tile_m)); j1 = int(np.floor((oy + G) * cell / tile_m - 1e-9))
    # the flight covers tiles with negative indices (P4.7 skipped exactly these)
    assert i0 < 0 and j0 < 0
    # and every covered tile of the actual run fits the canvas
    covered = [(i, j) for i in (-2, -1, 0, 1) for j in (-3, -2, -1, 0, 1)]
    for (i, j) in covered:
        r = (j1 - j) * tile; c = (i - i0) * tile
        rows = (j1 - j0 + 1) * tile; cols = (i1 - i0 + 1) * tile
        assert 0 <= r and 0 <= c and r + tile <= rows and c + tile <= cols
