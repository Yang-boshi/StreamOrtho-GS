"""P2b: FOV bbox/tiles, mosaic dumping, dump cadence, and no flush_all in the loop."""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.fov import footprint_bbox, fov_tiles
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.mosaic import save_mosaic
from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig


def _engine(render_fn=None):
    g = OrthoGrid(T_WP=np.eye(4), gsd=0.30, tile=64)
    cache = TileCache()
    if render_fn is None:
        def render_fn(grid, i, j):
            return (np.full((grid.tile, grid.tile, 3), 0.5, np.float32),
                    np.ones((grid.tile, grid.tile), np.float32),
                    np.zeros((grid.tile, grid.tile), np.float32), {})
    return OrthoEngine(g, cache, render_fn, SchedulerConfig(tile=64, budget_ms=1e9,
                                                            max_tiles_per_flush=8))


def test_fov_bbox_and_margin():
    pts = np.array([[0.0, 0.0], [10.0, 5.0], [-2.0, 3.0]])
    assert footprint_bbox(pts) == (-2.0, 0.0, 10.0, 5.0)
    assert footprint_bbox(pts, margin_m=1.0) == (-3.0, -1.0, 11.0, 6.0)
    with pytest.raises(ValueError):
        footprint_bbox(np.zeros((0, 2)))


def test_fov_tiles_cover_the_bbox():
    eng = _engine()
    bbox = (0.0, 0.0, 40.0, 40.0)          # tile = 64 * 0.3 = 19.2 m
    keys = fov_tiles(eng, bbox, margin_tiles=0)
    assert (0, 0) in keys and (2, 2) in keys


def test_mosaic_save_has_alpha_and_size(tmp_path):
    eng = _engine()
    eng.mark_footprint((0.0, 0.0, 30.0, 30.0), kf_id=0)
    eng.flush(budget_ms=1e9, max_tiles=8)
    info = save_mosaic(eng.cache, tmp_path / "m.png", tile_size=64)
    assert info is not None and info["tiles"] >= 1
    from PIL import Image
    im = Image.open(tmp_path / "m.png")
    assert im.mode == "RGBA"
    a = np.asarray(im)
    assert a.shape[2] == 4
    assert a[..., 3].max() > 0                    # something was drawn
    assert set(np.unique(a[..., 3])) <= {0, 255}  # alpha is 0/1 only


def test_mosaic_returns_none_when_empty(tmp_path):
    assert save_mosaic(TileCache(), tmp_path / "empty.png") is None


def test_no_flush_all_in_the_loop():
    """The streaming loop must only call flush(); flush_all is reserved for the end."""
    calls = {"flush": 0, "flush_all": 0}
    eng = _engine()
    orig_flush, orig_all = eng.flush, eng.flush_all
    eng.flush = lambda *a, **k: (calls.__setitem__("flush", calls["flush"] + 1),
                                 orig_flush(*a, **k))[1]
    eng.flush_all = lambda *a, **k: (calls.__setitem__("flush_all", calls["flush_all"] + 1),
                                     orig_all(*a, **k))[1]
    for n in range(5):
        eng.mark_footprint((n * 10.0, 0.0, n * 10.0 + 20.0, 20.0), kf_id=n)
        eng.flush(budget_ms=1e9, max_tiles=4)
    assert calls["flush"] == 5 and calls["flush_all"] == 0


def test_dump_every_produces_the_expected_count(tmp_path):
    """10 frames with dump_every=5 -> mosaics at frame 5 and 10."""
    eng = _engine()
    made = 0
    for n in range(1, 11):
        eng.mark_footprint((0.0, 0.0, 30.0, 30.0), kf_id=n)
        eng.flush(budget_ms=1e9, max_tiles=4)
        if n % 5 == 0:
            info = save_mosaic(eng.cache, tmp_path / f"mosaic_{n:05d}.png", tile_size=64)
            made += int(info is not None)
    assert made == 2
    assert (tmp_path / "mosaic_00005.png").exists()
    assert (tmp_path / "mosaic_00010.png").exists()
