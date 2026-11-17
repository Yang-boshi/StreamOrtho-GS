"""Async mosaic writer: copies on submit, bounded queue, atomic writes, final never dropped."""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

from stream_ortho.app.mosaic_writer import AsyncMosaicWriter, snapshot_tiles, stitch_arrays


class FakeTile:
    def __init__(self, rgb, renders=1):
        self.rgb = rgb
        self.alpha = np.ones(rgb.shape[:2], np.float32)
        self.renders = renders


class FakeCache:
    def __init__(self, tiles):
        self.tiles = tiles


def _tiles(n=2, size=8):
    out = []
    for k in range(n):
        rgb = np.full((size, size, 3), 0.2 * (k + 1), np.float32)
        out.append(((0, k), FakeTile(rgb)))
    return out


def test_snapshot_copies_the_arrays():
    tiles = _tiles()
    cache = FakeCache(dict(tiles))
    snap = snapshot_tiles(cache)
    cache.tiles[(0, 0)].rgb[:] = 9.0               # mutate the cache afterwards
    assert np.allclose(snap[0][1], 0.2), "snapshot must be a copy, not a view"


def test_stitch_arrays_places_tiles_side_by_side():
    # i indexes columns, so two tiles (0,0) and (0,1) stack vertically (j is the row)
    snap = snapshot_tiles(FakeCache(dict(_tiles(2, size=8))))
    rgb, a = stitch_arrays(snap)
    assert rgb.shape == (16, 8, 3)
    assert a.shape == (16, 8)


def test_close_writes_all_and_leaves_no_tmp(tmp_path):
    w = AsyncMosaicWriter(maxsize=4)
    for k in range(3):
        w.submit(tmp_path / f"m{k}.png", snapshot_tiles(FakeCache(dict(_tiles()))))
    w.close()
    assert w.stats()["written"] == 3
    assert not list(tmp_path.glob("*.tmp"))
    for k in range(3):
        assert (tmp_path / f"m{k}.png").stat().st_size > 0


def test_queue_drops_old_preview_but_never_final(tmp_path, monkeypatch):
    """A full queue evicts the oldest preview; a final is written anyway."""
    import stream_ortho.app.mosaic_writer as mw
    orig = mw._save_rgba
    monkeypatch.setattr(mw, "_save_rgba", lambda *a, **k: (time.sleep(0.3), orig(*a, **k))[1])
    w = AsyncMosaicWriter(maxsize=1)
    snap = snapshot_tiles(FakeCache(dict(_tiles())))
    for k in range(4):
        w.submit(tmp_path / f"p{k}.png", snap)
    assert w.dropped >= 1, "previews must be dropped when the queue is full"
    w.submit(tmp_path / "final.png", snap, final=True)
    w.close()
    assert (tmp_path / "final.png").exists() and (tmp_path / "final.png").stat().st_size > 0


def test_final_is_written(tmp_path):
    w = AsyncMosaicWriter(maxsize=2)
    w.submit(tmp_path / "final.png", snapshot_tiles(FakeCache(dict(_tiles()))), final=True)
    w.close()
    assert (tmp_path / "final.png").exists()
    assert (tmp_path / "final.png").stat().st_size > 0


def test_errors_are_collected_not_raised(tmp_path):
    w = AsyncMosaicWriter(maxsize=1)
    w.submit(tmp_path / "ok.png", [])              # empty tile list -> no image, no crash
    w.close()
    assert isinstance(w.stats()["errors"], list)


def test_slow_writer_keeps_the_loop_fast(tmp_path):
    """Simulate a 1 s encode: submitting 20 snapshots must not block the caller."""
    w = AsyncMosaicWriter(maxsize=1)
    w._worker_orig = w._worker
    snap = snapshot_tiles(FakeCache(dict(_tiles(1, size=32))))
    t0 = time.perf_counter()
    for k in range(20):
        w.submit(tmp_path / f"m{k}.png", snap)
    dt = time.perf_counter() - t0
    w.close()
    assert dt < 1.0, f"submit blocked for {dt:.2f} s"
