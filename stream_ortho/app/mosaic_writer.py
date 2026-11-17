"""Asynchronous mosaic writer: the streaming loop must not block on a 115 MB PNG.

A preview mosaic at 5632x5120 RGBA takes 1-1.5 s to encode, which is fine as a background
task but ruinous inside ``on_frame``.  The loop therefore only **snapshots** the tile cache
(copying the arrays, so the worker never races the renderer) and hands the snapshot to a
bounded queue; a daemon thread stitches and writes it.

Guarantees:

* ``final`` mosaics are never dropped -- they are retried/blocked instead;
* preview mosaics may be dropped when the queue is full (bounded memory, bounded latency);
* every file is written to ``<name>.tmp`` and then ``os.replace``d, so a reader never sees a
  half-written PNG;
* exceptions are collected in ``errors`` instead of killing the run.
"""
from __future__ import annotations

import os
import pathlib
import queue
import threading
import time

import numpy as np


def snapshot_tiles(cache):
    """Copy the rendered tiles out of the cache (main thread only)."""
    out = []
    for (i, j), t in cache.tiles.items():
        if t is None or getattr(t, "renders", 0) == 0:
            continue
        out.append(((int(i), int(j)), np.array(t.rgb, copy=True), np.array(t.alpha, copy=True)))
    return out


def stitch_arrays(tiles):
    """Stitch ``[(key, rgb, alpha)]`` into ``(rgb, alpha)``.

    The axes follow :func:`stream_ortho.ortho.tiles.stitch`: ``i`` indexes columns, ``j``
    indexes rows and grows "up" in the plane, so rows are built from ``jmax - j``.  Getting
    this backwards transposes the mosaic (the "tiles in the wrong place" look).
    """
    if not tiles:
        return None, None
    imin = min(k[0] for k, _, _ in tiles)
    imax = max(k[0] for k, _, _ in tiles)
    jmin = min(k[1] for k, _, _ in tiles)
    jmax = max(k[1] for k, _, _ in tiles)
    s = tiles[0][1].shape[0]
    rows, cols = (jmax - jmin + 1), (imax - imin + 1)
    rgb = np.zeros((rows * s, cols * s, 3), np.float32)
    a = np.zeros((rows * s, cols * s), np.float32)
    for (i, j), trgb, ta in tiles:
        y0 = (jmax - j) * s
        x0 = (i - imin) * s
        rgb[y0:y0 + s, x0:x0 + s] = trgb
        a[y0:y0 + s, x0:x0 + s] = ta
    return rgb, a


def _save_rgba(path: pathlib.Path, rgb, alpha, fmt: str = "png", scale: float = 1.0):
    from PIL import Image
    rgb = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0) if alpha is not None else np.ones(
        rgb.shape[:2], np.float32)
    if scale and scale != 1.0:
        h, w = rgb.shape[:2]
        size = (max(int(w * scale), 1), max(int(h * scale), 1))
        rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(size, Image.BILINEAR),
                         np.float32) / 255.0
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.BILINEAR),
                       np.float32) / 255.0
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt.lower() in ("jpg", "jpeg"):
        im = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        pil_fmt = "JPEG"
    else:
        im = Image.fromarray((np.concatenate([rgb, a[..., None]], -1) * 255).astype(np.uint8),
                             mode="RGBA")
        pil_fmt = "PNG"
    # pass the format explicitly: the atomic temp name ends in ".tmp" and PIL cannot infer it
    im.save(tmp, format=pil_fmt)
    os.replace(tmp, path)


class AsyncMosaicWriter:
    def __init__(self, maxsize: int = 2, fmt: str = "png", scale: float = 1.0):
        self.q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.fmt, self.scale = fmt, scale
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.written = 0
        self.dropped = 0
        self.errors: list = []
        self.last_ms = 0.0
        self.total_ms = 0.0

    # ------------------------------------------------------------------
    def submit(self, path, tiles, final: bool = False, scale: float | None = None):
        """Non-blocking; ``tiles`` must already be a private copy."""
        item = (pathlib.Path(path), tiles, bool(final),
                self.scale if scale is None else float(scale))
        try:
            self.q.put_nowait(item)
        except queue.Full:
            if final:
                # never drop a final; if the worker cannot drain in time, write it inline
                try:
                    self.q.put(item, timeout=5.0)
                except queue.Full:
                    self._write_now(item)
                return
            try:
                old = self.q.get_nowait()
                self.q.task_done()
                if old[2]:                             # the oldest was a final: keep it
                    self.q.put(old)
                    self.q.put(item)
                    return
                self.dropped += 1
            except queue.Empty:
                pass
            self.q.put(item)

    def _write_now(self, item):
        """Synchronous fallback for a final mosaic when the queue is stuck."""
        path, tiles, _final, scale = item
        t0 = time.perf_counter()
        rgb, a = stitch_arrays(tiles)
        if rgb is not None:
            _save_rgba(path, rgb, a, self.fmt, scale)
            self.written += 1
            self.sync_writes = getattr(self, "sync_writes", 0) + 1
        self.last_ms = (time.perf_counter() - t0) * 1000

    def _worker(self):
        while True:
            path, tiles, final, scale = self.q.get()
            try:
                if path is None:
                    break
                t0 = time.perf_counter()
                rgb, a = stitch_arrays(tiles)
                if rgb is not None:
                    _save_rgba(path, rgb, a, self.fmt, scale)
                    self.written += 1
                dt = (time.perf_counter() - t0) * 1000
                self.last_ms, self.total_ms = dt, self.total_ms + dt
            except Exception as exc:                   # never kill the loop
                self.errors.append(repr(exc))
            finally:
                self.q.task_done()

    def close(self, wait: bool = True, timeout: float = 60.0):
        self.q.put((None, None, False, 1.0))
        if wait:
            self.q.join()
            self.thread.join(timeout=timeout)

    def stats(self) -> dict:
        return {"written": self.written, "sync_writes": getattr(self, "sync_writes", 0), "dropped": self.dropped, "queue": self.q.qsize(),
                "last_ms": round(self.last_ms, 1), "total_ms": round(self.total_ms, 1),
                "errors": self.errors[-3:]}
