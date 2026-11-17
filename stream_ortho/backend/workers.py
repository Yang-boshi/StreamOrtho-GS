"""Async tile worker: priority queue + per-frame budget (design doc §7.8 item 4).

The MVP keeps a **synchronous fallback**: ``pump(budget_ms=None)`` called right
after ``submit`` is exactly the week-6 behaviour.  What the worker adds is

* de-duplication (a tile already queued is not enqueued twice),
* priority (pose updates > new footprints; within a class: never-rendered first,
  then the oldest version),
* a per-pump budget so the caller can spread a burst over several frames.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from stream_ortho.ortho.cache import TileCache

TileKey = Tuple[int, int]


@dataclass
class WorkerConfig:
    budget_ms: float = 250.0
    max_tiles: int = 8


class TileWorker:
    """A tiny single-threaded "async" worker (queue now, pump later)."""

    def __init__(self, cache: TileCache, render_fn: Callable, grid,
                 cfg: Optional[WorkerConfig] = None):
        self.cache = cache
        self.render_fn = render_fn
        self.grid = grid
        self.cfg = cfg or WorkerConfig()
        self._heap: List[Tuple[float, int, TileKey]] = []
        self._queued: set = set()
        self._seq = 0
        self.rendered = 0
        self.render_ms = 0.0
        self.deduped = 0
        self.pumps = 0

    # ------------------------------------------------------------------
    def submit(self, keys: Iterable[TileKey], priority: float = 0.0,
               reason: str = "") -> int:
        """Enqueue tiles; returns how many were newly queued."""
        n = 0
        for k in keys:
            if k in self._queued:
                self.deduped += 1
                continue
            t = self.cache.tiles.get(k)
            never = 0 if (t is None or t.renders == 0) else 1
            ver = t.version if t is not None else 0
            self._seq += 1
            # lower tuple sorts first: high priority, never-rendered, oldest version
            heapq.heappush(self._heap, (-float(priority), never, ver, self._seq, tuple(k)))
            self._queued.add(tuple(k))
            self.cache.mark_dirty([k], reason=reason)
            n += 1
        return n

    def pending(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------
    def pump(self, budget_ms: Optional[float] = None,
             max_tiles: Optional[int] = None) -> dict:
        budget = self.cfg.budget_ms if budget_ms is None else budget_ms
        cap = self.cfg.max_tiles if max_tiles is None else max_tiles
        t0 = time.perf_counter()
        done, spent = 0, 0.0
        while self._heap and done < cap and spent < budget:
            _, _, _, _, key = heapq.heappop(self._heap)
            self._queued.discard(tuple(key))
            s = time.perf_counter()
            res = self.render_fn(self.grid, key[0], key[1])
            ms = (time.perf_counter() - s) * 1000.0
            rgb, alpha, depth = res[0], res[1], res[2]
            self.cache.put(tuple(key), rgb, alpha, depth, ms=ms)
            done += 1
            spent += ms
            self.rendered += 1
            self.render_ms += ms
        self.pumps += 1
        return {"rendered": done, "ms": round(spent, 2),
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "pending": len(self._heap)}

    def drain(self, budget_ms: float = 1e9, max_tiles: int = 32) -> dict:
        total = 0
        while self._heap:
            r = self.pump(budget_ms=budget_ms, max_tiles=max_tiles)
            total += r["rendered"]
            if r["rendered"] == 0:
                break
        return {"rendered": total, "pending": len(self._heap)}

    def stats(self) -> dict:
        return {"queued": len(self._heap), "rendered": self.rendered,
                "render_ms_total": round(self.render_ms, 1),
                "render_ms_mean": round(self.render_ms / max(self.rendered, 1), 2),
                "deduped": self.deduped, "pumps": self.pumps}
