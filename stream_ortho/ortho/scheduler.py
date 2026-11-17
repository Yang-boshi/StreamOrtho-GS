"""Dirty-tile scheduler for the streaming ortho back-end (design doc §7.7 item 1).

The engine owns a **global** tile grid (absolute ``(i, j)`` indices, so tiles can
be negative just like :class:`OrthoGrid`) and a :class:`TileCache`.  Two events
mark tiles dirty:

* ``mark_footprint(bbox)``   -- a new keyframe arrived: its ground footprint
  (already projected to the plane) covers a bbox -> all tiles intersecting it
  (plus a 1-tile margin) become dirty;
* ``mark_pose_changed(kf_ids)`` -- a pose was updated: only the tiles that contain
  Gaussians whose ``owner`` is one of those keyframes become dirty (the interface
  week 7's pose graph will call).

``flush(budget_ms)`` renders dirty tiles by priority until the budget is spent,
so a single keyframe can never trigger an unbounded re-render.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.grid import OrthoGrid

BBox = Tuple[float, float, float, float]


@dataclass
class SchedulerConfig:
    margin_tiles: int = 1          # guard ring around a footprint bbox
    max_tiles_per_flush: int = 8   # cap per keyframe (design doc: 4-8)
    budget_ms: float = 500.0       # render time budget per flush
    priority: str = "coverage"     # coverage | newest | round_robin
    tile: int = 512


class OrthoEngine:
    def __init__(self, grid: OrthoGrid, cache: TileCache,
                 render_fn: Callable[[OrthoGrid, int, int], tuple],
                 cfg: Optional[SchedulerConfig] = None):
        self.grid = grid
        self.cache = cache
        self.render_fn = render_fn
        self.cfg = cfg or SchedulerConfig()
        self.tile = int(cfg.tile) if cfg is not None else 512
        self.footprints: List[BBox] = []
        self.covered: Set[Tuple[int, int]] = set()      # tiles ever marked dirty
        self.rendered: Set[Tuple[int, int]] = set()      # tiles rendered at least once
        self.log: List[dict] = []
        self.n_marked = 0

    # ------------------------------------------------------------------
    # marking
    # ------------------------------------------------------------------
    def tiles_of_bbox(self, bbox: BBox, margin_tiles: Optional[int] = None) -> List[Tuple[int, int]]:
        """All tile indices intersecting ``bbox`` (plane frame) + margin ring."""
        m = self.cfg.margin_tiles if margin_tiles is None else margin_tiles
        x0, y0, x1, y1 = bbox
        g = self.grid.gsd
        i0 = int(np.floor(x0 / (self.tile * g))) - m
        i1 = int(np.floor(x1 / (self.tile * g))) + m
        j0 = int(np.floor(y0 / (self.tile * g))) - m
        j1 = int(np.floor(y1 / (self.tile * g))) + m
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def mark_footprint(self, bbox: BBox, kf_id: int = -1, reason: str = "new-kf") -> dict:
        keys = self.tiles_of_bbox(bbox)
        newly = self.cache.mark_dirty(keys, reason=f"{reason}:kf{kf_id}")
        self.covered.update(keys)
        self.footprints.append(bbox)
        self.n_marked += len(keys)
        return {"kf": kf_id, "tiles": len(keys), "newly_dirty": newly,
                "dirty_total": len(self.cache.dirty_keys())}

    def mark_pose_changed(self, kf_ids: Iterable[int], gmap) -> dict:
        """Mark the tiles that contain Gaussians owned by ``kf_ids``."""
        ids = set(int(k) for k in kf_ids)
        owner = gmap.committed["owner"] if gmap.n_committed else np.zeros(0, np.int32)
        means = gmap.committed["means"] if gmap.n_committed else np.zeros((0, 3), np.float32)
        sel = np.isin(owner, np.fromiter(ids, np.int32, len(ids))) if len(owner) else np.zeros(0, bool)
        if sel.sum() == 0:
            return {"kf_ids": sorted(ids), "tiles": 0, "newly_dirty": 0}
        P = (means[sel].astype(np.float64) - self.grid.T_WP[:3, 3]) @ self.grid.T_WP[:3, :3]
        g = self.grid.gsd
        ij = np.floor(P[:, :2] / (self.tile * g)).astype(np.int64)
        # a Gaussian near a tile border can splat into the neighbour -> 1-tile ring
        keys = set()
        for i, j in ij:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    keys.add((int(i) + di, int(j) + dj))
        newly = self.cache.mark_dirty(keys, reason=f"pose:kf{sorted(ids)}")
        self.covered.update(keys)
        return {"kf_ids": sorted(ids), "tiles": len(keys), "newly_dirty": newly,
                "dirty_total": len(self.cache.dirty_keys())}

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _priority_key(self, key: Tuple[int, int]):
        t = self.cache.tiles.get(key)
        never = 0 if (t is None or t.renders == 0) else 1
        ver = t.version if t is not None else 0
        if self.cfg.priority == "newest":
            # prefer tiles near the newest footprint centre
            if self.footprints:
                x0, y0, x1, y1 = self.footprints[-1]
                cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            else:
                cx = cy = 0.0
            g = self.grid.gsd
            d = abs((key[0] + 0.5) * self.tile * g - cx) + abs((key[1] + 0.5) * self.tile * g - cy)
            return (never, ver, d)
        return (never, ver, 0.0)

    def flush(self, budget_ms: Optional[float] = None,
              max_tiles: Optional[int] = None) -> dict:
        """Render dirty tiles in priority order until the budget is exhausted."""
        budget = self.cfg.budget_ms if budget_ms is None else budget_ms
        cap = self.cfg.max_tiles_per_flush if max_tiles is None else max_tiles
        keys = sorted(self.cache.dirty_keys(), key=self._priority_key)
        t_start = time.perf_counter()
        rendered, spent = 0, 0.0
        for key in keys:
            if rendered >= cap or spent >= budget:
                break
            t0 = time.perf_counter()
            rgb, alpha, depth, _ = self.render_fn(self.grid, key[0], key[1])
            ms = (time.perf_counter() - t0) * 1000.0
            self.cache.put(key, rgb, alpha, depth, ms=ms)
            self.rendered.add(key)
            rendered += 1
            spent += ms
        return {"rendered": rendered, "ms": round(spent, 2),
                "wall_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                "dirty_left": len(self.cache.dirty_keys())}

    def flush_all(self, max_iters: int = 200) -> dict:
        """Drain the dirty queue (used at the end of a replay / for tests)."""
        total = 0
        for _ in range(max_iters):
            r = self.flush(budget_ms=1e9, max_tiles=self.cfg.max_tiles_per_flush)
            total += r["rendered"]
            if r["dirty_left"] == 0:
                break
        return {"rendered": total, "dirty_left": len(self.cache.dirty_keys())}

    # ------------------------------------------------------------------
    def coverage_complete(self) -> dict:
        """No missed tiles: everything ever marked must have been rendered once."""
        missing = sorted(self.covered - self.rendered)
        return {"covered": len(self.covered), "rendered": len(self.rendered),
                "missing": len(missing), "missing_keys": missing[:20],
                "complete": len(missing) == 0}

    def stats(self) -> dict:
        return {
            "n_covered": len(self.covered),
            "n_rendered": len(self.rendered),
            "n_marked_total": self.n_marked,
            "dirty_now": len(self.cache.dirty_keys()),
            "cache": self.cache.stats(),
            "cfg": {"margin_tiles": self.cfg.margin_tiles,
                    "max_tiles_per_flush": self.cfg.max_tiles_per_flush,
                    "budget_ms": self.cfg.budget_ms, "tile": self.tile,
                    "priority": self.cfg.priority},
        }
