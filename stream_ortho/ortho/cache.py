"""Tile cache for the streaming ortho back-end (design doc §7.7 item 2).

One entry per ortho tile: the rendered colour/alpha/depth plus a monotonically
increasing ``version`` and a ``dirty`` flag.  The cache is what makes "only
re-render what changed" possible; it also records render counts so the week-6
report can draw a dirty/render heatmap.

Everything is numpy/CPU (tiles are 512x512x3 float32 ~ 3 MB each, so a 20x20
grid is ~1.2 GB at worst; the engine's budget keeps the number small).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TileKey = Tuple[int, int]


@dataclass
class CachedTile:
    rgb: np.ndarray              # (S,S,3) float32 in [0,1]
    alpha: np.ndarray            # (S,S) float32
    depth: np.ndarray            # (S,S) float32
    version: int = 0
    dirty: bool = False
    renders: int = 0
    last_ms: float = 0.0
    dirty_reason: str = ""

    @property
    def shape(self):
        return self.rgb.shape


class TileCache:
    def __init__(self):
        self.tiles: Dict[TileKey, CachedTile] = {}
        self.version = 0
        self.renders = 0
        self.render_ms = 0.0
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.tiles)

    def __contains__(self, key: TileKey):
        return key in self.tiles

    def get(self, key: TileKey) -> Optional[CachedTile]:
        t = self.tiles.get(key)
        if t is None:
            self.misses += 1
        else:
            self.hits += 1
        return t

    def put(self, key: TileKey, rgb: np.ndarray, alpha: np.ndarray, depth: np.ndarray,
            ms: float = 0.0, reason: str = "") -> CachedTile:
        self.version += 1
        old = self.tiles.get(key)
        t = CachedTile(rgb=rgb, alpha=alpha, depth=depth, version=self.version,
                       dirty=False, renders=(old.renders + 1 if old else 1),
                       last_ms=ms, dirty_reason="")
        self.tiles[key] = t
        self.renders += 1
        self.render_ms += ms
        return t

    # ------------------------------------------------------------------
    def mark_dirty(self, keys: Iterable[TileKey], reason: str = "") -> int:
        """Mark tiles dirty (idempotent: returns how many were *newly* dirty)."""
        n = 0
        for k in keys:
            t = self.tiles.get(k)
            if t is None:
                # unknown tile -> remember it as an empty dirty placeholder
                self.tiles[k] = CachedTile(
                    rgb=np.zeros((1, 1, 3), np.float32), alpha=np.zeros((1, 1), np.float32),
                    depth=np.zeros((1, 1), np.float32), version=0, dirty=True,
                    renders=0, dirty_reason=reason)
                n += 1
            elif not t.dirty:
                t.dirty = True
                t.dirty_reason = reason
                n += 1
        return n

    def dirty_keys(self) -> List[TileKey]:
        return [k for k, t in self.tiles.items() if t.dirty]

    def is_dirty(self, key: TileKey) -> bool:
        t = self.tiles.get(key)
        return bool(t and t.dirty)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        n = len(self.tiles)
        rendered = sum(1 for t in self.tiles.values() if t.renders > 0)
        return {
            "n_tiles": n,
            "n_rendered": rendered,
            "n_dirty": len(self.dirty_keys()),
            "renders": self.renders,
            "render_ms_total": round(self.render_ms, 1),
            "render_ms_mean": round(self.render_ms / max(self.renders, 1), 2),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "version": self.version,
        }

    # ------------------------------------------------------------------
    def enforce_capacity(self, n_max: int) -> dict:
        """LRU eviction: drop the least recently rendered clean tiles (week-7)."""
        if n_max <= 0 or len(self.tiles) <= n_max:
            return {"evicted": 0, "n_tiles": len(self.tiles)}
        clean = [(t.version, k) for k, t in self.tiles.items() if not t.dirty]
        clean.sort()
        n_evict = len(self.tiles) - n_max
        evicted = 0
        for _, k in clean[:n_evict]:
            self.tiles.pop(k, None)
            evicted += 1
        return {"evicted": evicted, "n_tiles": len(self.tiles)}

    def mosaic(self, keys: List[TileKey], tile_size: int) -> np.ndarray:
        """Assemble cached tiles into one image (for saving / comparison)."""
        if not keys:
            return np.zeros((tile_size, tile_size, 3), np.float32)
        i0 = min(k[0] for k in keys)
        i1 = max(k[0] for k in keys)
        j0 = min(k[1] for k in keys)
        j1 = max(k[1] for k in keys)
        out = np.zeros(((j1 - j0 + 1) * tile_size, (i1 - i0 + 1) * tile_size, 3), np.float32)
        for (i, j) in keys:
            t = self.tiles.get((i, j))
            if t is None or t.renders == 0:
                continue
            y = (j1 - j) * tile_size     # j grows *up* in the plane, image rows grow down
            x = (i - i0) * tile_size
            h = min(tile_size, t.rgb.shape[0])
            w = min(tile_size, t.rgb.shape[1])
            out[y:y + h, x:x + w] = t.rgb[:h, :w]
        return out
