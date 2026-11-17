"""Field-of-view helpers: the bbox a frame covers and the tiles it touches.

A point-cloud bbox is conservative (it can only be larger than the true frustum footprint) and
therefore can never miss a tile, which is the property the scheduler needs.
"""
from __future__ import annotations

import numpy as np


def footprint_bbox(points_plane, margin_m: float = 0.0):
    """``(x0, y0, x1, y1)`` of a frame's plane points, expanded by ``margin_m``."""
    p = np.asarray(points_plane, np.float64)
    if p.size == 0:
        raise ValueError("empty point set")
    x, y = p[:, 0], p[:, 1]
    return (float(x.min()) - margin_m, float(y.min()) - margin_m,
            float(x.max()) + margin_m, float(y.max()) + margin_m)


def fov_tiles(engine, bbox, margin_tiles: int = 1):
    """Tile indices touched by ``bbox`` (plus a guard ring)."""
    return engine.tiles_of_bbox(bbox, margin_tiles=margin_tiles)
