"""Tile stitching and seam metrics (design doc S4 / §7.4 item 5).

The P1 gate asks for ``seam gradient < 1.5x interior gradient``.  We measure it
on the *stitched mosaic*: mean absolute image gradient across the seam lines
(the tile boundaries) divided by the mean absolute gradient in the tile
interiors (a band away from any seam).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class Tile:
    i: int
    j: int
    rgb: np.ndarray      # (S,S,3) float32 in [0,1]
    alpha: np.ndarray    # (S,S) float32 in [0,1]

    @property
    def size(self) -> int:
        return int(self.rgb.shape[0])


def stitch(tiles: Iterable[Tile], shape: Tuple[int, int] = None,
           origin: Tuple[int, int] = None,
           jmax_anchor: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch tiles into a mosaic; returns ``(rgb, alpha)``.

    ``shape=(rows, cols)`` in *tile* units; if omitted it is inferred from the index ranges.
    ``origin=(imin, jmin)`` anchors the canvas so that *every* frame of a growing map lands on
    the same pixel grid -- without it the canvas is anchored to the current maximum index and
    each frame has a different size (which silently breaks a video writer).
    Empty areas stay black with ``alpha=0``.
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError("no tiles")
    imin = min(t.i for t in tiles) if origin is None else int(origin[0])
    jmin = min(t.j for t in tiles) if origin is None else int(origin[1])
    imax = max(t.i for t in tiles)
    jmax = max(t.j for t in tiles)
    if jmax_anchor is not None:
        jmax = int(jmax_anchor)          # fixed anchor: rows must not move as tiles arrive
    rows = (jmax - jmin + 1) if shape is None else int(shape[0])
    cols = (imax - imin + 1) if shape is None else int(shape[1])
    S = tiles[0].size
    rgb = np.zeros((rows * S, cols * S, 3), np.float32)
    a = np.zeros((rows * S, cols * S), np.float32)
    for t in tiles:
        r = (jmax - t.j) * S           # j grows "up" in the plane, rows grow down
        c = (t.i - imin) * S
        if r < 0 or c < 0 or r + S > rows * S or c + S > cols * S:
            continue                    # a tile outside the anchored canvas is skipped
        rgb[r:r + S, c:c + S] = t.rgb
        a[r:r + S, c:c + S] = t.alpha
    return rgb, a


def seam_gradient_ratio(rgb: np.ndarray, tile: int, band: int = 2,
                        seam_band: int = 1) -> dict:
    """Seam vs interior gradient ratio.

    * vertical seams at columns ``tile, 2*tile, ...``; horizontal at the rows;
    * ``seam``  = mean |gradient| in a ``+-seam_band`` window around those lines;
    * ``interior`` = mean |gradient| in pixels at least ``band`` away from any
      seam (a robust reference).
    """
    g = rgb.mean(axis=-1).astype(np.float64)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    H, W = g.shape
    col_seams = list(range(tile, W, tile))
    row_seams = list(range(tile, H, tile))

    def mask_cols(cols, width):
        m = np.zeros(W - 1, bool)
        for c in cols:
            m[max(0, c - width):min(W - 1, c + width + 1)] = True
        return m

    def mask_rows(rows, width):
        m = np.zeros(H - 1, bool)
        for r in rows:
            m[max(0, r - width):min(H - 1, r + width + 1)] = True
        return m

    seam_cols = mask_cols(col_seams, seam_band)
    seam_rows = mask_rows(row_seams, seam_band)
    # interior: away from seams by `band`
    far_cols = ~mask_cols(col_seams, band)
    far_rows = ~mask_rows(row_seams, band)

    seam = float(np.mean(np.concatenate([gx[:, seam_cols].ravel(), gy[seam_rows, :].ravel()]))) \
        if (seam_cols.any() or seam_rows.any()) else float("nan")
    interior = float(np.mean(np.concatenate([gx[:, far_cols].ravel(), gy[far_rows, :].ravel()]))) \
        if (far_cols.any() or far_rows.any()) else float("nan")
    if np.isfinite(interior) and interior > 1e-9:
        ratio = seam / interior
    else:
        # a perfectly flat mosaic has no gradient anywhere; a zero seam is not a defect
        ratio = 1.0 if (np.isfinite(seam) and seam <= 1e-9) else float("inf")
    return {"seam_gradient": seam, "interior_gradient": interior,
            "ratio": ratio, "n_col_seams": len(col_seams), "n_row_seams": len(row_seams)}


def feather_blend(tiles, tile_size: int, shape=None, feather: int = 8,
                  origin=None, jmax_anchor=None):
    """Optional cosine feather across tile borders (visualisation only).

    The docstring used to claim the tiles overlap by ``halo``.  They do **not**: streaming
    tiles are exact 512x512 blocks with no overlap, so every mosaic pixel is covered by a
    single tile and the border ring would get weight 0 -> ``0 / 1e-6 = 0`` -> a black cross
    wherever a border crosses content.  Two guards make it safe for both cases:

    * the cosine ramp is floored at 0.25, so a border pixel never loses all of its weight;
    * wherever the accumulated weight is still zero, the plain stitch is used instead.

    ``origin``/``shape``/``jmax_anchor`` must match the caller's :func:`stitch` arguments so
    the blend lands on the same (possibly anchored, larger) canvas.
    """
    base_rgb, base_a = stitch(tiles, shape=shape, origin=origin, jmax_anchor=jmax_anchor)
    if feather <= 0:
        return base_rgb
    tiles = list(tiles)
    if not tiles:
        raise ValueError("no tiles")
    imin = min(t.i for t in tiles) if origin is None else int(origin[0])
    jmax = max(t.j for t in tiles)
    if jmax_anchor is not None:
        jmax = int(jmax_anchor)          # same anchor as stitch: rows must not move
    rows, cols = base_rgb.shape[:2]
    S = int(tile_size)
    acc = np.zeros((rows, cols, 3), np.float32)
    wsum = np.zeros((rows, cols, 1), np.float32)
    r = np.ones(S, np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, feather, dtype=np.float32))
    ramp = np.maximum(ramp, 0.25)                 # never a zero weight
    r[:feather] = ramp
    r[-feather:] = ramp[::-1]
    w2 = np.outer(r, r)[..., None]
    for t in tiles:
        rr = (jmax - t.j) * S
        cc = (t.i - imin) * S
        if rr < 0 or cc < 0 or rr + S > rows or cc + S > cols:
            continue                               # outside the canvas, like stitch
        acc[rr:rr + S, cc:cc + S] += t.rgb * w2
        wsum[rr:rr + S, cc:cc + S] += w2
    out = acc / np.maximum(wsum, 1e-6)
    return np.where(wsum > 1e-6, out, base_rgb)
