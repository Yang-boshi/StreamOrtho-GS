"""Stitch the scheduler's tile cache into a mosaic PNG.

The cache holds whatever has been rendered so far, so dumping it every N frames produces the
"tiles appear as the flight proceeds" sequence the system is meant to show.
"""
from __future__ import annotations

import pathlib

import numpy as np

from stream_ortho.ortho.tiles import Tile, feather_blend, stitch


def cache_to_tiles_fixed(cache, origin, shape):
    return cache_to_tiles(cache)


def cache_to_tiles(cache):
    tiles = []
    for (i, j), t in cache.tiles.items():
        if t is None or getattr(t, "renders", 0) == 0:
            continue
        tiles.append(Tile(i=int(i), j=int(j), rgb=np.asarray(t.rgb, np.float32),
                          alpha=np.asarray(t.alpha, np.float32)))
    return tiles


def content_crop(rgb, alpha, thr: float = 0.0):
    """Crop canvas-aligned rgb/alpha to the alpha>``thr`` content bounding box.

    Returns ``(rgb_crop, (r0, r1, c0, c1))`` with the box in FULL-canvas
    coordinates (the same convention the update-video union placement uses).
    """
    rgb = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    occ = np.nonzero(a > thr)
    if occ[0].size == 0:
        return rgb, (0, rgb.shape[0], 0, rgb.shape[1])
    r0, r1 = int(occ[0].min()), int(occ[0].max()) + 1
    c0, c1 = int(occ[1].min()), int(occ[1].max()) + 1
    return rgb[r0:r1, c0:c1], (r0, r1, c0, c1)


def save_mosaic(cache, out_path, tile_size: int = 512, feather: int = 0, crop: bool = False,
                scale: float = 1.0, origin=None, shape=None, crop_box=None,
                jmax_anchor=None):
    """Write an RGBA mosaic; returns ``{tiles, shape}`` or ``None`` when nothing is rendered."""
    from PIL import Image
    tiles = cache_to_tiles(cache)
    if not tiles:
        return None
    if feather:
        rgb = feather_blend(tiles, tile_size, feather=feather, shape=shape,
                            origin=origin, jmax_anchor=jmax_anchor)
        _, a = stitch(tiles, shape=shape, origin=origin, jmax_anchor=jmax_anchor)
    else:
        rgb, a = stitch(tiles, shape=shape, origin=origin, jmax_anchor=jmax_anchor)
    rgb = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    a = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
    rgba = np.concatenate([rgb, a[..., None]], -1)
    box = None
    if crop_box is not None:
        # a *shared* box for every frame: the same pixel window on each update, so the video
        # writer sees a constant size
        r0, r1, c0, c1 = [int(v) for v in crop_box]
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, rgba.shape[0]), min(c1, rgba.shape[1])
        rgba = rgba[r0:r1, c0:c1]
        rgb = rgb[r0:r1, c0:c1]
        a = a[r0:r1, c0:c1]
        box = (r0, r1, c0, c1)
    elif crop:
        # the canvas is the covered-tile bounding box, which is mostly empty for a line
        # flight; cropping to the content keeps the real pixels and the real resolution
        occ = np.nonzero(a > 0)
        if occ[0].size:
            r0, r1 = int(occ[0].min()), int(occ[0].max()) + 1
            c0, c1 = int(occ[1].min()), int(occ[1].max()) + 1
            rgba = rgba[r0:r1, c0:c1]
            rgb = rgb[r0:r1, c0:c1]
            a = a[r0:r1, c0:c1]
            box = (r0, r1, c0, c1)
    sc = float(scale) if scale else 1.0
    if sc != 1.0:
        from PIL import Image as _Im
        h, w = rgba.shape[:2]
        rgba = np.asarray(_Im.fromarray((rgba * 255).astype(np.uint8), "RGBA").resize(
            (max(int(w * sc), 1), max(int(h * sc), 1)), _Im.BILINEAR), np.float32) / 255.0
        rgb, a = rgba[..., :3], rgba[..., 3]
        if box is not None:
            # express the box in the *saved* image's coordinates (PIL truncates with int())
            r0 = int(box[0] * sc)
            c0 = int(box[2] * sc)
            box = (r0, r0 + rgba.shape[0], c0, c0 + rgba.shape[1])
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA").save(p)
    return {"tiles": len(tiles), "shape": [int(rgb.shape[0]), int(rgb.shape[1])],
            "covered_frac": float((a > 0.0).mean()), "path": str(p),
            "crop_box": [int(v) for v in box] if box is not None else None}
