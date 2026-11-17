"""Laplacian pyramid blending for the P5 multi-keyframe ortho mosaic (P5.1b Task 4).

Why not plain alpha feathering: keyframe tiles overlap with slightly different
exposure / geometry; a linear cross-fade leaves a visible brightness step where
the masks meet.  Blending at several pyramid scales smooths the LOW-frequency
transition while keeping each image's HIGH-frequency detail.

Inputs are canvas-aligned (H, W, 3) RGBs and (H, W) float masks; the caller
(accumulation) guarantees identical shapes, so no per-image resizing happens
here.  Masks are feathered with a distance transform: pixels near a mask's
boundary get a lower weight than its interior, which removes the hard step at
the coverage edge.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

_GAUSS = None


def _gauss_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    global _GAUSS
    if _GAUSS is None or _GAUSS.shape[0] != ksize:
        _GAUSS = cv2.getGaussianKernel(ksize, sigma)
    return _GAUSS


def _downsample(img: np.ndarray) -> np.ndarray:
    """Blur with the 5-tap kernel, then keep every 2nd pixel."""
    k = _gauss_kernel()
    if img.ndim == 3:
        img = cv2.sepFilter2D(img, -1, k, k, borderType=cv2.BORDER_REFLECT)
    else:
        img = cv2.sepFilter2D(img, -1, k, k, borderType=cv2.BORDER_REFLECT)
    return img[::2, ::2]


def _upsample(img: np.ndarray, shape) -> np.ndarray:
    """DC-exact doubling (bilinear).  The classic zero-insert + 4x-blur expand
    self-consistently round-trips its own pyramid but multiplies the DC of any
    *blended* level set by 4 per level, destroying the reconstruction."""
    h, w = shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(_downsample(pyr[-1]))
    return pyr


def laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    """L[i] = G[i] - upsample(G[i+1]); the last level is the coarsest Gaussian."""
    g = gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels - 1):
        up = _upsample(g[i + 1], g[i].shape[:2])
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def reconstruct_from_laplacian(pyr: list) -> np.ndarray:
    img = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        img = _upsample(img, pyr[i].shape[:2]) + pyr[i]
    return img


def feather_mask(mask: np.ndarray, feather_px: int = 24) -> np.ndarray:
    """Distance-transform feathering: interior stays 1, boundary ramps to 0.

    ``mask`` float (H, W) in [0, 1].  The ramp width ``feather_px`` is measured
    from the mask edge inwards; 0 returns the mask unchanged.
    """
    m = (np.asarray(mask, np.float32) > 0.5).astype(np.uint8)
    if feather_px <= 0 or m.sum() == 0:
        return np.asarray(mask, np.float32) * (m > 0)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    w = np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32)
    return w


def laplacian_blend(images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                    levels: int = 4, feather_px: int = 24) -> tuple:
    """Blend canvas-aligned keyframe renders.

    Returns ``(rgb, weight_sum)``: the blended mosaic and the total feathered
    weight (its >0 region is the coverage; pixels where it is 0 stay black).
    """
    if len(images) != len(masks):
        raise ValueError("images/masks length mismatch")
    if not images:
        raise ValueError("no images")
    h, w = np.asarray(images[0]).shape[:2]
    for im, mk in zip(images, masks):
        if np.asarray(im).shape[:2] != (h, w) or np.asarray(mk).shape[:2] != (h, w):
            raise ValueError("all images/masks must share one canvas shape")
    return laplacian_blend_accumulate(zip(images, masks), (h, w), levels, feather_px)


def laplacian_blend_accumulate(tiles, shape, levels: int = 4, feather_px: int = 24) -> tuple:
    """Streaming form of :func:`laplacian_blend` for many tiles.

    ``tiles`` is an iterable of ``(rgb, alpha)`` canvas-aligned pairs; only the
    per-level accumulators (canvas-sized) are held, never all tiles' pyramids
    at once -- 200 keyframes x ~600px tiles would otherwise need several GB.
    """
    h, w = int(shape[0]), int(shape[1])
    num: list = [None] * levels
    den: list = [None] * levels
    wsum = np.zeros((h, w), np.float32)
    n = 0
    for im, mk in tiles:
        if np.asarray(im).shape[:2] != (h, w) or np.asarray(mk).shape[:2] != (h, w):
            raise ValueError("all images/masks must share one canvas shape")
        wt = feather_mask(mk, feather_px)
        lp_i = laplacian_pyramid(np.asarray(im, np.float32), levels)
        lp_w = gaussian_pyramid(wt, levels)
        for i in range(levels):
            term = lp_i[i] * lp_w[i][..., None]
            if num[i] is None:
                num[i], den[i] = np.zeros_like(term), np.zeros_like(lp_w[i])
            num[i] += term
            den[i] += lp_w[i]
        wsum += wt
        n += 1
    if n == 0:
        raise ValueError("no tiles")
    blended_pyr = [num[i] / np.maximum(den[i], 1e-6)[..., None] for i in range(levels)]
    rgb = reconstruct_from_laplacian(blended_pyr)
    return rgb, wsum
