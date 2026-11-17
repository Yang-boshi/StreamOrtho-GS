"""Floaters on the product grid (C2 evidence, week9-08).

A cell is a **floater** when it has no support (one observation) and its height is far
from its neighbourhood's supported surface:

    support   = observations of the cell in the top layer (n_obs for A, or 1/2 decoded
                from the product opacity, which is min(1, n/2))
    bad       = support < 2  and  |z - median5(z_supported)| > h_tol

Reported as bad cells, connected clusters (8-connectivity) and the cluster-size p90.
"""
from __future__ import annotations

import numpy as np


def grid_top(means, cell: float, support=None):
    """(origin, W, H, z_top, sup_top) on a ``cell``-sized grid."""
    P = np.asarray(means, np.float64)
    ix = np.floor(P[:, 0] / cell).astype(np.int64)
    iy = np.floor(P[:, 1] / cell).astype(np.int64)
    x0, y0 = int(ix.min()), int(iy.min())
    W = int(ix.max()) - x0 + 1
    H = int(iy.max()) - y0 + 1
    z = np.full((H, W), -np.inf, np.float64)
    sup = np.zeros((H, W), np.int32)
    sup_arr = np.ones(len(P), np.int32) if support is None else np.asarray(support, np.int32)
    flat = (iy - y0) * W + (ix - x0)
    np.maximum.at(z.reshape(-1), flat, P[:, 2])
    np.maximum.at(sup.reshape(-1), flat, sup_arr)
    return (x0, y0, W, H, z, sup)


def support_from_opacity(opacity, floor: float = 0.9):
    """Product opacity is min(1, n/2): 0.5 -> 1 observation, >=floor -> 2 or more."""
    o = np.asarray(opacity, np.float32)
    return np.where(o >= floor, 2, 1).astype(np.int32)


def bad_mask(z, sup, cell: float, h_tol: float = 1.5):
    from scipy.ndimage import median_filter

    occ = np.isfinite(z)
    zs = np.where(occ & (sup >= 2), z, np.nan)
    if not np.isfinite(zs).any():
        return np.zeros_like(z, bool)
    fill = np.nanmedian(zs)
    zf = np.where(np.isfinite(zs), zs, fill)
    med = median_filter(zf, size=5, mode="nearest")
    return occ & (sup < 2) & (np.abs(z - med) > h_tol)


def clusters(mask: np.ndarray):
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas > 0]
    return int(n - 1), areas


def summarize(means, cell: float, support=None, h_tol: float = 1.5) -> dict:
    x0, y0, W, H, z, sup = grid_top(means, cell, support)
    bad = bad_mask(z, sup, cell, h_tol)
    n_clu, areas = clusters(bad)
    return {"bad_cells": int(bad.sum()), "clusters": n_clu,
            "cluster_p90": float(np.percentile(areas, 90)) if areas.size else 0.0,
            "cluster_max": int(areas.max()) if areas.size else 0,
            "cells_total": int(np.isfinite(z).sum()), "h_tol": h_tol}
