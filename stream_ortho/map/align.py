"""Frame-to-map alignment: make a new keyframe's patch land on the existing map.

Why this exists (week8-end-18/19): the voxel dedup never fired (``merges = 0``)
because two frames' point sets overlap in only 0.0-2.0% of voxels -- every frame's
depth surface has a slightly different *shape* and is offset by metres, so the map
piled the frames up instead of fusing them.  Before ingesting a new patch we
therefore estimate a small correction so that its points fall onto the map:

* **lateral (dx, dy)** in the plane frame, from phase correlation between a top-down
  density image of the patch's ground points and the same image built from the map,
* **depth (a, b)**: ``d <- a*d + b`` fitted on the ground against the DSM (the caller
  can pass ``affine_fix=True`` to :func:`head_adapt_patch`, which does the same fit
  *before* the placement; here it is only used to report the residual).

The lateral shift is applied to the patch means in the plane frame, so the geometry
stays a rigid translation (no stretching) and the colours are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignResult:
    dx: float
    dy: float
    ok: bool
    score: float
    reason: str = ""


def _ground_density(points_plane: np.ndarray, x0: float, y0: float, gsd: float,
                    nx: int, ny: int, z_max: float = 2.0) -> np.ndarray:
    """Top-down occupancy image of the ground points (plane frame, metres)."""
    import cv2

    ok = np.isfinite(points_plane).all(1) & (points_plane[:, 2] < z_max)
    p = points_plane[ok]
    img = np.zeros((ny, nx), np.float32)
    if len(p) == 0:
        return img
    i = np.clip(((p[:, 0] - x0) / gsd).astype(np.int64), 0, nx - 1)
    j = np.clip(((p[:, 1] - y0) / gsd).astype(np.int64), 0, ny - 1)
    np.add.at(img, (j, i), 1.0)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def align_patch_to_map(gmap, patch, T_WP: np.ndarray, gsd: float = 0.5,
                       max_shift_m: float = 8.0, z_max: float = 2.0) -> AlignResult:
    """Estimate the (dx, dy) that best registers ``patch`` onto the map's ground."""
    import cv2

    c = gmap.committed
    if c["means"].shape[0] < 5000:
        return AlignResult(0.0, 0.0, False, 0.0, "map-too-small")

    m_plane = (c["means"] - T_WP[:3, 3]) @ T_WP[:3, :3]
    p_plane = (patch.means - T_WP[:3, 3]) @ T_WP[:3, :3]
    lo = np.minimum(m_plane.min(0), p_plane.min(0))[:2]
    hi = np.maximum(m_plane.max(0), p_plane.max(0))[:2]
    nx = int(np.ceil((hi[0] - lo[0]) / gsd)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / gsd)) + 1
    if nx * ny > 4_000_000 or nx < 8 or ny < 8:
        return AlignResult(0.0, 0.0, False, 0.0, "bad-extent")

    A = _ground_density(p_plane, lo[0], lo[1], gsd, nx, ny, z_max)   # new frame
    B = _ground_density(m_plane, lo[0], lo[1], gsd, nx, ny, z_max)   # current map
    if A.sum() < 100 or B.sum() < 100:
        return AlignResult(0.0, 0.0, False, 0.0, "no-ground")

    # phase correlation wants zero-mean, non-negative-ish inputs
    a = A - A.mean()
    b = B - B.mean()
    (sx, sy), resp = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64))
    dx, dy = float(sx) * gsd, float(sy) * gsd
    if not np.isfinite(dx) or not np.isfinite(dy):
        return AlignResult(0.0, 0.0, False, float(resp), "nan")
    if abs(dx) > max_shift_m or abs(dy) > max_shift_m:
        return AlignResult(0.0, 0.0, False, float(resp), "shift-too-large")
    return AlignResult(dx, dy, True, float(resp))


def apply_shift(patch, dx: float, dy: float, T_WP: np.ndarray):
    """Translate the patch by (dx, dy) metres in the plane frame (in place)."""
    w = T_WP[:3, :3] @ np.array([dx, dy, 0.0], np.float64)
    patch.means = (patch.means + w.reshape(1, 3)).astype(np.float32)
    return patch


def voxel_overlap(means_a: np.ndarray, means_b: np.ndarray, voxel: float = 0.25,
                  sample: int = 20000) -> float:
    """Fraction of ``a``'s voxels that also contain a point of ``b`` (diagnostic)."""
    if len(means_a) == 0 or len(means_b) == 0:
        return 0.0
    a = np.floor(means_a[:: max(1, len(means_a) // sample)] / voxel).astype(np.int64)
    b = np.floor(means_b[:: max(1, len(means_b) // sample)] / voxel).astype(np.int64)
    sa = {tuple(x) for x in a}
    sb = {tuple(x) for x in b}
    return len(sa & sb) / max(len(sa), 1)


# ---------------------------------------------------------------------------
def validate_shift(patch_means, map_means, T_WP, dx: float, dy: float,
                   tol: float = 0.5, sample: int = 4000) -> dict:
    """Inlier fraction of the patch's ground points near a map ground point.

    Used as the **validation** step of the alignment (On-the-Fly3R style): a shift is
    only accepted when it actually improves the number of points that have a map
    neighbour, not merely because phase correlation produced a number.
    """
    def ground(pts):
        p = (np.asarray(pts, np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
        return p[(p[:, 2] > -20.0) & (p[:, 2] < 2.0)][:, :2]

    a = ground(patch_means)
    b = ground(map_means)
    if len(a) < 100 or len(b) < 100:
        return {"inlier": 0.0, "before": 0.0, "gain": 0.0, "n": 0}
    step = max(1, len(a) // sample)
    a = a[::step] + np.array([dx, dy])
    bstep = max(1, len(b) // (sample * 4))
    b = b[::bstep]
    # nearest neighbour via a coarse grid
    cell = tol
    keys = set((int(x / cell), int(y / cell)) for x, y in b)
    hit = 0
    for x, y in a:
        kx, ky = int(x / cell), int(y / cell)
        if any((kx + ox, ky + oy) in keys for ox in (-1, 0, 1) for oy in (-1, 0, 1)):
            hit += 1
    after = hit / len(a)
    a0 = ground(patch_means)[::step]
    hit0 = 0
    for x, y in a0:
        kx, ky = int(x / cell), int(y / cell)
        if any((kx + ox, ky + oy) in keys for ox in (-1, 0, 1) for oy in (-1, 0, 1)):
            hit0 += 1
    before = hit0 / len(a0)
    return {"inlier": float(after), "before": float(before),
            "gain": float(after - before), "n": int(len(a))}


def align_and_validate(gmap, patch, T_WP, gsd=0.5, max_shift_m=8.0,
                       min_response: float = 0.2, min_gain: float = 0.05) -> dict:
    """Phase-correlation shift + validation; returns the accepted (dx, dy) or zeros."""
    r = align_patch_to_map(gmap, patch, T_WP, gsd=gsd, max_shift_m=max_shift_m)
    if not r.ok:
        return {"accepted": False, "dx": 0.0, "dy": 0.0, "reason": r.reason,
                "response": r.score}
    v = validate_shift(patch.means, gmap.committed["means"], T_WP, r.dx, r.dy)
    ok = (r.score >= min_response) and (v["gain"] >= min_gain)
    return {"accepted": bool(ok), "dx": float(r.dx), "dy": float(r.dy),
            "response": float(r.score), "inlier_before": v["before"],
            "inlier_after": v["inlier"], "gain": v["gain"],
            "reason": "ok" if ok else ("low-response" if r.score < min_response else "no-gain")}


def align_photometric(gmap, patch, grid, i: int, j: int, max_shift: float = 3.0,
                      coarse: float = 1.5, fine: float = 0.5) -> dict:
    """Dense-photometric 3-DoF alignment of a patch to the current map (P2').

    Phase correlation on a 0.5 m density raster was measured to be unreliable
    (response 0.01-0.02), so this replaces it: the patch is re-rendered at a few
    candidate lateral shifts and the shift that maximises the **normalised cross
    correlation** with the map's render of the same tile is kept -- provided it is
    within ``max_shift`` and improves the NCC by at least 2%.

    Returns ``{"ok", "dx", "dy", "ncc_before", "ncc_after", "n_evals"}``.
    """
    import torch

    from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile

    def _ncc(a, b, m):
        a = a[m]; b = b[m]
        a = a - a.mean(); b = b - b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-9
        return float((a * b).sum() / d)

    def _cloud(p):
        dev = torch.device("cuda:0")
        return GaussianCloud(
            means=torch.as_tensor(p.means, device=dev),
            quats=torch.as_tensor(p.quats, device=dev),
            scales=torch.as_tensor(p.scales, device=dev),
            opacity=torch.as_tensor(p.opacity, device=dev),
            colors=torch.as_tensor(p.colors, device=dev),
            owner=torch.zeros(len(p), dtype=torch.long, device=dev), meta={})

    map_rgb, map_al, _, _ = render_tile(gmap.to_cloud(), grid, i, j)
    base_m = map_al > 0.5
    if base_m.sum() < 5000:
        return {"ok": False, "dx": 0.0, "dy": 0.0, "ncc_before": 0.0,
                "ncc_after": 0.0, "n_evals": 0, "reason": "map-empty"}

    orig = patch.means.copy()
    best = {"ncc": -1.0, "dx": 0.0, "dy": 0.0}
    n = 0

    def try_shift(dx, dy):
        nonlocal n
        delta = (grid.T_WP[:3, :3] @ np.array([dx, dy, 0.0])).reshape(1, 3)
        patch.means = (orig + delta).astype(np.float32)
        rgb, al, _, _ = render_tile(_cloud(patch), grid, i, j)
        m = base_m & (al > 0.5)
        n += 1
        if m.sum() < 5000:
            return -1.0
        g1 = rgb.mean(-1); g2 = map_rgb.mean(-1)
        return _ncc(g1, g2, m)

    ncc0 = try_shift(0.0, 0.0)
    best.update(ncc=ncc0)
    steps = np.arange(-max_shift, max_shift + 1e-9, coarse)
    for dx in steps:
        for dy in steps:
            v = try_shift(float(dx), float(dy))
            if v > best["ncc"]:
                best.update(ncc=v, dx=float(dx), dy=float(dy))
    for dx in np.arange(best["dx"] - coarse, best["dx"] + coarse + 1e-9, fine):
        for dy in np.arange(best["dy"] - coarse, best["dy"] + coarse + 1e-9, fine):
            v = try_shift(float(dx), float(dy))
            if v > best["ncc"]:
                best.update(ncc=v, dx=float(dx), dy=float(dy))
    patch.means = orig
    ok = (abs(best["dx"]) <= max_shift and abs(best["dy"]) <= max_shift
          and best["ncc"] >= ncc0 * 1.02)
    return {"ok": bool(ok), "dx": best["dx"], "dy": best["dy"],
            "ncc_before": float(ncc0), "ncc_after": float(best["ncc"]),
            "n_evals": int(n), "reason": "ok" if ok else "no-gain"}
