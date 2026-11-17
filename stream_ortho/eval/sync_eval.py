"""E1 -- the relative-synchronisation death experiment (``refs/06`` §P0-3, ``04-P0-3``).

Question: does a **frame-to-frame relative** deformation model beat a **frame-to-product
absolute** correction on held-out frames?

Five methods, identical folds, identical product builder, identical evaluation:

    raw        no correction; product = build frames' raw z
    abs_t3     each held frame fitted to the raw product's near layer (3-DOF)
    abs_s7     same with a depth scale (s in [0.9, 1.1])
    sync_t3    relative sync over the build frames (3-DOF, frame-cell graph), product rebuilt
    field4x4   each held frame fitted to a 16-coefficient bilinear field

Protocol (do not mix): 5 contiguous folds of the keyframe order; product and ``eval_cells``
come from the **build** frames only; held-out frames only ever fit *themselves*.

The 3-DOF sync is ill-conditioned on a chained graph (frames with disjoint support make
``b`` and ``tx``/``ty`` nearly collinear): an unregularised solve reaches the same pairwise
residual with ``b`` blowing up to +/- 23 m and the pooled height IQR doubling.  ``SyncConfig.ridge``
is therefore a **conditioning prior** (residual vs ridge is reported), not a tuning knob for
the gate.

Usage::

    python -m stream_ortho.eval.sync_eval --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-sync/ieu --folds 5 [--head-obs ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np

Z_GAP = 2.0
MAX_LAYERS = 3


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class SyncConfig:
    min_shared: int = 500
    max_dz: float = 3.0
    huber_iters: int = 4
    max_pairs: int = 2000
    ridge: float = 1e-3
    gauge: int = 0


# --------------------------------------------------------------------------------------
# loading / indexing
# --------------------------------------------------------------------------------------
def load_obs(path) -> dict:
    z = np.load(pathlib.Path(path) / "obs_merged.npz")
    d = {k: z[k] for k in ("key", "kf", "z", "x", "y", "conf", "cos_inc", "n", "z_std",
                           "layer", "rgb")}
    d["cell"] = float(z["cell"])
    d["x"] = d["x"].astype(np.float64)
    d["y"] = d["y"].astype(np.float64)
    d["z"] = d["z"].astype(np.float64)
    d["conf"] = d["conf"].astype(np.float64)
    d["cos_inc"] = d["cos_inc"].astype(np.float64)
    return d


def frame_slices(kf):
    """Row slices per keyframe (the table is frame-major)."""
    order = np.argsort(kf, kind="stable")
    uq, start = np.unique(kf[order], return_index=True)
    cnt = np.diff(np.r_[start, order.size])
    return {int(k): order[s:s + c] for k, s, c in zip(uq, start, cnt)}


def single_layer_cells(idx_key):
    k, c = np.unique(idx_key, return_counts=True)
    return k[c == 1]


def _intersect_sorted(a, b):
    """Intersection of two sorted int64 arrays (searchsorted, no re-sort)."""
    if b.size and b[0] > b[-1]:
        b = b[::-1]
    pos = np.searchsorted(b, a)
    pos = np.clip(pos, 0, max(b.size - 1, 0))
    hit = b[pos] == a
    return a[hit]


def _lookup(sorted_key, sorted_val, keys):
    pos = np.searchsorted(sorted_key, keys)
    return sorted_val[pos]


# --------------------------------------------------------------------------------------
# product
# --------------------------------------------------------------------------------------
def build_product(ix, iy, key, z, w, n, cell, bbox, L=MAX_LAYERS, z_gap=Z_GAP, kf=None,
                  rgb=None, return_assign=False):
    """Aggregate ``(key, layer)`` rows of the build frames into a dense ``(H, W, L)`` grid.

    ``kf`` (optional, per row) records the **owner** of every cell-layer: the frame that
    contributed the largest weight.  SEI needs it -- a seam is a neighbour pair whose chosen
    layer comes from different frames.
    """
    x0, y0, x1, y1 = bbox
    ox, oy = int(np.floor(x0 / cell)), int(np.floor(y0 / cell))
    W = int(np.ceil(x1 / cell)) - ox + 1
    H = int(np.ceil(y1 / cell)) - oy + 1

    order = np.lexsort((z, key))
    ks, zs, ws, ns = key[order], z[order], w[order], n[order]
    new = np.r_[True, (ks[1:] != ks[:-1]) | (np.diff(zs) > z_gap)]
    gid = np.cumsum(new) - 1
    start = np.r_[0, np.nonzero(np.diff(gid))[0] + 1]
    cnt = np.diff(np.r_[start, gid.size])

    sw = np.add.reduceat(ws, start)
    swz = np.add.reduceat(ws * zs, start)
    swz2 = np.add.reduceat(ws * zs * zs, start)
    sn = np.add.reduceat(ns.astype(np.float64), start)
    z_ref = swz / np.maximum(sw, 1e-12)

    gkey = ks[start]
    key_change = np.r_[True, gkey[1:] != gkey[:-1]]
    key_id = np.cumsum(key_change) - 1
    n_per_key = np.bincount(key_id)
    first_of_key = np.nonzero(key_change)[0]
    idx_in_key = np.arange(gkey.size) - first_of_key[key_id]
    keep = (n_per_key[key_id] - 1 - idx_in_key) < L
    slot = (n_per_key[key_id] - 1 - idx_in_key)[keep]

    # all rows of a group share one key, so take the first row's index (reduceat would SUM
    # them and push almost every cell outside the bbox)
    _ixs = np.asarray(ix, np.int64)[order]
    _iys = np.asarray(iy, np.int64)[order]
    gix = _ixs[start][keep]
    giy = _iys[start][keep]
    j, i = giy - oy, gix - ox
    inside = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    flat = (j * W + i)[inside] * L + slot[inside]

    n_grid = np.zeros(H * W * L, np.float64)
    z_grid = np.zeros(H * W * L, np.float64)
    sw_grid = np.zeros(H * W * L, np.float64)
    swz_grid = np.zeros(H * W * L, np.float64)
    swz2_grid = np.zeros(H * W * L, np.float64)
    np.add.at(n_grid, flat, sn[keep][inside])
    np.add.at(sw_grid, flat, sw[keep][inside])
    np.add.at(swz_grid, flat, swz[keep][inside])
    np.add.at(swz2_grid, flat, swz2[keep][inside])
    np.add.at(z_grid, flat, swz[keep][inside])
    z_ref = np.where(sw_grid > 0, z_grid / np.maximum(sw_grid, 1e-12), 0.0)

    rgb_grid = np.zeros((H * W * L, 3), np.float64)
    if rgb is not None:
        rgbs = np.asarray(rgb, np.float64)[order]
        for ch in range(3):
            np.add.at(rgb_grid[:, ch], flat, np.add.reduceat(ws * rgbs[:, ch], start)[keep][inside])
    # per-row assignment: which (cell, slot) each observation was merged into.  Freezing this
    # is what lets a later round update z WITHOUT re-grouping layers or re-electing owners.
    assign = None
    if return_assign:
        flat_per_group = np.full(gkey.size, -1, np.int64)
        flat_per_group[keep] = flat
        assign = np.full(key.size, -1, np.int64)
        assign[order] = flat_per_group[gid]

    owner_grid = np.full(H * W * L, -1, np.int64)
    if kf is not None:
        kf_s = np.asarray(kf, np.int64)[order]
        o2 = np.lexsort((ws, gid))                    # group major, weight minor
        g2 = gid[o2]
        s2 = np.r_[0, np.nonzero(np.diff(g2))[0] + 1]
        c2 = np.diff(np.r_[s2, g2.size])
        rep = o2[s2 + c2 - 1]                          # max-weight row of each group
        owner_grid[flat] = kf_s[rep][keep][inside]   # one owner per (cell, slot)

    return {
        "n": n_grid.reshape(H, W, L), "z_ref": z_ref.reshape(H, W, L),
        "owner": owner_grid.reshape(H, W, L),
        "rgb": np.where(sw_grid[:, None] > 0,
                        rgb_grid / np.maximum(sw_grid[:, None], 1e-12), 0.0
                        ).reshape(H, W, L, 3),
        "sum_w": sw_grid.reshape(H, W, L), "sum_wz": swz_grid.reshape(H, W, L),
        "sum_wz2": swz2_grid.reshape(H, W, L),
        "x0": ox * cell, "y0": oy * cell, "cell": cell, "W": W, "H": H, "L": L,
        "ox": ox, "oy": oy, "n_overflow_layers": int((~keep).sum()),
        "assign": assign,
    }


def product_layers_at(prod, ix, iy):
    j, i = iy - prod["oy"], ix - prod["ox"]
    ok = (i >= 0) & (i < prod["W"]) & (j >= 0) & (j < prod["H"])
    out_n = np.zeros((ix.size, prod["L"]))
    out_z = np.zeros((ix.size, prod["L"]))
    idx = np.nonzero(ok)[0]
    out_n[idx] = prod["n"][j[idx], i[idx]]
    out_z[idx] = prod["z_ref"][j[idx], i[idx]]
    return out_n, out_z, ok


def eval_cells_of(prod) -> np.ndarray:
    n = prod["n"]
    occ = n > 0
    sw = np.maximum(prod["sum_w"], 1e-12)
    var = np.maximum(prod["sum_wz2"] / sw - (prod["sum_wz"] / sw) ** 2, 0.0)
    idx = occ.argmax(-1)
    sig = np.take_along_axis(var, idx[..., None], -1)[..., 0]
    sup = np.take_along_axis(n, idx[..., None], -1)[..., 0]
    flat = (occ.sum(-1) == 1) & (np.sqrt(sig) <= 0.3) & (sup >= 2)
    jj, ii = np.nonzero(flat)
    from stream_ortho.map.selection import pack_cell
    # sorted: `_intersect_sorted` requires it, and pack_cell's key order is not the
    # (row, col) order that np.nonzero produces
    return np.sort(pack_cell(ii + prod["ox"], jj + prod["oy"]))


# --------------------------------------------------------------------------------------
# HFR / PC
# --------------------------------------------------------------------------------------
def hfr_rows(rows, prod, tail=3.0) -> dict:
    """HFR of a set of rows (one frame) against the product, **without truncation**."""
    from stream_ortho.eval.hfr import hfr
    from stream_ortho.map.selection import unpack_cell

    n_prod = prod["n"].reshape(-1, prod["L"])
    z_prod = prod["z_ref"].reshape(-1, prod["L"])
    ix, iy = unpack_cell(rows["key"])
    i, j = ix - prod["ox"], iy - prod["oy"]
    ok = (i >= 0) & (i < prod["W"]) & (j >= 0) & (j < prod["H"])
    if not ok.any():
        return {"near": np.zeros(0), "top": np.zeros(0), "n": 0, "outside": int(rows["key"].size)}
    cell = (j[ok] * prod["W"] + i[ok])
    occ = n_prod[cell] > 0
    zz = z_prod[cell]
    zo = rows["z"][ok]
    has = occ.any(-1)
    cell, occ, zz, zo = cell[has], occ[has], zz[has], zo[has]
    if zo.size == 0:
        return {"near": np.zeros(0), "top": np.zeros(0), "n": 0, "outside": int(rows["key"].size)}
    z_top = np.where(occ, zz, -np.inf).max(-1)
    d = np.where(occ, np.abs(zo[:, None] - zz), np.inf)
    near = d.argmin(-1)
    z_near = zz[np.arange(near.size), near]
    rn, rt = np.abs(zo - z_near), np.abs(zo - z_top)
    keep = rn <= 3.0                      # LEGACY ONLY (week9-17 used this truncation)
    return {"near": rn, "top": rt, "n": int(zo.size),
            "near_trunc3": float(np.median(rn[keep])) if keep.any() else float("nan"),
            "drop_frac": float(1 - keep.mean()),
            "outside": int(rows["key"].size - zo.size)}


def summarise_hfr(collected: dict, tail=3.0) -> dict:
    near = np.concatenate(collected["near"]) if collected["near"] else np.zeros(0)
    top = np.concatenate(collected["top"]) if collected["top"] else np.zeros(0)
    if near.size == 0:
        return {"hfr_near": float("nan"), "hfr_top": float("nan"), "n": 0}
    return {"hfr_near": float(np.median(near)), "hfr_top": float(np.median(top)),
            "hfr_near_p90": float(np.percentile(near, 90)),
            "hfr_top_p90": float(np.percentile(top, 90)),
            "tail_frac": float(np.mean(near > tail)), "n": int(near.size)}


# --------------------------------------------------------------------------------------
# pair fitting / sync
# --------------------------------------------------------------------------------------
def _prep_frames(d, slices, keys_need=None):
    fr = {}
    for k, idx in slices.items():
        kk = d["key"][idx]
        o = np.argsort(kk, kind="stable")
        sl = idx[o]
        kk = d["key"][sl]
        flat_row = (d["z_std"][sl] <= 0.3) & (d["n"][sl] >= 3)
        fr[k] = {"rows": sl, "key": kk, "z": d["z"][sl], "x": d["x"][sl],
                 "y": d["y"][sl], "conf": d["conf"][sl], "cos": d["cos_inc"][sl],
                 "single": single_layer_cells(kk),
                 # keys that are single-layer AND whose single row is flat (the batch rank-3
                 # constraint set); the review asks for this as an edge variant
                 "flat": np.sort(kk[flat_row & np.isin(kk, single_layer_cells(kk))])}
    return fr


def fit_pair(fi, fj, eval_cells, origin, cfg: SyncConfig, flat_only: bool = False) -> dict:
    """3-DOF pair fit on single-layer cells, one global origin, Huber IRLS.

    ``flat_only`` restricts the shared cells further to rows with ``z_std <= 0.3`` and
    ``n >= 3`` (the same constraint set the batch rank-3 ladder uses).
    """
    ka, kb = ("flat", "flat") if flat_only else ("single", "single")
    common = _intersect_sorted(fi[ka], fj[kb])
    if eval_cells is not None:
        common = _intersect_sorted(common, eval_cells)
    if common.size < cfg.min_shared:
        return {"ok": False, "n": int(common.size), "reason": "few_shared"}
    zi = _lookup(fi["key"], fi["z"], common)
    zj = _lookup(fj["key"], fj["z"], common)
    xi = _lookup(fi["key"], fi["x"], common)
    xj = _lookup(fj["key"], fj["x"], common)
    yi = _lookup(fi["key"], fi["y"], common)
    yj = _lookup(fj["key"], fj["y"], common)
    x = 0.5 * (xi + xj)
    y = 0.5 * (yi + yj)
    w = (_lookup(fi["key"], fi["conf"], common) * _lookup(fj["key"], fj["conf"], common) *
         _lookup(fi["key"], fi["cos"], common) * _lookup(fj["key"], fj["cos"], common))
    d = zi - zj
    A = np.stack([np.ones_like(x), x - origin[0], y - origin[1]], 1)
    w = np.clip(np.nan_to_num(w, nan=1.0), 1e-3, None)
    p = np.zeros(3)
    for _ in range(cfg.huber_iters):
        r = d - A @ p
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        h = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
        ww = w * h
        Aw = A * ww[:, None]
        p = np.linalg.lstsq(Aw.T @ A, Aw.T @ d, rcond=None)[0]
    r = d - A @ p
    gross = float(np.mean(np.abs(d) > cfg.max_dz))
    mad = float(np.median(np.abs(r)))
    s2 = float(np.sum(w * r * r) / max(np.sum(w) - 3, 1))
    cov = s2 * np.linalg.inv(A.T @ (w[:, None] * A) + 1e-9 * np.eye(3))
    return {"ok": bool(gross <= 0.5 and mad <= 3.0), "n": int(common.size),
            "p": p, "cov": cov, "gross_frac": gross, "mad": mad,
            "resid": float(np.median(np.abs(d)))}


def build_edges(frames, eval_cells, origin, cfg: SyncConfig, kfs):
    edges = []
    n = len(kfs)
    for a in range(n):
        for b in range(a + 1, n):
            e = fit_pair(frames[kfs[a]], frames[kfs[b]], eval_cells, origin, cfg)
            e.update({"i": kfs[a], "j": kfs[b]})
            edges.append(e)
    return edges


def solve_sync(edges, kfs, cfg: SyncConfig):
    """Sparse 3-DOF synchronisation; frame 0 of the component is the gauge."""
    from scipy.sparse import coo_matrix
    F = int(max(kfs)) + 1
    good = [e for e in edges if e["ok"] and "p" in e]
    if not good:
        return np.zeros((F, 3)), 0, np.zeros((0, 3))
    rows, cols, data, rhs = [], [], [], []
    r = 0
    for e in good:
        try:
            A = np.linalg.cholesky(np.linalg.inv(e["cov"] + 1e-6 * np.eye(3))).T
        except np.linalg.LinAlgError:
            A = np.eye(3)
        # A @ (theta_i - theta_j - p) = 0, row by row: row k couples ALL three components
        # (using only A[k,k] while the right-hand side uses the full row is inconsistent and
        # biases theta whenever cov has off-diagonal terms -- which it always does, because
        # the design [1, x, y] has x,y of order hundreds of metres)
        for k in range(3):
            for c in range(3):
                rows += [r, r]
                cols += [3 * e["i"] + c, 3 * e["j"] + c]
                data += [A[k, c], -A[k, c]]
            rhs.append(A[k] @ e["p"])
            r += 1
    # gauge rows (weight 1e3) on the lowest frame index
    g = int(min(kfs))
    for k in range(3):
        rows.append(r); cols.append(3 * g + k); data.append(1e3); rhs.append(0.0); r += 1
    A = coo_matrix((data, (rows, cols)), shape=(r, 3 * F)).tocsr()
    ATA = (A.T @ A).toarray()
    ATb = np.asarray(A.T @ np.asarray(rhs)).ravel()
    scale = float(np.mean(np.diag(ATA))) or 1.0
    theta = np.linalg.solve(ATA + cfg.ridge * scale * np.eye(ATA.shape[0]), ATb).reshape(-1, 3)
    resid = np.array([np.linalg.norm(A[k * 3:(k + 1) * 3] @ theta.ravel() - rhs[k * 3:(k + 1) * 3])
                      for k in range(len(good))])
    return theta, len(good), resid


# --------------------------------------------------------------------------------------
# absolute baselines
# --------------------------------------------------------------------------------------
def fit_to_product(rows, prod, eval_cells, origin, mode="t3", cfg=SyncConfig()) -> np.ndarray:
    """Fit one frame to the product's near layer; returns the correction ``z -= corr``."""
    from stream_ortho.map.selection import unpack_cell

    single = set(single_layer_cells(rows["key"]).tolist())
    if eval_cells is not None:
        single &= set(np.asarray(eval_cells).tolist())
    ii = np.array([i for i, k in enumerate(rows["key"]) if k in single], np.int64)
    if ii.size < 50:
        return np.zeros(3)
    kk = rows["key"][ii]
    ix, iy = unpack_cell(kk)
    pn, pz, ok = product_layers_at(prod, ix, iy)
    occ = pn > 0
    has = occ.any(-1)
    zo = rows["z"][ii][has]
    zz = pz[has]
    occ = occ[has]
    x = rows["x"][ii][has] - origin[0]
    y = rows["y"][ii][has] - origin[1]
    w = rows["conf"][ii][has] * rows["cos"][ii][has]
    d = np.where(occ, np.abs(zo[:, None] - zz), np.inf)
    near = d.argmin(-1)
    z_near = zz[np.arange(near.size), near]
    resid = zo - z_near
    w = np.clip(np.nan_to_num(w, nan=1.0), 1e-3, None)

    if mode == "t3":
        A = np.stack([np.ones_like(x), x, y], 1)
        p = np.zeros(3)
    elif mode == "s7":
        A = np.stack([z_near, np.ones_like(x), x, y], 1)
        p = np.zeros(4)
    elif mode == "f4":
        A = _bilinear4(x, y)
        p = np.zeros(A.shape[1])
    else:
        raise ValueError(mode)

    for _ in range(cfg.huber_iters):
        r = resid - A @ p
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        h = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
        ww = w * h
        Aw = A * ww[:, None]
        p = np.linalg.lstsq(Aw.T @ A, Aw.T @ resid, rcond=None)[0]
        if mode == "s7":
            p[0] = float(np.clip(p[0], -0.1, 0.1))
    return p


def _bilinear4(x, y, n=4):
    """16 bilinear basis functions over a 4x4 partition (the ``field4x4`` baseline)."""
    u = np.clip((x / 600.0 + 1) * 0.5, 0, 1) * (n - 1)
    v = np.clip((y / 600.0 + 1) * 0.5, 0, 1) * (n - 1)
    iu, iv = np.floor(u).astype(int).clip(0, n - 2), np.floor(v).astype(int).clip(0, n - 2)
    fu, fv = u - iu, v - iv
    cols = []
    for a in range(n):
        for b in range(n):
            cols.append((np.abs(iu - a) <= 1) * (np.abs(iv - b) <= 1) *
                        (1 - np.abs(u - a)) * (1 - np.abs(v - b)))
    return np.stack(cols, 1)


def apply_correction(rows, corr, origin, mode="t3"):
    """Return corrected z for a frame's rows."""
    z = rows["z"].copy()
    x = rows["x"] - origin[0]
    y = rows["y"] - origin[1]
    if mode == "raw":
        return z
    if mode == "t3":
        return z - (corr[0] + corr[1] * x + corr[2] * y)
    if mode == "s7":
        raise NotImplementedError("s7 needs the product layer height")
    if mode == "f4":
        A = _bilinear4(x, y)
        return z - A @ corr
    raise ValueError(mode)


# --------------------------------------------------------------------------------------
# driver helpers
# --------------------------------------------------------------------------------------
def _top_surface(prod):
    """Chosen surface for the grid metrics: the highest occupied layer of every cell."""
    n = prod["n"]
    occ = n > 0
    z = prod["z_ref"]
    z_top = np.where(occ, z, -np.inf).max(-1)
    live = occ.any(-1)
    return {"z": np.where(live, z_top, 0.0), "rgb": np.zeros(live.shape + (3,)),
            "n": np.take_along_axis(n, occ.argmax(-1)[..., None], -1)[..., 0],
            "live": live, "label": occ.argmax(-1).astype(np.int16)}


def _surface_metrics(prod) -> dict:
    from stream_ortho.eval.grid_metrics import coverage_cells, layer_consistency, ripple_amplitude
    sel = _top_surface(prod)
    lcr = layer_consistency(prod, sel)
    ra = ripple_amplitude(prod, sel)
    cov = coverage_cells(prod)
    return {"lcr": lcr["lcr"], "lcr_n_shared_pairs": lcr["n_shared_pairs"],
            "top_preserved_frac": lcr["top_preserved_frac"],
            "ra": ra["ra"], "ra_n_cells": ra["n_cells"],
            "coverage_cells": cov["coverage_cells"]}


def _frame_guard(z_raw, z_corr) -> dict:
    """Did the correction flatten or shift this frame? (review 11 §4)"""
    q = np.percentile(z_raw, [25, 75])
    qc = np.percentile(z_corr, [25, 75])
    iqr_raw = float(q[1] - q[0])
    return {"z_iqr_ratio": float((qc[1] - qc[0]) / iqr_raw) if iqr_raw > 0 else float("nan"),
            "median_z_shift": float(abs(np.median(z_corr) - np.median(z_raw)))}


def _guards(prod, prod_raw) -> dict:
    from stream_ortho.eval.hfr import surface_guards
    return surface_guards(prod, prod_raw)


def _rows(d, idx):
    return {k: d[k][idx] for k in ("key", "z", "x", "y", "conf", "cos_inc", "n", "z_std",
                                   "layer", "rgb")}


def product_near_z(rows, prod):
    """Per-row height of the nearest occupied product layer (NaN where none)."""
    from stream_ortho.map.selection import unpack_cell
    ix, iy = unpack_cell(rows["key"])
    pn, pz, ok = product_layers_at(prod, ix, iy)
    occ = pn > 0
    has = occ.any(-1)
    z_near = np.full(rows["key"].size, np.nan)
    if not has.any():
        return z_near, has
    zz = pz[has]
    zo = rows["z"][has]
    d = np.where(occ[has], np.abs(zo[:, None] - zz), np.inf)
    near = d.argmin(-1)
    z_near[has] = zz[np.arange(near.size), near]
    return z_near, has


def _fit_t3(z, z_near, x, y, w, cfg, scale=False):
    """Huber IRLS fit of ``z - z_near = b + tx u + ty v`` (optionally with a scale on z_near)."""
    ok = np.isfinite(z_near)
    if ok.sum() < 50:
        return None
    d = z[ok] - z_near[ok]
    u, v = x[ok], y[ok]
    ww = np.clip(np.nan_to_num(w[ok], nan=1.0), 1e-3, None)
    A = (np.stack([z_near[ok], np.ones_like(u), u, v], 1) if scale
         else np.stack([np.ones_like(u), u, v], 1))
    p = np.zeros(A.shape[1])
    for _ in range(cfg.huber_iters):
        r = d - A @ p
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        h = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
        wwk = ww * h
        Aw = A * wwk[:, None]
        p = np.linalg.lstsq(Aw.T @ A, Aw.T @ d, rcond=None)[0]
        if scale:
            p[0] = float(np.clip(p[0], -0.1, 0.1))
    return p


def correct_frame(rows, prod, eval_cells, origin, mode, cfg):
    """Corrected z for one frame under one of the *absolute* methods."""
    z = rows["z"].copy()
    if mode == "raw":
        return z
    z_near, _ = product_near_z(rows, prod)
    x = rows["x"] - origin[0]
    y = rows["y"] - origin[1]
    w = rows["conf"] * rows["cos_inc"]

    if mode in ("t3", "s7"):
        p = _fit_t3(z, z_near, x, y, w, cfg, scale=(mode == "s7"))
        if p is None:
            return z
        if mode == "t3":
            return z - (p[0] + p[1] * x + p[2] * y)
        corr = p[0] * np.where(np.isfinite(z_near), z_near, 0.0) + p[1] + p[2] * x + p[3] * y
        return z - corr
    if mode == "f4":
        ok = np.isfinite(z_near)
        if ok.sum() < 200:
            return z
        A = _bilinear4(x, y)
        d = np.where(ok, z - np.where(ok, z_near, 0.0), 0.0)
        ww = np.where(ok, np.clip(np.nan_to_num(w, nan=1.0), 1e-3, None), 0.0)
        p = np.zeros(A.shape[1])
        for _ in range(cfg.huber_iters):
            r = d - A @ p
            s = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))) + 1e-6
            h = np.where(ok, np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9)), 0.0)
            wwk = ww * h
            Aw = A * wwk[:, None]
            p = np.linalg.lstsq(Aw.T @ A, Aw.T @ d, rcond=None)[0]
        return z - A @ p
    raise ValueError(mode)


def _spec(z, prod, origin, theta):
    """Apply a 3-DOF theta (per frame) to rows."""
    return z - (theta[0] + theta[1] * (0.0) + theta[2] * (0.0))


def _apply_theta_frame(rows, th, origin):
    return rows["z"] - (th[0] + th[1] * (rows["x"] - origin[0]) + th[2] * (rows["y"] - origin[1]))


def _components(edges, kfs):
    """Union-find over the edges -> number of connected components and their sizes."""
    parent = {k: k for k in kfs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        if e.get("ok") and "p" in e:
            ra, rb = find(e["i"]), find(e["j"])
            if ra != rb:
                parent[ra] = rb
    from collections import Counter
    c = Counter(find(k) for k in kfs)
    return len(c), sorted(c.values(), reverse=True)


def _cycle_stats(edges, origin):
    """Triangle inconsistency ``p_ij + p_jk - p_ik`` over the edge graph."""
    by = {}
    for e in edges:
        if e.get("ok") and "p" in e:
            by[(e["i"], e["j"])] = e["p"]
            by[(e["j"], e["i"])] = -e["p"]
    tris = []
    keys = list(by)
    for (i, j) in keys:
        if i >= j:
            continue
        for k in (k for (a, k) in keys if a == j):
            p_ik = by.get((i, k))
            if p_ik is None:
                continue
            r = by[(i, j)] + by[(j, k)] - p_ik
            tris.append(np.linalg.norm(r))
    if not tris:
        return {"cycle_median_m": float("nan"), "cycle_p90_m": float("nan"), "n_triangles": 0}
    return {"cycle_median_m": float(np.median(tris)),
            "cycle_p90_m": float(np.percentile(tris, 90)), "n_triangles": int(len(tris))}


def pc_held(d, slices, held_k, build_k, eval_cells, origin, cfg, correct_fn, n_neighbors=6):
    """Pairwise consistency between held frames and their nearest build frames."""
    out = []
    for k in held_k:
        rk = _rows(d, slices[k])
        zk = correct_fn(k, rk)
        near = [b for b in build_k if abs(b - k) <= 40][:n_neighbors] or build_k[:n_neighbors]
        for j in near:
            rj = _rows(d, slices[j])
            zj = correct_fn(j, rj)
            ki = single_layer_cells(rk["key"]); kj = single_layer_cells(rj["key"])
            common = _intersect_sorted(ki, kj)
            if eval_cells is not None:
                common = _intersect_sorted(common, eval_cells)
            if common.size < cfg.min_shared:
                continue
            a = _lookup(np.sort(rk["key"]), zk[np.argsort(rk["key"], kind="stable")], common)
            b = _lookup(np.sort(rj["key"]), zj[np.argsort(rj["key"], kind="stable")], common)
            dd = np.abs(a - b)
            out.append({"i": int(k), "j": int(j), "n": int(dd.size),
                        "gross_frac": float(np.mean(dd > cfg.max_dz)),
                        "median": float(np.median(dd))})
    return out


def summarise_pc(pairs):
    good = [p for p in pairs if p["gross_frac"] <= 0.5]
    if not pairs:
        return {"pc": float("nan"), "pc_all": float("nan"), "n_pairs": 0, "n_pairs_ok": 0}
    return {"pc": float(np.median([p["median"] for p in good])) if good else float("nan"),
            "pc_all": float(np.median([p["median"] for p in pairs])),
            "n_pairs": len(pairs), "n_pairs_ok": len(good)}


def pc_pass(d, slices, held_k, nbr, eval_cells, cfg, z_corrected):
    """PC between held frames and their top-overlap build neighbours, after correction.

    ``z_corrected(k, rows)`` returns the corrected heights of frame ``k`` (raw if unchanged).
    """
    out = {}
    for k in held_k:
        rk = _rows(d, slices[k])
        zk = z_corrected(k, rk)
        ok_k = np.argsort(rk["key"], kind="stable")
        ks_i, zs_i = rk["key"][ok_k], zk[ok_k]
        sin_k = single_layer_cells(rk["key"])
        for j in nbr[k]:
            rj = _rows(d, slices[j])
            zj = z_corrected(j, rj)
            common = _intersect_sorted(sin_k, single_layer_cells(rj["key"]))
            if eval_cells is not None:
                common = _intersect_sorted(common, eval_cells)
            if common.size < cfg.min_shared:
                continue
            ok_j = np.argsort(rj["key"], kind="stable")
            a = _lookup(ks_i, zs_i, common)
            b = _lookup(rj["key"][ok_j], zj[ok_j], common)
            dd = np.abs(a - b)
            out.setdefault(k, []).append({"j": int(j), "n": int(dd.size),
                                          "gross_frac": float(np.mean(dd > cfg.max_dz)),
                                          "median": float(np.median(dd))})
    flat = [p for v in out.values() for p in v]
    good = [p for p in flat if p["gross_frac"] <= 0.5]
    return {"pc": float(np.median([p["median"] for p in good])) if good else float("nan"),
            "pc_all": float(np.median([p["median"] for p in flat])) if flat else float("nan"),
            "n_pairs": len(flat), "n_pairs_ok": len(good)}


def run_e1(obs_path, out_dir, seq="ieu", source="pointmap", n_folds=5, max_frames=0,
           cfg=None, neighbor_builds=6, run_sync=True, only_fold=None,
           variants=("all", "flat")) -> dict:
    cfg = cfg or SyncConfig()
    t0 = time.time()
    obs = np.load(pathlib.Path(obs_path) / "obs_merged.npz")
    d = load_obs(obs_path)
    ix_all, iy_all = obs["ix"].astype(np.int64), obs["iy"].astype(np.int64)
    slices = frame_slices(d["kf"])
    kfs = sorted(slices)
    if max_frames:
        kfs = kfs[:max_frames]
        slices = {k: slices[k] for k in kfs}

    folds = np.array_split(np.arange(len(kfs)), n_folds)
    if only_fold is not None:
        folds = folds[only_fold:only_fold + 1]
    methods = ("raw", "abs_t3", "abs_s7", "field4x4", "sync_t3_all", "sync_t3")
    acc = {m: {"near": [], "top": [], "pc": [], "trunc3": [], "drop": []} for m in methods}
    frame_guards = {m: [] for m in methods}
    surf = {"raw": [], "sync_t3_all": [], "sync_t3": []}
    guards = {m: [] for m in methods}
    folds_info, drift, cyc = [], [], []
    theta_hist = []

    for fi, fold in enumerate(folds):
        held_k = [kfs[i] for i in fold]
        build_k = [k for k in kfs if k not in held_k]
        idx_b = np.sort(np.concatenate([slices[k] for k in build_k]))
        rb = _rows(d, idx_b)
        kf_b = d["kf"][idx_b]
        ixb, iyb = ix_all[idx_b], iy_all[idx_b]
        x0 = (ixb.min() - 2) * d["cell"]; y0 = (iyb.min() - 2) * d["cell"]
        x1 = (ixb.max() + 2) * d["cell"]; y1 = (iyb.max() + 2) * d["cell"]
        bbox = (x0, y0, x1, y1)
        origin = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

        prod_raw = build_product(ixb, iyb, rb["key"], rb["z"],
                                 rb["conf"] * rb["n"], rb["n"].astype(np.float64), d["cell"], bbox)
        eval_cells = eval_cells_of(prod_raw)
        surf["raw"].append(_surface_metrics(prod_raw))
        guards["raw"].append(_guards(prod_raw, prod_raw))

        # ---- absolute methods (product stays raw) ----
        for mode, name in (("t3", "abs_t3"), ("s7", "abs_s7"), ("f4", "field4x4")):
            for k in held_k:
                r = _rows(d, slices[k])
                zc = dict(r); zc["z"] = correct_frame(r, prod_raw, eval_cells, origin, mode, cfg)
                h = hfr_rows(zc, prod_raw)
                acc[name]["near"].append(h["near"]); acc[name]["top"].append(h["top"])
                frame_guards[name].append(_frame_guard(r["z"], zc["z"]))

        # ---- raw ----
        for k in held_k:
            r = _rows(d, slices[k])
            h = hfr_rows(r, prod_raw)
            acc["raw"]["near"].append(h["near"]); acc["raw"]["top"].append(h["top"])
            frame_guards["raw"].append(_frame_guard(r["z"], r["z"]))

        # top-overlap build neighbours for the held frames (used by PC and by the sync)
        nbr = {}
        for k in held_k:
            rk = _rows(d, slices[k])
            sk = single_layer_cells(rk["key"])
            ov = sorted(((_intersect_sorted(sk, single_layer_cells(_rows(d, slices[j])["key"])).size, j)
                         for j in build_k), reverse=True)
            nbr[k] = [j for _, j in ov[:neighbor_builds]]
        acc["raw"]["pc"].append(pc_pass(d, slices, held_k, nbr, eval_cells, cfg,
                                        lambda k, rows: rows["z"]))
        acc["abs_t3"]["pc"].append(pc_pass(
            d, slices, held_k, nbr, eval_cells, cfg,
            lambda k, rows: correct_frame(rows, prod_raw, eval_cells, origin, "t3", cfg)
            if k in held_k else rows["z"]))
        acc["field4x4"]["pc"].append(pc_pass(
            d, slices, held_k, nbr, eval_cells, cfg,
            lambda k, rows: correct_frame(rows, prod_raw, eval_cells, origin, "f4", cfg)
            if k in held_k else rows["z"]))

        # ---- relative sync on the build frames: two edge variants ----
        # "all"  = single-layer cells (the previous run)
        # "flat" = additionally z_std <= 0.3 and n >= 3 (the batch rank-3 constraint set)
        for variant, flat_only in [("all", False), ("flat", True)]:
            if variant not in variants:
                continue
            theta = np.zeros((max(kfs) + 1, 3))
            if not run_sync:
                break
            fr = _prep_frames(d, {k: slices[k] for k in build_k})
            edges = []
            for a in range(len(build_k)):
                for b in range(a + 1, len(build_k)):
                    e = fit_pair(fr[build_k[a]], fr[build_k[b]], eval_cells, origin, cfg,
                                 flat_only=flat_only)
                    e.update({"i": build_k[a], "j": build_k[b]})
                    edges.append(e)
            theta, n_ok, edge_res = solve_sync(edges, build_k, cfg)
            n_comp, comp_sizes = _components(edges, build_k)
            cyc.append(dict(_cycle_stats(edges, origin), variant=variant, fold=fi))
            drift.append({"fold": fi, "variant": variant, "theta_b": theta[build_k, 0].tolist(),
                          "n_edges": len(edges), "n_ok": int(n_ok),
                          "n_components": int(n_comp), "component_sizes": comp_sizes})
            # post-hoc constant gauge: mean b over build frames -> 0
            theta[build_k, 0] -= float(np.mean(theta[build_k, 0]))
            z_b = rb["z"] - (theta[kf_b, 0] + theta[kf_b, 1] * (rb["x"] - origin[0]) +
                             theta[kf_b, 2] * (rb["y"] - origin[1]))
            prod_sync = build_product(ixb, iyb, rb["key"], z_b,
                                      rb["conf"] * rb["n"], rb["n"].astype(np.float64),
                                      d["cell"], bbox)
            # constant alignment on build frames only
            occ_r, occ_s = prod_raw["n"] > 0, prod_sync["n"] > 0
            c = float(np.median(prod_raw["z_ref"][occ_r]) - np.median(prod_sync["z_ref"][occ_s]))
            prod_sync["z_ref"][occ_s] += c
            prod_sync["sum_wz"][occ_s] += c * prod_sync["sum_w"][occ_s]
            name = "sync_t3" if variant == "flat" else "sync_t3_all"
            surf[name] = surf.get(name, [])
            surf[name].append(_surface_metrics(prod_sync))
            guards[name] = guards.get(name, [])
            guards[name].append(_guards(prod_sync, prod_raw))

            # held frames: theta from edges to build frames, then constant c
            _held_theta = {}
            for k in held_k:
                r = _rows(d, slices[k])
                pk = []
                frk = _prep_frames(d, {k: slices[k]})[k]
                ov = []
                for j in build_k:
                    ov.append((_intersect_sorted(frk["flat" if flat_only else "single"],
                                                 fr[j]["flat" if flat_only else "single"]).size, j))
                ov.sort(reverse=True)
                for _, j in ov[:neighbor_builds]:
                    if _ < cfg.min_shared:
                        break
                    e = fit_pair(frk, fr[j], eval_cells, origin, cfg, flat_only=flat_only)
                    if e.get("ok") and "p" in e:
                        pk.append(theta[j] + e["p"])
                th = np.median(np.stack(pk), 0) if pk else theta[build_k[0]]
                _held_theta[k] = th
                zc = dict(r)
                zc["z"] = _apply_theta_frame(r, th, origin) + c
                h = hfr_rows(zc, prod_sync)
                acc[name]["near"].append(h["near"]); acc[name]["top"].append(h["top"])
                fg = _frame_guard(r["z"], zc["z"])
                frame_guards[name].append(fg)

            def _z_sync(k, rows, _ht=_held_theta, _th=theta, _c=c):
                if k in _ht:
                    return _apply_theta_frame(rows, _ht[k], origin) + _c
                return _apply_theta_frame(rows, _th[k], origin) + _c

            acc[name]["pc"].append(pc_pass(d, slices, held_k, nbr, eval_cells, cfg, _z_sync))
            theta_hist.append({"fold": fi, "variant": variant, "b": theta[build_k, 0].tolist(),
                               "tx": theta[build_k, 1].tolist(), "ty": theta[build_k, 2].tolist(),
                               "build_k": build_k})
        folds_info.append({"fold": fi, "n_held": len(held_k), "n_build": len(build_k),
                           "eval_cells": int(eval_cells.size), "bbox": list(bbox),
                           "n_overflow_layers": prod_raw["n_overflow_layers"]})

    out = {"seq": seq, "source": source, "cell": d["cell"], "sync_cell": d["cell"],
           "n_frames": len(kfs), "folds": f"contiguous-{n_folds}",
           "ridge": cfg.ridge, "min_shared": cfg.min_shared,
           "folds_info": folds_info, "cycle": cyc, "drift": drift,
           "methods": {}, "wall_s": time.time() - t0}
    for m in methods:
        s = summarise_hfr(acc[m])
        if acc[m].get("pc"):
            pcs = [p for fold in acc[m]["pc"] for p in ([fold] if isinstance(fold, dict) else [])]
            if pcs:
                s["pc"] = float(np.median([p["pc"] for p in pcs if np.isfinite(p["pc"])]))
                s["pc_all"] = float(np.median([p["pc_all"] for p in pcs if np.isfinite(p["pc_all"])]))
                s["pc_held"] = s["pc"]
                s["n_pairs_pc"] = int(sum(p["n_pairs"] for p in pcs))
        sm = surf["raw"] if m in ("raw", "abs_t3", "abs_s7", "field4x4") else surf.get("sync_t3", [])
        if sm:
            s.update({k: float(np.median([x[k] for x in sm if np.isfinite(x.get(k, np.nan))]))
                      for k in ("lcr", "ra", "coverage_cells", "top_preserved_frac")
                      if any(np.isfinite(x.get(k, np.nan)) for x in sm)})
        g = [x for x in guards[m] if x]
        if g:
            s["prod_z_iqr_ratio"] = float(np.median([x["z_iqr_ratio"] for x in g]))
            s["prod_median_z_shift"] = float(np.median([x["median_z_shift"] for x in g]))
        fg = [x for x in frame_guards[m] if x]
        if fg:
            # the review's guard for the *corrected held frames* (abs methods included)
            s["z_iqr_ratio"] = float(np.median([x["z_iqr_ratio"] for x in fg]))
            s["median_z_shift"] = float(np.median([x["median_z_shift"] for x in fg]))
            s["held_guard_ok"] = bool(0.95 <= s["z_iqr_ratio"] <= 1.05 and s["median_z_shift"] < 0.5)
        out["methods"][m] = s
    return out


# --------------------------------------------------------------------------------------
# gate + output
# --------------------------------------------------------------------------------------
def gate_report(res: dict) -> dict:
    """Apply the E1 gate of ``04-P0-3`` §1 to the measured methods."""
    m = res["methods"]
    sync = m.get("sync_t3", {})
    absm = {k: v for k, v in m.items() if k.startswith("abs_")}
    best_abs = min((v.get("hfr_near", np.inf) for v in absm.values()), default=float("inf"))
    best_abs_name = min(absm, key=lambda k: absm[k].get("hfr_near", np.inf)) if absm else None
    raw = m.get("raw", {}).get("hfr_near", float("nan"))
    sn = sync.get("hfr_near", float("nan"))
    d_abs = 100.0 * (1 - sn / best_abs) if best_abs and np.isfinite(best_abs) else float("nan")
    d_raw = 100.0 * (1 - sn / raw) if raw and np.isfinite(raw) else float("nan")
    checks = {
        "hfr_vs_best_abs_ge_25": bool(d_abs >= 25),
        "hfr_vs_raw_ge_40": bool(d_raw >= 40),
        "tail_frac_ok": bool(sync.get("tail_frac", 1) <= m.get("raw", {}).get("tail_frac", 0) + 0.02),
        "z_iqr_ratio_ok": bool(0.95 <= sync.get("z_iqr_ratio", 0) <= 1.05),
        "median_z_shift_ok": bool(sync.get("median_z_shift", 1) < 0.5),
        "coverage_ok": bool(sync.get("coverage_cells", 0) >= 0.99 * m.get("raw", {}).get("coverage_cells", 0)),
        "product_translation": bool(
            (sync.get("lcr", 0) >= m.get("raw", {}).get("lcr", 0) + 0.05) or
            (sync.get("ra", np.inf) <= 0.75 * m.get("raw", {}).get("ra", np.inf))),
    }
    return {"sync_vs_best_abs_hfr_pct": d_abs, "sync_vs_raw_hfr_pct": d_raw,
            "best_abs": best_abs_name, "checks": checks,
            "passed": bool(all(checks.values()))}


def save_metrics(res: dict, out_dir, git_hash: str = "", command: str = "") -> pathlib.Path:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(res)
    res["git_hash"] = git_hash
    res["command"] = command
    res["px_gate"] = "removed: z-only model does not change (x,y)"
    res["gate"] = gate_report(res)
    name = f"metrics_{res.get('source', 'na')}.json"
    (out / name).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    if res.get("source") == "pointmap":                # main table = pointmap (review 10 §4.1)
        (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return out / name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="obs dir containing obs_merged.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--source", default="pointmap")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--min-shared", type=int, default=500)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--variants", default="all,flat")
    a = ap.parse_args(argv)
    import subprocess
    import sys
    res = run_e1(a.obs, a.out, seq=a.seq, source=a.source, n_folds=a.folds,
                 max_frames=a.max_frames, cfg=SyncConfig(ridge=a.ridge, min_shared=a.min_shared),
                 only_fold=a.only_fold,
                 variants=tuple(v.strip() for v in a.variants.split(",") if v.strip()))
    root = pathlib.Path(__file__).resolve().parents[2]
    gh = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
    p = save_metrics(res, a.out, git_hash=gh,
                     command=" ".join([sys.executable, "-m", "stream_ortho.eval.sync_eval"] + (argv or sys.argv[1:])))
    g = gate_report(res)
    print(json.dumps({m: {k: round(v, 3) for k, v in res["methods"][m].items()
                          if k in ("hfr_near", "hfr_top", "lcr", "ra", "coverage_cells", "z_iqr_ratio")}
                      for m in res["methods"]}, indent=1, ensure_ascii=False))
    print(json.dumps(g, indent=1, ensure_ascii=False))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
