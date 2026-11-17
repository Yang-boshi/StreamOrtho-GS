"""Cell-grid metrics for P0 -- **frozen** definitions from ``refs/06-任务卡-P0.md`` §2.

Every function takes

``store``  a ``map.columns.ColumnStore`` (or any object/dict with the same fields:
           ``cell, ox, oy, W, H, L, z_ref, n, sum_w, sum_wz, sum_wz2, rgb``)
``sel``    the chosen surface ``{"z","rgb","n","live","label"}`` with ``(H,W)`` arrays

and returns a ``dict`` containing the value, its ``n``/parameters and its unit, so a JSON
record can be written without recomputing anything.

Units are metres; directions are stated per function.  Notation: ``occ = n > 0``,
``single-layer flat`` = ``occ.sum(-1) == 1 & sigma_z <= 0.3 & n >= 2``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------------------
# generic access
# --------------------------------------------------------------------------------------
def _g(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _phys(store):
    """(H, W, L) occupancy, heights, sigma and support."""
    n = np.asarray(_g(store, "n"))
    occ = n > 0
    z = np.asarray(_g(store, "z_ref"), np.float64)
    sw = np.maximum(np.asarray(_g(store, "sum_w"), np.float64), EPS)
    mz = np.asarray(_g(store, "sum_wz"), np.float64) / sw
    var = np.maximum(np.asarray(_g(store, "sum_wz2"), np.float64) / sw - mz * mz, 0.0)
    return occ, z, np.sqrt(var), n


def _take_occupied(occ, arr):
    """Value of ``arr`` on each cell's first occupied layer (meaningful only if unique)."""
    idx = occ.argmax(-1)
    return np.take_along_axis(arr, idx[..., None], -1)[..., 0]


def _flat_mask(store, live=None, sigma_max: float = 0.3, min_n: int = 2):
    occ, _, sigma, n = _phys(store)
    single = occ.sum(-1) == 1
    sigma1 = _take_occupied(occ, sigma)
    n1 = _take_occupied(occ, n)
    m = single & (sigma1 <= sigma_max) & (n1 >= min_n)
    if live is not None:
        m &= np.asarray(live, bool)
    return m


def _neighbour_pairs(mask_or_shape, axis):
    """Elementwise-aligned slices for right (axis=1) / down (axis=0) neighbours."""
    H, W = mask_or_shape if isinstance(mask_or_shape, tuple) else mask_or_shape.shape
    if axis == 1:
        return (slice(None), slice(0, W - 1)), (slice(None), slice(1, W))
    return (slice(0, H - 1), slice(None)), (slice(1, H), slice(None))


# --------------------------------------------------------------------------------------
# PC -- pairwise consistency (§2.2)
# --------------------------------------------------------------------------------------
def single_layer_keys(frame) -> np.ndarray:
    """Keys that occur exactly once in a row-wise observation table."""
    k, cnt = np.unique(np.asarray(frame["key"], np.int64), return_counts=True)
    return k[cnt == 1]


def lookup_field(frame, name: str, keys) -> np.ndarray:
    """Field value for ``keys`` (which must all be present exactly once)."""
    key = np.asarray(frame["key"], np.int64)
    val = np.asarray(frame[name])
    order = np.argsort(key, kind="stable")
    ks = key[order]
    pos = np.searchsorted(ks, np.asarray(keys, np.int64))
    pos = np.clip(pos, 0, len(ks) - 1)
    if ks[pos].size and not np.array_equal(ks[pos], np.asarray(keys, np.int64)):
        raise KeyError(f"{name}: keys missing from the frame")
    return val[order][pos]


def fit_pair_t3(zi, zj, x, y, origin=(0.0, 0.0), wi=None, wj=None, iters: int = 4) -> dict:
    """Robust 3-DOF pairwise fit ``z_i - z_j = (theta_i - theta_j) . (1, x-x0, y-y0)``.

    Huber IRLS, weights ``conf_i * conf_j`` (times ``|cos|`` by the caller).  ``origin`` is
    **one global** ``(x0, y0)`` for every pair in an experiment: with per-pair centring the
    constant term absorbs the tilt and the estimates can no longer be composed (that is the
    week9-14/17 bug).  Nothing is hard-dropped; outliers are down-weighted and summarised by
    ``gross_frac`` / ``resid`` / ``ok``.

    Returns ``{"p","cov","n","resid","ok","gross_frac","max_dz"}``.
    """
    zi = np.asarray(zi, np.float64)
    zj = np.asarray(zj, np.float64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    d = zi - zj
    A = np.stack([np.ones_like(x), x - origin[0], y - origin[1]], 1)
    if wi is None and wj is None:
        w = np.ones_like(d)
    else:
        wi = np.ones_like(d) if wi is None else np.nan_to_num(np.asarray(wi, np.float64), nan=1.0)
        wj = np.ones_like(d) if wj is None else np.nan_to_num(np.asarray(wj, np.float64), nan=1.0)
        w = np.clip(wi, 1e-3, None) * np.clip(wj, 1e-3, None)

    p = np.zeros(3)
    for _ in range(max(1, int(iters))):
        r = d - A @ p
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + EPS
        h = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
        ww = w * h
        Aw = A * ww[:, None]
        p = np.linalg.lstsq(Aw.T @ A, Aw.T @ d, rcond=None)[0]

    r = d - A @ p
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + EPS
    h = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
    ww = w * h
    s2 = float(np.sum(ww * r * r) / max(np.sum(ww) - 3.0, 1.0))
    cov = s2 * np.linalg.inv(A.T @ (ww[:, None] * A) + 1e-9 * np.eye(3))
    raw = np.abs(d)
    gross = float(np.mean(raw > 3.0))
    resid = float(np.median(np.abs(r)))
    return {"p": p, "cov": cov, "n": int(len(d)), "resid": resid,
            "gross_frac": gross, "ok": bool(gross <= 0.5 and resid <= 3.0),
            "max_dz": 3.0}


def pairwise_consistency(frames, eval_cells=None, min_shared: int = 500,
                         max_dz: float = 3.0) -> dict:
    """PC over every frame pair, matched **by cell key** (§2.2).

    Only cells where *both* frames have exactly one layer are used (otherwise ``z_i`` is
    ambiguous), intersected with ``eval_cells``.  ``np.sort`` pairing is forbidden: it pairs
    ranks across differently populated cells and silently cancels real offsets.

    ``pc``       median over good pairs of the pair's median |dz|
    ``pc_all``   same but including pairs with ``gross_frac > 0.5`` (audited, not dropped
                 silently); ``pc_pooled`` pools every cell of every good pair.
    """
    n = len(frames)
    out, all_d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ki = single_layer_keys(frames[i])
            kj = single_layer_keys(frames[j])
            common = np.intersect1d(ki, kj)
            if eval_cells is not None:
                common = np.intersect1d(common, np.asarray(eval_cells, np.int64))
            if common.size < min_shared:
                continue
            zi = lookup_field(frames[i], "z", common).astype(np.float64)
            zj = lookup_field(frames[j], "z", common).astype(np.float64)
            d = np.abs(zi - zj)
            gross = float(np.mean(d > max_dz))
            all_d.append(d)
            out.append({"i": i, "j": j, "n": int(d.size), "gross_frac": gross,
                        "ok": bool(gross <= 0.5),
                        "median": float(np.median(d)),
                        "p90": float(np.percentile(d, 90))})
    good = [o for o in out if o["ok"]]
    return {
        "pc": float(np.median([o["median"] for o in good])) if good else float("nan"),
        "pc_all": float(np.median([o["median"] for o in out])) if out else float("nan"),
        "pc_pooled": float(np.median(np.concatenate(all_d))) if all_d else float("nan"),
        "n_pairs": len(out), "n_pairs_ok": len(good),
        "min_shared": int(min_shared), "max_dz": float(max_dz), "pairs": out,
    }


# --------------------------------------------------------------------------------------
# LCR -- layer consistency rate (§2.3)
# --------------------------------------------------------------------------------------
def _pair_match_counts(zA, oA, zB, oB, z_gap: float, chunk: int = 200_000):
    """(m_A, m_B): how many layers of each cell find a partner within ``z_gap``."""
    n, L = zA.shape
    mA = np.zeros(n, np.int16)
    mB = np.zeros(n, np.int16)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = np.abs(zA[s:e, :, None] - zB[s:e, None, :])          # (m, L, L)
        ok = diff <= z_gap
        mA[s:e] = (ok & oB[s:e, None, :]).any(-1).sum(-1) & oA[s:e].sum(-1)
        mB[s:e] = (ok & oA[s:e, :, None]).any(1).sum(-1) & oB[s:e].sum(-1)
    return mA, mB


def layer_consistency(store, sel, z_gap: float = 2.0, h_tol: float = 1.0,
                      top_preserve: float = 0.5, chunk: int = 200_000) -> dict:
    """LCR over right/down 4-neighbour pairs (§2.3).

    ``shared2``  both cells observe **at least two common layers** (``min(match_fwd,
                 match_bwd) >= 2``).  A true roof/facade edge shares only one layer and is
                 therefore *not* penalised; a double wall (both cells see roof and ground,
                 but the selector picks different ones) is.
    ``jump``     chosen heights differ by more than ``h_tol``.
    ``LCR``      1 - #(shared2 & jump) / #shared2.

    ``top_preserved_frac`` is the anti-cheat guard: the selected surface may not drift more
    than ``top_preserve`` metres below the top layer for more than 5% of live cells.
    """
    occ, z, _, n = _phys(store)
    live = np.asarray(sel["live"], bool)
    z_sel = np.asarray(sel["z"], np.float64)

    n_shared = n_inc = 0
    for axis in (1, 0):
        a_sl, b_sl = _neighbour_pairs(occ.shape[:2], axis)
        oA, oB = occ[a_sl], occ[b_sl]
        zA, zB = z[a_sl], z[b_sl]
        mA, mB = _pair_match_counts(zA.reshape(-1, zA.shape[-1]), oA.reshape(-1, oA.shape[-1]),
                                    zB.reshape(-1, zB.shape[-1]), oB.reshape(-1, oB.shape[-1]),
                                    z_gap, chunk)
        shared = (np.minimum(mA, mB) >= 2).reshape(oA.shape[:-1])
        both_live = live[a_sl] & live[b_sl]
        jump = np.abs(z_sel[a_sl] - z_sel[b_sl]) > h_tol
        s2 = shared & both_live
        n_shared += int(s2.sum())
        n_inc += int((s2 & jump).sum())

    z_top = np.where(occ, z, -np.inf).max(-1)
    top_ok = live & (np.abs(z_sel - z_top) <= top_preserve)
    return {
        "lcr": float(1.0 - n_inc / max(n_shared, 1)),
        "n_shared_pairs": int(n_shared), "n_inconsistent": int(n_inc),
        "top_preserved_frac": float(top_ok.sum() / max(int(live.sum()), 1)),
        "z_gap": float(z_gap), "h_tol": float(h_tol),
        "top_preserve": float(top_preserve),
    }


# --------------------------------------------------------------------------------------
# FCD -- flat colour discontinuity (§2.4)
# --------------------------------------------------------------------------------------
def flat_color_discontinuity(store, sel, h_tol: float = 0.2,
                             sigma_max: float = 0.3) -> dict:
    """Mean colour difference across *flat, same-surface* neighbour pairs (§2.4).

    Seam-only colour metrics (the old CDS) reward removing a fake seam: what is left are the
    real roof/ground edges, whose colour difference is large.  FCD therefore measures only
    neighbours that are both single-layer flat and within ``h_tol`` metres of height.
    Unit: 0-255 (mean of per-channel |delta|).  Lower is better.
    """
    live = np.asarray(sel["live"], bool)
    flat = _flat_mask(store, live, sigma_max=sigma_max)
    z = np.asarray(sel["z"], np.float64)
    rgb = np.asarray(sel["rgb"], np.float64)
    vals = []
    for axis in (1, 0):
        a_sl, b_sl = _neighbour_pairs(z.shape, axis)
        both = flat[a_sl] & flat[b_sl]
        same = np.abs(z[a_sl] - z[b_sl]) <= h_tol
        m = both & same
        diff = np.abs(rgb[a_sl] - rgb[b_sl]).mean(-1)
        vals.append(diff[m])
    v = np.concatenate(vals) if vals else np.zeros(0)
    return {"fcd": float(v.mean()) if v.size else float("nan"),
            "fcd_p90": float(np.percentile(v, 90)) if v.size else float("nan"),
            "n_pairs": int(v.size), "h_tol": float(h_tol),
            "sigma_max": float(sigma_max)}


# --------------------------------------------------------------------------------------
# DER_h -- height-field double edge rate (§2.5b)
# --------------------------------------------------------------------------------------
def _segments(mask_line):
    """Start/stop slices of contiguous True runs."""
    idx = np.nonzero(mask_line)[0]
    if idx.size == 0:
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [slice(s, e + 1) for s, e in zip(starts, ends)]


def double_edge_rate(sel, h_tol: float = 1.0, d_max: int = 6) -> dict:
    """Share of height steps that are immediately undone (§2.5b) -- annotation-free.

    A real roof edge steps once; a double wall steps up and back down within ``d_max`` cells
    and returns to the height it started from.  Scans every row and every column inside
    contiguous live segments of at least ``2 * d_max`` cells.  Unit: ratio.  Lower is better.
    """
    z = np.asarray(sel["z"], np.float64)
    live = np.asarray(sel["live"], bool)
    n_step = n_double = 0
    rates = []
    for axis in (0, 1):
        zz_all, ll_all = (z, live) if axis == 0 else (z.T, live.T)
        for zz, ll in zip(zz_all, ll_all):
            for seg in _segments(ll):
                if seg.stop - seg.start < max(2, 2 * d_max):
                    continue
                zs = zz[seg]
                dz = np.diff(zs)
                steps = np.nonzero(np.abs(dz) > h_tol)[0]
                n_step += int(steps.size)
                dbl = 0
                for a in range(steps.size):
                    i = steps[a]
                    for b in range(a + 1, steps.size):
                        j = steps[b]
                        if j - i > d_max:
                            break
                        if np.sign(dz[i]) == -np.sign(dz[j]) and abs(zs[i] - zs[j + 1]) <= h_tol:
                            dbl += 1
                            break
                n_double += dbl
                if steps.size:
                    rates.append(dbl / steps.size)
    return {"der_h": float(n_double / max(n_step, 1)),
            "der_h_p90": float(np.percentile(rates, 90)) if rates else float("nan"),
            "n_steps": int(n_step), "n_double": int(n_double),
            "n_lines": int(len(rates)), "h_tol": float(h_tol), "d_max": int(d_max)}


# --------------------------------------------------------------------------------------
# RA -- ripple amplitude (§2.5)
# --------------------------------------------------------------------------------------
def ripple_amplitude(store, sel, sigma_cells: float = 3.0,
                     flat_sigma: float = 0.3) -> dict:
    """RMS of the high-pass of the selected height field (§2.5), in metres.

    ``r(c) = z(c) - gaussian_filter(z, sigma=3 cells)``; only cells whose whole
    ``7x7`` neighbourhood is live and flat are used, because a hole boundary is not ripple.
    Holes are filled with the median before filtering so the filter does not ring into them.

    Units: m.  Lower is better; always report ``EW`` next to it (over-smoothing also lowers RA).
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    z = np.asarray(sel["z"], np.float64)
    live = np.asarray(sel["live"], bool)
    flat = _flat_mask(store, live, sigma_max=flat_sigma, min_n=2)
    if not live.any():
        return {"ra": float("nan"), "ra_p90": float("nan"), "n_cells": 0,
                "sigma_cells": float(sigma_cells)}
    fill = float(np.nanmedian(np.where(live, z, np.nan)))
    zf = np.where(live, z, fill)
    g = gaussian_filter(zf, sigma=sigma_cells, mode="wrap")
    r = z - g

    rad = int(np.ceil(sigma_cells))
    size = 2 * rad + 1
    full = uniform_filter(live.astype(np.float64), size=size, mode="wrap") >= 1.0 - 1e-9
    m = flat & full
    if not m.any():
        return {"ra": float("nan"), "ra_p90": float("nan"), "n_cells": 0,
                "sigma_cells": float(sigma_cells)}
    rv = r[m]
    return {"ra": float(np.sqrt(np.mean(rv ** 2))),
            "ra_p90": float(np.percentile(np.abs(rv), 90)),
            "n_cells": int(m.sum()), "sigma_cells": float(sigma_cells)}


# --------------------------------------------------------------------------------------
# EW -- edge width (§2.6)
# --------------------------------------------------------------------------------------
def edge_width(store, sel, cell: float | None = None, pct: float = 99.0) -> dict:
    """10-90% width of strong height edges, in metres (§2.6).

    Sampled along the gradient direction over ``[-6, +6]`` cells.  This is the guard against
    "smoothing the product until the ripple disappears": EW may not degrade by more than 10%
    against the baseline.  Lower is better.
    """
    from scipy.ndimage import map_coordinates

    z = np.asarray(sel["z"], np.float64)
    live = np.asarray(sel["live"], bool)
    cell = float(_g(store, "cell")) if cell is None else float(cell)
    if live.sum() < 9:
        return {"ew_m": float("nan"), "n_edges": 0, "pct": float(pct)}
    zf = np.where(live, z, np.nan)
    gy, gx = np.gradient(zf)
    g = np.hypot(gx, gy)
    gv = g[live & np.isfinite(g)]
    if gv.size == 0:
        return {"ew_m": float("nan"), "n_edges": 0, "pct": float(pct)}
    thr = float(np.percentile(gv, pct))
    ys, xs = np.nonzero(live & np.isfinite(g) & (g >= thr) & (g > 0))
    widths = []
    t = np.linspace(-6.0, 6.0, 49)
    for y, x in zip(ys, xs):
        ux, uy = gx[y, x] / g[y, x], gy[y, x] / g[y, x]
        yy, xx = y + t * uy, x + t * ux
        p = map_coordinates(zf, [yy, xx], order=1, mode="nearest")
        if not np.all(np.isfinite(p)):
            continue
        lo, hi = float(p.min()), float(p.max())
        if hi - lo < 1.0:
            continue
        i10 = int(np.argmax(p >= lo + 0.1 * (hi - lo)))
        i90 = int(np.argmax(p >= lo + 0.9 * (hi - lo)))
        if i90 <= i10:
            continue
        widths.append((t[i90] - t[i10]) * cell)
    return {"ew_m": float(np.median(widths)) if widths else float("nan"),
            "n_edges": int(len(widths)), "pct": float(pct)}


# --------------------------------------------------------------------------------------
# FR -- floater rate (§2.7)
# --------------------------------------------------------------------------------------
def _nanmedian5(a, chunk: int = 512):
    """5x5 nan-median with wrap-free edge handling (edges use whatever is available)."""
    H, W = a.shape
    out = np.full((H, W), np.nan)
    for r0 in range(0, H, chunk):
        r1 = min(r0 + chunk, H)
        stack = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                b = np.full((r1 - r0, W), np.nan)
                ys = slice(max(0, -dy), min(r1 - r0, H - r0 - dy))
                xs = slice(max(0, -dx), min(W, W - dx))
                src_y = slice(r0 + ys.start + dy, r0 + ys.stop + dy)
                src_x = slice(xs.start + dx, xs.stop + dx)
                b[ys, xs] = a[src_y, src_x]
                stack.append(b)
        with np.errstate(all="ignore"):
            out[r0:r1] = np.nanmedian(np.stack(stack), axis=0)
    return out


def floater_rate(store, sel, h_tol: float = 1.5) -> dict:
    """Share of live cells that are unsupported and far from their neighbourhood (§2.7).

    ``bad(c) = n_sel(c) < 2 and |z(c) - median5(z_supported)| > h_tol``.
    Support comes from the real ``n`` (never from ``opacity = min(1, n/2)``).  Unit: ratio.
    Lower is better.
    """
    from scipy.ndimage import label

    z = np.asarray(sel["z"], np.float64)
    live = np.asarray(sel["live"], bool)
    n_sel = np.asarray(sel["n"], np.float64)
    supported = live & (n_sel >= 2)
    med5 = _nanmedian5(np.where(supported, z, np.nan))
    bad = live & (n_sel < 2) & np.isfinite(med5) & (np.abs(z - med5) > h_tol)
    lab, n_clusters = label(bad)
    sizes = np.bincount(lab.ravel())[1:] if n_clusters else np.zeros(0)
    return {"fr": float(bad.sum() / max(int(live.sum()), 1)),
            "n_bad": int(bad.sum()), "n_live": int(live.sum()),
            "clusters": int(n_clusters),
            "cluster_p90": float(np.percentile(sizes, 90)) if sizes.size else 0.0,
            "h_tol": float(h_tol)}


# --------------------------------------------------------------------------------------
# coverage / eval cells
# --------------------------------------------------------------------------------------
def coverage_cells(store, bbox_cells: int | None = None) -> dict:
    """Live cells over the fixed bbox (§2.8).  A guard: no method may delete cells to win."""
    n = np.asarray(_g(store, "n"))
    live = (n > 0).any(-1)
    total = int(live.size) if bbox_cells is None else int(bbox_cells)
    return {"coverage_cells": float(live.sum() / max(total, 1)),
            "n_live": int(live.sum()), "n_cells": total}


def eval_cells_from_raw(store) -> np.ndarray:
    """``single-layer flat`` keys of the **build-frame raw product** (§2.0 rule 8).

    Computed once per fold from the build frames only; every method in that fold reuses it.
    Using held-out frames here would leak.
    """
    from stream_ortho.map.selection import pack_cell

    occ, _, sigma, n = _phys(store)
    flat = (occ.sum(-1) == 1) & (_take_occupied(occ, sigma) <= 0.3) & (_take_occupied(occ, n) >= 2)
    jj, ii = np.nonzero(flat)
    return pack_cell(ii + int(_g(store, "ox")), jj + int(_g(store, "oy")))
