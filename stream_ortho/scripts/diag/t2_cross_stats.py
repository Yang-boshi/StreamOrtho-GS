"""Cross-frame consistency statistics of the observation table (the G1 gate numbers).

Reads ``obs_merged.npz`` and writes ``cross_stats.json`` next to it, plus
``cross_hist.npz`` (histograms + the frames/cell raster) for the figures.

Definitions (``03-P0-2`` §4; implementations in ``eval/obs_stats.py``):

* ``dz_two_obs_median``  ``|z_i - z_j|`` over columns with **exactly two** frames -- 5.85 m
* ``dz_two_obs_flat``    the same restricted to **flat** rows (``z_std<0.12 & n>=3``), i.e. the
                         set the rank ladders are fitted on
* ``rank1_after``        after one constant per frame (``z' = z - b_i``) -- 1.59 m
* ``rank4_after``        after a per-frame affine ``z' = a z + b + c x + d y`` -- 1.16 m
* ``fcr``                relief (max-min) of flat columns inside one frame -- 5.42 m

These are descriptive: no gate is attached here (the gate is 4.277 frames/cell).

Usage::

    python -m stream_ortho.scripts.diag.t2_cross_stats \
        --merged result-new/P0/obs/ieu/pointmap/obs_merged.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

FLAT_SIGMA = 0.12
FLAT_MIN_N = 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-samples", type=int, default=2_000_000,
                    help="subsample kept for the histograms")
    a = ap.parse_args(argv)

    from stream_ortho.eval.obs_stats import (
        cross_frame_dz, fcr, fit_rank1, fit_rank4, frames_per_cell, two_observation_pairs)
    from stream_ortho.map.selection import unpack_cell

    path = pathlib.Path(a.merged)
    z = np.load(path)
    key, kf = z["key"], z["kf"].astype(np.int64)
    zv, z_std, nn = z["z"].astype(np.float64), z["z_std"].astype(np.float64), z["n"].astype(np.float64)
    xv, yv = z["x"].astype(np.float64), z["y"].astype(np.float64)
    flat = (z_std < FLAT_SIGMA) & (nn >= FLAT_MIN_N)

    d_all = cross_frame_dz(key, kf, zv)
    d_flat = cross_frame_dz(key, kf, zv, mask=flat)
    r1 = fit_rank1(key, kf, zv, mask=flat)
    r4 = fit_rank4(key, kf, zv, x=xv, y=yv, mask=flat)

    # held-out check: fit the per-frame affine on the even pairs, evaluate on the odd ones.
    # A model that only wins on its own training pairs is over-fitting the ladder.
    import numpy as _np
    from stream_ortho.eval.obs_stats import _pair_rows
    _i0, _i1 = _pair_rows(key, kf, mask=flat)
    _even = (_np.arange(_i0.size) % 2) == 0
    r4h = fit_rank4(key, kf, zv, x=xv, y=yv, mask=flat, pair_mask=~_even)
    r4h_fit = fit_rank4(key, kf, zv, x=xv, y=yv, mask=flat, pair_mask=_even)
    fc = fcr(key, kf, zv, z_std, nn, flat_sigma=FLAT_SIGMA, min_n=FLAT_MIN_N)

    # residuals on the flat two-observation set, for the figure (and a cross-check)
    i0, i1 = two_observation_pairs(key, kf)
    keep = flat[i0] & flat[i1]
    i0, i1 = i0[keep], i1[keep]
    b = np.asarray(r1["b"], np.float64)
    res1 = np.abs((zv[i0] - b[kf[i0]]) - (zv[i1] - b[kf[i1]]))
    pr = r4["params"]

    def _tf(idx):
        pk = kf[idx]
        out = np.empty(idx.size)
        for k in np.unique(pk):
            m = pk == k
            av, bv, cv, dv = pr[int(k)]
            out[m] = av * zv[idx][m] + bv + cv * xv[idx][m] + dv * yv[idx][m]
        return out

    res4 = np.abs(_tf(i0) - _tf(i1))
    dz_flat = np.abs(zv[i0] - zv[i1])

    stats = {
        "merged": str(path), "n_rows": int(key.size), "flat_frac": float(flat.mean()),
        "flat_sigma": FLAT_SIGMA, "flat_min_n": FLAT_MIN_N,
        "dz_two_obs_median": d_all["median_abs_dz"],
        "dz_two_obs_p90": d_all.get("p90_abs_dz"),
        "n_two_obs_columns": d_all["n_columns"],
        "dz_two_obs_flat": d_flat["median_abs_dz"],
        "n_two_obs_flat_columns": d_flat["n_columns"],
        "rank1_after": r1["median_abs_dz_after"], "rank1_n_columns": r1["n_columns"],
        "rank4_after": r4["median_abs_dz_after"], "rank4_n_columns": r4["n_columns"],
        "rank4_holdout_after": r4h["median_abs_dz_after"],
        "rank4_train_after": r4h_fit["median_abs_dz_after"],
        "rank4_holdout_n_columns": r4h["n_columns"],
        "rank1_reduction_pct": 100.0 * (1 - r1["median_abs_dz_after"] / d_flat["median_abs_dz"]),
        "rank4_reduction_pct": 100.0 * (1 - r4["median_abs_dz_after"] / d_flat["median_abs_dz"]),
        "residual_check_rank1": float(np.median(res1)),
        "residual_check_rank4": float(np.median(res4)),
        "fcr": fc["fcr"], "fcr_p90": fc.get("fcr_p90"), "n_flat_columns": fc["n_columns"],
        "rank1_offsets": [float(v) for v in b],
        "rank4_params": {int(k): [float(t) for t in v] for k, v in pr.items()},
    }

    # ---- histograms + frames/cell raster for the figures ----
    comp = np.unique(key * 64 + kf)
    cells, cnt = np.unique(comp // 64, return_counts=True)
    ix, iy = unpack_cell(cells)
    H, W = int(iy.max() - iy.min() + 1), int(ix.max() - ix.min() + 1)
    grid = np.zeros((H, W), np.float32)
    grid[iy - iy.min(), ix - ix.min()] = cnt.astype(np.float32)

    rng = np.random.default_rng(0)

    def sub(v):
        v = np.asarray(v, np.float64)
        return v if v.size <= a.max_samples else rng.choice(v, a.max_samples, replace=False)

    out = pathlib.Path(a.out) if a.out else path.with_name("cross_hist.npz")
    np.savez(out, grid_frames_per_cell=grid, grid_ix0=int(ix.min()), grid_iy0=int(iy.min()),
             cell=float(z["cell"]),
             dz_flat=sub(dz_flat).astype(np.float32),
             dz_rank1=sub(res1).astype(np.float32),
             dz_rank4=sub(res4).astype(np.float32),
             rank1_offsets=b.astype(np.float32))

    (path.with_name("cross_stats.json")).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in (
        "dz_two_obs_median", "dz_two_obs_flat", "rank1_after", "rank4_after", "fcr",
        "n_two_obs_columns", "rank1_reduction_pct", "rank4_reduction_pct")},
        indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
