"""Render orthophoto maps (RGB + hillshade) for a product, raw vs calibrated.

The product stores a colour per layer (``build_product(..., rgb=...)``); the frozen rebuild
used by the calibration only re-estimates the heights, so the colours are re-aggregated over
the same frozen assignment.  Two products are rendered:

* **raw**: the uncalibrated product;
* **hybrid**: the frozen main method (neighbour consensus + 0.1 product prior, 3 rounds).

Outputs (``<out>/ortho/``):

    ortho_raw.png / ortho_hybrid.png        full extent, 2x downsampled
    crop_<name>_compare.png                 raw | hybrid side by side at full resolution
    hillshade_compare.png                   the same crops in hillshade

Usage::

    python -m stream_ortho.scripts.diag.render_ortho --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-pivotb/ieu --seq ieu --iters 3
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho.eval import pivotb_preflight as P
from stream_ortho.eval import sync_eval as S
from stream_ortho.map.calib import CalibConfig, FrozenProduct, apply_t3, fit_t3
from stream_ortho.scripts.diag.e6_neighbor_fit import fit_objective, weighted_median_rows


def calibrate_hybrid(d, slices, kfs, prod_raw, fp, rows, origin, cfg, iters, lam_p, k,
                     return_z=False):
    """Run the frozen hybrid calibration; returns the final product (layers/owner frozen)."""
    from stream_ortho.scripts.diag.e6_neighbor_fit import fit_objective, weighted_median_rows
    from stream_ortho.scripts.run_pivotb import neighbour_lists as NL

    nbrs = NL(d, slices, kfs, k)
    sub_of, order_of = {}, {}
    for kk in kfs:
        r = S._rows(d, slices[kk])
        o = np.argsort(r["key"], kind="stable")
        order_of[kk] = o
        sub_of[kk] = {k2: v[o] for k2, v in r.items()}

    prod = prod_raw
    z_cur = {kk: sub_of[kk]["z"].copy() for kk in kfs}
    for _ in range(iters):
        z_rows = rows["z"].copy()
        for kk in kfs:
            sub = sub_of[kk]
            zk = z_cur[kk]
            eval_cells = S.eval_cells_of(prod)
            base = S._intersect_sorted(S.single_layer_cells(sub["key"]), eval_cells)
            if base.size < 200:
                continue
            V = np.full((base.size, len(nbrs[kk])), np.nan)
            Wt = np.zeros((base.size, len(nbrs[kk])))
            for i, j in enumerate(nbrs[kk]):
                sj, zj = sub_of[j], z_cur[j]
                pos = np.clip(np.searchsorted(sj["key"], base), 0, max(sj["key"].size - 1, 0))
                hit = sj["key"][pos] == base
                V[hit, i] = zj[pos[hit]]
                Wt[hit, i] = 1.0
            z_ref = weighted_median_rows(V, Wt)
            wc = Wt.sum(1)
            xv = S._lookup(sub["key"], sub["x"], base)
            yv = S._lookup(sub["key"], sub["y"], base)
            zk_c = S._lookup(sub["key"], zk, base)
            pv, _ = S.product_near_z({**sub, "z": zk}, prod)
            z_prior = S._lookup(sub["key"], pv, base)
            A = np.stack([np.ones_like(xv), xv - origin[0], yv - origin[1]], 1)
            th = fit_objective(zk_c, z_ref, z_prior, A, wc, lam_p, cfg)
            A_full = np.stack([np.ones(sub["key"].size), sub["x"] - origin[0],
                               sub["y"] - origin[1]], 1)
            zk2 = zk - A_full @ th
            z_cur[kk] = zk2
            z_rows[slices[kk][order_of[kk]]] = zk2
        prod = fp.rebuild(z_rows, rows["conf"] * rows["n"])
    if return_z:
        return prod, z_cur
    return prod


def rgb_grid(assign, rgb, w, H, W, L):
    size = H * W * L
    ok = assign >= 0
    a = assign[ok]
    den = np.bincount(a, weights=w[ok], minlength=size)
    out = np.zeros((size, 3))
    for c in range(3):
        num = np.bincount(a, weights=(w * rgb[:, c])[ok], minlength=size)
        out[:, c] = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return out.reshape(H, W, L, 3), den.reshape(H, W, L)


def top_surface(prod, rgb_g):
    occ = prod["n"] > 0
    idx = np.where(occ, prod["z_ref"], -np.inf).argmax(-1)
    z = np.take_along_axis(prod["z_ref"], idx[..., None], -1)[..., 0]
    c = np.take_along_axis(rgb_g, idx[..., None, None], 2)[..., 0, :]
    live = occ.any(-1)
    return z, c, live


def shade(z, cell, live):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LightSource, Normalize
    zz = np.where(live, z, np.nanmedian(z[live]))
    lo, hi = np.percentile(zz[live], [2, 98])
    return LightSource(azdeg=315, altdeg=45).shade(
        zz, cmap=plt.get_cmap("gray"), norm=Normalize(lo, hi), blend_mode="soft",
        vert_exag=1.5, dx=cell, dy=cell)


def fill_holes(z, c, live):
    """Fill cells enclosed by coverage but never observed (display only).

    The fill copies the nearest covered cell; ``filled_frac`` is reported so the figure never
    silently invents a surface.
    """
    from scipy.ndimage import binary_fill_holes, distance_transform_edt
    inside = binary_fill_holes(live)
    holes = inside & ~live
    if not holes.any():
        return z, c, live, 0.0
    z2, c2 = z.copy(), c.copy()
    ind = distance_transform_edt(~live, return_distances=False, return_indices=True)
    z2[holes] = z[tuple(ind[:, holes])]
    for ch in range(3):
        c2[..., ch][holes] = c[..., ch][tuple(ind[:, holes])]
    return z2, c2, live | holes, float(holes.sum() / max(inside.sum(), 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="filtered observation table")
    ap.add_argument("--obs-raw", default=None,
                    help="unfiltered table for the raw panel (defaults to --obs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--lam-p", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--downsample", type=int, default=2)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = CalibConfig()
    d, ix, iy = P.load(a.obs)
    # the raw panel must come from the UNFILTERED table, otherwise filtering is invisible
    d_raw = d
    if a.obs_raw and pathlib.Path(a.obs_raw).resolve() != pathlib.Path(a.obs).resolve():
        d_raw, ix_raw, iy_raw = P.load(a.obs_raw)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    if d_raw is not d:
        bbox_raw = P._bbox(ix_raw, iy_raw, d_raw["cell"])
        bbox = (min(bbox[0], bbox_raw[0]), min(bbox[1], bbox_raw[1]),
                max(bbox[2], bbox_raw[2]), max(bbox[3], bbox_raw[3]))
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    # panel 1: raw (unfiltered, uncalibrated)
    if d_raw is not d:
        sl_raw = S.frame_slices(d_raw["kf"])
        kfs_raw = sorted(sl_raw)
        prod_unf, _, rows_unf = P.product(d_raw, ix_raw, iy_raw, kfs_raw, sl_raw, bbox,
                                          return_assign=True)
        rgb_unf, _ = rgb_grid(prod_unf["assign"], rows_unf["rgb"],
                              rows_unf["conf"] * rows_unf["n"], *prod_unf["n"].shape)
        z_unf, c_unf, live_unf = top_surface(prod_unf, rgb_unf)
    else:
        prod_unf, z_unf, c_unf, live_unf = None, None, None, None
    # panel 2: filtered, uncalibrated
    prod_raw, _, rows = P.product(d, ix, iy, kfs, slices, bbox, return_assign=True)
    fp = FrozenProduct.from_product(prod_raw)
    H, W, L = prod_raw["n"].shape
    assign = prod_raw["assign"]
    w_all = rows["conf"] * rows["n"]
    rgb_g_raw, _ = rgb_grid(assign, rows["rgb"], w_all, H, W, L)

    prod = calibrate_hybrid(d, slices, kfs, prod_raw, fp, rows, origin, cfg, a.iters, a.lam_p, a.k)

    z_hyb, c_hyb, live = top_surface(prod, rgb_g_raw)
    z_raw, c_raw, _ = top_surface(prod_raw, rgb_g_raw)

    # ---- optional product cleanup: drop isolated high floaters (NOT a method contribution) ----
    from scipy.ndimage import median_filter
    z_f = np.where(live, z_hyb, np.nan)
    med = median_filter(np.where(live, z_hyb, np.nanmedian(z_hyb[live])), size=21, mode="nearest")
    floater = live & ((z_f - med) > 2.5)
    live_clean = live & ~floater
    # fill the holes from the nearest valid cell so the figure has no black speckle
    from scipy.ndimage import distance_transform_edt
    hole = live & ~live_clean
    if hole.any():
        ind = distance_transform_edt(hole, return_distances=False, return_indices=True)
        z_fill = z_hyb.copy()
        c_fill = c_hyb.copy()
        z_fill[hole] = z_hyb[tuple(ind[:, hole])]
        for ch in range(3):
            c_fill[..., ch][hole] = c_hyb[..., ch][tuple(ind[:, hole])]
    else:
        z_fill, c_fill = z_hyb, c_hyb
    meta_clean = {"floater_frac_of_live": float(floater.sum() / max(live.sum(), 1)),
                  "coverage_after": float(live_clean.sum() / max(live.sum(), 1)),
                  "filled_frac": float(hole.sum() / max(live.sum(), 1)),
                  "coverage_after_fill": float((live_clean | hole).sum() / max(live.sum(), 1))}

    out = pathlib.Path(a.out) / "ortho"
    out.mkdir(parents=True, exist_ok=True)
    ds = a.downsample
    # three panels: raw (unfiltered) | filtered | filtered + frozen.  Showing the filter's own
    # effect is the point: with one table for both, raw and hybrid only differ by centimetres.
    panels = []
    if z_unf is not None:
        panels.append((*fill_holes(z_unf, c_unf, live_unf), "raw (unfiltered, uncalibrated)"))
    panels += [(*fill_holes(z_raw, c_raw, live), "filtered, uncalibrated"),
               (*fill_holes(z_hyb, c_hyb, live), "filtered + 3 rounds frozen")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.6))
    axes = np.atleast_1d(axes)
    for ax, (z, c, lv, filled, ttl) in zip(axes, panels):
        img = np.clip(c, 0, 255).astype(np.uint8)[::ds, ::ds]
        hs = shade(z, prod_raw["cell"], lv)[::ds, ::ds, :3]
        comp = np.clip(0.75 * img / 255.0 + 0.35 * hs, 0, 1)
        ax.imshow(comp, interpolation="nearest")
        cov = float(lv.mean())
        ax.set_title(f"{a.seq}: {ttl}\ncoverage {cov:.3f}, filled {filled:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(out / "ortho_compare.png"); plt.close(fig)
    (out / "ortho_panels.json").write_text(json.dumps(
        {"seq": a.seq, "obs": str(a.obs), "obs_raw": str(a.obs_raw or a.obs),
         "panels": [{"title": t, "coverage": float(lv.mean()), "filled_frac": filled}
                    for _z, _c, lv, filled, t in panels]},
        indent=2, ensure_ascii=False))

    # three-panel: raw | hybrid | hybrid + floater cleanup
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    hs_l = shade(z_hyb, prod_raw["cell"], live)
    for ax, (z, c, lv, ttl) in zip(axes, ((z_raw, c_raw, live, "raw"),
                                          (z_hyb, c_hyb, live, "hybrid0.1"),
                                          (z_fill, c_fill, live, "hybrid + floater fill"))):
        img = np.clip(c, 0, 255).astype(np.uint8)[::ds, ::ds]
        hs = shade(z, prod_raw["cell"], lv)[::ds, ::ds, :3]
        comp = np.clip(0.75 * img / 255.0 + 0.35 * hs, 0, 1)
        comp[~lv[::ds, ::ds]] = 1.0   # no-data -> paper white, never black
        ax.imshow(comp, interpolation="nearest")
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"{a.seq}: floaters are inherited from the feed-forward geometry -- "
                 f"{100*meta_clean['floater_frac_of_live']:.1f}% flagged, filled from the nearest "
                 f"valid cell (coverage {100*meta_clean['coverage_after_fill']:.1f}%)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ortho_cleanup.png"); plt.close(fig)

    # ---- crops ----
    ext = np.nonzero(live)
    ci, cj = ext[0], ext[1]
    centres = [(int(np.median(ci)), int(np.median(cj))),
               (int(np.percentile(ci, 25)), int(np.percentile(cj, 70))),
               (int(np.percentile(ci, 75)), int(np.percentile(cj, 30)))]
    half = 150
    for ci_, cj_ in centres:
        i0, i1 = max(ci_ - half, 0), min(ci_ + half, H)
        j0, j1 = max(cj_ - half, 0), min(cj_ + half, W)
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6))
        axes = np.atleast_1d(axes)
        for ax, (z, c, lv, filled, ttl) in zip(axes, panels):
            img = np.clip(c[i0:i1, j0:j1], 0, 255).astype(np.uint8)
            hs = shade(z[i0:i1, j0:j1], prod_raw["cell"], lv[i0:i1, j0:j1])[..., :3]
            ax.imshow(np.clip(0.75 * img / 255.0 + 0.35 * hs, 0, 1), interpolation="nearest")
            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.suptitle(f"crop at i={ci_}, j={cj_} (150 cells = {150*prod_raw['cell']:.0f} m)",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"crop_{ci_}_{cj_}_compare.png"); plt.close(fig)

    meta = {"seq": a.seq, "cell": prod_raw["cell"], "iters": a.iters, "lam_p": a.lam_p, "K": a.k,
            "grid": [int(H), int(W), int(L)], "live_cells": int(live.sum()),
            "crops": [[int(c[0]), int(c[1])] for c in centres],
            "z_shift_median_m": float(np.median(z_hyb[live] - z_raw[live])),
            "z_shift_p95_abs_m": float(np.percentile(np.abs(z_hyb[live] - z_raw[live]), 95)),
            **meta_clean}
    (out / "ortho_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
