"""P3.5 diagnostics: are roofs/ground missing because of the product, or the rendering?

For a run directory (which holds ``offline/product_frozen.npz``) plus the observation table it
came from, this computes:

  coverage            live cells / cells the observations touch
  roof_retention      cells whose observations reach >3 m above ground and whose product still
                      has a live layer near that height
  ground_retention    the same for cells whose observations are <1 m above ground
  floater_count       product cells whose top is >2.5 m above the local p10
  edge_strip_score    inside the edge band, the mean (n - 1) per live cell: duplicated
                      observations, i.e. the "stripes" nearest-neighbour upsampling leaves
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load_product(path):
    d = np.load(path)
    n = d["n"]
    z = d["z_ref"]
    occ = n > 0
    top = np.where(occ, z, -np.inf).argmax(-1)
    z_top = np.take_along_axis(z, top[..., None], -1)[..., 0]
    live = occ.any(-1)
    return dict(n=n, z=z, live=live, z_top=z_top)


def load_obs(path):
    d = np.load(path)
    return d


def grid_of(prod, obs):
    """Align the observation cells with the product grid (same origin/cell)."""
    cell = float(obs["cell"])
    ix = np.asarray(obs["ix"], np.int64)
    iy = np.asarray(obs["iy"], np.int64)
    ox, oy = ix.min(), iy.min()
    H, W = prod["n"].shape[:2]
    jx = np.clip(ix - ox, 0, W - 1)
    jy = np.clip(iy - oy, 0, H - 1)
    return jx, jy, cell


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/system/p35_*")
    ap.add_argument("--obs", required=True, help="obs table dir (…/pointmap)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell", type=float, default=None)
    ap.add_argument("--visual", action="store_true")
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out or a.run) / "p35_diag.json"
    prod = load_product(pathlib.Path(a.run) / "offline" / "product_frozen.npz")
    obs = load_obs(pathlib.Path(a.obs) / "obs_merged.npz")
    jx, jy, cell = grid_of(prod, obs)
    zobs = np.asarray(obs["z"], np.float64)
    ground = float(np.percentile(zobs, 5))
    H, W = prod["n"].shape[:2]

    # per-cell observation extremes
    zmax = np.full((H, W), -np.inf); zmin = np.full((H, W), np.inf)
    np.maximum.at(zmax, (jy, jx), zobs)
    np.minimum.at(zmin, (jy, jx), zobs)
    seen = np.isfinite(zmax)
    roof_cells = seen & (zmax > ground + 3.0)
    ground_cells = seen & (zmin < ground + 1.0)

    live = prod["live"]; z_top = prod["z_top"]
    def retained(cells):
        if not cells.any():
            return 0.0
        ok = live & np.isfinite(z_top)
        return float((ok & cells).sum() / cells.sum())

    # roof: the product must keep something near the observed roof height, not just any cell
    roof_ok = np.zeros_like(live)
    occ = prod["n"] > 0
    z3 = prod["z"]
    for j in range(0, 1):        # vectorised below instead of a python loop
        pass
    near = (np.abs(z3 - zmax[..., None]) < 1.5) & occ
    roof_ok = near.any(-1)
    # a max-based test counts a floater above the roof as a "missing roof"; the per-cell
    # median of the elevated rows is the roof itself
    hi = zobs > ground + 2.0
    zmed = np.full((H, W), np.nan)
    if hi.any():
        sums = np.zeros((H, W)); cnt = np.zeros((H, W))
        np.add.at(sums, (jy[hi], jx[hi]), zobs[hi])
        np.add.at(cnt, (jy[hi], jx[hi]), 1.0)
        zmed = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    roof_cells_med = seen & np.isfinite(zmed) & (zmed > ground + 3.0)
    near_med = (np.abs(z3 - zmed[..., None]) < 1.5) & occ
    roof_ret_med = float((near_med.any(-1) & roof_cells_med).sum() /
                         max(roof_cells_med.sum(), 1))
    # dense roofs only: >=5 elevated observations.  Sparse cells are usually a stray floater,
    # which has no product support by design and should not count as a "lost roof".
    dense = roof_cells_med & (cnt >= 5) if hi.any() else roof_cells_med
    roof_ret_dense = float((near_med.any(-1) & dense).sum() / max(dense.sum(), 1))
    roof_ret = float((roof_ok & roof_cells).sum() / max(roof_cells.sum(), 1))
    ground_ret = retained(ground_cells)

    # floater count on the top surface
    from scipy.ndimage import minimum_filter
    zf = np.where(live, z_top, np.nan)
    med = minimum_filter(np.where(live, z_top, np.nanmax(z_top[live])), size=5)
    floaters = live & ((zf - med) > 2.5)

    # edge strip score: duplicates per cell in the edge band
    g = np.hypot(*np.gradient(np.where(live, z_top, np.nanmean(z_top[live]))))
    band = live & (g > np.percentile(g[live], 85))
    nsum = prod["n"].sum(-1)
    strip = float((nsum[band] - 1).mean()) if band.any() else 0.0

    res = {"run": a.run, "obs": a.obs, "cell": cell,
           "coverage": float(live.sum() / max(seen.sum(), 1)),
           "roof_retention": roof_ret,
           "roof_retention_median": roof_ret_med,
           "roof_retention_dense": roof_ret_dense,
           "ground_retention": ground_ret,
           "floater_count": int(floaters.sum()),
           "floater_frac_of_live": float(floaters.sum() / max(live.sum(), 1)),
           "edge_strip_score": strip,
           "n_cells_obs": int(seen.sum()), "n_cells_live": int(live.sum()),
           "ground_z": ground}
    # visual: where the product top is *below* the observed top by >2 m (lost roofs) and
    # where it is above by >2 m (floaters), overlaid on the top surface as a height map
    if a.visual:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        diff = np.where(live & np.isfinite(zmax), zmax - z_top, np.nan)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(np.where(live, z_top, np.nan), cmap="viridis")
        ax[0].set_title("product top surface", fontsize=9); plt.colorbar(im, ax=ax[0], fraction=0.04)
        ov = np.zeros((*diff.shape, 3))
        ov[..., 0] = np.clip(np.nan_to_num(diff, nan=0) / 4.0, 0, 1)      # red: roof missing
        ov[..., 2] = np.clip(-np.nan_to_num(diff, nan=0) / 4.0, 0, 1)     # blue: floater
        ax[1].imshow(np.clip(ov, 0, 1))
        ax[1].set_title("red = product below observed top (>0), blue = above (<0)", fontsize=9)
        for axx in ax:
            axx.set_xticks([]); axx.set_yticks([])
        fig.tight_layout(); fig.savefig(out.parent / "p35_missing_roofs.png"); plt.close(fig)
        res["missing_roof_cells"] = int(np.nansum(diff > 2.0))
        res["floater_cells"] = int(np.nansum(diff < -2.0))
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
