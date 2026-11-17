"""P4-B3: why does the product keep only ~87% of the roofs?

For every cell whose observations reach >3 m above ground, the audit asks what happened to it:

  layers_in_obs      how many distinct layers the observation table gives the cell
  truncated          would max_layers have cut a layer off? (layers_in_obs > max_layers)
  merged_by_zgap     do the roof and the ground rows fall inside one z_gap, i.e. did the
                     layering merge them into a single height?
  in_product         does the product have any live layer for the cell
  roof_hit           does it have one within +-1.5 m of the observed roof height

The counts tell whether the loss comes from the layer budget, the z-gap, or the rebuild.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--max-layers", type=int, default=3)
    ap.add_argument("--z-gap", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--visual", action="store_true")
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out or a.run)
    prod = np.load(pathlib.Path(a.run) / "offline" / "product_frozen.npz")
    n, z = prod["n"], prod["z_ref"]
    occ = n > 0
    z_top = np.take_along_axis(z, np.where(occ, z, -np.inf).argmax(-1)[..., None], -1)[..., 0]
    live = occ.any(-1)
    H, W = n.shape[:2]

    obs = np.load(pathlib.Path(a.obs) / "obs_merged.npz")
    ix, iy = np.asarray(obs["ix"], np.int64), np.asarray(obs["iy"], np.int64)
    zz = np.asarray(obs["z"], np.float64)
    lay = np.asarray(obs["layer"], np.int32)
    ox, oy = ix.min(), iy.min()
    jx = np.clip(ix - ox, 0, W - 1); jy = np.clip(iy - oy, 0, H - 1)
    ground = float(np.percentile(zz, 5))

    cell = (jy * W + jx).reshape(-1)
    nlay = np.zeros(H * W, np.int32)
    zlo = np.full(H * W, np.inf); zhi = np.full(H * W, -np.inf)
    np.maximum.at(nlay, cell, lay)
    np.minimum.at(zlo, cell, zz); np.maximum.at(zhi, cell, zz)
    nlay = nlay + 1
    seen = np.isfinite(zlo)
    nlay[~seen] = 0
    nlay2 = nlay.reshape(H, W); zlo2 = zlo.reshape(H, W); zhi2 = zhi.reshape(H, W)
    roof = seen.reshape(H, W) & (zhi2 > ground + 3.0)

    zhi_g = np.where(roof, zhi2, np.nan)
    near = (np.abs(z - zhi_g[..., None]) < 1.5) & occ
    roof_hit = near.any(-1)
    r = roof
    nr = max(int(r.sum()), 1)
    trunc = r & (nlay2 > a.max_layers)
    keep = r & (nlay2 <= a.max_layers)
    res = {"run": a.run, "cell": float(obs["cell"]), "max_layers": a.max_layers,
           "z_gap": a.z_gap, "n_roof_cells": int(r.sum()),
           "roof_in_product": float((roof_hit & r).sum() / nr),
           "truncated_cells": int(trunc.sum()),
           "truncated_frac": float(trunc.sum() / nr),
           "merged_by_zgap_frac": float((r & ((zhi2 - zlo2) < 1.5)).sum() / nr),
           "roof_in_product_truncated": float((roof_hit & trunc).sum() / max(trunc.sum(), 1)),
           "roof_in_product_kept": float((roof_hit & keep).sum() / max(keep.sum(), 1)),
           # split the loss: is the cell missing entirely, or present at the wrong height?
           "cell_live_frac": float((live & r).sum() / nr),
           "roof_hit_given_live": float((roof_hit & live & r).sum() / max((live & r).sum(), 1)),
           "z_err_p50_m": float(np.nanmedian(np.where(live & r, zhi2 - z_top, np.nan))),
           "z_err_p90_m": float(np.nanpercentile(np.where(live & r, zhi2 - z_top, np.nan), 90)),
           "layer_count_hist": {int(k): int(v) for k, v in
                                zip(*np.unique(nlay2[r], return_counts=True))}}
    (out / "layer_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
