"""The **single** observation-to-product matching used by every HFR/audit script.

Two definitions, always reported together (week9-17 audit):

* ``top``  -- residual to the **highest** occupied layer of the cell (what the product
  renders by default);
* ``near`` -- residual to the **closest** occupied layer (what a fused/selective consumer
  would use; it keeps the layer-mixing observations that ``top`` throws away).

The previous ``eval/hfr.py`` used ``min(-1)``, i.e. the **lowest** layer (ground), and then
dropped everything with ``|dz| > 3 m`` -- rooftop observations silently fell out and the
metric measured the ground only.  That is why the number was not comparable with the audits
(which used the top layer).
"""
from __future__ import annotations

import numpy as np


def layers_of(frame, store, cell: float):
    """Per-observation (z_obs, z_top, z_near, n_layers, sigma_z_top, support_top)."""
    from stream_ortho.map.selection import unpack_cell

    ix, iy = unpack_cell(np.asarray(frame.cell_key, np.int64))
    j = iy - store.oy
    i = ix - store.ox
    ok = (i >= 0) & (i < store.W) & (j >= 0) & (j < store.H)
    z = np.asarray(frame.z, np.float64)
    L = store.n.shape[-1]
    out = {"ok": ok}
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return out
    jj, ii = j[idx], i[idx]
    occ = store.n[jj, ii] > 0                      # (N, L)
    zz = store.z_ref[jj, ii]                       # (N, L)
    top = np.where(occ, zz, -np.inf).argmax(-1)
    ar = np.arange(len(top))
    z_top = zz[ar, top]
    d = np.where(occ, np.abs(z[idx][:, None] - zz), np.inf)
    near = d.argmin(-1)
    z_near = zz[ar, near]
    sw = np.maximum(store.sum_w[jj, ii], 1e-6)
    mz = store.sum_wz[jj, ii] / sw
    var = np.maximum(store.sum_wz2[jj, ii] / sw - mz * mz, 0.0)
    out.update(idx=idx, z_obs=z[idx], z_top=z_top, z_near=z_near,
               n_layers=occ.sum(-1), support_top=store.n[jj, ii][ar, top],
               sigma_top=np.sqrt(var[ar, top]), near_is_top=(near == top),
               layer=occ)
    return out


def residuals(frame, store, cell: float, max_dz: float = 3.0, no_drop: bool = False) -> dict:
    """Residual sets plus the dropped fraction, for both definitions.

    ``no_drop=False`` keeps the week9 behaviour (``|dz| > max_dz`` dropped) so archived
    numbers stay reproducible; ``no_drop=True`` is the **frozen P0** behaviour (nothing is
    dropped, the tail is reported by ``hfr.tail_frac`` instead).
    """
    d = layers_of(frame, store, cell)
    if "idx" not in d:
        return {"top": np.zeros(0), "near": np.zeros(0), "dropped_frac": np.nan,
                "n": 0, "idx": np.zeros(0, int), "d": d}
    r_top = np.abs(d["z_obs"] - d["z_top"])
    r_near = np.abs(d["z_obs"] - d["z_near"])
    if no_drop:
        # frozen P0 definition (06 §2.0 rule 7): never drop a single observation
        keep_top = np.isfinite(r_top)
        keep_near = np.isfinite(r_near)
    else:
        keep_top = np.isfinite(r_top) & (r_top <= max_dz)
        keep_near = np.isfinite(r_near) & (r_near <= max_dz)
    return {"top": r_top[keep_top], "near": r_near[keep_near],
            "dropped_frac": float(1.0 - keep_near.mean()),
            "n": int(len(d["z_obs"])), "idx": d["idx"],
            "keep_top": keep_top, "keep_near": keep_near, "d": d}


def median_pair(frame, store, cell: float, max_dz: float = 3.0, no_drop: bool = False) -> dict:
    r = residuals(frame, store, cell, max_dz, no_drop=no_drop)
    return {"top": float(np.median(r["top"])) if r["top"].size else float("nan"),
            "near": float(np.median(r["near"])) if r["near"].size else float("nan"),
            "dropped_frac": r["dropped_frac"], "n": r["n"]}
