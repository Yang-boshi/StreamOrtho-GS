"""HFR -- held-out frame residual, **frozen** definition (``refs/06-任务卡-P0.md`` §2.1).

Definition
----------
For every held-out observation ``o = (kf, x, y, z_o)`` the product grid cell is found from
the *real* plane coordinates ``(x, y)`` (never from a cell index, because ``sync_cell`` and
``prod_cell`` may differ), then two residuals are formed:

``r_top``   distance to the **highest** occupied layer (what the product renders by default)
``r_near``  distance to the **closest** occupied layer (per observation)

``top`` is the rendered surface but throws away observations under a roof; ``near`` keeps
them but adapts to the observation.  Both are always reported; the E1 gate uses ``near``.

Hard rules (the bugs this file exists to avoid)
----------------------------------------------
* **no truncation**: ``|dz| > tail`` is *not* dropped from the median; it is summarised by
  ``tail_frac``.  Truncating is non-monotone: repairing a 10 m outlier to 2.9 m would look
  *worse* if the three good rows also moved by 0.1 m.
* **row-wise**: one row per ``(cell, layer)``; a multi-layer frame is never collapsed with
  ``f.z[idx, None]`` (that silently misaligns facade cells).
* the caller guarantees that held-out frames never entered the product (leakage rule).

Units: metres.  Direction: lower is better.
"""
from __future__ import annotations

import numpy as np


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def hfr(held_frames, product, tail: float = 3.0) -> dict:
    """Held-out frame residual.

    Parameters
    ----------
    held_frames : list of dict
        Each dict has ``kf`` (int), ``x`` (N,), ``y`` (N,), ``z`` (N,) -- one row per
        ``(cell, layer)`` observation.  Rows of different frames may have different sizes.
    product : dict or ColumnStore-like
        Fields ``n`` (H,W,L), ``z_ref`` (H,W,L), ``x0``, ``y0``, ``cell``, ``W``, ``H``.
    tail : float
        Threshold for ``tail_frac`` only; never removes a row from the medians.

    Returns
    -------
    dict with ``top_pooled``, ``top_p90``, ``near_pooled``, ``near_p90``, ``near_frame``,
    ``tail_frac``, ``outside_frac``, ``empty_frac``, ``n_obs``, ``per_frame``.
    """
    x0, y0 = float(_get(product, "x0")), float(_get(product, "y0"))
    cell = float(_get(product, "cell"))
    W, Hgt = int(_get(product, "W")), int(_get(product, "H"))
    n_arr = np.asarray(_get(product, "n"))
    z_arr = np.asarray(_get(product, "z_ref"))
    n_flat = n_arr.reshape(-1, n_arr.shape[-1])
    z_flat = z_arr.reshape(-1, z_arr.shape[-1])

    R_top, R_near, per_frame = [], [], []
    n_rows = n_out = n_empty = 0

    for f in held_frames:
        xs = np.asarray(f["x"], np.float64)
        ys = np.asarray(f["y"], np.float64)
        zs = np.asarray(f["z"], np.float64)
        i = np.floor((xs - x0) / cell).astype(np.int64)
        j = np.floor((ys - y0) / cell).astype(np.int64)
        ok = (i >= 0) & (i < W) & (j >= 0) & (j < Hgt)
        rows = np.nonzero(ok)[0]
        n_rows += len(zs)
        n_out += len(zs) - len(rows)
        if rows.size == 0:
            continue
        cell_idx = j[rows] * W + i[rows]
        occ = n_flat[cell_idx] > 0                      # (M, L)
        zz = z_flat[cell_idx]                           # (M, L)
        z_o = zs[rows]
        has = occ.any(-1)
        n_empty += int((~has).sum())
        if not has.any():
            continue
        occ, zz, z_o = occ[has], zz[has], z_o[has]

        # top: highest occupied layer, independent of the observation
        z_top = np.where(occ, zz, -np.inf).max(-1)
        # near: closest occupied layer, per observation
        d = np.where(occ, np.abs(z_o[:, None] - zz), np.inf)
        near = d.argmin(-1)
        ar = np.arange(len(near))
        z_near = zz[ar, near]

        r_top = np.abs(z_o - z_top)
        r_near = np.abs(z_o - z_near)
        R_top.append(r_top)
        R_near.append(r_near)
        per_frame.append((int(f["kf"]) if "kf" in f else -1, float(np.median(r_near))))

    n_in = n_rows - n_out
    out = {
        "top_pooled": float("nan"), "top_p90": float("nan"),
        "near_pooled": float("nan"), "near_p90": float("nan"),
        "near_frame": float("nan"), "tail_frac": float("nan"),
        "outside_frac": float(n_out / max(n_rows, 1)),
        "empty_frac": float("nan"),
        "n_obs": 0, "per_frame": [],
        "n_rows": int(n_rows), "n_outside": int(n_out), "n_empty": int(n_empty),
        "tail": float(tail),
    }
    if not R_top:
        return out
    Rt, Rn = np.concatenate(R_top), np.concatenate(R_near)
    out.update({
        "top_pooled": float(np.median(Rt)),
        "top_p90": float(np.percentile(Rt, 90)),
        "near_pooled": float(np.median(Rn)),
        "near_p90": float(np.percentile(Rn, 90)),
        "near_frame": float(np.median([v for _, v in per_frame])),
        "tail_frac": float(np.mean(Rn > tail)),
        "empty_frac": float(n_empty / max(n_in, 1)),
        "n_obs": int(len(Rn)),
        "per_frame": per_frame,
    })
    return out


def surface_guards(product, product_raw, iqr_tol=(0.95, 1.05), shift_max: float = 0.5) -> dict:
    """Anti-cheat guards: a method must not buy HFR by flattening or shifting the product.

    ``z_IQR_ratio``    IQR(product z) / IQR(raw product z) -- must stay inside ``iqr_tol``
    ``median_z_shift`` |median(product z) - median(raw z)| -- must stay below ``shift_max``

    Only occupied layers (``n > 0``) are used; empty slots hold ``z_ref = 0`` and would
    otherwise dominate both statistics.
    """
    def _values(p):
        z = np.asarray(_get(p, "z_ref"), np.float64)
        n = np.asarray(_get(p, "n"))
        return z[n > 0]

    z_new, z_raw = _values(product), _values(product_raw)
    q25, q75 = np.percentile(z_raw, [25, 75])
    iqr_raw = float(q75 - q25)
    if z_new.size == 0 or iqr_raw <= 0:
        ratio = float("nan")
    else:
        q25n, q75n = np.percentile(z_new, [25, 75])
        ratio = float((q75n - q25n) / iqr_raw)
    shift = (float(abs(np.median(z_new) - np.median(z_raw)))
             if z_new.size and z_raw.size else float("nan"))
    iqr_ok = bool(np.isfinite(ratio) and iqr_tol[0] <= ratio <= iqr_tol[1])
    shift_ok = bool(np.isfinite(shift) and shift < shift_max)
    return {"z_iqr_ratio": ratio, "median_z_shift": shift,
            "iqr_ok": iqr_ok, "shift_ok": shift_ok,
            "pass": bool(iqr_ok and shift_ok),
            "iqr_tol": list(iqr_tol), "shift_max": float(shift_max)}


# --------------------------------------------------------------------------------------
# backwards-compatible helper (week9 callers imported this name)
# --------------------------------------------------------------------------------------
def residual(frame, store, cell: float, max_dz: float = 3.0):
    """Legacy nearest-layer median; delegates to the ported ``eval.match`` implementation."""
    from stream_ortho.eval.match import median_pair
    return median_pair(frame, store, cell, max_dz)["near"]
