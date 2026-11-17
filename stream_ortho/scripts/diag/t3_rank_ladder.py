"""T3 pre-step: the **guarded rank ladder** on the 2-observation columns.

T2 fitted an unconstrained per-frame affine and reported 0.006 m for the head source; the
review showed that fit drives the depth scale to ``a = 0.0001`` and flattens the relief to 55%
of its IQR, i.e. the residual was bought by destroying the surface.  This script redoes the
ladder the way the review requires:

    rank-1   z' = z - b_i
    rank-3   z' = z - (b_i + tx_i u + ty_i v)          (a = 1: relief preserved)
    rank-4   z' = a_i z + b_i + tx_i u + ty_i v        (with a penalty sweep or a in [0.95,1.05])

and reports, for every rung, the residual on the 2-observation flat columns **plus** the
guards ``z_iqr_ratio`` and ``median_z_shift``.  A rung whose IQR ratio leaves [0.95, 1.05] is
marked ``flattened`` and must not enter the E1 main table.

Usage::

    python -m stream_ortho.scripts.diag.t3_rank_ladder \
        --merged result-new/P0/obs/ieu/pointmap/obs_merged.npz \
        --out results/week10-sync/ieu
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

FLAT_SIGMA, FLAT_MIN_N = 0.12, 3
PENALTIES = (0.0, 1e-2, 1.0, 1e2, 1e4)


def _row(name, r, *, extra=None):
    p = r.get("params", {})
    a = np.array([v[0] for k, v in p.items() if k != 0]) if p else np.zeros(0)
    out = {"rung": name, "residual_m": r["median_abs_dz_after"], "n_columns": r["n_columns"],
           "z_iqr_ratio": r.get("z_iqr_ratio"), "median_z_shift": r.get("median_z_shift"),
           "median_z_shift_raw": r.get("median_z_shift_raw"),
           "guard_fail": bool(r.get("z_iqr_ratio") is not None and
                              not (0.95 <= r.get("z_iqr_ratio", 0) <= 1.05)),
           "guard_reason": (None if r.get("z_iqr_ratio") is None else
                            ("flattened" if r["z_iqr_ratio"] < 0.95 else
                             ("inflated" if r["z_iqr_ratio"] > 1.05 else None)))}
    if a.size:
        out["a_median"] = float(np.median(a))
        out["a_min"] = float(a.min())
        out["a_max"] = float(a.max())
    if extra:
        out.update(extra)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--source", default="pointmap")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from stream_ortho.eval.obs_stats import fit_rank1, fit_rank3, fit_rank4

    z = np.load(a.merged)
    key, kf = z["key"], z["kf"].astype(np.int64)
    zv, zs, nn = z["z"].astype(np.float64), z["z_std"].astype(np.float64), z["n"].astype(np.float64)
    xv, yv = z["x"].astype(np.float64), z["y"].astype(np.float64)
    flat = (zs < FLAT_SIGMA) & (nn >= FLAT_MIN_N)

    def fit(fn, **kw):
        return fn(key, kf, zv, mask=flat, two_obs_only=True, **kw)

    r1 = fit(fit_rank1)
    r3 = fit(fit_rank3, x=xv, y=yv)
    r4_free = fit(fit_rank4, x=xv, y=yv)
    r4_clip = fit(fit_rank4, x=xv, y=yv, a_range=(0.95, 1.05))
    ridge = 1e-3                                      # the conditioning used by E1
    r1_cond = fit_rank1(key, kf, zv, mask=flat, two_obs_only=True)
    r3_cond = fit(fit_rank3, x=xv, y=yv, ridge=ridge)
    r4_cond = fit(fit_rank4, x=xv, y=yv, a_range=(0.95, 1.05), ridge=ridge)

    ladders = [_row("rank-1 (b)", r1), _row("rank-3 (b,tx,ty)", r3),
               _row("rank-4 free (a z)", r4_free),
               _row("rank-4 a∈[0.95,1.05]", r4_clip),
               _row("rank-1 ridge=1e-3", r1_cond),
               _row("rank-3 ridge=1e-3", r3_cond),
               _row("rank-4 ridge=1e-3 a-clip", r4_cond)]

    sweep = []
    for lam in PENALTIES:
        if lam == 0.0:
            continue
        r = fit(fit_rank4, x=xv, y=yv, a_penalty=lam)
        sweep.append(_row(f"rank-4 penalty λ={lam:g}", r, extra={"lambda": lam}))

    payload = {
        "seq": a.seq, "source": a.source, "merged": a.merged,
        "definition": "2-observation columns, both rows flat (z_std<0.12, n>=3); "
                      "gauge = frame 0 identity; guards measured after constant alignment",
        "n_rows": int(key.size), "flat_frac": float(flat.mean()),
        "ladder": ladders, "penalty_sweep": sweep,
        "raw_2obs_median_m": None,
    }
    from stream_ortho.eval.obs_stats import cross_frame_dz
    payload["raw_2obs_median_m"] = cross_frame_dz(key, kf, zv)["median_abs_dz"]
    payload["raw_2obs_flat_m"] = cross_frame_dz(key, kf, zv, mask=flat)["median_abs_dz"]

    out = pathlib.Path(a.out) if a.out else pathlib.Path("results/week10-sync") / a.seq
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"rank_ladder_{a.source}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{a.seq}/{a.source}  raw 2-obs {payload['raw_2obs_median_m']:.3f} m "
          f"(flat {payload['raw_2obs_flat_m']:.3f})")
    print(f"{'rung':24s} {'resid(m)':>9s} {'IQR':>6s} {'shift':>6s} {'a_med':>7s} flatten")
    for row in ladders + sweep:
        iqr = row.get("z_iqr_ratio")
        sh = row.get("median_z_shift")
        am = row.get("a_median")
        print(f"{row['rung']:24s} {row['residual_m']:9.3f} "
              f"{iqr if iqr is not None else float('nan'):6.3f} "
              f"{sh if sh is not None else float('nan'):6.3f} "
              f"{am if am is not None else float('nan'):7.3f} {row.get('guard_reason')}")
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
