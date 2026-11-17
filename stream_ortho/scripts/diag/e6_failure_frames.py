"""T7-A -- why do a handful of held-out frames get WORSE after the 3-DOF correction?

E6 is a structure effect (not coverage), but the median hides frames like kf22 (-77.7%).
For the failing frames and two good controls per direction this dumps the fit:

    n_fit, n_cells, resid median/p90, huber_frac_lt_0.1,
    layer_hist of the matched product layer, layer_match_frac (== the product's top layer),
    b / tx-span / ty-span, PC before/after, and a counterfactual PC when the fit uses
    ONLY single-layer flat cells.

Verdict:
  layer_mismatch   bad frames match a non-top layer AND the flat-only fit fixes the PC
  theta_unstable   |b| > 3 m or few rows / high residual, and flat-only does not fix it
  product_eval     layer match is high and theta looks normal, but the PC still regresses
  mixed            otherwise

Usage::

    python -m stream_ortho.scripts.diag.e6_failure_frames \
        --obs result-new/P0/obs/ieu/pointmap --out results/week10-pivotb/ieu/e6 --seq ieu
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

import numpy as np

from stream_ortho.eval import pivotb_preflight as P
from stream_ortho.eval import sync_eval as S
from stream_ortho.map.calib import CalibConfig, apply_t3, fit_t3
from stream_ortho.scripts.diag.e6_directional import frame_pc

FLAT_SIGMA, FLAT_MIN_N = 0.3, 2


def near_with_layer(rows, prod):
    """(z_near, layer_idx, top_idx, has) for each row against the product."""
    ix = np.floor(rows["x"] / prod["cell"]).astype(np.int64)
    iy = np.floor(rows["y"] / prod["cell"]).astype(np.int64)
    pn, pz, ok = S.product_layers_at(prod, ix, iy)
    occ = pn > 0
    has = occ.any(-1)
    z_near = np.full(ix.size, np.nan)
    layer = np.full(ix.size, -1, np.int64)
    top = np.full(ix.size, -1, np.int64)
    if has.any():
        zz = pz[has]
        occ_h = occ[has]
        zo = rows["z"][has]
        d = np.where(occ_h, np.abs(zo[:, None] - zz), np.inf)
        idx = d.argmin(-1)
        z_near[has] = zz[np.arange(idx.size), idx]
        layer[has] = idx
        top[has] = np.where(occ_h, zz, -np.inf).argmax(-1)
    return z_near, layer, top, has


def fit_stats(z, z_near, layer, top, x, y, w, cfg, origin):
    ok = np.isfinite(z_near)
    th = fit_t3(z, z_near, x, y, w, cfg, origin)
    if th is None:
        return {"n_fit": int(ok.sum()), "theta": None}
    d = z[ok] - z_near[ok]
    A = np.stack([np.ones(ok.sum()), x[ok] - origin[0], y[ok] - origin[1]], 1)
    r = d - A @ th
    s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-2)
    h = np.minimum(1.0, cfg.huber_k * s / np.maximum(np.abs(r), 1e-9))
    hist = np.bincount(layer[ok], minlength=3)[:3]
    match = float(np.mean(layer[ok] == top[ok]))
    return {"n_fit": int(ok.sum()), "n_cells": int(np.unique(np.floor(x[ok] / 0.45)).size),
            "resid_median": float(np.median(np.abs(r))), "resid_p90": float(np.percentile(np.abs(r), 90)),
            "huber_frac_lt_0.1": float(np.mean(h < 0.1)),
            "layer_hist": [int(v) for v in hist], "layer_match_frac": match,
            "b": float(th[0]), "tx": float(th[1]), "ty": float(th[2]),
            "tx_span_m": float(abs(th[1]) * (np.percentile(x[ok], 95) - np.percentile(x[ok], 5))),
            "ty_span_m": float(abs(th[2]) * (np.percentile(y[ok], 95) - np.percentile(y[ok], 5))),
            "theta": [float(v) for v in th]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--ridge", type=float, default=0.0)
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = CalibConfig(ridge=a.ridge)
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    even, odd = kfs[0::2], kfs[1::2]

    targets = {"odd_build_even_held": {"build": odd, "bad": [22, 10, 8, 4], "good": [13, 3]},
               "even_build_odd_held": {"build": even, "bad": [7, 1], "good": [13, 9]}}

    out = {"seq": a.seq, "cell": d["cell"], "directions": {}}
    for name, spec in targets.items():
        bk = spec["build"]
        prod_b, _, _ = P.product(d, ix, iy, bk, slices, bbox, return_assign=True)
        eval_cells = S.eval_cells_of(prod_b)
        occ = prod_b["n"] > 0
        single = occ.sum(-1) == 1
        rows_all = []
        for tag in ("bad", "good"):
            for k in spec[tag]:
                if k not in slices:
                    continue
                r = S._rows(d, slices[k])
                z_near, layer, top, has = near_with_layer(r, prod_b)
                w = r["conf"] * r["cos_inc"]
                st = fit_stats(r["z"], z_near, layer, top, r["x"], r["y"], w, cfg, origin)
                # PC before/after (the frame's own consistency vs its build neighbours)
                pc_b, npr = frame_pc(d, slices, k, bk, eval_cells, lambda kk, rr: rr["z"])
                zc = (apply_t3(r["z"], r["x"], r["y"], st["theta"], origin)
                      if st.get("theta") is not None else r["z"])
                pc_a, _ = frame_pc(d, slices, k, bk, eval_cells,
                                   lambda kk, rr, _z=zc: _z if kk == k else rr["z"])
                # counterfactual: fit only on single-layer flat cells of product and frame
                key = r["key"]
                cell_flat = ((np.floor(r["x"] / prod_b["cell"]).astype(np.int64) - prod_b["ox"]) +
                             (np.floor(r["y"] / prod_b["cell"]).astype(np.int64) - prod_b["oy"]) * prod_b["W"])
                cell_single = single.reshape(-1)[np.clip(cell_flat, 0, prod_b["H"] * prod_b["W"] - 1)]
                kk, cc = np.unique(key, return_counts=True)
                frame_single = np.isin(key, kk[cc == 1])
                flat = ((r["z_std"] <= FLAT_SIGMA) & (r["n"] >= FLAT_MIN_N) &
                        cell_single & frame_single & has)
                th_flat = (fit_t3(r["z"][flat], z_near[flat], r["x"][flat], r["y"][flat],
                                  w[flat], cfg, origin) if flat.sum() >= 50 else None)
                if th_flat is not None:
                    zf = apply_t3(r["z"], r["x"], r["y"], th_flat, origin)
                    pc_f, _ = frame_pc(d, slices, k, bk, eval_cells,
                                       lambda kk2, rr, _z=zf: _z if kk2 == k else rr["z"])
                else:
                    pc_f = float("nan")
                rec = {"kf": int(k), "dir": name, "kind": tag, **{kk2: v for kk2, v in st.items() if kk2 != "theta"},
                       "pc_before": pc_b, "pc_after": pc_a,
                       "pc_after_flat": pc_f, "n_flat_fit": int(flat.sum()),
                       "pc_rel_pct": float(100 * (1 - pc_a / pc_b)) if pc_b else float("nan"),
                       "pc_rel_flat_pct": float(100 * (1 - pc_f / pc_b)) if (pc_b and np.isfinite(pc_f)) else float("nan")}
                rows_all.append(rec)
                print(f"{name[:4]} kf{k:2d} [{tag}] n_fit {rec['n_fit']:6d} match {rec['layer_match_frac']:.2f} "
                      f"|b| {abs(rec['b']):.2f} resid {rec['resid_median']:.2f}  "
                      f"PC {pc_b:.3f}->{pc_a:.3f} ({rec['pc_rel_pct']:+.1f}%)  flat {pc_f:.3f} "
                      f"({rec['pc_rel_flat_pct']:+.1f}%)")
        # ---- theta series over ALL held frames: are the bad frames fit outliers? ----
        series = []
        for k in (bk[0:0] or []):
            pass
        held_k = [kk for kk in (even if name.startswith("odd") else odd)]
        for k in held_k:
            r = S._rows(d, slices[k])
            z_near, layer, top, has = near_with_layer(r, prod_b)
            th = fit_t3(r["z"], z_near, r["x"], r["y"], r["conf"] * r["cos_inc"], cfg, origin)
            if th is None:
                continue
            pc_b, _ = frame_pc(d, slices, k, bk, eval_cells, lambda kk, rr: rr["z"])
            zc = apply_t3(r["z"], r["x"], r["y"], th, origin)
            pc_a, _ = frame_pc(d, slices, k, bk, eval_cells,
                               lambda kk, rr, _z=zc: _z if kk == k else rr["z"])
            series.append({"kf": int(k), "b": float(th[0]), "tx": float(th[1]), "ty": float(th[2]),
                           "pc_rel_pct": float(100 * (1 - pc_a / pc_b)) if pc_b else float("nan")})
        order = sorted(s_["kf"] for s_ in series)
        bs = {s_["kf"]: s_["b"] for s_ in series}
        nbr_of = {k: [order[i - 1], order[i + 1]] for i, k in enumerate(order)
                  if 0 < i < len(order) - 1}
        for s_ in series:
            nb = [bs[j] for j in nbr_of.get(s_["kf"], [])]
            s_["db_vs_neighbours"] = float(abs(s_["b"] - np.median(nb))) if nb else float("nan")
        rel = np.array([s_["pc_rel_pct"] for s_ in series])
        dbv = np.array([s_["db_vs_neighbours"] for s_ in series])
        ok = np.isfinite(rel) & np.isfinite(dbv)
        corr = float(np.corrcoef(dbv[ok], rel[ok])[0, 1]) if ok.sum() > 3 else float("nan")
        bad_kf = set(spec["bad"])
        bad_db = [s_["db_vs_neighbours"] for s_ in series if s_["kf"] in bad_kf]
        out["directions"][name] = {"frames": rows_all,
                                   "theta_series": series,
                                   "theta_corr_db_pcrel": corr,
                                   "bad_db_median": float(np.median(bad_db)) if bad_db else float("nan"),
                                   "all_db_median": float(np.nanmedian(dbv)),
                                   "n_neg_frames": int((rel < 0).sum()),
                                   "n_frames": int(len(series))}

    # ---- verdict ----
    bad = [f for d_ in out["directions"].values() for f in d_["frames"] if f["kind"] == "bad"]
    good = [f for d_ in out["directions"].values() for f in d_["frames"] if f["kind"] == "good"]
    mismatch = [f for f in bad if f["layer_match_frac"] < 0.6]
    flat_fixes = [f for f in mismatch if np.isfinite(f["pc_rel_flat_pct"]) and
                  f["pc_rel_flat_pct"] > f["pc_rel_pct"] + 5]
    unstable = [f for f in bad if abs(f["b"]) > 3.0 or f["n_fit"] < 500]
    if mismatch and flat_fixes:
        verdict = "layer_mismatch"
    elif unstable:
        verdict = "theta_unstable"
    elif all(f["layer_match_frac"] > 0.8 for f in bad):
        verdict = "product_eval"
    else:
        verdict = "mixed"
    out["verdict"] = verdict
    out["summary"] = {
        "n_bad": len(bad), "n_good": len(good),
        "bad_layer_match_median": float(np.median([f["layer_match_frac"] for f in bad])),
        "good_layer_match_median": float(np.median([f["layer_match_frac"] for f in good])),
        "bad_absb_median": float(np.median([abs(f["b"]) for f in bad])),
        "good_absb_median": float(np.median([abs(f["b"]) for f in good])),
        "bad_pc_rel_median": float(np.median([f["pc_rel_pct"] for f in bad])),
        "bad_pc_rel_flat_median": float(np.median([f["pc_rel_flat_pct"] for f in bad])),
        "n_mismatch": len(mismatch), "n_flat_fixes": len(flat_fixes), "n_unstable": len(unstable),
    }
    out["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    out["command"] = "python -m stream_ortho.scripts.diag.e6_failure_frames " + " ".join(argv or [])
    out["wall_s"] = time.time() - t0
    p = pathlib.Path(a.out) / "failure_frames.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("verdict:", verdict, json.dumps(out["summary"], ensure_ascii=False))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
