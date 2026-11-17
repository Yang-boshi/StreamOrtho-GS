"""T9-A -- per-frame selection between the neighbour-consensus and the product objective.

Both candidates are scored by **half-cell cross-validation**: the cells are split by
``hash(key) % 2`` (so the split cannot leak spatially), each candidate is fitted on one half
and scored on the other (the score is the residual to the neighbour consensus, available on
every cell), and the two directions are averaged.

Selection per frame:

* ``nbr`` wins if it beats ``prod`` by >= 5% on the held halves;
* ``prod`` is kept otherwise;
* if neither beats the identity (theta = 0) on the held halves, the frame keeps its raw
  heights and is counted in ``fallback_frac``.

Reported: ``pc_rel_median``, ``pct_improved``, ``min_pc_rel``, ``worst_kf``,
``fallback_frac`` for both even/odd directions and the 5 folds.

Usage::

    python -m stream_ortho.scripts.diag.e6_selector --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-pivotb/ieu/e6 --seq ieu
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
from stream_ortho.scripts.diag.e6_neighbor_fit import (neighbour_consensus, product_prior,
                                                       fit_objective)
from stream_ortho.scripts.diag.e6_heldout import assign_rows, rebuild
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics

GAIN = 1.05          # nbr must be 5% better to be selected


def _parity(keys):
    """Stable hash-free split: the packed cell key is already a bijection, so use its parity."""
    return (np.asarray(keys, np.int64) & 1) == 0


def _resid_median(z, target, A, theta):
    r = np.abs(z - target - A @ theta)
    return float(np.median(r)) if r.size else float("nan")


def select_frame(d, slices, k, build_k, prod_b, eval_cells, cfg, origin):
    """Fit both candidates, score them on held halves, return the chosen theta + diagnostics."""
    r = S._rows(d, slices[k])
    common, z_ref, w_c, nbrs = neighbour_consensus(d, slices, k, build_k, eval_cells)
    if common.size < 200:
        return None, {"reason": "too_few_cells", "n": int(common.size)}
    o = np.argsort(r["key"], kind="stable")
    ks = r["key"][o]
    zk = S._lookup(ks, r["z"][o], common)
    xc = S._lookup(ks, r["x"][o], common)
    yc = S._lookup(ks, r["y"][o], common)
    A = np.stack([np.ones_like(xc), xc - origin[0], yc - origin[1]], 1)
    z_near = S._lookup(ks, product_prior(r, None, prod_b, "near")[o], common)
    ok = np.isfinite(z_ref) & np.isfinite(zk)
    ha = _parity(common) & ok
    hb = (~_parity(common)) & ok
    if ha.sum() < 100 or hb.sum() < 100:
        return None, {"reason": "half_too_small", "na": int(ha.sum()), "nb": int(hb.sum())}

    th_nbr_a = fit_objective(zk[ha], z_ref[ha], None, A[ha], w_c[ha], 0.0, cfg)
    th_nbr_b = fit_objective(zk[hb], z_ref[hb], None, A[hb], w_c[hb], 0.0, cfg)
    s_nbr = 0.5 * (_resid_median(zk[hb], z_ref[hb], A[hb], th_nbr_a) +
                   _resid_median(zk[ha], z_ref[ha], A[ha], th_nbr_b))

    th_prod_a = fit_t3(zk[ha], z_near[ha], xc[ha], yc[ha], w_c[ha], cfg, origin)
    th_prod_b = fit_t3(zk[hb], z_near[hb], xc[hb], yc[hb], w_c[hb], cfg, origin)
    if th_prod_a is None or th_prod_b is None:
        th_prod_a = th_prod_b = np.zeros(3)
    s_prod = 0.5 * (_resid_median(zk[hb], z_ref[hb], A[hb], th_prod_a) +
                    _resid_median(zk[ha], z_ref[ha], A[ha], th_prod_b))

    s_raw = 0.5 * (_resid_median(zk[hb], z_ref[hb], A[hb], np.zeros(3)) +
                   _resid_median(zk[ha], z_ref[ha], A[ha], np.zeros(3)))

    # refit the winner on ALL usable cells before applying
    if s_nbr * GAIN < s_prod and s_nbr < s_raw:
        pick = "nbr"
        theta = fit_objective(zk[ok], z_ref[ok], None, A[ok], w_c[ok], 0.0, cfg)
    elif s_prod < s_raw:
        pick = "prod"
        theta = fit_t3(zk[ok], z_near[ok], xc[ok], yc[ok], w_c[ok], cfg, origin)
    else:
        pick = "identity"
        theta = np.zeros(3)
    diag = {"pick": pick, "s_nbr": s_nbr, "s_prod": s_prod, "s_raw": s_raw,
            "n_cells": int(ok.sum())}
    return theta, diag


def evaluate(d, slices, build_k, held_k, prod_b, cfg, origin, eval_cells,
             z_raw_top, masks_b, raw_iqr, base_cov, rows_b, w_b, n_b, z_b, a_b):
    rows_h_idx = np.sort(np.concatenate([slices[k] for k in held_k]))
    rows_h = S._rows(d, rows_h_idx)
    kf_h = d["kf"][rows_h_idx]
    z_cor = rows_h["z"].copy()
    rels, picks = [], []
    for k in held_k:
        theta, diag = select_frame(d, slices, k, build_k, prod_b, eval_cells, cfg, origin)
        m = kf_h == k
        picks.append(diag["pick"] if isinstance(diag, dict) else "skip")
        if theta is not None:
            sub = {kk: vv[m] for kk, vv in rows_h.items()}
            z_cor[m] = apply_t3(sub["z"], sub["x"], sub["y"], theta, origin)
        pc_b, _ = frame_pc(d, slices, k, build_k, eval_cells, lambda kk, rr: rr["z"])
        pc_a, _ = frame_pc(d, slices, k, build_k, eval_cells,
                           lambda kk, rr, _z=z_cor[m]: _z if kk == k else rr["z"])
        if pc_b:
            rels.append(100.0 * (1 - pc_a / pc_b))
    # uncorrected counterpart: the LCR gate is relative to the same rebuild with raw held z
    a_h_raw = assign_rows(rows_h, rows_h["z"], prod_b)
    prod_eval_raw = rebuild(prod_b, np.r_[a_b, a_h_raw], np.r_[z_b, rows_h["z"]],
                            np.r_[w_b, rows_h["conf"] * rows_h["n"]],
                            np.r_[n_b, rows_h["n"].astype(np.float64)])
    m_raw = surf_metrics(prod_eval_raw, masks_b, z_raw_top, raw_iqr)

    a_h = assign_rows(rows_h, z_cor, prod_b)
    prod_eval = rebuild(prod_b, np.r_[a_b, a_h], np.r_[z_b, z_cor],
                        np.r_[w_b, rows_h["conf"] * rows_h["n"]],
                        np.r_[n_b, rows_h["n"].astype(np.float64)])
    m = surf_metrics(prod_eval, masks_b, z_raw_top, raw_iqr)
    m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
    m["lcr_raw_eval"] = m_raw["lcr_fixed"]
    m["pc_rel_median"] = float(np.nanmedian(rels)) if rels else float("nan")
    m["pct_improved"] = float(np.mean(np.array(rels) > 0) * 100) if rels else float("nan")
    m["min_pc_rel"] = float(np.min(rels)) if rels else float("nan")
    m["worst_kf"] = int(held_k[int(np.argmin(rels))]) if rels else -1
    m["fallback_frac"] = float(np.mean(np.array(picks) == "identity")) if picks else float("nan")
    m["pick_hist"] = {p: int(sum(1 for q in picks if q == p)) for p in set(picks)}
    m["pc_rel_all"] = rels
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--skip-folds", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = CalibConfig()
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    even, odd = kfs[0::2], kfs[1::2]

    res = {"seq": a.seq, "cell": d["cell"], "gain": GAIN, "directions": {}}
    for name, (bk, hk) in {"even_build_odd_held": (even, odd),
                           "odd_build_even_held": (odd, even)}.items():
        prod_b, _, _ = P.product(d, ix, iy, bk, slices, bbox, return_assign=True)
        eval_cells = S.eval_cells_of(prod_b)
        masks_b = shared2_pairs(prod_b)
        z_top, _ = top_surface(prod_b)
        iqr = float(np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 75) -
                    np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 25))
        cov = float((prod_b["n"] > 0).any(-1).mean())
        idx_b = np.sort(np.concatenate([slices[k] for k in bk]))
        rb = S._rows(d, idx_b)
        m = evaluate(d, slices, bk, hk, prod_b, cfg, origin, eval_cells, z_top, masks_b, iqr,
                     cov, rb, rb["conf"] * rb["n"], rb["n"].astype(np.float64), rb["z"],
                     prod_b["assign"])
        res["directions"][name] = m
        print(f"{name}: median {m['pc_rel_median']:+.1f}%  improved {m['pct_improved']:.1f}%  "
              f"worst {m['min_pc_rel']:+.1f}% (kf{m['worst_kf']})  fallback {m['fallback_frac']:.2f}  "
              f"picks {m['pick_hist']}  LCR {m['lcr_fixed']:.4f}")

    if not a.skip_folds:
        folds = []
        for fi, fold in enumerate(np.array_split(np.arange(len(kfs)), 5)):
            hk = [kfs[i] for i in fold]
            bk = [k for k in kfs if k not in hk]
            prod_b, _, _ = P.product(d, ix, iy, bk, slices, bbox, return_assign=True)
            eval_cells = S.eval_cells_of(prod_b)
            masks_b = shared2_pairs(prod_b)
            z_top, _ = top_surface(prod_b)
            iqr = float(np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 75) -
                        np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 25))
            cov = float((prod_b["n"] > 0).any(-1).mean())
            idx_b = np.sort(np.concatenate([slices[k] for k in bk]))
            rb = S._rows(d, idx_b)
            m = evaluate(d, slices, bk, hk, prod_b, cfg, origin, eval_cells, z_top, masks_b,
                         iqr, cov, rb, rb["conf"] * rb["n"], rb["n"].astype(np.float64),
                         rb["z"], prod_b["assign"])
            m["fold"] = fi
            folds.append(m)
            print(f"fold {fi}: median {m['pc_rel_median']:+.1f}%  improved {m['pct_improved']:.1f}%  "
                  f"worst {m['min_pc_rel']:+.1f}%  fallback {m['fallback_frac']:.2f}")
        res["folds"] = folds

    g = {}
    d_ = res["directions"]
    g["e6a_both_ge20"] = bool(all(d_[k]["pc_rel_median"] >= 20 for k in d_))
    g["pct_improved_ge90"] = bool(all(d_[k]["pct_improved"] >= 90 for k in d_))
    g["worst_ge_minus10"] = bool(all(d_[k]["min_pc_rel"] >= -10 for k in d_))
    g["lcr_ok"] = bool(all(d_[k]["lcr_fixed"] >= d_[k]["lcr_raw_eval"] - 0.002 for k in d_))
    g["z_iqr_ok"] = bool(all(0.95 <= d_[k]["z_iqr_ratio"] <= 1.05 for k in d_))
    if res.get("folds"):
        g["e6b_5of5"] = bool(all(f["pc_rel_median"] > 0 for f in res["folds"]))
    g["coverage_ok"] = bool(all(abs(d_[k]["coverage_frac_delta"]) <= 0.01 for k in d_))
    g["passed"] = bool(g["e6a_both_ge20"] and g["pct_improved_ge90"] and
                       g["worst_ge_minus10"] and g["z_iqr_ok"] and g["lcr_ok"] and
                       g["coverage_ok"] and g.get("e6b_5of5", True))
    res["gate"] = g
    res["verdict"] = "selector_passes" if g["passed"] else "selector_misses"
    res["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    res["command"] = "python -m stream_ortho.scripts.diag.e6_selector " + " ".join(argv or [])
    res["wall_s"] = time.time() - t0
    p = pathlib.Path(a.out) / "e6_selector.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("gate:", json.dumps({k: v for k, v in g.items()}, ensure_ascii=False))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
