"""T8 -- change the fit target from "product near layer" to "neighbour consensus + prior".

The E6 failure frames are a target mismatch: ``fit_t3`` pulls a held frame onto the
product's near layer, while PC scores it against its neighbouring frames.  Where the product
layer is a surface only some frames see, the correction pushes the frame *away* from its
neighbours (the same keyframe 13 improves +59.8% under one build set and degrades -8.5% under
the other).  Here the objective becomes

    min_theta  sum_c w_c (z_k(c) - z_ref(c) - A_c theta)^2
             + lam_p sum_c w_c (z_k(c) - z_prior(c) - A_c theta)^2

with ``z_ref`` the **weighted median** of the top-K build neighbours on the shared
single-layer flat cells, ``w_c`` the summed neighbour weight, and ``z_prior`` either the
product's near layer (``prod``) or its top layer (``top``).  ``lam_p = 0`` is the pure
neighbour fit (the E1 failure mode is the warning), ``lam_p = 1`` weights both equally.

Configs: ``raw``, ``prod`` (current PIVOT-B objective), ``nbr`` (lam_p=0),
``hybrid0.1``, ``hybrid1``, ``nbr_top`` (neighbour + product **top** prior).

Usage::

    python -m stream_ortho.scripts.diag.e6_neighbor_fit --obs result-new/P0/obs/ieu/pointmap \
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
from stream_ortho.scripts.diag.e6_heldout import assign_rows, rebuild
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics

K_NEIGHBOURS = 6


# --------------------------------------------------------------------------------------
def weighted_median_rows(V, W):
    """Per-row weighted median; rows with no weight become NaN.

    ``V``/``W`` may contain NaN / zero weights (a neighbour that does not observe that
    cell); those entries are simply ignored.
    """
    V = np.where(np.isfinite(V) & (W > 0), V, np.nan)
    order = np.argsort(np.where(np.isfinite(V), V, np.inf), axis=1)
    Vs = np.take_along_axis(V, order, 1)
    Ws = np.take_along_axis(np.where(np.isfinite(Vs), np.take_along_axis(W, order, 1), 0.0), order * 0 + np.arange(V.shape[1])[None, :] * 0, 1)
    Ws = np.take_along_axis(W, order, 1)
    Ws = np.where(np.isfinite(Vs), Ws, 0.0)
    cw = np.cumsum(Ws, 1)
    tot = cw[:, -1]
    thr = 0.5 * tot
    idx = (cw >= thr[:, None]).argmax(1)
    out = Vs[np.arange(V.shape[0]), idx]
    out[tot <= 0] = np.nan
    return out


def neighbour_consensus(d, slices, k, build_k, eval_cells, n_neighbors=K_NEIGHBOURS):
    """(keys, z_ref, w_c, neighbours) over the frame's single-layer flat cells.

    The base set is the frame ∩ the product's flat cells; a neighbour that does not observe
    a base cell simply contributes nothing there (NaN / zero weight), so the consensus never
    collapses just because one neighbour has a hole.
    """
    rk = S._rows(d, slices[k])
    sk = S.single_layer_cells(rk["key"])
    ov = sorted(((S._intersect_sorted(sk, S.single_layer_cells(
        S._rows(d, slices[j])["key"])).size, j) for j in build_k), reverse=True)
    nbrs = [j for _, j in ov[:n_neighbors]]
    base = sk if eval_cells is None else S._intersect_sorted(sk, eval_cells)
    if base.size == 0:
        return base, np.zeros(0), np.zeros(0), nbrs
    V = np.full((base.size, len(nbrs)), np.nan)
    W = np.zeros((base.size, len(nbrs)))
    for i, j in enumerate(nbrs):
        rj = S._rows(d, slices[j])
        o = np.argsort(rj["key"], kind="stable")
        bk, bz = rj["key"][o], rj["z"][o]
        bw = (rj["conf"] * rj["cos_inc"])[o]
        pos = np.searchsorted(bk, base)
        pos_c = np.clip(pos, 0, bk.size - 1)
        hit = bk[pos_c] == base
        V[hit, i] = bz[pos_c][hit]
        W[hit, i] = bw[pos_c][hit]
    z_ref = weighted_median_rows(V, W)
    return base, z_ref, W.sum(1), nbrs


def fit_objective(z_k, z_ref, z_prior, A, w, lam_p, cfg: CalibConfig):
    """Huber IRLS on the stacked (neighbour consensus, product prior) rows."""
    # rows without a consensus value (a cell no neighbour observed) cannot constrain the
    # fit; they are excluded from the *fit* only -- the correction is still applied to every
    # observation when scoring
    ok0 = np.isfinite(z_ref) & np.isfinite(z_k) & np.isfinite(A).all(1)
    d1 = (z_k - z_ref)[ok0]
    rows = [A[ok0]]
    vals = [d1]
    wts = [w[ok0]]
    if lam_p > 0 and z_prior is not None:
        ok = np.isfinite(z_prior) & ok0
        if ok.any():
            rows.append(A[ok])
            vals.append((z_k - z_prior)[ok])
            # the card's objective is  + lam_p * sum w (...)^2 , so the *rows* are scaled by
            # sqrt(lam_p); scaling them by lam_p would give an effective coefficient lam_p^2
            wts.append(np.sqrt(lam_p) * w[ok])
    if not vals:
        return np.zeros(3)
    Aa = np.vstack(rows)
    dd = np.concatenate(vals)
    ww = np.clip(np.nan_to_num(np.concatenate(wts), nan=0.0), 1e-3, None)
    keep = np.isfinite(dd) & np.isfinite(Aa).all(1)
    Aa, dd, ww = Aa[keep], dd[keep], ww[keep]
    if Aa.shape[0] < 50:
        return np.zeros(3)
    p = np.linalg.lstsq(Aa * ww[:, None], dd * ww, rcond=None)[0]
    for _ in range(cfg.huber_iters):
        r = dd - Aa @ p
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-2)
        h = np.minimum(1.0, cfg.huber_k * s / np.maximum(np.abs(r), 1e-9))
        wh = ww * h
        p = np.linalg.lstsq(Aa * wh[:, None], dd * wh, rcond=None)[0]
    return p


def product_prior(rows, keys, prod, layer="near"):
    ix = np.floor(rows["x"] / prod["cell"]).astype(np.int64)
    iy = np.floor(rows["y"] / prod["cell"]).astype(np.int64)
    pn, pz, ok = S.product_layers_at(prod, ix, iy)
    occ = pn > 0
    if layer == "near":
        zo = rows["z"]
        d = np.where(occ, np.abs(zo[:, None] - pz), np.inf)
        idx = d.argmin(-1)
    else:
        idx = np.where(occ, pz, -np.inf).argmax(-1)
    n = np.arange(idx.size)
    out = np.full(idx.size, np.nan)
    good = ok & occ[n, idx]
    out[good] = pz[n, idx][good]
    return out


# --------------------------------------------------------------------------------------
def evaluate_config(d, slices, build_k, held_k, prod_b, cfg, origin, configs, eval_cells,
                    z_raw_top, masks_b, raw_iqr, base_cov, rows_b, w_b, n_b, z_b, a_b):
    """Per config: per-frame PC + a frozen rebuild of the product for the secondary metrics.

    ``configs`` maps name -> (kind, lam_p) with kind in
    ``{"raw", "prod", "nbr", "near", "top"}``: ``prod`` = the current PIVOT-B objective
    (fit to the product's near layer over all rows), ``nbr`` = neighbour consensus only,
    ``near``/``top`` = consensus + product prior on the shared cells.
    """
    held_idx = np.sort(np.concatenate([slices[k] for k in held_k]))
    rows_h = S._rows(d, held_idx)
    kf_h = d["kf"][held_idx]
    cons = {k: neighbour_consensus(d, slices, k, build_k, eval_cells) for k in held_k}
    prior_all = {}
    for k in held_k:
        r = S._rows(d, slices[k])
        prior_all[k] = {"near": product_prior(r, None, prod_b, "near"),
                        "top": product_prior(r, None, prod_b, "top")}

    out = {}
    for name, (kind, lam_p) in configs.items():
        z_cor = rows_h["z"].copy()
        rels = []
        for k in held_k:
            m = kf_h == k
            sub = {kk: vv[m] for kk, vv in rows_h.items()}
            o = np.argsort(sub["key"], kind="stable")
            ks = sub["key"][o]
            w_row = sub["conf"] * sub["cos_inc"]
            if kind == "raw":
                theta = None
            elif kind == "prod":
                theta = fit_t3(sub["z"], prior_all[k]["near"], sub["x"], sub["y"], w_row,
                               cfg, origin)
            else:
                common, z_ref, w_c, _ = cons[k]
                if common.size < 50:
                    continue
                zk_c = S._lookup(ks, sub["z"][o], common)
                xc = S._lookup(ks, sub["x"][o], common)
                yc = S._lookup(ks, sub["y"][o], common)
                A = np.stack([np.ones_like(xc), xc - origin[0], yc - origin[1]], 1)
                z_prior = None
                if kind in ("near", "top"):
                    z_prior = S._lookup(ks, prior_all[k][kind][o], common)
                theta = fit_objective(zk_c, z_ref, z_prior, A, w_c, lam_p, cfg)
            if theta is not None:
                z_cor[m] = apply_t3(sub["z"], sub["x"], sub["y"], theta, origin)
            pc_b, _ = frame_pc(d, slices, k, build_k, eval_cells, lambda kk, rr: rr["z"])
            pc_a, _ = frame_pc(d, slices, k, build_k, eval_cells,
                               lambda kk, rr, _z=z_cor[m]: _z if kk == k else rr["z"])
            if pc_b:
                rels.append(100.0 * (1 - pc_a / pc_b))
        a_h = assign_rows(rows_h, z_cor, prod_b)
        prod_eval = rebuild(prod_b, np.r_[a_b, a_h], np.r_[z_b, z_cor],
                            np.r_[w_b, rows_h["conf"] * rows_h["n"]],
                            np.r_[n_b, rows_h["n"].astype(np.float64)])
        m = surf_metrics(prod_eval, masks_b, z_raw_top, raw_iqr)
        m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
        m["pc_rel_median"] = float(np.nanmedian(rels)) if rels else float("nan")
        m["pc_rel_mean"] = float(np.nanmean(rels)) if rels else float("nan")
        m["n_frames"] = len(rels)
        m["n_negative"] = int(np.sum(np.array(rels) < 0))
        m["pct_improved"] = float(np.mean(np.array(rels) > 0) * 100) if rels else float("nan")
        m["min_pc_rel"] = float(np.min(rels)) if rels else float("nan")
        m["worst_kf"] = int(held_k[int(np.argmin(rels))]) if rels else -1
        out[name] = m
        print(f"    {name:10s} PC rel {m['pc_rel_median']:+6.1f}% (mean {m['pc_rel_mean']:+.1f}) "
              f"improved {m['pct_improved']:5.1f}%  worst {m['min_pc_rel']:+6.1f}% "
              f"LCR {m['lcr_fixed']:.4f} RA {m['ra']:.3f} IQR {m['z_iqr_ratio']:.3f}")
    return out


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--skip-folds", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = CalibConfig(ridge=a.ridge)
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    configs = {"raw": ("raw", 0.0), "prod": ("prod", 0.0), "nbr": ("nbr", 0.0),
               "hybrid0.1": ("near", 0.1), "hybrid1": ("near", 1.0),
               "nbr_top": ("top", 0.1)}

    even, odd = kfs[0::2], kfs[1::2]
    res = {"seq": a.seq, "cell": d["cell"], "K_neighbours": K_NEIGHBOURS, "directions": {}}

    for name, (bk, hk) in {"even_build_odd_held": (even, odd),
                           "odd_build_even_held": (odd, even)}.items():
        prod_b, _, _ = P.product(d, ix, iy, bk, slices, bbox, return_assign=True)
        eval_cells = S.eval_cells_of(prod_b)
        masks_b = shared2_pairs(prod_b)
        z_top, _ = top_surface(prod_b)
        raw_iqr = float(np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 75) -
                        np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 25))
        base_cov = float((prod_b["n"] > 0).any(-1).mean())
        idx_b = np.sort(np.concatenate([slices[k] for k in bk]))
        rows_b = S._rows(d, idx_b)
        w_b = rows_b["conf"] * rows_b["n"]
        n_b = rows_b["n"].astype(np.float64)
        print(f"== {name}")
        res["directions"][name] = evaluate_config(
            d, slices, bk, hk, prod_b, cfg, origin, configs, eval_cells, z_top, masks_b,
            raw_iqr, base_cov, rows_b, w_b, n_b, rows_b["z"], prod_b["assign"])

    # ---- E6-B: 5 folds, three configs only ----
    if not a.skip_folds:
        fold_res = []
        for fi, fold in enumerate(np.array_split(np.arange(len(kfs)), 5)):
            hk = [kfs[i] for i in fold]
            bk = [k for k in kfs if k not in hk]
            prod_b, _, _ = P.product(d, ix, iy, bk, slices, bbox, return_assign=True)
            eval_cells = S.eval_cells_of(prod_b)
            masks_b = shared2_pairs(prod_b)
            z_top, _ = top_surface(prod_b)
            raw_iqr = float(np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 75) -
                            np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 25))
            base_cov = float((prod_b["n"] > 0).any(-1).mean())
            idx_b = np.sort(np.concatenate([slices[k] for k in bk]))
            rows_b = S._rows(d, idx_b)
            print(f"== fold {fi}")
            m = evaluate_config(d, slices, bk, hk, prod_b, cfg, origin,
                                {k: configs[k] for k in ("raw", "prod", "nbr", "hybrid0.1")},
                                eval_cells, z_top, masks_b, raw_iqr, base_cov, rows_b,
                                rows_b["conf"] * rows_b["n"], rows_b["n"].astype(np.float64),
                                rows_b["z"], prod_b["assign"])
            m["fold"] = fi
            fold_res.append(m)
        res["folds"] = fold_res

    # ---- gate ----
    def gate(dirs, folds):
        g = {}
        for cfgname in ("prod", "nbr", "hybrid0.1", "hybrid1", "nbr_top"):
            rel = [dirs[d_][cfgname]["pc_rel_median"] for d_ in dirs]
            g[cfgname] = {
                "e6a_both_20": bool(all(r >= 20 for r in rel)),
                "pct_improved_80": bool(all(dirs[d_][cfgname]["pct_improved"] >= 80 for d_ in dirs)),
                "worst_ge_-10": bool(all(dirs[d_][cfgname]["min_pc_rel"] >= -10 for d_ in dirs)),
                "lcr_ok": bool(all(dirs[d_][cfgname]["lcr_fixed"] >=
                                   dirs[d_]["raw"]["lcr_fixed"] - 0.002 for d_ in dirs)),
            }
            if folds and all(cfgname in f for f in folds):
                g[cfgname]["e6b_folds_4of5"] = bool(sum(
                    f[cfgname]["pc_rel_median"] >= 20 for f in folds) >= 4)
            g[cfgname]["passed"] = bool(all(v for k, v in g[cfgname].items()))
        return g

    res["gate"] = gate(res["directions"], res.get("folds"))
    best = [k for k, v in res["gate"].items() if v["passed"]]
    res["verdict"] = ("hybrid_passes" if any(k.startswith("hybrid") for k in best) else
                      ("nbr_passes" if "nbr" in best else "none_pass"))
    res["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    res["command"] = "python -m stream_ortho.scripts.diag.e6_neighbor_fit " + " ".join(argv or [])
    res["wall_s"] = time.time() - t0
    p = pathlib.Path(a.out) / "e6_neighbor_fit.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("gate:", json.dumps({k: v["passed"] for k, v in res["gate"].items()}, ensure_ascii=False))
    print("verdict:", res["verdict"])
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
