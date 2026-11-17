"""T9-B -- does the neighbour objective survive the **full-frame** product build?

T8 only measured held-out PC.  The deployed method rebuilds the product from all 55 frames, so
the three objectives are run there:

    prod       fit to the product's near layer (current PIVOT-B)
    nbr        fit to the weighted-median consensus of the top-6 other frames (lambda_p = 0)
    hybrid0.1  consensus + 0.1 x product-near prior

Layers/owners stay **frozen** (the PIVOT-B fix), three rounds, each round using the previous
round's corrected heights for the neighbours (Gauss-Seidel: a frame's consensus comes from
*other* frames, so there is no self-reference).

Reported per round and objective: LCR_fixed, n_jump, RA, SEI, coverage, z_IQR,
top_preserved, plus the theta drift (b_std / max adjacent step).

Usage::

    python -m stream_ortho.scripts.diag.fullframe_neighbor \
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
from stream_ortho.map.calib import CalibConfig, FrozenProduct, apply_t3, fit_t3
from stream_ortho.scripts.diag.e6_neighbor_fit import fit_objective, weighted_median_rows
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics

K_NEIGHBOURS = 6


def neighbour_lists(d, slices, kfs, n=K_NEIGHBOURS):
    """Top-n other frames by shared single-layer cell count (computed once from raw keys)."""
    singles = {k: S.single_layer_cells(d["key"][slices[k]]) for k in kfs}
    out = {}
    for k in kfs:
        ov = sorted(((S._intersect_sorted(singles[k], singles[j]).size, j)
                     for j in kfs if j != k), reverse=True)
        out[k] = [j for _, j in ov[:n]]
    return out


def consensus_from(z_by_k, key_by_k, k, nbrs):
    """Weighted-median consensus over ``nbrs`` on frame k's single-layer cells."""
    base = key_by_k[k]
    V = np.full((base.size, len(nbrs)), np.nan)
    W = np.zeros((base.size, len(nbrs)))
    for i, j in enumerate(nbrs):
        kj = key_by_k[j]
        pos = np.clip(np.searchsorted(kj, base), 0, max(kj.size - 1, 0))
        hit = kj[pos] == base
        V[hit, i] = z_by_k[j][pos[hit]]
        W[hit, i] = 1.0
    return weighted_median_rows(V, W), W.sum(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iters", type=int, default=3)
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

    prod_raw, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox, return_assign=True)
    fp = FrozenProduct.from_product(prod_raw)
    masks = shared2_pairs(prod_raw)
    z_top, _ = top_surface(prod_raw)
    raw_iqr = float(np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 75) -
                    np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 25))
    base_cov = float((prod_raw["n"] > 0).any(-1).mean())
    w_all = rows_all["conf"] * rows_all["n"]

    # per-frame arrays in key order (the consensus is built on sorted keys)
    order_of, sub_of = {}, {}
    for k in kfs:
        r = S._rows(d, slices[k])
        o = np.argsort(r["key"], kind="stable")
        order_of[k] = o
        sub_of[k] = {k2: v[o] for k2, v in r.items()}
    nbrs = neighbour_lists(d, slices, kfs)

    def ev(prod_):
        m = surf_metrics(prod_, masks, z_top, raw_iqr)
        m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
        return m

    raw_m = ev(prod_raw)
    print("raw:", {k: round(v, 4) for k, v in raw_m.items() if isinstance(v, (int, float))})

    results = {}
    for objective in ("prod", "nbr", "hybrid0.1"):
        lam_p = 0.0 if objective != "hybrid0.1" else 0.1
        prod = prod_raw
        z_cur = {k: sub_of[k]["z"].copy() for k in kfs}
        rounds = []
        for it in range(1, a.iters + 1):
            thetas = {}
            for k in kfs:
                sub = dict(sub_of[k])
                zk = z_cur[k]
                sub["z"] = zk
                eval_cells = S.eval_cells_of(prod)
                if objective == "prod":
                    z_near, _ = S.product_near_z(sub, prod)
                    th = fit_t3(zk, z_near, sub["x"], sub["y"],
                                sub["conf"] * sub["cos_inc"], cfg, origin)
                else:
                    base = S._intersect_sorted(S.single_layer_cells(sub["key"]), eval_cells)
                    if base.size < 200:
                        thetas[k] = None
                        continue
                    V = np.full((base.size, len(nbrs[k])), np.nan)
                    W = np.zeros((base.size, len(nbrs[k])))
                    for i, j in enumerate(nbrs[k]):
                        kj = sub_of[j]["key"]
                        pos = np.clip(np.searchsorted(kj, base), 0, max(kj.size - 1, 0))
                        hit = kj[pos] == base
                        V[hit, i] = z_cur[j][pos[hit]]
                        W[hit, i] = 1.0
                    z_ref = weighted_median_rows(V, W)
                    wc = W.sum(1)
                    z_prior = None
                    if lam_p > 0:
                        pn, _ = S.product_near_z(sub, prod)
                        z_prior = S._lookup(sub["key"], pn, base)
                    xv = S._lookup(sub["key"], sub["x"], base)
                    yv = S._lookup(sub["key"], sub["y"], base)
                    A = np.stack([np.ones_like(xv), xv - origin[0], yv - origin[1]], 1)
                    zk_c = S._lookup(sub["key"], zk, base)
                    th = fit_objective(zk_c, z_ref, z_prior, A, wc, lam_p, cfg)
                thetas[k] = th
            z_rows = rows_all["z"].copy()
            z_new = {}
            for k in kfs:
                sub = sub_of[k]
                zk = z_cur[k]
                if thetas.get(k) is not None:
                    zk2 = apply_t3(zk, sub["x"], sub["y"], thetas[k], origin)
                else:
                    zk2 = zk
                z_new[k] = zk2
                z_rows[slices[k][order_of[k]]] = zk2
            prod = fp.rebuild(z_rows, w_all)
            z_cur = z_new
            B = np.array([thetas[k] if thetas.get(k) is not None else np.zeros(3) for k in kfs])
            step = np.abs(np.diff(B, axis=0))
            m = ev(prod)
            m.update({"iter": it, "b_std": float(np.nanstd(B[:, 0])),
                      "max_step": float(np.nanmax(step)), "median_step": float(np.nanmedian(step))})
            rounds.append(m)
            print(f"{objective:10s} it{it}: LCR {m['lcr_fixed']:.4f} jump {m['n_jump']:6d} "
                  f"RA {m['ra']:.3f} SEI {m['sei']:.2f} IQR {m['z_iqr_ratio']:.3f} "
                  f"b_std {m['b_std']:.2f} max_step {m['max_step']:.2f}")
        results[objective] = rounds

    # ---- gate on the last round ----
    def gate(m):
        return {"lcr_fixed": bool(m["lcr_fixed"] >= raw_m["lcr_fixed"] - 0.002),
                "n_jump": bool(m["n_jump"] <= raw_m["n_jump"] * 1.05),
                "ra": bool(m["ra"] <= raw_m["ra"] * 0.95),
                "sei": bool(m["sei"] <= raw_m["sei"] * 1.02),
                "coverage": bool(abs(m["coverage_frac_delta"]) <= 0.01),
                "z_iqr": bool(0.95 <= m["z_iqr_ratio"] <= 1.05)}

    gate_out = {}
    for obj, rounds in results.items():
        g = gate(rounds[-1])
        g["passed"] = bool(all(v for k, v in g.items()))
        gate_out[obj] = g

    res = {"seq": a.seq, "cell": d["cell"], "iters": a.iters, "K": K_NEIGHBOURS,
           "raw": raw_m, "objectives": results, "gate": gate_out,
           "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "command": "python -m stream_ortho.scripts.diag.fullframe_neighbor " + " ".join(argv or []),
           "wall_s": time.time() - t0}
    p = pathlib.Path(a.out) / "fullframe_neighbor.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("gate:", json.dumps({k: v["passed"] for k, v in gate_out.items()}, ensure_ascii=False))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
