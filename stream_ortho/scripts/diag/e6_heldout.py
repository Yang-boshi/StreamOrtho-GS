"""E6 -- is the frozen calibration a self-fulfilling loop?  Held-out evaluation.

The PIVOT-B product is built from **all** frames and theta is fitted to that same product,
so nothing was ever held out.  Here the calibration is fitted on build frames only and
applied to frames that never entered the product:

* **E6-A even/odd**: build from even keyframes, hold out odd (and the reverse);
* **E6-B 5 folds**: leave-one-contiguous-fold-out;
* **E6-C reference**: no external ortho exists, so the even-built and odd-built products are
  cross-evaluated [非外部参考].

Per held frame: ``theta = fit_t3(frame, prod_build.near)``; the corrected heights are then
used for the held-side metrics and for a **frozen rebuild** of the product (build rows raw +
held rows corrected, assigned into the build product's layer slots; cells that only the held
frames observe are no-data and reported as ``held_only_cell_frac``).

Usage::

    python -m stream_ortho.scripts.diag.e6_heldout --obs result-new/P0/obs/ieu/pointmap \
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
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics

Z_GAP = 2.0


# --------------------------------------------------------------------------------------
def build(prod_b, masks_b, z_raw_top, raw_iqr):
    base_cov = float((prod_b["n"] > 0).any(-1).mean())
    return masks_b, z_raw_top, raw_iqr, base_cov


def assign_rows(rows, z_vals, prod_b):
    """Assign held observations to the build product's frozen layer slots."""
    ix = np.floor(rows["x"] / prod_b["cell"]).astype(np.int64)
    iy = np.floor(rows["y"] / prod_b["cell"]).astype(np.int64)
    n_l, z_l, ok = S.product_layers_at(prod_b, ix, iy)
    occ = n_l > 0
    d = np.where(occ, np.abs(z_vals[:, None] - z_l), np.inf)
    best = d.argmin(-1)
    keep = ok & occ.any(-1) & (d[np.arange(len(best)), best] <= Z_GAP)
    flat = ((iy - prod_b["oy"]) * prod_b["W"] + (ix - prod_b["ox"])) * prod_b["L"] + best
    out = np.full(len(z_vals), -1, np.int64)
    out[keep] = flat[keep]
    return out


def rebuild(prod_b, assign_all, z_all, w_all, n_all):
    """Frozen-slot rebuild over the union (build rows + held rows)."""
    L = prod_b["L"]; H, W = prod_b["H"], prod_b["W"]
    size = H * W * L
    ok = assign_all >= 0
    a = assign_all[ok]
    num = np.bincount(a, weights=(w_all * z_all)[ok], minlength=size)
    den = np.bincount(a, weights=w_all[ok], minlength=size)
    swz2 = np.bincount(a, weights=(w_all * z_all * z_all)[ok], minlength=size)
    nn = np.bincount(a, weights=n_all[ok], minlength=size)
    z_ref = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return {"n": nn.reshape(H, W, L), "z_ref": z_ref.reshape(H, W, L),
            "owner": prod_b["owner"].copy(),
            "sum_w": den.reshape(H, W, L), "sum_wz": num.reshape(H, W, L),
            "sum_wz2": swz2.reshape(H, W, L),
            "cell": prod_b["cell"], "W": W, "H": H, "L": L,
            "ox": prod_b["ox"], "oy": prod_b["oy"],
            "x0": prod_b["ox"] * prod_b["cell"], "y0": prod_b["oy"] * prod_b["cell"]}


def evaluate(d, slices, build_k, held_k, prod_b, masks_b, z_raw_top, raw_iqr, base_cov,
             cfg, origin):
    """Held-out evaluation for one split (raw vs calibrated held frames)."""
    # assignment of the build rows is already in prod_b
    idx_b = np.sort(np.concatenate([slices[k] for k in build_k]))
    rows_b = S._rows(d, idx_b)
    a_b = prod_b["assign"]
    w_b = rows_b["conf"] * rows_b["n"]
    n_b = rows_b["n"].astype(np.float64)
    z_b = rows_b["z"]

    held_idx = np.sort(np.concatenate([slices[k] for k in held_k]))
    rows_h = S._rows(d, held_idx)
    w_h = rows_h["conf"] * rows_h["n"]
    n_h = rows_h["n"].astype(np.float64)
    z_h_raw = rows_h["z"]

    # fit + apply per held frame
    z_h_cor = z_h_raw.copy()
    kf_h = d["kf"][held_idx]
    for k in held_k:
        m = kf_h == k
        sub = {kk: vv[m] for kk, vv in rows_h.items()}
        z_near, _ = S.product_near_z(sub, prod_b)
        th = fit_t3(sub["z"], z_near, sub["x"], sub["y"],
                    sub["conf"] * sub["cos_inc"], cfg, origin)
        if th is None:
            continue
        z_h_cor[m] = apply_t3(sub["z"], sub["x"], sub["y"], th, origin)

    eval_cells = S.eval_cells_of(prod_b)
    nbr = {}
    sample = held_k[:: max(1, len(held_k) // 6)][:6]
    for k in sample:
        rk = S._rows(d, slices[k])
        sk = S.single_layer_cells(rk["key"])
        ov = sorted(((S._intersect_sorted(sk, S.single_layer_cells(
            S._rows(d, slices[j])["key"])).size, j) for j in build_k), reverse=True)
        nbr[k] = [j for _, j in ov[:6]]
    zmap = {}

    def zfun(k, rows):
        if k in held_k:
            m = kf_h == k
            return z_h_cor[m]
        return rows["z"]

    out = {}
    for tag, z_h in (("raw", z_h_raw), ("frozen", z_h_cor)):
        pcs = S.pc_pass(d, slices, sample, nbr, eval_cells,
                        S.SyncConfig(), lambda k, rows, _z=z_h: (
                            _z[kf_h == k] if k in held_k else rows["z"]))
        a_h = assign_rows(rows_h, z_h, prod_b)
        prod_eval = rebuild(prod_b, np.r_[a_b, a_h], np.r_[z_b, z_h],
                            np.r_[w_b, w_h], np.r_[n_b, n_h])
        m = surf_metrics(prod_eval, masks_b, z_raw_top, raw_iqr)
        m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
        occ = prod_eval["n"] > 0
        zlow = np.where(occ, prod_eval["z_ref"], np.inf).min(-1)
        ztop = prod_eval["z_ref"].max(-1)
        multi = occ.sum(-1) >= 2
        hgt = np.where(multi & occ.any(-1), ztop - zlow, np.nan)
        m["height_median"] = float(np.nanmedian(hgt)) if np.isfinite(hgt).any() else float("nan")
        m["pc"] = pcs["pc"]
        m["n_pairs_pc"] = pcs["n_pairs"]
        m["held_only_cell_frac"] = float(np.mean(
            (prod_eval["n"] > 0).any(-1) & ~(prod_b["n"] > 0).any(-1)))
        out[tag] = m
    return out


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = CalibConfig(ridge=a.ridge)
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    def prep(build_k):
        prod_b, _, _ = P.product(d, ix, iy, build_k, slices, bbox, return_assign=True)
        masks_b = shared2_pairs(prod_b)
        z_top, _ = top_surface(prod_b)
        iqr = float(np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 75) -
                    np.percentile(prod_b["z_ref"][prod_b["n"] > 0], 25))
        cov = float((prod_b["n"] > 0).any(-1).mean())
        return prod_b, masks_b, z_top, iqr, cov

    res = {"seq": a.seq, "cell": d["cell"], "n_frames": len(kfs), "ridge": a.ridge}

    # ---- E6-A even/odd, both directions ----
    even, odd = kfs[0::2], kfs[1::2]
    ea = {}
    for name, (bk, hk) in {"even_build_odd_held": (even, odd),
                           "odd_build_even_held": (odd, even)}.items():
        prod_b, masks_b, z_top, iqr, cov = prep(bk)
        ea[name] = evaluate(d, slices, bk, hk, prod_b, masks_b, z_top, iqr, cov, cfg, origin)
        ea[name]["n_build"], ea[name]["n_held"] = len(bk), len(hk)
        print(f"{name}: raw PC {ea[name]['raw']['pc']:.3f} -> frozen {ea[name]['frozen']['pc']:.3f} "
              f"({100*(1-ea[name]['frozen']['pc']/ea[name]['raw']['pc']):+.1f}%)  "
              f"LCR {ea[name]['raw']['lcr_fixed']:.4f} -> {ea[name]['frozen']['lcr_fixed']:.4f}")
    res["even_odd"] = ea

    # ---- E6-B five contiguous folds ----
    folds = []
    for fi, fold in enumerate(np.array_split(np.arange(len(kfs)), 5)):
        hk = [kfs[i] for i in fold]
        bk = [k for k in kfs if k not in hk]
        prod_b, masks_b, z_top, iqr, cov = prep(bk)
        m = evaluate(d, slices, bk, hk, prod_b, masks_b, z_top, iqr, cov, cfg, origin)
        m["fold"] = fi
        folds.append(m)
        print(f"fold {fi}: raw PC {m['raw']['pc']:.3f} -> frozen {m['frozen']['pc']:.3f}  "
              f"LCR {m['raw']['lcr_fixed']:.4f} -> {m['frozen']['lcr_fixed']:.4f}")
    res["folds"] = folds

    # ---- gates ----
    def gate_of(one):
        r, f = one["raw"], one["frozen"]
        return {
            "pc_down_20": bool(f["pc"] <= 0.8 * r["pc"]),
            "lcr_fixed": bool(f["lcr_fixed"] >= r["lcr_fixed"] - 0.002),
            "ra": bool(f["ra"] <= r["ra"] * 0.95),
            "sei": bool(f["sei"] <= r["sei"] * 1.02),
            "coverage": bool(abs(f["coverage_frac_delta"]) <= 0.01),
            "z_iqr": bool(0.95 <= f["z_iqr_ratio"] <= 1.05),
        }

    res["gate"] = {name: {**gate_of(v), "passed": bool(all(gate_of(v).values()))}
                   for name, v in ea.items()}
    res["gate"]["folds_passed"] = int(sum(
        all(gate_of(f).values()) for f in folds))
    res["gate"]["both_directions"] = bool(
        res["gate"]["even_build_odd_held"]["passed"] and res["gate"]["odd_build_even_held"]["passed"])
    res["gate"]["e6_a"] = bool(res["gate"]["both_directions"])
    res["gate"]["e6_b"] = bool(res["gate"]["folds_passed"] >= 4)

    # ---- E6-C reference (even/odd products) ----
    if a.reference:
        res["reference"] = P.reference_even_odd(d, ix, iy, kfs, slices)
    res["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    res["command"] = "python -m stream_ortho.scripts.diag.e6_heldout " + " ".join(argv or [])
    res["wall_s"] = time.time() - t0

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("gate:", json.dumps({k: (v if not isinstance(v, dict) else v["passed"])
                              for k, v in res["gate"].items()}, ensure_ascii=False))
    print("->", out / "heldout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
