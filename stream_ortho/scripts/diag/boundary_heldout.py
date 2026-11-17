"""T37-D -- the owner-boundary bias on **held-out** frames (does the effect survive without loop?).

The full-frame boundary audit (T13) uses a product built from every frame, so a reviewer can
argue the calibration and the evaluation share the same frames.  Here the product comes from
**build** frames only, the held frames are corrected against it, and the boundary bias is
measured **exclusively from the held frames' observations**:

    for every owner boundary (c, c') of the build product:
        z_c  = median over the held observations at cell c   (raw / calibrated)
        z_c' = same for c'
        bias = z_c - z_c'          (skip a pair if either side has no held observation)

Reported per direction and per fold: median_bias_reduction, robust weighted reduction
(|bias_raw| >= 0.2 m), boundary_bias_p95, double_wall_rate, frame_bias_std, n_pairs,
n_skipped.

Usage::

    python -m stream_ortho.scripts.diag.boundary_heldout \
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


def cell_median(keys, vals, flat_cells):
    """Per-cell median (upper median) of ``vals``; NaN where the cell has no row."""
    order = np.argsort(keys, kind="stable")
    ks, vs = keys[order], np.asarray(vals)[order]
    lo = np.searchsorted(ks, flat_cells, "left")
    hi = np.searchsorted(ks, flat_cells, "right")
    size = hi - lo
    mid = np.clip(lo + np.maximum(size // 2, 0), 0, max(vs.size - 1, 0))
    out = vs[mid].astype(np.float64)
    out[size == 0] = np.nan
    return out


def boundaries(prod, live_flat_shape):
    """(key_a, key_b) **packed global cell keys** of 4-neighbour pairs with different owners."""
    from stream_ortho.map.selection import pack_cell
    H, W = prod["H"], prod["W"]
    ox, oy = prod["ox"], prod["oy"]
    occ = prod["n"] > 0
    own = np.where(occ, prod["owner"], -2).max(-1)
    live = occ.any(-1)
    out = []
    d = live[:, :-1] & live[:, 1:] & (own[:, :-1] != own[:, 1:]) & (own[:, :-1] >= 0) & (own[:, 1:] >= 0)
    i, j = np.nonzero(d)
    out.append((pack_cell(ox + j, oy + i), pack_cell(ox + j + 1, oy + i)))
    d = live[:-1, :] & live[1:, :] & (own[:-1, :] != own[1:, :]) & (own[:-1, :] >= 0) & (own[1:, :] >= 0)
    i, j = np.nonzero(d)
    out.append((pack_cell(ox + j, oy + i), pack_cell(ox + j, oy + i + 1)))
    a = np.concatenate([o[0] for o in out]); b = np.concatenate([o[1] for o in out])
    return a, b


def held_build_pairs(d, slices, build_k, held_k, prod_b, z_cor, pos_of, min_shared=100):
    """Frame-pair bias between every held frame and every build frame on their shared cells.

    ``bias = median_c (z_held(c) - z_build(c))`` -- identical in spirit to the full-frame
    audit, but the held frame never entered the product.  Build frames keep their raw
    heights (documented as the "held-only correction" limitation).
    """
    eval_cells = S.eval_cells_of(prod_b)
    singles = {k: S.single_layer_cells(S._rows(d, slices[k])["key"]) for k in build_k + held_k}
    keys, zs = {}, {}
    for k in build_k + held_k:
        r = S._rows(d, slices[k])
        o = np.argsort(r["key"], kind="stable")
        keys[k] = r["key"][o]
        zs[k] = r["z"][o]
    out = []
    for k in held_k:
        zk_raw = zs[k]
        zk_cal = z_cor[pos_of[k]]
        for j in build_k:
            common = S._intersect_sorted(S._intersect_sorted(singles[k], singles[j]), eval_cells)
            if common.size < min_shared:
                continue
            a_r = S._lookup(keys[k], zk_raw, common)
            a_c = S._lookup(keys[k], zk_cal, common)
            b = S._lookup(keys[j], zs[j], common)
            out.append((float(np.median(a_r - b)), float(np.median(a_c - b))))
    if not out:
        return None
    br = np.array([o[0] for o in out]); bc = np.array([o[1] for o in out])
    abs_r, abs_c = np.abs(br), np.abs(bc)
    red = 100.0 * (1 - abs_c / np.maximum(abs_r, 1e-9))
    big = abs_r >= 0.2
    return {"n_pairs": int(br.size),
            "median_abs_bias_raw": float(np.median(abs_r)),
            "median_abs_bias_cal": float(np.median(abs_c)),
            "median_bias_reduction_pct": float(np.median(red)),
            "robust_weighted_reduction_pct": float(np.average(red[big], weights=abs_r[big]))
            if big.any() else float("nan"),
            "frac_pairs_improved": float(np.mean(red > 0)),
            "boundary_bias_p95_raw": float(np.percentile(abs_r, 95)),
            "boundary_bias_p95_cal": float(np.percentile(abs_c, 95)),
            "double_wall_rate_raw": float(np.mean(abs_r > 1.0)),
            "double_wall_rate_cal": float(np.mean(abs_c > 1.0)),
            "frame_bias_std_raw": float(np.std(br)),
            "frame_bias_std_cal": float(np.std(bc))}


def boundary_cells(prod, band=2):
    """Packed keys of live cells within ``band`` cells of an owner boundary."""
    from scipy.ndimage import distance_transform_edt
    from stream_ortho.map.selection import pack_cell
    H, W = prod["H"], prod["W"]
    occ = prod["n"] > 0
    own = np.where(occ, prod["owner"], -2).max(-1)
    live = occ.any(-1)
    b = np.zeros_like(live)
    d = live[:, :-1] & live[:, 1:] & (own[:, :-1] != own[:, 1:])
    b[:, :-1] |= d; b[:, 1:] |= d
    d = live[:-1, :] & live[1:, :] & (own[:-1, :] != own[1:, :])
    b[:-1, :] |= d; b[1:, :] |= d
    near = distance_transform_edt(~b) <= band
    i, j = np.nonzero(near & live)
    return pack_cell(prod["ox"] + j, prod["oy"] + i)


def held_pair_bias(d, slices, held_k, prod_b, z_cor, pos_of, cfg, origin, band=2,
                   n_neighbors=6):
    """Frame-pair agreement of HELD frames, restricted to the boundary band."""
    near = boundary_cells(prod_b, band)
    if near.size == 0:
        return None
    singles = {k: S.single_layer_cells(S._rows(d, slices[k])["key"]) for k in held_k}
    out = {"raw": [], "cal": []}
    for k in held_k:
        rk = S._rows(d, slices[k])
        ov = sorted(((S._intersect_sorted(singles[k], singles[j]).size, j)
                     for j in held_k if j != k), reverse=True)
        for _, j in ov[:n_neighbors]:
            common = S._intersect_sorted(S._intersect_sorted(singles[k], singles[j]), near)
            if common.size < 100:
                continue
            rj = S._rows(d, slices[j])
            ok_, zk = np.argsort(rk["key"], kind="stable"), rk["z"]
            oj_, zj = np.argsort(rj["key"], kind="stable"), rj["z"]
            a = S._lookup(rk["key"][ok_], zk[ok_], common)
            b = S._lookup(rj["key"][oj_], zj[oj_], common)
            out["raw"].append(float(np.median(np.abs(a - b))))
            zk_c = z_cor[pos_of[k]][ok_]
            zj_c = z_cor[pos_of[j]][oj_]
            ac = S._lookup(rk["key"][ok_], zk_c, common)
            bc = S._lookup(rj["key"][oj_], zj_c, common)
            out["cal"].append(float(np.median(np.abs(ac - bc))))
    if not out["raw"]:
        return None
    raw = float(np.median(out["raw"]))
    cal = float(np.median(out["cal"]))
    return {"n_pairs": len(out["raw"]), "pair_bias_raw": raw, "pair_bias_cal": cal,
            "pair_bias_reduction_pct": float(100 * (1 - cal / raw)) if raw else float("nan")}


def run_split(d, ix, iy, slices, kfs, build_k, held_k, bbox, origin, cfg):
    prod_b, _, _ = P.product(d, ix, iy, build_k, slices, bbox, return_assign=True)
    H, W = prod_b["H"], prod_b["W"]
    cell = prod_b["cell"]

    held_idx = np.sort(np.concatenate([slices[k] for k in held_k]))
    rows = S._rows(d, held_idx)
    kf_h = d["kf"][held_idx]

    # per held frame: fit against the build product, apply
    z_cor = rows["z"].copy()
    for k in held_k:
        m = kf_h == k
        sub = {kk: vv[m] for kk, vv in rows.items()}
        z_near, _ = S.product_near_z(sub, prod_b)
        th = fit_t3(sub["z"], z_near, sub["x"], sub["y"],
                    sub["conf"] * sub["cos_inc"], cfg, origin)
        if th is not None:
            z_cor[m] = apply_t3(sub["z"], sub["x"], sub["y"], th, origin)

    pos_of = {k: np.searchsorted(held_idx, slices[k]) for k in held_k}
    fp = held_build_pairs(d, slices, build_k, held_k, prod_b, z_cor, pos_of)
    a, b = boundaries(prod_b, None)
    if a.size == 0:
        return None
    za_r = cell_median(rows["key"], rows["z"], a)
    zb_r = cell_median(rows["key"], rows["z"], b)
    za_c = cell_median(rows["key"], z_cor, a)
    zb_c = cell_median(rows["key"], z_cor, b)
    ok = np.isfinite(za_r) & np.isfinite(zb_r) & np.isfinite(za_c) & np.isfinite(zb_c)
    n_skip = int((~ok).sum())
    bias_r = (za_r - zb_r)[ok]
    bias_c = (za_c - zb_c)[ok]
    if bias_r.size == 0:
        return None
    abs_r, abs_c = np.abs(bias_r), np.abs(bias_c)
    red = 100.0 * (1 - abs_c / np.maximum(abs_r, 1e-9))
    big = abs_r >= 0.2
    res = {
        "n_pairs": int(bias_r.size), "n_skipped": n_skip,
        "median_abs_bias_raw": float(np.median(abs_r)),
        "median_abs_bias_cal": float(np.median(abs_c)),
        "median_bias_reduction_pct": float(np.median(red)),
        "robust_weighted_reduction_pct": float(np.average(red[big], weights=abs_r[big]))
        if big.any() else float("nan"),
        "frac_pairs_improved": float(np.mean(red > 0)),
        "boundary_bias_p95_raw": float(np.percentile(abs_r, 95)),
        "boundary_bias_p95_cal": float(np.percentile(abs_c, 95)),
        "double_wall_rate_raw": float(np.mean(abs_r > 1.0)),
        "double_wall_rate_cal": float(np.mean(abs_c > 1.0)),
        "frame_bias_std_raw": float(np.std(bias_r)),
        "frame_bias_std_cal": float(np.std(bias_c)),
        "_pairs": (abs_r.tolist(), abs_c.tolist()),
    }
    if fp:
        res["frame_pairs"] = fp
    return res


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

    res = {"seq": a.seq, "cell": d["cell"], "directions": {}, "folds": []}
    for name, (bk, hk) in {"even_build_odd_held": (even, odd),
                           "odd_build_even_held": (odd, even)}.items():
        r = run_split(d, ix, iy, slices, kfs, bk, hk, bbox, origin, cfg)
        if r is None:
            continue
        res["directions"][name] = {k: v for k, v in r.items() if k != "_pairs"}
        fp = r.get("frame_pairs") or {}
        print(f"{name}: [frame pairs] n {fp.get('n_pairs', 0)}  median red "
              f"{fp.get('median_bias_reduction_pct', float('nan')):+.1f}%  robust "
              f"{fp.get('robust_weighted_reduction_pct', float('nan')):+.1f}%  "
              f"p95 {fp.get('boundary_bias_p95_raw', float('nan')):.3f}->"
              f"{fp.get('boundary_bias_p95_cal', float('nan')):.3f}  double-wall "
              f"{fp.get('double_wall_rate_raw', float('nan')):.3f}->"
              f"{fp.get('double_wall_rate_cal', float('nan')):.3f}  | [cell step] "
              f"{r['median_bias_reduction_pct']:+.1f}%")

    for fi, fold in enumerate(np.array_split(np.arange(len(kfs)), 5)):
        hk = [kfs[i] for i in fold]
        bk = [k for k in kfs if k not in hk]
        r = run_split(d, ix, iy, slices, kfs, bk, hk, bbox, origin, cfg)
        if r is None:
            continue
        res["folds"].append({"fold": fi, **{k: v for k, v in r.items() if k != "_pairs"}})
        fp = r.get("frame_pairs") or {}
        print(f"fold {fi}: [frame pairs] n {fp.get('n_pairs', 0)}  median red "
              f"{fp.get('median_bias_reduction_pct', float('nan')):+.1f}%  double-wall "
              f"{fp.get('double_wall_rate_raw', float('nan')):.3f}->"
              f"{fp.get('double_wall_rate_cal', float('nan')):.3f}")

    dirs_ok = [v.get("frame_pairs", {}).get("median_bias_reduction_pct", -1e9) >= 20
               for v in res["directions"].values()]
    folds_ok = [f.get("frame_pairs", {}).get("median_bias_reduction_pct", -1e9) >= 20
                for f in res["folds"]]
    res["gate"] = {
        "both_directions_ge20": bool(all(dirs_ok)) if dirs_ok else False,
        "folds_4of5_ge20": bool(sum(folds_ok) >= 4) if folds_ok else False,
        "p95_down": bool(all(v["frame_pairs"]["boundary_bias_p95_cal"] <=
                             v["frame_pairs"]["boundary_bias_p95_raw"]
                             for v in res["directions"].values() if v.get("frame_pairs")))
        if res["directions"] else False,
        "double_wall_down": bool(all(v["frame_pairs"]["double_wall_rate_cal"] <=
                                     v["frame_pairs"]["double_wall_rate_raw"]
                                     for v in res["directions"].values() if v.get("frame_pairs")))
        if res["directions"] else False,
        "n_pairs": {k: v.get("frame_pairs", {}).get("n_pairs", 0)
                    for k, v in res["directions"].items()},
    }
    res["gate"]["passed"] = bool(res["gate"]["both_directions_ge20"] and
                                 res["gate"]["folds_4of5_ge20"] and
                                 res["gate"]["p95_down"] and res["gate"]["double_wall_down"])
    res["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    res["command"] = "python -m stream_ortho.scripts.diag.boundary_heldout " + " ".join(argv or [])
    res["wall_s"] = time.time() - t0
    p = pathlib.Path(a.out) / "boundary_bias_heldout.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("gate:", json.dumps(res["gate"], ensure_ascii=False))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
