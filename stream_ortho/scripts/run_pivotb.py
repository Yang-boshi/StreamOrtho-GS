"""PIVOT-B pipeline: absolute 3-DOF all-frame calibration with **frozen layers/owner**.

    prod0 = build_product(all frames, raw z)          # records the per-row assignment
    for it in 1..N:
        theta_k = fit_t3(frame_k, prod_current)       # every frame, near-layer matching
        z_k'    = z_k - (b + tx u + ty v)
        prod_current = FrozenProduct.rebuild(z', w)   # no re-grouping, no owner re-election

Modes (``--modes``):

    frozen      layers and owner frozen (the method)
    unfrozen    rebuild by re-grouping every round (the ablation that regresses LCR)
    owner_only  re-group the layers but restore the raw owner label afterwards

Correction models (``--model``): ``hybrid`` (frozen main method: neighbour-consensus
objective + 0.1 x product-near prior), ``t3`` (previous PIVOT-B: product-near only),
``r1`` (offset only), ``r4`` (+ scale), ``f4`` (16-coefficient bilinear field).

Usage::

    python -m stream_ortho.scripts.run_pivotb --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-pivotb/ieu --seq ieu --modes frozen,unfrozen,owner_only --iters 3
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
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import flat_mask, metrics as surf_metrics


# --------------------------------------------------------------------------------------
def bilinear16(x, y, half=600.0, n=4):
    u = np.clip((x / half + 1) * 0.5, 0, 1) * (n - 1)
    v = np.clip((y / half + 1) * 0.5, 0, 1) * (n - 1)
    iu, iv = np.floor(u).astype(int).clip(0, n - 2), np.floor(v).astype(int).clip(0, n - 2)
    cols = []
    for a in range(n):
        for b in range(n):
            cols.append((np.abs(iu - a) <= 1) * (np.abs(iv - b) <= 1) *
                        (1 - np.abs(u - a)) * (1 - np.abs(v - b)))
    return np.stack(cols, 1)


def neighbour_lists(d, slices, kfs, n=6):
    """Top-n other frames by shared single-layer cell count (computed once from raw keys)."""
    singles = {k: S.single_layer_cells(d["key"][slices[k]]) for k in kfs}
    out = {}
    for k in kfs:
        ov = sorted(((S._intersect_sorted(singles[k], singles[j]).size, j)
                     for j in kfs if j != k), reverse=True)
        out[k] = [j for _, j in ov[:n]]
    return out


def fit_hybrid(z, z_near, x, y, w, origin, z_ref, w_ref, lam_p, cfg):
    """Neighbour-consensus objective plus a product-near prior (the frozen main method)."""
    from stream_ortho.scripts.diag.e6_neighbor_fit import fit_objective
    A = np.stack([np.ones_like(x), x - origin[0], y - origin[1]], 1)
    return fit_objective(z, z_ref, z_near, A, w_ref, lam_p, cfg)


def fit_model(z, z_near, x, y, w, origin, model, cfg):
    """Per-frame correction coefficients for one of the four models."""
    ok = np.isfinite(z_near)
    if ok.sum() < cfg.min_obs:
        return None
    d = z[ok] - z_near[ok]
    u = x[ok] - origin[0]
    v = y[ok] - origin[1]
    ww = np.clip(np.nan_to_num(w[ok], nan=1.0), 1e-3, None)
    if model == "t3":
        return fit_t3(z, z_near, x, y, w, cfg, origin)
    if model == "r1":
        A = np.ones((ok.sum(), 1))
    elif model == "r4":
        A = np.stack([np.ones_like(u), u, v, z_near[ok]], 1)
    elif model == "f4":
        A = bilinear16(u + origin[0], v + origin[1])
    else:
        raise ValueError(model)
    p = np.linalg.lstsq(A * ww[:, None], d * ww, rcond=None)[0]
    for _ in range(cfg.huber_iters):
        r = d - A @ p
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-2)
        h = np.minimum(1.0, cfg.huber_k * s / np.maximum(np.abs(r), 1e-9))
        wh = ww * h
        p = np.linalg.lstsq(A * wh[:, None], d * wh, rcond=None)[0]
    return p


def apply_model(z, x, y, z_near, p, origin, model):
    if p is None:
        return z
    if model == "t3":
        return apply_t3(z, x, y, p, origin)
    if model == "r1":
        return z - p[0]
    if model == "r4":
        return z - (p[0] + p[1] * (x - origin[0]) + p[2] * (y - origin[1]) + p[3] * z_near)
    if model == "f4":
        return z - bilinear16(x, y) @ p
    raise ValueError(model)


# --------------------------------------------------------------------------------------
def run(obs, out, seq, modes, iters, model, ridge, tag="", lam_p=0.1, k=6):
    t0 = time.time()
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    cfg = CalibConfig(ridge=ridge)
    d, ix, iy = P.load(obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    t_build = time.time()
    prod_raw, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox, return_assign=True)
    build_s = time.time() - t_build
    fp = FrozenProduct.from_product(prod_raw)
    masks = shared2_pairs(prod_raw)
    z_raw_top, live = top_surface(prod_raw)
    raw_iqr = float(np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 75) -
                    np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 25))
    base_cov = float((prod_raw["n"] > 0).any(-1).mean())
    w_all = rows_all["conf"] * rows_all["n"]
    n_all = rows_all["n"].astype(np.float64)
    PC = np.zeros(3)          # unused placeholder

    def ev(prod_, extra=None):
        m = surf_metrics(prod_, masks, z_raw_top, raw_iqr)
        m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
        # building-height retention: selected surface above the cell's own lowest layer,
        # only where the cell actually has >= 2 layers (single-layer cells carry no height)
        occ = prod_["n"] > 0
        zlow = np.where(occ, prod_["z_ref"], np.inf).min(-1)
        ztop_sel = prod_["z_ref"].max(-1)
        multi = occ.sum(-1) >= 2
        hgt = np.where(multi & (prod_["n"] > 0).any(-1), ztop_sel - zlow, np.nan)
        m["height_median"] = float(np.nanmedian(hgt)) if np.isfinite(hgt).any() else float("nan")
        m["n_multi_layer"] = int(np.isfinite(hgt).sum())
        # --- T11 guard triple: did the selection change the layer or the owner? ---
        idx1 = np.where(occ, prod_["z_ref"], -np.inf).argmax(-1).ravel()
        own1 = np.take_along_axis(
            prod_["owner"].reshape(-1, prod_["L"]), idx1[:, None], 1)[:, 0]
        z1 = np.take_along_axis(
            prod_["z_ref"].reshape(-1, prod_["L"]), idx1[:, None], 1)[:, 0]
        live1 = (prod_["n"] > 0).any(-1).ravel()
        both = live1 & raw_live_flat
        m["layer_index_preserved"] = (float(np.mean(idx1[both] == raw_top_idx[both]))
                                      if both.any() else float("nan"))
        m["owner_preserved"] = (float(np.mean(own1[both] == raw_top_owner[both]))
                                if both.any() else float("nan"))
        dz = (z1 - raw_top_z)[both]
        m["z_shift_median"] = float(np.median(dz)) if dz.size else float("nan")
        m["z_shift_p95_abs"] = float(np.percentile(np.abs(dz), 95)) if dz.size else float("nan")
        m["n_cells_compared"] = int(both.sum())
        if extra:
            m.update(extra)
        return m

    _occ0 = prod_raw["n"] > 0
    _idx0 = np.where(_occ0, prod_raw["z_ref"], -np.inf).argmax(-1)
    raw_top_idx = _idx0.ravel()
    raw_top_owner = np.take_along_axis(
        prod_raw["owner"].reshape(-1, prod_raw["L"]), _idx0.reshape(-1, 1), 1)[:, 0]
    raw_top_z = np.take_along_axis(
        prod_raw["z_ref"].reshape(-1, prod_raw["L"]), _idx0.reshape(-1, 1), 1)[:, 0]
    raw_live_flat = _occ0.any(-1).ravel()

    raw_m = ev(prod_raw)

    # hybrid needs each frame's neighbours and the current (sorted) z of those neighbours
    if model == "hybrid":
        nbrs = neighbour_lists(d, slices, kfs, k)
        order_of = {kk: np.argsort(S._rows(d, slices[kk])["key"], kind="stable") for kk in kfs}
        sub_of = {kk: {k2: v[order_of[kk]] for k2, v in S._rows(d, slices[kk]).items()}
                  for kk in kfs}

    results = {}
    timings = []
    for mode in modes:
        prod = prod_raw
        z_cur = rows_all["z"].copy()
        it_stats = []
        rebuild_ms = []
        thetas = {}
        for it in range(1, iters + 1):
            eval_cells = S.eval_cells_of(prod)
            z_new = rows_all["z"].copy()
            fit_ms = []
            for k in kfs:
                r = S._rows(d, slices[k])
                z_near, _ = S.product_near_z(r, prod)
                tf0 = time.perf_counter()
                if model == "hybrid":
                    sub = sub_of[k]
                    zk = z_cur[slices[k]][order_of[k]]
                    z_cur_k = zk
                    base = S._intersect_sorted(S.single_layer_cells(sub["key"]), eval_cells)
                    if base.size < 200:
                        p = fit_t3(zk, z_near, r["x"], r["y"], r["conf"] * r["cos_inc"],
                                   cfg, origin)
                    else:
                        V = np.full((base.size, len(nbrs[k])), np.nan)
                        W = np.zeros((base.size, len(nbrs[k])))
                        for i, j in enumerate(nbrs[k]):
                            sj = sub_of[j]
                            zj = z_cur[slices[j]][order_of[j]]
                            pos = np.clip(np.searchsorted(sj["key"], base), 0,
                                          max(sj["key"].size - 1, 0))
                            hit = sj["key"][pos] == base
                            V[hit, i] = zj[pos[hit]]
                            W[hit, i] = 1.0
                        from stream_ortho.scripts.diag.e6_neighbor_fit import weighted_median_rows
                        z_ref = weighted_median_rows(V, W)
                        wc = W.sum(1)
                        xv = S._lookup(sub["key"], sub["x"], base)
                        yv = S._lookup(sub["key"], sub["y"], base)
                        zk_c = S._lookup(sub["key"], zk, base)
                        pv, _ = S.product_near_z({**sub, "z": zk}, prod)
                        z_prior = S._lookup(sub["key"], pv, base)
                        p = fit_hybrid(zk_c, z_prior, xv, yv, wc, origin, z_ref, wc, lam_p, cfg)
                    fit_ms.append((time.perf_counter() - tf0) * 1000)
                    thetas[int(k)] = p
                    if p is not None:
                        A_full = np.stack([np.ones(sub["key"].size, np.float64),
                                           sub["x"] - origin[0], sub["y"] - origin[1]], 1)
                        z_new[slices[k]] = zk - A_full @ p
                    continue
                p = fit_model(r["z"], z_near, r["x"], r["y"], r["conf"] * r["cos_inc"],
                              origin, model, cfg)
                fit_ms.append((time.perf_counter() - tf0) * 1000)
                thetas[int(k)] = p
                if p is not None:
                    z_new[slices[k]] = apply_model(r["z"], r["x"], r["y"],
                                                   np.nan_to_num(z_near, nan=0.0), p, origin, model)
            if mode == "frozen":
                _t = time.perf_counter()
                prod = fp.rebuild(z_new, w_all)
                rebuild_ms.append((time.perf_counter() - _t) * 1000)
            else:
                zc = z_new
                prod = S.build_product(ix[allidx], iy[allidx], rows_all["key"], zc,
                                       w_all, n_all, d["cell"], bbox,
                                       kf=d["kf"][allidx], rgb=rows_all["rgb"])
                if mode == "owner_only":
                    same = prod["n"].shape == prod_raw["n"].shape
                    if same:
                        m = prod["n"] > 0
                        prod["owner"] = np.where(m, prod_raw["owner"], prod["owner"])
            z_cur = z_new
            m = ev(prod, {"iter": it, "fit_ms_median": float(np.median(fit_ms)),
                          "fit_ms_p95": float(np.percentile(fit_ms, 95)),
                          "rebuild_ms": float(np.median(rebuild_ms)) if rebuild_ms else float("nan")})
            it_stats.append(m)
            timings.append({"mode": mode, "iter": it, "fit_ms_p50": float(np.median(fit_ms)),
                            "fit_ms_p95": float(np.percentile(fit_ms, 95))})
        prod_final = prod
        results[mode] = {"iters": it_stats, "final": it_stats[-1], "z_final": z_new}

        # product snapshot (primary mode only, to keep the disk small)
        if mode == "frozen" and iters == 3:
            np.savez_compressed(pathlib.Path(out) / "product_frozen.npz",
                                n=prod_final["n"].astype(np.uint16),
                                z_ref=prod_final["z_ref"].astype(np.float32),
                                owner=prod_final["owner"].astype(np.int32),
                                cell=prod_final["cell"], ox=prod_final["ox"],
                                oy=prod_final["oy"])

    # ---- frame-to-frame PC on the calibrated frames (independent of the product) ----
    sample = kfs[:: max(1, len(kfs) // 10)][:10]
    nbr = {}
    for k in sample:
        rk = S._rows(d, slices[k])
        sk = S.single_layer_cells(rk["key"])
        ov = sorted(((S._intersect_sorted(sk, S.single_layer_cells(
            S._rows(d, slices[j])["key"])).size, j) for j in kfs), reverse=True)
        nbr[k] = [j for _, j in ov[:6]]
    pc_ec = S.eval_cells_of(prod_final)
    for mode in results:
        zf = results[mode].get("z_final")
        if zf is None:
            results[mode]["pc"] = float("nan")
            continue
        pcs = S.pc_pass(d, slices, sample, nbr, pc_ec, S.SyncConfig(),
                        lambda k, rows, _z=zf: _z[slices[k]])
        results[mode]["pc"] = pcs["pc"]
        results[mode]["n_pairs_pc"] = pcs["n_pairs"]

    # ---- machine-readable gate (review 21 §2.2) ----
    def gate_of(mode):
        f = results[mode]["final"]
        vals = {"lcr_fixed": f["lcr_fixed"], "n_jump": f["n_jump"], "pc": results[mode]["pc"],
                "ra": f["ra"], "sei": f["sei"], "coverage": f["coverage_frac_delta"],
                "layer_index_preserved": f["layer_index_preserved"],
                "owner_preserved": f["owner_preserved"],
                "top_preserved_appendix": 1.0 - f["top_changed_frac"],
                "z_shift_p95_abs": f["z_shift_p95_abs"],
                "building_height": (f["height_median"] / raw_m["height_median"]
                                    if raw_m.get("height_median") else float("nan")),
                "z_iqr": f["z_iqr_ratio"]}
        checks = {
            "lcr_fixed": bool(vals["lcr_fixed"] >= raw_m["lcr_fixed"] - 0.002),
            "n_jump": bool(vals["n_jump"] <= raw_m["n_jump"] * 1.05),
            "pc": bool(vals["pc"] <= 0.30),
            "ra": bool(vals["ra"] <= raw_m["ra"] * 0.95),
            "sei": bool(vals["sei"] <= raw_m["sei"] * 1.02),
            "coverage": bool(abs(vals["coverage"]) <= 0.01),
            "layer_index_preserved": bool(vals["layer_index_preserved"] >= 0.99),
            "owner_preserved": bool(vals["owner_preserved"] >= 0.99),
            "building_height": bool(vals["building_height"] >= 0.90),
            "z_iqr": bool(0.95 <= vals["z_iqr"] <= 1.05),
        }
        return {**checks, "values": {k: float(v) for k, v in vals.items()},
                "thresholds": {"lcr_fixed": raw_m["lcr_fixed"] - 0.002,
                               "n_jump": raw_m["n_jump"] * 1.05, "pc": 0.30,
                               "ra": raw_m["ra"] * 0.95, "sei": raw_m["sei"] * 1.02,
                               "coverage": 0.01, "layer_index_preserved": 0.99,
                               "owner_preserved": 0.99,
                               "building_height": 0.90, "z_iqr": [0.95, 1.05]},
                "passed": bool(all(checks.values()))}
    res_gate = {m: gate_of(m) for m in results}

    res = {"seq": seq, "obs": obs, "cell": d["cell"], "n_frames": len(kfs), "iters": iters,
           "gate": res_gate,
           "model": model, "ridge": ridge, "raw": raw_m, "modes": results,
           "timings": timings,
           "memory": {"assign_mb_int64": fp.memory_mb(np.int64),
                      "assign_mb_int32": fp.memory_mb(np.int32),
                      "n_rows": int(fp.assign.size)},
           "raw_build_s": build_s,
           "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "wall_s": time.time() - t0}
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(out) / f"pivotb_{model}{tag}.json"
    slim = dict(res)
    slim["modes"] = {m: {k: v for k, v in d_.items() if k != "z_final"}
                     for m, d_ in res["modes"].items()}
    p.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    return res, p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--modes", default="frozen,unfrozen,owner_only")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--model", default="hybrid", choices=("hybrid", "t3", "r1", "r4", "f4"))
    ap.add_argument("--lam-p", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    res, p = run(a.obs, a.out, a.seq, [m.strip() for m in a.modes.split(",") if m.strip()],
                 a.iters, a.model, a.ridge, a.tag, lam_p=a.lam_p, k=a.k)
    raw = res["raw"]
    print(f"{res['seq']}  model={res['model']}  iters={res['iters']}  cell={res['cell']}")
    print(f"{'mode':11s} {'LCR_fixed':>9s} {'n_jump':>8s} {'switch':>7s} {'RA':>6s} {'SEI':>6s} {'cover':>6s} {'IQRr':>5s}")
    print(f"{'raw':11s} {raw['lcr_fixed']:9.5f} {raw['n_jump']:8d} {raw['owner_switch']:7d} "
          f"{raw['ra']:6.3f} {raw['sei']:6.2f} {raw['coverage_cells']:6.3f} {raw['z_iqr_ratio']:5.3f}")
    for m, v in res["modes"].items():
        f = v["final"]
        print(f"{m:11s} {f['lcr_fixed']:9.5f} {f['n_jump']:8d} {f['owner_switch']:7d} "
              f"{f['ra']:6.3f} {f['sei']:6.2f} {f['coverage_cells']:6.3f} {f['z_iqr_ratio']:5.3f}")
    print("memory:", res["memory"])
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
