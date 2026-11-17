"""T4 pilot -- owner/layer-coherent first-surface selection on two sectors.

E2 GO came from the T3.8 classification: 99.75% of the 187k newly created owner boundaries
have a *matchable common layer*, i.e. the two cells do see the same place but pick different
frames/layers.  This script tests, on two dense sectors only, whether a selection that
**penalises owner changes** can repair them without flattening the surface.

Energy (per cell ``c`` and candidate layer ``l``)::

    U(c,l) = -[ w_s log(1+n_l) + w_f flatness_l + w_v visibility_l + w_d diversity_l ]
    phi(c,l;c',l') = lam_o * 1[owner_c(l) != owner_c'(l')]
                   + lam_h * exp(-||dRGB||/sigma_c) * |z_c(l) - z_c'(l')|
                   + lam_c * ||dRGB|| / 255

Solved with ICM (red-black sweeps); ``lam_o = 0`` reproduces a purely geometric MRF and is
reported as the ablation that shows why the owner term is needed.

Protocol: two 341x341-cell sectors at 0.45 m (= 153.5 m, the area-equivalent of the card's
512x512 @ 0.30 m), chosen from the ``new_boundary`` density **once**, +1 cell halo for the
optimisation, metrics on the interior only.  Baselines: raw, it3, top, quality (+ column is
not applicable in a cell-layer framework and is reported as N/A).

PC is frame-to-frame and E2 never touches the frames, so PC is invariant by construction; the
value is inherited from it3 and flagged as such.

Usage::

    python -m stream_ortho.scripts.diag.e2_pilot --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-e2/ieu --seq ieu
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

H_TOL = 1.0
Z_GAP = 2.0
MIN_N = 2
HALO = 1
TILE = 341                     # 341 * 0.45 m = 153.5 m  (the card's 512 x 0.30 m)
TAU_ND = 10.0                  # no-data cost (fixed, not tuned)


# --------------------------------------------------------------------------------------
def calibrate(d, ix, iy, kfs, slices, bbox, origin, cfg, n_iter=3):
    prod, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox)
    prod_raw = prod
    for _ in range(n_iter):
        eval_cells = S.eval_cells_of(prod)
        z_new = rows_all["z"].copy()
        for k in kfs:
            r = S._rows(d, slices[k])
            z_near, _ = S.product_near_z(r, prod)
            w = r["conf"] * r["cos_inc"]
            th = S._fit_t3(r["z"], z_near, r["x"] - origin[0], r["y"] - origin[1], w, cfg)
            if th is None:
                continue
            z_new[slices[k]] = r["z"] - (th[0] + th[1] * (r["x"] - origin[0]) +
                                         th[2] * (r["y"] - origin[1]))
        prod = P.product(d, ix, iy, kfs, slices, bbox, z_override=z_new)[0]
    return prod_raw, prod


def top_of(prod):
    occ = prod["n"] > 0
    top = np.where(occ, prod["z_ref"], -np.inf).argmax(-1)
    z = np.where(occ.any(-1), np.max(np.where(occ, prod["z_ref"], -np.inf), -1), np.nan)
    own = np.take_along_axis(prod["owner"], top[..., None], -1)[..., 0]
    return z, np.where(occ.any(-1), own, -1)


def pick_tiles(own_raw, own_it3, tile=TILE):
    """Two fixed sectors with the highest density of NEW owner boundaries."""
    live = (own_raw >= 0) & (own_it3 >= 0)
    d1 = np.zeros_like(live)
    d1[:, :-1] = live[:, :-1] & live[:, 1:] & (own_it3[:, :-1] != own_it3[:, 1:])
    d2 = np.zeros_like(live)
    d2[:-1, :] = live[:-1, :] & live[1:, :] & (own_it3[:-1, :] != own_it3[1:, :])
    r1 = np.zeros_like(live)
    r1[:, :-1] = live[:, :-1] & live[:, 1:] & (own_raw[:, :-1] != own_raw[:, 1:])
    r2 = np.zeros_like(live)
    r2[:-1, :] = live[:-1, :] & live[1:, :] & (own_raw[:-1, :] != own_raw[1:, :])
    new = (d1 | d2) & ~(r1 | r2)

    # integral image for O(1) tile sums
    ii = np.pad(new.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = new.shape
    if H < tile or W < tile:
        raise SystemExit("grid smaller than the tile")
    sums = (ii[tile:, tile:] - ii[:-tile, tile:] - ii[tile:, :-tile] + ii[:-tile, :-tile])
    order = np.argsort(sums.ravel())[::-1]
    picked = []
    for flat in order:
        j0, i0 = divmod(int(flat), sums.shape[1])
        if j0 + tile >= H or i0 + tile >= W:
            continue
        if any(abs(j0 - a) < tile and abs(i0 - b) < tile for a, b in picked):
            continue
        picked.append((j0, i0))
        if len(picked) == 2:
            break
    return [{"id": k, "j0": int(j), "i0": int(i), "h": tile, "w": tile,
             "new_boundaries": int(sums[j, i])} for k, (j, i) in enumerate(picked)]


# --------------------------------------------------------------------------------------
def tile_view(prod, j0, i0, h, w, halo=HALO):
    J0, I0 = max(j0 - halo, 0), max(i0 - halo, 0)
    J1, I1 = min(j0 + h + halo, prod["H"]), min(i0 + w + halo, prod["W"])
    n = prod["n"][J0:J1, I0:I1]
    return {"n": n, "z": prod["z_ref"][J0:J1, I0:I1], "owner": prod["owner"][J0:J1, I0:I1],
            "rgb": prod["rgb"][J0:J1, I0:I1],
            "sum_w": prod["sum_w"][J0:J1, I0:I1], "sum_wz": prod["sum_wz"][J0:J1, I0:I1],
            "sum_wz2": prod["sum_wz2"][J0:J1, I0:I1], "J0": J0, "I0": I0, "J1": J1, "I1": I1,
            "j0": j0, "i0": i0, "h": h, "w": w}


def unary(tv, w_s=1.0, w_f=1.0, w_v=0.8, w_d=0.3):
    n, z = tv["n"].astype(np.float64), tv["z"]
    occ = n > 0
    top = np.where(occ, z, -np.inf).max(-1, keepdims=True)
    sw = np.maximum(tv["sum_w"], 1e-9)
    z_std = np.sqrt(np.maximum(tv["sum_wz2"] / sw - (tv["sum_wz"] / sw) ** 2, 0.0))
    z_std = np.nan_to_num(z_std)
    flatness = np.exp(-np.maximum(z_std - 0.3, 0) / 0.5)
    visibility = np.where(occ, np.exp(-np.maximum(top - z, 0) / 1.0), 0.0)
    diversity = np.clip(n / 4.0, 0, 1)
    U = -(w_s * np.log1p(n) + w_f * flatness + w_v * visibility + w_d * diversity)
    U = np.where(occ & (n >= MIN_N), U, TAU_ND)
    return U


def icm(tv, U, lam_o=1.0, lam_h=0.5, lam_c=0.3, sigma_c=40.0, iters=15, check_nd=True):
    """Red-black ICM on the candidate layers; returns labels and the selected arrays."""
    n, z, own, rgb = tv["n"], tv["z"], tv["owner"], tv["rgb"]
    H, W, L = n.shape
    occ = (n > 0) & (n >= MIN_N)
    valid = occ.any(-1)
    init = np.where(occ, z, -np.inf).argmax(-1)
    lab = init.copy()

    def sel(arr):
        return np.take_along_axis(arr, lab[..., None], -1)[..., 0]

    zc = np.where(occ, z, 0.0)
    ownc = own.astype(np.float64)
    rgbc = rgb
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(iters):
        for color in (0, 1):
            m = ((yy + xx) % 2 == color) & valid
            if not m.any():
                continue
            cost = U.copy()
            for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                zj0, zj1 = max(0, dj), H + min(0, dj)
                zi0, zi1 = max(0, di), W + min(0, di)
                src = (slice(zj0, zj1), slice(zi0, zi1))
                dst = (slice(max(0, -dj), H + min(0, -dj)), slice(max(0, -di), W + min(0, -di)))
                nb_lab = lab[src]
                nb_own = np.take_along_axis(own[src], nb_lab[..., None], -1)[..., 0]
                nb_z = np.take_along_axis(z[src], nb_lab[..., None], -1)[..., 0]
                nb_rgb = np.take_along_axis(rgb[src], nb_lab[..., None, None], axis=2)[:, :, 0, :]
                drgb = np.linalg.norm(rgbc[dst] - nb_rgb[..., None, :], axis=-1)      # (h,w,L)
                dz = np.abs(zc[dst] - nb_z[..., None])
                do = (ownc[dst] != nb_own[..., None]).astype(np.float64)
                lam_h_eff = lam_h * np.exp(-drgb / sigma_c)
                cost[dst] += (lam_o * do + lam_h_eff * dz + lam_c * drgb / 255.0)
            cost = np.where(occ, cost, np.inf)
            new = cost.argmin(-1)
            lab = np.where(m, new, lab)

    z_sel = np.take_along_axis(z, lab[..., None], -1)[..., 0]
    own_sel = np.take_along_axis(own, lab[..., None], -1)[..., 0]
    rgb_sel = np.take_along_axis(rgb, lab[..., None, None], axis=2)[:, :, 0, :]
    z_sel = np.where(valid, z_sel, np.nan)
    return {"label": lab, "z": z_sel, "owner": np.where(valid, own_sel, -1), "rgb": rgb_sel,
            "live": valid}


def quality_select(tv):
    n, z, own = tv["n"].astype(np.float64), tv["z"], tv["owner"]
    occ = (n > 0) & (n >= MIN_N)
    score = np.where(occ, n, -np.inf)
    lab = score.argmax(-1)
    valid = occ.any(-1)
    return {"label": lab, "z": np.where(valid, np.take_along_axis(z, lab[..., None], -1)[..., 0], np.nan),
            "owner": np.where(valid, np.take_along_axis(own, lab[..., None], -1)[..., 0], -1),
            "live": valid}


def top_select(tv):
    n, z, own = tv["n"].astype(np.float64), tv["z"], tv["owner"]
    occ = n > 0
    lab = np.where(occ, z, -np.inf).argmax(-1)
    valid = occ.any(-1)
    return {"label": lab, "z": np.where(valid, np.max(np.where(occ, z, -np.inf), -1), np.nan),
            "owner": np.where(valid, np.take_along_axis(own, lab[..., None], -1)[..., 0], -1),
            "live": valid}


# --------------------------------------------------------------------------------------
def tile_metrics(tv, sel, masks_full):
    """Metrics on the tile interior, on the fixed raw shared2 pairs."""
    j0, i0, h, w = tv["j0"], tv["i0"], tv["h"], tv["w"]
    z, own, live = sel["z"], sel["owner"], sel["live"]
    # interior window inside the halo
    oj, oi = j0 - tv["J0"], i0 - tv["I0"]
    zz = z[oj:oj + h, oi:oi + w]
    oo = own[oj:oj + h, oi:oi + w]
    ll = live[oj:oj + h, oi:oi + w]

    n_pairs = n_jump = n_switch = 0
    for axis in (0, 1):
        m = masks_full[axis][j0:j0 + (h - 1 if axis == 0 else h),
                             i0:i0 + (w if axis == 0 else w - 1)]
        if axis == 0:                       # down neighbour: mask is (h-1, w)
            za, zb = zz[:-1, :], zz[1:, :]
            oa, ob = oo[:-1, :], oo[1:, :]
            la, lb = ll[:-1, :], ll[1:, :]
        else:                               # right neighbour: mask is (h, w-1)
            za, zb = zz[:, :-1], zz[:, 1:]
            oa, ob = oo[:, :-1], oo[:, 1:]
            la, lb = ll[:, :-1], ll[:, 1:]
        ok = m & la & lb
        d = np.abs(za - zb)
        n_pairs += int(ok.sum())
        n_jump += int((ok & (d > H_TOL)).sum())
        n_switch += int((ok & (oa != ob)).sum())

    from stream_ortho.eval.grid_metrics import ripple_amplitude
    from stream_ortho.eval.pivotb_preflight import seam_index
    tile_prod = {"n": tv["n"], "z_ref": tv["z"], "owner": tv["owner"], "rgb": tv["rgb"],
                 "sum_w": tv["sum_w"], "sum_wz": tv["sum_wz"], "sum_wz2": tv["sum_wz2"],
                 "cell": 0.45, "W": tv["n"].shape[1], "H": tv["n"].shape[0], "L": tv["n"].shape[2],
                 "ox": 0, "oy": 0, "x0": 0, "y0": 0}
    sel_full = {"z": np.nan_to_num(z, nan=0.0), "rgb": sel.get("rgb", np.zeros(z.shape + (3,))),
                "n": np.zeros(z.shape), "live": live, "label": sel["label"].astype(np.int16)}
    ra = ripple_amplitude(tile_prod, sel_full)["ra"]
    sei = seam_index(tile_prod)["sei"]

    # building height: selected surface above the cell's own lowest layer
    occ = tv["n"] > 0
    zlow = np.where(occ, tv["z"], np.inf).min(-1)
    n_layers = occ.sum(-1)
    # building height only means something where the cell actually has >= 2 layers
    hgt = np.where(ll & (n_layers[oj:oj + h, oi:oi + w] >= 2),
                   zz - zlow[oj:oj + h, oi:oi + w], np.nan)
    hgt_top = np.where(ll & (n_layers[oj:oj + h, oi:oi + w] >= 2),
                       np.where(occ, tv["z"], -np.inf).max(-1)[oj:oj + h, oi:oi + w]
                       - zlow[oj:oj + h, oi:oi + w], np.nan)
    ztop = np.where(occ, tv["z"], -np.inf).max(-1)[oj:oj + h, oi:oi + w]
    top_ok = ll & (np.abs(zz - ztop) <= 0.5)
    return {"n_pairs": n_pairs, "n_jump": n_jump, "lcr_fixed": 1 - n_jump / max(n_pairs, 1),
            "owner_switch": n_switch, "ra": float(ra), "sei": float(sei),
            "coverage": float(ll.mean()), "top_preserved_frac": float(top_ok.sum() / max(ll.sum(), 1)),
            "height_median": float(np.nanmedian(hgt)) if np.isfinite(hgt).any() else float("nan"),
            "height_top_median": float(np.nanmedian(hgt_top)) if np.isfinite(hgt_top).any() else float("nan"),
            "n_multi_layer": int(np.isfinite(hgt).sum()),
            "n_cells": int(ll.sum())}


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iter", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--icm-iters", type=int, default=15)
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = S.SyncConfig(ridge=a.ridge)
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    prod_raw, prod_it3 = calibrate(d, ix, iy, kfs, slices, bbox, origin, cfg, a.iter)
    z_raw, own_raw = top_of(prod_raw)
    z_it3, own_it3 = top_of(prod_it3)
    tiles = pick_tiles(own_raw, own_it3)
    for t in tiles:
        t["bbox_m"] = [t["i0"] * d["cell"], t["j0"] * d["cell"],
                       (t["i0"] + t["w"]) * d["cell"], (t["j0"] + t["h"]) * d["cell"]]

    from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs
    masks_full = shared2_pairs(prod_raw)

    configs = [dict(lam_o=1.0, lam_h=0.5, lam_c=0.3, name="default"),
               dict(lam_o=0.0, lam_h=0.5, lam_c=0.3, name="no-owner (ablation)"),
               dict(lam_o=0.5, lam_h=0.5, lam_c=0.3, name="weak owner"),
               dict(lam_o=2.0, lam_h=0.5, lam_c=0.3, name="strong owner"),
               dict(lam_o=1.0, lam_h=0.25, lam_c=0.3, name="weak height"),
               dict(lam_o=1.0, lam_h=1.0, lam_c=0.3, name="strong height")]

    out_tiles = []
    for t in tiles:
        tv_raw = tile_view(prod_raw, t["j0"], t["i0"], t["h"], t["w"])
        tv_it3 = tile_view(prod_it3, t["j0"], t["i0"], t["h"], t["w"])
        # rgb: rebuild products with rgb is expensive; approximate by zeroing -> colour term off
        # (documented in README: rgb is not tracked in the product cache of this run)
        base = {
            "raw": tile_metrics(tv_raw, top_select(tv_raw), masks_full),
            "it3": tile_metrics(tv_it3, top_select(tv_it3), masks_full),
            "top": tile_metrics(tv_it3, top_select(tv_it3), masks_full),
            "quality": tile_metrics(tv_it3, quality_select(tv_it3), masks_full),
        }
        U = unary(tv_it3)
        mrf = {}
        for c in configs:
            sel = icm(tv_it3, U, lam_o=c["lam_o"], lam_h=c["lam_h"], lam_c=c["lam_c"],
                      iters=a.icm_iters)
            m = tile_metrics(tv_it3, sel, masks_full)
            m["params"] = {k: v for k, v in c.items() if k != "name"}
            mrf[c["name"]] = m
        out_tiles.append({"tile": t, "baselines": base, "mrf": mrf})

    # ---- gate on the default configuration ----
    def gate(m, raw, it3, top_base):
        return {
            "lcr_fixed": bool(m["lcr_fixed"] >= raw["lcr_fixed"] - 0.002),
            "n_jump": bool(m["n_jump"] <= raw["n_jump"]),
            "owner_switch": bool(m["owner_switch"] <= raw["owner_switch"] and
                                 m["owner_switch"] <= 0.8 * it3["owner_switch"]),
            "pc": True,          # E2 never touches frames -> PC invariant (documented)
            "ra": bool(m["ra"] <= raw["ra"] * 0.95),
            "sei": bool(m["sei"] <= raw["sei"]),
            "coverage": bool(abs(m["coverage"] - raw["coverage"]) <= 0.01),
            "top_preserved": bool(m["top_preserved_frac"] >= 0.95),
            "height_retention": bool(m["height_median"] >= 0.90 * top_base["height_median"]),
        }

    # ---- rasters for the figures ----
    rasters = {}
    for ot in out_tiles:
        tid = ot["tile"]["id"]
        tv_r = tile_view(prod_raw, ot["tile"]["j0"], ot["tile"]["i0"], ot["tile"]["h"], ot["tile"]["w"], halo=0)
        tv_i = tile_view(prod_it3, ot["tile"]["j0"], ot["tile"]["i0"], ot["tile"]["h"], ot["tile"]["w"], halo=0)
        U = unary(tv_i)
        rasters[f"t{tid}_raw_z"] = top_select(tv_r)["z"]
        rasters[f"t{tid}_raw_o"] = top_select(tv_r)["owner"]
        rasters[f"t{tid}_it3_z"] = top_select(tv_i)["z"]
        rasters[f"t{tid}_it3_o"] = top_select(tv_i)["owner"]
        sel = icm(tv_i, U, iters=a.icm_iters)
        rasters[f"t{tid}_mrf_z"] = sel["z"]
        rasters[f"t{tid}_mrf_o"] = sel["owner"]
        sel0 = icm(tv_i, U, lam_o=0.0, iters=a.icm_iters)
        rasters[f"t{tid}_noown_z"] = sel0["z"]
        rasters[f"t{tid}_noown_o"] = sel0["owner"]
    np.savez(pathlib.Path(a.out) / "pilot_maps.npz", **rasters,
             cell=np.float64(d["cell"]),
             **{f"t{ot['tile']['id']}_j0": ot["tile"]["j0"] for ot in out_tiles},
             **{f"t{ot['tile']['id']}_i0": ot["tile"]["i0"] for ot in out_tiles})

    gate_all, failures = {}, []
    for ot in out_tiles:
        raw, it3 = ot["baselines"]["raw"], ot["baselines"]["it3"]
        m = ot["mrf"]["default"]
        g = gate(m, raw, it3, ot["baselines"]["top"])
        gate_all[f"tile{ot['tile']['id']}"] = g
        failures += [f"tile{ot['tile']['id']}:{k}" for k, v in g.items() if not v]

    passed = all(v for g in gate_all.values() for v in g.values())
    res = {"seq": a.seq, "cell": d["cell"], "n_frames": len(kfs),
           "tiles": tiles, "params_default": {"lam_o": 1.0, "lam_h": 0.5, "lam_c": 0.3,
                                              "icm_iters": a.icm_iters, "tau_nd": TAU_ND,
                                              "min_n": MIN_N},
           "configs": configs, "tiles_result": out_tiles,
           "gate": gate_all, "passed": bool(passed), "failures": failures,
           "notes": {"pc": "invariant: E2 changes the product selection, not the frames",
                     "rgb": "products now carry conf-weighted rgb; lambda_c and the colour "
                            "edge-awareness act on it",
                     "column_baseline": "N/A in a cell-layer framework"},
           "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "command": "python -m stream_ortho.scripts.diag.e2_pilot " + " ".join(argv or []),
           "wall_s": time.time() - t0}

    p = pathlib.Path(a.out) / "pilot.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for ot in out_tiles:
        print(f"--- tile {ot['tile']['id']} bbox {ot['tile']['bbox_m']} "
              f"new_boundaries {ot['tile']['new_boundaries']}")
        for k, m in list(ot["baselines"].items()) + list(ot["mrf"].items()):
            print(f"  {k:20s} LCR {m['lcr_fixed']:.4f} jump {m['n_jump']:5d} switch {m['owner_switch']:5d} "
                  f"RA {m['ra']:.3f} SEI {m['sei']:5.2f} top {m['top_preserved_frac']:.3f} "
                  f"hgt {m['height_median']:.2f}")
    print("gate:", json.dumps(gate_all, ensure_ascii=False), "passed:", passed)
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
