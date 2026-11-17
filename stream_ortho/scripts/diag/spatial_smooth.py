"""T3.7 -- spatial smoothing of the 3-DOF calibration field (schemes S1 and S2).

The all-frame calibration aligns every frame to the product (PC -56%, RA -9.6%) but each
frame carries its own (b, tx, ty); where neighbouring cells are topped by *different*
frames, the two offsets meet at a cell boundary and create a new seam: on the frozen raw
pair set the jumps went 198k -> 244k (+23%) and LCR_fixed -2.12 pp (T3.6).  The field is
discontinuous in space, not in time.

Two repairs, both applied to the *correction field* C(c) = b_k + tx_k dx + ty_k dy of the
owning frame k(c) (no new representation, no training):

* **S1** Gaussian smoothing of C over flat / low-gradient cells (sigma in cells); the
  correction is then ``z -= (C_smooth - C_own)``, i.e. the field is made spatially
  continuous but keeps its mean.
* **S2** per-cell consensus: every frame that observes the cell contributes its own
  correction, weighted by ``conf * n``; ``C(c) = sum(w C) / sum(w)``.

Metrics (same frozen pair set as T3.6): LCR_fixed, n_jump, RA, SEI, top_changed_frac,
z_IQR, coverage.

Usage::

    python -m stream_ortho.scripts.diag.spatial_smooth \
        --obs result-new/P0/obs/ieu/pointmap --out results/week10-pivotb/ieu/round2
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
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, jump_on, top_surface

H_TOL = 1.0
RAWS = {"lcr_fixed": 0.90750, "n_jump": 198464, "ra": 0.8010, "sei": 8.6271}


# --------------------------------------------------------------------------------------
def calibrate(d, ix, iy, kfs, slices, bbox, origin, cfg, n_iter=3):
    """Round-2 all-frame iteration; returns (prod_raw, prod_it3, theta, z_all, rows)."""
    prod, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox)
    prod_raw = prod
    theta = {}
    for _ in range(n_iter):
        eval_cells = S.eval_cells_of(prod)
        z_new = rows_all["z"].copy()
        for k in kfs:
            r = S._rows(d, slices[k])
            z_near, _ = S.product_near_z(r, prod)
            w = r["conf"] * r["cos_inc"]
            th = S._fit_t3(r["z"], z_near, r["x"] - origin[0], r["y"] - origin[1], w, cfg)
            if th is None:
                theta[int(k)] = np.zeros(3)
                continue
            theta[int(k)] = th
            z_new[slices[k]] = r["z"] - (th[0] + th[1] * (r["x"] - origin[0]) +
                                         th[2] * (r["y"] - origin[1]))
        prod = P.product(d, ix, iy, kfs, slices, bbox, z_override=z_new)[0]
    return prod_raw, prod, theta, z_new, rows_all


def correction_field(prod, theta, origin, cell):
    """C0(c) for every cell: the owning frame's offset+tilt evaluated at the cell centre."""
    H, W = prod["n"].shape[:2]
    occ = prod["n"] > 0
    top = np.where(occ, prod["z_ref"], -np.inf).argmax(-1)
    owner = np.take_along_axis(prod["owner"], top[..., None], -1)[..., 0]
    live = occ.any(-1)
    yy, xx = np.mgrid[0:H, 0:W]
    xc = (xx + prod["ox"] + 0.5) * cell
    yc = (yy + prod["oy"] + 0.5) * cell
    dx, dy = xc - origin[0], yc - origin[1]
    C0 = np.zeros((H, W))
    for k, v in theta.items():
        m = owner == k
        if m.any():
            C0[m] = v[0] + v[1] * dx[m] + v[2] * dy[m]
    return C0, owner, live, top


def flat_mask(prod_raw, pct=60.0):
    """Low-gradient cells of the raw product (buildings keep their original value)."""
    z = np.where(prod_raw["n"] > 0, prod_raw["z_ref"], np.nan)
    zt = np.max(np.where(np.isfinite(z), z, -np.inf), axis=-1)
    zt = np.where(np.isfinite(zt), zt, np.nan)
    gy, gx = np.gradient(zt)
    g = np.hypot(gx, gy)
    live = np.isfinite(zt)
    if live.sum() == 0:
        return live
    thr = np.nanpercentile(g[live], pct)
    return live & np.isfinite(g) & (g <= thr)


def apply_delta(prod, delta):
    """Subtract a per-cell correction from every occupied layer (keeps layer gaps)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prod.items()}
    occ = out["n"] > 0
    for l in range(out["n"].shape[-1]):
        m = occ[..., l]
        out["z_ref"][..., l][m] -= delta[m]
    return out


def metrics(prod, masks, z_raw_top, raw_iqr):
    from stream_ortho.eval.grid_metrics import layer_consistency, ripple_amplitude, coverage_cells
    z_top, live = top_surface(prod)
    sel = {"z": z_top, "rgb": np.zeros(prod["n"].shape[:2] + (3,)),
           "n": np.zeros(prod["n"].shape[:2]), "live": live,
           "label": np.zeros(prod["n"].shape[:2], np.int16)}
    lcr = layer_consistency(prod, sel)
    ra = ripple_amplitude(prod, sel)
    n_pairs, n_jump = jump_on(masks, z_top)
    occ = prod["n"] > 0
    own_top = np.take_along_axis(prod["owner"],
                                 np.where(occ, prod["z_ref"], -np.inf).argmax(-1)[..., None],
                                 -1)[..., 0]
    n_switch = 0
    for axis, m in masks.items():
        if axis == 1:
            aa = (slice(None), slice(0, -1)); bb = (slice(None), slice(1, None))
        else:
            aa = (slice(0, -1), slice(None)); bb = (slice(1, None), slice(None))
        n_switch += int((m & (own_top[aa] != own_top[bb])).sum())
    sei = P.seam_index(prod)
    zz = prod["z_ref"][prod["n"] > 0]
    iqr = float(np.percentile(zz, 75) - np.percentile(zz, 25))
    return {
        "lcr_fixed": float(1.0 - n_jump / max(n_pairs, 1)),
        "n_jump": int(n_jump), "n_pairs": int(n_pairs), "owner_switch": int(n_switch),
        "lcr_std": float(lcr["lcr"]), "ra": float(ra["ra"]),
        "sei": float(sei["sei"]) if np.isfinite(sei["sei"]) else float("nan"),
        "top_changed_frac": float(np.mean(np.abs(np.nan_to_num(z_top - z_raw_top)) > H_TOL)),
        "z_iqr": iqr, "z_iqr_ratio": float(iqr / raw_iqr) if raw_iqr else float("nan"),
        "coverage_cells": float(coverage_cells(prod)["coverage_cells"]),
    }


def owner_split(prod, masks, h_tol=H_TOL):
    """Jump rate at owner boundaries vs inside one frame's patch (diagnosis)."""
    occ = prod["n"] > 0
    top = np.where(occ, prod["z_ref"], -np.inf).argmax(-1)
    own = np.take_along_axis(prod["owner"], top[..., None], -1)[..., 0]
    z = np.where(occ.any(-1), np.max(np.where(occ, prod["z_ref"], -np.inf), -1), np.nan)
    acc = {"same": [0, 0], "diff": [0, 0]}          # [pairs, jumps]
    for axis, m in masks.items():
        if axis == 1:
            a = (slice(None), slice(0, -1)); b = (slice(None), slice(1, None))
        else:
            a = (slice(0, -1), slice(None)); b = (slice(1, None), slice(None))
        d = np.abs(z[a] - z[b])
        jump = m & np.isfinite(d) & (d > h_tol)
        same = m & (own[a] == own[b])
        for name, sel in (("same", same), ("diff", m & ~same)):
            acc[name][0] += int(sel.sum())
            acc[name][1] += int((jump & sel).sum())
    return {k: {"n": v[0], "jumps": v[1],
                "rate": float(v[1] / max(v[0], 1))} for k, v in acc.items()}


def gate(m, raw):
    return {
        "lcr_fixed_ok": bool(m["lcr_fixed"] >= raw["lcr_fixed"] - 0.002),
        "n_jump_ok": bool(m["n_jump"] <= raw["n_jump"] * 1.05),
        "ra_ok": bool(m["ra"] <= raw["ra"] * 0.80),
        "sei_ok": bool(np.isfinite(m["sei"]) and m["sei"] <= raw["sei"]),
        "top_changed_ok": bool(m["top_changed_frac"] < 0.064),     # below the it3 value
        "z_iqr_ok": bool(0.95 <= m["z_iqr_ratio"] <= 1.05),
        "coverage_ok": bool(abs(m["coverage_frac_delta"]) <= 0.01),
    }


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iter", type=int, default=3)
    ap.add_argument("--sigmas", default="0,1,2,4")
    ap.add_argument("--ridge", type=float, default=1e-3)
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = S.SyncConfig(ridge=a.ridge)
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    prod_raw, prod_it3, theta, z_it3, rows_all = calibrate(d, ix, iy, kfs, slices, bbox,
                                                           origin, cfg, a.iter)
    masks = shared2_pairs(prod_raw)
    z_raw_top, live_raw = top_surface(prod_raw)
    raw_iqr = float(np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 75) -
                    np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 25))
    base_cov = float((prod_raw["n"] > 0).any(-1).mean())

    C0, owner, live, top = correction_field(prod_it3, theta, origin, d["cell"])
    flat = flat_mask(prod_raw) & (owner >= 0)

    def eval_prod(prod_):
        m = metrics(prod_, masks, z_raw_top, raw_iqr)
        m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
        m["owner_split"] = owner_split(prod_, masks)
        return m

    results = []
    # baseline: raw and it3 (sigma = 0)
    raw_m = eval_prod(prod_raw)
    it3_m = eval_prod(prod_it3)
    results.append({"method": "raw", "sigma": None, **raw_m, "coverage_frac_delta": 0.0})
    results.append({"method": "it3", "sigma": 0.0, **it3_m})

    # ---- S1: Gaussian smoothing of the owning correction, applied on flat cells ----
    from scipy.ndimage import gaussian_filter
    for sig in [float(s) for s in a.sigmas.split(",") if float(s) > 0]:
        w = flat.astype(np.float64)
        num = gaussian_filter(C0 * w, sig, mode="nearest")
        den = gaussian_filter(w, sig, mode="nearest")
        Cs = np.where(den > 1e-6, num / np.maximum(den, 1e-9), C0)
        delta = np.where(flat & live, Cs - C0, 0.0)
        prod_s = apply_delta(prod_it3, delta)
        m = eval_prod(prod_s)
        results.append({"method": "S1", "sigma": sig, **m})

    # ---- S2: per-cell multi-frame consensus of the correction ----
    cell_flat = (iy[allidx] - prod_it3["oy"]) * prod_it3["W"] + (ix[allidx] - prod_it3["ox"])
    B = np.array([theta[int(k)] for k in kfs])
    idx = np.searchsorted(np.array(kfs), d["kf"][allidx])
    xc = (ix[allidx] + 0.5) * d["cell"]; yc = (iy[allidx] + 0.5) * d["cell"]
    corr = B[idx, 0] + B[idx, 1] * (xc - origin[0]) + B[idx, 2] * (yc - origin[1])
    wgt = rows_all["conf"] * rows_all["n"]
    n_cells = prod_it3["H"] * prod_it3["W"]
    num = np.bincount(cell_flat, weights=corr * wgt, minlength=n_cells)
    den = np.bincount(cell_flat, weights=wgt, minlength=n_cells)
    C2 = np.zeros(n_cells)
    ok = den > 0
    C2[ok] = num[ok] / den[ok]
    C2 = C2.reshape(prod_it3["H"], prod_it3["W"])
    delta2 = np.where(live, C2 - C0, 0.0)
    prod_s2 = apply_delta(prod_it3, delta2)
    m2 = eval_prod(prod_s2)
    results.append({"method": "S2", "sigma": None, **m2})

    # ---- gate evaluation ----
    raw_ref = {"lcr_fixed": raw_m["lcr_fixed"], "n_jump": raw_m["n_jump"],
               "ra": raw_m["ra"], "sei": raw_m["sei"]}
    for r in results:
        r["gate"] = gate(r, raw_ref)
        r["gate_passed"] = bool(all(r["gate"].values()))

    passed = [r for r in results if r["method"] in ("S1", "S2") and r["gate_passed"]]
    best = None
    if passed:
        best = max(passed, key=lambda r: (r["lcr_fixed"], -r["n_jump"]))
    verdict = ("pass" if passed else
               ("partial" if any(r["method"] in ("S1", "S2") and
                                 r["lcr_fixed"] >= raw_m["lcr_fixed"] - 0.010 for r in results)
                else "fail"))

    out = {"seq": a.seq, "cell": d["cell"], "n_frames": len(kfs), "iter": a.iter,
           "flat_frac": float(flat.mean()), "raw_ref": raw_ref, "results": results,
           "verdict": verdict, "best": best["method"] + (f"-sigma{best['sigma']}" if best.get("sigma") else "") if best else None,
           "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "command": "python -m stream_ortho.scripts.diag.spatial_smooth " + " ".join(argv or []),
           "wall_s": time.time() - t0}
    p = pathlib.Path(a.out) / "spatial_smooth.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"{'method':10s} {'sigma':>5s} {'LCR_fix':>8s} {'n_jump':>8s} {'RA':>6s} {'SEI':>6s} "
          f"{'topchg':>7s} {'IQRr':>5s} {'pass':>5s}")
    for r in results:
        print(f"{r['method']:10s} {str(r['sigma']):>5s} {r['lcr_fixed']:8.5f} {r['n_jump']:8d} "
              f"{r['ra']:6.3f} {r['sei']:6.2f} {r['top_changed_frac']:7.3f} "
              f"{r['z_iqr_ratio']:5.3f} {str(r['gate_passed']):>5s}")
    print("verdict:", verdict, "best:", out["best"])
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
