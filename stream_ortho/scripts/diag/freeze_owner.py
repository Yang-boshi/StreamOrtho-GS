"""F1 -- freeze the layer/owner assignment during the all-frame calibration.

T3.8/T4 showed that E2 (selecting among existing layers) cannot repair the seams, because
``owner`` is a property of the **layer**, which the calibration re-creates every round: the
number of owner switches on the fixed pair set grows 3-4x (13.9k -> 45.6k on a pilot tile).

This experiment tests the structural hypothesis: keep the raw product's
``(cell, layer-slot, owner)`` **frozen** and only let ``theta_k`` move the observations' z.
Concretely, ``build_product`` now returns a per-row ``assign`` (which (cell, slot) each
observation was merged into).  Each round:

1. fit ``abs_t3`` for every frame against the current product;
2. apply it to every observation;
3. rebuild the product by a **weighted mean over the frozen assignment** -- no re-grouping,
   no owner re-election.

If the frozen run keeps PC/RA and brings LCR_fixed back to raw, then "calibration + frozen
layers" is the PIVOT-B method; if PC/RA collapse, freezing blocks the calibration.

Usage::

    python -m stream_ortho.scripts.diag.freeze_owner \
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
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import flat_mask, metrics as surf_metrics


def rebuild_frozen(prod, assign, z, w, n):
    """Rebuild z_ref/sum_* over the frozen (cell, slot) assignment; owner and n untouched."""
    L = prod["n"].shape[-1]
    H, W = prod["n"].shape[:2]
    size = H * W * L
    ok = assign >= 0
    a = assign[ok]
    zz, ww, nn = z[ok], w[ok], n[ok]
    num = np.bincount(a, weights=ww * zz, minlength=size)
    den = np.bincount(a, weights=ww, minlength=size)
    swz2 = np.bincount(a, weights=ww * zz * zz, minlength=size)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prod.items()}
    zref = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0).reshape(H, W, L)
    out["z_ref"] = zref
    out["sum_w"] = den.reshape(H, W, L)
    out["sum_wz"] = num.reshape(H, W, L)
    out["sum_wz2"] = swz2.reshape(H, W, L)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iter", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--context", default=None, help="round2/spatial_smooth.json for it3")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = S.SyncConfig(ridge=a.ridge)
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    prod_raw, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox, return_assign=True)
    assign = prod_raw["assign"]
    masks = shared2_pairs(prod_raw)
    z_raw_top, live = top_surface(prod_raw)
    raw_iqr = float(np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 75) -
                    np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 25))
    base_cov = float((prod_raw["n"] > 0).any(-1).mean())

    w_all = rows_all["conf"] * rows_all["n"]
    n_all = rows_all["n"].astype(np.float64)

    def ev(prod_, extra=None):
        m = surf_metrics(prod_, masks, z_raw_top, raw_iqr)
        m["coverage_frac_delta"] = m["coverage_cells"] - base_cov
        if extra:
            m.update(extra)
        return m

    raw_m = ev(prod_raw)

    prod = prod_raw
    z_cur = rows_all["z"].copy()
    iters = []
    prev_owner = prod_raw["owner"]
    for it in range(1, a.iter + 1):
        eval_cells = S.eval_cells_of(prod)
        z_new = rows_all["z"].copy()
        theta = {}
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
        prod = rebuild_frozen(prod, assign, z_new, w_all, n_all)
        z_cur = z_new
        owner_same = float(np.mean(prod["owner"] == prev_owner))
        m = ev(prod, {"iter": it, "owner_preserved_frac": owner_same})
        iters.append(m)

    # PC on the calibrated frames (frame-to-frame; the frozen product does not enter)
    sample = kfs[:: max(1, len(kfs) // 10)][:10]
    nbr = {}
    for k in sample:
        rk = S._rows(d, slices[k])
        sk = S.single_layer_cells(rk["key"])
        ov = sorted(((S._intersect_sorted(sk, S.single_layer_cells(S._rows(d, slices[j])["key"])).size, j)
                     for j in kfs), reverse=True)
        nbr[k] = [j for _, j in ov[:6]]
    pc = S.pc_pass(d, slices, sample, nbr, S.eval_cells_of(prod), cfg,
                   lambda k, rows: z_cur[slices[k]])

    # ---- gate (owner_switch is a diagnostic, not a gate -- review 19 §3.2) ----
    fin = iters[-1]
    ctx = json.loads(pathlib.Path(a.context).read_text()) if a.context else None
    it3 = None
    if ctx:
        for r in ctx.get("results", []):
            if r["method"] == "it3":
                it3 = r
    gate = {
        "lcr_fixed": bool(fin["lcr_fixed"] >= raw_m["lcr_fixed"] - 0.002),
        "n_jump": bool(fin["n_jump"] <= raw_m["n_jump"] * 1.05),
        "pc": bool(pc["pc"] <= 0.30),
        "ra": bool(fin["ra"] <= raw_m["ra"] * 0.95),
        "sei": bool(fin["sei"] <= raw_m["sei"]),
        "coverage": bool(abs(fin["coverage_frac_delta"]) <= 0.01),
        "top_preserved": bool(fin["top_changed_frac"] < 0.064),
        "z_iqr": bool(0.95 <= fin["z_iqr_ratio"] <= 1.05),
    }
    passed = all(gate.values())

    res = {"seq": a.seq, "cell": d["cell"], "n_frames": len(kfs), "iter": a.iter,
           "method": "F1 freeze layer/owner (theta only)",
           "raw": raw_m, "iters": iters, "pc": pc["pc"],
           "it3_reference": it3, "gate": gate, "passed": bool(passed),
           "note": "owner_switch is a diagnostic here (not a gate): owner is frozen to raw, "
                   "so it equals raw's value by construction",
           "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "command": "python -m stream_ortho.scripts.diag.freeze_owner " + " ".join(argv or []),
           "wall_s": time.time() - t0}
    p = pathlib.Path(a.out) / "freeze_owner.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"{'':10s} {'LCR_fixed':>9s} {'n_jump':>8s} {'switch':>7s} {'RA':>6s} {'SEI':>6s} {'PC':>6s} {'IQRr':>5s}")
    print(f"{'raw':10s} {raw_m['lcr_fixed']:9.5f} {raw_m['n_jump']:8d} {raw_m['owner_switch']:7d} "
          f"{raw_m['ra']:6.3f} {raw_m['sei']:6.2f} {'-':>6s} {raw_m['z_iqr_ratio']:5.3f}")
    if it3:
        print(f"{'it3':10s} {it3['lcr_fixed']:9.5f} {it3['n_jump']:8d} {it3.get('owner_switch', -1):7d} "
              f"{it3['ra']:6.3f} {it3['sei']:6.2f} {'-':>6s} {it3['z_iqr_ratio']:5.3f}")
    for m in iters:
        print(f"{'frozen it'+str(m['iter']):10s} {m['lcr_fixed']:9.5f} {m['n_jump']:8d} "
              f"{m['owner_switch']:7d} {m['ra']:6.3f} {m['sei']:6.2f} {'-':>6s} {m['z_iqr_ratio']:5.3f}")
    print(f"PC (frozen, calibrated frames): {pc['pc']:.3f}   it3 PC: "
          f"{ctx.get('fullframe', {}).get('median', {}).get('abs_t3', {}).get('pc', float('nan')) if ctx else float('nan')}")
    print("gate:", json.dumps(gate), "passed:", passed)
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
