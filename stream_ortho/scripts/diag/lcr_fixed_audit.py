"""T3.6 task A -- is the LCR drop a denominator artefact?

The iterative calibration shrank ``n_shared_pairs`` from 2.15 M to 1.73 M (-19.5%) while the
*absolute* number of inconsistent pairs fell from 198 k to 193 k.  A ratio whose denominator
moves 20% cannot be compared across runs (06 §2.3: only when ``n_shared >= 0.9 x raw``), so
this script freezes the denominator:

1. rebuild the raw 55-frame product;
2. rerun the round-2 all-frame 3-round calibration to obtain ``z_it3`` and ``prod_it3``;
3. take the **shared2** pairs of the **raw** product (4-neighbour, ``min(match_fwd,
   match_bwd) >= 2``), ~2.15 M of them;
4. on that same fixed pair set compute ``jump = |z_sel(a) - z_sel(b)| > 1 m`` for raw and for
   it3, and report ``LCR_fixed = 1 - #jump / #pairs`` for both, plus the standard LCR
   numerator/denominator.

Verdict: ``LCR_fixed_it3 - LCR_fixed_raw >= -0.002`` -> the -1.9 pp was an artefact;
``-0.005 .. -0.002`` -> weak regression; ``< -0.005`` -> real regression (then also report
``top_changed_frac`` and the worst tiles, and do **not** write the PIVOT-B card).

Output: ``results/week10-pivotb/{seq}/round2/lcr_fixed.json`` (+ optionally theta_final for
the task-C stability table).

Usage::

    python -m stream_ortho.scripts.diag.lcr_fixed_audit \
        --obs result-new/P0/obs/ieu/pointmap --out results/week10-pivotb/ieu/round2 --seq ieu
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
from stream_ortho.eval.grid_metrics import _pair_match_counts

Z_GAP = 2.0
H_TOL = 1.0


def top_surface(prod):
    """Chosen surface z and liveness of a product (top layer)."""
    t = P._top_and_owner(prod)
    return t["z"], t["live"]


def shared2_pairs(prod, z_gap=Z_GAP):
    """Boolean masks of the shared2 neighbour pairs of ``prod`` (right, down)."""
    occ, z = prod["n"] > 0, prod["z_ref"]
    live = occ.any(-1)
    out = {}
    for axis in (1, 0):
        if axis == 1:
            a = (slice(None), slice(0, -1)); b = (slice(None), slice(1, None))
        else:
            a = (slice(0, -1), slice(None)); b = (slice(1, None), slice(None))
        oA, oB = occ[a], occ[b]
        zA, zB = z[a], z[b]
        mA, mB = _pair_match_counts(zA.reshape(-1, zA.shape[-1]), oA.reshape(-1, oA.shape[-1]),
                                    zB.reshape(-1, zB.shape[-1]), oB.reshape(-1, oB.shape[-1]),
                                    z_gap)
        shared = (np.minimum(mA, mB) >= 2).reshape(oA.shape[:-1])
        out[axis] = (shared & live[a] & live[b])
    return out


def jump_on(masks, z_sel, h_tol=H_TOL) -> tuple[int, int]:
    """(#pairs, #jumps) on a fixed pair set."""
    n = n_jump = 0
    for axis, m in masks.items():
        if axis == 1:
            a = (slice(None), slice(0, -1)); b = (slice(None), slice(1, None))
        else:
            a = (slice(0, -1), slice(None)); b = (slice(1, None), slice(None))
        d = np.abs(z_sel[a] - z_sel[b])
        n += int(m.sum())
        n_jump += int((m & (d > h_tol)).sum())
    return n, n_jump


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iter", type=int, default=3)
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

    prod_raw, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox)
    z_raw, live_raw = top_surface(prod_raw)
    masks = shared2_pairs(prod_raw)
    n_pairs, n_jump_raw = jump_on(masks, z_raw)

    # ---- reproduce the round-2 all-frame iteration ----
    prod = prod_raw
    theta = {}
    for it in range(1, a.iter + 1):
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

    z_it3, live_it3 = top_surface(prod)
    # fixed denominator: the raw shared2 pairs; only the chosen heights change
    n_pairs_it, n_jump_it3 = jump_on(masks, z_it3)

    # standard LCR (own denominator each) for the record
    def std_lcr(prod_):
        from stream_ortho.eval.grid_metrics import layer_consistency
        sel = {"z": top_surface(prod_)[0], "rgb": np.zeros(prod_["n"].shape[:2] + (3,)),
               "n": np.zeros(prod_["n"].shape[:2]), "live": top_surface(prod_)[1],
               "label": np.zeros(prod_["n"].shape[:2], np.int16)}
        r = layer_consistency(prod_, sel)
        return r["lcr"], r["n_shared_pairs"], r["n_inconsistent"]

    lcr_r, sh_r, inc_r = std_lcr(prod_raw)
    lcr_i, sh_i, inc_i = std_lcr(prod)

    lcr_fixed_raw = 1.0 - n_jump_raw / max(n_pairs, 1)
    lcr_fixed_it3 = 1.0 - n_jump_it3 / max(n_pairs, 1)
    delta_pp = 100.0 * (lcr_fixed_it3 - lcr_fixed_raw)

    if delta_pp >= -0.2:
        verdict = "artefact"
    elif delta_pp >= -0.5:
        verdict = "weak_regression"
    else:
        verdict = "real_regression"

    # task C: theta stability on the final iteration
    ks = sorted(theta)
    B = np.array([theta[k] for k in ks])
    step = np.abs(np.diff(B, axis=0))
    jumps = [int(ks[i + 1]) for i in np.nonzero(step[:, 0] > 1.0)[0]]
    fold_bounds = [int(kfs[11 * i]) for i in range(1, 5)]

    out = {
        "seq": a.seq, "cell": d["cell"], "n_frames": len(kfs), "iter": a.iter,
        "n_pairs": int(n_pairs), "n_pairs_it3_check": int(n_pairs_it),
        "lcr_fixed_raw": float(lcr_fixed_raw), "lcr_fixed_it3": float(lcr_fixed_it3),
        "n_jump_raw": int(n_jump_raw), "n_jump_it3": int(n_jump_it3),
        "delta_pp": float(delta_pp), "verdict": verdict,
        "lcr_std_raw": float(lcr_r), "lcr_std_it3": float(lcr_i),
        "n_shared_raw": int(sh_r), "n_shared_it3": int(sh_i),
        "n_inconsistent_raw": int(inc_r), "n_inconsistent_it3": int(inc_i),
        "shared_ratio": float(sh_i / max(sh_r, 1)),
        "z_iqr_raw": float(np.nanpercentile(z_raw[live_raw], 75) - np.nanpercentile(z_raw[live_raw], 25)),
        "z_iqr_it3": float(np.nanpercentile(z_it3[live_it3], 75) - np.nanpercentile(z_it3[live_it3], 25)),
        "theta_stability": {
            "b_std": float(np.nanstd(B[:, 0])), "b_range": [float(np.nanmin(B[:, 0])), float(np.nanmax(B[:, 0]))],
            "max_step": float(np.nanmax(step)), "median_step": float(np.nanmedian(step)),
            "frames_delta_b_gt_1m": jumps,
            "fold_boundaries": fold_bounds,
            "jumps_at_fold_boundary": [f for f in jumps if f in fold_bounds],
        },
        "theta_final": {int(k): [float(v) for v in theta[k]] for k in ks},
        "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "command": "python -m stream_ortho.scripts.diag.lcr_fixed_audit " + " ".join(argv or []),
        "wall_s": time.time() - t0,
    }

    # if it is a real regression, add the diagnostics the review asks for
    if verdict == "real_regression":
        from stream_ortho.eval.grid_metrics import layer_consistency  # noqa: F401
        out["top_changed_frac"] = float(np.mean(np.abs(z_it3 - z_raw) > 1.0))
        worst = []
        for axis, m in masks.items():
            if axis == 1:
                aa = (slice(None), slice(0, -1)); bb = (slice(None), slice(1, None))
            else:
                aa = (slice(0, -1), slice(None)); bb = (slice(1, None), slice(None))
            d_ = np.abs(z_it3[aa] - z_it3[bb])
            bad = m & (d_ > H_TOL)
            if bad.any():
                jj, ii = np.nonzero(bad)
                worst.append({"axis": axis, "n": int(bad.sum()),
                              "tile_x": [int(ii.min()), int(ii.max())],
                              "tile_y": [int(jj.min()), int(jj.max())]})
        out["worst_tiles"] = worst

    path = pathlib.Path(a.out) / "lcr_fixed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_pairs", "lcr_fixed_raw", "lcr_fixed_it3", "delta_pp",
                                          "n_jump_raw", "n_jump_it3", "lcr_std_raw", "lcr_std_it3",
                                          "n_shared_raw", "n_shared_it3", "n_inconsistent_raw",
                                          "n_inconsistent_it3", "verdict")}, indent=1))
    print(json.dumps(out["theta_stability"], indent=1))
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
