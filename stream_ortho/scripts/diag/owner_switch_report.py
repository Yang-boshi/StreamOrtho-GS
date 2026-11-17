"""T3.8-A -- classify the owner-boundary pairs: can E2 (owner/layer-coherent selection) fix them?

T3.7 showed the LCR regression is not "wider seams" but "+46% owner-boundary pairs": the
calibration rewrites z, the layers regroup, and 6.4% of the cells change which frame owns
their top layer.  Before committing a week to E2 we need to know whether those new
boundaries are *fixable*: do the two cells share a frame, or at least a layer that can be
matched within ``z_gap``?

Per fixed pair (the raw product's shared2 neighbours):

    stable_same          same owner in raw and in it3
    boundary_removed     different in raw, same in it3      (improvement)
    new_boundary         same in raw, different in it3      (created by the calibration)
    persistent_boundary  different in both

For every ``new_boundary`` / ``persistent_boundary`` pair:

* ``with_common_frame``  the two cells share at least one keyframe (64-bit kf mask per cell)
* ``with_common_layer``  both have an occupied layer (n >= 2) within ``z_gap`` of each other
* ``coverage_only``      neither -> no layer choice can make them agree

Gate: ``new_boundary.with_common_layer / new_boundary.n``
    >= 50%  -> E2 GO
    20-50%  -> E2 partial (pilot on 1-2 sectors)
    <20%    -> E2 cannot fix it

The script also stores downsampled top-surface grids per iteration so the map evolution can
be rendered (``maps_evolution.npz`` -> ``figs/map_iter_*.png``).

Usage::

    python -m stream_ortho.scripts.diag.owner_switch_report \
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

Z_GAP = 2.0
MIN_SUPPORT = 2
DOWN = 4


# --------------------------------------------------------------------------------------
def top_of(prod):
    occ = prod["n"] > 0
    top = np.where(occ, prod["z_ref"], -np.inf).argmax(-1)
    z = np.where(occ.any(-1), np.max(np.where(occ, prod["z_ref"], -np.inf), -1), np.nan)
    own = np.take_along_axis(prod["owner"], top[..., None], -1)[..., 0]
    return z, np.where(occ.any(-1), own, -1)


def cell_kf_mask(d, prod, n_cells):
    """64-bit mask of "which keyframes observed this cell" (kf < 64 for P0)."""
    cell = (d["iy"] - prod["oy"]) * prod["W"] + (d["ix"] - prod["ox"])
    kf = d["kf"].astype(np.int64)
    ok = (cell >= 0) & (cell < n_cells)
    comp = np.unique(cell[ok].astype(np.int64) * 64 + kf[ok])
    c, k = comp // 64, comp % 64
    mask = np.zeros(n_cells, np.uint64)
    np.bitwise_or.at(mask, c, np.left_shift(np.uint64(1), k.astype(np.uint64)))
    return mask


def cell_layers(prod, min_support=MIN_SUPPORT):
    """(n_cells, L) heights of the supported layers, NaN where empty."""
    n = prod["n"].reshape(-1, prod["L"])
    z = prod["z_ref"].reshape(-1, prod["L"]).copy()
    z[n < min_support] = np.nan
    return z


def pair_indices(mask, W, axis):
    jj, ii = np.nonzero(mask)
    a = jj * W + ii
    b = jj * W + ii + (1 if axis == 1 else W)
    return a, b


def has_common_layer(za, zb, z_gap=Z_GAP):
    if za.size == 0:
        return np.zeros(0, bool)
    diff = np.abs(za[:, :, None] - zb[:, None, :])       # (M, L, L)
    m = (diff <= z_gap) & np.isfinite(diff)
    return m.any(-1).any(-1)


# --------------------------------------------------------------------------------------
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
    d["ix"], d["iy"] = ix, iy
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs

    prod_raw, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox)
    masks = shared2_pairs(prod_raw)
    W = prod_raw["W"]
    n_cells = prod_raw["H"] * W

    # ---- all-frame iteration, capturing the top surface after every round ----
    snaps = {}
    zr, own_r = top_of(prod_raw)
    snaps["raw"] = (zr[::DOWN, ::DOWN].astype(np.float32), own_r[::DOWN, ::DOWN].astype(np.int32))
    prod = prod_raw
    for it in range(1, a.iter + 1):
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
        zi, oi = top_of(prod)
        snaps[f"it{it}"] = (zi[::DOWN, ::DOWN].astype(np.float32), oi[::DOWN, ::DOWN].astype(np.int32))
    prod_it3 = prod
    zi, own_i = top_of(prod_it3)

    # ---- classify the fixed pairs ----
    counts = {"stable_same": 0, "boundary_removed": 0, "new_boundary": 0, "persistent_boundary": 0}
    new_pairs, pers_pairs = [], []
    for axis, m in masks.items():
        a_, b_ = pair_indices(m, W, axis)
        same_r = own_r.ravel()[a_] == own_r.ravel()[b_]
        same_i = own_i.ravel()[a_] == own_i.ravel()[b_]
        cls = np.where(same_r,
                       np.where(same_i, "stable_same", "new_boundary"),
                       np.where(same_i, "boundary_removed", "persistent_boundary"))
        for c in counts:
            counts[c] += int((cls == c).sum())
        new_pairs.append((a_[cls == "new_boundary"], b_[cls == "new_boundary"]))
        pers_pairs.append((a_[cls == "persistent_boundary"], b_[cls == "persistent_boundary"]))

    maskkf = cell_kf_mask(d, prod_raw, n_cells)
    z3 = cell_layers(prod_it3)

    def classify_boundary(pairs):
        a_ = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros(0, np.int64)
        b_ = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros(0, np.int64)
        if a_.size == 0:
            return {"n": 0, "with_common_frame": 0, "with_common_layer": 0, "coverage_only": 0}
        kf_common = (maskkf[a_] & maskkf[b_]) != np.uint64(0)
        lyr_common = has_common_layer(z3[a_], z3[b_])
        return {"n": int(a_.size), "with_common_frame": int(kf_common.sum()),
                "with_common_layer": int(lyr_common.sum()),
                "with_both": int((kf_common & lyr_common).sum()),
                "coverage_only": int((~kf_common & ~lyr_common).sum())}

    new_b = classify_boundary(new_pairs)
    pers_b = classify_boundary(pers_pairs)
    share_new = new_b["with_common_layer"] / max(new_b["n"], 1)
    share_pers = pers_b["with_common_layer"] / max(pers_b["n"], 1)

    if share_new >= 0.5:
        decision = "E2 GO"
    elif share_new >= 0.2:
        decision = "E2 partial"
    else:
        decision = "E2 cannot fix"

    out = {
        "seq": a.seq, "cell": d["cell"], "n_frames": len(kfs), "iter": a.iter,
        "n_pairs": int(sum(counts.values())), "classes": counts,
        "new_boundary": new_b, "persistent_boundary": pers_b,
        "share_new_with_common_layer": float(share_new),
        "share_persistent_with_common_layer": float(share_pers),
        "share_new_with_common_frame": float(new_b["with_common_frame"] / max(new_b["n"], 1)),
        "decision": decision,
        "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "command": "python -m stream_ortho.scripts.diag.owner_switch_report " + " ".join(argv or []),
        "wall_s": time.time() - t0,
    }
    p = pathlib.Path(a.out) / "owner_switch_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    np.savez(pathlib.Path(a.out) / "maps_evolution.npz",
             **{f"z_{k}": v[0] for k, v in snaps.items()},
             **{f"owner_{k}": v[1] for k, v in snaps.items()},
             down=DOWN, cell=d["cell"])

    print(json.dumps({k: out[k] for k in ("n_pairs", "classes", "new_boundary",
                                          "persistent_boundary", "share_new_with_common_layer",
                                          "share_persistent_with_common_layer", "decision")},
                     indent=1, ensure_ascii=False))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
