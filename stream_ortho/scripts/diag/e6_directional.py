"""T6 -- why is odd->even weaker than even->odd?  Per held-frame diagnosis.

For both even/odd directions and every held frame ``k``:

``overlap_frac``    |S_k ∩ S_build| / |S_k|, S_k = the frame's single-layer flat cells,
                    S_build = cells occupied by the build product
``dist_to_build``   distance from the frame's plane-frame centroid to the nearest build
                    frame's centroid (a camera-position proxy; ``idx_gap`` is also reported)
``pc_raw/frozen``   the frame's own pairwise consistency against its 6 best build
                    neighbours, before and after the 3-DOF correction
``ra_subset``       RA over the cells the held frames actually contribute to (the frozen
                    owner is inherited from the build product, so "owner == k" cannot happen;
                    the *corrected* cells are the ones the held frames touch)

Verdict: if ``pc_improvement`` falls with overlap and odd->even has clearly lower overlap,
the gap is **coverage**; if overlap is similar but the improvement differs, it is
**structure**; if the corrected-subset RA improves while the whole-product RA does not, the
RA miss is **dilution** (the product still has ~80% raw frames).

Usage::

    python -m stream_ortho.scripts.diag.e6_directional \
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
from stream_ortho.scripts.diag.e6_heldout import assign_rows, rebuild
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs

TOP_PCT = 99.0


def frame_pc(d, slices, k, build_k, eval_cells, zfun, n_neighbors=6):
    """Median |dz| of frame k against its best-overlapping build frames."""
    rk = S._rows(d, slices[k])
    sk = S.single_layer_cells(rk["key"])
    ov = sorted(((S._intersect_sorted(sk, S.single_layer_cells(
        S._rows(d, slices[j])["key"])).size, j) for j in build_k), reverse=True)
    zk = zfun(k, rk)
    o = np.argsort(rk["key"], kind="stable")
    ks, zs = rk["key"][o], zk[o]
    meds, npairs = [], 0
    for _, j in ov[:n_neighbors]:
        rj = S._rows(d, slices[j])
        common = S._intersect_sorted(sk, S.single_layer_cells(rj["key"]))
        if eval_cells is not None:
            common = S._intersect_sorted(common, eval_cells)
        if common.size < 50:
            continue
        oj = np.argsort(rj["key"], kind="stable")
        a = S._lookup(ks, zs, common)
        b = S._lookup(rj["key"][oj], zfun(j, rj)[oj], common)
        meds.append(float(np.median(np.abs(a - b))))
        npairs += int(common.size)
    return (float(np.median(meds)) if meds else float("nan")), npairs


def ra_on(prod, mask):
    """RA restricted to ``mask``: RMS of the high-pass of the top surface."""
    from scipy.ndimage import gaussian_filter
    occ = prod["n"] > 0
    z = np.where(occ, prod["z_ref"], np.nan)
    zt = np.max(np.where(np.isfinite(z), z, -np.inf), -1)
    live = np.isfinite(zt) & occ.any(-1)
    if not live.any():
        return float("nan")
    zf = np.where(live, zt, np.nanmedian(zt[live]))
    r = zt - gaussian_filter(zf, 3.0, mode="wrap")
    m = live & mask
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean(r[m] ** 2)))


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
    centroid = {k: (float(np.mean(d["x"][slices[k]])), float(np.mean(d["y"][slices[k]])))
                for k in kfs}

    even, odd = kfs[0::2], kfs[1::2]
    out = {"seq": a.seq, "cell": d["cell"], "directions": {}}
    for name, (bk, hk) in {"even_build_odd_held": (even, odd),
                           "odd_build_even_held": (odd, even)}.items():
        prod_b, _, _ = P.product(d, ix, iy, bk, slices, bbox, return_assign=True)
        eval_cells = S.eval_cells_of(prod_b)
        s_build = np.unique(np.nonzero((prod_b["n"] > 0).any(-1).ravel())[0])

        rows_h_idx = np.sort(np.concatenate([slices[k] for k in hk]))
        rows_h = S._rows(d, rows_h_idx)
        kf_h = d["kf"][rows_h_idx]
        z_cor = rows_h["z"].copy()
        z_near_by_k = {}
        for k in hk:
            m = kf_h == k
            sub = {kk: vv[m] for kk, vv in rows_h.items()}
            z_near, _ = S.product_near_z(sub, prod_b)
            z_near_by_k[k] = z_near
            th = fit_t3(sub["z"], z_near, sub["x"], sub["y"],
                        sub["conf"] * sub["cos_inc"], cfg, origin)
            if th is not None:
                z_cor[m] = apply_t3(sub["z"], sub["x"], sub["y"], th, origin)

        # per-frame metrics
        frames = []
        for k in hk:
            rk = S._rows(d, slices[k])
            sk = S.single_layer_cells(rk["key"])
            cell = ((rk["x"] / prod_b["cell"]).astype(np.int64) - prod_b["ox"]
                    + (rk["y"] / prod_b["cell"]).astype(np.int64) * prod_b["W"]
                    - prod_b["oy"] * prod_b["W"])
            sbuild = np.unique(np.nonzero((prod_b["n"] > 0).any(-1).ravel())[0])
            ks_in = np.intersect1d(np.sort(cell), sbuild).size
            overlap = ks_in / max(np.unique(cell).size, 1)
            cx, cy = centroid[k]
            dist = min(float(np.hypot(cx - centroid[j][0], cy - centroid[j][1])) for j in bk)
            idx_gap = min(abs(k - j) for j in bk)
            pc_r, npr = frame_pc(d, slices, k, bk, eval_cells, lambda kk, rr: rr["z"])
            pc_f, _ = frame_pc(d, slices, k, bk, eval_cells,
                               lambda kk, rr, _z=z_cor, _kf=kf_h: (
                                   _z[_kf == kk] if kk in hk else rr["z"]))
            frames.append({"kf": int(k), "overlap_frac": float(overlap),
                           "dist_to_build": float(dist), "idx_gap": int(idx_gap),
                           "pc_raw": pc_r, "pc_frozen": pc_f,
                           "pc_improvement": float(pc_r - pc_f),
                           "pc_rel_pct": float(100 * (1 - pc_f / pc_r)) if pc_r else float("nan"),
                           "n_pairs": int(npr)})

        # corrected-subset RA: cells the held frames touch, raw vs corrected
        a_h = assign_rows(rows_h, rows_h["z"], prod_b)
        touched = np.zeros(prod_b["H"] * prod_b["W"], bool)
        ok = a_h >= 0
        touched[(a_h[ok] // prod_b["L"])] = True
        touched = touched.reshape(prod_b["H"], prod_b["W"])
        idx_b = np.sort(np.concatenate([slices[k] for k in bk]))
        rows_b = S._rows(d, idx_b)
        w_b = rows_b["conf"] * rows_b["n"]
        n_b = rows_b["n"].astype(np.float64)
        prod_raw_eval = rebuild(prod_b, np.r_[prod_b["assign"], a_h],
                                np.r_[rows_b["z"], rows_h["z"]],
                                np.r_[w_b, rows_h["conf"] * rows_h["n"]],
                                np.r_[n_b, rows_h["n"].astype(np.float64)])
        prod_cor_eval = rebuild(prod_b, np.r_[prod_b["assign"], a_h],
                                np.r_[rows_b["z"], z_cor],
                                np.r_[w_b, rows_h["conf"] * rows_h["n"]],
                                np.r_[n_b, rows_h["n"].astype(np.float64)])
        ra = {"full_raw": ra_on(prod_raw_eval, np.ones_like(touched)),
              "full_frozen": ra_on(prod_cor_eval, np.ones_like(touched)),
              "subset_raw": ra_on(prod_raw_eval, touched),
              "subset_frozen": ra_on(prod_cor_eval, touched),
              "subset_frac": float(touched.mean())}

        # bins
        ov = np.array([f["overlap_frac"] for f in frames])
        im = np.array([f["pc_improvement"] for f in frames])
        rel = np.array([f["pc_rel_pct"] for f in frames])
        qs = np.quantile(ov, [0, .2, .4, .6, .8, 1.0])
        bins = []
        for i in range(5):
            m = (ov >= qs[i]) & (ov <= qs[i + 1] if i == 4 else ov < qs[i + 1])
            bins.append({"lo": float(qs[i]), "hi": float(qs[i + 1]), "n": int(m.sum()),
                         "pc_improvement_median": float(np.nanmedian(im[m])) if m.any() else float("nan"),
                         "pc_rel_pct_median": float(np.nanmedian(rel[m])) if m.any() else float("nan")})
        dbins = []
        edges = [0, 50, 100, 200, 1e9]
        dd = np.array([f["dist_to_build"] for f in frames])
        for i in range(4):
            m = (dd >= edges[i]) & (dd < edges[i + 1])
            dbins.append({"lo": edges[i], "hi": edges[i + 1], "n": int(m.sum()),
                          "pc_rel_pct_median": float(np.nanmedian(rel[m])) if m.any() else float("nan")})

        # build-product structure (why could the two directions differ at all?)
        occ = prod_b["n"] > 0
        nl = occ.sum(-1)
        own_top = np.take_along_axis(
            prod_b["owner"], np.where(occ, prod_b["z_ref"], -np.inf).argmax(-1)[..., None], -1)[..., 0]
        sw = 0
        for axis, m in shared2_pairs(prod_b).items():
            if axis == 1:
                sa = (slice(None), slice(0, -1)); sb = (slice(None), slice(1, None))
            else:
                sa = (slice(0, -1), slice(None)); sb = (slice(1, None), slice(None))
            sw += int((m & (own_top[sa] != own_top[sb])).sum())
        build_stats = {
            "n_live": int(occ.any(-1).sum()),
            "multi_layer_frac": float((nl >= 2).sum() / max(occ.any(-1).sum(), 1)),
            "mean_layers": float(nl[occ.any(-1)].mean()),
            "n_shared2": int(sum(int(m.sum()) for m in shared2_pairs(prod_b).values())),
            "owner_switch": sw,
        }
        out["directions"][name] = {
            "n_build": len(bk), "n_held": len(hk), "build_stats": build_stats,
            "frames": frames,
            "bins": {"overlap": bins, "distance": dbins},
            "summary": {
                "overlap_median": float(np.nanmedian(ov)),
                "overlap_mean": float(np.nanmean(ov)),
                "pc_improvement_median": float(np.nanmedian(im)),
                "pc_rel_pct_median": float(np.nanmedian(rel)),
                "pc_rel_pct_mean": float(np.nanmean(rel)),
                "dist_median": float(np.nanmedian(dd)),
                "corr_overlap_improvement": float(np.corrcoef(ov, rel)[0, 1])
                if ov.size > 2 and np.std(rel) > 0 else float("nan"),
            },
            "ra": ra,
        }
        s = out["directions"][name]["summary"]
        print(f"{name}: overlap {s['overlap_median']:.3f}  dist {s['dist_median']:.0f} m  "
              f"pc_rel {s['pc_rel_pct_median']:+.1f}%  corr(overlap,rel) "
              f"{s['corr_overlap_improvement']:+.2f}  RA full {ra['full_raw']:.3f}->{ra['full_frozen']:.3f} "
              f"subset {ra['subset_raw']:.3f}->{ra['subset_frozen']:.3f}")

    # ---- verdict ----
    a_ = out["directions"]["even_build_odd_held"]["summary"]
    b_ = out["directions"]["odd_build_even_held"]["summary"]
    ra_a = out["directions"]["even_build_odd_held"]["ra"]
    ra_b = out["directions"]["odd_build_even_held"]["ra"]
    overlap_gap = a_["overlap_median"] - b_["overlap_median"]
    pc_gap = a_["pc_rel_pct_median"] - b_["pc_rel_pct_median"]
    subset_improves = (ra_b["subset_frozen"] < ra_b["subset_raw"])
    if abs(overlap_gap) > 0.05 and pc_gap > 3:
        verdict = "coverage"
    elif subset_improves and not (ra_b["full_frozen"] < ra_b["full_raw"]):
        verdict = "dilution"
    else:
        verdict = "structure"
    out["verdict"] = verdict
    out["verdict_detail"] = {
        "overlap_gap": float(overlap_gap), "pc_gap_pp": float(pc_gap),
        "ra_subset_improves": bool(subset_improves),
        "ra_full_improves": bool(ra_b["full_frozen"] < ra_b["full_raw"]),
    }
    out["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    out["command"] = "python -m stream_ortho.scripts.diag.e6_directional " + " ".join(argv or [])
    out["wall_s"] = time.time() - t0
    p = pathlib.Path(a.out) / "e6_directional.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("verdict:", verdict, out["verdict_detail"])
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
