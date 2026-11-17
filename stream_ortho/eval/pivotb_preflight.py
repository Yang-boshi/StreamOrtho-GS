"""PIVOT-B pre-check (``13-任务卡-PIVOT-B预检``): is ``abs_t3``'s PC gain real?

Three experiments, no new representation and no training:

**E-A even/odd** -- build the product from even keyframes, hold out the odd ones (and vice
versa).  If the 3-DOF calibration still cuts the held<->build PC by >= 20% in *both*
directions, the gain cannot be an artefact of "the held frame saw the product it is scored
against"; if it vanishes, PIVOT-B has no evidence.

**E-B full-frame rebuild** -- E1 only corrected the held frames, so LCR/RA could not move.
Here the calibration is applied to *every* frame and the product is rebuilt, per fold, from
build frames + corrected held frames.  Compared with the raw full-frame product on LCR / RA
/ SEI / coverage.

**E-C reference, head, field stability** -- no external ortho exists, so the even-built and
odd-built products are cross-evaluated (NCC / Rel-RMSE, marked ``[非外部参考]``); ``abs_t3``
is repeated on the head source; the per-frame ``(b, tx, ty)`` sequence is checked for
smoothness (std, adjacent step).

Usage::

    python -m stream_ortho.eval.pivotb_preflight --obs result-new/P0/obs/ieu/pointmap \\
        --out results/week10-pivotb/ieu --seq ieu
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho.eval import sync_eval as S


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def load(obs_path):
    obs = np.load(pathlib.Path(obs_path) / "obs_merged.npz")
    d = S.load_obs(obs_path)
    return d, obs["ix"].astype(np.int64), obs["iy"].astype(np.int64)


def _bbox(ix, iy, cell, margin=2):
    return ((ix.min() - margin) * cell, (iy.min() - margin) * cell,
            (ix.max() + margin) * cell, (iy.max() + margin) * cell)


def product(d, ix, iy, kfs, slices, bbox, z_override=None, kf_override=None,
            return_assign=False):
    idx = np.sort(np.concatenate([slices[k] for k in kfs]))
    rows = S._rows(d, idx)
    z = rows["z"] if z_override is None else z_override[idx]
    kf = d["kf"][idx] if kf_override is None else kf_override[idx]
    return S.build_product(ix[idx], iy[idx], rows["key"], z,
                           rows["conf"] * rows["n"], rows["n"].astype(np.float64),
                           d["cell"], bbox, kf=kf, rgb=rows["rgb"], return_assign=return_assign), idx, rows


def _nbrs(d, slices, held_k, build_k, n=6, flat=True):
    out = {}
    for k in held_k:
        rk = S._rows(d, slices[k])
        sk = S.single_layer_cells(rk["key"])
        ov = sorted(((S._intersect_sorted(
            sk, S.single_layer_cells(S._rows(d, slices[j])["key"])).size, j) for j in build_k),
            reverse=True)
        out[k] = [j for _, j in ov[:n]]
    return out


def _guard(z_raw, z_corr):
    return S._frame_guard(z_raw, z_corr)


def _top_and_owner(prod):
    n, z, owner = prod["n"], prod["z_ref"], prod.get("owner")
    occ = n > 0
    idx = occ.argmax(-1)
    z_top = np.where(occ, z, -np.inf).max(-1)
    live = occ.any(-1)
    own = np.take_along_axis(owner, idx[..., None], -1)[..., 0] if owner is not None else None
    return {"z": np.where(live, z_top, np.nan), "live": live, "owner": own, "n": n, "z_ref": z}


def surface_metrics(prod) -> dict:
    """LCR / RA / SEI / coverage on a product (SEI = owner-boundary vs interior step)."""
    sel = {"z": _top_and_owner(prod)["z"], "rgb": np.zeros(prod["n"].shape[:2] + (3,)),
           "n": np.zeros(prod["n"].shape[:2]), "live": _top_and_owner(prod)["live"],
           "label": np.zeros(prod["n"].shape[:2], np.int16)}
    from stream_ortho.eval.grid_metrics import coverage_cells, layer_consistency, ripple_amplitude
    lcr = layer_consistency(prod, sel)
    ra = ripple_amplitude(prod, sel)
    cov = coverage_cells(prod)
    return {"lcr": lcr["lcr"], "lcr_n_shared_pairs": lcr["n_shared_pairs"],
            "ra": ra["ra"], "ra_n_cells": ra["n_cells"],
            "coverage_cells": cov["coverage_cells"],
            "top_preserved_frac": lcr["top_preserved_frac"],
            "z_iqr": float(np.percentile(prod["z_ref"][prod["n"] > 0], 75) -
                           np.percentile(prod["z_ref"][prod["n"] > 0], 25)),
            "z_median": float(np.median(prod["z_ref"][prod["n"] > 0])),
            **{k: v for k, v in seam_index(prod).items() if k != "n_seam"},
            "n_seam": seam_index(prod)["n_seam"]}


def seam_index(prod) -> dict:
    """Owner-boundary step over interior step (SEI).  No owner -> NaN.

    A seam is a 4-neighbour pair whose *chosen* layer comes from different frames; the ratio
    of the mean height step across those pairs to the mean step inside a single frame's
    patch is a direct, annotation-free seam indicator.  Lower is better; 1.0 means the
    product is indistinguishable from a single-frame mosaic.
    """
    t = _top_and_owner(prod)
    z, live, own = t["z"], t["live"], t["owner"]
    if own is None:
        return {"sei": float("nan"), "n_seam": 0}
    seam, interior = [], []
    for axis in (1, 0):
        a = (slice(None), slice(0, -1)) if axis == 1 else (slice(0, -1), slice(None))
        b = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
        ok = live[a] & live[b]
        if not ok.any():
            continue
        d = np.abs(z[a] - z[b])[ok]
        diff_owner = (own[a] != own[b])[ok]
        seam.append(d[diff_owner])
        interior.append(d[~diff_owner])
    s = np.concatenate(seam) if seam else np.zeros(0)
    i = np.concatenate(interior) if interior else np.zeros(0)
    if s.size < 50 or i.size < 50:
        return {"sei": float("nan"), "n_seam": int(s.size)}
    return {"sei": float(np.mean(s) / max(np.mean(i), 1e-9)), "n_seam": int(s.size),
            "seam_step_m": float(np.mean(s)), "interior_step_m": float(np.mean(i))}


# --------------------------------------------------------------------------------------
# E-A even/odd
# --------------------------------------------------------------------------------------
def even_odd(d, ix, iy, kfs, slices, cfg, methods=("raw", "abs_t3", "field4x4")) -> dict:
    out = {}
    for name, (build_k, held_k) in {
            "even_build_odd_held": (kfs[0::2], kfs[1::2]),
            "odd_build_even_held": (kfs[1::2], kfs[0::2])}.items():
        _b = np.sort(np.concatenate([slices[k] for k in build_k]))   # bbox from build only
        bbox = _bbox(ix[_b], iy[_b], d["cell"])
        prod, _, _ = product(d, ix, iy, build_k, slices, bbox)
        eval_cells = S.eval_cells_of(prod)
        origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
        nbr = _nbrs(d, slices, held_k, build_k)
        out[name] = {"n_build": len(build_k), "n_held": len(held_k),
                     "eval_cells": int(eval_cells.size), "methods": {}}
        for m in methods:
            if m == "raw":
                zfun = lambda k, rows: rows["z"]
            else:
                zfun = (lambda mode: (lambda k, rows: S.correct_frame(
                    rows, prod, eval_cells, origin, mode, cfg)))({"abs_t3": "t3",
                                                                  "field4x4": "f4"}[m])
            pc = S.pc_pass(d, slices, held_k, nbr, eval_cells, cfg, zfun)
            guard, hfr = [], {"near": [], "top": []}
            for k in held_k:
                r = S._rows(d, slices[k])
                zc = zfun(k, r)
                hfr["near"].append(S.hfr_rows({**r, "z": zc}, prod)["near"])
                hfr["top"].append(S.hfr_rows({**r, "z": zc}, prod)["top"])
                guard.append(_guard(r["z"], zc))
            h = S.summarise_hfr(hfr)
            h.update({"pc": pc["pc"], "pc_all": pc["pc_all"], "n_pairs": pc["n_pairs"],
                      "z_iqr_ratio": float(np.median([g["z_iqr_ratio"] for g in guard])),
                      "median_z_shift": float(np.median([g["median_z_shift"] for g in guard]))})
            h["held_guard_ok"] = bool(0.95 <= h["z_iqr_ratio"] <= 1.05 and
                                      h["median_z_shift"] < 0.5)
            out[name]["methods"][m] = h

        raw_pc = out[name]["methods"]["raw"]["pc"]
        for m in methods:
            v = out[name]["methods"][m]
            v["pc_vs_raw_pct"] = float(100.0 * (1 - v["pc"] / raw_pc)) if raw_pc else float("nan")
            v["hfr_vs_raw_pct"] = float(100.0 * (1 - v["hfr_near"] /
                                                 out[name]["methods"]["raw"]["hfr_near"]))
    return out


# --------------------------------------------------------------------------------------
# E-B full-frame rebuild
# --------------------------------------------------------------------------------------

def fullframe_iterative(d, ix, iy, kfs, slices, cfg, n_iter=3, pc_frames=10, pc_nbrs=6):
    """Deployment protocol: **every** frame is calibrated, then the product is rebuilt.

    The first version of this function corrected only the held fold (11 of 55 frames) and
    left 44 raw, so the product could not move -- that measured 1/5 of the correction, not
    the method (review 14 §2).  Here each round fits ``abs_t3`` for **all** frames against
    the current product, applies it, rebuilds, and records the product metrics, the frame
    consistency (PC) and the theta statistics (b_std / max adjacent step), so the
    convergence is visible.

    Circularity is covered by E-A (even/odd), not by this loop.
    """
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = _bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    prod_raw, _, rows_all = product(d, ix, iy, kfs, slices, bbox)
    out = {"raw": surface_metrics(prod_raw), "iters": [], "n_iter": int(n_iter)}
    out["raw"]["theta"] = None

    # a fixed frame sample + its top-overlap neighbours: PC is frame-to-frame, independent
    # of the product, and comparable across rounds
    sample = kfs[:: max(1, len(kfs) // pc_frames)][:pc_frames]
    nbr = _nbrs(d, slices, sample, kfs, n=pc_nbrs)

    z_cur = rows_all["z"].copy()
    prod = prod_raw
    prev_b = None
    for it in range(1, int(n_iter) + 1):
        eval_cells = S.eval_cells_of(prod)
        thetas = {}
        z_new = rows_all["z"].copy()
        for k in kfs:
            r = S._rows(d, slices[k])
            z_near, _ = S.product_near_z(r, prod)
            w = r["conf"] * r["cos_inc"]
            th = S._fit_t3(r["z"], z_near, r["x"] - origin[0], r["y"] - origin[1], w, cfg,
                           scale=False)
            if th is None:
                thetas[int(k)] = np.zeros(3)
                continue
            thetas[int(k)] = th
            z_new[slices[k]] = r["z"] - (th[0] + th[1] * (r["x"] - origin[0]) +
                                         th[2] * (r["y"] - origin[1]))
        prod = product(d, ix, iy, kfs, slices, bbox, z_override=z_new)[0]
        z_cur = z_new

        B = np.array([thetas[int(k)] for k in kfs])
        step = np.abs(np.diff(B, axis=0)) if B.shape[0] > 1 else np.zeros((1, 3))
        pcs = S.pc_pass(d, slices, sample, nbr, eval_cells, cfg,
                        lambda k, rows: z_new[slices[k]]) 
        m = surface_metrics(prod)
        m.update({"pc": pcs["pc"], "pc_all": pcs["pc_all"], "n_pairs": pcs["n_pairs"],
                  "b_std": float(np.nanstd(B[:, 0])), "b_range":
                            [float(np.nanmin(B[:, 0])), float(np.nanmax(B[:, 0]))],
                  "max_step": float(np.nanmax(step)), "median_step": float(np.nanmedian(step))})
        if prev_b is not None:
            m["b_std_change"] = float(m["b_std"] - prev_b)
        out["iters"].append(m)
        prev_b = m["b_std"]
    return out


def fullframe(d, ix, iy, kfs, slices, cfg, n_folds=5) -> dict:
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = _bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    prod_raw, idx_all, rows_all = product(d, ix, iy, kfs, slices, bbox)
    out = {"raw": surface_metrics(prod_raw), "folds": []}

    folds = np.array_split(np.arange(len(kfs)), n_folds)
    z_corr_all = rows_all["z"].copy()
    for fi, fold in enumerate(folds):
        held_k = [kfs[i] for i in fold]
        build_k = [k for k in kfs if k not in held_k]
        prod_b, idx_b, _ = product(d, ix, iy, build_k, slices, bbox)
        eval_cells = S.eval_cells_of(prod_b)
        ms = {}
        for m, mode in (("raw", None), ("abs_t3", "t3")):
            z = rows_all["z"].copy()
            if mode is not None:
                for k in held_k:
                    r = S._rows(d, slices[k])
                    zc = S.correct_frame(r, prod_b, eval_cells, origin, mode, cfg)
                    z[slices[k]] = zc
            p, _, _ = product(d, ix, iy, kfs, slices, bbox, z_override=z)
            ms[m] = surface_metrics(p)
            if mode is not None:
                z_corr_all = z
        out["folds"].append({"fold": fi, "n_held": len(held_k), "methods": ms})
    med = {}
    for m in ("raw", "abs_t3"):
        keys = out["raw"].keys() if m == "raw" else out["folds"][0]["methods"]["abs_t3"].keys()
        med[m] = {k: float(np.median([f["methods"][m].get(k, np.nan) for f in out["folds"]]))
                  for k in keys if k != "lcr_n_shared_pairs" and k != "ra_n_cells"}
        med[m]["lcr_n_shared_pairs"] = int(np.median(
            [f["methods"][m].get("lcr_n_shared_pairs", 0) for f in out["folds"]]))
        med[m]["ra_n_cells"] = int(np.median(
            [f["methods"][m].get("ra_n_cells", 0) for f in out["folds"]]))
    out["median"] = med
    out["delta"] = {
        "lcr": med["abs_t3"]["lcr"] - med["raw"]["lcr"],
        "ra_pct": 100.0 * (1 - med["abs_t3"]["ra"] / med["raw"]["ra"]) if med["raw"]["ra"] else float("nan"),
        "coverage": med["abs_t3"]["coverage_cells"] - med["raw"]["coverage_cells"]}
    return out


# --------------------------------------------------------------------------------------
# E-C reference / head / field stability
# --------------------------------------------------------------------------------------
def reference_even_odd(d, ix, iy, kfs, slices) -> dict:
    """Cross-evaluate the even-built and odd-built products (no external reference exists)."""
    idx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = _bbox(ix[idx], iy[idx], d["cell"])
    pe, _, _ = product(d, ix, iy, kfs[0::2], slices, bbox)
    po, _, _ = product(d, ix, iy, kfs[1::2], slices, bbox)
    ze, zo = pe["z_ref"], po["z_ref"]
    ne, no = pe["n"], po["n"]
    ok = (ne > 0).any(-1) & (no > 0).any(-1)
    if ok.sum() < 1000:
        return {"found": False, "type": "none"}
    a = np.where((ne > 0), ze, np.nan).max(-1)[ok]
    b = np.where((no > 0), zo, np.nan).max(-1)[ok]
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
    return {"found": True, "type": "even_odd_internal", "label": "[非外部参考]",
            "n_cells": int(a.size),
            "rel_rmse": float(np.median(np.abs(a - b)) / max(iqr, 1e-9)),
            "ncc": float(np.corrcoef(a, b)[0, 1]) if a.size > 10 else float("nan"),
            "median_abs_m": float(np.median(np.abs(a - b)))}


def theta_stability(d, ix, iy, kfs, slices, cfg, n_folds=5) -> dict:
    """Per-frame (b, tx, ty) from the 5-fold **out-of-fold** abs_t3 fits."""
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = _bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    folds = np.array_split(np.arange(len(kfs)), n_folds)
    theta = {}
    for fold in folds:
        held_k = [kfs[i] for i in fold]
        build_k = [k for k in kfs if k not in held_k]
        prod_b, _, _ = product(d, ix, iy, build_k, slices, bbox)
        eval_cells = S.eval_cells_of(prod_b)
        for k in held_k:
            r = S._rows(d, slices[k])
            z_near, _ = S.product_near_z(r, prod_b)
            x = r["x"] - origin[0]; y = r["y"] - origin[1]
            w = r["conf"] * r["cos_inc"]
            p = S._fit_t3(r["z"], z_near, x, y, w, cfg, scale=False)
            theta[int(k)] = p.tolist() if p is not None else [float("nan")] * 3
    ks = sorted(theta)
    arr = np.array([theta[k] for k in ks])
    step = np.abs(np.diff(arr, axis=0))
    return {"frames": ks, "theta": arr.tolist(),
            "b_std": float(np.nanstd(arr[:, 0])), "tx_std": float(np.nanstd(arr[:, 1])),
            "ty_std": float(np.nanstd(arr[:, 2])),
            "max_step": float(np.nanmax(step)) if step.size else float("nan"),
            "median_step": float(np.nanmedian(step)) if step.size else float("nan"),
            "b_range": [float(np.nanmin(arr[:, 0])), float(np.nanmax(arr[:, 0]))]}


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--head-obs", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--iter", type=int, default=3)
    ap.add_argument("--only", default="all", choices=("all", "iterative"))
    ap.add_argument("--context", default=None, help="round-1 preflight.json, for the decision")
    a = ap.parse_args(argv)

    cfg = S.SyncConfig(ridge=a.ridge)
    t0 = time.time()
    d, ix, iy = load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)

    res = {"seq": a.seq, "source": "pointmap", "cell": d["cell"], "n_frames": len(kfs)}
    if a.only == "iterative":
        res["fullframe_iterative"] = fullframe_iterative(d, ix, iy, kfs, slices, cfg,
                                                         n_iter=a.iter)
        ctx = json.loads(pathlib.Path(a.context).read_text()) if a.context else None
        res["context"] = {"even_odd_gate": ctx["gate"] if ctx else None,
                          "round1_fullframe": ctx["fullframe"] if ctx else None}
        it0, itN = res["fullframe_iterative"]["iters"][0], res["fullframe_iterative"]["iters"][-1]
        raw = res["fullframe_iterative"]["raw"]
        d_lcr = itN["lcr"] - raw["lcr"]
        d_ra = 100.0 * (1 - itN["ra"] / raw["ra"]) if raw["ra"] else float("nan")
        d_sei = 100.0 * (1 - itN["sei"] / raw["sei"]) if raw.get("sei") and np.isfinite(raw["sei"]) else float("nan")
        res["decision"] = {
            "lcr_delta": float(d_lcr), "ra_pct": float(d_ra), "sei_pct": float(d_sei),
            "coverage_delta": float(itN["coverage_cells"] - raw["coverage_cells"]),
            "theta_b_std": [float(x["b_std"]) for x in res["fullframe_iterative"]["iters"]],
            "theta_max_step": [float(x["max_step"]) for x in res["fullframe_iterative"]["iters"]],
            "G_B2_calibration_reaches_product": bool(
                (d_lcr >= 0.005 or d_ra >= 5 or d_sei >= 5) and
                abs(itN["coverage_cells"] - raw["coverage_cells"]) <= 0.01),
            "theta_converged": bool(
                abs(res["fullframe_iterative"]["iters"][-1]["b_std"] -
                    res["fullframe_iterative"]["iters"][0]["b_std"]) < 0.3),
        }
        res["wall_s"] = time.time() - t0
        out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "preflight_iterative.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(json.dumps({"raw": {k: round(v, 4) for k, v in res["fullframe_iterative"]["raw"].items()
                                  if isinstance(v, float)},
                          "iters": [{k: round(v, 4) for k, v in it.items()
                                     if isinstance(v, float)} for it in res["fullframe_iterative"]["iters"]],
                          "decision": res["decision"]}, indent=1, ensure_ascii=False))
        print("->", out / "preflight_iterative.json")
        return 0

    res["even_odd"] = even_odd(d, ix, iy, kfs, slices, cfg)
    res["fullframe"] = fullframe(d, ix, iy, kfs, slices, cfg, n_folds=a.folds)
    res["theta_stability"] = theta_stability(d, ix, iy, kfs, slices, cfg, n_folds=a.folds)
    res["reference"] = reference_even_odd(d, ix, iy, kfs, slices)

    if a.head_obs:
        dh, ixh, iyh = load(a.head_obs)
        sh = S.frame_slices(dh["kf"])
        kh = sorted(sh)
        res["head"] = even_odd(dh, ixh, iyh, kh, sh, cfg,
                               methods=("raw", "abs_t3"))["even_build_odd_held"]

    # ---- gates ----
    ea = res["even_odd"]
    ok = {d_: ea[d_]["methods"]["abs_t3"]["pc_vs_raw_pct"] >= 20 for d_ in ea}
    res["gate"] = {
        "G_A_both_directions": bool(all(ok.values())),
        "G_A_one_direction": bool(any(ok.values())),
        "G_A_detail": {k: float(v) for k, v in
                       {d_: ea[d_]["methods"]["abs_t3"]["pc_vs_raw_pct"] for d_ in ea}.items()},
        "G_B_lcr_not_worse": bool(res["fullframe"]["delta"]["lcr"] >= -1e-3),
        "G_B_ra_not_worse": bool(res["fullframe"]["delta"]["ra_pct"] >= -1.0),
        "G_B_coverage": bool(abs(res["fullframe"]["delta"]["coverage"]) <= 0.01),
        "head_pc_vs_raw_pct": (res.get("head", {}).get("methods", {}).get("abs_t3", {})
                               .get("pc_vs_raw_pct", None)),
    }
    res["wall_s"] = time.time() - t0
    res["command"] = " ".join(["python", "-m", "stream_ortho.eval.pivotb_preflight"] +
                              (argv if argv else []))
    import subprocess
    res["git_hash"] = subprocess.run(
        ["git", "-C", str(pathlib.Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({"gate": res["gate"],
                      "even_odd": {k: {m: round(v["pc"], 4) for m, v in
                                       ea[k]["methods"].items()} for k in ea},
                      "fullframe": res["fullframe"]["median"],
                      "theta_std": [round(res["theta_stability"]["b_std"], 3),
                                    round(res["theta_stability"]["max_step"], 3)],
                      "reference": res["reference"]}, indent=1, ensure_ascii=False))
    print("->", out / "preflight.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
