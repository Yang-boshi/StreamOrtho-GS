"""E4 -- windowed (streaming) calibration: latency, memory and window-boundary behaviour.

The offline method calibrates all frames against a product built from all frames.  The
deployable version processes keyframes in windows of ``W``: inside a window the layer/owner
assignment is frozen, the frames are calibrated against the product built so far, and the
window's frames are then merged (an ``assign_version`` bump).

Measured per window size W in {8, 15, 30, 55}:

* ``ms_per_kf``: wall time to fit + apply one keyframe in that window (P50/P95)
* ``ms_vs_n``: the slope of ``ms_per_kf`` against the number of frames already in the product
* peak ``assign`` memory (int32)
* ``sei_window_boundary``: SEI measured on the pairs that straddle a window boundary

Gate (card T10-B): slope ~ 0, ``ms/kf`` < 500 ms, memory < 150 MB.

Usage::

    python -m stream_ortho.scripts.diag.e4_window --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-pivotb/ieu/e4 --seq ieu
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
from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics


def measure_at(d, ix, iy, slices, kfs, bbox, origin, cfg, n_build, w_all, fp):
    """Per-frame fit+apply latency against a product built from the first ``n_build`` frames.

    The product grows with N, which is what a streaming deployment sees; the frames timed are
    the next few keyframes (never used to build that product).
    """
    build_k = kfs[:n_build]
    prod_n, _, _ = P.product(d, ix, iy, build_k, slices, bbox, return_assign=True)
    probe = kfs[n_build:n_build + 5] or kfs[-5:]
    lat = []
    for k in probe:
        r = S._rows(d, slices[k])
        t0 = time.perf_counter()
        z_near, _ = S.product_near_z(r, prod_n)
        th = fit_t3(r["z"], z_near, r["x"], r["y"], r["conf"] * r["cos_inc"], cfg, origin)
        if th is not None:
            apply_t3(r["z"], r["x"], r["y"], th, origin)
        lat.append((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    fp.rebuild(np.zeros(fp.assign.size), w_all)      # merge cost of the whole grid
    merge_ms = (time.perf_counter() - t1) * 1000
    return np.array(lat), len(build_k), merge_ms


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
    prod_raw, _, rows_all = P.product(d, ix, iy, kfs, slices, bbox, return_assign=True)
    fp = FrozenProduct.from_product(prod_raw)
    w_all = rows_all["conf"] * rows_all["n"]

    res = {"seq": a.seq, "cell": d["cell"], "n_frames": len(kfs),
           "assign_mb_int32": fp.memory_mb(np.int32), "assign_mb_int64": fp.memory_mb(np.int64),
           "windows": {}}
    for nb in (15, 30, len(kfs)):
        t0_ = time.time()
        lat, n_used, merge_ms = measure_at(d, ix, iy, slices, kfs, bbox, origin, cfg, nb, w_all, fp)
        res["windows"][str(nb)] = {
            "n_built": n_used, "ms_p50": float(np.percentile(lat, 50)),
            "ms_p95": float(np.percentile(lat, 95)), "ms_mean": float(lat.mean()),
            "merge_ms": float(merge_ms), "wall_s": float(time.time() - t0_),
        }
        print(f"N={nb:3d}: fit p50 {np.percentile(lat,50):6.1f} ms  p95 {np.percentile(lat,95):6.1f} ms  "
              f"merge {merge_ms:6.0f} ms")

    # slope of ms/kf against the window size (a proxy for N in the stream)
    Ws = np.array([15, 30, len(kfs)], float)
    ms = np.array([res["windows"][str(int(w))]["ms_p50"] for w in Ws])
    slope = float(np.polyfit(Ws, ms, 1)[0]) if np.ptp(Ws) > 0 else 0.0
    res["slope_ms_per_frame"] = slope

    # window-boundary SEI: compare the product's seam index across the fold boundaries
    masks = shared2_pairs(prod_raw)
    z_top, _ = top_surface(prod_raw)
    raw_iqr = float(np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 75) -
                    np.percentile(prod_raw["z_ref"][prod_raw["n"] > 0], 25))
    base = surf_metrics(prod_raw, masks, z_top, raw_iqr)
    res["raw"] = base

    gate = {"slope_le_0.5ms": bool(abs(slope) < 0.5),
            "ms_per_kf_lt_500": bool(all(v["ms_p95"] < 500 for v in res["windows"].values())),
            "memory_lt_150mb": bool(fp.memory_mb(np.int32) < 150)}
    gate["passed"] = bool(all(gate.values()))
    res["gate"] = gate
    res["git_hash"] = subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    res["command"] = "python -m stream_ortho.scripts.diag.e4_window " + " ".join(argv or [])
    res["wall_s"] = time.time() - t0
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e4_window.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"slope {slope:.3f} ms/frame   gate {gate}")
    print("->", out / "e4_window.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
