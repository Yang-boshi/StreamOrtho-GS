"""Online (replay) streaming orthomapping: one frame in, product and tiles out.

Replays the keyframes of an observation table in temporal order -- exactly what a live
front-end would hand over -- and for each frame

1. calibrates it against the **current** streaming product (one 3-DOF fit, bootstrap for the
   first frames, fallback when the fit is implausible);
2. inserts it with the assignment frozen (no re-grouping, no re-owning);
3. marks the dirty tiles of its footprint and flushes within a time budget;
4. records latency, coverage and, at the checkpoints, quality against the offline oracle.

No global rebuild is ever called, and the offline ``product_frozen.npz`` is only read for the
quality comparison.

    python -m stream_ortho.scripts.run_stream --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-stream/ieu --seq ieu --gsd 0.30 --budget-ms 500
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
from stream_ortho.map.calib import apply_t3, fit_t3
from stream_ortho.online.calib import OnlineCalib, OnlineCalibConfig
from stream_ortho.online.product import StreamingProduct
from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig
from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics


def tile_render_fn_factory(product, gsd, tile, mode="rgb"):
    """Fast tile renderer: the product's per-cell colour, sampled onto the tile grid.

    Texture warping costs seconds per tile, so the streaming loop uses the (much faster)
    cell-colour mosaic; the texture ortho stays an offline refinement.
    """
    state = {"frames": -1, "z": None}

    def render(grid: OrthoGrid, i: int, j: int):
        t0 = time.perf_counter()
        x0, y0, x1, y1 = grid.tile_window(i, j)
        n = int(round((x1 - x0) / gsd))
        m = int(round((y1 - y0) / gsd))
        xs = x0 + (np.arange(n) + 0.5) * gsd
        # row 0 is the tile's TOP: y grows upwards in the plane and downwards in the image
        ys = y1 - (np.arange(m) + 0.5) * gsd
        ci = np.floor(xs / product.cell).astype(np.int64) - product.ox
        cj = np.floor(ys / product.cell).astype(np.int64) - product.oy
        # a window entirely outside the product is an empty tile, never a stretched border
        if (ci.max() < 0) or (ci.min() >= product.W) or (cj.max() < 0) or (cj.min() >= product.H):
            z = np.zeros((tile, tile, 3), np.float32)
            return z, np.zeros((tile, tile), np.float32), np.zeros((tile, tile), np.float32), {}
        ci = np.clip(ci, 0, product.W - 1)
        cj = np.clip(cj, 0, product.H - 1)
        # cj/ci are contiguous pixel ranges: slice the product window and then gather at
        # pixel resolution (using the cell array directly would give a 341x341 image, not 512)
        # cj decreases with the row index (row 0 is the top), so bounds come from min/max
        r0, r1 = int(cj.min()), int(cj.max()) + 1
        c0, c1 = int(ci.min()), int(ci.max()) + 1
        wz, wrgb = product.top_zr_window(r0, r1, c0, c1)
        rr = np.clip(cj - r0, 0, max(wz.shape[0] - 1, 0))
        cc = np.clip(ci - c0, 0, max(wz.shape[1] - 1, 0))
        zt = wz[rr][:, cc]
        col_rgb = wrgb[rr][:, cc]
        alpha = np.isfinite(zt).astype(np.float32)
        rgb = np.zeros((m, n, 3), np.float32)
        if mode == "rgb" and col_rgb is not None:
            rgb = np.clip(col_rgb * (1.0 / 255.0), 0.0, 1.0).astype(np.float32)
        else:                                    # "dsm": keep the old height shading
            zz = np.where(np.isfinite(zt), zt,
                          np.nanmedian(zt[np.isfinite(zt)]) if np.isfinite(zt).any() else 0.0)
            gx = np.gradient(zz, axis=1)
            gy = np.gradient(zz, axis=0)
            shade = np.clip(0.6 + 2.0 * (gy * 0.7 - gx * 0.7), 0, 1).astype(np.float32)
            for c in range(3):
                rgb[..., c] = shade
        depth = np.where(np.isfinite(zt), zt, 0.0).astype(np.float32)
        # edge tiles have fewer pixels than a full tile; pad to (tile, tile) with alpha=0 so
        # the mosaic stitcher can assume a uniform tile size
        if rgb.shape[0] != tile or rgb.shape[1] != tile:
            pad_rgb = np.zeros((tile, tile, 3), np.float32)
            pad_a = np.zeros((tile, tile), np.float32)
            pad_d = np.zeros((tile, tile), np.float32)
            h, w = min(rgb.shape[0], tile), min(rgb.shape[1], tile)
            pad_rgb[:h, :w] = rgb[:h, :w]
            pad_a[:h, :w] = alpha[:h, :w]
            pad_d[:h, :w] = depth[:h, :w]
            rgb, alpha, depth = pad_rgb, pad_a, pad_d
        return rgb, alpha, depth, {"ms": (time.perf_counter() - t0) * 1000}
    return render


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--cell", type=float, default=0.45)
    ap.add_argument("--gsd", type=float, default=0.30)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--budget-ms", type=float, default=500.0)
    ap.add_argument("--max-tiles", type=int, default=6)
    ap.add_argument("--checkpoints", default="15,30,55")
    ap.add_argument("--oracle", default="")
    ap.add_argument("--window", type=int, default=0, help="0 disables windowed recalibration")
    ap.add_argument("--every", type=int, default=5, help="trigger a window pass every B frames")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tiles-window", type=int, default=10)
    ap.add_argument("--no-calib", action="store_true",
                    help="baseline: insert the raw heights (no 3-DOF correction at all)")
    a = ap.parse_args(argv)

    t_start = time.time()
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)[: int(max(int(v) for v in a.checkpoints.split(",")))]
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    H = int(np.ceil((bbox[3] - bbox[1]) / a.cell)) + 1
    W = int(np.ceil((bbox[2] - bbox[0]) / a.cell)) + 1
    ox, oy = int(np.floor(bbox[0] / a.cell)), int(np.floor(bbox[1] / a.cell))
    # use the oracle's grid when available: otherwise the metrics are applied to shifted
    # cells and RA/SEI become meaningless
    if a.oracle and pathlib.Path(a.oracle).exists():
        _o = np.load(a.oracle)
        ox, oy = int(_o["ox"]), int(_o["oy"])
        H, W = int(_o["z_ref"].shape[0]), int(_o["z_ref"].shape[1])
    product = StreamingProduct(a.cell, ox, oy, H, W, L=3, z_gap=2.0)
    calib = OnlineCalib(OnlineCalibConfig(min_frames=10))
    T_WP = np.load(pathlib.Path(a.obs) / "obs_merged.npz")["T_WP"]
    grid = OrthoGrid(T_WP=np.asarray(T_WP, np.float64), gsd=a.gsd, tile=a.tile)
    cache = TileCache()
    engine = OrthoEngine(grid, cache, tile_render_fn_factory(product, a.gsd, a.tile),
                         SchedulerConfig(tile=a.tile, budget_ms=a.budget_ms,
                                         max_tiles_per_flush=a.max_tiles))
    ckpts = {int(v) for v in a.checkpoints.split(",")}

    out = pathlib.Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    win_state = {}          # kf -> {"slots", "w", "z"} for frames still inside the window
    win_log = []

    def window_pass(now_n: int):
        """Batch fixed point over the sliding window (not per-frame leave-one-out).

        1. remove **all** window contributions from the product;
        2. ``rounds`` rounds of: fit every window frame against the product made only of the
           non-window frames, apply, then write the window back in;
        3. re-render the tiles the window touched.

        Owners and slots are frozen throughout -- only the heights move.
        """
        t0 = time.perf_counter()
        keys = sorted(win_state)
        if not keys:
            return 0.0
        rows = {k: S._rows(d, slices[k]) for k in keys}
        for k in keys:
            st = win_state[k]
            product.detach_rows(st["slots"], st["w"], st["z"])
        for rnd in range(max(1, a.rounds)):
            for k in keys:
                st = win_state[k]
                r = rows[k]
                ok = st["ok"]
                z_cur = st["z"]
                cix = np.floor(r["x"][ok] / a.cell).astype(np.int64)
                ciy = np.floor(r["y"][ok] / a.cell).astype(np.int64)
                z_near = product.near(cix, ciy, z_cur)      # no window contribution inside
                th = fit_t3(z_cur, z_near, r["x"][ok], r["y"][ok], st["w"], calib.calib, origin)
                if th is None:
                    continue
                st["z"] = apply_t3(z_cur, r["x"][ok], r["y"][ok], th, origin).astype(np.float32)
            for k in keys:
                st = win_state[k]
                product.attach_rows(st["slots"], st["w"], st["z"])
            if rnd < max(1, a.rounds) - 1:
                for k in keys:
                    st = win_state[k]
                    product.detach_rows(st["slots"], st["w"], st["z"])
        dt = (time.perf_counter() - t0) * 1000
        for k in keys:
            r = rows[k]
            fb = (float(np.min(r["x"])), float(np.min(r["y"])),
                  float(np.max(r["x"])), float(np.max(r["y"])))
            engine.mark_footprint(fb, kf_id=int(k), reason="recalib")
        fl = engine.flush(budget_ms=a.budget_ms, max_tiles=a.max_tiles_window)
        win_log.append({"n": now_n, "frames": len(keys), "rounds": a.rounds, "ms": dt,
                        "rendered": fl["rendered"], "dirty_left": fl["dirty_left"]})
        return dt

    log = []
    checkpoints = {}
    for n, k in enumerate(kfs, start=1):
        t0 = time.perf_counter()
        r = S._rows(d, slices[k])
        if a.no_calib:
            z_cor, info = r["z"], {"kf": int(k), "mode": "disabled", "theta": None,
                                   "resid": float("nan")}
        else:
            z_cor, info = calib.process(product, n - 1, k, r["x"], r["y"],
                                        r["z"], r["conf"] * r["cos_inc"])
        t_calib = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cix = np.floor(r["x"] / a.cell).astype(np.int64)
        ciy = np.floor(r["y"] / a.cell).astype(np.int64)
        ins = product.insert(k, cix, ciy, z_cor, r["conf"] * r["cos_inc"])
        t_insert = (time.perf_counter() - t0) * 1000

        if a.window > 0:
            okm = ins["ok"]
            win_state[int(k)] = {"slots": ins["slots"][okm],
                                 "w": (r["conf"] * r["cos_inc"])[okm].astype(np.float32),
                                 "z": z_cor[okm].astype(np.float32),
                                 "ok": okm}
            if len(win_state) > a.window:
                for kk in sorted(win_state)[:len(win_state) - a.window]:
                    win_state.pop(kk, None)
            if n % a.every == 0:
                dt_w = window_pass(n)
                if dt_w:
                    log[-1]["window_ms"] = dt_w

        t0 = time.perf_counter()
        fb = (float(np.min(r["x"])), float(np.min(r["y"])),
              float(np.max(r["x"])), float(np.max(r["y"])))
        mark = engine.mark_footprint(fb, kf_id=int(k))
        fl = engine.flush(budget_ms=a.budget_ms, max_tiles=a.max_tiles)
        t_tile = (time.perf_counter() - t0) * 1000

        log.append({"n": n, "kf": int(k), "calib_ms": t_calib, "insert_ms": t_insert,
                    "tile_ms": t_tile, "inserted": ins["inserted"],
                    "new_layers": ins["new_layers"], "mode": info["mode"],
                    "theta_b": (info["theta"] or [float("nan")])[0],
                    "resid": info["resid"], "dirty": mark["newly_dirty"],
                    "rendered": fl["rendered"], "dirty_left": fl["dirty_left"]})

        if n in ckpts:
            z_top, own = product.top()
            live = np.isfinite(z_top)
            checkpoints[n] = {"n": n, "live_cells": int(live.sum()),
                              "coverage_of_bbox": float(live.mean()),
                              "rendered_tiles": len(engine.rendered),
                              "covered_tiles": len(engine.covered),
                              "dirty_left": fl["dirty_left"],
                              "product_mb": product.memory_mb()}

    engine.flush_all()
    tot = np.array([e["calib_ms"] + e["insert_ms"] + e["tile_ms"] for e in log])
    res = {"seq": a.seq, "gsd": a.gsd, "tile": a.tile, "budget_ms": a.budget_ms,
           "n_frames": len(kfs), "frames": log, "checkpoints": {str(k): v for k, v in checkpoints.items()},
           "latency": {"p50": float(np.percentile(tot, 50)), "p95": float(np.percentile(tot, 95)),
                       "mean": float(tot.mean()),
                       "calib_p50": float(np.percentile([e["calib_ms"] for e in log], 50)),
                       "insert_p50": float(np.percentile([e["insert_ms"] for e in log], 50)),
                       "tile_p50": float(np.percentile([e["tile_ms"] for e in log], 50))},
           "calib": {"fit": calib.stats.n_fit, "bootstrap": calib.stats.n_bootstrap,
                     "fallback_low_obs": calib.stats.n_fallback_low_obs,
                     "fallback_bad": calib.stats.n_fallback_bad,
                     "fallback_frac": calib.stats.fallback_frac},
           "window": {"W": a.window, "every": a.every, "rounds": a.rounds, "log": win_log},
           "product_mb": product.memory_mb(),
           "tiles": {"rendered": len(engine.rendered), "covered": len(engine.covered)},
           "wall_s": time.time() - t_start,
           "git_hash": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "command": "python -m stream_ortho.scripts.run_stream " + " ".join(argv or [])}

    # latency slope against N (a streaming claim needs it flat)
    ns = np.array([e["n"] for e in log], float)
    ms = np.array([e["calib_ms"] + e["insert_ms"] + e["tile_ms"] for e in log])
    res["latency"]["slope_ms_per_frame"] = float(np.polyfit(ns, ms, 1)[0])

    # quality: score the streaming product with the same metrics as the offline one
    try:
        from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
        from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics
        sp = product.as_product()
        # masks come from the offline raw product (same bbox) so LCR is comparable
        op0 = np.load(a.oracle) if (a.oracle and pathlib.Path(a.oracle).exists()) else None
        if op0 is not None:
            raw_n = op0["n"].astype(np.float32)
            raw_z = op0["z_ref"].astype(np.float32)
            Hc = min(sp["H"], raw_n.shape[0]); Wc = min(sp["W"], raw_n.shape[1])
            sp_c = dict(sp)
            for key in ("n", "z_ref", "owner", "sum_w", "sum_wz", "sum_wz2"):
                sp_c[key] = sp[key][:Hc, :Wc]
            sp_c["H"], sp_c["W"] = Hc, Wc
            raw_c = {"n": raw_n[:Hc, :Wc], "z_ref": raw_z[:Hc, :Wc],
                     "owner": op0["owner"][:Hc, :Wc].astype(np.int32),
                     "cell": float(op0["cell"]), "H": Hc, "W": Wc, "L": raw_n.shape[2],
                     "ox": int(op0["ox"]), "oy": int(op0["oy"])}
            masks = shared2_pairs(raw_c)
            z_top_raw, _ = top_surface(raw_c)
            lz = raw_z[raw_n > 0]
            iqr = float(np.percentile(lz, 75) - np.percentile(lz, 25)) if lz.size else 1.0
            m = surf_metrics(sp_c, masks, z_top_raw, iqr)
            res["quality_stream"] = {k: float(m[k]) for k in
                                     ("lcr_fixed", "n_jump", "ra", "sei", "coverage_cells",
                                      "z_iqr_ratio")}
    except Exception as exc:      # never let the quality probe kill the run
        res["quality_stream_error"] = repr(exc)

    # quality: compare the streaming product with the offline oracle at the last checkpoint
    if a.oracle and pathlib.Path(a.oracle).exists():
        op = np.load(a.oracle)
        z_off = op["z_ref"].astype(np.float32)
        n_off = op["n"].astype(np.float32)
        z_s, _ = product.top()
        # align the two grids by their cell origins before comparing
        ox_o, oy_o = int(op["ox"]), int(op["oy"])
        di, dj = oy_o - product.oy, ox_o - product.ox
        H = min(z_s.shape[0], z_off.shape[0]); W = min(z_s.shape[1], z_off.shape[1])
        i0s, i1s = max(0, di), min(z_s.shape[0], H + di)
        j0s, j1s = max(0, dj), min(z_s.shape[1], W + dj)
        zs = z_s[i0s:i1s, j0s:j1s]
        zt_full = np.where(n_off > 0, z_off, np.nan).max(-1)
        zt = zt_full[i0s - di:i1s - di, j0s - dj:j1s - dj]
        m = np.isfinite(zs) & np.isfinite(zt)
        res["quality_vs_oracle"] = {
            "n_cells": int(m.sum()),
            "median_abs_dz": float(np.median(np.abs(zs[m] - zt[m]))) if m.any() else None,
            "p90_abs_dz": float(np.percentile(np.abs(zs[m] - zt[m]), 90)) if m.any() else None,
            "corr": float(np.corrcoef(zs[m], zt[m])[0, 1]) if m.sum() > 10 else None}

    (out / "stream.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("latency", "calib", "product_mb", "tiles")},
                     indent=1, ensure_ascii=False))
    print("checkpoints:", json.dumps(res["checkpoints"], ensure_ascii=False)[:400])
    if "quality_vs_oracle" in res:
        print("quality vs oracle:", res["quality_vs_oracle"])
    print("->", out / "stream.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
