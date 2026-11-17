"""Build the layered observation table ``(kf, cell, layer)`` -- one row per layer.

Why this table exists (``refs/06-任务卡`` §P0-2): the 55-frame flight sees **4.28 keyframes per
0.45 m cell** and **20.1 points per cell**; the "1.42 obs/column, 65% single observation"
figure came from a 10-frame subset *after* ``collapse_cells(keep_n=1)``.  Every later
experiment (E1/E2) reads this table, never the collapsed product.

Per ``(kf, cell)``: sort by ``z``, cut a new layer whenever the gap exceeds ``z_gap`` (2 m),
keep at most ``max_layers`` (3, the highest ones) and emit **one row per layer**:

    z = median, x/y = mean of the layer's points (NOT the cell centre -- E1 compares z_i - z_j
    on the same (x, y)), z_std = std, n = count, conf = max, rgb = colour of the max-conf point

Fields (strict schema, ``03-P0-2`` §3.1)::

    kf, frame_id, key, ix, iy, x, y, layer, z, z_std, n, rgb, conf, cos_inc, D, uv, owner
    (+ T_WP, cell stored once per file)

``D`` (ray length) and ``uv`` (normalised pixel coordinates) are **computed for real** by
reprojecting each world point into its keyframe (``X = ((Pw - t) @ R) / s``): the patch grid is
the full-resolution image, so the projection lands exactly on the original pixel grid (verified:
u in [0, W-1], v in [0, H-1] for every point of a frame).

Usage::

    python -m stream_ortho.scripts.diag.build_obs_min \
        --run results/stage-S2/ieu_calib --seq ieu --cell 0.45 --z-gap 2.0 \
        --stride 1 --conf-min 0.2 --sources pointmap --out result-new/P0/obs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

SCHEMA_INT = {"kf": np.int32, "frame_id": np.int32, "key": np.int64, "ix": np.int32,
              "iy": np.int32, "layer": np.int16, "n": np.int16, "owner": np.int32}
SCHEMA_F32 = {"x": np.float32, "y": np.float32, "z": np.float32, "z_std": np.float32,
              "D": np.float32}
SCHEMA_F16 = {"conf": np.float16, "cos_inc": np.float16}


# --------------------------------------------------------------------------------------
def keyframe_indices(npz) -> np.ndarray:
    """The exact keyframe rule of ``scripts/run_incremental_video.py`` (+-30% Sim3 scale)."""
    sc = np.asarray(npz["T_WC"][:, 7], np.float64)
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    return kfs[np.argsort(npz["frame_id"][kfs])]


def camera_arrays(npz, sim3_vec, k, T_WP):
    """(R, t, s) of the keyframe and the plane-frame camera centre."""
    from stream_ortho.geometry import compose_sim3, sim3_from_vec8

    R, t, s = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec))
    t = np.asarray(t, np.float64)
    cam_P = (t - T_WP[:3, 3]) @ T_WP[:3, :3]          # world -> plane for the camera centre
    return np.asarray(R), t, float(s), cam_P


def project(Pw, R, t, s, K):
    """Camera-frame points, metric ray length ``D`` and normalised pixel ``uv``."""
    X = ((Pw - t) @ R) / s                            # world -> camera (canonical units)
    D = s * np.linalg.norm(X, axis=1)                 # metric distance to the camera
    zc = X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * X[:, 0] / zc + K[0, 2]
        v = K[1, 1] * X[:, 1] / zc + K[1, 2]
    return X, D, u, v, zc


def layer_table(P, key, ix, iy, z, rgb01, conf, cos_inc, D, uv, kf, frame_id, owner,
                z_gap=2.0, max_layers=3):
    """Group one frame's points into ``(key, layer)`` rows."""
    order = np.lexsort((z, key))
    ks, zs = key[order], z[order]
    if ks.size == 0:
        return _empty_table()
    new = np.r_[True, (ks[1:] != ks[:-1]) | (np.diff(zs) > z_gap)]
    gid = np.cumsum(new) - 1
    start = np.r_[0, np.nonzero(np.diff(gid))[0] + 1]
    cnt = np.diff(np.r_[start, ks.size])
    ng = start.size

    z_ord = zs
    med = 0.5 * (z_ord[start + (cnt - 1) // 2] + z_ord[start + cnt // 2])
    z_std = np.sqrt(np.maximum(
        np.add.reduceat(z_ord * z_ord, start) / cnt - (np.add.reduceat(z_ord, start) / cnt) ** 2, 0.0))
    P_ord = P[order]
    x_m = np.add.reduceat(P_ord[:, 0], start) / cnt
    y_m = np.add.reduceat(P_ord[:, 1], start) / cnt
    cos_m = np.add.reduceat(cos_inc[order], start) / cnt
    D_m = np.add.reduceat(D[order], start) / cnt

    # representative colour = colour of the highest-conf point of the group
    o2 = np.lexsort((conf[order], gid))
    g2 = gid[o2]
    s2 = np.r_[0, np.nonzero(np.diff(g2))[0] + 1]
    c2 = np.diff(np.r_[s2, g2.size])
    # o2 is sorted by gid first, so the per-group maxima already come out in group order
    rep = o2[s2 + c2 - 1]
    rgb_rep = rgb01[order][rep]
    conf_max = conf[order][rep]
    uv_m = np.stack([
        np.add.reduceat(uv[order, 0], start) / cnt,
        np.add.reduceat(uv[order, 1], start) / cnt], 1)

    # keep at most max_layers per cell, highest z first; layer 0 = highest
    gkey = ks[start]
    key_change = np.r_[True, gkey[1:] != gkey[:-1]]
    key_id = np.cumsum(key_change) - 1
    n_per_key = np.bincount(key_id)
    first_of_key = np.nonzero(key_change)[0]          # key_change[0] is always True
    idx_in_key = np.arange(ng) - first_of_key[key_id]
    keep = (n_per_key[key_id] - 1 - idx_in_key) < max_layers
    layer_no = (n_per_key[key_id] - 1 - idx_in_key).astype(np.int16)

    gk = gkey[keep]
    tbl = {
        "kf": np.full(gk.size, kf, np.int32),
        "frame_id": np.full(gk.size, frame_id, np.int32),
        "key": gk.astype(np.int64),
        "ix": ix[order][start][keep].astype(np.int32),
        "iy": iy[order][start][keep].astype(np.int32),
        "x": x_m[keep].astype(np.float32),
        "y": y_m[keep].astype(np.float32),
        "layer": layer_no[keep],
        "z": med[keep].astype(np.float32),
        "z_std": z_std[keep].astype(np.float32),
        "n": cnt[keep].astype(np.int16),
        "rgb": np.clip(rgb_rep[keep] * 255.0 + 0.5, 0, 255).astype(np.uint8),
        "conf": conf_max[keep].astype(np.float16),
        "cos_inc": cos_m[keep].astype(np.float16),
        "D": D_m[keep].astype(np.float32),
        "uv": uv_m[keep].astype(np.float32),
        "owner": np.full(gk.size, owner, np.int32),
    }
    # emit rows in (key, layer) order so layer 0 is always the highest surface
    ordv = np.lexsort((tbl["layer"], tbl["key"]))
    tbl = {k: v[ordv] for k, v in tbl.items()}
    return tbl, int(ng - gk.size)


def _empty_table():
    return ({k: np.zeros(0, d) for k, d in {**SCHEMA_INT, **SCHEMA_F32, **SCHEMA_F16}.items()}
            | {"uv": np.zeros((0, 2), np.float32), "rgb": np.zeros((0, 3), np.uint8)}, 0)


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--cell", type=float, default=0.45)
    ap.add_argument("--extra-cells", default="0.30")     # stats only, recorded not gated
    ap.add_argument("--z-gap", type=float, default=2.0)
    ap.add_argument("--max-layers", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--depth-upsample", choices=("linear", "nearest", "edge"), default="nearest",
                    help="bilinear invents in-between depths at roof/ground steps; nearest keeps "
                         "the full-resolution density without new depth values")
    ap.add_argument("--edge-tau-m", type=float, default=0.75,
                    help="edge mode: 2x2 depth range above which nearest replaces bilinear")
    ap.add_argument("--geometry-source", choices=("raw", "adapter"), default="raw",
                    help="raw: the pointmap's own grid (upstream); adapter: the legacy "
                         "bilinear depth upsample")
    ap.add_argument("--conf-stat", choices=("avg", "max"), default="avg",
                    help="avg = C/N (upstream's get_average_conf), max = legacy raw C")
    ap.add_argument("--conf-min", type=float, default=1.0)
    ap.add_argument("--sigma-m", type=float, default=0.20)
    ap.add_argument("--sources", default="pointmap")
    ap.add_argument("--out", default="result-new/P0/obs")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-per-frame", action="store_true",
                    help="skip the per-frame npz (sensitivity runs); still write the merged table")
    ap.add_argument("--stats-only", action="store_true",
                    help="do not write per-frame npz; only the cell statistics (for 0.30)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    from stream_ortho import config as zcfg
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset
    from stream_ortho.eval.obs_stats import frames_per_cell, points_per_cell
    from stream_ortho.frontend.gaussian_head import load_gaussian_head
    from stream_ortho.frontend.predict import load_model
    from stream_ortho.geometry import R_to_quat_xyzw
    from stream_ortho.map.build import head_adapt_patch
    from stream_ortho.map.selection import pack_cell

    out_root = pathlib.Path(a.out)
    rd = pathlib.Path(a.run)
    info = json.loads((rd / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(rd / "keyframes.npz")
    m = json.loads((rd / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    kfs = keyframe_indices(npz)
    if a.max_frames:
        kfs = kfs[:a.max_frames]
    extra = [float(v) for v in str(a.extra_cells).split(",") if v.strip()]

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)

    t_start = time.time()
    for source in [s.strip() for s in a.sources.split(",") if s.strip()]:
        out_dir = out_root / a.seq / source
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, timings, n_overflow = [], [], 0
        conf_stats = {"removed": 0, "total": 0, "p50": []}
        extra_keys = {c: [] for c in extra}

        for n, k in enumerate(kfs):
            t0 = time.time()
            # conf_min is applied as a mask at model resolution: upstream's gate is on C / N
            # (pointmap), while the patch's own conf comes from the Gaussian head
            n_upd0 = float(npz["N"][int(k)]) if "N" in npz.files else 1.0
            if a.geometry_source == "raw":
                # raw pointmap path: the points are the model's own 288x512 grid, exactly as
                # upstream's save_reconstruction (no cv2.resize of the depth, which would
                # interpolate fake points between a roof and the ground at every edge)
                import cv2 as _cv2
                from stream_ortho.geometry import compose_sim3 as _compose
                from stream_ortho.geometry import sim3_from_vec8 as _s8
                Xc = np.asarray(npz["X_canon"][int(k)], np.float32)
                h, w = Xc.shape[:2]
                Cc = np.asarray(npz["C"][int(k)], np.float32).reshape(h, w)
                conf_avg = Cc.reshape(-1) / max(n_upd0, 1.0)
                _R, _t, _sc = _s8(_compose(npz["T_WC"][int(k)], sg))
                Pw = float(_sc) * (Xc.reshape(-1, 3).astype(np.float64) @ _R.T) + _t
                rgb_full = ds.read_img(int(npz["frame_id"][int(k)]))
                rgb01 = (_cv2.resize(rgb_full, (w, h), interpolation=_cv2.INTER_AREA)
                         .reshape(-1, 3).astype(np.float64) / 255.0)
                conf_raw = conf_avg
                sel_raw = np.isfinite(conf_avg) & np.isfinite(Pw).all(1) & (conf_avg >= a.conf_min)
                step = max(int(a.stride), 1)
                sel_raw &= (np.arange(sel_raw.size) % step) == 0 if step > 1 else True
                Pw, conf_raw, rgb01 = Pw[sel_raw], conf_raw[sel_raw], rgb01[sel_raw]
            else:
                cmask = None
                if a.conf_stat == "avg":
                    _c = np.asarray(npz["C"][int(k)], np.float32).reshape(
                        np.asarray(npz["X_canon"][int(k)]).shape[:2])
                    cmask = (_c / max(n_upd0, 1.0)) >= a.conf_min
                patch = head_adapt_patch(ds, npz, int(k), sg, model, head, stride=a.stride,
                                         depth_source=source, sigma_m=a.sigma_m,
                                         conf_min=0.0, device=zcfg.DEVICE, conf_mask=cmask,
                                         depth_upsample=a.depth_upsample,
                                         edge_tau_m=a.edge_tau_m)
                Pw = np.asarray(patch.means, np.float64)
                conf_raw = np.asarray(patch.conf, np.float64)
                rgb01 = np.asarray(patch.colors, np.float64)
            # upstream save_reconstruction: valid = get_average_conf() > threshold, i.e. C / N
            n_upd = float(npz["N"][int(k)]) if "N" in npz.files else 1.0
            if "N" not in npz.files and n == 0:
                print("[obs] warning: keyframes.npz has no N; conf_avg falls back to C")
            conf = conf_raw                     # raw path already carries C/N
            sel = np.isfinite(conf) & np.isfinite(Pw).all(1)
            conf_removed = int((~sel).sum())
            conf_p50 = float(np.median(conf[sel])) if sel.any() else 0.0
            Pw, conf, rgb01 = Pw[sel], conf[sel], rgb01[sel]
            R, t, s, cam_P = camera_arrays(npz, sg, int(k), ds.T_WP)

            X, D, u, v, _ = project(Pw, R, t, s, ds.K_orig)
            H, W = ds.read_img(int(npz["frame_id"][k])).shape[:2]
            uv = np.stack([2.0 * u / max(W - 1, 1) - 1.0, 2.0 * v / max(H - 1, 1) - 1.0], 1)

            P = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]          # world -> plane
            ix = np.floor(P[:, 0] / a.cell).astype(np.int64)
            iy = np.floor(P[:, 1] / a.cell).astype(np.int64)
            key = pack_cell(ix, iy)

            ray_P = (Pw - t) @ ds.T_WP[:3, :3]
            cos_inc = np.abs(ray_P[:, 2]) / np.maximum(np.linalg.norm(ray_P, axis=1), 1e-9)

            conf_stats["removed"] += conf_removed
            conf_stats["total"] += int(conf_removed + sel.sum())
            conf_stats["p50"].append(conf_p50)
            tbl, n_over = layer_table(P, key, ix, iy, P[:, 2], rgb01, conf, cos_inc, D, uv,
                                      int(k), int(npz["frame_id"][k]), int(k),
                                      z_gap=a.z_gap, max_layers=a.max_layers)
            n_overflow += n_over
            if not (a.stats_only or a.no_per_frame):
                np.savez(out_dir / f"kf_{int(k):04d}.npz", T_WP=ds.T_WP,
                         cell=np.float32(a.cell), **tbl)
            rows.append(tbl)

            # extra cell sizes: only distinct keys per frame are kept
            for c in extra:
                ixe = np.floor(P[:, 0] / c).astype(np.int64)
                iye = np.floor(P[:, 1] / c).astype(np.int64)
                uq, inv = np.unique(pack_cell(ixe, iye), return_inverse=True)
                extra_keys[c].append((uq, np.bincount(inv).astype(np.int32), int(k)))
            timings.append((int(k), time.time() - t0, int(Pw.shape[0])))

        merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        stats = {
            "seq": a.seq, "source": source, "cell": a.cell, "z_gap": a.z_gap,
            "max_layers": a.max_layers, "stride": a.stride, "conf_min": a.conf_min,
            "conf_stat": a.conf_stat, "geometry_source": a.geometry_source,
            "depth_upsample": a.depth_upsample,
            "conf_removed_frac": (conf_stats["removed"] / max(conf_stats["total"], 1)),
            "conf_avg_p50": (float(np.median(conf_stats["p50"])) if conf_stats["p50"] else 0.0),
            "sigma_m": a.sigma_m, "n_frames": len(kfs), "kfs": [int(k) for k in kfs],
            "dataset": info["dataset"], "run": str(rd),
            "n_obs_rows": int(merged["key"].size),
            "n_overflow_layers": int(n_overflow),
            "n_points_total": int(merged["n"].sum()),
            **frames_per_cell(merged["key"], merged["kf"]),
            **points_per_cell(merged["key"], merged["n"]),
            "z_std_p50": float(np.median(merged["z_std"])),
            "z_std_p90": float(np.percentile(merged["z_std"], 90)),
            "n_p50": float(np.median(merged["n"])),
            "cos_inc_mean": float(np.mean(merged["cos_inc"].astype(np.float64))),
            "cos_inc_frac_lt_0.5": float(np.mean(merged["cos_inc"] < 0.5)),
            "D_p50": float(np.median(merged["D"])),
            "layer_hist": {str(k): int(v) for k, v in
                           zip(*np.unique(merged["layer"], return_counts=True))},
            "extra_cells": {},
            "patch_ms_p50": float(np.median([t[1] for t in timings]) * 1000),
            "wall_s": float(time.time() - t_start),
            "git_head": subprocess.run(["git", "-C", "code/Ortho-3dgs", "rev-parse", "HEAD"],
                                       capture_output=True, text=True, cwd=pathlib.Path(__file__).resolve().parents[3]).stdout.strip(),
            "command": " ".join([sys.executable, "-m", "stream_ortho.scripts.diag.build_obs_min"] + (argv or sys.argv[1:])),
        }
        for c, pairs in extra_keys.items():
            kk = np.concatenate([p[0] for p in pairs])
            kf_arr = np.concatenate([np.full(p[0].size, p[2], np.int64) for p in pairs])
            cc = np.concatenate([p[1] for p in pairs]).astype(np.float64)
            st = frames_per_cell(kk, kf_arr)
            per_cell = points_per_cell(kk, cc)
            stats["extra_cells"][f"{c}"] = {
                "frames_per_cell": st["frames_per_cell"],
                "points_per_cell": per_cell["points_per_cell"],
                "n_cells": st["n_cells"], "n_points": per_cell["n_points"],
                "hist": st["hist"]}
            del kk, kf_arr, cc

        if not a.stats_only:
            np.savez(out_dir / "obs_merged.npz", T_WP=ds.T_WP, cell=np.float32(a.cell), **merged)
            (out_dir / "manifest.json").write_text(json.dumps(
                {k: stats[k] for k in ("seq", "source", "cell", "z_gap", "max_layers", "stride",
                                       "conf_min", "sigma_m", "n_frames", "dataset", "run",
                                       "command", "git_head", "wall_s")},
                indent=2, ensure_ascii=False))
            name = "obs_stats.json"
        else:
            name = f"obs_stats_cell{str(a.cell).replace('.', '')}{a.tag}.json"
        (out_dir / name).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"[{a.seq}/{source}] cell={a.cell} frames={len(kfs)} rows={stats['n_obs_rows']} "
              f"frames/cell={stats['frames_per_cell']:.3f} pts/cell={stats['points_per_cell']:.2f} "
              f"-> {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
