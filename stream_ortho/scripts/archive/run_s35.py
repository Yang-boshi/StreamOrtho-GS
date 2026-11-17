"""S35 -- S1 (cross-view depth gating) -> S2 (per-frame depth calibration).

Three policies on the same keyframes, all with the 2.5D cell selection at the end:

* ``baseline``  : no gating, no calibration,
* ``s1``        : drop pixels that disagree with neighbouring keyframes' depth,
* ``s1s2``      : S1 + a robust per-frame ``z' = a*z + b`` fitted to the committed
  surface (rejected when implausible).

The headline metric is the **drift** of an already-stable surface: snapshot the per-cell
top surface after the first 5 keyframes, add 4 more, and measure how much the *cells
that were already there* moved.  This is exactly the user-visible symptom (a well-formed
building pulled sideways by a later frame), and it complements coverage / layer / PSNR.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8
from stream_ortho.map.build import head_adapt_patch
from stream_ortho.map.consistency import (cell_surface, cross_view_mask, drift_metric,
                                          fit_depth_calibration)
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.selection import layer_metric, quality_score, select_by_cell
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, stitch


def make_cloud(M, Q, SC, OP, CO, device="cuda:0"):
    import torch

    d = torch.device(device)
    return GaussianCloud(means=torch.as_tensor(np.ascontiguousarray(M), device=d),
                         quats=torch.as_tensor(np.ascontiguousarray(Q), device=d),
                         scales=torch.as_tensor(np.ascontiguousarray(SC), device=d),
                         opacity=torch.as_tensor(np.ascontiguousarray(OP), device=d),
                         colors=torch.as_tensor(np.ascontiguousarray(CO), device=d),
                         owner=torch.zeros(len(M), dtype=torch.long, device=d), meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--sigma", type=float, default=0.45)
    p.add_argument("--cell", type=float, default=0.45)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=2.0, help="S1 depth tolerance (m)")
    p.add_argument("--max-kf", type=int, default=9)
    p.add_argument("--stable-after", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()[: a.max_kf]
    intr = (ds.K_orig, ds.cfg.height, ds.cfg.width)

    def sim3_fn(k):
        return sim3_from_vec8(compose_sim3(T[k], sg))

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S35" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in ("baseline", "s1", "s1s2"):
        gm = GaussianMap(MapConfig(voxel_size=0.25, dedup=True, gate=False,
                                   commit_min_obs=2, commit_max_age=2, fusion="weighted"),
                         ds.T_WP)
        stats = {"dropped": 0, "pixels": 0, "calib": []}
        snap_before = None
        t0 = time.time()
        for n, kf in enumerate(kfs):
            patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                     depth_source="pointmap", sigma_m=a.sigma,
                                     conf_min=a.conf_min, device=zcfg.DEVICE)
            if policy in ("s1", "s1s2") and n > 0:
                prev = kfs[max(0, n - 2):n]
                keep, gt = cross_view_mask(npz, intr, sim3_fn, kf, prev, tau=a.tau)
                stats["dropped"] += int(gt["frac_dropped"] * keep.size * a.stride * a.stride)
                stats["pixels"] += keep.size
                # the patch lives on the **full-resolution strided** grid, while the
                # mask is on the head grid: upsample the mask first (an earlier version
                # strided the head mask directly, so the length never matched and the
                # gating silently did nothing -- the A/B came out identical)
                import cv2 as _cv2
                keep_full = _cv2.resize(keep.astype(np.uint8), (ds.cfg.width, ds.cfg.height),
                                        interpolation=_cv2.INTER_NEAREST)[::a.stride, ::a.stride]
                kd = keep_full.reshape(-1).astype(bool)
                assert len(kd) == len(patch.means), (len(kd), len(patch.means))
                patch = patch.subset(kd)
            if policy == "s1s2" and n > 0 and gm.n_committed > 10000:
                r = fit_depth_calibration(patch.means, gm.committed["means"], ds.T_WP,
                                          cell=0.5)
                stats["calib"].append({k: r[k] for k in ("ok", "a", "b", "n", "resid")})
                if r["ok"]:
                    P = (patch.means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
                    P[:, 2] = r["a"] * P[:, 2] + r["b"]
                    patch.means = (P @ ds.T_WP[:3, :3] + ds.T_WP[:3, 3]).astype(np.float32)
            gm.ingest(patch, kf)
            gm.commit(kf)
            if n + 1 == a.stable_after:
                live = gm.committed["dead"] == 0
                qq = quality_score(conf=gm.committed["conf"][live],
                                   opacity=gm.committed["opacity"][live],
                                   wsum=gm.committed["wsum"][live],
                                   n_obs=gm.committed["n_obs"][live])
                keys, zs, _ = cell_surface(gm.committed["means"][live], ds.T_WP, cell=0.5,
                                           q=qq, mode="quality")
                snap_before = {int(k): float(z) for k, z in zip(keys, zs)}
        live = gm.committed["dead"] == 0
        M = gm.committed["means"][live]
        q = quality_score(conf=gm.committed["conf"][live], opacity=gm.committed["opacity"][live],
                          wsum=gm.committed["wsum"][live], n_obs=gm.committed["n_obs"][live])
        idx = np.where(live)[0]
        sel = idx[select_by_cell(M, ds.T_WP, cell=a.cell, q=q, prefer="quality")]
        c = gm.committed
        cloud = make_cloud(c["means"][sel], c["quats"][sel], c["scales"][sel],
                           np.clip(c["opacity"][sel], 0.95, 1.0), c["colors"][sel])
        Keys, Zs, _ = cell_surface(c["means"][sel], ds.T_WP, cell=0.5, mode="quality")
        snap_after = {int(k): float(z) for k, z in zip(Keys, Zs)}
        drift = drift_metric(snap_before or {}, snap_after) if snap_before else {"n": 0}

        P = (c["means"][sel] - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        i0, j0 = grid.tile_of(float(P[:, 0].min()), float(P[:, 1].min()))
        i1, j1 = grid.tile_of(float(P[:, 0].max()), float(P[:, 1].max()))
        i0, i1, j0, j1 = min(i0, i1), max(i0, i1), min(j0, j1), max(j0, j1)
        tiles = []
        for jj in range(j0, j1 + 1):
            for ii in range(i0, i1 + 1):
                rgb, alpha, _, _ = render_tile(cloud, grid, ii, jj)
                tiles.append(Tile(ii, jj, rgb, alpha))
        mosaic, cov = stitch(tiles, shape=(j1 - j0 + 1, i1 - i0 + 1))
        ref = PlaneReferenceMosaic(ds, a.gsd, 0.0, stride=1)
        ps = []
        for jj in range(j0, min(j0 + 3, j1 + 1)):
            for ii in range(i0, min(i0 + 3, i1 + 1)):
                rgb, al, _, _ = render_tile(cloud, grid, ii, jj)
                tgt, cv_, _, _ = ref.render_tile(grid, ii, jj, max_frames=1)
                mk = (al > 0.5) & (cv_ > 0.5)
                if mk.sum() > 5000:
                    ps.append(psnr(rgb[mk], tgt[mk]))
        row = {"policy": policy, "n": int(len(sel)),
               "coverage": float((cov > 0.5).mean()), "layer": layer_metric(c["means"][sel], ds.T_WP),
               "psnr": float(np.mean(ps)) if ps else None,
               "dropped_px": int(stats["dropped"]),
               "dropped_frac": float(stats["dropped"] / max(stats["pixels"], 1)) if stats["pixels"] else 0.0,
               "calib_ok": sum(1 for r in stats["calib"] if r["ok"]),
               "calib_tries": len(stats["calib"]),
               "drift": drift, "s": round(time.time() - t0, 1)}
        rows.append(row)
        import cv2
        cv2.imwrite(str(out / f"{policy}.png"),
                    (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        d = drift
        print(f"[s35] {policy:8s} N={row['n']:>8,} cov={100*row['coverage']:5.1f}% "
              f"layer={row['layer']:.2f} psnr={row['psnr']:.2f} "
              f"dropped={100*row['dropped_frac']:4.1f}% calib={row['calib_ok']}/{row['calib_tries']} "
              f"drift med={d.get('median_dz', float('nan')):.2f} m "
              f"p95={d.get('p95_dz', float('nan')):.2f} moved={100*d.get('frac_moved', 0):4.1f}% "
              f"({row['s']}s)")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
