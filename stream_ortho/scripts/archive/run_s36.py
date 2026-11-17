"""S36 -- S3: admission control + freeze + rehearsal.

Three policies on the same keyframes (2.5D cell selection at the end):

* ``baseline`` : nothing (S35 baseline),
* ``s3``       : **admission control** (a new point floating > tau above the established
                 surface is rejected unless two independent frames propose the same
                 height) + **freeze** (a cell that kept its height between two
                 snapshots is locked, later frames can no longer replace it),
* ``s3r``      : ``s3`` + **rehearsal** (per-tile multi-view refinement that always
                 includes an *early* keyframe, so the optimiser cannot satisfy the new
                 frames by breaking the old geometry -- MonoGS's trick).

Reported together (a single metric can be fooled): floating-new-cell fraction, drift
p95 of already-stable cells, coverage, layer, PSNR, and the admission counters.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.backend.multiview import MultiViewConfig, refine_multiview
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch
from stream_ortho.map.consistency import (admission_mask, cell_surface, drift_metric,
                                          floating_new_cells, freeze_update)
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


def surface(gm, mode="quality"):
    live = gm.committed["dead"] == 0
    M = gm.committed["means"][live]
    q = quality_score(conf=gm.committed["conf"][live], opacity=gm.committed["opacity"][live],
                      wsum=gm.committed["wsum"][live], n_obs=gm.committed["n_obs"][live])
    k, z, _ = cell_surface(M, ds.T_WP, cell=0.5, q=q, mode=mode)
    return {int(a): float(b) for a, b in zip(k, z)}


ds = None


def main(argv=None):
    global ds
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--sigma", type=float, default=0.45)
    p.add_argument("--cell", type=float, default=0.45)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--tau-adm", type=float, default=1.5)
    p.add_argument("--adm-rad", type=int, default=1)
    p.add_argument("--collapse-every", type=int, default=3)
    p.add_argument("--max-kf", type=int, default=9)
    p.add_argument("--stable-after", type=int, default=5)
    p.add_argument("--policies", default="baseline,s3,s3r")
    p.add_argument("--mv-tiles", type=int, default=6)
    p.add_argument("--mv-iters", type=int, default=60)
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
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S36" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in [x.strip() for x in a.policies.split(",") if x.strip()]:
        gm = GaussianMap(MapConfig(voxel_size=0.25, dedup=True, gate=False,
                                   commit_min_obs=2, commit_max_age=2, fusion="weighted"),
                         ds.T_WP)
        pending, locked = {}, {}
        snap, prev_snap = None, None
        cnt = {"rejected": 0, "admitted_late": 0, "judged": 0, "locked": 0}
        t0 = time.time()
        for n, kf in enumerate(kfs):
            patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                     depth_source="pointmap", sigma_m=a.sigma,
                                     conf_min=a.conf_min, device=zcfg.DEVICE)
            if policy in ("s3", "s3r") and gm.n_committed > 10000:
                keep, inf = admission_mask(patch.means, gm.committed["means"][gm.committed["dead"] == 0],
                                           ds.T_WP, cell=0.5, tau=a.tau_adm,
                                           pending=pending, support=2, frame=int(kf),
                                           rad=a.adm_rad)
                cnt["rejected"] += inf["rejected"]; cnt["judged"] += inf["judged"]
                cnt["admitted_late"] += inf["admitted_late"]
                if keep.sum() < len(keep):
                    patch = patch.subset(keep)
            gm.ingest(patch, kf)
            gm.commit(kf)
            if policy in ("s3", "s3r") and (n + 1) % a.collapse_every == 0:
                gm.collapse_cells(grid, a.cell, prefer="quality",
                                  locked=(locked if locked else None))
                cur = surface(gm)
                if prev_snap:
                    locked.update(freeze_update(prev_snap, cur))
                    cnt["locked"] = len(locked)
                prev_snap = cur
            if n + 1 == a.stable_after:
                snap = surface(gm)
        live = gm.committed["dead"] == 0
        M = gm.committed["means"][live]
        q = quality_score(conf=gm.committed["conf"][live], opacity=gm.committed["opacity"][live],
                          wsum=gm.committed["wsum"][live], n_obs=gm.committed["n_obs"][live])
        idx = np.where(live)[0]
        sel = idx[select_by_cell(M, ds.T_WP, cell=a.cell, q=q, prefer="quality")]
        c = gm.committed
        PM, PQ, PSC, POP, PCO = (c["means"][sel], c["quats"][sel], c["scales"][sel],
                                 np.clip(c["opacity"][sel], 0.95, 1.0), c["colors"][sel])
        after = surface(gm)
        drift = drift_metric(snap or {}, after) if snap else {"n": 0}
        flo = floating_new_cells(snap or {}, after)

        # ---- rehearsal (S3R): refine the busiest tiles with an *early* view included ----
        mv = {"tiles": 0}
        if policy == "s3r":
            cams, imgs = {}, {}
            for kk in kfs:
                Tg = sim3_matrix(compose_sim3(T[kk], sg))
                R_, t_ = Tg[:3, :3], Tg[:3, 3]
                vm = np.eye(4); vm[:3, :3] = R_.T; vm[:3, 3] = -R_.T @ t_
                fid = int(npz["frame_id"][kk])
                cams[fid] = (vm, ds.K_orig, ds.cfg.height, ds.cfg.width)
                imgs[fid] = ds.read_img(fid)
            Pp = (PM - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
            st = grid.tile * grid.gsd
            kcell = np.floor(Pp[:, :2] / st).astype(np.int64)
            uq, cn = np.unique(kcell, axis=0, return_counts=True)
            order = np.argsort(cn)[::-1][: a.mv_tiles]
            for oi in order:
                ti, tj = int(uq[oi][0]), int(uq[oi][1])
                inside = (kcell[:, 0] == ti) & (kcell[:, 1] == tj)
                if inside.sum() < 20000:
                    continue
                ssel = np.where(inside)[0]
                cxy = grid.tile_xy(ti, tj)
                Pw_t = ds.T_WP[:3, :3] @ np.array([cxy[0], cxy[1], 0.0]) + ds.T_WP[:3, 3]
                seen = []
                for fid, (vm, K, H, W) in cams.items():
                    Pc = vm[:3, :3] @ Pw_t + vm[:3, 3]
                    if Pc[2] <= 1e-6:
                        continue
                    u = K[0, 0] * Pc[0] / Pc[2] + K[0, 2]
                    v = K[1, 1] * Pc[1] / Pc[2] + K[1, 2]
                    if 0 <= u < W and 0 <= v < H:
                        seen.append(fid)
                if len(seen) < 2:
                    continue
                # **rehearsal**: oldest seeing view + the two newest ones
                views = [seen[0]] + seen[-2:]
                views = list(dict.fromkeys(views))[:4]
                res = refine_multiview(PM[ssel], PQ[ssel], PSC[ssel], POP[ssel], PCO[ssel],
                                       [cams[f] for f in views], [imgs[f] for f in views],
                                       MultiViewConfig(iters=a.mv_iters), device=zcfg.DEVICE)
                PM[ssel], PQ[ssel], PSC[ssel] = res["means"], res["quats"], res["scales"]
                POP[ssel], PCO[ssel] = res["opacity"], res["colors"]
                mv["tiles"] += 1

        cloud = make_cloud(PM, PQ, PSC, POP, PCO)
        Pp = (PM - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        i0, j0 = grid.tile_of(float(Pp[:, 0].min()), float(Pp[:, 1].min()))
        i1, j1 = grid.tile_of(float(Pp[:, 0].max()), float(Pp[:, 1].max()))
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
        row = {"policy": policy, "n": int(len(sel)), "coverage": float((cov > 0.5).mean()),
               "layer": layer_metric(PM, ds.T_WP),
               "psnr": float(np.mean(ps)) if ps else None,
               "drift_p95": drift.get("p95_dz"), "drift_moved": drift.get("frac_moved"),
               "new_cells": flo.get("n_new"), "new_median_dz": flo.get("median_dz"),
               "new_p90_dz": flo.get("p90_dz"), "new_frac_gt1m": flo.get("frac_gt1m"),
               "rejected": cnt["rejected"], "judged": cnt["judged"],
               "admitted_late": cnt["admitted_late"], "locked": cnt["locked"],
               "mv_tiles": mv["tiles"], "s": round(time.time() - t0, 1)}
        rows.append(row)
        import cv2
        cv2.imwrite(str(out / f"{policy}.png"),
                    (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        print(f"[s36] {policy:8s} N={row['n']:>8,} cov={100*row['coverage']:5.1f}% "
              f"layer={row['layer']:.2f} psnr={row['psnr']:.2f} | drift p95={row['drift_p95']:.2f} "
              f"| new cells {row['new_cells']:,} med={row['new_median_dz']:.2f} "
              f">1m={100*(row['new_frac_gt1m'] or 0):4.1f}% | rej={row['rejected']:,} "
              f"late={row['admitted_late']:,} locked={row['locked']:,} mv={row['mv_tiles']} "
              f"({row['s']}s)")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
