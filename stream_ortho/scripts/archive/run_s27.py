"""S27 -- P4': multi-view joint refinement on every *selected* tile.

After P1' each tile is owned by one keyframe (the other layers were evicted).  This
script takes the live Gaussians of each tile, finds the K keyframes that actually see
that tile, and refines the tile's Gaussians against **all of them at once** (poses
fixed, means almost frozen).  It then re-renders the tile and reports the ortho PSNR
before/after plus the per-tile cost.

This is the only optimisation in the pipeline that uses more than one view, so it is the
one that can reduce the cross-frame disagreement behind the blur.
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
from stream_ortho.frontend.keyframe_policy import bbox_from_pose
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def detail_score(rgb, mask):
    import cv2

    if mask.sum() < 1000:
        return -1.0
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                          cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))[mask].mean())
    return grad * float(mask.mean())


def cloud_from(means, quats, scales, opacity, colors, device="cuda:0"):
    import torch

    dev = torch.device(device)
    return GaussianCloud(
        means=torch.as_tensor(means, device=dev), quats=torch.as_tensor(quats, device=dev),
        scales=torch.as_tensor(scales, device=dev), opacity=torch.as_tensor(opacity, device=dev),
        colors=torch.as_tensor(colors, device=dev),
        owner=torch.zeros(len(means), dtype=torch.long, device=dev), meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--tiles", default="2x2")
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-kf", type=int, default=8)
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
    mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth / float(ds.K_orig[0, 0])
    grid = OrthoGrid(ds.T_WP, gsd, tile=512, halo=32, z_ground=0.0, z_top=800.0)
    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    dsm = build_piecewise_dsm(ds, npz, sg, kfs[:4], gsd=0.5)
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)

    rows, cols = (int(x) for x in a.tiles.lower().split("x"))
    # anchor the block on the **middle keyframe's** own ground points: the union over
    # all keyframes is fine too, but only the middle frame is guaranteed to be seen by
    # several neighbours.
    allP = []
    for kf in [mid]:
        X = np.asarray(npz["X_canon"][kf], np.float32)[::4, ::4].reshape(-1, 3)
        R_, t_, s_ = sim3_from_vec8(compose_sim3(T[kf], sg))
        Pw = float(s_) * (X @ R_.T) + t_
        Pp = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        allP.append(Pp[(Pp[:, 2] > -20.0) & (Pp[:, 2] < 2.0)][:, :2])
    s_tile = grid.tile * grid.gsd
    keys = np.floor(np.concatenate(allP) / s_tile).astype(np.int64)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    # candidate tiles: any tile that contains ground points (scan a window around the
    # densest one); the "seen >= 2 views" filter is applied later, so a tile that no
    # keyframe footprint covers is simply skipped.  (Points and footprints disagree by
    # tens of metres -- another symptom of the per-frame depth inconsistency.)
    ic, jc = (int(uniq[int(np.argmax(counts))][0]), int(uniq[int(np.argmax(counts))][1]))
    tiles = [(ic + di, jc + dj) for dj in range(-3, rows + 3) for di in range(-3, cols + 3)]
    print(f"[s27] scanning {len(tiles)} tiles around ({ic},{jc})")

    # ---- build the map exactly as P1' does: score -> evict -> ingest ----
    gm = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                               commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    owner, score = {}, {}
    for kf in kfs:
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 dsm=dsm, depth_source="head", device=zcfg.DEVICE)
        cand = cloud_from(patch.means, patch.quats, patch.scales, patch.opacity, patch.colors)
        if gm.n_committed > 0:
            cloud = gm.to_cloud()
            for (i, j) in tiles:
                rgb_c, al_c, _, _ = render_tile(cand, grid, i, j)
                s_new = detail_score(rgb_c, al_c > 0.5)
                if s_new < 0:
                    continue
                if (i, j) not in owner:
                    rgb_o, al_o, _, _ = render_tile(cloud, grid, i, j)
                    if s_new >= detail_score(rgb_o, al_o > 0.5):
                        gm.evict_tile(grid, i, j)
                        owner[(i, j)] = int(kf); score[(i, j)] = s_new
                elif s_new > score[(i, j)] * 1.02:
                    gm.evict_tile(grid, i, j)
                    owner[(i, j)] = int(kf); score[(i, j)] = s_new
        gm.ingest(patch, kf)
        gm.commit(kf)

    # ---- P4': refine each tile's live content against the views that see it ----
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S27" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2

    # keyframe cameras / images (world -> camera)
    cams, imgs = {}, {}
    for kf in kfs:
        Tg = sim3_matrix(compose_sim3(T[kf], sg))
        R_, t_ = Tg[:3, :3], Tg[:3, 3]
        vm = np.eye(4); vm[:3, :3] = R_.T; vm[:3, 3] = -R_.T @ t_
        fid = int(npz["frame_id"][kf])
        cams[fid] = (vm, ds.K_orig, ds.cfg.height, ds.cfg.width)
        imgs[fid] = ds.read_img(fid)

    c = gm.committed
    live = c["dead"] == 0
    P = (c["means"] - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    rows_out = []
    for (i, j) in tiles:
        inside = live & ((P[:, 0] >= i * s_tile) & (P[:, 0] < (i + 1) * s_tile) &
                         (P[:, 1] >= j * s_tile) & (P[:, 1] < (j + 1) * s_tile))
        if inside.sum() < 5000:
            continue
        sel = np.where(inside)[0]
        if len(sel) > 120000:
            sel = sel[:: max(1, len(sel) // 120000)]
        # views whose footprint contains the tile
        # which keyframes actually see this tile?  Project the tile's centre (on the
        # ground plane) into each camera and test the image bounds -- robust, unlike the
        # footprint bbox (points and footprints use slightly different conventions).
        tgt_xy = grid.tile_xy(i, j)
        Pw_t = ds.T_WP[:3, :3] @ np.array([tgt_xy[0], tgt_xy[1], 0.0]) + ds.T_WP[:3, 3]
        seen = []
        for fid, (vm, K, H, W) in cams.items():
            Pc = vm[:3, :3] @ Pw_t + vm[:3, 3]
            if Pc[2] <= 1e-6:
                continue
            u = K[0, 0] * Pc[0] / Pc[2] + K[0, 2]
            v = K[1, 1] * Pc[1] / Pc[2] + K[1, 2]
            if 0 <= u < W and 0 <= v < H:
                seen.append(fid)
        seen = seen[: a.views]
        if len(seen) < 2:
            continue
        t0 = time.time()
        res = refine_multiview(c["means"][sel], c["quats"][sel], c["scales"][sel],
                               c["opacity"][sel], c["colors"][sel],
                               [cams[f] for f in seen],
                               [imgs[f] for f in seen],
                               MultiViewConfig(iters=a.iters), device=zcfg.DEVICE)
        dt = time.time() - t0
        tgt, cov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
        before = render_tile(cloud_from(c["means"][sel], c["quats"][sel], c["scales"][sel],
                                        c["opacity"][sel], c["colors"][sel]), grid, i, j)
        after = render_tile(cloud_from(res["means"], res["quats"], res["scales"],
                                       res["opacity"], res["colors"]), grid, i, j)
        mk = (before[1] > 0.5) & (cov > 0.5)
        mk2 = (after[1] > 0.5) & (cov > 0.5)
        if mk.sum() < 5000 or mk2.sum() < 5000:
            continue
        r = {"tile": [int(i), int(j)], "n": int(len(sel)), "views": len(seen),
             "psnr_before": float(psnr(before[0][mk], tgt[mk])),
             "psnr_after": float(psnr(after[0][mk2], tgt[mk2])),
             "view_psnr_before": res["mean_before"], "view_psnr_after": res["mean_after"],
             "s": round(dt, 1)}
        r["gain"] = r["psnr_after"] - r["psnr_before"]
        rows_out.append(r)
        cv2.imwrite(str(out / f"before_tile_{i}_{j}.png"),
                    (np.clip(before[0], 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"after_tile_{i}_{j}.png"),
                    (np.clip(after[0], 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"reference_tile_{i}_{j}.png"),
                    (np.clip(tgt, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        print(f"[s27] tile ({i},{j}) views={len(seen)} n={len(sel):,} "
              f"ortho {r['psnr_before']:.2f} -> {r['psnr_after']:.2f} dB ({r['gain']:+.2f}) "
              f"| view {res['mean_before']:.2f} -> {res['mean_after']:.2f} | {dt:.0f} s")

    if rows_out:
        summary = {"tiles": len(rows_out),
                   "mean_gain": float(np.mean([r["gain"] for r in rows_out])),
                   "mean_before": float(np.mean([r["psnr_before"] for r in rows_out])),
                   "mean_after": float(np.mean([r["psnr_after"] for r in rows_out])),
                   "mean_s": float(np.mean([r["s"] for r in rows_out]))}
        (out / "metrics.json").write_text(json.dumps({"rows": rows_out, "summary": summary},
                                                     indent=2))
        print(f"[s27] mean: {summary['mean_before']:.2f} -> {summary['mean_after']:.2f} dB "
              f"({summary['mean_gain']:+.2f}) over {summary['tiles']} tiles, "
              f"{summary['mean_s']:.0f} s/tile")


if __name__ == "__main__":
    main()
