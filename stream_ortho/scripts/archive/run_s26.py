"""S26: P1' -- tile-ownership selection with **real eviction** in the streaming loop.

Order per keyframe (this order matters):

1. score the **new patch alone** against the **current map content** on every tile it
   covers (detail x coverage),
2. for the tiles the new keyframe wins, **evict everything currently inside them**
   (``GaussianMap.evict_tile`` without an owner filter -- the tile's content is being
   replaced, not merged),
3. then ingest the new patch, so the winning tiles are repopulated by the new frame
   while the other tiles keep their existing content.

This is strategy C at tile granularity with real replacement, so the map no longer
grows by stacking one layer per keyframe (``N`` stays bounded per tile).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.selection import layer_metric
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, stitch


def detail_score(rgb, mask):
    import cv2

    if mask.sum() < 1000:
        return -1.0
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                          cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))[mask].mean())
    return grad * float(mask.mean())


def edge_width(gray):
    import cv2

    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.hypot(gx, gy)
    th = np.percentile(mag, 99.0)
    ys, xs = np.nonzero(mag >= th)
    W = []
    for y, x in zip(ys[::25], xs[::25]):
        if not (12 <= y < g.shape[0] - 12 and 12 <= x < g.shape[1] - 12):
            continue
        prof = g[y, x - 8:x + 9] if abs(gx[y, x]) > abs(gy[y, x]) else g[y - 8:y + 9, x]
        if prof.size != 17 or np.abs(np.diff(prof)).max() < 8:
            continue
        lo, hi = prof.min(), prof.max()
        if hi - lo < 12:
            continue
        f = (prof - lo) / (hi - lo)
        i10 = int(np.argmax(f >= 0.1))
        i90 = int(np.argmax(f >= 0.9))
        if i90 > i10:
            W.append(i90 - i10)
    return (float(np.median(W)) if W else float("nan"))


def lowfreq_psnr(a, b, mask, sigma=8.0):
    import cv2

    aa = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
    bb = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigma)
    return float(psnr(aa[mask], bb[mask]))


def single_patch_cloud(patch, device="cuda:0"):
    import torch

    dev = torch.device(device)
    return GaussianCloud(
        means=torch.as_tensor(patch.means, device=dev),
        quats=torch.as_tensor(patch.quats, device=dev),
        scales=torch.as_tensor(patch.scales, device=dev),
        opacity=torch.as_tensor(patch.opacity, device=dev),
        colors=torch.as_tensor(patch.colors, device=dev),
        owner=torch.zeros(len(patch), dtype=torch.long, device=dev), meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--tiles", default="2x2")
    p.add_argument("--margin", type=float, default=1.02)
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
    # pick the block from the **densest area of the map's ground points**, not from the
    # middle camera's nadir: the two differ by ~10-30 m because each frame's surface is
    # offset, and a nadir-centred block can fall outside the point extent (measured).
    allP = []
    for kf in kfs:
        R_, t_, s_ = sim3_from_vec8(compose_sim3(T[kf], sg))
        X = np.asarray(npz["X_canon"][kf], np.float32)[::8, ::8].reshape(-1, 3)
        Pw = float(s_) * (X @ R_.T) + t_
        Pp = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        allP.append(Pp[(Pp[:, 2] > -20.0) & (Pp[:, 2] < 2.0)][:, :2])
    Pg = np.concatenate(allP)
    s_tile = grid.tile * grid.gsd
    keys = np.floor(Pg / s_tile).astype(np.int64)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    best = uniq[int(np.argmax(counts))]
    ic, jc = int(best[0]), int(best[1])
    print(f"[s26] densest ground tile ({ic},{jc}) with {int(counts.max())} samples")
    i0, j0 = ic - cols // 2, jc - rows // 2
    tiles = [(i0 + di, j0 + dj) for dj in range(rows) for di in range(cols)]
    print(f"[s26] block ({i0},{j0}) {cols}x{rows}: {len(tiles)} tiles, {len(kfs)} keyframes")

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S26" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    for tag, select in (("stack", False), ("select_evict", True)):
        gm = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                   commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
        owner, score = {}, {}
        swaps = 0
        t0 = time.time()
        for n, kf in enumerate(kfs):
            patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                     dsm=dsm, depth_source="head", device=zcfg.DEVICE)
            if select and gm.n_committed > 0:
                cand = single_patch_cloud(patch)
                cloud = gm.to_cloud()
                for (i, j) in tiles:
                    rgb_c, al_c, _, _ = render_tile(cand, grid, i, j)
                    s_new = detail_score(rgb_c, al_c > 0.5)
                    if s_new < 0:
                        continue
                    if (i, j) not in owner:
                        rgb_o, al_o, _, _ = render_tile(cloud, grid, i, j)
                        s_old = detail_score(rgb_o, al_o > 0.5)
                        if s_new >= s_old:
                            gm.evict_tile(grid, i, j)          # replace the tile
                            owner[(i, j)] = int(kf)
                            score[(i, j)] = s_new
                            swaps += 1
                    elif s_new > score[(i, j)] * a.margin:
                        gm.evict_tile(grid, i, j)
                        owner[(i, j)] = int(kf)
                        score[(i, j)] = s_new
                        swaps += 1
            gm.ingest(patch, kf)
            gm.commit(kf)
        cloud = gm.to_cloud()
        ps, lf, ew = [], [], []
        for (i, j) in tiles:
            rgb, al, _, _ = render_tile(cloud, grid, i, j)
            tgt, cov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
            mk = (al > 0.5) & (cov > 0.5)
            if mk.sum() < 5000:
                continue
            ps.append(psnr(rgb[mk], tgt[mk]))
            lf.append(lowfreq_psnr(rgb, tgt, mk))
            ew.append(edge_width(__import__("cv2").cvtColor(
                (np.clip(rgb, 0, 1) * 255).astype(np.uint8), __import__("cv2").COLOR_RGB2GRAY)))
            __import__("cv2").imwrite(str(out / f"{tag}_tile_{i}_{j}.png"),
                                      (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        lay = layer_metric(cloud.means.detach().cpu().numpy(), ds.T_WP)
        rows_out = {"variant": tag, "psnr": float(np.mean(ps)), "psnr_low": float(np.mean(lf)),
                    "edge_width": float(np.nanmean(ew)), "layer": lay, "n": int(len(cloud)),
                    "evicted": int(gm.n_evicted), "swaps": int(swaps),
                    "s": round(time.time() - t0, 1)}
        print(f"[s26] {tag:12s} PSNR={rows_out['psnr']:5.2f} low={rows_out['psnr_low']:5.2f} "
              f"edge={rows_out['edge_width']:.1f}px layer={lay:.2f} N={rows_out['n']:,} "
              f"evicted={rows_out['evicted']:,} swaps={swaps} ({rows_out['s']} s)")
        (out / f"metrics_{tag}.json").write_text(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
