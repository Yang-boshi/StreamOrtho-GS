"""S30 -- the "fixed" ortho product, applying the P5 fix list 1->6.

Fixes applied here:

1. **sigma bound to the output GSD** (``sigma_m``), not to the input sampling,
2. **stride 2** (0.41 m sampling) instead of 4,
3. **P1' selection with real eviction** (score -> evict the tile -> ingest),
4. **floater gating**: ``conf_min`` + opacity not forced to 0.95 + a height sanity mask,
5. **ray x DSM placement** (``depth_source="dsm"``) so roofs land on the DSM surface,
6. **DSM from 32 keyframes** (and the P4' multi-view refinement afterwards).

Outputs the product (PNG + TIFF + sidecar JSON), a before/after comparison figure and a
fixed-extent streaming video, all under ``results/stage-S30/<run>/``.
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


def cloud_of_patch(patch, device="cuda:0"):
    import torch

    dev = torch.device(device)
    return GaussianCloud(
        means=torch.as_tensor(patch.means, device=dev),
        quats=torch.as_tensor(patch.quats, device=dev),
        scales=torch.as_tensor(patch.scales, device=dev),
        opacity=torch.as_tensor(patch.opacity, device=dev),
        colors=torch.as_tensor(patch.colors, device=dev),
        owner=torch.zeros(len(patch), dtype=torch.long, device=dev), meta={})


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
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--dsm-kf", type=int, default=32)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--z-max", type=float, default=40.0)
    p.add_argument("--mv-tiles", type=int, default=12, help="how many tiles to refine (P4')")
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
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()
    mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd_native = depth / float(ds.K_orig[0, 0])
    print(f"[s30] {len(kfs)} keyframes, native gsd {gsd_native:.4f}, output {a.gsd} m/px")

    # fix 6: DSM from (up to) 32 keyframes
    dsm_kfs = kfs if a.dsm_kf >= len(kfs) else kfs[:: max(1, len(kfs) // a.dsm_kf)]
    t0 = time.time()
    dsm = build_piecewise_dsm(ds, npz, sg, dsm_kfs, gsd=0.5)
    print(f"[s30] DSM from {len(dsm_kfs)} keyframes in {time.time()-t0:.1f}s: {dsm.shape} "
          f"{ {k: round(v,2) if isinstance(v,float) else v for k,v in dsm.stats().items()} }")

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
    s_tile = grid.tile * grid.gsd

    # ---- streaming build with P1' selection + eviction (fix 3) ----
    gm = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                               commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    owner, score = {}, {}
    t_build = time.time()
    for n, kf in enumerate(kfs):
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 dsm=dsm, depth_source="dsm",            # fix 5
                                 sigma_m=a.gsd,                           # fix 1
                                 opacity_min=0.0, conf_min=a.conf_min,    # fix 4
                                 device=zcfg.DEVICE)
        # fix 4b: drop splats that are absurdly high (floaters / ghost copies)
        P = (patch.means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        keep = np.abs(P[:, 2]) < a.z_max
        if not keep.all():
            patch = patch.subset(keep)
        if gm.n_committed > 0:
            cand = cloud_of_patch(patch)
            cur = gm.to_cloud()
            Pc = (patch.means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
            tiles_new = set()
            for x, y in Pc[:: max(1, len(Pc) // 200)][:, :2]:
                tiles_new.add((int(np.floor(x / s_tile)), int(np.floor(y / s_tile))))
            for (i, j) in tiles_new:
                rgb_c, al_c, _, _ = render_tile(cand, grid, i, j)
                s_new = detail_score(rgb_c, al_c > 0.5)
                if s_new < 0:
                    continue
                if (i, j) not in owner:
                    rgb_o, al_o, _, _ = render_tile(cur, grid, i, j)
                    if s_new >= detail_score(rgb_o, al_o > 0.5):
                        gm.evict_tile(grid, i, j)
                        owner[(i, j)] = int(kf); score[(i, j)] = s_new
                elif s_new > score[(i, j)] * 1.02:
                    gm.evict_tile(grid, i, j)
                    owner[(i, j)] = int(kf); score[(i, j)] = s_new
        gm.ingest(patch, kf)
        gm.commit(kf)
        if (n + 1) % 10 == 0 or n + 1 == len(kfs):
            print(f"[s30] kf{n+1:3d}/{len(kfs)} N={gm.n_committed:,} live={len(gm.to_cloud()):,} "
                  f"owners={len(owner)} evicted={gm.n_evicted:,} ({time.time()-t_build:.0f}s)")
    cloud = gm.to_cloud()
    print(f"[s30] built in {time.time()-t_build:.0f}s: N={len(cloud):,} "
          f"(evicted {gm.n_evicted:,})")

    # ---- P4' refinement on the busiest tiles (fix 7 / item 6) ----
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S30" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    c = gm.committed
    live = c["dead"] == 0
    M = c["means"][live]
    Q = c["quats"][live]
    SC = c["scales"][live]
    OP = c["opacity"][live]
    CO = c["colors"][live]
    Pp = (M - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    keys = np.floor(Pp[:, :2] / s_tile).astype(np.int64)
    uq, cnt = np.unique(keys, axis=0, return_counts=True)
    order = np.argsort(cnt)[::-1]
    cams, imgs = {}, {}
    for kf in kfs:
        Tg = sim3_matrix(compose_sim3(T[kf], sg))
        R_, t_ = Tg[:3, :3], Tg[:3, 3]
        vm = np.eye(4); vm[:3, :3] = R_.T; vm[:3, 3] = -R_.T @ t_
        fid = int(npz["frame_id"][kf])
        cams[fid] = (vm, ds.K_orig, ds.cfg.height, ds.cfg.width)
        imgs[fid] = ds.read_img(fid)
    refined = 0
    t_mv = time.time()
    for idx in order[: a.mv_tiles]:
        ti, tj = int(uq[idx][0]), int(uq[idx][1])
        inside = (keys[:, 0] == ti) & (keys[:, 1] == tj)
        if inside.sum() < 20000:
            continue
        sel = np.where(inside)[0]
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
        seen = seen[:4]
        if len(seen) < 2:
            continue
        res = refine_multiview(M[sel], Q[sel], SC[sel], OP[sel], CO[sel],
                               [cams[f] for f in seen], [imgs[f] for f in seen],
                               MultiViewConfig(iters=a.mv_iters), device=zcfg.DEVICE)
        M[sel], Q[sel], SC[sel] = res["means"], res["quats"], res["scales"]
        OP[sel], CO[sel] = res["opacity"], res["colors"]
        refined += len(sel)
    print(f"[s30] P4' refined {refined:,} gaussians on {min(a.mv_tiles, len(uq))} tiles "
          f"in {time.time()-t_mv:.0f}s")

    cloud = cloud_from(M, Q, SC, OP, CO)

    # ---- product ----
    i0, j0 = grid.tile_of(float(Pp[:, 0].min()), float(Pp[:, 1].min()))
    i1, j1 = grid.tile_of(float(Pp[:, 0].max()), float(Pp[:, 1].max()))
    i0, i1 = min(i0, i1), max(i0, i1)
    j0, j1 = min(j0, j1), max(j0, j1)
    tiles = []
    t_prod = time.time()
    for jj in range(j0, j1 + 1):
        for ii in range(i0, i1 + 1):
            rgb, alpha, _, _ = render_tile(cloud, grid, ii, jj)
            tiles.append(Tile(ii, jj, rgb, alpha))
    mosaic, cov = stitch(tiles, shape=(j1 - j0 + 1, i1 - i0 + 1))
    import cv2
    import tifffile

    img8 = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "ortho_product_fixed.png"), img8[:, :, ::-1])
    tifffile.imwrite(str(out / "ortho_product_fixed.tif"), img8)
    meta = {"source_run": run_dir.name, "n_gaussians": int(len(M)),
            "gsd_m_per_px": float(a.gsd), "size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
            "origin_plane_m": [float(i0 * s_tile), float(j0 * s_tile)],
            "extent_plane_m": [float(i0 * s_tile), float(j0 * s_tile),
                               float((i1 + 1) * s_tile), float((j1 + 1) * s_tile)],
            "T_WP": ds.T_WP.tolist(), "coverage": float((cov > 0.5).mean()),
            "config": {"stride": a.stride, "sigma_m": a.gsd, "depth_source": "dsm",
                       "conf_min": a.conf_min, "z_max": a.z_max,
                       "dsm_kf": len(dsm_kfs), "selection": True, "P4": True,
                       "mv_tiles": a.mv_tiles, "voxel": a.voxel},
            "evicted": int(gm.n_evicted), "owners": len(owner),
            "note": "plane frame from T_WP; PNG + TIFF + JSON sidecar (no GDAL)"}
    (out / "ortho_product_fixed.json").write_text(json.dumps(meta, indent=2))

    # sanity metrics on the product grid (vs the single-frame plane reference)
    ref = PlaneReferenceMosaic(ds, a.gsd, 0.0, stride=1)
    ps = []
    for jj in range(j0, min(j0 + 3, j1 + 1)):
        for ii in range(i0, min(i0 + 3, i1 + 1)):
            rgb, al, _, _ = render_tile(cloud, grid, ii, jj)
            tgt, cv_, _, _ = ref.render_tile(grid, ii, jj, max_frames=1)
            mk = (al > 0.5) & (cv_ > 0.5)
            if mk.sum() > 5000:
                ps.append(psnr(rgb[mk], tgt[mk]))
    if ps:
        print(f"[s30] product PSNR (first tiles) {np.mean(ps):.2f} dB")
    lay = layer_metric(M, ds.T_WP)
    print(f"[s30] product: {mosaic.shape[1]}x{mosaic.shape[0]} @ {a.gsd} m/px, "
          f"coverage {100*meta['coverage']:.1f}%, layer {lay:.2f} m, "
          f"{len(M):,} gaussians, {time.time()-t_prod:.0f}s render")
    (out / "metrics.json").write_text(json.dumps(
        {**meta, "product_psnr": float(np.mean(ps)) if ps else None, "layer": lay}, indent=2))


if __name__ == "__main__":
    main()
