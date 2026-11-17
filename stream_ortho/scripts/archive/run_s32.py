"""S32 -- per-frame point-map placement + P1' selection, no DSM at all.

Rationale (week8end-30): the global DSM is the *inconsistent* part -- its roofs sit
~15 m away from the per-frame point maps -- and every DSM-based placement measured
worse than no DSM at all.  The 3DGS path does not need a height field: the splats carry
their own 3D position, and an orthographic camera renders them directly.  So this run

* places every splat with the **frame's own point map** (``depth_source="pointmap"``),
* keeps one keyframe per tile (**P1' selection with real eviction**),
* keeps a fixed budget, sigma tied to the output GSD, conf gating and head opacity,
* exports the product (PNG + TIFF + JSON) and the tiles for a comparison figure.
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
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.selection import layer_metric
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


def patch_cloud(patch, device="cuda:0"):
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
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--sigma", type=float, default=0.45)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--max-points", type=int, default=8000000)
    p.add_argument("--margin", type=float, default=1.02)
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
    print(f"[s32] {len(kfs)} keyframes, output {a.gsd} m/px, sigma {a.sigma} m, "
          f"budget {a.max_points:,}, NO DSM")
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
    s_tile = grid.tile * grid.gsd

    gm = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                               commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    owner, score, clouds = {}, {}, {}
    t_b = time.time()
    swaps = 0
    for n, kf in enumerate(kfs):
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 depth_source="pointmap", sigma_m=a.sigma,
                                 conf_min=a.conf_min, device=zcfg.DEVICE)
        clouds[int(kf)] = patch_cloud(patch)
        P = (patch.means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        P = P[np.isfinite(P).all(1)]
        tiles_new = set()
        for x, y in P[:: max(1, len(P) // 300)][:, :2]:
            tiles_new.add((int(np.floor(x / s_tile)), int(np.floor(y / s_tile))))
        to_evict = []
        if gm.n_committed > 0:
            cand = clouds[int(kf)]
            for (i, j) in tiles_new:
                rgb_c, al_c, _, _ = render_tile(cand, grid, i, j)
                s_new = detail_score(rgb_c, al_c > 0.5)
                if s_new < 0:
                    continue
                if (i, j) not in owner:
                    to_evict.append((i, j))
                    owner[(i, j)] = int(kf); score[(i, j)] = s_new; swaps += 1
                elif s_new > score[(i, j)] * a.margin:
                    old_kf = owner.get((i, j))
                    rgb_o, al_o, _, _ = render_tile(clouds.get(old_kf, cand), grid, i, j)
                    if s_new > detail_score(rgb_o, al_o > 0.5) * a.margin:
                        to_evict.append((i, j))
                        owner[(i, j)] = int(kf); score[(i, j)] = s_new; swaps += 1
            if to_evict:
                gm.evict_tiles(grid, to_evict)
        gm.ingest(patch, kf)
        gm.commit(kf)
        if a.max_points:
            gm.enforce_budget(a.max_points, "ortho3")
        if (n + 1) % 10 == 0 or n + 1 == len(kfs):
            print(f"[s32] kf{n+1:3d}/{len(kfs)} N={gm.n_committed:,} live={len(gm.to_cloud()):,} "
                  f"owners={len(owner)} evicted={gm.n_evicted:,} swaps={swaps} "
                  f"({time.time()-t_b:.0f}s)")
    cloud = gm.to_cloud()
    M = cloud.means.detach().cpu().numpy()
    print(f"[s32] built in {time.time()-t_b:.0f}s: live {len(M):,}, evicted {gm.n_evicted:,}")

    P = (M - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    i0, j0 = grid.tile_of(float(P[:, 0].min()), float(P[:, 1].min()))
    i1, j1 = grid.tile_of(float(P[:, 0].max()), float(P[:, 1].max()))
    i0, i1, j0, j1 = min(i0, i1), max(i0, i1), min(j0, j1), max(j0, j1)
    tiles = []
    t_r = time.time()
    for jj in range(j0, j1 + 1):
        for ii in range(i0, i1 + 1):
            rgb, alpha, _, _ = render_tile(cloud, grid, ii, jj)
            tiles.append(Tile(ii, jj, rgb, alpha))
    mosaic, cov = stitch(tiles, shape=(j1 - j0 + 1, i1 - i0 + 1))
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S32" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2
    import tifffile

    img8 = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "ortho_selected_pointmap.png"), img8[:, :, ::-1])
    tifffile.imwrite(str(out / "ortho_selected_pointmap.tif"), img8)
    ref = PlaneReferenceMosaic(ds, a.gsd, 0.0, stride=1)
    ps = []
    for jj in range(j0, min(j0 + 4, j1 + 1)):
        for ii in range(i0, min(i0 + 4, i1 + 1)):
            rgb, al, _, _ = render_tile(cloud, grid, ii, jj)
            tgt, cv_, _, _ = ref.render_tile(grid, ii, jj, max_frames=1)
            mk = (al > 0.5) & (cv_ > 0.5)
            if mk.sum() > 5000:
                ps.append(psnr(rgb[mk], tgt[mk]))
    meta = {"source_run": run_dir.name, "n_gaussians": int(len(M)),
            "gsd_m_per_px": float(a.gsd), "sigma_m": float(a.sigma),
            "size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
            "origin_plane_m": [float(i0 * s_tile), float(j0 * s_tile)],
            "extent_plane_m": [float(i0 * s_tile), float(j0 * s_tile),
                               float((i1 + 1) * s_tile), float((j1 + 1) * s_tile)],
            "T_WP": ds.T_WP.tolist(), "coverage": float((cov > 0.5).mean()),
            "config": {"stride": a.stride, "depth_source": "pointmap", "dsm": False,
                       "selection": True, "evicted": int(gm.n_evicted), "swaps": int(swaps),
                       "budget": a.max_points, "conf_min": a.conf_min},
            "layer_ground": layer_metric(M, ds.T_WP),
            "psnr_first_tiles": float(np.mean(ps)) if ps else None,
            "render_s": round(time.time() - t_r, 1)}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"[s32] product {mosaic.shape[1]}x{mosaic.shape[0]} @ {a.gsd} m/px "
          f"cov {100*meta['coverage']:.1f}% layer {meta['layer_ground']:.2f} m "
          f"PSNR {meta['psnr_first_tiles']}")


if __name__ == "__main__":
    main()
