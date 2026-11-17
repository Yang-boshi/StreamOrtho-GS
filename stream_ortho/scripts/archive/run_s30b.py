"""S30b -- "fixed" product with fixes 1,2,4,5,6 (no selection / no P4 for speed).

* sigma bound to the output GSD (1 px),
* stride 2,
* floater gating (conf + height),
* ray x DSM placement (``depth_source="dsm"``),
* DSM from 32 keyframes,
* fixed budget 2 M so the streaming cost stays bounded.

Selection (fix 3) and the multi-view refinement (fix 7) are measured separately
(week8end-25 / -26) and need the streaming integration; this run isolates the effect of
the *rendering-side* fixes on the product.
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
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, stitch


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--dsm-kf", type=int, default=32)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--z-max", type=float, default=40.0)
    p.add_argument("--max-points", type=int, default=2000000)
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
    print(f"[s30b] {len(kfs)} keyframes, native gsd {depth/ds.K_orig[0,0]:.4f}, out {a.gsd}")
    dsm_kfs = kfs if a.dsm_kf >= len(kfs) else kfs[:: max(1, len(kfs) // a.dsm_kf)]
    t0 = time.time()
    dsm = build_piecewise_dsm(ds, npz, sg, dsm_kfs, gsd=0.5)
    print(f"[s30b] DSM {dsm.shape} from {len(dsm_kfs)} kf in {time.time()-t0:.1f}s")
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    gm = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                               commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    t_b = time.time()
    dropped = 0
    for n, kf in enumerate(kfs):
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride, dsm=dsm,
                                 depth_source="dsm", sigma_m=a.gsd, opacity_min=0.0,
                                 conf_min=a.conf_min, device=zcfg.DEVICE)
        P = (patch.means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        keep = np.abs(P[:, 2]) < a.z_max
        dropped += int((~keep).sum())
        if not keep.all():
            patch = patch.subset(keep)
        gm.ingest(patch, kf)
        gm.commit(kf)
        if a.max_points:
            gm.enforce_budget(a.max_points, "ortho3")
        if (n + 1) % 10 == 0 or n + 1 == len(kfs):
            print(f"[s30b] kf{n+1:3d}/{len(kfs)} N={gm.n_committed:,} "
                  f"dropped={dropped:,} ({time.time()-t_b:.0f}s)")
    cloud = gm.to_cloud()
    M = cloud.means.detach().cpu().numpy()
    print(f"[s30b] built: N={len(cloud):,}, floaters dropped {dropped:,}")

    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
    s_tile = grid.tile * grid.gsd
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
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S30b" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2
    import tifffile

    img8 = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "ortho_product_fixed.png"), img8[:, :, ::-1])
    tifffile.imwrite(str(out / "ortho_product_fixed.tif"), img8)
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
            "gsd_m_per_px": float(a.gsd), "size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
            "origin_plane_m": [float(i0 * s_tile), float(j0 * s_tile)],
            "extent_plane_m": [float(i0 * s_tile), float(j0 * s_tile),
                               float((i1 + 1) * s_tile), float((j1 + 1) * s_tile)],
            "T_WP": ds.T_WP.tolist(), "coverage": float((cov > 0.5).mean()),
            "config": {"stride": a.stride, "sigma_m": a.gsd, "depth_source": "dsm",
                       "conf_min": a.conf_min, "z_max": a.z_max, "dsm_kf": len(dsm_kfs),
                       "max_points": a.max_points, "selection": False, "P4": False},
            "dropped_floaters": int(dropped),
            "layer_ground": layer_metric(M, ds.T_WP),
            "psnr_first_tiles": float(np.mean(ps)) if ps else None,
            "render_s": round(time.time() - t_r, 1),
            "note": "PNG + TIFF + JSON sidecar (no GDAL)"}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"[s30b] product {mosaic.shape[1]}x{mosaic.shape[0]} @ {a.gsd} m/px "
          f"cov {100*meta['coverage']:.1f}% layer {meta['layer_ground']:.2f} m "
          f"PSNR(first tiles) {meta['psnr_first_tiles']}")


if __name__ == "__main__":
    main()
