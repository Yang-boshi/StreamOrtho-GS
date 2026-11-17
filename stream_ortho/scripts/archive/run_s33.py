"""S33 -- 2.5D **cell selection** (P1''): every (x, y) keeps its best observation.

Replaces the tile-level replacement of S32, which deleted a whole tile's content and
therefore punched tile-shaped holes wherever the winning keyframe did not cover the
tile (measured: 8 M points -> 4.18 M live, coverage 61.3% -> 53.0%).

Here the map is built **without** any eviction and the selection is applied once at the
end: per ``--cell``-metre cell of the plane frame keep the observation with the best
quality ``q = conf * opacity * wsum * min(n_obs,2)/2``.  Coverage is preserved by
construction, layers collapse to one surface, and no voxel-hash match is required
(which is why the voxel merge never fired).
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
from stream_ortho.map.selection import layer_metric, quality_score, select_by_cell
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, stitch


def build_cloud(means, quats, scales, opacity, colors, device="cuda:0"):
    import torch

    dev = torch.device(device)
    return GaussianCloud(
        means=torch.as_tensor(np.ascontiguousarray(means), device=dev),
        quats=torch.as_tensor(np.ascontiguousarray(quats), device=dev),
        scales=torch.as_tensor(np.ascontiguousarray(scales), device=dev),
        opacity=torch.as_tensor(np.ascontiguousarray(opacity), device=dev),
        colors=torch.as_tensor(np.ascontiguousarray(colors), device=dev),
        owner=torch.zeros(len(means), dtype=torch.long, device=dev), meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--sigma", type=float, default=0.45)
    p.add_argument("--cell", type=float, default=0.45)
    p.add_argument("--keep-n", type=int, default=1)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--edge-reject", type=int, default=0)
    p.add_argument("--cell-color", action="store_true",
                   help="average the colours of all samples inside a cell (C)")
    p.add_argument("--max-points", type=int, default=8000000)
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
    print(f"[s33] {len(kfs)} keyframes, out {a.gsd} m/px, sigma {a.sigma} m, cell {a.cell} m, "
          f"keep {a.keep_n}, budget {a.max_points:,}, NO DSM, NO eviction")

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    gm = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                               commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    t_b = time.time()
    for n, kf in enumerate(kfs):
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 depth_source="pointmap", sigma_m=a.sigma,
                                 conf_min=a.conf_min, device=zcfg.DEVICE)
        gm.ingest(patch, kf)
        gm.commit(kf)
        if a.max_points:
            gm.enforce_budget(a.max_points, "ortho3")
        if (n + 1) % 10 == 0 or n + 1 == len(kfs):
            print(f"[s33] kf{n+1:3d}/{len(kfs)} N={gm.n_committed:,} ({time.time()-t_b:.0f}s)")
    c = gm.committed
    live = c["dead"] == 0 if "dead" in c else np.ones(gm.n_committed, bool)
    print(f"[s33] built {gm.n_committed:,} in {time.time()-t_b:.0f}s")

    # ---- P1'': 2.5D cell selection ----
    t_s = time.time()
    q = quality_score(conf=c["conf"][live], opacity=c["opacity"][live],
                      wsum=c["wsum"][live], n_obs=c["n_obs"][live])
    idx_live = np.where(live)[0]
    sel = idx_live[select_by_cell(c["means"][live], ds.T_WP, cell=a.cell, q=q,
                                  keep_n=a.keep_n)]
    print(f"[s33] cell selection: {len(idx_live):,} -> {len(sel):,} "
          f"({100*len(sel)/max(len(idx_live),1):.1f}%) in {time.time()-t_s:.1f}s")
    M, Q, SC, OP, CO = (c["means"][sel], c["quats"][sel], c["scales"][sel],
                        c["opacity"][sel], c["colors"][sel])
    if a.cell_color:
        from stream_ortho.map.selection import mean_colors_by_cell

        CO = mean_colors_by_cell(M, CO, ds.T_WP, cell=a.cell)
        print(f"[s33] C: cell colours averaged (cell {a.cell} m)")
    cloud = build_cloud(M, Q, SC, OP, CO)

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
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S33" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2
    import tifffile

    img8 = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "ortho_cell_selected.png"), img8[:, :, ::-1])
    tifffile.imwrite(str(out / "ortho_cell_selected.tif"), img8)

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
            "gsd_m_per_px": float(a.gsd), "sigma_m": float(a.sigma), "cell_m": float(a.cell),
            "size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
            "origin_plane_m": [float(i0 * s_tile), float(j0 * s_tile)],
            "extent_plane_m": [float(i0 * s_tile), float(j0 * s_tile),
                               float((i1 + 1) * s_tile), float((j1 + 1) * s_tile)],
            "T_WP": ds.T_WP.tolist(), "coverage": float((cov > 0.5).mean()),
            "layer_ground": layer_metric(M, ds.T_WP),
            "psnr_first_tiles": float(np.mean(ps)) if ps else None,
            "config": {"stride": a.stride, "depth_source": "pointmap", "dsm": False,
                       "selection": "cell", "keep_n": a.keep_n, "budget": a.max_points},
            "render_s": round(time.time() - t_r, 1)}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"[s33] product {mosaic.shape[1]}x{mosaic.shape[0]} @ {a.gsd} m/px "
          f"cov {100*meta['coverage']:.1f}% layer {meta['layer_ground']:.2f} m "
          f"PSNR {meta['psnr_first_tiles']}")


if __name__ == "__main__":
    main()
