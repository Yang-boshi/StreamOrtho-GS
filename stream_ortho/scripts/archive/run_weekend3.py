"""week8-end-3: texture-aware evaluation + densifying refinement, per sequence.

Tasks (see ``results/week8end-3/README.md``):

2. **protocol**: find the most detailed block of the flight by rendering the
   *reference* on a coarse grid over the whole corridor (a 7x7 window around the
   nadir is often a featureless roof -- that is how week8-end-1 got PSNR 26 dB
   on an empty grey image);
3. three sequences: full-res 2x2 block at that location -> PSNR/SSIM/sharpness;
1. densifying refinement (`backend/refine_dense.py`, borrowed
   ``gsplat.strategy.DefaultStrategy``) on 2 tiles of the block;
4. ``iso`` vs ``prior`` shapes, both **with** densification.

Outputs ``results/stage-S8/<seq>/{metrics.json,ortho.png,reference.png,compare.png,
refine/*.png}``.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.backend.refine import RefineConfig, refine_tile
from stream_ortho.backend.refine_dense import DenseConfig, refine_tile_dense
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr, ssim
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, seam_gradient_ratio, stitch


def pick_textured_block(ref, grid, gsd, bbox, coarse_tile=128, map_points=None):
    """Coarse scan of the whole corridor -> the most detailed coarse tile."""
    gsd_c = gsd * 4.0
    grid_c = OrthoGrid(grid.T_WP, gsd_c, tile=coarse_tile, halo=coarse_tile // 4,
                       z_ground=grid.z_ground, z_top=grid.z_top)
    x0, y0, x1, y1 = bbox
    step = coarse_tile * gsd_c
    best, best_score = None, -1.0
    for i in range(int(np.floor(x0 / step)), int(np.floor(x1 / step)) + 1):
        for j in range(int(np.floor(y0 / step)), int(np.floor(y1 / step)) + 1):
            rgb, cov, _, _ = ref.render_tile(grid_c, i, j, max_frames=1)
            if (cov > 0.5).mean() < 0.9:
                continue
            if map_points is not None:      # the map must cover it too (L-shaped corridor)
                x0c, y0c, x1c, y1c = grid_c.tile_window(i, j)
                n_in = int(np.sum((map_points[:, 0] >= x0c) & (map_points[:, 0] <= x1c) &
                                  (map_points[:, 1] >= y0c) & (map_points[:, 1] <= y1c)))
                if n_in < 2000:
                    continue
            g = rgb.mean(axis=2)
            score = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
            if score > best_score:
                best, best_score = (i, j), score
    return best, best_score


def main(argv=None):
    p = argparse.ArgumentParser(description="week8-end-3 pipeline")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--shape", default="iso", choices=["iso", "prior"])
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--grid", default="2x2")
    p.add_argument("--fusion", default="weighted", choices=["mean", "weighted", "best"])
    p.add_argument("--dsm-source", default="pointmap", choices=["pointmap", "da3"],
                   help="geometry source for the DSM")
    p.add_argument("--da3-npz", default=None, help="path to da3_depth*.npz")
    p.add_argument("--dsm", action="store_true", help="lift the splats onto the 2.5D DSM")
    p.add_argument("--ray-dsm", action="store_true",
                   help="iterative ray x DSM intersection (fixes roof/facade lateral offset)")
    p.add_argument("--dsm-gsd", type=float, default=0.5)
    p.add_argument("--dsm-all-kf", action="store_true",
                   help="build the DSM from every keyframe, not just the replayed ones")
    p.add_argument("--register-sign", type=float, default=1.0,
                   help="flip the phase-correlation sign (convention check)")
    p.add_argument("--register", action="store_true",
                   help="apply phase-correlation per-keyframe ground offsets")
    p.add_argument("--seam", action="store_true",
                   help="evaluate a block centred *between* two keyframes (registration test)")
    p.add_argument("--skip-refine", action="store_true")
    p.add_argument("--timing", action="store_true", help="per-keyframe pipeline timing")
    p.add_argument("--refine-tiles", type=int, default=2)
    p.add_argument("--dense-iters", type=int, default=150)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    ds = NPUDroneMapDataset(pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"])
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0][: a.max_kf].tolist()
    mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth / float(ds.K_orig[0, 0])
    grid = OrthoGrid(ds.T_WP, gsd, tile=a.tile, halo=a.halo, z_ground=0.0, z_top=800.0)

    gmap = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2, fusion=a.fusion), ds.T_WP)
    dsm = None
    if a.dsm:
        from stream_ortho.ortho.dsm import build_from_keyframes

        dsm_kfs = (np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0].tolist()
                   if a.dsm_all_kf else kfs)
        print(f"[we3] building DSM from {len(dsm_kfs)} keyframes at {a.dsm_gsd} m ...")
        if a.dsm_source == "da3":
            from stream_ortho.ortho.dsm import build_from_da3_depth

            src = a.da3_npz or str(run_dir.parent.parent / "stage-S12" / run_dir.name / "da3_depth_1008.npz")
            print(f"[we3] DSM from DA3 depth: {src}")
            dsm = build_from_da3_depth(ds, npz, sg, dsm_kfs, src, gsd=a.dsm_gsd)
        else:
            dsm = build_from_keyframes(ds, npz, sg, dsm_kfs, gsd=a.dsm_gsd)
        print(f"[we3] DSM {dsm.shape} {dsm.stats()}")
    offsets = {}
    if a.register:
        from stream_ortho.map.register import estimate_offsets

        print("[we3] estimating inter-frame registration ...")
        offsets = estimate_offsets(ds, npz, sg, kfs, verbose=False)
        print(f"[we3] offsets: {({k: (round(v[0],2), round(v[1],2)) for k,v in offsets.items()})}")
    import time as _t

    t_ingest = []
    for kf in kfs:
        off = offsets.get(kf, (0.0, 0.0))
        w_off = a.register_sign * (ds.T_WP[:3, :3] @ np.array([off[0], off[1], 0.0]))
        _t0 = _t.perf_counter()
        gmap.ingest(plane_patch(ds, npz, kf, sg, stride=a.stride, shape=a.shape, dsm=dsm,
                                ray_dsm=a.ray_dsm,
                                world_offset=w_off if a.register else None), kf)
        gmap.commit(kf)
        t_ingest.append((_t.perf_counter() - _t0) * 1000.0)
    cloud = gmap.to_cloud()
    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)

    # ---- task 2: find the most detailed block over the whole corridor ----
    P = (gmap.committed["means"].astype(np.float64) - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    bbox = (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max())
    (ci, cj), score = pick_textured_block(ref, grid, gsd, bbox, map_points=P[:, :2])
    if a.seam:
        # move the block centre to the midpoint between two adjacent keyframes
        cs = []
        for kf in kfs:
            Tv = sim3_matrix(compose_sim3(T[kf], sg))
            pp = (Tv[:3, 3] - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
            cs.append(pp[:2])
        cs = np.asarray(cs)
        mid_xy = 0.5 * (cs[len(cs) // 2 - 1] + cs[len(cs) // 2])
        ci, cj = grid.tile_of(float(mid_xy[0]), float(mid_xy[1]))
        print(f"[we3] seam block: centred between kf{kfs[len(kfs)//2-1]} and kf{kfs[len(kfs)//2]} -> ({ci},{cj})")
    # coarse tile = tile/4 pixels at 4*gsd -> exactly the same ground window as one
    # full-resolution tile, so the index maps 1:1 (not 4:1)
    ic, jc = ci, cj
    rows, cols = (int(x) for x in a.grid.lower().split("x"))
    i0, j0 = ic - cols // 2, jc - rows // 2
    print(f"[we3] {run_dir.name}: coarse block ({ci},{cj}) score={score:.4f} "
          f"-> full-res block ({i0},{j0}) {cols}x{rows}")

    # ---- task 3: full-res 2x2 block ----
    tiles = []
    for dj in range(rows):
        for di in range(cols):
            rgb, alpha, _, _ = render_tile(cloud, grid, i0 + di, j0 + dj)
            tiles.append(Tile(i0 + di, j0 + dj, rgb, alpha))
    mosaic, cov = stitch(tiles, shape=(rows, cols))
    ref_rgb, ref_cov, _ = ref.render_block(grid, i0, j0, rows, cols, max_frames=1)
    both = (cov > 0.5) & (ref_cov > 0.5)
    ps = psnr(mosaic[both], ref_rgb[both])
    ys, xs = np.nonzero(both)
    ss = ssim(mosaic[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
              ref_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    import cv2

    def grad(x):
        g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))

    gsharp = float(grad(mosaic)[both].mean() / max(grad(ref_rgb)[both].mean(), 1e-9))
    seam = seam_gradient_ratio(mosaic, a.tile)
    if a.timing:
        print(f"[we3] timing: ingest+commit {np.mean(t_ingest):.0f} ms/kf "
              f"(max {np.max(t_ingest):.0f}), {len(t_ingest)} keyframes, "
              f"N={gmap.n_committed:,}")
    print(f"[we3] block PSNR={ps:.2f} SSIM={ss:.4f} gsharp={gsharp:.3f} seam={seam['ratio']:.2f}")

    # ---- tasks 1+4: refinement with densification on N tiles ----
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S8" / run_dir.name
    (out / "refine").mkdir(parents=True, exist_ok=True)
    ref_rows = []
    for n in range(0 if a.skip_refine else a.refine_tiles):
        i, j = i0 + n, j0
        tgt, tcov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
        rgb, alpha, _, _ = render_tile(cloud, grid, i, j)
        mk = (alpha > 0.5) & (tcov > 0.5)
        if mk.sum() < 5000:
            continue
        p0 = psnr(rgb[mk], tgt[mk])
        plain = refine_tile(cloud, grid, i, j, tgt, tcov, RefineConfig(iters_a=120))
        dense = refine_tile_dense(cloud, grid, i, j, tgt, tcov,
                                  DenseConfig(iters=a.dense_iters))
        p_plain = psnr(plain["rgb_after"][mk], tgt[mk])
        p_dense = psnr(dense["rgb_after"][mk], tgt[mk])
        ref_rows.append({"tile": [int(i), int(j)], "psnr_before": float(p0),
                         "psnr_plain": float(p_plain), "psnr_dense": float(p_dense),
                         "gain_plain": float(p_plain - p0), "gain_dense": float(p_dense - p0),
                         "n_before": dense["n_before"], "n_after": dense["n_after"],
                         "time_plain": plain["time_s"], "time_dense": dense["time_s"]})
        for tag, arr in (("before", rgb), ("plain", plain["rgb_after"]), ("dense", dense["rgb_after"]),
                         ("target", tgt)):
            cv2.imwrite(str(out / "refine" / f"tile_{i}_{j}_{tag}.png"),
                        (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        print(f"  tile ({i},{j}) {p0:.2f} -> plain {p_plain:.2f} ({p_plain-p0:+.2f}) "
              f"/ dense {p_dense:.2f} ({p_dense-p0:+.2f})  N {dense['n_before']:,}->{dense['n_after']:,}")

    def save(name, arr):
        cv2.imwrite(str(out / name), (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    save("ortho.png", mosaic)
    save("reference.png", ref_rgb)
    gap = np.ones((mosaic.shape[0], 8, 3), np.float32)
    save("compare.png", np.concatenate([mosaic, gap, ref_rgb], axis=1))

    metrics = {
        "run": run_dir.name, "seq": ds.name, "shape": a.shape, "stride": a.stride,
        "tile": a.tile, "gsd_m_per_px": gsd, "n_kfs": len(kfs),
        "n_committed": gmap.n_committed,
        "coarse_block": [int(ci), int(cj)], "coarse_score": score,
        "block": [int(i0), int(j0), rows, cols],
        "psnr": float(ps), "ssim": float(ss), "gsharp": gsharp, "seam": seam["ratio"],
        "fusion": a.fusion, "dsm": bool(a.dsm), "registered": bool(a.register),
        "ingest_ms_mean": float(np.mean(t_ingest)), "ingest_ms_max": float(np.max(t_ingest)), "ray_dsm": bool(a.ray_dsm),
        "dsm_gsd": a.dsm_gsd if a.dsm else None, "dsm_all_kf": bool(a.dsm_all_kf),
        "refine": ref_rows,
        "mean_gain_plain": float(np.mean([r["gain_plain"] for r in ref_rows])) if ref_rows else 0.0,
        "mean_gain_dense": float(np.mean([r["gain_dense"] for r in ref_rows])) if ref_rows else 0.0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[we3] mean refine gain: plain {metrics['mean_gain_plain']:+.2f} / "
          f"dense {metrics['mean_gain_dense']:+.2f} dB -> {out}")


if __name__ == "__main__":
    main()
