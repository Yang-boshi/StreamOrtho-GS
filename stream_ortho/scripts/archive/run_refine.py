"""Run the per-tile photometric refinement (week8-end-1 task 3) on N tiles.

Builds the map once (same config as the ortho driver), then for each sampled tile
renders it, refines only that tile's Gaussians, and reports PSNR/sharpness before
and after.  Saves before/after/curve figures.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.backend.refine import RefineConfig, refine_tile
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def main(argv=None):
    p = argparse.ArgumentParser(description="per-tile refinement")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--tiles", type=int, default=3)
    p.add_argument("--iters", type=int, default=120)
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
    grid = OrthoGrid(ds.T_WP, gsd, tile=512, halo=32, z_ground=0.0, z_top=800.0)

    gmap = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2), ds.T_WP)
    for kf in kfs:
        gmap.ingest(plane_patch(ds, npz, kf, sg, stride=a.stride, shape="prior"), kf)
        gmap.commit(kf)
    cloud = gmap.to_cloud()
    print(f"[refine] {run_dir.name}: N={len(cloud):,} stride={a.stride} gsd={gsd:.4f}")

    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))
    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S7c" / run_dir.name / "refine"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    import cv2

    for n in range(a.tiles):
        i, j = ic + n, jc
        tgt, cov, nfr, _ = ref.render_tile(grid, i, j, max_frames=1)
        rgb, alpha, _, _ = render_tile(cloud, grid, i, j)
        mk = (alpha > 0.5) & (cov > 0.5)
        if mk.sum() < 5000:
            continue
        p0 = psnr(rgb[mk], tgt[mk])
        res = refine_tile(cloud, grid, i, j, tgt, cov, RefineConfig(iters_a=a.iters))
        p1 = psnr(res["rgb_after"][mk], tgt[mk])
        rows.append({"tile": [int(i), int(j)], "psnr_before": float(p0),
                     "psnr_after": float(p1), "gain": float(p1 - p0),
                     "best": res["best_psnr"], "n_refined": res["n_refined"],
                     "time_s": res["time_s"], "curve": res["curve"]})
        cv2.imwrite(str(out / f"tile_{i}_{j}_before.png"),
                    (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"tile_{i}_{j}_after.png"),
                    (np.clip(res["rgb_after"], 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"tile_{i}_{j}_target.png"),
                    (np.clip(tgt, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        print(f"  tile ({i},{j}) PSNR {p0:5.2f} -> {p1:5.2f} ({p1-p0:+.2f})  "
              f"best {res['best_psnr']:.2f}  {res['time_s']}s  N_ref={res['n_refined']:,}")

    if rows:
        mean_gain = float(np.mean([r["gain"] for r in rows]))
        mean_best = float(np.mean([r["best"] - r["psnr_before"] for r in rows]))
        summary = {"run": run_dir.name, "seq": ds.name, "n_tiles": len(rows),
                   "mean_gain": mean_gain, "mean_best_gain": mean_best,
                   "mean_time_s": float(np.mean([r["time_s"] for r in rows])),
                   "rows": rows}
        (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
        print(f"[refine] mean gain {mean_gain:+.2f} dB (best {mean_best:+.2f}), "
              f"{summary['mean_time_s']:.2f} s/tile -> {out}")


if __name__ == "__main__":
    main()
