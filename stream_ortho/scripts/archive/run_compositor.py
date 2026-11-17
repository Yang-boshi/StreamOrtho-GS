"""Week8-end-6: ortho compositor vs 3DGS rendering (design doc v3 §4.6.1).

For N tiles of a textured block this script reports, for every tile:

* **consensus** = per-pixel median of all valid frame warps (metric target),
* **3DGS render** (our current product) vs consensus,
* **compositor** output (fused or single-frame fallback) vs consensus,
* **best single frame** (the floor the compositor must never go below),
* sharpness of each, the chosen path, and the time per tile.

Outputs ``results/stage-S11/<run>/``: ``metrics.json``, ``compare_<i>_<j>.png``
(consensus | 3DGS | compositor | chosen-mask) and a short summary table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.compositor import CompositorConfig, OrthoCompositor, _grad
from stream_ortho.ortho.dsm import build_from_keyframes
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile


def main(argv=None):
    p = argparse.ArgumentParser(description="compositor vs 3DGS")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--k-frames", type=int, default=3)
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--poses", default="slam", choices=["slam", "gt"],
                   help="slam = the map's own poses (what we have in reality); "
                        "gt = GT poses (upper bound for the compositor)")
    p.add_argument("--dsm-source", default="pointmap", choices=["pointmap", "da3"],
                   help="geometry source for the DSM")
    p.add_argument("--da3-npz", default=None, help="path to da3_depth*.npz")
    p.add_argument("--dsm", action="store_true")
    p.add_argument("--dsm-gsd", type=float, default=0.5)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=zcfg.DEVICE)
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

    # camera->GT world poses for the candidate frames (all keyframes, not only the replayed ones)
    T_gt = {}
    all_kf = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0].tolist()
    for k in all_kf:
        fid_k = int(npz["frame_id"][k])
        if a.poses == "slam":
            T_gt[fid_k] = sim3_matrix(compose_sim3(T[k], sg))
        else:
            Tg = ds.gt_pose(fid_k)
            if Tg is not None:
                T_gt[fid_k] = Tg
    print(f"[comp] poses={a.poses}, {len(T_gt)} candidate frames")

    dsm = None
    if a.dsm:
        if a.dsm_source == "da3":
            from stream_ortho.ortho.dsm import build_from_da3_depth

            src = a.da3_npz or str(run_dir.parent.parent / "stage-S12" / run_dir.name / "da3_depth_1008.npz")
            print(f"[comp] DSM from DA3 depth: {src}")
            dsm = build_from_da3_depth(ds, npz, sg, kfs, src, gsd=a.dsm_gsd)
        else:
            dsm = build_from_keyframes(ds, npz, sg, kfs, gsd=a.dsm_gsd)
        print(f"[comp] DSM {dsm.shape} {dsm.stats()}")

    # 3DGS map (current product path)
    gmap = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    for kf in kfs:
        gmap.ingest(plane_patch(ds, npz, kf, sg, stride=a.stride, shape="iso", dsm=dsm,
                                ray_dsm=bool(dsm)), kf)
        gmap.commit(kf)
    cloud = gmap.to_cloud()

    comp = OrthoCompositor(ds, gsd, z_ground=0.0, dsm=dsm,
                           cfg=CompositorConfig(max_frames=a.k_frames))
    # evaluation tiles: around the middle keyframe's nadir
    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S11" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2

    rows = []
    for n in range(a.tiles):
        i, j = ic + n, jc
        t0 = time.time()
        res = comp.compose(grid, i, j, sorted(T_gt.keys()), T_gt)
        dt = (time.time() - t0) * 1000.0
        if not res.get("ok"):
            continue
        # 3DGS render
        rgb3, alpha3, _, _ = render_tile(cloud, grid, i, j)
        # consensus from the candidate warps (nan-aware median over valid pixels)
        cands = comp.candidates(grid, i, j, sorted(T_gt.keys()), T_gt)
        if len(cands) < 2:
            continue
        stack = np.stack([c["rgb"] for c in cands], 0)
        valid = np.stack([c["valid"] for c in cands], 0)
        stack_m = np.where(valid[..., None], stack, np.nan)
        with np.errstate(all="ignore"):
            consensus = np.nanmedian(stack_m, axis=0)
        vmask = valid.any(0) & np.isfinite(consensus).all(axis=-1)
        if vmask.sum() < 20000:
            continue
        m3 = vmask & (alpha3 > 0.5)
        ps_3dgs = psnr(rgb3[m3], consensus[m3]) if m3.sum() > 5000 else float("nan")
        mc = vmask & res["valid"]
        ps_comp = psnr(res["rgb"][mc], consensus[mc]) if mc.sum() > 5000 else float("nan")
        kb = int(np.argmax([c["sharp"] for c in cands]))
        ps_single = psnr(stack[kb][mc], consensus[mc]) if mc.sum() > 5000 else float("nan")
        sh = {"3dgs": float(_grad(rgb3)[m3].mean()) if m3.sum() > 5000 else float("nan"),
              "comp": float(_grad(res["rgb"])[mc].mean()) if mc.sum() > 5000 else float("nan"),
              "single": float(_grad(stack[kb])[mc].mean()) if mc.sum() > 5000 else float("nan")}
        rows.append({"tile": [int(i), int(j)], "chosen": res["chosen"], "k": res["k"],
                     "psnr_3dgs": float(ps_3dgs), "psnr_comp": float(ps_comp),
                     "psnr_single": float(ps_single),
                     "sharp_3dgs": sh["3dgs"], "sharp_comp": sh["comp"],
                     "sharp_single": sh["single"], "ms": round(dt, 1),
                     "fids": res["fids"], "fid_best": res["fid_best"]})
        # 4-panel figure: consensus | 3DGS | compositor | |comp-3dgs|
        def vis(x):
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)
        gap = np.full((a.tile, 8, 3), 255, np.uint8)
        diff = np.clip(np.abs(res["rgb"] - rgb3).max(axis=2) * 8, 0, 1)
        diff = np.repeat((diff * 255).astype(np.uint8)[..., None], 3, axis=2)
        panels = [vis(consensus), vis(rgb3), vis(res["rgb"]), diff]
        cv2.imwrite(str(out / f"compare_{i}_{j}.png"),
                    np.concatenate([panels[0], gap, panels[1], gap, panels[2], gap,
                                    panels[3]], axis=1)[:, :, ::-1])
        print(f"  tile ({i},{j}) {res['chosen']:6s} k={res['k']}  "
              f"PSNR(consensus): 3dgs {ps_3dgs:5.2f} / comp {ps_comp:5.2f} / single {ps_single:5.2f}  "
              f"sharp 3dgs {sh['3dgs']:.1f} comp {sh['comp']:.1f} single {sh['single']:.1f}  {dt:.0f} ms")

    if not rows:
        print("[comp] no valid tile")
        return
    summary = {
        "run": run_dir.name, "seq": ds.name, "n_tiles": len(rows), "k_frames": a.k_frames,
        "dsm": bool(a.dsm), "stride": a.stride,
        "fallback_rate": float(np.mean([r["chosen"] == "single" for r in rows])),
        "psnr_3dgs": float(np.nanmean([r["psnr_3dgs"] for r in rows])),
        "psnr_comp": float(np.nanmean([r["psnr_comp"] for r in rows])),
        "psnr_single": float(np.nanmean([r["psnr_single"] for r in rows])),
        "sharp_3dgs": float(np.nanmean([r["sharp_3dgs"] for r in rows])),
        "sharp_comp": float(np.nanmean([r["sharp_comp"] for r in rows])),
        "sharp_single": float(np.nanmean([r["sharp_single"] for r in rows])),
        "never_worse_than_single": bool(all(r["psnr_comp"] >= r["psnr_single"] - 0.05 for r in rows)),
        "ms_per_tile": float(np.mean([r["ms"] for r in rows])),
        "rows": rows,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[comp] mean PSNR(consensus): 3dgs {summary['psnr_3dgs']:.2f} / "
          f"comp {summary['psnr_comp']:.2f} / single {summary['psnr_single']:.2f} dB; "
          f"fallback {100*summary['fallback_rate']:.0f}%; "
          f"sharp comp {summary['sharp_comp']:.1f} vs 3dgs {summary['sharp_3dgs']:.1f}; "
          f"{summary['ms_per_tile']:.0f} ms/tile -> {out}")


if __name__ == "__main__":
    main()
