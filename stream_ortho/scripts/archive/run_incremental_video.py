"""Incremental 3DGS map over a whole sequence, with a video + point-cloud export.

For each keyframe **in flight order**:

1. build its Gaussians (`head_adapt_patch`: per-pixel depth + ortho adapter),
2. `ingest` + `commit` into the streaming map (voxel dedup + merging + budget),
3. render a fixed 2x2-tile block around the middle keyframe's nadir,

and every step is written as one video frame (ortho + statistics panel), so the video
shows the incremental map growth: N gaussians, merge hits, conflicts, coverage.

At the end the final map is exported for inspection:

* ``cloud.npz``  -- means / colors / scales / opacity (the renderable data),
* ``cloud.ply``  -- same, binary little endian (inspection only; the **main path never
  writes a .ply**, see design doc §2.5).

Usage::

    python -m stream_ortho.scripts.run_incremental_video --run <run.json dir> --out <dir>
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
from stream_ortho.frontend.predict import load_model
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch, plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.tiles import stitch


def save_ply(path: pathlib.Path, means: np.ndarray, colors: np.ndarray,
             scales: np.ndarray, opacity: np.ndarray):
    """Binary little-endian PLY of the Gaussians (inspection export only)."""
    n = len(means)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float scale\nproperty float alpha\n"
        "end_header\n"
    )
    c = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                             ("scale", "<f4"), ("alpha", "<f4")])
    rec["x"], rec["y"], rec["z"] = means[:, 0], means[:, 1], means[:, 2]
    rec["red"], rec["green"], rec["blue"] = c[:, 0], c[:, 1], c[:, 2]
    rec["scale"] = np.asarray(scales).max(axis=1)
    rec["alpha"] = opacity
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--fusion", default="weighted")
    p.add_argument("--depth-source", default="head",
                   choices=["head", "pointmap", "dsm", "fused"])
    p.add_argument("--sigma-m", type=float, default=0.0,
                   help="splat sigma in metres (0 = the old stride*d/fx rule)")
    p.add_argument("--budget-every", type=int, default=1,
                   help="enforce the point budget every K keyframes (1 = every frame)")
    p.add_argument("--edge-reject", type=int, default=0,
                   help="depth-discontinuity band width (A)")
    p.add_argument("--edge-opacity", type=float, default=0.25,
                   help="band splats keep this opacity fraction (A')")
    p.add_argument("--render-every", type=int, default=1,
                   help="re-render every K frames and reuse the last mosaic otherwise; "
                        "ingestion ALWAYS happens (E')")
    p.add_argument("--column-filter", type=float, default=0.0,
                   help="P1-4: drop splats more than this many metres below their "
                        "column's top surface (0 = off)")
    p.add_argument("--mv-band", action="store_true",
                   help="delete a band splat only where the cell has >=2 clean views")
    p.add_argument("--async-worker", action="store_true",
                   help="run ingest/commit on a background thread (F): the renderer "
                        "always uses the latest published snapshot")
    p.add_argument("--nfp", action="store_true",
                   help="enable the nadir free-space prior veto (map/freespace.py)")
    p.add_argument("--nfp-cell", type=float, default=1.0)
    p.add_argument("--nfp-eps", type=float, default=1.0)
    p.add_argument("--collapse-every", type=int, default=0,
                   help="run the 2.5D cell selection every K keyframes (0 = never)")
    p.add_argument("--cell", type=float, default=0.45)
    p.add_argument("--cell-prefer", default="quality",
                   choices=["top", "quality", "column"],
                   help="'top' = argmax z (what caused the blocky colour patchwork), "
                        "'quality' = conf/opacity, 'column' = robust top surface (P0-2)")
    p.add_argument("--opacity-floor", type=float, default=0.0,
                   help="raise splat opacity to this value (product wants ~1)")
    p.add_argument("--tiles", default="2x2")
    p.add_argument("--max-kf", type=int, default=0, help="0 = all keyframes")
    p.add_argument("--dsm-kf", type=int, default=20, help="keyframes used to build the DSM")
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--fixed-extent", action="store_true",
                   help="render every frame on ONE grid computed from the whole sequence "
                        "(no per-frame bbox rescaling -> no apparent drift)")
    p.add_argument("--full-scene", action="store_true",
                   help="render the whole map bbox every frame (coarse gsd) instead of a fixed block")
    p.add_argument("--full-gsd", type=float, default=0.45)
    p.add_argument("--video-size", type=int, default=1024,
                   help="long edge of the written video frame")
    p.add_argument("--max-points", type=int, default=0,
                   help="fixed budget N_max (0 = unbounded, which grows O(N^2) in time)")
    p.add_argument("--budget-strategy", default="ortho3")
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    # flight order: keep the keyframes in the order they were captured
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()
    if a.max_kf:
        kfs = kfs[: a.max_kf]
    print(f"[vid] {ds.name}: {len(kfs)} keyframes in flight order")

    mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth / float(ds.K_orig[0, 0])
    grid = OrthoGrid(ds.T_WP, gsd, tile=512, halo=32, z_ground=0.0, z_top=800.0)
    grid_full = (OrthoGrid(ds.T_WP, a.full_gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
                 if a.full_scene else None)
    fixed_tiles = None
    if a.fixed_extent and grid_full is not None:
        # union of the ground points of **all** keyframes -> one extent for the video
        XP = []
        for kf in kfs:
            R_, t_, s_ = sim3_from_vec8(compose_sim3(T[kf], sg))
            X = np.asarray(npz["X_canon"][kf], np.float32)[::8, ::8].reshape(-1, 3)
            Pw = float(s_) * (X @ R_.T) + t_
            XP.append((Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3])
        XP = np.concatenate(XP)
        XP = XP[(XP[:, 2] > -20.0) & (XP[:, 2] < 2.0)]
        fi0, fj0 = grid_full.tile_of(float(XP[:, 0].min()), float(XP[:, 1].min()))
        fi1, fj1 = grid_full.tile_of(float(XP[:, 0].max()), float(XP[:, 1].max()))
        fixed_tiles = [(ii, jj) for jj in range(int(fj0), int(fj1) + 1)
                       for ii in range(int(fi0), int(fi1) + 1)]
        print(f"[vid] fixed extent: {len(fixed_tiles)} tiles "
              f"({int(fi0)}..{int(fi1)}, {int(fj0)}..{int(fj1)})")
    dsm_kfs = kfs if a.dsm_kf <= 0 or a.dsm_kf >= len(kfs) else kfs[:: max(1, len(kfs) // a.dsm_kf)]
    print(f"[vid] building DSM from {len(dsm_kfs)}/{len(kfs)} keyframes ...")
    dsm = build_piecewise_dsm(ds, npz, sg, dsm_kfs, gsd=0.5)
    rows, cols = (int(x) for x in a.tiles.lower().split("x"))
    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))
    i0, j0 = ic - cols // 2, jc - rows // 2
    print(f"[vid] block tiles ({i0},{j0}) {cols}x{rows}, gsd {gsd:.3f} m/px, DSM {dsm.shape}")

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S21" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    gmap = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2, fusion=a.fusion,
                                 max_points=a.max_points,
                                 budget_strategy=a.budget_strategy), ds.T_WP)
    if a.nfp:
        from stream_ortho.map.freespace import NFPConfig, NadirHeightField

        gmap.nfp = NadirHeightField(NFPConfig(cell=a.nfp_cell, eps=a.nfp_eps), ds.T_WP)
        print(f"[vid] NFP on (cell {a.nfp_cell} m, eps {a.nfp_eps} m)")
    import cv2

    video_path = out / "incremental.mp4"
    vw = None
    # ---- F: async map worker ----
    # Ingest/commit is ~1.1 s of numpy work; rendering is ~0.2 s of GPU work.  Running
    # them on separate threads lets the GPU render overlap the (GIL-releasing) numpy
    # passes instead of serialising them.  No keyframe is ever dropped: the queue has
    # room for every patch and the worker drains it in order.
    import queue as _queue
    import threading

    worker_q = _queue.Queue()
    worker_err = []
    snapshot = {"cloud": None, "n": 0}

    def _worker():
        while True:
            item = worker_q.get()
            if item is None:
                break
            patch, kf = item
            try:
                gmap.ingest(patch, kf)
                gmap.commit(kf)
                if a.max_points and (a.budget_every <= 1 or
                                     (kf + 1) % a.budget_every == 0):
                    gmap.enforce_budget(a.max_points, a.budget_strategy)
                snapshot["cloud"] = gmap.to_cloud()
                snapshot["n"] = int(gmap.n_committed)
            except Exception as exc:                     # surface it in the main thread
                worker_err.append(exc)
                break

    worker = threading.Thread(target=_worker, daemon=True) if a.async_worker else None
    if worker is not None:
        worker.start()
        print("[vid] async worker started (ingest/commit off the main thread)")

    t_all = time.time()
    stats = []
    for n, kf in enumerate(kfs):
        t0 = time.perf_counter()
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 dsm=dsm, depth_source=a.depth_source,
                                 sigma_m=a.sigma_m, opacity_min=a.opacity_floor,
                                 edge_reject=a.edge_reject, edge_opacity=a.edge_opacity,
                                 device=zcfg.DEVICE)
        t_patch = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        # E': every keyframe enters the map (skipping ingestion lost 41.5% of the
        # content in the S45 experiment); --render-every only throttles rendering.
        if a.column_filter > 0:
            from stream_ortho.map.selection import column_top_filter

            ref = None
            if gmap.n_committed > 50000:
                lv = gmap.committed["dead"] == 0 if "dead" in gmap.committed else slice(None)
                ref = gmap.committed["means"][lv][::7]      # subsampled reference
            kf_keep = column_top_filter(patch.means, ds.T_WP, ref, cell=a.cell,
                                        h_tol=a.column_filter)
            if not kf_keep.all():
                patch = patch.subset(kf_keep)
        if a.mv_band:
            from stream_ortho.map.selection import mv_band_filter

            kp = mv_band_filter(patch.means, patch.opacity, ds.T_WP, cell=a.cell,
                                min_support=2)
            if not kp.all():
                patch = patch.subset(kp)
        if worker is not None:
            worker_q.put((patch, kf))
            if worker_err:
                raise worker_err[0]
        else:
            gmap.ingest(patch, kf)
            gmap.commit(kf)
        if a.collapse_every and (n + 1) % a.collapse_every == 0:
            gmap.collapse_cells(grid, a.cell, prefer=a.cell_prefer)
        if worker is None and a.max_points and \
                (a.budget_every <= 1 or (n + 1) % a.budget_every == 0):
            gmap.enforce_budget(a.max_points, a.budget_strategy)
        t_map = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        if worker is not None:
            worker_q.join()
            cloud = snapshot["cloud"] if snapshot["cloud"] is not None else gmap.to_cloud()
        else:
            cloud = gmap.to_cloud()
        from stream_ortho.ortho.tiles import Tile

        if a.fixed_extent and fixed_tiles is not None:
            tiles = []
            for (ii, jj) in fixed_tiles:
                rgb, alpha, _, _ = render_tile(cloud, grid_full, ii, jj)
                tiles.append(Tile(ii, jj, rgb, alpha))
            ii_min = min(t[0] for t in fixed_tiles); jj_min = min(t[1] for t in fixed_tiles)
            ii_max = max(t[0] for t in fixed_tiles); jj_max = max(t[1] for t in fixed_tiles)
            mosaic, cov = stitch(tiles, shape=(jj_max - jj_min + 1, ii_max - ii_min + 1))
        elif a.full_scene and gmap.n_committed > 50000:
            # whole-scene view: enumerate the tiles covering the map bbox at a coarse gsd
            P = (gmap.committed["means"] - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
            fi0, fj0 = grid_full.tile_of(float(P[:, 0].min()), float(P[:, 1].min()))
            fi1, fj1 = grid_full.tile_of(float(P[:, 0].max()), float(P[:, 1].max()))
            tiles = []
            for jj in range(min(fj0, fj1), max(fj0, fj1) + 1):
                for ii in range(min(fi0, fi1), max(fi0, fi1) + 1):
                    rgb, alpha, _, _ = render_tile(cloud, grid_full, ii, jj)
                    tiles.append(Tile(ii, jj, rgb, alpha))
            rr = max(fj0, fj1) - min(fj0, fj1) + 1
            cc = max(fi0, fi1) - min(fi0, fi1) + 1
            mosaic, cov = stitch(tiles, shape=(rr, cc))
        else:
            tiles = []
            for dj in range(rows):
                for di in range(cols):
                    rgb, alpha, _, _ = render_tile(cloud, grid, i0 + di, j0 + dj)
                    tiles.append(Tile(i0 + di, j0 + dj, rgb, alpha))
            mosaic, cov = stitch(tiles, shape=(rows, cols))
        t_render = (time.perf_counter() - t2) * 1000

        img = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
        if a.full_scene:
            f = float(a.video_size) / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)),
                             interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 54), (0, 0, 0), -1)
        cv2.putText(img, f"keyframe {n+1}/{len(kfs)}  (frame_id {int(npz['frame_id'][kf])})",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"N={len(cloud):,}  merges={gmap.n_merge_hits:,}  "
                         f"conflicts={gmap.n_merge_conflicts:,}  cov={100*cov.mean():.0f}%",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        cv2.imwrite(str(frames_dir / f"{n:04d}.png"), img)
        if vw is None:
            vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (img.shape[1], img.shape[0]))
        vw.write(img)
        stats.append({"n": n + 1, "frame_id": int(npz["frame_id"][kf]), "n_gauss": int(len(cloud)),
                      "merge_hits": int(gmap.n_merge_hits),
                      "conflicts": int(gmap.n_merge_conflicts),
                      "coverage": float(cov.mean()),
                      "ms_patch": round(t_patch, 1), "ms_map": round(t_map, 1),
                      "ms_render": round(t_render, 1)})
        if (n + 1) % 5 == 0 or n + 1 == len(kfs):
            print(f"[vid] {n+1:3d}/{len(kfs)}  N={len(cloud):,}  "
                  f"patch {t_patch:.0f} ms / map {t_map:.0f} ms / render {t_render:.0f} ms")
    if vw is not None:
        vw.release()

    # ---- export the final cloud ----
    c = gmap.committed
    # only the **live** points: after the streaming cell selection the committed arrays
    # still contain the collapsed-away rows (``dead == 1``), and saving them made the
    # exported cloud 8 M instead of ~3.1 M and wrecked the product render
    live = (c["dead"] == 0) if "dead" in c else np.ones(len(c["means"]), bool)
    if worker is not None:
        worker_q.put(None)
        worker.join(timeout=10)
    np.savez_compressed(out / "cloud.npz", means=c["means"][live], colors=c["colors"][live],
                        scales=c["scales"][live], opacity=c["opacity"][live],
                        quats=c["quats"][live], n_obs=c["n_obs"][live])
    print(f"[vid] saved {int(live.sum()):,} live gaussians (of {len(c['means']):,} committed)")
    # inspection export goes *outside* results/ so the "no .ply on the main path"
    # guard (tests/test_gaussian_head.py) keeps its meaning
    ply_dir = zcfg.RESULTS_DIR.parent / "exports"
    ply_dir.mkdir(parents=True, exist_ok=True)
    save_ply(ply_dir / f"{run_dir.name}_cloud.ply", c["means"], c["colors"],
             c["scales"], c["opacity"])
    (out / "stats.json").write_text(json.dumps({"seq": ds.name, "n_kf": len(kfs),
                                                "frames": stats}, indent=2))
    print(f"[vid] -> {video_path} ({len(kfs)} frames), cloud.npz ({gmap.n_committed:,} gaussians), "
          f"cloud.ply; total {time.time()-t_all:.0f} s")


if __name__ == "__main__":
    main()
