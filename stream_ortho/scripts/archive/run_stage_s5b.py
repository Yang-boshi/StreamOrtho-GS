"""S5b: fixed budget + first end-to-end orthomosaic (design doc §7.6).

Builds the map from one sequence's cruise keyframes, optionally enforces a budget
with a scoring policy, renders an ``RxC`` tile block at a fixed GSD and compares
it with the GT-pose plane-warp reference mosaic.

The week-5 comparison (and the test of the user's edge idea)::

    --shape iso   --strategy append   # baseline, no budget
    --shape iso   --strategy conf     # budget by confidence
    --shape prior --strategy conf     # same retention, edge-aware *shapes*
    --shape prior --strategy edge     # user's idea: keep all edges + fill area

Outputs (``results/stage-S5b/<run>/<shape>_<strategy>/``):
    ortho.png, reference.png, coverage.png, compare.png, metrics.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr, ssim
from stream_ortho.geometry import R_to_quat_xyzw, sim3_from_vec8
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, seam_gradient_ratio, stitch


def build_parser():
    p = argparse.ArgumentParser(description="S5b: budgeted map -> orthomosaic")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--shape", default="prior", choices=["iso", "prior"])
    p.add_argument("--strategy", default="edge",
                   choices=["append", "conf", "voxel", "ortho", "edge", "random"])
    p.add_argument("--n-max", type=int, default=150000)
    p.add_argument("--grid", default="4x4")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--gsd", type=float, default=0.0, help="0 = native GSD of the centre KF")
    p.add_argument("--max-kf", type=int, default=12)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--voxel", type=float, default=0.4)
    p.add_argument("--radius-factor", type=float, default=1.0,
                   help="splat sigma in source-pixel footprints (1.0 with alpha normalisation)")
    p.add_argument("--no-alpha-norm", action="store_true")
    p.add_argument("--edge-low", type=int, default=40)
    p.add_argument("--edge-high", type=int, default=120)
    p.add_argument("--pick-textured", action="store_true",
                   help="choose the evaluation block where the reference has the most detail "
                        "(a uniform roof makes PSNR look good and the figure look empty)")
    p.add_argument("--register", action="store_true",
                   help="estimate per-keyframe 2-D ground offsets (phase correlation) and apply them")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=zcfg.DEVICE)
    return p


def cruise_keyframes(npz, n_max):
    T = npz["T_WC"]
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(len(T))
    cruise = np.where(np.abs(np.log(s / np.median(s))) < np.log(1.3))[0]
    if len(cruise) == 0:
        cruise = np.arange(len(T))
    return cruise[:n_max].tolist()


def edge_sharpness(ours, ref, mask, top: float = 0.10):
    """Mean |grad| of ``ours`` on the reference's strongest edges (1.0 = as sharp)."""
    import cv2

    g_o = cv2.cvtColor((np.clip(ours, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_r = cv2.cvtColor((np.clip(ref, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    go = np.hypot(cv2.Sobel(g_o, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g_o, cv2.CV_32F, 0, 1, 3))
    gr = np.hypot(cv2.Sobel(g_r, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g_r, cv2.CV_32F, 0, 1, 3))
    vals = gr[mask]
    if vals.size < 1000:
        return {"edge_sharpness": float("nan"), "ref_edge_grad": float("nan")}
    thr = np.percentile(vals, 100 * (1 - top))
    edge_mask = mask & (gr >= thr)
    sharp = float(go[edge_mask].mean())
    ref_s = float(gr[edge_mask].mean())
    return {"edge_sharpness": sharp, "ref_edge_grad": ref_s,
            "edge_ratio": sharp / max(ref_s, 1e-6)}


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"]
    tag = f"{a.shape}_{a.strategy}"
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S5b" / run_dir.name / tag
    out.mkdir(parents=True, exist_ok=True)

    import stream_ortho.map.build as _build
    import stream_ortho.ortho.rasterizer as _rz

    _build.SPLAT_RADIUS_FACTOR = float(a.radius_factor)
    _rz.ALPHA_NORMALISE = not a.no_alpha_norm
    ds = NPUDroneMapDataset(seq_dir)
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])),
                         [m["scale"]]])

    gmap = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2),
                       ds.T_WP, z_ground=0.0, device=a.device)
    kfs = cruise_keyframes(npz, a.max_kf)
    offsets = {}
    if a.register:
        from stream_ortho.map.register import estimate_offsets

        print("[S5b] estimating inter-frame ground registration ...")
        offsets = estimate_offsets(ds, npz, sg, kfs, verbose=True)
    print(f"[S5b] {run_dir.name} shape={a.shape} strategy={a.strategy} kfs={len(kfs)}")
    for kf in kfs:
        from stream_ortho.map.edges import EdgeConfig

        ecfg = EdgeConfig(canny_low=a.edge_low, canny_high=a.edge_high)
        off = offsets.get(kf, (0.0, 0.0))
        w_off = ds.T_WP[:3, :3] @ np.array([off[0], off[1], 0.0])
        gmap.ingest(plane_patch(ds, npz, kf, sg, stride=a.stride, shape=a.shape,
                                device=a.device, edge_cfg=ecfg,
                                world_offset=w_off if a.register else None), kf)
        gmap.commit(kf)
    n_before = gmap.n_committed
    budget = None
    if a.strategy != "append":
        budget = gmap.enforce_budget(a.n_max, a.strategy)
    print(f"[S5b] N before={n_before}  after budget={gmap.n_committed}  {budget}")

    # ---- mosaic -------------------------------------------------------
    rows, cols = (int(x) for x in a.grid.lower().split("x"))
    kf_center = kfs[len(kfs) // 2]
    T = npz["T_WC"]
    _, _, s_c = sim3_from_vec8(T[kf_center])
    depth_m = float(np.median(npz["X_canon"][kf_center][..., 2].astype(np.float32))) * s_c
    gsd = a.gsd or (depth_m / float(ds.K_orig[0, 0]))
    grid = OrthoGrid(ds.T_WP, gsd, tile=a.tile, halo=a.halo, z_ground=0.0, z_top=800.0)

    from stream_ortho.geometry import compose_sim3, sim3_matrix

    cam_w = sim3_matrix(compose_sim3(T[kf_center], sg))[:3, 3]
    p = (cam_w - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(p[0]), float(p[1]))
    if a.pick_textured:
        ref_scan = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
        best, best_score = (ic, jc), -1.0
        for di in range(-3, 4):
            for dj in range(-3, 4):
                rgb_s, cov_s, _, _ = ref_scan.render_tile(grid, ic + di, jc + dj, max_frames=1)
                if (cov_s > 0.5).mean() < 0.95:
                    continue
                g = rgb_s.mean(axis=2)
                score = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
                if score > best_score:
                    best, best_score = (ic + di, jc + dj), score
        ic, jc = best
        print(f"[S5b] textured block selected at ({ic},{jc}) score={best_score:.4f}")
    i0, j0 = ic - cols // 2, jc - rows // 2

    cloud = gmap.to_cloud()
    tiles = []
    for dj in range(rows):
        for di in range(cols):
            rgb, alpha, _, _ = render_tile(cloud, grid, i0 + di, j0 + dj)
            tiles.append(Tile(i0 + di, j0 + dj, rgb, alpha))
    mosaic, cov = stitch(tiles, shape=(rows, cols))

    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    ref_mosaic, ref_cov, n_ref = ref.render_block(grid, i0, j0, rows, cols, max_frames=1)

    both = (cov > 0.5) & (ref_cov > 0.5)
    ps = psnr(mosaic[both], ref_mosaic[both]) if both.sum() > 1000 else float("nan")
    if both.sum() > 1000:
        ys, xs = np.nonzero(both)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ss = ssim(mosaic[y0:y1, x0:x1], ref_mosaic[y0:y1, x0:x1])
    else:
        ss = float("nan")
    sharp = edge_sharpness(mosaic, ref_mosaic, both)
    # global sharpness (alignment-free): mean |grad| ratio over the covered area
    import cv2 as _cv2
    _go = np.hypot(_cv2.Sobel(_cv2.cvtColor((mosaic*255).astype(np.uint8), _cv2.COLOR_RGB2GRAY).astype(np.float32), _cv2.CV_32F, 1, 0, 3),
                   _cv2.Sobel(_cv2.cvtColor((mosaic*255).astype(np.uint8), _cv2.COLOR_RGB2GRAY).astype(np.float32), _cv2.CV_32F, 0, 1, 3))
    _gr = np.hypot(_cv2.Sobel(_cv2.cvtColor((ref_mosaic*255).astype(np.uint8), _cv2.COLOR_RGB2GRAY).astype(np.float32), _cv2.CV_32F, 1, 0, 3),
                   _cv2.Sobel(_cv2.cvtColor((ref_mosaic*255).astype(np.uint8), _cv2.COLOR_RGB2GRAY).astype(np.float32), _cv2.CV_32F, 0, 1, 3))
    sharp["global_sharpness"] = float(_go[both].mean()) if both.any() else float("nan")
    sharp["global_sharpness_ref"] = float(_gr[both].mean()) if both.any() else float("nan")
    sharp["global_sharpness_ratio"] = (sharp["global_sharpness"] / max(sharp["global_sharpness_ref"], 1e-6))
    seam = seam_gradient_ratio(mosaic, a.tile)
    holes = float(((cov < 0.5) & (ref_cov > 0.5)).sum() / max((ref_cov > 0.5).sum(), 1))

    import cv2

    def save(name, arr):
        cv2.imwrite(str(out / name), (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    save("ortho.png", mosaic)
    save("reference.png", ref_mosaic)
    save("coverage.png", np.repeat(cov[..., None], 3, axis=-1))
    gap = np.ones((mosaic.shape[0], 8, 3), np.float32)
    save("compare.png", np.concatenate([mosaic, gap, ref_mosaic, gap,
                                        np.repeat(cov[..., None], 3, axis=-1)], axis=1))

    stats = {
        "run": run_dir.name, "seq": ds.name, "shape": a.shape, "strategy": a.strategy,
        "n_max": a.n_max, "n_before_budget": n_before, "n_committed": gmap.n_committed,
        "budget": budget, "grid": [rows, cols], "tile": a.tile, "gsd_m_per_px": gsd,
        "coverage": float((cov > 0.5).mean()),
        "holes_vs_reference": holes,
        "psnr_vs_reference": ps, "ssim_vs_reference": ss,
        "edge": sharp, "seam_ratio": seam["ratio"],
        "memory_mb": round(gmap.memory_mb(), 2),
        "edge_ratio_map": float(gmap.committed["kind"].mean()) if len(gmap.committed["kind"]) else 0.0,
        "n_edge_map": int(gmap.committed["kind"].sum()),
        "reference_frames_per_tile": n_ref,
    }
    (out / "metrics.json").write_text(json.dumps(stats, indent=2))
    print(f"[S5b] {tag}: N={stats['n_committed']} PSNR={ps:.2f} SSIM={ss:.3f} "
          f"edge={sharp.get('edge_ratio', float('nan')):.3f} "
          f"gsharp={sharp.get('global_sharpness_ratio', float('nan')):.3f} "
          f"holes={100*holes:.1f}% seam={seam['ratio']:.2f} mem={stats['memory_mb']:.0f}MB")
    return stats


if __name__ == "__main__":
    main()
