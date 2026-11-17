"""S4: render one orthographic tile / a tile block from Gaussian keyframe patches.

End-to-end for the P1 gate:

1. pick the keyframes whose Gaussian patches cover the requested area
   (cruise keyframes only, i.e. skip take-off / landing);
2. run the Gaussian head on each, **ray-constrain** the point map, align the
   depth scale to MASt3R-SLAM's point map, transform to the SLAM world and then
   to the GT world with the est->GT Sim3 from ``run_stage_s2_ate``;
3. estimate the ground level in the plane frame from the pooled clouds;
4. render ``rows x cols`` tiles at ``gsd`` with a halo, stitch them;
5. build the **GT-pose plane-warp reference** on the same grid and compare
   (PSNR / SSIM / cross-correlation shift / seam gradient).

Examples::

    # one 512 tile
    python -m stream_ortho.scripts.render_mosaic --run .../npu_calib --grid 1x1
    # 2x2 block, i.e. the P1 seam test
    python -m stream_ortho.scripts.render_mosaic --run .../npu_calib --grid 2x2
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.data.plane import world_to_plane
from stream_ortho.frontend import predict as P
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.geometry import compose_sim3, sim3_from_vec8, transform_gaussians_to_world
from stream_ortho.ortho.grid import OrthoGrid, gsd_from_altitude
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic, estimate_ground_level
from stream_ortho.ortho.tiles import Tile, seam_gradient_ratio, stitch


def build_parser():
    p = argparse.ArgumentParser(description="S4: orthographic tile(s) from Gaussian patches")
    p.add_argument("--run", required=True, help="stage-S2 run (keyframes.npz + sim3_est_to_gt.json)")
    p.add_argument("--seq", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--grid", default="1x1", help="RxC tiles, e.g. 2x2")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--gsd", type=float, default=0.0, help="metres/px (0 = native GSD of the centre keyframe)")
    p.add_argument("--gsd-ratio", type=float, default=1.0,
                   help=">1 renders finer than native (exposes the density gap)")
    p.add_argument("--n-kf", type=int, default=6, help="keyframes contributing Gaussians")
    p.add_argument("--kf", type=int, default=-1, help="centre keyframe (-1 = cruise middle)")
    p.add_argument("--anchor", choices=["head", "mast3r"], default="head")
    p.add_argument("--cloud", choices=["head", "plane"], default="plane",
                   help="head: feed-forward Gaussian patch transformed with the Sim3 chain; "
                        "plane: project the keyframe's pixels onto the ground plane "
                        "(design-doc fallback B, geometrically exact on flat ground)")
    p.add_argument("--no-align-depth", action="store_true",
                   help="skip the per-keyframe depth-scale alignment to the SLAM map")
    p.add_argument("--z-ground", default="cloud",
                   help="'cloud' (from the Gaussian cloud z-quantile), 'plane' (plane origin), "
                        "or a number (plane-frame z in metres)")
    p.add_argument("--ref-frames", type=int, default=3,
                   help="how many near-nadir images the reference blends (1 = sharpest)")
    p.add_argument("--device", default=zcfg.DEVICE)
    return p


# ---------------------------------------------------------------------------

def pick_keyframes(npz, kf: int, n_kf: int):
    """Centre keyframe + the ``n_kf-1`` cruise keyframes closest to it (SLAM world)."""
    T = npz["T_WC"]
    n = len(T)
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(n)
    cruise = np.where(np.abs(np.log(s / np.median(s))) < np.log(1.3))[0]
    if len(cruise) == 0:
        cruise = np.arange(n)
    if kf < 0:
        kf = int(cruise[len(cruise) // 2])
    pos = T[:, :3]
    order = [k for k in cruise if k != kf]
    order.sort(key=lambda k: float(np.linalg.norm(pos[k] - pos[kf])))
    return kf, [kf] + order[: max(0, n_kf - 1)]


def build_cloud(model, head, ds, npz, kf_list, sim3_gt, anchor="head",
                align_depth=True, device="cuda:0", verbose=True):
    """Gaussian cloud in the **GT world** frame from a set of keyframes."""
    clouds = []
    per_kf = []
    for k in kf_list:
        fid = int(npz["frame_id"][k])
        enc = P.encode_image(model, ds.read_img(fid), device=device)
        g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
        K = ds.K_frame()
        mu = P.constrain_to_rays(g["mu_cam"], K)
        mu_slam = P.constrain_to_rays(
            torch.as_tensor(npz["X_canon"][k], device=mu.device, dtype=torch.float32), K)
        k_align = 1.0
        if align_depth:
            k_align = float((mu_slam[..., 2].median() / mu[..., 2].median()).item())
        if anchor == "mast3r":
            mu = mu_slam * k_align if not align_depth else mu_slam
            k_align = 1.0
        else:
            mu = mu * k_align

        img = enc["uimg"].astype(np.float32) / 255.0
        colors = torch.as_tensor(img, device=device)
        scales = g["scales"] * k_align

        # camera -> SLAM world -> GT world
        T_gt_cam = compose_sim3(npz["T_WC"][k], sim3_gt)
        mu_w, q_w, sc_w = transform_gaussians_to_world(mu, g["quats"], scales, T_gt_cam, device=device)
        op = g["opacity"].reshape(-1).to(device)
        clouds.append(GaussianCloud(mu_w, q_w, sc_w, op, colors.reshape(-1, 3)))
        per_kf.append({
            "kf": int(k), "frame_id": fid, "k_align": round(k_align, 4),
            "depth_head_median": float(g["mu_cam"][..., 2].median()),
            "depth_slam_median": float(npz["X_canon"][k][..., 2].astype(np.float32).mean()),
            "n_gaussians": int(mu.reshape(-1, 3).shape[0]),
        })
        if verbose:
            print(f"    kf{k:3d} frame{fid:4d}  k_align={k_align:.3f}  "
                  f"depth head/slam={per_kf[-1]['depth_head_median']:.2f}/"
                  f"{per_kf[-1]['depth_slam_median']:.2f}")
    return GaussianCloud.concat(clouds), per_kf


def build_cloud_plane(ds, npz, kf_list, sim3_vec_gt, device="cuda:0", stride=2):
    """Fallback B: every keyframe pixel becomes a Gaussian on the ground plane.

    Ray (from ``K_frame``) x plane (``T_WP``) gives an exact planar footprint, so
    this cloud is *by construction* consistent with the plane-warp reference.
    Splat radius: half a source pixel footprint (``depth/f`` / 2).
    """
    from stream_ortho.data.plane import world_to_plane
    from stream_ortho.geometry import sim3_act

    K = ds.K_orig
    H, W = int(ds.cfg.height), int(ds.cfg.width)
    uu, vv = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0],
                     (vv - K[1, 2]) / K[1, 1],
                     np.ones_like(uu, dtype=np.float64)], axis=-1)
    T_WP = ds.T_WP
    n = T_WP[:3, 2]
    p0 = T_WP[:3, 3]

    clouds, per_kf = [], []
    for k in kf_list:
        fid = int(npz["frame_id"][k])
        img = ds.read_img(fid)
        import cv2
        small = (img[::stride, ::stride].astype(np.float32) / 255.0)
        # camera -> GT world (pose chain), same as the head path
        from stream_ortho.geometry import sim3_from_vec8, compose_sim3, transform_gaussians_to_world
        T_gt_cam = compose_sim3(npz["T_WC"][k], sim3_vec_gt)
        R, t, s = sim3_from_vec8(T_gt_cam)
        cam = t
        z_gt = float(((p0 - cam) @ n) / (np.array([0, 0, 1.0]) @ n))  # camera height above the plane origin
        depth = float(np.median(npz["X_canon"][k][..., 2].astype(np.float32))) * s
        d_world = (rays @ R.T)  # rays in GT world (camera +z is the optical axis)
        dirs = rays @ R.T
        denom = dirs @ n
        tt = ((p0 - cam) @ n) / denom
        ok = (tt > 0) & np.isfinite(tt)
        pts_w = cam + tt[..., None] * dirs
        colors = torch.as_tensor(small.reshape(-1, 3), device=device)
        mu = torch.as_tensor(pts_w.reshape(-1, 3), dtype=torch.float32, device=device)
        fp = max(depth / float(K[0, 0]) * stride, 1e-3)   # one (strided) pixel footprint
        fp *= 1.5                                          # splat radius: avoid gaps/darkening
        scales = torch.full_like(mu, fp)
        quats = torch.zeros((mu.shape[0], 4), device=device)
        quats[:, 0] = 1.0
        opac = torch.ones(mu.shape[0], device=device)
        clouds.append(GaussianCloud(mu[ok.reshape(-1)], quats[ok.reshape(-1)], scales[ok.reshape(-1)],
                                    opac[ok.reshape(-1)], colors[ok.reshape(-1)]))
        per_kf.append({"kf": int(k), "frame_id": fid, "depth_world": depth,
                       "footprint_m": fp * 2, "n_gaussians": int(ok.sum())})
        print(f"    [plane] kf{k:3d} frame{fid:4d} depth={depth:.1f} m  footprint={fp*2:.3f} m")
    return GaussianCloud.concat(clouds), per_kf


def px_correlation(a, b, valid):
    """Phase-correlation shift between two tiles (robust to border effects)."""
    import cv2

    ga = cv2.cvtColor((a * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor((b * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = valid.astype(np.float32)
    win = np.outer(np.hanning(ga.shape[0]), np.hanning(ga.shape[1])).astype(np.float32)
    w = m * win
    try:
        (dx, dy), resp = cv2.phaseCorrelate(ga * w, gb * w)
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    return {"dx_px": float(dx), "dy_px": float(dy), "response": float(resp),
            "valid": bool(resp > 0.05)}


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = pathlib.Path(args.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(args.seq) if args.seq else zcfg.dataset_root() / info["dataset"]
    rows, cols = (int(x) for x in args.grid.lower().split("x"))
    out = pathlib.Path(args.out) if args.out else (
        zcfg.RESULTS_DIR / "stage-S4" / run_dir.name / f"grid{rows}x{cols}")
    out.mkdir(parents=True, exist_ok=True)

    ds = NPUDroneMapDataset(seq_dir)
    npz = np.load(run_dir / "keyframes.npz")
    sim3_gt = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sim3_vec_gt = np.concatenate([sim3_gt["translation"],
                                  # rotation matrix -> quaternion (xyzw)
                                  _mat_to_quat(np.asarray(sim3_gt["rotation"])),
                                  [sim3_gt["scale"]]])

    kf_center, kf_list = pick_keyframes(npz, args.kf, args.n_kf)
    print(f"[S4] run={run_dir.name} seq={ds.name} kf={kf_center} cloud_kfs={kf_list}")

    if args.cloud == "plane":
        cloud, per_kf = build_cloud_plane(ds, npz, kf_list, sim3_vec_gt, args.device)
    else:
        model = P.load_model(device=args.device)
        head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=args.device, verbose=False)
        cloud, per_kf = build_cloud(model, head, ds, npz, kf_list, sim3_vec_gt,
                                    anchor=args.anchor, align_depth=not args.no_align_depth,
                                    device=args.device)

    # ---- ground level + grid ------------------------------------------
    T_WP = ds.T_WP
    pts_P = world_to_plane(T_WP, cloud.means.detach().cpu().numpy())
    if args.z_ground == "cloud":
        z_ground = estimate_ground_level([pts_P], percentile=5.0)
    elif args.z_ground == "plane":
        z_ground = 0.0
    else:
        z_ground = float(args.z_ground)

    T = npz["T_WC"]
    _, _, s_c = sim3_from_vec8(T[kf_center])
    Xc = npz["X_canon"][kf_center][..., 2].astype(np.float32)
    depth_m = float(np.median(Xc)) * s_c * float(sim3_gt["scale"])   # metres (metric after est->GT)
    # real GSD is metres per *original* pixel: altitude / fx_orig (1220 for phantom3)
    fx_src = float(ds.K_orig[0, 0] if args.cloud == "plane" else ds.K_frame()[0, 0])
    gsd_native = depth_m / fx_src
    gsd = args.gsd if args.gsd > 0 else gsd_native / max(args.gsd_ratio, 1e-6)
    print(f"[S4] source fx={fx_src:.1f}  native gsd={gsd_native:.4f} m/px  used gsd={gsd:.4f}")

    grid = OrthoGrid(T_WP=T_WP, gsd=gsd, tile=args.tile, halo=args.halo,
                     z_ground=z_ground, z_top=max(50.0, 3.0 * (depth_m + 1.0)))

    # tile block centred on the centre keyframe's ground position
    cam_gt = _cam_centre_gt(npz, sim3_vec_gt, kf_center)
    cam_P = world_to_plane(T_WP, cam_gt[None])[0]
    # tile containing the camera nadir, then shift so the block is centred on it
    ic, jc = grid.tile_of(float(cam_P[0]), float(cam_P[1]))
    i0, j0 = ic - cols // 2, jc - rows // 2
    print(f"[S4] gsd={gsd:.4f} m/px  z_ground={z_ground:.2f}  tiles i0={i0} j0={j0}  "
          f"extent={rows*grid.tile*gsd:.1f}x{cols*grid.tile*gsd:.1f} m  N={len(cloud)}")

    # ---- render + reference -------------------------------------------
    tiles, tile_metrics = [], {}
    for dj in range(rows):
        for di in range(cols):
            i, j = i0 + di, j0 + dj
            rgb, alpha, _, _ = render_tile(cloud, grid, i, j)
            tiles.append(Tile(i, j, rgb, alpha))
            tile_metrics[f"{i},{j}"] = {
                "coverage": float((alpha > 0.5).mean()),
                "alpha_mean": float(alpha.mean()),
            }

    mosaic, cov = stitch(tiles, shape=(rows, cols))
    ref = PlaneReferenceMosaic(ds, gsd, z_ground, stride=1)
    ref_mosaic, ref_cov, ref_frames = ref.render_block(grid, i0, j0, rows, cols,
                                                      max_frames=args.ref_frames)

    both = (cov > 0.5) & (ref_cov > 0.5)
    psnr = P.psnr(mosaic[both], ref_mosaic[both]) if both.sum() > 1000 else float("nan")
    ssim = P.ssim(mosaic[both], ref_mosaic[both]) if both.sum() > 1000 else float("nan")
    shift = px_correlation(mosaic, ref_mosaic, both)

    # seam test on our mosaic, and on the reference as a control
    seam_ours = seam_gradient_ratio(mosaic, grid.tile)
    seam_ref = seam_gradient_ratio(ref_mosaic, grid.tile)

    # ---- save ----------------------------------------------------------
    import cv2

    def save(name, arr):
        a = np.clip(arr, 0, 1)
        cv2.imwrite(str(out / name), (a * 255).astype(np.uint8)[:, :, ::-1])

    save("mosaic.png", mosaic)
    save("reference.png", ref_mosaic)
    save("coverage.png", np.repeat(cov[..., None], 3, axis=-1))
    gap = np.ones((mosaic.shape[0], 8, 3), np.float32)
    save("compare.png", np.concatenate([mosaic, gap, ref_mosaic, gap,
                                        np.repeat(cov[..., None], 3, axis=-1)], axis=1))
    for t in tiles:
        save(f"tile_{t.i}_{t.j}_rgb.png", t.rgb)
        save(f"tile_{t.i}_{t.j}_alpha.png", np.repeat(t.alpha[..., None], 3, axis=-1))

    metrics = {
        "run": run_dir.name,
        "seq": ds.name,
        "grid": [rows, cols],
        "tile": args.tile,
        "halo": args.halo,
        "gsd_m_per_px": gsd,
        "gsd_native": gsd_native,
        "gsd_ratio": args.gsd_ratio,
        "z_ground": z_ground,
        "z_ground_mode": str(args.z_ground),
        "ref_frames": args.ref_frames,
        "z_top": grid.z_top,
        "center_kf": int(kf_center),
        "cloud_kfs": [int(k) for k in kf_list],
        "n_gaussians": len(cloud),
        "anchor": args.anchor,
        "cloud": args.cloud,
        "align_depth": not args.no_align_depth,
        "cloud_stats": cloud.stats(),
        "per_kf": per_kf,
        "coverage": float((cov > 0.5).mean()),
        "reference_coverage": float((ref_cov > 0.5).mean()),
        "psnr_vs_reference": psnr,
        "ssim_vs_reference": ssim,
        "phase_correlation": shift,
        "seam_ours": seam_ours,
        "seam_reference": seam_ref,
        "seam_ratio": seam_ours["ratio"],
        "tiles": tile_metrics,
        "reference_frames_per_tile": ref_frames,
        "sim3_est_to_gt": sim3_gt,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[S4] coverage={metrics['coverage']:.3f}  PSNR={psnr:.2f} dB  SSIM={ssim:.3f}  "
          f"shift={shift}")
    print(f"[S4] seam ratio={seam_ours['ratio']:.2f} (ours) vs {seam_ref['ratio']:.2f} (reference)")
    print(f"[S4] -> {out}")
    return metrics


# ---------------------------------------------------------------------------

def _mat_to_quat(R):
    from stream_ortho.geometry import R_to_quat_xyzw
    return R_to_quat_xyzw(R)


def _cam_centre_gt(npz, sim3_vec_gt, kf):
    from stream_ortho.geometry import sim3_act
    return sim3_act(sim3_vec_gt, npz["T_WC"][kf][:3][None])[0]


if __name__ == "__main__":
    main()
