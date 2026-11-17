"""S3 stage (second half): one 1024x1024 orthographic tile from a keyframe patch.

Week-2 scope (design doc §7.3 items 3-4): this tile is a **coordinate / GSD /
orientation sanity check**, not the P1 gate.  The true `gsplat` ortho
rasteriser + seams + `result.png` comparison belong to week 3.

Since `phantom3-centralPark-kfs/result.png` is no longer available, the
reference is built from the *same keyframe* by warping it onto the ground plane
with the GT/estimated pose (`--ref-warp`).  That is exact for flat ground and
checks exactly what week 2 needs: plane origin, axis signs, GSD and tile
registration.  Buildings off the plane will disagree by design.

Pipeline:
    keyframe -> Gaussian head -> (optional) depth-scale alignment to the SLAM
    point map -> world (full Sim3, incl. per-keyframe scale) -> plane frame
    -> orthographic camera -> gsplat ortho rasterisation -> crop halo -> tile

Outputs (``results/stage-S3/<run>/ortho_tile/``):
    tile_rgb.png, tile_alpha.png, tile_depth.png, tile_reference.png,
    tile_compare.png, tile.json
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
from stream_ortho.geometry import (
    ortho_K,
    ortho_viewmat,
    sim3_act_inv,
    sim3_from_vec8,
    transform_gaussians_to_world,
)


def build_parser():
    p = argparse.ArgumentParser(description="S3: single orthographic tile")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--kf", type=int, default=-1, help="keyframe index (-1 = cruise middle)")
    p.add_argument("--out", default=None)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--anchor", choices=["head", "mast3r"], default="head")
    p.add_argument("--no-constrain-rays", action="store_true",
                   help="disable the pixel-ray constraint (ablation; default is ON)")
    p.add_argument("--align-mode", choices=["none", "slam", "area"], default="area",
                   help="depth-scale alignment of the head patch: none (k=1), "
                        "slam (match the SLAM point-map depth), "
                        "area (match the ground-plane footprint area; one extra render)")
    p.add_argument("--sh-mode", choices=["sh", "sh+image"], default="sh+image")
    p.add_argument("--gsd", type=float, default=0.0, help="override GSD [world units/px]")
    p.add_argument("--gsd-ratio", type=float, default=1.0,
                   help="sample the ground this many times finer than the source pixels "
                        "(1 = native GSD = depth/f); >1 exposes the density gap")
    p.add_argument("--color-mode", choices=["image", "sh", "sh+image"], default="image",
                   help="image: observed pixel colour (orthomosaic semantics); "
                        "sh: learned SH DC only; sh+image: Splatt3R's recipe")
    p.add_argument("--max-points", type=int, default=0, help="subsample Gaussians (0 = all)")
    p.add_argument("--device", default=zcfg.DEVICE)
    return p


def pick_keyframe(npz, kf: int) -> int:
    T = npz["T_WC"]
    n = len(T)
    if kf >= 0:
        return kf
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(n)
    cruise = np.where(np.abs(np.log(s / np.median(s))) < np.log(1.3))[0]
    if len(cruise) == 0:
        cruise = np.arange(n)
    return int(cruise[len(cruise) // 2])


def reference_warp(uimg, T_WC_vec, ds, T_WP, cx, cy, gsd, z_ground, H, W):
    """Warp the keyframe image onto the ground plane at the tile's GSD.

    For each tile pixel -> plane point (z = z_ground) -> camera pixel of the
    keyframe -> bilinear sample.  Returns ``(rgb, valid)``.
    """
    import cv2

    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    x_P = cx - 0.5 * W * gsd + uu * gsd
    y_P = cy + 0.5 * H * gsd - vv * gsd
    z_P = np.full_like(x_P, z_ground)
    p_P = np.stack([x_P, y_P, z_P], axis=-1).reshape(-1, 3)

    p_W = p_P @ T_WP[:3, :3].T + T_WP[:3, 3]
    p_C = sim3_act_inv(T_WC_vec, p_W)
    z = p_C[:, 2]
    K = ds.K_frame()
    uv = p_C @ K.T
    px = uv[:, 0] / np.maximum(uv[:, 2], 1e-9)
    py = uv[:, 1] / np.maximum(uv[:, 2], 1e-9)

    h, w = uimg.shape[:2]
    valid = (z > 1e-6) & (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
    map_x = np.where(valid, px, 0).astype(np.float32).reshape(H, W)
    map_y = np.where(valid, py, 0).astype(np.float32).reshape(H, W)
    rgb = cv2.remap(uimg.astype(np.float32) / 255.0, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rgb, valid.reshape(H, W)


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = pathlib.Path(args.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(args.seq) if args.seq else zcfg.dataset_root() / info["dataset"]
    out = pathlib.Path(args.out) if args.out else zcfg.RESULTS_DIR / "stage-S3" / run_dir.name / "ortho_tile"
    out.mkdir(parents=True, exist_ok=True)

    npz = np.load(run_dir / "keyframes.npz")
    X, T = npz["X_canon"], npz["T_WC"]
    fids = npz["frame_id"]
    Hf, Wf = int(npz["hw"][0]), int(npz["hw"][1])
    ds = NPUDroneMapDataset(seq_dir)
    k = pick_keyframe(npz, args.kf)
    fid = int(fids[k])
    T_WC = T[k]
    R, t, s = sim3_from_vec8(T_WC)

    model = P.load_model(device=args.device)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=args.device)

    enc = P.encode_image(model, ds.read_img(fid), device=args.device)
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    uimg = enc["uimg"]

    mu = g["mu_cam"]
    x_slam = torch.as_tensor(X[k], device=mu.device, dtype=torch.float32)
    reproj_raw = P.reprojection_error_px(mu, ds.K_frame())
    if not args.no_constrain_rays:
        mu = P.constrain_to_rays(mu, ds.K_frame())
        x_slam = P.constrain_to_rays(x_slam, ds.K_frame())
    if args.anchor == "mast3r":
        mu = x_slam
        k_align = 1.0
    elif args.align_mode == "slam":
        k_align = float((x_slam[..., 2].median() / mu[..., 2].median()).item())
    else:
        k_align = 1.0
    mu = mu * k_align

    scales_raw = g["scales"] * k_align
    # colours (before any subsampling so that shapes stay aligned)
    img_col = torch.as_tensor(uimg / 255.0, device=g["sh"].device, dtype=torch.float32)
    if args.color_mode == "image":
        rgb_full = img_col.clamp(0, 1)
    elif args.color_mode == "sh+image":
        rgb_full = P.sh_to_rgb(g["sh"] + P.rgb_to_sh(img_col))
    else:
        rgb_full = P.sh_to_rgb(g["sh"])

    if args.max_points and args.max_points < mu.reshape(-1, 3).shape[0]:
        # stride subsample (keeps the image grid layout)
        stride = int(np.ceil(np.sqrt(mu.reshape(-1, 3).shape[0] / args.max_points)))
        mu = mu[::stride, ::stride]
        scales_raw = scales_raw[::stride, ::stride]
        g["quats"] = g["quats"][::stride, ::stride]
        g["opacity"] = g["opacity"][::stride, ::stride]
        g["sh"] = g["sh"][::stride, ::stride]
        rgb_full = rgb_full[::stride, ::stride]

    rgb = rgb_full

    mu_w, q_w, sc_w = transform_gaussians_to_world(mu, g["quats"], scales_raw, T_WC, device=args.device)
    op_w = g["opacity"].reshape(-1).to(args.device)

    # ---- plane frame / tile geometry -----------------------------------
    T_WP = ds.T_WP
    mu_p = world_to_plane(T_WP, mu_w.cpu().numpy())
    z_ground = float(np.percentile(mu_p[:, 2], 5.0))
    cam_p = world_to_plane(T_WP, t[None])[0]
    cx, cy = float(cam_p[0]), float(cam_p[1])
    depth_cam = float(mu[..., 2].median()) * s          # world units
    gsd_native = depth_cam / float(ds.K_frame()[0, 0])
    gsd = args.gsd if args.gsd > 0 else gsd_native / max(args.gsd_ratio, 1e-6)
    z_top = float(np.percentile(mu_p[:, 2], 99.0) + 5.0 * gsd)
    W = H = args.tile + 2 * args.halo

    viewmat = torch.as_tensor(ortho_viewmat(T_WP, cx, cy, z_top), dtype=torch.float32,
                              device=args.device)[None]
    K_ortho = ortho_K(gsd, W, H)

    img, alpha, _ = P.render_ortho(mu_w, sc_w, q_w, op_w, rgb, viewmat, K_ortho, H, W)
    tile = img[0].clamp(0, 1).cpu().numpy()
    alpha_np = alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()
    h0 = args.halo
    tile_c = tile[h0:h0 + args.tile, h0:h0 + args.tile]
    alpha_c = alpha_np[h0:h0 + args.tile, h0:h0 + args.tile]

    ref, ref_valid = reference_warp(uimg, T_WC, ds, T_WP, cx, cy, gsd, z_ground,
                                    args.tile, args.tile)
    if args.align_mode == "area" and args.anchor == "head":
        # both regions are the same ground patch: match their areas
        a_rend = float((alpha_c > 0.5).sum())
        a_ref = float(ref_valid.sum())
        if a_rend > 100 and a_ref > 100:
            k_scale = float(np.sqrt(a_ref / a_rend))
            k_align *= k_scale
            mu = mu * k_scale
            scales_raw = scales_raw * k_scale
            mu_w, q_w, sc_w = transform_gaussians_to_world(
                mu, g["quats"], scales_raw, T_WC, device=args.device)
            depth_cam = float(mu[..., 2].median()) * s
            gsd_native = depth_cam / float(ds.K_frame()[0, 0])
            gsd = args.gsd if args.gsd > 0 else gsd_native / max(args.gsd_ratio, 1e-6)
            z_top = float(np.percentile(mu_p[:, 2], 99.0) + 5.0 * gsd)
            viewmat = torch.as_tensor(ortho_viewmat(T_WP, cx, cy, z_top),
                                      dtype=torch.float32, device=args.device)[None]
            K_ortho = ortho_K(gsd, W, H)
            img, alpha, _ = P.render_ortho(mu_w, sc_w, q_w, op_w, rgb, viewmat,
                                           K_ortho, H, W)
            tile = img[0].clamp(0, 1).cpu().numpy()
            alpha_np = alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()
            tile_c = tile[h0:h0 + args.tile, h0:h0 + args.tile]
            alpha_c = alpha_np[h0:h0 + args.tile, h0:h0 + args.tile]
            ref, ref_valid = reference_warp(uimg, T_WC, ds, T_WP, cx, cy, gsd,
                                            z_ground, args.tile, args.tile)

    mask = alpha_c > 0.05
    both = mask & ref_valid
    # density gap: holes *inside* the keyframe's true ground footprint
    holes = float(((alpha_c < 0.5) & ref_valid).sum() / max(ref_valid.sum(), 1))
    psnr = P.psnr(tile_c[both], ref[both]) if both.sum() > 100 else float("nan")
    ssim = P.ssim(tile_c[both], ref[both]) if both.sum() > 100 else float("nan")

    # translation check: phase correlation between tile and reference (grey)
    shift = None
    if both.sum() > 1000:
        import cv2

        g1 = cv2.cvtColor((tile_c * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g1 = cv2.GaussianBlur(g1, (5, 5), 0)
        g2 = cv2.GaussianBlur(g2, (5, 5), 0)
        try:
            (dx, dy), resp = cv2.phaseCorrelate(g1, g2)
            shift = {"dx_px": float(dx), "dy_px": float(dy), "response": float(resp)}
        except Exception as exc:  # pragma: no cover
            shift = {"error": str(exc)}

    import cv2

    def save(name, arr):
        a = np.clip(arr, 0, 1)
        cv2.imwrite(str(out / name), (a * 255).astype(np.uint8)[:, :, ::-1]
                    if a.ndim == 3 else (a * 255).astype(np.uint8))

    save("tile_rgb.png", tile_c)
    save("tile_alpha.png", np.repeat(alpha_c[..., None], 3, axis=-1))
    save("tile_reference.png", ref)
    gap = np.ones((args.tile, 8, 3), np.float32)
    save("tile_compare.png", np.concatenate(
        [tile_c, gap, ref, gap, np.repeat(alpha_c[..., None], 3, axis=-1)], axis=1))

    info_out = {
        "run": run_dir.name,
        "seq": ds.name,
        "keyframe": int(k),
        "frame_id": fid,
        "tile_size": args.tile,
        "halo": args.halo,
        "gsd_world_units_per_px": gsd,
        "gsd_native": gsd_native,
        "plane": {
            "cx_m": cx, "cy_m": cy, "z_ground_m": z_ground, "z_top_m": z_top,
            "T_WP": T_WP.tolist(),
        },
        "camera": {
            "world_t": t.tolist(), "sim3_scale": s,
            "depth_cam": depth_cam, "k_align": k_align,
            "align_mode": args.align_mode, "anchor": args.anchor,
        },
        "coverage_rendered": float((alpha_c > 0.5).mean()),
        "holes_inside_footprint": holes,
        "mean_alpha_inside_footprint": float(alpha_c[ref_valid].mean()) if ref_valid.any() else float("nan"),
        "coverage_reference": float(ref_valid.mean()),
        "psnr_vs_plane_warp": psnr,
        "ssim_vs_plane_warp": ssim,
        "phase_correlation": shift,
        "phase_correlation_valid": bool(shift and shift.get("response", 0) > 0.05),
        "n_gaussians": int(mu_w.shape[0]),
        "color_mode": args.color_mode,
        "constrain_rays": not args.no_constrain_rays,
        "reproj_raw_px": reproj_raw,
        "gsd_ratio": args.gsd_ratio,
    }
    (out / "tile.json").write_text(json.dumps(info_out, indent=2))
    print(f"[S3-tile] kf{k} frame{fid}: gsd={gsd:.4f} m/px  coverage={info_out['coverage_rendered']:.3f}  "
          f"PSNR(vs plane warp)={psnr:.2f} dB  SSIM={ssim:.3f}  shift={shift}")
    print(f"[S3-tile] -> {out}")


if __name__ == "__main__":
    main()
