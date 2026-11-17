#!/usr/bin/env python3
"""P5.1a Task 3: single keyframe integration test.

image -> MASt3r-anchored Gaussians -> gsplat ORTHO render -> PNG + metrics

Conventions fixed after the numeric audit of 2026-11-16 (all three were render-black bugs):

1. ``camera_model="ortho"`` in gsplat (``u = fx * x``); "pinhole" divides by the
   camera-space z and collapses the whole keyframe into a few centre pixels.
2. SH DC = RGB2SH(full-resolution original image) + the head's residual -- the
   head's own ``sh`` output is only a small residual (+-0.08 -> flat grey).
3. The head's ``scales`` are already metric in point-map units; they are NOT
   rescaled by depth/fx.

The default GSD and the Gaussian scale floor are derived from the data (the
no-calib reconstruction is monocular -- "0.2 m/px" is meaningless on it):
    gsd    = median(depth) / fx_eff * sim3_s      (one source pixel, in plane units)
    floor  = sqrt(extent_area / N) * --floor-mul  (so the splats tile densely)

Usage (from the Ortho-3dgs repo root, env 2dgs-slam):
    PYTHONPATH=third_party/MASt3R-SLAM:third_party/MASt3R-SLAM/thirdparty/mast3r \
    python -m stream_ortho.scripts.test_p5_single_kf \
        --run results/system/p48_nocalib --kf 10
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.resolve()))

import argparse
import json
import time

import cv2
import numpy as np
import torch

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8

from stream_ortho.adapters.splatt3r import mast3r_anchored_gaussians, Gaussians
from stream_ortho.ortho.gsplat_ortho import render_ortho_tile


def load_keyframe_data(run_dir, ds, kf_idx):
    """Load keyframe data (image, X_canon, Sim3, frame_id)."""
    npz = np.load(run_dir / "keyframes.npz")

    frame_id = int(npz["frame_id"][kf_idx])
    T_WC = npz["T_WC"][kf_idx]
    X_canon = npz["X_canon"][kf_idx]

    img_u8 = ds.read_img(frame_id)

    sim3_align_path = run_dir / "sim3_est_to_gt.json"
    if sim3_align_path.exists():
        m = json.loads(sim3_align_path.read_text())
        sim3_gt = np.concatenate([
            m["translation"],
            R_to_quat_xyzw(np.asarray(m["rotation"])),
            [m["scale"]]
        ])
        T_WC_final = compose_sim3(T_WC, sim3_gt)
    else:
        T_WC_final = T_WC

    return {
        "frame_id": frame_id,
        "img_u8": img_u8,
        "X_canon": X_canon,
        "T_WC": T_WC_final,
        "T_WP": ds.T_WP,
    }


def effective_fx(X_canon) -> float:
    """Least-squares focal of the point map vs the pixel grid (see predict.fit_effective_intrinsics)."""
    from stream_ortho.frontend.predict import fit_effective_intrinsics
    K_guess = np.array([[0.5 * X_canon.shape[1], 0, 0.5 * X_canon.shape[1]],
                        [0, 0.5 * X_canon.shape[0], 0.5 * X_canon.shape[0]],
                        [0, 0, 1.]])
    X_t = torch.as_tensor(X_canon, dtype=torch.float32)
    eff = fit_effective_intrinsics(X_t, K_guess)
    return float(eff["x"]["f_eff"])


def compute_edge_sharpness(img_u8, mask=None):
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    if mask is not None:
        if magnitude[mask].size < 100:
            return {"mean": float("nan"), "p95": float("nan"), "n": int(mask.sum())}
        mag = magnitude[mask]
    else:
        mag = magnitude.ravel()
    return {"mean": float(mag.mean()), "p95": float(np.percentile(mag, 95)), "n": int(mag.size)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Run directory (contains keyframes.npz)")
    p.add_argument("--kf", type=int, default=10, help="Keyframe index to test (default: 10)")
    p.add_argument("--out", default=None, help="Output directory (default results/system/phase2_p5_3dgs/p5.1a_single_kf)")
    p.add_argument("--gsd", type=float, default=0.0, help="GSD in plane units; 0 = auto (one source pixel)")
    p.add_argument("--floor-mul", type=float, default=1.0,
                   help="Gaussian scale floor multiplier on sqrt(extent_area/N); 0 disables")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        return 1

    out_dir = pathlib.Path(a.out) if a.out else (
        zcfg.RESULTS_DIR / "system" / "phase2_p5_3dgs" / "p5.1a_single_kf"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((run_dir / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.DATA_DIR / info["dataset"])

    print(f"[test] Dataset: {ds.name}  kf index {a.kf}")
    kf_data = load_keyframe_data(run_dir, ds, a.kf)
    print(f"[test] frame_id {kf_data['frame_id']}  img {kf_data['img_u8'].shape}  X_canon {kf_data['X_canon'].shape}")

    # adapter
    t0 = time.time()
    gaussians = mast3r_anchored_gaussians(
        img_u8=kf_data["img_u8"], X_canon=kf_data["X_canon"], sim3_vec=kf_data["T_WC"],
        T_WP=kf_data["T_WP"], keyframe_id=kf_data["frame_id"], device=a.device,
    )
    t_adapter = time.time() - t0
    print(f"[test] Adapter: {gaussians.means.shape[0]} Gaussians in {t_adapter:.2f}s")

    # scene extent / centre in the plane frame
    means_np = gaussians.means.detach().cpu().numpy()
    T_PW = np.linalg.inv(kf_data["T_WP"])
    means_plane = (T_PW @ np.concatenate([means_np, np.ones((len(means_np), 1))], 1).T).T[:, :3]
    cx, cy = means_plane[:, 0].mean(), means_plane[:, 1].mean()
    z_max = float(means_plane[:, 2].max())
    ext = means_plane[:, :2].max(0) - means_plane[:, :2].min(0)

    # auto GSD: one source pixel expressed in plane units
    _, _, s3 = sim3_from_vec8(kf_data["T_WC"])
    src_gsd = float(np.median(kf_data["X_canon"][..., 2])) / effective_fx(kf_data["X_canon"]) * float(s3)
    gsd = float(a.gsd) if a.gsd > 0 else src_gsd
    scale_floor = float(np.sqrt(float(ext[0] * ext[1]) / len(means_np))) * a.floor_mul
    print(f"[test] extent {ext[0]:.2f}x{ext[1]:.2f}m  z_max {z_max:.2f}  centre ({cx:.1f},{cy:.1f})")
    print(f"[test] gsd {gsd:.5f} m/px ({gsd/src_gsd:.2f}x source)  scale floor {scale_floor:.5f} ({scale_floor/src_gsd:.2f} src px)")

    # render
    t0 = time.time()
    rgb, alpha, depth = render_ortho_tile(
        gaussians=gaussians, T_WP=kf_data["T_WP"], cx=cx, cy=cy, z_max=z_max,
        gsd=gsd, tile_size=a.tile_size, scale_floor=scale_floor, device=a.device,
    )
    t_render = time.time() - t0
    am = alpha.squeeze().detach().cpu().numpy()
    rgb_u8 = (torch.clamp(rgb, 0, 1) * 255).cpu().numpy().astype(np.uint8)

    # metrics on SOLID content only (alpha > 0.95): holes would fake a black image
    ys, xs = np.where(am > 0.95)
    solid = (am > 0.95)
    edge = compute_edge_sharpness(rgb_u8, solid)

    print(f"[test] Render {rgb.shape} in {t_render:.2f}s")
    print(f"[test] alpha>0.95 frac {solid.mean():.3f}  alpha med {np.median(am):.3f}")
    print(f"[test] RGB range [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"[test] Edge sharpness (solid content): mean={edge['mean']:.2f} p95={edge['p95']:.2f}")

    cv2.imwrite(str(out_dir / f"test_kf{a.kf}.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"test_kf{a.kf}_alpha.png"), (am * 255).astype(np.uint8))

    metrics = {
        "keyframe_id": kf_data["frame_id"], "keyframe_idx": a.kf,
        "n_gaussians": int(gaussians.means.shape[0]),
        "time_adapter_s": t_adapter, "time_render_s": t_render,
        "gsd": gsd, "gsd_is_source_px": a.gsd == 0, "scale_floor": scale_floor,
        "tile_size_px": a.tile_size,
        "rgb_range": [float(rgb.min()), float(rgb.max())],
        "alpha_solid_frac": float(solid.mean()), "alpha_median": float(np.median(am)),
        "edge_sharpness": edge,
        "scene_z_range": [float(means_plane[:, 2].min()), z_max],
    }
    with open(out_dir / f"test_kf{a.kf}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ok = (solid.mean() > 0.1 and rgb.max() > 0.2 and t_render < 5.0)
    print(f"[test] {'PASS' if ok else 'FAIL'} -> {out_dir / f'test_kf{a.kf}.png'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
