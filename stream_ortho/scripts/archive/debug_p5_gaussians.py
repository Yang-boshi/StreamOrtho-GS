#!/usr/bin/env python3
"""Debug script to inspect Gaussian positions."""
import json
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8
from stream_ortho.adapters.splatt3r import mast3r_anchored_gaussians

run_dir = zcfg.RESULTS_DIR / "system" / "e2e_calib"
kf_idx = 10

# Load dataset
info = json.loads((run_dir / "run.json").read_text())
ds = NPUDroneMapDataset(zcfg.DATA_DIR / info["dataset"])

# Load keyframe
npz = np.load(run_dir / "keyframes.npz")
frame_id = int(npz["frame_id"][kf_idx])
T_WC = npz["T_WC"][kf_idx]
X_canon = npz["X_canon"][kf_idx]
img_u8 = ds.read_img(frame_id)

print(f"Frame {frame_id}, X_canon shape: {X_canon.shape}")
print(f"X_canon stats (camera frame):")
print(f"  X: [{X_canon[..., 0].min():.2f}, {X_canon[..., 0].max():.2f}]")
print(f"  Y: [{X_canon[..., 1].min():.2f}, {X_canon[..., 1].max():.2f}]")
print(f"  Z (depth): [{X_canon[..., 2].min():.2f}, {X_canon[..., 2].max():.2f}]")

# Generate Gaussians
gaussians = mast3r_anchored_gaussians(
    img_u8=img_u8,
    X_canon=X_canon,
    sim3_vec=T_WC,
    T_WP=ds.T_WP,
    keyframe_id=frame_id,
    device="cuda:0",
)

print(f"\nGaussians (world frame): {gaussians.means.shape[0]} points")
means_np = gaussians.means.detach().cpu().numpy()
print(f"World coordinates:")
print(f"  X: [{means_np[:, 0].min():.2f}, {means_np[:, 0].max():.2f}]")
print(f"  Y: [{means_np[:, 1].min():.2f}, {means_np[:, 1].max():.2f}]")
print(f"  Z: [{means_np[:, 2].min():.2f}, {means_np[:, 2].max():.2f}]")

# Transform to plane frame
means_homo = np.concatenate([means_np, np.ones((means_np.shape[0], 1))], axis=1)
T_PW = np.linalg.inv(ds.T_WP)
means_plane = (T_PW @ means_homo.T).T[:, :3]
print(f"\nPlane coordinates:")
print(f"  X: [{means_plane[:, 0].min():.2f}, {means_plane[:, 0].max():.2f}]")
print(f"  Y: [{means_plane[:, 1].min():.2f}, {means_plane[:, 1].max():.2f}]")
print(f"  Z: [{means_plane[:, 2].min():.2f}, {means_plane[:, 2].max():.2f}]")

# Camera position
R, t, s = sim3_from_vec8(T_WC)
T_WC_mat = np.eye(4)
T_WC_mat[:3, :3] = R * s
T_WC_mat[:3, 3] = t
cam_world = T_WC_mat[:3, 3]
cam_plane_homo = T_PW @ np.append(cam_world, 1.0)
cx, cy, cz = cam_plane_homo[:3]

print(f"\nCamera position (plane): ({cx:.2f}, {cy:.2f}, {cz:.2f})")
print(f"Tile center: ({cx:.2f}, {cy:.2f})")
print(f"Z_max: {means_plane[:, 2].max():.2f}")
print(f"Camera height above z_max: {cz - means_plane[:, 2].max():.2f}m")

# Check colors
colors_np = gaussians.sh_dc.detach().cpu().numpy()
C0 = 0.28209479177387814
rgb = (colors_np * C0 + 0.5).clip(0, 1)
print(f"\nColors (RGB after SH conversion):")
print(f"  R: [{rgb[:, 0].min():.3f}, {rgb[:, 0].max():.3f}], mean={rgb[:, 0].mean():.3f}")
print(f"  G: [{rgb[:, 1].min():.3f}, {rgb[:, 1].max():.3f}], mean={rgb[:, 1].mean():.3f}")
print(f"  B: [{rgb[:, 2].min():.3f}, {rgb[:, 2].max():.3f}], mean={rgb[:, 2].mean():.3f}")

# Check opacity
opacity_np = gaussians.opacity.detach().cpu().numpy().ravel()
print(f"\nOpacity: [{opacity_np.min():.3f}, {opacity_np.max():.3f}], mean={opacity_np.mean():.3f}")

# Check scales
scales_np = gaussians.scales.detach().cpu().numpy()
print(f"\nScales (log-space):")
print(f"  X: [{scales_np[:, 0].min():.3f}, {scales_np[:, 0].max():.3f}], mean={scales_np[:, 0].mean():.3f}")
print(f"  Y: [{scales_np[:, 1].min():.3f}, {scales_np[:, 1].max():.3f}], mean={scales_np[:, 1].mean():.3f}")
print(f"  Z: [{scales_np[:, 2].min():.3f}, {scales_np[:, 2].max():.3f}], mean={scales_np[:, 2].mean():.3f}")
