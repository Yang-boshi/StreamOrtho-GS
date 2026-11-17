"""Splatt3r adapter: MASt3r-anchored 3D Gaussians for high-resolution orthophotos.

Core innovation (P5 design doc §2.1):
    means = X_canon (MASt3R-SLAM's fused point map) -- NOT head's predicted mu!
    appearance = Splatt3r head (scales, quats, opacity) + image colours

This decouples geometry (metric, stable) from appearance (detailed, per-frame).
The head was trained with mu=pts3d, but we override that prediction with the
canonical geometry to prevent drift and floaters.

Two facts about the released Splatt3R head that the first implementation got
wrong (both verified against ``splatt3r/main.py`` and the released ckpt):

* the head's ``sh`` output is a small RESIDUAL; the released model adds
  ``RGB2SH(original_img)`` on top (``learn_residual = True``), so the adapter
  must sample the ORIGINAL full-resolution image for the colour term;
* the head's ``scales`` (after its ``exp()`` activation) are already METRIC in
  point-map units (median scales/footprint ratio 0.6 on NPU frames) -- they
  must NOT be multiplied by depth/fx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from stream_ortho import config as zcfg
from stream_ortho.filter.pointmap import FilterConfig, PointmapFilter
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, encode_image, predict_gaussians
from stream_ortho.geometry import transform_gaussians_to_world

SH_C0 = 0.28209479177387814


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> SH DC coefficient (the 3DGS convention)."""
    return (rgb - 0.5) / SH_C0


def sample_fullres_colors(img_u8: np.ndarray, H: int, W: int, device: str = "cuda:0") -> torch.Tensor:
    """Sample ``img_u8`` (H0,W0,3) uint8 RGB at the ``H``x``W`` head grid, bilinearly.

    The head grid is the MASt3R resize of the same image, so source pixel ``(u, v)``
    maps to the original pixel ``((u+0.5)*W0/W - 0.5, (v+0.5)*H0/H - 0.5)`` -- a pure
    uniform rescale with no parallax.  Returns (H, W, 3) RGB in [0, 1].
    """
    H0, W0 = img_u8.shape[:2]
    img_f = torch.as_tensor(img_u8, dtype=torch.float32, device=device) / 255.0
    uu = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    vv = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    gx = ((uu + 0.5) * W0 / W - 0.5) / (W0 - 1) * 2 - 1
    gy = ((vv + 0.5) * H0 / H - 0.5) / (H0 - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)                       # (H, W, 2)
    col = F.grid_sample(img_f.permute(2, 0, 1)[None], grid[None], mode="bilinear",
                        padding_mode="border", align_corners=True)
    return col[0].permute(1, 2, 0)


def _sh_dc_from_image(img_u8: np.ndarray, H: int, W: int, device: str = "cuda:0") -> torch.Tensor:
    return RGB2SH(sample_fullres_colors(img_u8, H, W, device))


@dataclass
class Gaussians:
    """MASt3r-anchored Gaussian representation (P5.1a design).
    
    All tensors are (N,C) where N is the number of Gaussians after filtering.
    Coordinates are in world frame after Sim3 transform + plane projection.
    """
    means: torch.Tensor           # (N, 3) world-frame XYZ, MASt3r-anchored
    scales: torch.Tensor          # (N, 3) Gaussian scale (metric, point-map units × Sim3 s)
    quats: torch.Tensor           # (N, 4) wxyz quaternions (gsplat convention)
    opacity: torch.Tensor         # (N, 1) opacity in [0,1]
    sh_dc: torch.Tensor           # (N, 3) SH DC = RGB2SH(full-res image) + head residual
    keyframe_id: int              # which keyframe these came from
    confidence: Optional[torch.Tensor] = None  # (N, 1) head confidence


def mast3r_anchored_gaussians(
    img_u8: np.ndarray,
    X_canon: np.ndarray,
    sim3_vec: np.ndarray,
    T_WP: np.ndarray,
    keyframe_id: int,
    model=None,
    head=None,
    filter_cfg: Optional[FilterConfig] = None,
    device: str = "cuda:0",
) -> Gaussians:
    """Create MASt3r-anchored Gaussians from one keyframe.
    
    Pipeline (P5.1a Task 1):
        1. Encode image through MASt3R encoder
        2. Run Splatt3r head prediction → scales, quats, opacity, sh residual
        3. **CRITICAL**: Use X_canon as means (NOT head's mu!)
        4. sh DC = RGB2SH(original image, full-res) + head residual
        5. Transform: camera → world (Sim3) → plane (T_WP)
        6. Filter: confidence ≥0.5, NFP above, prune isolated
        7. Return Gaussians dataclass
    
    Args:
        img_u8: (H, W, 3) uint8 RGB image
        X_canon: (H, W, 3) float64 canonical 3D points in camera frame (MASt3R-SLAM)
        sim3_vec: (8,) Sim3 transform [tx, ty, tz, qx, qy, qz, qw, s]
        T_WP: (4, 4) world→plane homogeneous transform
        keyframe_id: integer keyframe identifier
        model: MASt3R model (loaded if None)
        head: Splatt3r GaussianHead (loaded if None)
        filter_cfg: PointmapFilter config (default if None)
        device: torch device
    
    Returns:
        Gaussians with N points after filtering (N ≤ H*W)
    """
    # Load models if not provided
    if model is None:
        model = load_model(device=device)
    if head is None:
        head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=device, verbose=False)
    
    # 1. Encode image
    enc = encode_image(model, img_u8, device=device, img_size=512)
    
    # 2. Run Splatt3r head prediction
    pred = predict_gaussians(
        model, head, enc["feat"], enc["pos"], enc["shape_hw"], anchor="head"
    )
    
    # 3. Extract appearance parameters from head (ignore mu_cam!)
    #    NOTE: the head's `sh` output is only a small RESIDUAL: the released
    #    Splatt3R model adds `RGB2SH(original_img)` on top of it at inference
    #    (`learn_residual = True` in splatt3r/main.py).  The residual alone is
    #    +-0.08 SH units -> a flat grey image; with the image term the colours
    #    span the full [-1.8, 1.8] range.
    scales_raw = pred["scales"]   # (H, W, 3) metric, in point-map units (see note below)
    quats = pred["quats"]         # (H, W, 4) wxyz
    opacity = pred["opacity"]     # (H, W, 1)
    sh_residual = pred["sh"]      # (H, W, 3) SH DC residual (see note above)
    conf = pred["conf"]           # (H, W, 1)
    
    H, W = scales_raw.shape[:2]
    
    # 4. CRITICAL: Use X_canon as means (MASt3r-anchored geometry)
    X_canon_torch = torch.as_tensor(X_canon, dtype=torch.float32, device=device)
    if X_canon_torch.shape != (H, W, 3):
        raise ValueError(f"X_canon shape {X_canon.shape} != expected {(H, W, 3)}")
    
    means_cam = X_canon_torch  # This is the KEY innovation!
    
    # 5. Scales: the head's `exp()` output is already METRIC in point-map units --
    #    measured median ratio scales/pixel-footprint is 0.6 (p95 1.6) on NPU
    #    frames, matching the released Splatt3R output statistics.  Do NOT
    #    rescale by depth/fx: that shrinks them another ~100x (measured) and
    #    the render vanishes.  The Sim3 scale `s` is applied in step 6.
    scales = scales_raw.clamp_min(1e-8)
    
    # 6. Transform to world frame (Sim3: camera → world)
    means_world, quats_world, scales_world = transform_gaussians_to_world(
        means_cam, quats, scales, sim3_vec, device=device
    )
    
    # 7. Appearance: SH DC = RGB2SH(original image, sampled at full resolution)
    #    + the head's residual.  Sampling the FULL-RES image is what gives P5 its
    #    detail: the head grid is 288x512 but img_u8 is the original 1080x1920,
    #    so each Gaussian carries 3.75x finer colour than the downsampled grid.
    sh_dc = _sh_dc_from_image(img_u8, H, W, device) + sh_residual
    sh_dc = sh_dc.clamp(-3.0, 3.0)
    
    # 8. Flatten to (N, C) format
    N = H * W
    means_world = means_world.reshape(N, 3)
    scales_world = scales_world.reshape(N, 3)
    quats_world = quats_world.reshape(N, 4)
    opacity_flat = opacity.reshape(N, 1)
    sh_dc_flat = sh_dc.reshape(N, 3)
    conf_flat = conf.reshape(N, 1)
    
    # 9. Transform to plane frame for filtering
    # Convert to numpy for filter
    means_np = means_world.detach().cpu().numpy()
    conf_np = conf_flat.detach().cpu().numpy().ravel()
    
    # Apply T_WP to get plane coordinates
    means_homo = np.concatenate([means_np, np.ones((N, 1))], axis=1)
    T_PW = np.linalg.inv(T_WP)
    means_plane = (T_PW @ means_homo.T).T[:, :3]
    
    x_plane, y_plane, z_plane = means_plane[:, 0], means_plane[:, 1], means_plane[:, 2]
    
    # 10. Apply filtering (confidence, NFP, isolated pruning)
    if filter_cfg is None:
        # Compute grid origin from min coordinates (avoid negative grid indices)
        x_min, y_min = float(x_plane.min()), float(y_plane.min())
        cell_size = 0.45
        origin_x = np.floor(x_min / cell_size) * cell_size - cell_size  # add margin
        origin_y = np.floor(y_min / cell_size) * cell_size - cell_size
        
        filter_cfg = FilterConfig(
            enabled=True,
            min_conf=0.5,
            nfp=True,
            prune=True,
            iso_min_support=2,
            iso_height_gap_m=1.2,
            cell=cell_size,
            origin=(origin_x, origin_y),
        )
    
    pf = PointmapFilter(filter_cfg)
    keep, reasons = pf.apply(x_plane, y_plane, z_plane, conf_np)
    
    # Apply filter mask
    n_kept = keep.sum()
    keep_torch = torch.as_tensor(keep, dtype=torch.bool, device=device)
    
    means_filtered = means_world[keep_torch]
    scales_filtered = scales_world[keep_torch]
    quats_filtered = quats_world[keep_torch]
    opacity_filtered = opacity_flat[keep_torch]
    sh_dc_filtered = sh_dc_flat[keep_torch]
    conf_filtered = conf_flat[keep_torch]
    
    print(f"[mast3r_anchored_gaussians] Keyframe {keyframe_id}: "
          f"{N} → {n_kept} Gaussians after filtering "
          f"(gate={reasons['gate']}, nfp={reasons['nfp']}, prune={reasons['prune']})")
    
    return Gaussians(
        means=means_filtered,
        scales=scales_filtered,
        quats=quats_filtered,
        opacity=opacity_filtered,
        sh_dc=sh_dc_filtered,
        keyframe_id=keyframe_id,
        confidence=conf_filtered,
    )
