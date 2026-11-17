"""Orthographic renderer using gsplat for 3D Gaussian splatting.

This module implements orthographic projection rendering (P5.1a Task 2),
converting world-frame Gaussians into high-resolution RGB orthophotos.

Key difference from perspective rendering:
    - Camera positioned directly above tile center at z_max + offset
    - Orthographic projection (fx = fy = 1/gsd pixels per meter)
    - Looking straight down (-Z axis in world frame)
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch

from stream_ortho.geometry import ortho_viewmat, ortho_K


def render_ortho_tile(
    gaussians,
    T_WP: np.ndarray,
    cx: float,
    cy: float,
    z_max: float,
    gsd: float = 0.20,
    tile_size: int = 512,
    z_offset: float = 10.0,
    scale_floor: float = 0.0,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render one orthographic tile from world-frame Gaussians.
    
    Pipeline (P5.1a Task 2):
        1. Setup orthographic camera positioned above tile center
        2. Construct viewmat: world → ortho camera
        3. Construct K: orthographic projection (fx=fy=1/gsd)
        4. Convert SH DC to RGB colors
        5. Call gsplat.rasterization() with antialiased mode
        6. Return (rgb, alpha, depth) tensors
    
    Args:
        gaussians: Gaussians dataclass from mast3r_anchored_gaussians()
        T_WP: (4, 4) world→plane homogeneous transform
        cx, cy: Tile center in plane coordinates (meters)
        z_max: Maximum Z value in the scene (meters, plane frame)
        gsd: Ground sampling distance (meters per pixel), default 0.20m
        tile_size: Output tile size (pixels), default 512
        z_offset: Camera height above z_max (meters), default 10m
        scale_floor: Minimum per-axis Gaussian scale (plane units).  One Splatt3R
            head Gaussian per source pixel is too sparse to cover a tile by
            itself; raise it to ~sqrt(tile_area/N) so the splats tile densely.
            0 keeps the raw head scales (for multi-keyframe accumulation).
        device: torch device
    
    Returns:
        rgb: (H, W, 3) float32 tensor in [0, 1]
        alpha: (H, W, 1) float32 tensor in [0, 1]
        depth: (H, W, 1) float32 tensor (camera-space depth)
    
    Notes:
        - Empty tiles (no Gaussians visible) return alpha=0.0 transparent image
        - Camera looks straight down: view direction = (0, 0, -1) in world
        - ``camera_model="ortho"``: gsplat projects ``u = fx * x`` (pixels per
          metre).  The default "pinhole" model divides by z and collapses the
          whole scene into a few pixels at the fx a metre-per-pixel K implies.
    """
    from gsplat import rasterization
    
    # Handle empty Gaussians
    if gaussians.means.shape[0] == 0:
        rgb = torch.zeros((tile_size, tile_size, 3), dtype=torch.float32, device=device)
        alpha = torch.zeros((tile_size, tile_size, 1), dtype=torch.float32, device=device)
        depth = torch.full((tile_size, tile_size, 1), float('inf'), dtype=torch.float32, device=device)
        return rgb, alpha, depth
    
    # 1. Setup orthographic camera
    # Camera positioned at (cx, cy, z_max + z_offset) in plane frame,
    # looking straight down (-Z in plane frame)
    z_top = z_max + z_offset
    viewmat_np = ortho_viewmat(T_WP, cx, cy, z_top)  # (4, 4) world→camera
    
    # 2. Construct intrinsics (orthographic: fx=fy=1/gsd pixels per meter)
    K_np = ortho_K(gsd, tile_size, tile_size)  # (3, 3)
    
    # Convert to torch tensors with batch dimension
    viewmats = torch.as_tensor(
        viewmat_np, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 4, 4)
    
    Ks = torch.as_tensor(
        K_np, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 3, 3)
    
    # 3. Convert SH DC to RGB colors
    # SH DC basis: C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    colors = (gaussians.sh_dc * C0 + 0.5).clamp(0.0, 1.0)  # (N, 3)
    
    # 4. Prepare Gaussian parameters
    means = gaussians.means.contiguous()      # (N, 3)
    quats = gaussians.quats.contiguous()      # (N, 4) wxyz
    scales = gaussians.scales.contiguous()    # (N, 3)
    if scale_floor > 0.0:
        # keep the head's anisotropy, only lift the floor so the splats tile
        # densely enough to cover the tile (one head gaussian per source pixel
        # is too sparse on its own for a single-keyframe render)
        scales = torch.maximum(scales, torch.full_like(scales, float(scale_floor)))
    opacities = gaussians.opacity.squeeze(-1).contiguous()  # (N,)
    
    # 5. Call gsplat rasterization (orthographic mode)
    try:
        rgb, alpha, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=tile_size,
            height=tile_size,
            # CRITICAL: "ortho" projects u = fx * x (pixels per metre); the
            # default "pinhole" divides by the camera-space z and collapses
            # the scene into a few centre pixels at ortho fx values
            camera_model="ortho",
            # No spherical harmonics beyond DC (we already converted to RGB)
            sh_degree=None,
            packed=False,  # We don't use packed mode for simplicity
            render_mode="RGB",
            rasterize_mode="antialiased",  # Smooth anti-aliased rendering
        )
    except Exception as e:
        print(f"[render_ortho_tile] gsplat.rasterization failed: {e}")
        print(f"  means: {means.shape}, quats: {quats.shape}, scales: {scales.shape}")
        print(f"  opacities: {opacities.shape}, colors: {colors.shape}")
        raise
    
    # 6. Extract outputs (gsplat returns (1, H, W, C), we want (H, W, C))
    rgb = rgb[0]       # (H, W, 3)
    alpha = alpha[0]   # (H, W, 1)
    
    # Depth from metadata (if available)
    if "depth" in meta:
        depth = meta["depth"][0]  # (H, W, 1)
    else:
        # Fallback: compute depth from means and viewmat
        means_homo = torch.cat([means, torch.ones((means.shape[0], 1), device=device)], dim=1)
        means_cam = (viewmats[0] @ means_homo.T).T[:, :3]  # (N, 3)
        depth_values = means_cam[:, 2]  # Z in camera frame
        # This is a crude fallback; gsplat should provide depth
        depth = torch.zeros((tile_size, tile_size, 1), dtype=torch.float32, device=device)
    
    return rgb, alpha, depth


def render_ortho_tile_batch(
    gaussians_list,
    T_WP: np.ndarray,
    tile_centers: np.ndarray,
    z_max: float,
    gsd: float = 0.20,
    tile_size: int = 512,
    z_offset: float = 10.0,
    scale_floor: float = 0.0,
    device: str = "cuda:0",
) -> list:
    """Render multiple orthographic tiles in a batch (future optimization).
    
    Args:
        gaussians_list: List of Gaussians dataclasses
        T_WP: (4, 4) world→plane transform
        tile_centers: (N, 2) array of (cx, cy) tile centers
        z_max: Maximum Z value in scene
        gsd, tile_size, z_offset: Same as render_ortho_tile()
        device: torch device
    
    Returns:
        List of (rgb, alpha, depth) tuples
    """
    results = []
    for i, gaussians in enumerate(gaussians_list):
        cx, cy = tile_centers[i]
        rgb, alpha, depth = render_ortho_tile(
            gaussians, T_WP, cx, cy, z_max,
            gsd=gsd, tile_size=tile_size, z_offset=z_offset,
            scale_floor=scale_floor, device=device
        )
        results.append((rgb, alpha, depth))
    return results
