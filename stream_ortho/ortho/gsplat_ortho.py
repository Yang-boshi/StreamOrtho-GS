"""Orthographic renderer using gsplat for 3D Gaussian splatting.

Implements:
* :func:`render_ortho_tile` — standalone single-tile renderer (P5.1a/b/c).
* :func:`make_streaming_tile_render_fn` — spatial-indexed ``OrthoEngine``
  renderer over :class:`~stream_ortho.online.gs_map.StreamingGaussianMap` with
  halo guard bands and alpha un-premultiplication for seamless high-res tiles.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch

from stream_ortho.geometry import ortho_K, ortho_viewmat

_SH_C0 = 0.28209479177387814


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
    alpha_normalise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render one orthographic tile from world-frame Gaussians."""
    from gsplat import rasterization

    if gaussians.means.shape[0] == 0:
        rgb = torch.zeros((tile_size, tile_size, 3), dtype=torch.float32, device=device)
        alpha = torch.zeros((tile_size, tile_size, 1), dtype=torch.float32, device=device)
        depth = torch.full((tile_size, tile_size, 1), float("inf"), dtype=torch.float32, device=device)
        return rgb, alpha, depth

    z_top = float(z_max) + float(z_offset)
    viewmat_np = ortho_viewmat(T_WP, cx, cy, z_top)
    K_np = ortho_K(gsd, tile_size, tile_size)

    viewmats = torch.as_tensor(viewmat_np, dtype=torch.float32, device=device).unsqueeze(0)
    Ks = torch.as_tensor(K_np, dtype=torch.float32, device=device).unsqueeze(0)

    colors = (gaussians.sh_dc * _SH_C0 + 0.5).clamp(0.0, 1.0)
    means = gaussians.means.contiguous()
    quats = gaussians.quats.contiguous()
    scales = gaussians.scales.contiguous()
    if scale_floor > 0.0:
        scales = torch.maximum(scales, torch.full_like(scales, float(scale_floor)))
    opacities = gaussians.opacity.squeeze(-1).contiguous()

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
        camera_model="ortho",
        sh_degree=None,
        packed=True,
        render_mode="RGB",
        rasterize_mode="antialiased",
    )

    rgb = rgb[0]       # (H, W, 3)
    alpha = alpha[0]   # (H, W, 1)
    if alpha_normalise:
        a_clamped = alpha.clamp(min=1e-3)
        rgb = torch.where(alpha > 0.02, (rgb / a_clamped).clamp(0.0, 1.0), torch.zeros_like(rgb))

    if "depth" in meta:
        depth = meta["depth"][0]
    else:
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
    """Render multiple orthographic tiles sequentially."""
    results = []
    for i, gaussians in enumerate(gaussians_list):
        cx, cy = tile_centers[i]
        rgb, alpha, depth = render_ortho_tile(
            gaussians, T_WP, cx, cy, z_max,
            gsd=gsd, tile_size=tile_size, z_offset=z_offset,
            scale_floor=scale_floor, device=device,
        )
        results.append((rgb, alpha, depth))
    return results


def make_streaming_tile_render_fn(
    gs_map,
    gsd: float,
    tile_size: int = 512,
    halo: int = 16,
    device: str = "cuda:0",
):
    """Create an ``OrthoEngine`` tile render callback backed by ``StreamingGaussianMap``.

    Spatially queries only Gaussians intersecting the tile + halo guard band,
    rasterizes ``(tile + 2*halo) x (tile + 2*halo)`` via ``gsplat`` ortho mode,
    alpha-normalizes, and crops the centre ``(tile, tile)`` for seamless borders.
    """
    from gsplat import rasterization

    halo = max(int(halo), 0)
    tile_size = int(tile_size)
    render_sz = tile_size + 2 * halo
    K_np = ortho_K(gsd, render_sz, render_sz)
    Ks_t = torch.as_tensor(K_np, dtype=torch.float32, device=device).unsqueeze(0)
    I4 = np.eye(4, dtype=np.float64)
    halo_m = halo * float(gsd)

    @torch.inference_mode()
    def render(grid, i: int, j: int):
        t0 = time.perf_counter()
        x0, y0, x1, y1 = grid.tile_window(i, j)
        pad = halo_m + 6.0 * float(gsd)
        idx = gs_map.query_bbox(x0, y0, x1, y1, pad=pad)
        if idx.size == 0:
            z_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            z_a = np.zeros((tile_size, tile_size), dtype=np.float32)
            z_d = np.zeros((tile_size, tile_size), dtype=np.float32)
            return z_rgb, z_a, z_d, {"ms": (time.perf_counter() - t0) * 1000.0, "n_gs": 0}

        means_np = gs_map.means[idx]
        cx, cy = grid.tile_xy(i, j)
        z_top = float(means_np[:, 2].max()) + 20.0
        viewmat_np = ortho_viewmat(I4, cx, cy, z_top)

        means = torch.from_numpy(means_np).to(device=device, dtype=torch.float32).contiguous()
        quats = torch.from_numpy(gs_map.quats[idx]).to(device=device, dtype=torch.float32).contiguous()
        scales = torch.from_numpy(gs_map.scales[idx]).to(device=device, dtype=torch.float32).contiguous()
        opacities = torch.from_numpy(gs_map.opacity[idx]).to(device=device, dtype=torch.float32).contiguous()
        sh_dc = torch.from_numpy(gs_map.sh_dc[idx]).to(device=device, dtype=torch.float32)
        colors = (sh_dc * _SH_C0 + 0.5).clamp(0.0, 1.0).contiguous()
        viewmats = torch.as_tensor(viewmat_np, dtype=torch.float32, device=device).unsqueeze(0)

        img, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks_t,
            width=render_sz,
            height=render_sz,
            camera_model="ortho",
            sh_degree=None,
            packed=True,
            render_mode="RGB+ED",
            rasterize_mode="antialiased",
        )

        rgb_t = img[0, :, :, :3]
        a_t = alpha[0, :, :, 0].clamp(0.0, 1.0)
        # Alpha un-premultiply so sparse edges do not darken against black background
        rgb_t = torch.where(
            a_t.unsqueeze(-1) > 0.02,
            (rgb_t / a_t.clamp(min=1e-3).unsqueeze(-1)).clamp(0.0, 1.0),
            torch.zeros_like(rgb_t),
        )
        depth_t = img[0, :, :, 3] if img.shape[-1] >= 4 else torch.zeros_like(a_t)

        if halo > 0:
            sl = slice(halo, halo + tile_size)
            rgb_t = rgb_t[sl, sl]
            a_t = a_t[sl, sl]
            depth_t = depth_t[sl, sl]

        rgb_np = rgb_t.detach().cpu().numpy().astype(np.float32)
        a_np = a_t.detach().cpu().numpy().astype(np.float32)
        d_np = depth_t.detach().cpu().numpy().astype(np.float32)
        return rgb_np, a_np, d_np, {
            "ms": (time.perf_counter() - t0) * 1000.0,
            "n_gs": int(idx.size),
        }

    return render
