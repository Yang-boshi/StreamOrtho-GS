"""Orthographic rasterisation wrapper (design doc S4 / §7.4 item 2).

A thin layer over ``gsplat.rasterization(camera_model="ortho")``:

* everything is a :class:`GaussianCloud` (plain torch tensors, world frame,
  metric or estimator units -- the caller decides);
* tiles are rendered at ``S + 2*halo`` and cropped, so no Gaussian is cut at a
  seam and neighbouring tiles overlap in the guard band;
* ``packed=True`` + fp32 (8 GB-friendly; the week-1 numbers show we are fine
  on a 16 GB card).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from stream_ortho.ortho.grid import OrthoGrid


@dataclass
class GaussianCloud:
    """World-frame Gaussians as flat tensors."""

    means: torch.Tensor          # (N,3)
    quats: torch.Tensor          # (N,4) wxyz, unit
    scales: torch.Tensor         # (N,3) > 0
    opacity: torch.Tensor        # (N,) in [0,1]
    colors: torch.Tensor         # (N,3)
    owner: Optional[torch.Tensor] = None   # (N,) keyframe id (week 4+)
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.means.shape[0])

    def to(self, device):
        self.means = self.means.to(device)
        self.quats = self.quats.to(device)
        self.scales = self.scales.to(device)
        self.opacity = self.opacity.to(device)
        self.colors = self.colors.to(device)
        if self.owner is not None:
            self.owner = self.owner.to(device)
        return self

    @staticmethod
    def concat(clouds: list["GaussianCloud"]) -> "GaussianCloud":
        cats = lambda attr: torch.cat([getattr(c, attr) for c in clouds], dim=0)  # noqa: E731
        owner = None
        if all(c.owner is not None for c in clouds):
            owner = torch.cat([c.owner for c in clouds], dim=0)
        return GaussianCloud(cats("means"), cats("quats"), cats("scales"),
                             cats("opacity"), cats("colors"), owner=owner)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.means.detach().cpu().numpy()
        return m.min(axis=0), m.max(axis=0)

    def stats(self) -> dict:
        s = self.scales.detach().cpu().numpy()
        o = self.opacity.detach().cpu().numpy()
        return {
            "n": len(self),
            "scale_median": float(np.median(s)),
            "scale_p95": float(np.percentile(s, 95)),
            "opacity_median": float(np.median(o)),
            "opacity_frac_gt_0_99": float((o > 0.99).mean()),
        }


#: divide the composited colour by alpha (removes the "darkening" that forced us to
#: use a 1.5x splat radius; week8-end-1 task 1)
ALPHA_NORMALISE = True


@torch.inference_mode()
def render_tile(cloud: GaussianCloud, grid: OrthoGrid, i: int, j: int,
                render_mode: str = "RGB", crop: bool = True,
                alpha_normalise: bool = None):
    """Render tile ``(i, j)`` at ``S + 2*halo`` and crop the centre ``S``.

    Returns ``(rgb, alpha, depth, meta)`` with ``rgb/alpha`` as numpy arrays
    (``H x W x 3`` and ``H x W``) in the **cropped** size when ``crop=True``.
    """
    from gsplat import rasterization

    n = grid.render_size
    viewmat = torch.as_tensor(grid.viewmat(i, j), dtype=torch.float32,
                              device=cloud.means.device)[None]
    Ks = torch.as_tensor(grid.K(), dtype=torch.float32, device=cloud.means.device)[None]

    img, alpha, meta = rasterization(
        cloud.means.reshape(-1, 3).contiguous(),
        cloud.quats.reshape(-1, 4).contiguous(),
        cloud.scales.reshape(-1, 3).contiguous(),
        cloud.opacity.reshape(-1).contiguous(),
        cloud.colors.reshape(-1, 3).contiguous(),
        viewmat,
        Ks,
        n,
        n,
        camera_model="ortho",
        sh_degree=None,
        packed=True,
        render_mode=render_mode,
        rasterize_mode="antialiased",
    )
    rgb_t = img[0][..., :3]
    a_t = alpha[0, :, :, 0].clamp(0, 1)
    do_norm = ALPHA_NORMALISE if alpha_normalise is None else alpha_normalise
    if do_norm:      # un-premultiply: rgb/alpha, keep black where nothing was drawn
        rgb_t = rgb_t / a_t.clamp(min=1e-3).unsqueeze(-1)
        rgb_t = torch.where(a_t.unsqueeze(-1) > 0.02, rgb_t, torch.zeros_like(rgb_t))
    rgb = rgb_t.clamp(0, 1).detach().cpu().numpy()
    a = a_t.detach().cpu().numpy()
    depth = None
    if render_mode == "RGB+ED" and img.shape[-1] >= 4:
        depth = img[0, :, :, 3].detach().cpu().numpy()
    if crop:
        h = grid.halo
        s = grid.tile
        rgb = rgb[h:h + s, h:h + s]
        a = a[h:h + s, h:h + s]
        if depth is not None:
            depth = depth[h:h + s, h:h + s]
    return rgb, a, depth, meta


def ortho_depth_from_height(cloud: GaussianCloud, grid: OrthoGrid, i: int, j: int):
    """Median height of the Gaussians above ``z_ground`` inside tile ``(i, j)``."""
    from stream_ortho.data.plane import world_to_plane

    m = cloud.means.detach().cpu().numpy()
    p = world_to_plane(grid.T_WP, m)
    x0, y0, x1, y1 = grid.tile_window(i, j)
    sel = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    if not sel.any():
        return None
    h = p[sel, 2] - grid.z_ground
    return {"n": int(sel.sum()), "h_median": float(np.median(h)),
            "h_p95": float(np.percentile(h, 95))}
