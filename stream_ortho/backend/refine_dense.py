"""Per-tile refinement **with densification** (week8-end-3 task 1).

Borrows the standard 3DGS strategy straight from the installed ``gsplat``:
``gsplat.strategy.DefaultStrategy`` (clone/split by 2-D gradient, prune by
opacity/scale, opacity reset).  That is the piece the streaming map never had --
our map is analytic + voxel-averaged, so no Gaussian ever grows where the imagery
needs more detail.

Differences from :mod:`stream_ortho.backend.refine` (the densification-free one):

* parameters live in a **dict** because the strategy *replaces* tensors when it
  grows/prunes (sizes change), so a captured local tensor would go stale;
* ``rasterization(..., packed=True)`` returns ``meta`` with ``means2d`` /
  ``gaussian_ids`` / ``radii``, which the strategy consumes;
* rotations are frozen (``lr=0``) -- the plane prior already fixes the surface
  orientation, and the strategy would otherwise rotate splats freely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DenseConfig:
    iters: int = 300
    refine_start_iter: int = 30
    refine_every: int = 30
    grow_grad2d: float = 2e-4
    prune_opa: float = 5e-3
    prune_scale3d: float = 0.5      # x scene_scale
    lr_means: float = 4e-4
    lr_scales: float = 2e-3
    lr_opacity: float = 1e-2
    lr_colors: float = 5e-3
    ssim_w: float = 0.2
    w_means: float = 1e-3
    log_every: int = 50


def _render(means, quats, scales, opacity, colors, grid, i, j):
    from gsplat import rasterization

    n = grid.render_size
    viewmat = torch.as_tensor(grid.viewmat(i, j), dtype=torch.float32, device=means.device)[None]
    Ks = torch.as_tensor(grid.K(), dtype=torch.float32, device=means.device)[None]
    img, alpha, info = rasterization(
        means, quats, scales, opacity, colors, viewmat, Ks, n, n,
        camera_model="ortho", sh_degree=None, packed=True,
        rasterize_mode="antialiased")
    rgb = img[0][..., :3]
    a = alpha[0][..., 0].clamp(1e-3)
    rgb = rgb / a.unsqueeze(-1)
    h, s = grid.halo, grid.tile
    return rgb.permute(2, 0, 1)[None][..., h:h + s, h:h + s], a[None, None][..., h:h + s, h:h + s], info


def _ssim(x, y):
    k, pad = 11, 5
    mu_x = F.avg_pool2d(x, k, 1, pad)
    mu_y = F.avg_pool2d(y, k, 1, pad)
    sx = F.avg_pool2d(x * x, k, 1, pad) - mu_x ** 2
    sy = F.avg_pool2d(y * y, k, 1, pad) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, k, 1, pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2))
            / ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))).mean()


def refine_tile_dense(cloud, grid, i: int, j: int, target_rgb: np.ndarray,
                      target_cov: np.ndarray, cfg: DenseConfig = None,
                      device: str = "cuda:0", verbose: bool = True) -> dict:
    from gsplat.strategy import DefaultStrategy

    from stream_ortho.backend.refine import cull_for_tile
    from stream_ortho.ortho.rasterizer import GaussianCloud

    cfg = cfg or DenseConfig()
    t0 = time.time()
    idx, _ = cull_for_tile(cloud, grid, i, j, 0.5)
    if idx.size == 0:
        return {"psnr_before": float("nan"), "psnr_after": float("nan"), "gain": 0.0,
                "n_before": 0, "n_after": 0, "idx": idx, "curve": [], "time_s": 0.0}
    sub = GaussianCloud(means=cloud.means[idx], quats=cloud.quats[idx],
                        scales=cloud.scales[idx], opacity=cloud.opacity[idx],
                        colors=cloud.colors[idx])
    n_cull = len(sub)

    params = {
        "means": torch.nn.Parameter(sub.means.clone()),
        "scales": torch.nn.Parameter(torch.log(sub.scales.clamp_min(1e-6))),
        "quats": torch.nn.Parameter(sub.quats.clone()),
        "opacities": torch.nn.Parameter(torch.logit(sub.opacity.clamp(1e-3, 1 - 1e-3))),
        "colors": torch.nn.Parameter(torch.logit(sub.colors.clamp(1e-3, 1 - 1e-3))),
    }
    lrs = {"means": cfg.lr_means, "scales": cfg.lr_scales, "quats": 0.0,
           "opacities": cfg.lr_opacity, "colors": cfg.lr_colors}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}

    strategy = DefaultStrategy(
        refine_start_iter=cfg.refine_start_iter, refine_stop_iter=10 ** 9,
        refine_every=cfg.refine_every, reset_every=10 ** 9,
        grow_grad2d=cfg.grow_grad2d, prune_opa=cfg.prune_opa,
        prune_scale3d=cfg.prune_scale3d, verbose=False)
    strategy.check_sanity(params, optimizers)
    scene_scale = float(grid.tile * grid.gsd)          # metres across the tile
    state = strategy.initialize_state(scene_scale=scene_scale)

    tgt = torch.as_tensor(target_rgb, device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    with torch.no_grad():
        rgb0, a0, _ = _render(params["means"], params["quats"], torch.exp(params["scales"]),
                              torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
                              grid, i, j)
        m0 = (torch.as_tensor(target_cov, device=device, dtype=torch.float32)[None, None] > 0.5) & (a0 > 0.5)
    if m0.sum() < 5000:
        return {"psnr_before": float("nan"), "psnr_after": float("nan"), "gain": 0.0,
                "n_before": n_cull, "n_after": n_cull, "idx": idx, "curve": [], "time_s": 0.0}
    mask = m0.expand_as(tgt)

    def psnr_now():
        with torch.no_grad():
            rgb, _, _ = _render(params["means"], params["quats"], torch.exp(params["scales"]),
                               torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
                               grid, i, j)
            mse = F.mse_loss(rgb[mask], tgt[mask])
            return float(-10.0 * torch.log10(mse.clamp_min(1e-12))), rgb

    before, _ = psnr_now()
    best = (before, None, None)
    init_means = params["means"].detach().clone()
    curve = []
    for step in range(cfg.iters):
        rgb, alpha, info = _render(params["means"], params["quats"], torch.exp(params["scales"]),
                                   torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
                                   grid, i, j)
        l1 = F.l1_loss(rgb[mask], tgt[mask])
        ssim_l = 1.0 - _ssim(rgb * mask, tgt * mask)
        reg = ((params["means"] - init_means) ** 2).mean() if params["means"].shape == init_means.shape else 0.0
        loss = l1 + cfg.ssim_w * ssim_l + cfg.w_means * reg
        for o in optimizers.values():
            o.zero_grad(set_to_none=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        for o in optimizers.values():
            o.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.iters:
            p, _ = psnr_now()
            n_now = int(params["means"].shape[0])
            curve.append({"iter": step + 1, "loss": float(loss), "psnr": p, "n": n_now})
            if verbose:
                print(f"      it{step+1:4d}  loss={float(loss):.5f}  PSNR={p:5.2f}  N={n_now:,}")
            if p > best[0]:
                best = (p, step + 1, {k: v.detach().clone() for k, v in params.items()})

    if best[2] is not None:
        for k in params:
            params[k].data = best[2][k]
    after, rgb_t = psnr_now()
    return {
        "psnr_before": before, "psnr_after": after, "gain": after - before,
        "best_psnr": best[0], "best_iter": best[1],
        "n_before": n_cull, "n_after": int(params["means"].shape[0]),
        "n_grow": int(params["means"].shape[0]) - n_cull,
        "iters": cfg.iters, "time_s": round(time.time() - t0, 2),
        "idx": idx, "curve": curve,
        "rgb_after": rgb_t[0].permute(1, 2, 0).detach().cpu().numpy(),
        "means": params["means"].detach(), "scales": torch.exp(params["scales"]).detach(),
        "opacity": torch.sigmoid(params["opacities"]).detach(),
        "colors": torch.sigmoid(params["colors"]).detach(),
    }
