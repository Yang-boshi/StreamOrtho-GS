"""Per-tile photometric refinement ("local 3DGS optimisation", week8-end-1 task 3).

This is the piece the pipeline was missing: the streaming map is built from
*analytic* Gaussians (plane splats) and then voxel-averaged, so it never
optimises anything.  Here we take the Gaussians of **one tile** and run a short
standard 3DGS-style optimisation against the plane-warped source imagery:

* parameters: ``means`` (allowed to move, small weight pulling it back),
  ``scales`` (log), ``opacity`` (logit), ``colors`` (sigmoid);
* rotations are frozen (the plane prior gives them, and they are what makes the
  splats a *surface* rather than a cloud);
* loss: ``L1 + (1 - SSIM)`` inside the tile's valid mask;
* everything outside this tile is untouched (the design doc forbids full-map
  training/refinement, §2.3).

The target is the *plane-warped source image* (``PlaneReferenceMosaic``, one or
three frames), i.e. the same photometric evidence the streaming stage used -- no
hidden ground truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RefineConfig:
    # stage A: colour/opacity only (geometry frozen) -- the design doc's "freeze the
    # geometry first" rule; stage B: a *tiny* geometry nudge.
    iters_a: int = 120
    iters_b: int = 0            # geometry nudge: measured to *hurt* -> off by default
    lr_means: float = 2e-5
    lr_scales: float = 1e-4
    lr_opacity: float = 1e-2
    lr_colors: float = 5e-3
    w_means: float = 1e-2        # pulls means back to the initial (metric) position
    ssim_w: float = 0.2
    log_every: int = 20
    patience: int = 3           # early stop: #evaluations without a loss improvement
    margin_m: float = 0.5        # tile culling margin (metres)


def _render_raw(means, quats, scales, opacity, colors, grid, i, j):
    """Differentiable ortho render of the tile (crop included)."""
    from gsplat import rasterization

    n = grid.render_size
    viewmat = torch.as_tensor(grid.viewmat(i, j), dtype=torch.float32, device=means.device)[None]
    Ks = torch.as_tensor(grid.K(), dtype=torch.float32, device=means.device)[None]
    img, alpha, _ = rasterization(
        means, quats, scales, opacity, colors, viewmat, Ks, n, n,
        camera_model="ortho", sh_degree=None, packed=True,
        rasterize_mode="antialiased")
    rgb = img[0][..., :3]
    a = alpha[0][..., 0].clamp(1e-3)
    rgb = rgb / a.unsqueeze(-1)
    rgb = rgb.permute(2, 0, 1)[None]
    a = a[None, None]
    h = grid.halo
    s = grid.tile
    return rgb[..., h:h + s, h:h + s], a[..., h:h + s, h:h + s]


def _ssim_torch(x, y):
    """Minimal SSIM (Gaussian-free, 11x11 box) for a (1,3,H,W) pair."""
    import torch.nn.functional as F

    k = 11
    pad = k // 2
    mu_x = F.avg_pool2d(x, k, 1, pad)
    mu_y = F.avg_pool2d(y, k, 1, pad)
    sx = F.avg_pool2d(x * x, k, 1, pad) - mu_x ** 2
    sy = F.avg_pool2d(y * y, k, 1, pad) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, k, 1, pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return ssim.mean()


def cull_for_tile(cloud, grid, i: int, j: int, margin_m: float = 0.5):
    """Indices of the Gaussians that can influence tile ``(i, j)`` (+margin)."""
    x0, y0, x1, y1 = grid.tile_window(i, j)
    P = (cloud.means.detach().cpu().numpy().astype(np.float64) - grid.T_WP[:3, 3]) @ grid.T_WP[:3, :3]
    pad = margin_m + 2.0 * float(cloud.scales.detach().cpu().numpy().max())
    sel = ((P[:, 0] >= x0 - pad) & (P[:, 0] <= x1 + pad) &
           (P[:, 1] >= y0 - pad) & (P[:, 1] <= y1 + pad) &
           (P[:, 2] >= grid.z_ground - 20.0) & (P[:, 2] <= grid.z_top))
    return np.where(sel)[0], P


def refine_tile(cloud, grid, i: int, j: int, target_rgb: np.ndarray, target_cov: np.ndarray,
                cfg: RefineConfig = None, device: str = "cuda:0", verbose: bool = True,
                cull: bool = True) -> dict:
    """Optimise the Gaussians of tile ``(i,j)`` against ``target_rgb``.

    ``cloud`` is a :class:`~stream_ortho.ortho.rasterizer.GaussianCloud` (its tensors
    are modified in place on the returned cloud).
    """
    import torch.nn.functional as F

    cfg = cfg or RefineConfig()
    t0 = time.time()

    idx = None
    if cull:
        from stream_ortho.ortho.rasterizer import GaussianCloud

        idx, _ = cull_for_tile(cloud, grid, i, j, cfg.margin_m)
        if idx.size == 0:
            return {"psnr_before": float("nan"), "psnr_after": float("nan"), "gain": 0.0,
                    "iters": 0, "time_s": 0.0, "curve": [], "idx": idx}
        sub = GaussianCloud(means=cloud.means[idx], quats=cloud.quats[idx],
                            scales=cloud.scales[idx], opacity=cloud.opacity[idx],
                            colors=cloud.colors[idx])
        if verbose:
            print(f"      [refine] culled {idx.size:,}/{len(cloud):,} Gaussians for tile ({i},{j})")
        cloud = sub

    means = torch.nn.Parameter(cloud.means.to(device).clone())
    quats = cloud.quats.to(device).clone()
    scales = torch.nn.Parameter(torch.log(cloud.scales.to(device).clamp_min(1e-6)))
    opacity = torch.nn.Parameter(torch.logit(cloud.opacity.to(device).clamp(1e-3, 1 - 1e-3)))
    colors = torch.nn.Parameter(torch.logit(cloud.colors.to(device).clamp(1e-3, 1 - 1e-3)))
    init_means = means.detach().clone()

    tgt = torch.as_tensor(target_rgb, device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    # the valid mask must include *our* coverage too, otherwise the loss sees pixels
    # where the map draws nothing (alpha~0) and the reported PSNR is meaningless
    with torch.no_grad():
        _, a0 = _render_raw(means, quats, torch.exp(scales), torch.sigmoid(opacity),
                            torch.sigmoid(colors), grid, i, j)
        m0 = (torch.as_tensor(target_cov, device=device, dtype=torch.float32)[None, None] > 0.5) & (a0 > 0.5)
    mask = m0.float()
    if mask.sum() < 5000:
        return {"psnr_before": float("nan"), "psnr_after": float("nan"), "gain": 0.0,
                "best_psnr": float("nan"), "best_iter": 0, "iters": 0, "time_s": 0.0,
                "n_refined": int(idx.size) if idx is not None else len(cloud),
                "idx": idx, "curve": [], "rgb_after": np.zeros((1, 1, 3), np.float32)}

    opt = torch.optim.Adam([
        {"params": [means], "lr": cfg.lr_means},
        {"params": [scales], "lr": cfg.lr_scales},
        {"params": [opacity], "lr": cfg.lr_opacity},
        {"params": [colors], "lr": cfg.lr_colors},
    ])

    def psnr_now():
        with torch.no_grad():
            rgb, _ = _render_raw(means, quats, torch.exp(scales), torch.sigmoid(opacity),
                                 torch.sigmoid(colors), grid, i, j)
            mse = F.mse_loss(rgb[mask.expand_as(rgb).bool()], tgt[mask.expand_as(tgt).bool()])
            return float(-10.0 * torch.log10(mse.clamp_min(1e-12))), rgb

    before, _ = psnr_now()
    curve = []
    best = (before, None)
    best_state = None
    best_loss = float("inf")
    bad = 0
    total_iters = cfg.iters_a + cfg.iters_b
    for it in range(total_iters):
        stage_b = it >= cfg.iters_a
        if not stage_b:                       # freeze the geometry
            means.requires_grad_(False)
            scales.requires_grad_(False)
        else:
            means.requires_grad_(True)
            scales.requires_grad_(True)
        rgb, alpha = _render_raw(means, quats, torch.exp(scales), torch.sigmoid(opacity),
                                 torch.sigmoid(colors), grid, i, j)
        valid = mask.expand_as(rgb).bool()
        l1 = F.l1_loss(rgb[valid], tgt[valid])
        ssim_l = 1.0 - _ssim_torch(rgb * mask.expand_as(rgb), tgt * mask.expand_as(tgt))
        reg = ((means - init_means) ** 2).mean()
        loss = l1 + cfg.ssim_w * ssim_l + cfg.w_means * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([means, scales, opacity, colors], 1.0)
        opt.step()
        if (it + 1) % cfg.log_every == 0 or it + 1 == total_iters:
            with torch.no_grad():
                p, _ = psnr_now()
            curve.append({"iter": it + 1, "loss": float(loss), "psnr": p,
                          "stage": "B" if stage_b else "A"})
            if p > best[0]:
                best = (p, it + 1)
                best_state = (means.detach().clone(), torch.exp(scales).detach().clone(),
                              torch.sigmoid(opacity).detach().clone(),
                              torch.sigmoid(colors).detach().clone())
            if float(loss) < best_loss - 1e-5:
                best_loss, bad = float(loss), 0
            else:
                bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"      early stop at iter {it+1} (loss plateau)")
                break
            if verbose:
                print(f"      it{it+1:4d} [{'B' if stage_b else 'A'}]  loss={float(loss):.5f}  "
                      f"PSNR={p:5.2f} dB")

    if best_state is not None:            # restore the best (early stopping)
        m_b, s_b, o_b, c_b = best_state
        with torch.no_grad():
            means.copy_(m_b)
        scales.data = torch.log(s_b.clamp_min(1e-6))
        opacity.data = torch.logit(o_b.clamp(1e-3, 1 - 1e-3))
        colors.data = torch.logit(c_b.clamp(1e-3, 1 - 1e-3))
    after, rgb_t = psnr_now()
    return {
        "psnr_before": before, "psnr_after": after, "gain": after - before,
        "best_psnr": best[0], "best_iter": best[1],
        "iters": total_iters, "time_s": round(time.time() - t0, 2),
        "n_refined": int(idx.size) if idx is not None else len(cloud),
        "idx": idx, "curve": curve,
        "rgb_after": rgb_t[0].permute(1, 2, 0).detach().cpu().numpy(),
        "means": means.detach(), "scales": torch.exp(scales).detach(),
        "opacity": torch.sigmoid(opacity).detach(), "colors": torch.sigmoid(colors).detach(),
    }
