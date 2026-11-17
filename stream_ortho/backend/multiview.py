"""Multi-view joint refinement of a region's Gaussians (design doc P4 / strategy D).

Every other optimisation we tried was **single-view** (per-keyframe finetune: +0.04 dB;
per-tile ortho refine: self-consistent).  This one optimises the same set of Gaussians
against **K keyframe images at once** with the poses fixed, which is the only setup
that can force the geometry to explain more than one view -- i.e. the only one that can
reduce the cross-frame disagreement behind the "layering" and the blurred ground.

Positions move very little (``lr_means`` tiny), colours / opacity / scales / rotations
are free, no densification (the budget is fixed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MultiViewConfig:
    iters: int = 100
    render_scale: float = 0.25
    lr_colors: float = 1e-3
    lr_opacity: float = 3e-3
    lr_scales: float = 5e-4
    lr_quats: float = 1e-5
    lr_means: float = 5e-6
    ssim_w: float = 0.2
    log_every: int = 25


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


def _psnr(a, b, m):
    return float(-10.0 * torch.log10(F.mse_loss(a[m], b[m]).clamp_min(1e-12)))


def refine_multiview(means: np.ndarray, quats: np.ndarray, scales: np.ndarray,
                     opacity: np.ndarray, colors: np.ndarray,
                     cams, images, cfg: MultiViewConfig = None,
                     device: str = "cuda:0", verbose: bool = False) -> dict:
    """Optimise one Gaussian set against several perspective views.

    ``cams``   : list of ``(viewmat 4x4 world->camera, K 3x3, H, W)``
    ``images`` : list of ``HxWx3`` float arrays in [0, 1]
    """
    from gsplat import rasterization

    cfg = cfg or MultiViewConfig()
    dev = torch.device(device)
    params = {
        "means": torch.nn.Parameter(torch.as_tensor(means, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.as_tensor(scales, device=dev).clamp_min(1e-6))),
        "quats": torch.nn.Parameter(torch.as_tensor(quats, device=dev)),
        "opacities": torch.nn.Parameter(torch.logit(torch.as_tensor(opacity, device=dev).clamp(1e-3, 1 - 1e-3))),
        "colors": torch.nn.Parameter(torch.logit(torch.as_tensor(colors, device=dev).clamp(1e-3, 1 - 1e-3))),
    }
    opt = torch.optim.Adam([
        {"params": [params["colors"]], "lr": cfg.lr_colors},
        {"params": [params["opacities"]], "lr": cfg.lr_opacity},
        {"params": [params["scales"]], "lr": cfg.lr_scales},
        {"params": [params["quats"]], "lr": cfg.lr_quats},
        {"params": [params["means"]], "lr": cfg.lr_means},
    ])
    s2 = cfg.render_scale
    tgt, vms, Ks, hws = [], [], [], []
    for (vm, K, H, W), img in zip(cams, images):
        img = np.asarray(img)
        if img.dtype == np.uint8:                 # accept raw images too
            img = img.astype(np.float32) / 255.0
        H2, W2 = max(8, int(round(H * s2))), max(8, int(round(W * s2)))
        t = torch.as_tensor(img, device=dev).permute(2, 0, 1)[None]
        tgt.append(F.interpolate(t, size=(H2, W2), mode="area"))
        vms.append(torch.as_tensor(vm, dtype=torch.float32, device=dev))
        Kk = np.asarray(K, np.float64).copy()
        Kk[0, 0] *= s2; Kk[1, 1] *= s2; Kk[0, 2] *= s2; Kk[1, 2] *= s2
        Ks.append(torch.as_tensor(Kk, dtype=torch.float32, device=dev))
        hws.append((H2, W2))

    def render_all():
        outs = []
        for vm, Kk, (H2, W2) in zip(vms, Ks, hws):
            rgb, alpha, _ = rasterization(
                params["means"], F.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                torch.sigmoid(params["colors"]), vm[None], Kk[None], W2, H2,
                camera_model="pinhole", sh_degree=None, packed=False,
                rasterize_mode="antialiased")
            outs.append((rgb[0].permute(2, 0, 1)[None], alpha[0].permute(2, 0, 1)[None]))
        return outs

    MIN_PX = 1000

    def _valid(alpha):
        return (alpha > 0.05).sum() >= MIN_PX

    with torch.no_grad():
        outs0 = render_all()
        keep = [i for i, (r, a) in enumerate(outs0) if _valid(a)]
        before = [_psnr(outs0[i][0], tgt[i], (outs0[i][1] > 0.05).expand_as(tgt[i])) for i in keep]
    curve = []
    for it in range(cfg.iters):
        outs = render_all()
        loss = 0.0
        n_used = 0
        for i, (rgb, alpha) in enumerate(outs):
            if i not in keep:
                continue                       # this view does not see the region
            t = tgt[i]
            m = (alpha > 0.05).expand_as(t)
            loss = loss + F.l1_loss(rgb[m], t[m]) + cfg.ssim_w * (1.0 - _ssim(rgb * m, t * m))
            n_used += 1
        loss = loss / max(n_used, 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["quats"].copy_(F.normalize(params["quats"], dim=-1))
            if (it + 1) % cfg.log_every == 0 or it + 1 == cfg.iters:
                cur = [_psnr(outs[i][0], tgt[i], (outs[i][1] > 0.05).expand_as(tgt[i])) for i in keep]
                curve.append({"iter": it + 1, "loss": float(loss), "psnr": float(np.mean(cur))})
                if verbose:
                    print(f"    it{it+1:3d} loss={float(loss):.5f} mean-view PSNR={np.mean(cur):5.2f}")
    with torch.no_grad():
        outs1 = render_all()
        after = [_psnr(outs1[i][0], tgt[i], (outs1[i][1] > 0.05).expand_as(tgt[i])) for i in keep]
    return {
        "views": len(tgt), "views_used": len(keep), "iters": cfg.iters, "curve": curve,
        "psnr_before": [float(x) for x in before], "psnr_after": [float(x) for x in after],
        "mean_before": float(np.mean(before)), "mean_after": float(np.mean(after)),
        "means": params["means"].detach().cpu().numpy(),
        "quats": F.normalize(params["quats"], dim=-1).detach().cpu().numpy(),
        "scales": torch.exp(params["scales"]).detach().cpu().numpy(),
        "opacity": torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
        "colors": torch.sigmoid(params["colors"]).detach().cpu().numpy(),
    }
