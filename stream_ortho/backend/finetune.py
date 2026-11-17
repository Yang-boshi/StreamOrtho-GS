"""Per-keyframe 3DGS finetuning (the missing optimisation stage).

The Gaussian head only gives an *initial* shape per pixel: one disc of roughly one
pixel footprint, opacity 1, no SH, colour copied from the image.  Rendered
orthographically these discs are tiny and leave gaps (visible as a dot lattice), and
their colour/opacity/shape are never adapted to the multi-view evidence.

This module runs the standard 3DGS optimisation that the design doc promised
("先冻结/小步几何，主要优化颜色/不透明度/尺度/旋转"), following
``SLAM+GS/gsplat/examples/simple_trainer.py``:

* render the **keyframe itself** (perspective camera, gsplat),
* loss = L1 + 0.2 * (1 - SSIM) over the valid region,
* optimise colour / opacity / scale / rotation with normal LRs and the mean with a
  *tiny* LR (the user's requirement: positions should barely move),
* no densification (the budget is fixed by the streaming map), but the scales can grow
  to close the gaps -- which is exactly what the ortho needs.

Everything is per keyframe and independent, so it can run in the streaming backend
(≈50 iterations × a 960×540 render is a few hundred ms on a 4070 Ti Super).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from stream_ortho.geometry import compose_sim3, sim3_from_vec8


@dataclass
class FinetuneConfig:
    iters: int = 50
    render_scale: float = 0.5           # render the keyframe at this fraction of full res
    lr_colors: float = 1e-3             # tuned: 8e-3 diverged (PSNR 24.3 -> 21.3)
    lr_opacity: float = 3e-3
    lr_scales: float = 5e-4
    lr_quats: float = 1e-5
    lr_means: float = 5e-6              # positions move very little, by design
    ssim_w: float = 0.2
    min_scale_px: float = 1.0           # keep at least ~1 px sigma
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
    mse = F.mse_loss(a[m], b[m]).clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def finetune_patch(patch, ds, npz, k: int, sim3_vec_gt, cfg: FinetuneConfig = None,
                   device: str = "cuda:0", verbose: bool = False) -> dict:
    """Refine one keyframe's Gaussians against that keyframe's own image."""
    from gsplat import rasterization

    cfg = cfg or FinetuneConfig()
    fid = int(npz["frame_id"][k])
    img = ds.read_img(fid).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    s2 = cfg.render_scale
    H2, W2 = int(round(H * s2)), int(round(W * s2))
    img_t = torch.as_tensor(img, device=device).permute(2, 0, 1)[None]
    img_t = F.interpolate(img_t, size=(H2, W2), mode="area")

    R, t, s = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
    T_cw = np.eye(4)
    T_cw[:3, :3] = R.T
    T_cw[:3, 3] = -R.T @ t
    viewmat = torch.as_tensor(T_cw, dtype=torch.float32, device=device)[None]
    K = ds.K_orig.astype(np.float64).copy()
    K[0, 0] *= s2; K[1, 1] *= s2; K[0, 2] *= s2; K[1, 2] *= s2
    Ks = torch.as_tensor(K, dtype=torch.float32, device=device)[None]

    dev = torch.device(device)
    params = {
        "means": torch.nn.Parameter(torch.as_tensor(patch.means, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.as_tensor(patch.scales, device=dev).clamp_min(1e-6))),
        "quats": torch.nn.Parameter(torch.as_tensor(patch.quats, device=dev)),
        "opacities": torch.nn.Parameter(torch.logit(
            torch.as_tensor(patch.opacity, device=dev).clamp(1e-3, 1 - 1e-3))),
        "colors": torch.nn.Parameter(torch.logit(
            torch.as_tensor(patch.colors, device=dev).clamp(1e-3, 1 - 1e-3))),
    }
    opt = torch.optim.Adam([
        {"params": [params["colors"]], "lr": cfg.lr_colors},
        {"params": [params["opacities"]], "lr": cfg.lr_opacity},
        {"params": [params["scales"]], "lr": cfg.lr_scales},
        {"params": [params["quats"]], "lr": cfg.lr_quats},
        {"params": [params["means"]], "lr": cfg.lr_means},
    ])
    s_min = cfg.min_scale_px * (1.0 / float(ds.K_orig[0, 0] / 1.0)) * 1.0   # metres ~ px

    def render():
        rgb, alpha, _ = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"]),
            viewmat, Ks, W2, H2, camera_model="pinhole", sh_degree=None,
            packed=False, rasterize_mode="antialiased")
        return rgb[0].permute(2, 0, 1)[None], alpha[0].permute(2, 0, 1)[None]

    with torch.no_grad():
        rgb0, a0 = render()
        mask = a0 > 0.05
        p0 = _psnr(rgb0, img_t, mask.expand_as(img_t))
    curve = []
    for it in range(cfg.iters):
        rgb, alpha = render()
        m = (alpha > 0.05).expand_as(img_t)
        l1 = F.l1_loss(rgb[m], img_t[m])
        loss = l1 + cfg.ssim_w * (1.0 - _ssim(rgb * m, img_t * m))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["scales"].clamp_(max=float(np.log(50.0)))
            params["quats"].copy_(F.normalize(params["quats"], dim=-1))
            if (it + 1) % cfg.log_every == 0 or it + 1 == cfg.iters:
                p = _psnr(rgb.detach(), img_t, m.detach())
                curve.append({"iter": it + 1, "loss": float(loss), "psnr": p})
                if verbose:
                    print(f"    it{it+1:3d} loss={float(loss):.5f} PSNR={p:5.2f}")
    with torch.no_grad():
        rgb1, a1 = render()
        p1 = _psnr(rgb1, img_t, (a1 > 0.05).expand_as(img_t))
        sc = torch.exp(params["scales"]).detach().cpu().numpy()
    return {
        "psnr_before": p0, "psnr_after": p1, "curve": curve, "iters": cfg.iters,
        "means": params["means"].detach().cpu().numpy(),
        "quats": F.normalize(params["quats"], dim=-1).detach().cpu().numpy(),
        "scales": sc,
        "opacity": torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
        "colors": torch.sigmoid(params["colors"]).detach().cpu().numpy(),
        "scale_median_before": float(np.median(np.asarray(patch.scales))),
        "scale_median_after": float(np.median(sc)),
    }
