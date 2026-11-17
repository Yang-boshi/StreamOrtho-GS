"""Feed-forward Gaussian prediction on MASt3R-SLAM frames (S3).

Pipeline (design doc §11.5, keyframe-only):

    image -> MASt3R encoder -> decoder tokens -> GaussianHead
          -> {means, scales, quats, opacity, sh, conf}
          -> (optional) world transform with the full Sim3

The head is the vendored Splatt3R one (`gaussian_head.py`).  Two anchor modes
are supported and both are measured in week 2:

* ``head``   -- ``mu = head['pts3d']`` (what Splatt3R was trained with);
* ``mast3r`` -- ``mu = X_canon`` from MASt3R-SLAM's (fused) point map.

Note: the released checkpoint was trained with ``use_offsets=False``, so the
head's 3 offset channels are dead weights; ``mu = pts3d`` exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from stream_ortho import config as zcfg

C0 = 0.28209479177387814  # SH DC basis constant


# ---------------------------------------------------------------------------
# Model / feature extraction
# ---------------------------------------------------------------------------

def load_model(device: str = "cuda:0", ckpt_path=None):
    """Load the MASt3R backbone exactly as MASt3R-SLAM does."""
    from stream_ortho import compat

    compat.ensure_upstream_on_path()
    from mast3r_slam.mast3r_utils import load_mast3r

    ckpt = str(ckpt_path or zcfg.MAST3R_CKPT)
    model = load_mast3r(ckpt, device=device)
    model.eval()
    return model


@torch.inference_mode()
def encode_image(model, img_u8: np.ndarray, device: str = "cuda:0", img_size: int = 512):
    """Encode an HxWx3 uint8 RGB image the same way MASt3R-SLAM does.

    Returns a dict with ``feat``, ``pos``, ``shape_hw`` (feature-frame H,W) and
    ``uimg`` (the resized uint8 image, for the PSNR target).
    """
    from mast3r_slam.mast3r_utils import resize_img

    res = resize_img(img_u8, img_size)
    img = res["img"].to(device=device)
    shape = torch.tensor(res["true_shape"], device=device)
    feat, pos, _ = model._encode_image(img, shape)
    h, w = res["unnormalized_img"].shape[:2]
    return {
        "feat": feat,
        "pos": pos,
        "shape_hw": (int(h), int(w)),
        "uimg": res["unnormalized_img"],
    }


@torch.inference_mode()
def decoder_tokens(model, feat, pos):
    """``model._decoder`` on the self-pair (keyframe-only hook)."""
    dec1, _ = model._decoder(feat, pos, feat, pos)
    return dec1


@torch.inference_mode()
def predict_gaussians(model, head, feat, pos, shape_hw, anchor="head"):
    """Run the Gaussian head and return per-pixel Gaussian parameters.

    Returns dict of ``(H, W, ·)`` tensors on the compute device:
    ``mu_cam``, ``scales``, ``quats`` (wxyz), ``opacity``, ``sh``, ``conf``,
    ``pts3d`` plus the raw head dict under ``raw``.
    """
    dec1 = decoder_tokens(model, feat, pos)
    out = head(dec1, tuple(int(x) for x in shape_hw))

    # NOTE: `gaussian_postprocess` permutes to B,H,W,C, so every entry of the
    # returned dict is already NHWC (unlike the raw DPT output which is NCHW).
    def take(key):
        t = out[key]
        if isinstance(t, (list, tuple)):
            t = t[0]
        return t[0]                      # drop batch -> (H, W, C)

    res = {
        "mu_cam": take("means").contiguous(),        # (H,W,3)  (== pts3d, use_offsets=False)
        "pts3d": take("pts3d").contiguous(),         # (H,W,3)
        "scales": take("scales").contiguous(),       # (H,W,3)
        "quats": take("rotations").contiguous(),     # (H,W,4) wxyz
        "opacity": take("opacities").contiguous(),   # (H,W,1)
        "sh": take("sh")[..., 0].contiguous(),       # (H,W,3,1) -> (H,W,3) SH DC
        "conf": take("conf").unsqueeze(-1).contiguous(),  # (H,W,1)
        "raw": out,
    }
    return res


# ---------------------------------------------------------------------------
# Scale handling
# ---------------------------------------------------------------------------

def reprojection_error_px(mu_cam: torch.Tensor, K: np.ndarray) -> dict:
    """Median/p95 distance (px) between ``mu_cam``'s xy and the pixel it came from.

    MASt3R's feed-forward point maps are **not** consistent with the intrinsics
    in ``config.cfg``: measured median errors are 5-36 px on NPU aerial frames
    (the point map behaves like a camera with a ~1.2x longer focal and a
    different principal point).  Depth ``z`` is what we trust; xy must be
    replaced by the pixel ray -- see :func:`constrain_to_rays`.
    """
    H, W = mu_cam.shape[:2]
    z = mu_cam[..., 2].clamp_min(1e-6)
    u = mu_cam[..., 0] / z * float(K[0, 0]) + float(K[0, 2])
    v = mu_cam[..., 1] / z * float(K[1, 1]) + float(K[1, 2])
    uu = torch.arange(W, device=mu_cam.device).view(1, W).expand(H, W)
    vv = torch.arange(H, device=mu_cam.device).view(H, 1).expand(H, W)
    e = ((u - uu) ** 2 + (v - vv) ** 2).sqrt()
    return {"median_px": float(e.median()), "p95_px": float(e.flatten().quantile(0.95))}


def fit_effective_intrinsics(mu_cam: torch.Tensor, K: np.ndarray) -> dict:
    """Least-squares fit of ``pixel = a * (xy/z) + b`` -> effective f, c."""
    import numpy as _np

    H, W = mu_cam.shape[:2]
    z = mu_cam[..., 2].clamp_min(1e-6)
    out = {}
    for axis, xy, f, c, size in (("x", 0, K[0, 0], K[0, 2], W), ("y", 1, K[1, 1], K[1, 2], H)):
        n = (mu_cam[..., xy] / z).detach().cpu().numpy().ravel()
        grid = _np.arange(size, dtype=_np.float64).repeat(H if axis == "x" else 1)
        if axis == "x":
            g = _np.tile(_np.arange(W, dtype=_np.float64), H)
        else:
            g = _np.repeat(_np.arange(H, dtype=_np.float64), W)
        A = _np.vstack([n, _np.ones_like(n)]).T
        (a, b), *_ = _np.linalg.lstsq(A, g, rcond=None)
        resid = A @ _np.array([a, b]) - g
        out[axis] = {
            "f_eff": float(a), "c_eff": float(b),
            "f_ratio_vs_K": float(a / float(f)),
            "c_shift_px": float(b - float(c)),
            "resid_median_px": float(_np.median(_np.abs(resid))),
        }
    return out


def constrain_to_rays(mu_cam: torch.Tensor, K: np.ndarray) -> torch.Tensor:
    """Keep ``z`` and rebuild ``xy`` from each pixel's ray (OpenCV camera).

    This is the same operation MASt3R-SLAM applies in its ``use_calib`` path
    (``geometry.constrain_points_to_ray``); without it, rendering the point map
    through ``K`` is off by tens of pixels and both the perspective PSNR and
    the ortho registration degrade.
    """
    H, W = mu_cam.shape[:2]
    z = mu_cam[..., 2].clamp_min(1e-6)
    uu = torch.arange(W, device=mu_cam.device).view(1, W).expand(H, W)
    vv = torch.arange(H, device=mu_cam.device).view(H, 1).expand(H, W)
    x = (uu - float(K[0, 2])) / float(K[0, 0]) * z
    y = (vv - float(K[1, 2])) / float(K[1, 1]) * z
    return torch.stack([x, y, z], dim=-1)


def pixel_footprint(depth: torch.Tensor, K: np.ndarray) -> torch.Tensor:
    """Metric size of one pixel at ``depth`` (metres), per axis -> (H,W,3)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    s = depth / fx
    return torch.cat([s, depth / fy, s], dim=-1)


def rescale(scales: torch.Tensor, mode: str, depth: torch.Tensor, K: np.ndarray) -> torch.Tensor:
    """Scale reparameterisation (design doc S3 / §2.2).

    ``raw``    keep the head output;
    ``pixel``  multiply by the pixel footprint ``depth / f`` (the head's scales
               are then relative), which is the fallback if raw scales are
               far too small (Splatt3R initialises at ~9e-4).
    """
    if mode == "raw":
        return scales.clamp_min(1e-6)
    if mode == "pixel":
        return (scales * pixel_footprint(depth, K)).clamp_min(1e-8)
    raise ValueError(f"unknown scale mode {mode}")


# ---------------------------------------------------------------------------
# Rendering (gsplat)
# ---------------------------------------------------------------------------

def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    return (sh_dc * C0 + 0.5).clamp(0.0, 1.0)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


@torch.inference_mode()
def render_perspective(means, scales, quats, opacity, colors, K, H, W,
                       viewmat=None, render_mode="RGB"):
    """Pinhole rasterisation in the camera frame (identity view matrix)."""
    from gsplat import rasterization

    device = means.device
    n = means.shape[0]
    viewmats = torch.eye(4, device=device)[None] if viewmat is None else viewmat
    Ks = torch.as_tensor(np.asarray(K, dtype=np.float32), device=device)[None]
    img, alpha, meta = rasterization(
        means.reshape(-1, 3).contiguous(),
        quats.reshape(-1, 4).contiguous(),
        scales.reshape(-1, 3).contiguous(),
        opacity.reshape(-1).contiguous(),
        colors.reshape(-1, colors.shape[-1]).contiguous(),
        viewmats,
        Ks,
        int(W),
        int(H),
        camera_model="pinhole",
        sh_degree=None,
        packed=True,
        render_mode=render_mode,
        rasterize_mode="antialiased",
    )
    return img, alpha, meta


@torch.inference_mode()
def render_ortho(means, scales, quats, opacity, colors, viewmat, K, H, W):
    """Orthographic rasterisation (``fx = pixels per metre``)."""
    from gsplat import rasterization

    device = means.device
    Ks = torch.as_tensor(np.asarray(K, dtype=np.float32), device=device)[None]
    img, alpha, meta = rasterization(
        means.reshape(-1, 3).contiguous(),
        quats.reshape(-1, 4).contiguous(),
        scales.reshape(-1, 3).contiguous(),
        opacity.reshape(-1).contiguous(),
        colors.reshape(-1, 3).contiguous(),
        viewmat,
        Ks,
        int(W),
        int(H),
        camera_model="ortho",
        sh_degree=None,
        packed=True,
        rasterize_mode="antialiased",
    )
    return img, alpha, meta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    return float(10.0 * np.log10(1.0 / max(mse, eps)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Grayscale SSIM (11x11 Gaussian window), numpy only."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
        b = b.mean(axis=-1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    # simple 11x11 uniform-window moments (fast, stable, good enough for reports)
    from scipy.ndimage import uniform_filter

    mu_a = uniform_filter(a, 11)
    mu_b = uniform_filter(b, 11)
    va = uniform_filter(a * a, 11) - mu_a ** 2
    vb = uniform_filter(b * b, 11) - mu_b ** 2
    cab = uniform_filter(a * b, 11) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * cab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)
    return float((num / den).mean())
