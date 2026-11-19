"""Splatt3r adapter: MASt3r-anchored & edge-aware sub-pixel 3D Gaussians.

Two entry points:

* :func:`mast3r_anchored_gaussians` — preserved API for P5.1a/b/c tests & scripts.
* :func:`lift_keyframe_gaussians` — high-resolution streaming path that:
  1. Reuses MASt3R-SLAM's encoder ``(feat, pos)`` directly (no redundant ViT pass);
  2. Upsamples canonical depth with edge-preserving interpolation (prevents
     fake intermediate roof-to-ground floating points) and culls depth-jump curtains;
  3. Lifts native-resolution RGB pixels adaptively (``stride=1`` on edges/textures,
     ``flat_stride`` on smooth regions) with footprint-matched anisotropic scales;
  4. Computes a per-Gaussian ``nadir_score`` for winner-take-best voxel consensus.
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
from stream_ortho.frontend.predict import (
    encode_image,
    fit_effective_intrinsics,
    load_model,
    predict_gaussians,
)
from stream_ortho.geometry import sim3_from_vec8, transform_gaussians_to_world

SH_C0 = 0.28209479177387814


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> SH DC coefficient (the 3DGS convention)."""
    return (rgb - 0.5) / SH_C0


def sample_fullres_colors(img_u8: np.ndarray, H: int, W: int, device: str = "cuda:0") -> torch.Tensor:
    """Sample ``img_u8`` (H0,W0,3) uint8 RGB at the ``H``x``W`` head grid, bilinearly."""
    H0, W0 = img_u8.shape[:2]
    img_f = torch.as_tensor(img_u8, dtype=torch.float32, device=device) / 255.0
    uu = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    vv = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    gx = ((uu + 0.5) * W0 / W - 0.5) / max(W0 - 1, 1) * 2 - 1
    gy = ((vv + 0.5) * H0 / H - 0.5) / max(H0 - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)                       # (H, W, 2)
    col = F.grid_sample(img_f.permute(2, 0, 1)[None], grid[None], mode="bilinear",
                        padding_mode="border", align_corners=True)
    return col[0].permute(1, 2, 0)


def _sh_dc_from_image(img_u8: np.ndarray, H: int, W: int, device: str = "cuda:0") -> torch.Tensor:
    return RGB2SH(sample_fullres_colors(img_u8, H, W, device))


@dataclass
class Gaussians:
    """MASt3r-anchored Gaussian representation.

    ``means`` are in world frame (for legacy callers) or camera/world frame;
    ``nadir_score`` & ``cam_rays_w`` are populated by :func:`lift_keyframe_gaussians`
    for nadir-priority streaming consensus.
    """
    means: torch.Tensor                         # (N, 3) world-frame XYZ
    scales: torch.Tensor                        # (N, 3) linear scales (world units)
    quats: torch.Tensor                         # (N, 4) wxyz quaternions
    opacity: torch.Tensor                       # (N, 1) opacity in [0,1]
    sh_dc: torch.Tensor                         # (N, 3) SH DC
    keyframe_id: int
    confidence: Optional[torch.Tensor] = None   # (N, 1) head confidence
    nadir_score: Optional[torch.Tensor] = None  # (N,) view quality score
    cam_rays_w: Optional[torch.Tensor] = None   # (N, 3) unit viewing rays in world
    depth_map: Optional[np.ndarray] = None      # (h, w) canonical camera depth for free-space carving
    K_depth: Optional[tuple] = None             # (fx, fy, cx, cy) at depth_map resolution
    med_depth_cam: float = 0.0                  # median camera depth in canonical units


def _upsample_depth_edge_aware(d_h: np.ndarray, H0: int, W0: int,
                               mode: str = "edge", edge_tau: float = 0.05,
                               return_edge: bool = False,
                               d_edge: Optional[np.ndarray] = None):
    """Upsample (h, w) depth to (H0, W0) without inventing roof-to-ground bridges."""
    import cv2
    from scipy.ndimage import maximum_filter, minimum_filter

    d_h = np.asarray(d_h, np.float32)
    rng3 = maximum_filter(d_h, size=3, mode="nearest") - minimum_filter(d_h, size=3, mode="nearest")
    edge_jump = cv2.resize(rng3, (W0, H0), interpolation=cv2.INTER_NEAREST) > float(edge_tau)
    # True 7x7 canonical depth range on un-snapped depth catches multi-pixel smoothed
    # wall/roof depth ramps; only flag the background / lower side so foreground roofs keep 1.0x.
    d_ref = np.asarray(d_h if d_edge is None else d_edge, np.float32)
    d_min7 = minimum_filter(d_ref, size=7, mode="nearest")
    d_max7 = maximum_filter(d_ref, size=7, mode="nearest")
    rng7 = d_max7 - d_min7
    bg_jump7 = ((rng7 > float(edge_tau)) & ((d_ref - d_min7) > 0.18 * rng7)).astype(np.uint8)
    edge_jump_wide = cv2.resize(bg_jump7, (W0, H0), interpolation=cv2.INTER_NEAREST) > 0

    if d_h.shape == (H0, W0):
        out = d_h
    elif mode == "nearest":
        out = cv2.resize(d_h, (W0, H0), interpolation=cv2.INTER_NEAREST)
    elif mode == "linear":
        out = cv2.resize(d_h, (W0, H0), interpolation=cv2.INTER_LINEAR)
    else:
        # Edge-preserving: bilinear on smooth regions, nearest at depth jumps
        bil = cv2.resize(d_h, (W0, H0), interpolation=cv2.INTER_LINEAR)
        near = cv2.resize(d_h, (W0, H0), interpolation=cv2.INTER_NEAREST)
        out = np.where(edge_jump, near, bil)
    return (out, edge_jump_wide) if return_edge else out


def _resolve_fullres_K(X_canon: np.ndarray, K_orig: Optional[np.ndarray],
                       H0: int, W0: int) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) in full-resolution (W0, H0) pixel coordinates."""
    if K_orig is not None:
        K = np.asarray(K_orig, np.float64)
        if K.shape == (3, 3) and K[0, 0] > 1.0 and K[1, 1] > 1.0:
            return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    h, w = X_canon.shape[:2]
    K_guess = np.array([[0.8 * w, 0.0, 0.5 * w],
                        [0.0, 0.8 * w, 0.5 * h],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    eff = fit_effective_intrinsics(torch.as_tensor(X_canon, dtype=torch.float32), K_guess)
    sx, sy = W0 / float(w), H0 / float(h)
    fx = max(float(eff["x"]["f_eff"]) * sx, 10.0)
    fy = max(float(eff["y"]["f_eff"]) * sy, 10.0)
    cx = float(eff["x"]["c_eff"]) * sx
    cy = float(eff["y"]["c_eff"]) * sy
    return fx, fy, cx, cy


@torch.inference_mode()
def lift_keyframe_gaussians(
    img_u8: np.ndarray,
    X_canon: np.ndarray,
    sim3_vec: np.ndarray,
    keyframe_id: int,
    *,
    feat: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    shape_hw: Optional[tuple[int, int]] = None,
    K_orig: Optional[np.ndarray] = None,
    model=None,
    head=None,
    conf_threshold: float = 0.5,
    adaptive_stride: bool = True,
    flat_stride: int = 2,
    edge_grad_pct: float = 70.0,
    depth_upsample: str = "edge",
    edge_rel_tau: float = 0.03,
    edge_reject: int = 1,
    edge_pct: float = 95.0,
    slam_conf: Optional[np.ndarray] = None,
    device: str = "cuda:0",
) -> Gaussians:
    """Lift one keyframe into native-resolution, edge-aware MASt3R-anchored Gaussians.

    Reuses ``(feat, pos)`` from MASt3R-SLAM when available (avoiding a redundant
    ViT-Large encode pass) and lifts full-resolution RGB pixels adaptively:
    ``stride=1`` on high-gradient edges/textures and ``flat_stride`` on smooth regions.
    """
    import cv2

    if model is None:
        model = load_model(device=device)
    if head is None:
        head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=device, verbose=False)

    if feat is not None and pos is not None and shape_hw is not None:
        f_t = feat.to(device=device)
        p_t = pos.to(device=device)
        if f_t.ndim == 2:
            f_t = f_t.unsqueeze(0)
        if p_t.ndim == 2:
            p_t = p_t.unsqueeze(0)
        pred = predict_gaussians(model, head, f_t, p_t, shape_hw, anchor="head")
    else:
        enc = encode_image(model, img_u8, device=device, img_size=512)
        pred = predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"], anchor="head")

    scales_h = pred["scales"]     # (h, w, 3)
    quats_h = pred["quats"]       # (h, w, 4) wxyz
    opacity_h = pred["opacity"]   # (h, w, 1)
    sh_res_h = pred["sh"]         # (h, w, 3)
    conf_h = pred["conf"]         # (h, w, 1)
    h, w = scales_h.shape[:2]

    X_np = np.asarray(X_canon, dtype=np.float32).reshape(h, w, 3)
    d_h = X_np[..., 2].copy()
    fin = np.isfinite(d_h) & (d_h > 1e-4)
    if not fin.any():
        return Gaussians(
            means=torch.zeros((0, 3), dtype=torch.float32, device=device),
            scales=torch.zeros((0, 3), dtype=torch.float32, device=device),
            quats=torch.zeros((0, 4), dtype=torch.float32, device=device),
            opacity=torch.zeros((0, 1), dtype=torch.float32, device=device),
            sh_dc=torch.zeros((0, 3), dtype=torch.float32, device=device),
            keyframe_id=keyframe_id,
            confidence=torch.zeros((0, 1), dtype=torch.float32, device=device),
            nadir_score=torch.zeros((0,), dtype=torch.float32, device=device),
            cam_rays_w=torch.zeros((0, 3), dtype=torch.float32, device=device),
        )

    med_d = float(np.median(d_h[fin]))
    d_h = np.clip(np.where(fin, d_h, med_d), 0.25 * med_d, 3.0 * med_d)
    # Remove isolated 1-px flying depth spikes without blurring roof edges
    from scipy.ndimage import maximum_filter, median_filter, minimum_filter
    d_med3 = median_filter(d_h, size=3, mode="nearest")
    spike = np.abs(d_h - d_med3) > 0.08 * med_d
    d_h = np.where(spike, d_med3, d_h)

    H0, W0 = img_u8.shape[:2]
    # Calibrated MASt3R-SLAM tracks with constrain_points_to_ray(..., K_orig); uncalibrated
    # falls back to fit_effective_intrinsics(X_canon). Matching that exact K keeps poses aligned.
    fx0, fy0, cx0, cy0 = _resolve_fullres_K(X_np, K_orig, H0, W0)

    # Compute smooth surface normals in camera frame on the (h, w) depth grid
    # and identify grazing rubber-sheet depth ramps across building walls/eaves.
    h_g, w_g = X_np.shape[:2]
    uu_g, vv_g = np.meshgrid(np.arange(w_g, dtype=np.float32),
                             np.arange(h_g, dtype=np.float32))
    fx_h = float(fx0) * (w_g / float(W0))
    fy_h = float(fy0) * (h_g / float(H0))
    cx_h = float(cx0) * (w_g / float(W0))
    cy_h = float(cy0) * (h_g / float(H0))
    P_h = np.stack([
        (uu_g - cx_h) / max(fx_h, 1.0) * d_h,
        (vv_g - cy_h) / max(fy_h, 1.0) * d_h,
        d_h,
    ], axis=-1)
    tu = np.zeros_like(P_h)
    tv = np.zeros_like(P_h)
    tu[:, 1:-1] = P_h[:, 2:] - P_h[:, :-2]
    tu[:, 0] = P_h[:, 1] - P_h[:, 0]
    tu[:, -1] = P_h[:, -1] - P_h[:, -2]
    tv[1:-1, :] = P_h[2:, :] - P_h[:-2, :]
    tv[0, :] = P_h[1, :] - P_h[0, :]
    tv[-1, :] = P_h[-1, :] - P_h[-2, :]
    n_h = np.cross(tv, tu)  # points toward -Z_cam (toward camera)
    n_len = np.linalg.norm(n_h, axis=-1, keepdims=True)
    ray_back = -P_h / np.maximum(np.linalg.norm(P_h, axis=-1, keepdims=True), 1e-6)
    n_h = np.where(n_len < 1e-8, ray_back, n_h / np.maximum(n_len, 1e-8))
    dot_cam = np.sum(n_h * ray_back, axis=-1, keepdims=True)
    n_h = np.where(dot_cam < 0.0, -n_h, n_h)
    dot_cam = np.sum(n_h * ray_back, axis=-1, keepdims=True)
    dot_cam_2d = dot_cam[..., 0]

    # Foreground-restoring shock filter on smoothed depth-jump ramps (dot_cam < 0.55):
    # pulls upper-half roof-eave pixels onto foreground roof depth (d_min5) without
    # lifting wall/courtyard pixels out onto the roof; edge attenuation still uses d_orig.
    edge_tau = float(edge_rel_tau) * med_d
    d_orig = d_h.copy()
    d_min5 = minimum_filter(d_orig, size=5, mode="nearest")
    d_max5 = maximum_filter(d_orig, size=5, mode="nearest")
    is_ramp = ((d_max5 - d_min5) > edge_tau) & (dot_cam_2d < 0.55)
    d_h = np.where(
        is_ramp,
        np.where((d_orig - d_min5) <= 0.50 * (d_max5 - d_min5), d_min5, d_max5),
        d_orig,
    )

    # At steep depth-jump steps (|dot_cam| < 0.6), fall back to camera-facing normal
    # so surfels at roof borders stay flat rather than standing as vertical curtains.
    n_h = np.where(dot_cam < 0.6, ray_back, n_h)
    n_full = cv2.resize(n_h.astype(np.float32), (W0, H0), interpolation=cv2.INTER_LINEAR)
    n_full /= np.maximum(np.linalg.norm(n_full, axis=-1, keepdims=True), 1e-8)

    # Depth-discontinuity edge band on the (h, w) canonical depth grid:
    # compute background-ramp membership on un-snapped d_orig so steep wall ramps
    # yield cleanly to overlapping interior ground/courtyard views.
    edge_band_full = np.zeros((H0, W0), dtype=bool)
    if edge_reject > 0:
        inv_d = (1.0 / np.maximum(d_orig, 1e-6)).astype(np.float32)
        mag_d = np.hypot(cv2.Sobel(inv_d, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(inv_d, cv2.CV_32F, 0, 1, ksize=3))
        thr_d = float(np.percentile(mag_d, edge_pct))
        is_bg_side = (d_orig - d_min5) > 0.18 * (d_max5 - d_min5)
        band = (((mag_d >= thr_d) | (dot_cam_2d < 0.55)) & is_bg_side).astype(np.uint8)
        edge_band_full = cv2.resize(band, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

    d_full, edge_jump_full = _upsample_depth_edge_aware(
        d_h, H0, W0, mode=depth_upsample, edge_tau=edge_tau, return_edge=True, d_edge=d_orig
    )
    edge_band_full = edge_band_full | edge_jump_full

    # Adaptive gradient-driven sub-pixel sampling mask (no hard moat deletion)
    s_flat = max(int(flat_stride), 1)
    base_mask = np.zeros((H0, W0), dtype=bool)
    base_mask[::s_flat, ::s_flat] = True
    if adaptive_stride and s_flat > 1:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gmag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        hi_tex = gmag >= float(np.percentile(gmag, edge_grad_pct))
        sample_mask = base_mask | hi_tex | edge_band_full
        # Local spacing factor (1 px on promoted high-texture pixels, s_flat on smooth base)
        spacing_map = np.where(hi_tex | edge_band_full, 1.0, float(s_flat)).astype(np.float32)
    else:
        sample_mask = base_mask
        spacing_map = np.full((H0, W0), float(s_flat), dtype=np.float32)

    vv_np, uu_np = np.nonzero(sample_mask)
    if vv_np.size == 0:
        vv_np, uu_np = np.nonzero(base_mask)
    local_sp = spacing_map[vv_np, uu_np]

    # Unproject sampled native pixels using matched intrinsics + edge-aware upsampled depth
    z_np = d_full[vv_np, uu_np].astype(np.float32)
    x_np = ((uu_np.astype(np.float32) - np.float32(cx0)) / np.float32(fx0)) * z_np
    y_np = ((vv_np.astype(np.float32) - np.float32(cy0)) / np.float32(fy0)) * z_np

    means_cam = torch.from_numpy(np.stack([x_np, y_np, z_np], axis=-1)).to(device=device)
    sp_t = torch.from_numpy(local_sp).to(device=device)

    # Bilinearly sample Splatt3R appearance tensors at (uu_np, vv_np)
    uu_t = torch.from_numpy(uu_np.astype(np.float32)).to(device=device)
    vv_t = torch.from_numpy(vv_np.astype(np.float32)).to(device=device)
    gx = ((uu_t + 0.5) / float(W0) * float(w) - 0.5) / max(float(w - 1), 1.0) * 2.0 - 1.0
    gy = ((vv_t + 0.5) / float(H0) * float(h) - 0.5) / max(float(h - 1), 1.0) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)

    def _sample_hwc(t_hwc: torch.Tensor) -> torch.Tensor:
        out = F.grid_sample(
            t_hwc.permute(2, 0, 1).unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True
        )
        return out[0, :, 0, :].transpose(0, 1).contiguous()

    scales_s = _sample_hwc(scales_h)
    quats_s = _sample_hwc(quats_h)
    quats_s = quats_s / quats_s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    opac_s = _sample_hwc(opacity_h)
    sh_res_s = _sample_hwc(sh_res_h)
    conf_s = _sample_hwc(conf_h)

    # Keep smooth low-texture regions (e.g. water/pavement) hole-free; only drop
    # degenerate non-positive predictions and soft-downweight low conf_s below.
    keep = torch.isfinite(conf_s[:, 0]) & (conf_s[:, 0] > 0.0)
    if not keep.any():
        keep = torch.ones_like(keep)

    means_cam = means_cam[keep]
    scales_s = scales_s[keep]
    quats_s = quats_s[keep]
    opac_s = opac_s[keep]
    sh_res_s = sh_res_s[keep]
    conf_s = conf_s[keep]
    sp_t = sp_t[keep]
    uu_k = uu_np[keep.cpu().numpy()]
    vv_k = vv_np[keep.cpu().numpy()]

    # Native full-resolution RGB -> SH DC
    rgb_u8 = img_u8[vv_k, uu_k]
    rgb_f = torch.from_numpy(rgb_u8).to(device=device, dtype=torch.float32) / 255.0
    sh_dc = (RGB2SH(rgb_f) + sh_res_s).clamp(-3.0, 3.0)

    # Build surface-aligned surfel quaternions directly on GPU (local Z == surface normal)
    ez = torch.from_numpy(n_full[vv_k, uu_k]).to(device=device, dtype=torch.float32)
    ez = ez / ez.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ref = torch.zeros_like(ez)
    use_y = ez[:, 0].abs() > 0.9
    ref[~use_y, 0] = 1.0
    ref[use_y, 1] = 1.0
    ey = torch.cross(ez, ref, dim=-1)
    ey = ey / ey.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ex = torch.cross(ey, ez, dim=-1)
    # Convert R_surf = [ex, ey, ez] to wxyz quaternion (branch-stable Shepperd formula)
    m00, m11, m22 = ex[:, 0], ey[:, 1], ez[:, 2]
    tr = m00 + m11 + m22
    q0 = torch.stack([tr + 1.0, ez[:, 1] - ey[:, 2], ex[:, 2] - ez[:, 0], ey[:, 0] - ex[:, 1]], dim=-1)
    q1 = torch.stack([ez[:, 1] - ey[:, 2], 1.0 + m00 - m11 - m22, ey[:, 0] + ex[:, 1], ez[:, 0] + ex[:, 2]], dim=-1)
    q2 = torch.stack([ex[:, 2] - ez[:, 0], ey[:, 0] + ex[:, 1], 1.0 + m11 - m00 - m22, ez[:, 1] + ey[:, 2]], dim=-1)
    q3 = torch.stack([ey[:, 0] - ex[:, 1], ez[:, 0] + ex[:, 2], ez[:, 1] + ey[:, 2], 1.0 + m22 - m00 - m11], dim=-1)
    # Pick branch with largest diagonal sum for numerical stability
    diag_stack = torch.stack([tr, m00, m11, m22], dim=-1)
    best_b = torch.argmax(diag_stack, dim=-1)
    quats_surf = torch.where(
        (best_b == 0).unsqueeze(-1), q0,
        torch.where((best_b == 1).unsqueeze(-1), q1,
                    torch.where((best_b == 2).unsqueeze(-1), q2, q3))
    )
    quats_surf = quats_surf / quats_surf.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Footprint-matched disk scales covering native pixel spacing without pinholes or blur
    fp_px = (means_cam[:, 2] / float(fx0)) * sp_t
    s_xy = (1.05 * fp_px).unsqueeze(-1).expand(-1, 2)
    s_z = (0.15 * fp_px).unsqueeze(-1).clamp_min(1e-7)
    scales_cam = torch.cat([s_xy, s_z], dim=-1).clamp_min(1e-7)

    # Transform to world frame via Sim(3)
    means_world, quats_world, scales_world = transform_gaussians_to_world(
        means_cam, quats_surf, scales_cam, sim3_vec, device=device
    )

    # Compute world-frame viewing rays & radial/confidence quality prior
    R_wc, _, s_sim3 = sim3_from_vec8(sim3_vec)
    R_wc_t = torch.as_tensor(R_wc, dtype=torch.float32, device=device)
    rays_cam = means_cam / means_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cam_rays_w = rays_cam @ R_wc_t.T

    # Radial FOV centrality weight: center of frame has least distortion & cleanest geometry
    ux_n = (torch.from_numpy(uu_k.astype(np.float32)).to(device) - float(cx0)) / max(0.5 * W0, 1.0)
    vy_n = (torch.from_numpy(vv_k.astype(np.float32)).to(device) - float(cy0)) / max(0.5 * H0, 1.0)
    radial_w = torch.exp(-0.5 * (ux_n * ux_n + vy_n * vy_n))
    # Soft-clamp confidence at 1.0 above conf_threshold so high-frequency neural
    # confidence fluctuations on roof structures (solar panels, AC units, skylights)
    # cannot override geometric nadir centrality and flip-flop keyframe ownership.
    conf_thr = max(float(conf_threshold), 0.1)
    conf_w = (conf_s[:, 0] / conf_thr).clamp(0.1, 1.0)
    if float(conf_threshold) > 0.0:
        conf_w = torch.where(conf_s[:, 0] < float(conf_threshold), conf_w * 0.25, conf_w)
    if slam_conf is not None:
        sc_arr = np.asarray(slam_conf, dtype=np.float32).reshape(h, w)
        sc_pos = sc_arr[np.isfinite(sc_arr) & (sc_arr > 0.0)]
        med_sc = float(np.median(sc_pos)) if sc_pos.size else 1.0
        sc_rel = np.clip(np.where(np.isfinite(sc_arr), sc_arr / max(med_sc, 1e-4), 1.0), 0.1, 1.0)
        sc_full = cv2.resize(sc_rel.astype(np.float32), (W0, H0), interpolation=cv2.INTER_LINEAR)
        sc_t = torch.from_numpy(sc_full[vv_k, uu_k]).to(device=device, dtype=torch.float32)
        conf_w = conf_w * sc_t.sqrt().clamp(0.35, 1.0)
    # Finer ground GSD (smaller fp_px * s_sim3) gets higher quality, guarded against landing collapse
    gsd_m = (means_cam[:, 2] / float(fx0)) * max(float(s_sim3), 1e-6)
    base_score = (radial_w * radial_w) * conf_w / gsd_m.clamp_min(1e-5)
    # Strongly attenuate depth-discontinuity pixels so interior views own ground/roof
    # boundaries cleanly without red roof/wall bleed or leaving black moats.
    is_edge_t = torch.from_numpy(edge_band_full[vv_k, uu_k]).to(device=device)
    base_score = torch.where(is_edge_t, base_score * 0.01, base_score)

    return Gaussians(
        means=means_world,
        scales=scales_world,
        quats=quats_world,
        opacity=opac_s.clamp(1e-4, 1.0),
        sh_dc=sh_dc,
        keyframe_id=keyframe_id,
        confidence=conf_s,
        nadir_score=base_score,
        cam_rays_w=cam_rays_w,
        depth_map=d_h.astype(np.float32, copy=False),
        K_depth=(float(fx_h), float(fy_h), float(cx_h), float(cy_h)),
        med_depth_cam=float(med_d),
    )


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
    """Create MASt3r-anchored Gaussians from one keyframe (legacy P5.1a/b/c API)."""
    if model is None:
        model = load_model(device=device)
    if head is None:
        head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=device, verbose=False)

    enc = encode_image(model, img_u8, device=device, img_size=512)
    pred = predict_gaussians(
        model, head, enc["feat"], enc["pos"], enc["shape_hw"], anchor="head"
    )

    scales_raw = pred["scales"]
    quats = pred["quats"]
    opacity = pred["opacity"]
    sh_residual = pred["sh"]
    conf = pred["conf"]

    H, W = scales_raw.shape[:2]

    X_canon_torch = torch.as_tensor(X_canon, dtype=torch.float32, device=device)
    if X_canon_torch.shape != (H, W, 3):
        raise ValueError(f"X_canon shape {X_canon.shape} != expected {(H, W, 3)}")

    means_cam = X_canon_torch
    scales = scales_raw.clamp_min(1e-8)

    means_world, quats_world, scales_world = transform_gaussians_to_world(
        means_cam, quats, scales, sim3_vec, device=device
    )

    sh_dc = _sh_dc_from_image(img_u8, H, W, device) + sh_residual
    sh_dc = sh_dc.clamp(-3.0, 3.0)

    N = H * W
    means_world = means_world.reshape(N, 3)
    scales_world = scales_world.reshape(N, 3)
    quats_world = quats_world.reshape(N, 4)
    opacity_flat = opacity.reshape(N, 1)
    sh_dc_flat = sh_dc.reshape(N, 3)
    conf_flat = conf.reshape(N, 1)

    means_np = means_world.detach().cpu().numpy()
    conf_np = conf_flat.detach().cpu().numpy().ravel()

    means_homo = np.concatenate([means_np, np.ones((N, 1))], axis=1)
    T_PW = np.linalg.inv(T_WP)
    means_plane = (T_PW @ means_homo.T).T[:, :3]

    x_plane, y_plane, z_plane = means_plane[:, 0], means_plane[:, 1], means_plane[:, 2]

    if filter_cfg is None:
        x_min, y_min = float(x_plane.min()), float(y_plane.min())
        cell_size = 0.45
        origin_x = np.floor(x_min / cell_size) * cell_size - cell_size
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
