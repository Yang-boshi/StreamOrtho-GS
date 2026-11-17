"""Build a :class:`GaussianPatch` from one keyframe (plane-splat or head patch).

Both paths share the same pose chain: ``camera -> SLAM world -> GT world`` with
the est->GT Sim3 from ``run_stage_s2_ate.py``.  The difference is only *what the
Gaussian means/colours are*:

* ``plane`` (fallback B, the week-4 main path): ray x ground plane, exact on
  flat terrain, splat radius = 1.5 source pixels;
* ``head`` (design-doc path, still under investigation -- week 3 §3.2): the
  feed-forward point map, ray-constrained, depth-aligned to MASt3R-SLAM's map.
"""

from __future__ import annotations

import numpy as np
import torch

from stream_ortho.geometry import compose_sim3, sim3_from_vec8
from stream_ortho.map.gaussian import GaussianPatch


#: splat sigma in units of the (strided) source-pixel footprint.  1.5 was chosen in
#: week 5 to avoid alpha<1 gaps (a smaller value darkens the mosaic); it also caps
#: the achievable sharpness -- see ``diagnose_blur`` block (D).
#: last per-frame depth calibration (a, b) fitted by ``head_adapt_patch``
LAST_AFFINE: dict = {"a": 1.0, "b": 0.0}

SPLAT_RADIUS_FACTOR = 1.0


def _splat_radius(depth_m: float, fx: float, stride: int) -> float:
    return max(depth_m / float(fx) * stride * SPLAT_RADIUS_FACTOR, 1e-3)


def plane_patch(ds, npz, k, sim3_vec_gt, stride: int = 3, device: str = "cuda:0",
                shape: str = "iso", edge_cfg=None, T_gt_wc=None,
                world_offset=None, dsm=None, ray_dsm: bool = False,
                cull_facade: float = 0.0) -> GaussianPatch:
    """One keyframe -> Gaussians on the ground plane (world frame, metric).

    ``shape="iso"``   : isotropic discs (week-4 baseline);
    ``shape="prior"`` : edge-aware shapes + ``kind``/``theta`` tags (week-5 idea):
    line elements become elongated rods, area points stay flat discs.
    """
    fid = int(npz["frame_id"][k])
    img = ds.read_img(fid)
    K = ds.K_orig
    W, H = int(ds.cfg.width), int(ds.cfg.height)
    uu, vv = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0],
                     (vv - K[1, 2]) / K[1, 1],
                     np.ones_like(uu, dtype=np.float64)], axis=-1).reshape(-1, 3)

    if T_gt_wc is None:
        T_gt_cam = compose_sim3(npz["T_WC"][k], sim3_vec_gt)
    else:                          # explicit world pose (drift simulation, week 7)
        T_gt_cam = np.asarray(T_gt_wc, np.float64)
    R, t, s = sim3_from_vec8(T_gt_cam)
    n = ds.T_WP[:3, 2]
    p0 = ds.T_WP[:3, 3]
    dirs = rays @ R.T
    denom = dirs @ n
    tt = ((p0 - t) @ n) / np.where(np.abs(denom) > 1e-9, denom, np.nan)
    ok = np.isfinite(tt) & (tt > 0)
    pts = t + tt[:, None] * dirs
    if dsm is not None and ray_dsm:
        # iterative ray x DSM: the plane hit gives a lateral offset h*tan(theta) for
        # elevated points; re-intersecting with the DSM at the *corrected* xy removes it
        den = np.where(np.abs(denom) > 1e-9, denom, np.nan)
        ttv = tt.copy()
        for _ in range(3):
            P = (t + ttv[:, None] * dirs - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
            zz = dsm.lookup(P[:, 0], P[:, 1], fill=0.0).astype(np.float64)
            ttv = ((p0 + n[None] * zz[:, None] - t) @ n) / den
            ttv = np.where(np.isfinite(ttv) & (ttv > 0), ttv, tt)
        tt = ttv
        pts = t + tt[:, None] * dirs
        ok = np.isfinite(tt) & (tt > 0)

    if cull_facade > 0.0:
        # E2 facade culling: a pixel whose *own* point-map depth is much smaller than
        # the distance to the DSM/plane hit is looking at a vertical surface (a wall in
        # front of the roof plane).  Such pixels are not visible from above, so drop
        # them instead of smearing their colour onto the ground next to the building.
        X = np.asarray(npz["X_canon"][k], np.float32)
        hh, ww = X.shape[:2]
        xs = np.clip((uu * ww / W).astype(np.int64), 0, ww - 1)
        ys = np.clip((vv * hh / H).astype(np.int64), 0, hh - 1)
        d_self = float(s) * np.linalg.norm(X[ys, xs], axis=-1).reshape(-1)
        d_proxy = np.linalg.norm(t[None, :] + tt[:, None] * dirs - t[None, :], axis=-1)
        # the point map carries a per-frame depth *bias* (metres); without removing it a
        # 1 m threshold culls most of the ground too (measured: coverage dropped to 44%)
        good = np.isfinite(d_self) & (d_self > 0)
        bias = float(np.median(d_self[good] - d_proxy[good])) if good.sum() > 1000 else 0.0
        keep = d_self >= (d_proxy + bias - cull_facade)
        ok = ok & keep

    small = img[::stride, ::stride]
    colors = small.reshape(-1, 3).astype(np.float32) / 255.0
    depth_m = float(np.median(npz["X_canon"][k][..., 2].astype(np.float32)) * s)
    rad = _splat_radius(depth_m, K[0, 0], stride)

    if world_offset is not None:            # week8-end-1 task 2: registration
        pts = pts - np.asarray(world_offset, np.float64).reshape(1, 3)
    if dsm is not None and not ray_dsm:     # lift onto the DSM (z-only, keeps xy)
        P = (pts - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        P[:, 2] = dsm.lookup(P[:, 0], P[:, 1], fill=0.0)
        pts = P @ ds.T_WP[:3, :3] + ds.T_WP[:3, 3]
    # observation quality (Map2DFusion-style): 1 - r/r_max in the image, divided by
    # cos(incidence) so grazing views of the ground count less
    uu_c = (W - 1) * 0.5
    vv_c = (H - 1) * 0.5
    r_img = np.sqrt(((uu - uu_c) / max(uu_c, 1)) ** 2 + ((vv - vv_c) / max(vv_c, 1)) ** 2)
    w_view = np.clip(1.0 - 0.5 * r_img.reshape(-1), 0.05, 1.0)
    cos_inc = np.abs(dirs @ n) / np.maximum(np.linalg.norm(dirs, axis=1), 1e-9)
    weight = (w_view / np.maximum(cos_inc, 0.2)).astype(np.float32)
    n_ok = int(ok.sum())
    # edge tagging is always done (it feeds the budget), independently of the shaping
    from stream_ortho.map.edges import EdgeConfig, edge_map, shape_from_kind

    k2, th2, _ = edge_map(small, edge_cfg or EdgeConfig())
    kind = k2.reshape(-1).astype(np.int8)
    theta = th2.reshape(-1).astype(np.float32)
    if shape == "prior":
        gsd_ground = depth_m / float(K[0, 0]) * stride          # metres per strided pixel
        sc, quats_full = shape_from_kind(kind, theta, gsd_ground, R, n)
        scales = sc.astype(np.float32)
        quats = quats_full.astype(np.float32)
    else:
        scales = np.full((len(colors), 3), rad, np.float32)
        quats = np.zeros((len(colors), 4), np.float32)
        quats[:, 0] = 1.0

    patch = GaussianPatch(
        means=pts[ok], quats=quats[ok], scales=scales[ok],
        opacity=np.ones(n_ok, np.float32), colors=colors[ok],
        conf=np.full(n_ok, 2.0, np.float32),
        kind=kind[ok], theta=theta[ok], weight=weight[ok],
        owner=int(k), kf_id=int(k),
        T_WC=np.asarray(npz["T_WC"][k], np.float64),
        meta={"depth_m": depth_m, "frame_id": fid, "source": "plane",
              "stride": stride, "shape": shape},
    )
    return patch


def head_patch(ds, npz, k, sim3_vec_gt, model, head, align_depth: bool = True,
               device: str = "cuda:0") -> GaussianPatch:
    """One keyframe -> feed-forward Gaussians, mapped into the GT world."""
    from stream_ortho import config as zcfg
    from stream_ortho.frontend import predict as P
    from stream_ortho.geometry import transform_gaussians_to_world

    fid = int(npz["frame_id"][k])
    enc = P.encode_image(model, ds.read_img(fid), device=device)
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    K = ds.K_frame()
    mu = P.constrain_to_rays(g["mu_cam"], K)
    mu_slam = P.constrain_to_rays(
        torch.as_tensor(npz["X_canon"][k], device=mu.device, dtype=torch.float32), K)
    k_align = 1.0
    if align_depth:
        k_align = float((mu_slam[..., 2].median() / mu[..., 2].median()).item())
    mu = mu * k_align

    T_gt_cam = compose_sim3(npz["T_WC"][k], sim3_vec_gt)
    mu_w, q_w, sc_w = transform_gaussians_to_world(
        mu, g["quats"], g["scales"] * k_align, T_gt_cam, device=device)
    _, t, s = sim3_from_vec8(T_gt_cam)
    depth_m = float(g["mu_cam"][..., 2].median()) * k_align * float(s)
    colors = torch.as_tensor(enc["uimg"].astype(np.float32) / 255.0, device=device)

    return GaussianPatch(
        means=mu_w.detach().cpu().numpy(),
        quats=q_w.detach().cpu().numpy(),
        scales=sc_w.detach().cpu().numpy(),
        opacity=g["opacity"].reshape(-1).detach().cpu().numpy(),
        colors=colors.reshape(-1, 3).detach().cpu().numpy(),
        conf=g["conf"].reshape(-1).detach().cpu().numpy(),
        owner=int(k), kf_id=int(k), T_WC=np.asarray(npz["T_WC"][k], np.float64),
        meta={"depth_m": depth_m, "frame_id": fid, "source": "head",
              "k_align": k_align},
    )


def pointmap_patch(ds, npz, k, sim3_vec_gt, stride: int = 2, conf_thr: float = 0.0,
                   shape: str = "iso", align_plane: bool = True) -> GaussianPatch:
    """Place Gaussians at the **per-pixel 3D points** of the SLAM point map.

    This is the design doc's fallback B, kept for one reason: the point map already
    carries the *relative* height of every pixel (measured 20-23 m of relief on npu,
    i.e. buildings really are in it), while the plane/DSM path puts every splat at
    the ground x,y and therefore smears roofs and never produces a facade.

    The absolute scale comes from the est->GT Sim3 (metric); ``align_plane`` orients
    the discs along the ground plane, which is right for roofs and wrong for
    facades -- acceptable because roofs dominate an ortho.
    """
    from stream_ortho.geometry import compose_sim3, sim3_from_vec8

    fid = int(npz["frame_id"][k])
    X = np.asarray(npz["X_canon"][k], np.float32)          # (h,w,3) camera frame, SLAM units
    C = np.asarray(npz["C"][k], np.float32) if "C" in npz.files else None
    R, t, sc = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
    Pw = float(sc) * (X.reshape(-1, 3) @ R.T) + t          # metres, GT world
    depth = float(sc) * X[..., 2].reshape(-1)              # metres along the optical axis
    img = ds.read_img(fid).astype(np.float32) / 255.0      # (H,W,3)
    h, w = X.shape[:2]
    ys = np.linspace(0, img.shape[0] - 1, h).round().astype(int)
    xs = np.linspace(0, img.shape[1] - 1, w).round().astype(int)
    colors = img[np.ix_(ys, xs)].reshape(-1, 3)
    conf = C.reshape(-1) if C is not None else np.ones(depth.shape, np.float32)

    ok = np.isfinite(Pw).all(1) & np.isfinite(depth) & (depth > 1.0) & (conf >= conf_thr)
    sel = np.arange(depth.size)[ok][:: max(stride, 1)]
    Pw, colors, depth = Pw[sel], colors[sel], depth[sel]

    fx = float(ds.K_frame()[0, 0])
    sca = (depth / fx * stride) * np.float32(SPLAT_RADIUS_FACTOR * 1.0)
    if shape == "prior":
        sca = np.stack([sca * 0.55, sca, sca], 1)
    else:
        sca = np.stack([sca, sca, sca], 1)
    n = len(Pw)
    if align_plane:
        from stream_ortho.geometry import R_to_quat_wxyz

        q = R_to_quat_wxyz(ds.T_WP[:3, :3])
        quats = np.repeat(np.asarray(q, np.float32).reshape(1, 4), n, 0)
    else:
        quats = np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1))
    return GaussianPatch(
        means=Pw.astype(np.float32), quats=quats.astype(np.float32),
        scales=sca.astype(np.float32), opacity=np.ones(n, np.float32),
        colors=colors.astype(np.float32), owner=int(fid), conf=conf[sel])


def head_plane_patch(ds, npz, k, sim3_vec_gt, model, head, stride: int = 1,
                     device: str = "cuda:0", T_gt_wc=None) -> GaussianPatch:
    """Hybrid: **plane-constrained means** + head appearance (week-4 finding).

    The head's depth map is too flat (plane/head ratio 1.31 at the image centre,
    1.86 at the corners, i.e. a 1.35x spread), so a per-keyframe *constant* scale
    cannot place it on the ground.  Here the mean comes from the exact
    ray x plane intersection, while colour/opacity/scale still come from the
    head (scale converted to metres and stretched by the same depth ratio so the
    angular footprint is preserved).
    """
    from stream_ortho.frontend import predict as P

    fid = int(npz["frame_id"][k])
    enc = P.encode_image(model, ds.read_img(fid), device=device)
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    K = ds.K_frame()
    H, W = enc["shape_hw"]
    mu = P.constrain_to_rays(g["mu_cam"], K).detach().cpu().numpy()   # (H,W,3)
    z_head = np.maximum(mu[..., 2], 1e-6)

    if T_gt_wc is None:
        T_gt_cam = compose_sim3(npz["T_WC"][k], sim3_vec_gt)
    else:                          # explicit world pose (drift simulation, week 7)
        T_gt_cam = np.asarray(T_gt_wc, np.float64)
    R, t, s = sim3_from_vec8(T_gt_cam)
    n = ds.T_WP[:3, 2]
    p0 = ds.T_WP[:3, 3]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                     np.ones_like(uu, dtype=np.float64)], -1)
    dirs = rays @ R.T
    denom = dirs @ n
    tt = ((p0 - t) @ n) / np.where(np.abs(denom) > 1e-9, denom, np.nan)
    ok = np.isfinite(tt) & (tt > 0)
    pts = t + tt[..., None] * dirs

    depth_plane_m = tt * np.linalg.norm(dirs, axis=-1)          # metric, along the ray
    depth_head_m = z_head * float(s)
    stretch = np.clip(depth_plane_m / np.maximum(depth_head_m, 1e-6), 0.2, 5.0)

    scales = g["scales"].detach().cpu().numpy() * float(s) * stretch[..., None]
    colors = (enc["uimg"].astype(np.float32) / 255.0).reshape(-1, 3)
    ok_flat = ok.reshape(-1)
    n_ok = int(ok_flat.sum())
    quats = g["quats"].detach().cpu().numpy().reshape(-1, 4)[ok_flat]
    patch = GaussianPatch(
        means=pts.reshape(-1, 3)[ok_flat], quats=quats,
        scales=scales.reshape(-1, 3)[ok_flat],
        opacity=g["opacity"].reshape(-1).detach().cpu().numpy()[ok_flat],
        colors=colors[ok_flat],
        conf=g["conf"].reshape(-1).detach().cpu().numpy()[ok_flat],
        owner=int(k), kf_id=int(k), T_WC=np.asarray(npz["T_WC"][k], np.float64),
        meta={"depth_m": float(np.median(depth_plane_m)), "frame_id": fid,
              "source": "head_plane", "stretch_median": float(np.median(stretch))},
    )
    return patch


def fuse_pointmaps(ds, npz, sim3_vec_gt, kfs, stride: int = 2, voxel: float = 0.3,
                   max_resid: float = 2.0, min_conf: float = 0.0) -> GaussianPatch:
    """E1: fuse the keyframes' point maps into one **consensus** cloud.

    Each keyframe's ``X_canon`` is lifted to metric world coordinates and the union is
    averaged per voxel.  This is the missing step before per-pixel placement: it keeps
    the relative height (roofs) but removes the frame-to-frame disagreement that made
    per-pixel placement blurry (each view put the same roof corner 1-3 m apart).

    Two passes: voxel mean, then drop points > ``max_resid`` from their voxel mean and
    recompute -- a cheap robust consensus.
    """
    from stream_ortho.geometry import R_to_quat_wxyz, compose_sim3, sim3_from_vec8

    P_all, C_all = [], []
    for k in kfs:
        fid = int(npz["frame_id"][k])
        X = np.asarray(npz["X_canon"][k], np.float32)[::stride, ::stride].reshape(-1, 3)
        Cf = np.asarray(npz["C"][k], np.float32)[::stride, ::stride].reshape(-1)
        R, t, s = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
        Pw = float(s) * (X @ R.T) + t
        img = ds.read_img(fid).astype(np.float32) / 255.0
        h, w = X.shape[0] * 0 + np.asarray(npz["X_canon"][k]).shape[0], np.asarray(npz["X_canon"][k]).shape[1]
        ys = np.linspace(0, img.shape[0] - 1, h).round().astype(int)
        xs = np.linspace(0, img.shape[1] - 1, w).round().astype(int)
        cols = img[np.ix_(ys, xs)][::stride, ::stride].reshape(-1, 3)
        ok = np.isfinite(Pw).all(1) & (Cf >= min_conf) & (X[:, 2] > 0)
        P_all.append(Pw[ok]); C_all.append(cols[ok])
    if not P_all:
        raise ValueError("no points")
    P = np.concatenate(P_all); C = np.concatenate(C_all)
    origin = np.percentile(P, 0.5, axis=0)
    key = np.floor((P - origin) / voxel).astype(np.int64)
    key = (key[:, 0] << 42) ^ (key[:, 1] << 21) ^ key[:, 2]
    order = np.argsort(key, kind="stable")
    key_s, P_s, C_s = key[order], P[order], C[order]
    uniq, start, counts = np.unique(key_s, return_index=True, return_counts=True)
    sumP = np.add.reduceat(P_s, start, axis=0)
    sumC = np.add.reduceat(C_s, start, axis=0)
    meanP = sumP / counts[:, None]
    # pass 2: robust re-average (drop points > max_resid from their voxel mean)
    lab = np.repeat(np.arange(uniq.size), counts)
    resid = np.linalg.norm(P_s - meanP[lab], axis=1)
    good = resid <= max_resid
    meanP2 = np.add.reduceat(P_s * good[:, None], start, axis=0) / np.maximum(
        np.add.reduceat(good.astype(np.float64), start), 1)[:, None]
    meanC = sumC / counts[:, None]
    n = len(meanP2)
    sca = np.full((n, 3), voxel * 0.8, np.float32)
    q = R_to_quat_wxyz(ds.T_WP[:3, :3])
    return GaussianPatch(
        means=meanP2.astype(np.float32),
        quats=np.repeat(np.asarray(q, np.float32).reshape(1, 4), n, 0),
        scales=sca, opacity=np.ones(n, np.float32),
        colors=np.clip(meanC, 0, 1).astype(np.float32),
        owner=0, conf=counts.astype(np.float32))


# ---------------------------------------------------------------------------
def cleanse_depth(d: np.ndarray, guide: np.ndarray, radius: int = 4,
                  eps: float = 1e-3, outlier_sigma: float = 3.0) -> np.ndarray:
    """Denoise a per-pixel depth map while keeping its edges (guided filter).

    1. local-median outlier clamp (removes the metre-level spikes of the point map),
    2. guided filter with the *image* as guide (smooth in flat areas, sharp at edges).
    """
    import cv2

    d32 = d.astype(np.float32)
    med = cv2.blur(d32, (5, 5))
    resid = d32 - med
    sd = float(np.std(resid)) + 1e-6
    d32 = np.clip(d32, med - outlier_sigma * sd, med + outlier_sigma * sd)

    I = guide.astype(np.float32)
    if I.ndim == 3:
        I = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.blur(I, k)
    mean_p = cv2.blur(d32, k)
    corr_I = cv2.blur(I * I, k) - mean_I * mean_I
    corr_Ip = cv2.blur(I * d32, k) - mean_I * mean_p
    a = corr_Ip / (corr_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, k)
    mean_b = cv2.blur(b, k)
    return (mean_a * I + mean_b).astype(np.float32)


def raycast_dsm(dsm, T_WP, t_cam, dirs, z_top: float = 60.0, z_bot: float = -8.0,
                n_steps: int = 48, n_bisect: int = 8):
    """Nearest-hit ray x height-field intersection (P0-2).

    The old implementation was a fixed-point iteration *starting at the z=0 ground
    hit*: for a roof pixel the ground hit lands **outside** the building, where the DSM
    is the ground, so it converged to the ground and never reached the roof (measured:
    only 0.0-1.8% of roof pixels landed within +-3 m).  This marches **downward from
    above the tallest surface** and keeps the *first* crossing, which is the visible
    surface, then bisects it.

    ``dirs`` are the (H, W, 3) ray directions in the **world** frame (not normalised,
    z component 1).  Returns ``(depth_along_ray, hit_mask)`` with shape (H, W).
    """
    n = T_WP[:3, 2]
    den = dirs @ n
    den = np.where(np.abs(den) > 1e-9, den, np.nan)
    # z of the ray at parameter t: cam_z + t * (dirs.n)  ->  t(z) = (z - cam_z)/(dirs.n)
    cam_z = float(t_cam @ n)
    zs = np.linspace(z_top, z_bot, n_steps, dtype=np.float32)
    first = np.full(dirs.shape[:-1], -1, np.int32)
    z_prev = np.full(dirs.shape[:-1], np.nan, np.float32)
    z_hit = np.full(dirs.shape[:-1], np.nan, np.float32)
    for zi, zv in enumerate(zs):
        tz = (zv - cam_z) / den
        P = t_cam[None, None, :] + tz[..., None] * dirs
        Pp = (P - T_WP[:3, 3]) @ T_WP[:3, :3]
        z_s = dsm.lookup(Pp[..., 0], Pp[..., 1], fill=np.nan)
        below = np.isfinite(z_s) & (zv <= z_s)
        new = below & (first < 0)
        if new.any():
            first[new] = zi
            z_hit[new] = zv
            z_prev[new] = zs[zi - 1] if zi > 0 else zv
    # bisect between the last "above" and the first "below" z
    lo = z_prev.astype(np.float32)     # above the surface
    hi = z_hit.astype(np.float32)      # below the surface
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        tz = (mid - cam_z) / den
        P = t_cam[None, None, :] + tz[..., None] * dirs
        Pp = (P - T_WP[:3, 3]) @ T_WP[:3, :3]
        z_s = dsm.lookup(Pp[..., 0], Pp[..., 1], fill=np.nan)
        goes_below = np.isfinite(z_s) & (mid <= z_s)
        hi = np.where(goes_below, mid, hi)
        lo = np.where(goes_below, lo, mid)
    z_fin = 0.5 * (lo + hi)
    t_fin = (z_fin - cam_z) / den
    # return the **z-depth** (ray parameter with z-component 1), matching how every
    # other depth source in this module is used (``X_canon[...,2]``, head ``mu_cam``).
    # An earlier version returned the *length* along the ray, which the caller then
    # multiplied by an unnormalised ray -> up to ~4 m of error at the image edges.
    depth = np.where(first >= 0, t_fin, np.nan).astype(np.float32)
    return depth, (first >= 0)


def upsample_depth(d_h, H: int, W: int, mode: str = "linear", edge_tau_m: float = 0.75):
    """Resample a low-resolution depth map to the image grid.

    ``linear``   bilinear: invents depths between a roof and the ground at every edge
    ``nearest``  copies existing values only (no ramp) but duplicates one side across the edge
    ``edge``     bilinear on flat neighbourhoods, nearest where the 2x2 range exceeds tau
    """
    import cv2 as _cv2
    if mode == "nearest":
        return _cv2.resize(d_h, (W, H), interpolation=_cv2.INTER_NEAREST)
    if mode == "edge":
        bil = _cv2.resize(d_h, (W, H), interpolation=_cv2.INTER_LINEAR)
        near = _cv2.resize(d_h, (W, H), interpolation=_cv2.INTER_NEAREST)
        pad = np.pad(d_h, 1, mode="edge")
        h, w = d_h.shape[:2]
        local_max = np.maximum.reduce([pad[1 + di:1 + di + h, 1 + dj:1 + dj + w]
                                       for di in (-1, 0) for dj in (-1, 0)])
        local_min = np.minimum.reduce([pad[1 + di:1 + di + h, 1 + dj:1 + dj + w]
                                       for di in (-1, 0) for dj in (-1, 0)])
        rng = (local_max - local_min).astype(np.float32)
        # dilate by one model pixel: the target pixels *between* the last flat column and the
        # first step column take their nearest sample from the flat side, yet bilinear
        # interpolates across the step there -- they must be masked too
        from scipy.ndimage import maximum_filter
        rng = maximum_filter(rng, size=3, mode="nearest")
        edge = _cv2.resize(rng, (W, H), interpolation=_cv2.INTER_NEAREST) > float(edge_tau_m)
        return np.where(edge, near, bil)
    return _cv2.resize(d_h, (W, H), interpolation=_cv2.INTER_LINEAR)


def head_adapt_patch(ds, npz, k, sim3_vec_gt, model, head, stride: int = 2,
                     dsm=None, depth_source: str = "pointmap",
                     sigma_px: float = 1.0, opacity_min: float = 0.0,
                     sigma_m: float = 0.0,
                     flatten: float = 0.1, conf_min: float = 0.0,
                     smooth_radius: int = 0, clip_m: float = 3.0,
                     z_align: bool = False, cull_facade: float = 0.0,
                     edge_reject: int = 0, edge_pct: float = 99.0,
                     edge_opacity: float = 0.25,
                     device: str = "cuda:0", conf_mask=None,
                     depth_upsample: str = "nearest",
                     edge_tau_m: float = 0.75) -> GaussianPatch:
    """Head output + **ortho adapter** + **per-pixel depth** at full image resolution.

    Fixes three bugs of the first version (week8-end-14):
      1. colour came from the model's 512-px input image -> use the full-resolution image;
      2. splats were placed on the 288x512 grid (0.78 m) with sigma = stride*0.78 m
         (7.5 ortho px) -> use the **full-resolution pixel grid** with sigma ~ 1 pixel;
      3. the depth was over-smoothed by the guided filter and mis-scaled by the affine
         calibration -> only outlier-clamp (optionally a tiny guided pass) and no
         global re-scaling.


        depth_upsample: str = "nearest" by default to prevent ghosting artifacts at building
            edges. Bilinear interpolation ("linear") creates fake intermediate depth values
            between roof and ground (e.g., 25m, 35m between 50m roof and 0m ground), which
            accumulate as multiple offset building copies in multi-view fusion. Nearest-neighbor
            preserves original depth values only (roof OR ground). Advanced users can use
            "edge" mode for edge-aware interpolation that keeps sharp edges while smoothing
            flat regions.

    depth_source:
      ``"pointmap"`` : MASt3R-SLAM point-map depth (has the real 20 m relief),
      ``"head"``     : the head's own depth (flat shell, kept for the comparison).
    Adapter: rotations <- ground normal, scales <- (s_xy, s_xy, flatten*s_xy),
    colours <- full-resolution image, opacity clamped, conf gating.
    """
    import cv2

    from stream_ortho.frontend import predict as P

    cull_full = None      # must exist even when no branch assigns it
    edge_att = None
    from stream_ortho.geometry import R_to_quat_wxyz, compose_sim3, sim3_from_vec8

    fid = int(npz["frame_id"][k])
    img_u8 = ds.read_img(fid)
    H, W = img_u8.shape[:2]
    enc = P.encode_image(model, img_u8, device=device)
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    h, w = int(enc["shape_hw"][0]), int(enc["shape_hw"][1])
    R, t, sv = sim3_from_vec8(compose_sim3(npz["T_WC"][k], sim3_vec_gt))
    K = ds.K_orig

    # ---- depth at the head grid (metric), outlier-clamped ----
    cull_full = None
    if depth_source == "dsm":
        # P0-1: the old grid was ``meshgrid(arange(h), arange(w), indexing="xy")`` which
        # returns shape (w, h) with u/v swapped (u reached 1078 for a 1920-px image), so
        # the height field was sampled from the wrong columns.  Build the ray grid with
        # explicit pixel-centre coordinates of shape (h, w).
        uu_h, vv_h = np.meshgrid((np.arange(w, dtype=np.float64) + 0.5) * (W / float(w)),
                                 (np.arange(h, dtype=np.float64) + 0.5) * (H / float(h)),
                                 indexing="xy")
        assert uu_h.shape == (h, w), f"ray grid {uu_h.shape} != {(h, w)}"
        assert 0.0 <= uu_h.min() and uu_h.max() < W, (uu_h.min(), uu_h.max())
        assert 0.0 <= vv_h.min() and vv_h.max() < H, (vv_h.min(), vv_h.max())
        rays_d = np.stack([(uu_h - K[0, 2]) / K[0, 0],
                           (vv_h - K[1, 2]) / K[1, 1],
                           np.ones_like(uu_h)], -1)
        dirs_d = rays_d @ R.T
        # P0-2: nearest hit, marching down from above the tallest surface
        d_sur, hit = raycast_dsm(dsm, ds.T_WP, t, dirs_d)
        # P0-4: facade culling -- a wall pixel's own depth is *smaller* than the surface
        # behind it, so it must not be mapped onto the roof edge
        d_self = g["mu_cam"][..., 2].detach().cpu().numpy().astype(np.float32)
        xz = np.asarray(npz["X_canon"][k][..., 2], np.float32)
        d_self = d_self * (float(np.median(xz)) / max(float(np.median(d_self)), 1e-6)) * float(sv)
        if cull_facade > 0.0:
            cull_h = (d_self < (d_sur - float(cull_facade))) | (~hit)
        else:
            cull_h = ~hit
        cull_full = np.repeat(np.repeat(cull_h, max(1, H // h), 0),
                              max(1, W // w), 1)
        if cull_full.shape != (H, W):
            cull_full = cv2.resize(cull_h.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        d_h = d_sur
    elif depth_source == "head":
        d_h = g["mu_cam"][..., 2].detach().cpu().numpy().astype(np.float32)
        xz = np.asarray(npz["X_canon"][k][..., 2], np.float32)
        k_align = float(np.median(xz) / max(float(np.median(d_h)), 1e-6))
        d_h = d_h * k_align * float(sv)
    else:
        d_h = np.asarray(npz["X_canon"][k][..., 2], np.float32) * float(sv)
    med = np.median(d_h)
    d_h = np.clip(d_h, med * 0.5, med * 1.5)
    if smooth_radius > 0:
        d_h = cleanse_depth(d_h, enc["uimg"], radius=smooth_radius, eps=1e-4)

    # ---- upsample to full resolution and place one splat per (strided) pixel ----
    # A (week8-end-38): the raw 288x512 point map already carries a 3-9 pixel wide
    # depth-discontinuity band (measured) -- that band is what appears as "lines of
    # points hanging off the building edge".  Drop those pixels instead of trusting
    # them; the same roof edge is an *interior* pixel in other keyframes, so the
    # geometry is recovered from the views that see it properly.
    if edge_reject > 0:
        inv = (1.0 / np.maximum(d_h.astype(np.float64), 1e-6)).astype(np.float32)
        mag = np.hypot(cv2.Sobel(inv, cv2.CV_32F, 1, 0, 3),
                       cv2.Sobel(inv, cv2.CV_32F, 0, 1, 3))
        band = (mag >= np.percentile(mag, edge_pct)).astype(np.uint8)
        band = cv2.dilate(band, np.ones((2 * edge_reject + 1,) * 2, np.uint8))
        band_full = cv2.resize(band, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        # A' (week8-end-39): deleting the band in *every* frame left a permanent black
        # "moat" along every building outline (measured: 91% of the lost pixels were in
        # the edge band, and no view could refill them).  Attenuate instead: the points
        # still contribute when nothing else covers that spot, but they no longer form a
        # hard curtain.  ``edge_opacity=0`` restores the old hard delete.
        if edge_opacity <= 0.0:
            cull_full = band_full if cull_full is None else (cull_full | band_full)
        else:
            edge_att = band_full

    # bilinear upsampling invents depths *between* a roof and the ground at every edge;
    # nearest keeps the density without new values but copies one side across the edge.
    # "edge" interpolates only where the 2x2 model neighbourhood is flat.
    d_full = upsample_depth(np.asarray(d_h), H, W, depth_upsample, edge_tau_m)

    uu, vv = np.meshgrid(np.arange(0, W, stride, dtype=np.float64),
                         np.arange(0, H, stride, dtype=np.float64))
    d_s = d_full[::stride, ::stride]

    if depth_source == "fused":
        # geometry fusion: DSM surface (smooth) + the head's *local* deviation (detail),
        # with the deviation's ground median removed and its magnitude clipped, so the
        # head's noise cannot dominate but its height differences are kept
        if dsm is None:
            raise ValueError("depth_source='fused' needs a DSM")
        n = ds.T_WP[:3, 2]
        p0 = ds.T_WP[:3, 3]
        rays_f = np.stack([(uu - K[0, 2]) / K[0, 0],
                           (vv - K[1, 2]) / K[1, 1],
                           np.ones_like(uu)], -1)
        dirs_f = rays_f @ R.T
        den_f = np.where(np.abs(dirs_f @ n) > 1e-9, dirs_f @ n, np.nan)
        tt = np.broadcast_to(((p0 - t) @ n) / den_f, dirs_f.shape[:-1]).copy()
        for _ in range(3):
            P = t[None, None, :] + tt[..., None] * dirs_f
            Pp = (P - p0) @ ds.T_WP[:3, :3]
            zz = dsm.lookup(Pp[..., 0], Pp[..., 1], fill=0.0)
            tt_new = ((p0 + n[None, None, :] * zz[..., None] - t) @ n) / den_f
            tt = np.where(np.isfinite(tt_new) & (tt_new > 0), tt_new, tt)
        d_dsm = tt.reshape(-1)
        delta = d_s.reshape(-1) - d_dsm
        good = np.isfinite(delta)
        delta = delta - (float(np.median(delta[good])) if good.any() else 0.0)
        d_s = (d_dsm + np.clip(delta, -clip_m, clip_m)).reshape(d_s.shape)
    rays = np.stack([(uu - K[0, 2]) / K[0, 0],
                     (vv - K[1, 2]) / K[1, 1],
                     np.ones_like(uu)], -1)
    Pc = rays * d_s[..., None]
    Pw = Pc.reshape(-1, 3) @ R.T + t

    if z_align and dsm is not None:
        # EXPERIMENTAL (week8-end-20): does **not** work yet -- it zeroes the ground
        # median but the all-point median moves the wrong way (measured: kf4 +5.5 ->
        # +15.6) and the voxel overlap stays 0%.  Kept for the next debugging round,
        # default False.
        # per-frame **vertical offset** correction.  Each frame's surface sits at a
        # different plane-z (measured: median -4.5 m ... +5.5 m over 5 keyframes), so
        # the voxel dedup never matches and the map piles frames up instead of fusing
        # them.  Shifting the frame so that its *ground* sits on the DSM's ground is a
        # 1-DoF fix that removes the dominant term.
        Pp = (Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        zg = dsm.lookup(Pp[:, 0], Pp[:, 1], fill=0.0)
        ground = zg < 2.0
        if int(ground.sum()) > 5000:
            dz = -float(np.median(Pp[ground, 2]))
            dz = float(np.clip(dz, -15.0, 15.0))
            Pp[:, 2] += dz
            Pw = Pp @ ds.T_WP[:3, :3] + ds.T_WP[:3, 3]
            LAST_AFFINE.update(a=1.0, b=dz)


    # ---- adapter ----
    if sigma_m > 0:
        # sigma tied to the **output** GSD (P5 fix #1): a splat must cover about one
        # output pixel, otherwise the render is a 2.7 px blur brush at 0.30 m/px
        s_xy = np.full_like(d_s, sigma_m)
    else:
        s_xy = sigma_px * stride * d_s / float(K[0, 0])   # ~1 pixel footprint
    scales = np.stack([s_xy, s_xy, flatten * s_xy], -1).reshape(-1, 3)
    q = np.asarray(R_to_quat_wxyz(ds.T_WP[:3, :3]), np.float32).reshape(1, 4)
    quats = np.repeat(q, Pw.shape[0], 0)
    # head opacity / conf upsampled to the same grid (nearest is fine)
    op_h = g["opacity"].detach().cpu().numpy().astype(np.float32).reshape(h, w)
    cf_h = g["conf"].detach().cpu().numpy().astype(np.float32).reshape(h, w)
    op = np.clip(cv2.resize(op_h, (W, H), interpolation=cv2.INTER_NEAREST)[::stride, ::stride].reshape(-1),
                 opacity_min, 1.0)
    if edge_reject > 0 and edge_opacity > 0.0:
        att = edge_att[::stride, ::stride].reshape(-1)
        op = np.where(att, op * float(edge_opacity), op).astype(np.float32)
    conf = cv2.resize(cf_h, (W, H), interpolation=cv2.INTER_NEAREST)[::stride, ::stride].reshape(-1)
    colors = img_u8[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
    sel = np.arange(Pw.shape[0])
    if cull_full is not None:
        cf = cull_full[::stride, ::stride].reshape(-1)
        sel = sel[~cf[sel]]
    if conf_mask is not None:
        # upstream gates on the *pointmap* confidence C/N, which is a different quantity from
        # the head's per-Gaussian confidence; the mask carries it at model resolution
        mmk = np.asarray(conf_mask)
        mmk = cv2.resize(mmk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mmk = mmk[::stride, ::stride].reshape(-1).astype(bool)
        sel = sel[mmk[sel]]
    if conf_min > 0:
        sel = sel[conf[sel] >= conf_min]
    return GaussianPatch(
        means=Pw[sel].astype(np.float32), quats=quats[sel].astype(np.float32),
        scales=scales[sel].astype(np.float32), opacity=op[sel], colors=colors[sel],
        owner=int(fid), conf=conf[sel],
        meta={"depth_m": float(np.median(d_s)), "frame_id": fid,
              "source": f"head+adapter/{depth_source}"})

