"""P3.4: is the edge "splashing" caused by the depth upsampling in the adapter?

``head_adapt_patch`` takes the 288x512 pointmap depth and bilinearly resizes it to the full
image, then back-projects ``rays * d``.  At a roof/ground discontinuity that interpolation
creates depths *between* the roof and the ground, i.e. exactly the slanted "splashes" that
appear after texturing.  Upstream simply maps ``X_canon`` through ``T_WC`` and has no such
step.

This script builds both clouds from the same keyframe dump -- raw (model grid) and adapter
(the same geometry the pipeline uses) -- and compares their edge statistics and pictures.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho.geometry import compose_sim3, sim3_from_vec8


def _load(run):
    d = np.load(pathlib.Path(run) / "keyframes.npz")
    sim3 = pathlib.Path(run) / "sim3_est_to_gt.json"
    sg = (json.loads(sim3.read_text()) if sim3.exists() else None)
    if sg is None:
        sg = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}
    return d, sg


def _sim3_vec(sg):
    R = np.asarray(sg["rotation"], np.float64)
    from stream_ortho.geometry import R_to_quat_xyzw
    return np.r_[np.asarray(sg["translation"], np.float64), R_to_quat_xyzw(R), float(sg["scale"])]


def raw_cloud(d, sg, k, thr=0.2):
    X = np.asarray(d["X_canon"][k], np.float64).reshape(-1, 3)
    C = np.asarray(d["C"][k], np.float64).reshape(-1)
    n = float(d["N"][k]) if "N" in d.files else 1.0
    conf = C / max(n, 1.0)
    R, t, s = sim3_from_vec8(compose_sim3(d["T_WC"][k], _sim3_vec(sg)))
    Pw = float(s) * (X @ R.T) + t
    keep = np.isfinite(conf) & np.isfinite(Pw).all(1) & (conf >= thr)
    return Pw[keep], conf[keep]


def adapter_cloud(d, sg, k, thr=0.2, image_hw=(1080, 1920)):
    """Replicate head_adapt_patch's geometry: bilinear depth upsample + full-res rays."""
    import cv2
    X = np.asarray(d["X_canon"][k], np.float64)
    h, w = X.shape[:2]
    K = np.asarray(d["K"], np.float64).reshape(3, 3)
    # hw in the dump is the *model* grid; the adapter resamples to the full image
    H, W = int(image_hw[0]), int(image_hw[1])
    d_h = X[..., 2]
    d_full = cv2.resize(d_h, (W, H), interpolation=cv2.INTER_LINEAR)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                     np.ones_like(uu)], -1)
    R, t, s = sim3_from_vec8(compose_sim3(d["T_WC"][k], _sim3_vec(sg)))
    Pw = float(s) * ((rays * d_full[..., None]).reshape(-1, 3) @ R.T) + t
    C = np.asarray(d["C"][k], np.float64).reshape(h, w)
    n = float(d["N"][k]) if "N" in d.files else 1.0
    conf_lo = (C / max(n, 1.0)).reshape(-1)
    # nearest-sample the model-grid confidence on the upsampled grid
    ci = np.clip((uu * (w - 1) / max(W - 1, 1)).round().astype(int), 0, w - 1)
    cj = np.clip((vv * (h - 1) / max(H - 1, 1)).round().astype(int), 0, h - 1)
    conf = (C / max(n, 1.0))[cj, ci].reshape(-1)
    keep = np.isfinite(conf) & np.isfinite(Pw).all(1) & (conf >= thr)
    return Pw[keep], conf[keep]


def stats(P, conf, cell=0.45, gap=2.5):
    out = {"n": int(P.shape[0])}
    if not P.shape[0]:
        return {**out, "floater": 0, "n_edge": 0, "z_iqr_edge": 0.0, "coverage": 0.0}
    ix = np.floor(P[:, 0] / cell).astype(np.int64); iy = np.floor(P[:, 1] / cell).astype(np.int64)
    ix -= ix.min(); iy -= iy.min()
    z = np.full((iy.max() + 1, ix.max() + 1), -np.inf)
    np.maximum.at(z, (iy, ix), P[:, 2])
    z[~np.isfinite(z)] = np.nan
    live = np.isfinite(z)
    zf = np.where(live, z, np.nanmedian(z[live]))
    g = np.hypot(np.gradient(zf, axis=0), np.gradient(zf, axis=1))
    edge = live & (g > np.percentile(g[live], 85))
    from scipy.ndimage import minimum_filter
    low = minimum_filter(np.where(live, z, np.nanmax(z[live])), size=5)
    floaters = live & ((z - low) > gap)
    out.update({"floater": int(floaters.sum()),
                "floater_frac": float(floaters.sum() / max(live.sum(), 1)),
                "n_edge": int(edge.sum()),
                "z_iqr_edge": float(np.subtract(*np.percentile(z[edge], [75, 25]))) if edge.any() else 0.0,
                "coverage": float(live.mean()), "conf_p50": float(np.median(conf))})
    return out


def topview(P, cell=0.45, size=520):
    from PIL import Image
    if not P.shape[0]:
        return Image.new("RGB", (size, size), (255, 255, 255))
    ix = np.floor(P[:, 0] / cell).astype(np.int64); iy = np.floor(P[:, 1] / cell).astype(np.int64)
    ix -= ix.min(); iy -= iy.min()
    z = np.full((iy.max() + 1, ix.max() + 1), -np.inf)
    np.maximum.at(z, (iy, ix), P[:, 2])
    z[~np.isfinite(z)] = np.nan
    lo, hi = np.nanpercentile(z, [2, 98])
    g = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (np.where(np.isfinite(z), g, 1.0) * 255).astype(np.uint8)
    im = Image.fromarray(img).resize((size, size), Image.NEAREST)
    return im.convert("RGB")


def crops(P, cell, boxes, size=320):
    from PIL import Image
    im = topview(P, cell, size=1024)
    out = []
    for (x0, y0, x1, y1) in boxes:
        c = im.crop((int(x0 * 1024), int(y0 * 1024), int(x1 * 1024), int(y1 * 1024)))
        out.append(c.resize((size, size), Image.LANCZOS))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/stage-S2/ieu_calib")
    ap.add_argument("--out", default="results/system/p34_compare")
    ap.add_argument("--conf-min", type=float, default=0.2)
    ap.add_argument("--frames", default="0,5,10,20", help="keyframe indices to sample")
    ap.add_argument("--upstream-ply", default=None)
    ap.add_argument("--image-hw", default="1080,1920")
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d, sg = _load(a.run)
    kfs = [int(x) for x in a.frames.split(",") if int(x) < d["X_canon"].shape[0]]
    image_hw = tuple(int(v) for v in a.image_hw.split(","))
    res = {"run": a.run, "frames": kfs, "clouds": {}}
    clouds = {}
    for name, fn in (("raw", raw_cloud), ("adapter", adapter_cloud)):
        Ps, Cs = [], []
        for k in kfs:
            P, c = (fn(d, sg, k, a.conf_min, image_hw=image_hw) if fn is adapter_cloud
                    else fn(d, sg, k, a.conf_min))
            Ps.append(P); Cs.append(c)
        P = np.vstack(Ps) if Ps else np.zeros((0, 3)); C = np.concatenate(Cs) if Cs else np.zeros(0)
        clouds[name] = (P, C)
        res["clouds"][name] = stats(P, C)
    if a.upstream_ply and pathlib.Path(a.upstream_ply).exists():
        res["clouds"]["upstream"] = {"ply": a.upstream_ply}

    base = res["clouds"]["adapter"]["floater"]
    for k, v in res["clouds"].items():
        if "floater" in v:
            v["floater_vs_adapter"] = 1.0 - v["floater"] / base if base else 0.0
    (out / "compare.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # images: top views + 3 building crops
    from PIL import Image
    tv = [("raw (X_canon)", clouds["raw"][0]), ("adapter (depth upsampled)", clouds["adapter"][0])]
    imgs = [topview(P) for _, P in tv]
    W = sum(i.size[0] for i in imgs)
    canvas = Image.new("RGB", (W, imgs[0].size[1]), (255, 255, 255))
    x = 0
    for i in imgs:
        canvas.paste(i, (x, 0)); x += i.size[0]
    canvas.save(out / "topview_raw_vs_adapter.png")

    boxes = [(0.25, 0.05, 0.50, 0.30), (0.10, 0.35, 0.35, 0.60), (0.50, 0.45, 0.75, 0.70)]
    rows = []
    for name, P in (("raw", clouds["raw"][0]), ("adapter", clouds["adapter"][0])):
        rows.append(crops(P, 0.45, boxes))
    cw = rows[0][0].size[0]
    canv = Image.new("RGB", (cw * 3, cw * 2), (255, 255, 255))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            canv.paste(im, (c * cw, r * cw))
    canv.save(out / "crops_raw_top_adapter_bottom.png")

    # ---- the money shot: a height profile across a roof/ground step -------------------
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = kfs[0]
    X = np.asarray(d["X_canon"][k], np.float64)
    h, w = X.shape[:2]
    d_h = X[..., 2]
    d_up = cv2.resize(d_h, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
    gx = np.abs(np.diff(d_h, axis=1))
    row = int(np.argmax(gx.max(1)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(np.arange(w), d_h[row], "o-", ms=2.5, lw=1, label="raw (model grid)")
    ax[0].plot(np.linspace(0, w - 1, image_hw[1]), d_up[int(row * image_hw[0] / h)],
               "-", lw=1, label="adapter (bilinear upsample)")
    ax[0].set_title(f"depth along row {row}: the step becomes a ramp", fontsize=9)
    ax[0].set_xlabel("column (model grid)"); ax[0].set_ylabel("depth")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].plot(np.arange(w), d_h[row], "o-", ms=2.5, lw=1, label="raw (model grid)")
    ax[1].set_xlim(max(0, w // 2 - 3), min(w - 1, w // 2 + 3)) if False else None
    ax[1].set_title("same, zoom on the step", fontsize=9)
    j = int(np.argmax(gx[row]))
    ax[1].plot(np.arange(max(0, j - 3), min(w, j + 4)),
               d_h[row][max(0, j - 3):min(w, j + 4)], "o-", ms=5, lw=1.2)
    ax[1].set_xlim(max(0, j - 3), min(w - 1, j + 3)); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "edge_profile.png"); plt.close(fig)
    print(json.dumps(res["clouds"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
