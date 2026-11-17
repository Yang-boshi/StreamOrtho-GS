"""Run artefacts: an ortho-update video and a point cloud of the product surface."""
from __future__ import annotations

import pathlib

import numpy as np


def write_cloud_ply(path, product, max_points: int = 2_000_000):
    """Top surface of the product as an ASCII PLY (x, y, z plus the cell colour)."""
    path = pathlib.Path(path)
    z, n, _ = product.layers()
    occ = n[..., 0] > 0
    if not occ.any():
        path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\n"
                        "property float y\nproperty float z\nend_header\n")
        return {"points": 0, "path": str(path)}
    rgb = product.top_rgb()
    ii, jj = np.nonzero(occ)
    if ii.size > max_points:
        step = int(np.ceil(ii.size / max_points))
        ii, jj = ii[::step], jj[::step]
    cell = float(product.cell)
    x = (jj + product.ox) * cell
    y = (ii + product.oy) * cell
    zz = z[ii, jj, 0]
    c = rgb[ii, jj]
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {ii.size}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for k in range(ii.size):
            f.write(f"{x[k]:.4f} {y[k]:.4f} {zz[k]:.4f} "
                    f"{int(c[k, 0])} {int(c[k, 1])} {int(c[k, 2])}\n")
    return {"points": int(ii.size), "path": str(path)}


def write_points_ply(path, pts, max_points: int = 1_500_000):
    """Write raw points ``(N, 7) = x,y,z,r,g,b,kf`` as an ASCII PLY (with colour)."""
    path = pathlib.Path(path)
    pts = np.asarray(pts, np.float32).reshape(-1, 7) if len(pts) else np.zeros((0, 7), np.float32)
    if pts.shape[0] > max_points:
        step = int(np.ceil(pts.shape[0] / max_points))
        pts = pts[::step]
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float kf\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                    f"{int(max(0, min(255, p[3])))} {int(max(0, min(255, p[4]))) } "
                    f"{int(max(0, min(255, p[5])))} {p[6]:.0f}\n")
    return {"points": int(pts.shape[0]), "path": str(path)}


def write_update_video(path, frames, cache, tile: int, fps: int = 10, crop_box=None,
                       boxes=None):
    """Write the ortho-update clip on a fixed canvas and verify it by reading it back.

    ``boxes`` gives each still's content box in the shared (scaled) canvas coordinates;
    every frame is placed at its box offset inside the union box, so a growing map stays
    aligned instead of everything being pasted at the top-left corner.
    """
    path = pathlib.Path(path)
    paths = sorted(pathlib.Path(p) for p in (frames or [])
                   if isinstance(p, (str, pathlib.Path)) and pathlib.Path(p).exists())
    if not paths:
        return {"error": "no per-update mosaics"}
    try:
        import cv2
        from PIL import Image
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        if boxes is not None and len(boxes) == len(imgs) and all(b is not None for b in boxes):
            bx = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
            r0 = min(b[0] for b in bx); r1 = max(b[1] for b in bx)
            c0 = min(b[2] for b in bx); c1 = max(b[3] for b in bx)
            fixed = []
            for im, (br0, br1, bc0, bc1) in zip(imgs, bx):
                canvas = np.zeros((r1 - r0, c1 - c0, 3), np.uint8)
                hh, ww = min(im.shape[0], br1 - br0), min(im.shape[1], bc1 - bc0)
                canvas[br0 - r0:br0 - r0 + hh, bc0 - c0:bc0 - c0 + ww] = im[:hh, :ww]
                fixed.append(canvas)
        else:
            if crop_box is not None:
                r0, r1, c0, c1 = [int(v) for v in crop_box]
                r0, c0 = max(r0, 0), max(c0, 0)
                r1, c1 = min(r1, imgs[0].shape[0]), min(c1, imgs[0].shape[1])
                imgs = [im[r0:r1, c0:c1] for im in imgs]
            h = min(i.shape[0] for i in imgs)
            w = min(i.shape[1] for i in imgs)
            fixed = []
            for im in imgs:
                canvas = np.zeros((h, w, 3), np.uint8)
                hh, ww = min(h, im.shape[0]), min(w, im.shape[1])
                canvas[:hh, :ww] = im[:hh, :ww]
                fixed.append(canvas)
        h, w = fixed[0].shape[:2]
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for im in fixed:
            out.write(im[:, :, ::-1])
        out.release()
        written = len(fixed)
    except Exception as exc:
        return {"error": repr(exc)}

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diffs, prev = [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if prev is not None and prev.shape == fr.shape:
            diffs.append(float(np.abs(fr.astype(np.float32) - prev.astype(np.float32)).mean()))
        prev = fr
    cap.release()
    return {"path": str(path), "frames": int(n), "written": int(written),
            "size": [fw, fh], "fps": fps, "duration_s": float(n / max(fps, 1)),
            "frame_diff": float(np.mean(diffs)) if diffs else 0.0, "tool": "cv2"}
