"""Side-by-side comparison: 0.45 m per-cell colour vs texture ortho on the same grid."""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--tex", required=True, help="dir with tex_crop_*.npz")
    ap.add_argument("--crops", default="0,2")
    ap.add_argument("--npz-pattern", default="tex_crop_{ci}.npz")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = dict(np.load(pathlib.Path(a.obs) / "obs_merged.npz"))
    p = np.load(a.product)
    cell = float(p["cell"]); ox, oy = int(p["ox"]), int(p["oy"])
    H, W = p["z_ref"].shape[:2]

    ix = np.clip(np.asarray(obs["ix"], np.int64) - ox, 0, W - 1)
    iy = np.clip(np.asarray(obs["iy"], np.int64) - oy, 0, H - 1)
    w = np.asarray(obs["conf"], np.float32) * np.asarray(obs["n"], np.float32)
    flat = iy * W + ix
    den = np.bincount(flat, weights=w, minlength=H * W)
    cell_rgb = np.zeros((H * W, 3), np.float32)
    for ch in range(3):
        num = np.bincount(flat, weights=w * np.asarray(obs["rgb"][:, ch], np.float32),
                          minlength=H * W)
        cell_rgb[:, ch] = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    cell_rgb = (cell_rgb / 255.0).reshape(H, W, 3)

    tex_dir = pathlib.Path(a.tex)
    for ci in [int(v) for v in a.crops.split(",") if v.strip()]:
        f = tex_dir / (a.npz_pattern.format(ci=ci))
        if not f.exists():
            continue
        z = np.load(f)
        rgb_t = z["rgb"].astype(np.float32)
        x0, y0, gsd = float(z["x0"]), float(z["y0"]), float(z["gsd"])
        h, w_ = rgb_t.shape[:2]
        xs = np.clip(((x0 + (np.arange(w_) + 0.5) * gsd) // cell).astype(np.int64) - ox, 0, W - 1)
        ys = np.clip(((y0 + (np.arange(h) + 0.5) * gsd) // cell).astype(np.int64) - oy, 0, H - 1)
        rgb_c = cell_rgb[np.ix_(ys, xs)]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
        axes[0].imshow(np.clip(rgb_c, 0, 1))
        axes[0].set_title(f"per-cell colour @0.45 m, up-sampled to {gsd:.2f} m", fontsize=9)
        axes[1].imshow(np.clip(rgb_t, 0, 1))
        axes[1].set_title(f"texture ortho @{gsd:.2f} m", fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.tight_layout(); fig.savefig(tex_dir / f"compare_cell_vs_tex_{ci}.png"); plt.close(fig)
        print("wrote", tex_dir / f"compare_cell_vs_tex_{ci}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
