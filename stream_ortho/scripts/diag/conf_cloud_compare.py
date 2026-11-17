"""P3.3: build per-threshold clouds from the keyframe dump and compare them.

Upstream's ``save_reconstruction`` keeps a pixel when ``get_average_conf() > threshold``, i.e.
``C / N`` where ``N`` is the number of pointmap updates.  Our observation table used the raw
accumulator ``C`` (and a 0.2 floor), which keeps edge pixels whose average is far below 1.5 --
the "splashing" seen after texturing.

This script writes one PLY per threshold (plus the legacy ``max/0.2`` cloud) and the numbers the
card asks for: point counts, confidence percentiles, edge-band statistics, floater count.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho.geometry import sim3_matrix


def load_keyframes(path):
    d = np.load(path)
    X = d["X_canon"].astype(np.float64)
    C = d["C"].astype(np.float64).reshape(X.shape[0], -1)
    if "N" in d.files:
        N = d["N"].astype(np.float64).reshape(-1, 1)
        has_n = True
    else:
        N = np.ones((X.shape[0], 1), np.float64)
        has_n = False
    # sim3_matrix folds the lietorch scale into the rotation block; dropping it would leave
    # the cloud in a 2.5 m "SLAM world" and make a 0.45 m grid meaningless
    T = (np.stack([sim3_matrix(t) for t in d["T_WC"]]) if d["T_WC"].shape[1] == 8
         else d["T_WC"])
    return X, C, N, T, has_n


def cloud(X, C, N, T, thr, stat="avg"):
    conf = (C / N) if stat == "avg" else C
    pts, cols = [], []
    for k in range(X.shape[0]):
        x = X[k].reshape(-1, 3)
        c = conf[k].reshape(-1)
        keep = np.isfinite(c) & np.isfinite(x).all(1) & (c >= thr)
        if not keep.any():
            continue
        ph = np.c_[x[keep], np.ones(int(keep.sum()))]
        pts.append((ph @ T[k].T)[:, :3])
        cols.append(np.clip(c[keep] / max(thr, 1e-6), 0, 4)[:, None])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 1))
    return np.vstack(pts), np.vstack(cols)


def grid_stats(P, cell=0.45, gap_m=2.5):
    if P.shape[0] == 0:
        return {"n": 0, "n_edge": 0, "z_iqr_edge": 0.0, "floaters": 0, "coverage": 0.0}
    ix = np.floor(P[:, 0] / cell).astype(np.int64)
    iy = np.floor(P[:, 1] / cell).astype(np.int64)
    ix -= ix.min(); iy -= iy.min()
    H, W = iy.max() + 1, ix.max() + 1
    # np.maximum.at ignores NaN ordering, so start from -inf and restore NaN afterwards
    z = np.full((H, W), -np.inf)
    np.maximum.at(z, (iy, ix), P[:, 2])          # top surface per cell
    z[~np.isfinite(z)] = np.nan
    live = np.isfinite(z)
    zf = np.where(live, z, np.nanmedian(z[live]))
    gx = np.gradient(zf, axis=1); gy = np.gradient(zf, axis=0)
    grad = np.hypot(gx, gy)
    edge = live & (grad > np.percentile(np.abs(grad[live]), 85))
    # local low surface (p10 in a 5x5 window) for the floater test
    from scipy.ndimage import minimum_filter
    low = minimum_filter(np.where(live, z, np.nanmax(z[live])), size=5)
    floaters = live & ((z - low) > gap_m)
    return {"n": int(P.shape[0]), "n_edge": int(edge.sum()),
            "z_iqr_edge": float(np.subtract(*np.percentile(z[edge], [75, 25]))) if edge.any() else 0.0,
            "floaters": int(floaters.sum()), "coverage": float(live.mean())}


def write_ply(path, P, conf):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {P.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float conf\nend_header\n")
        for (x, y, z), c in zip(P, conf.reshape(-1)):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {c:.3f}\n")


def preview(path, clouds, names, cell=0.45):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(clouds), figsize=(5.2 * len(clouds), 5.0))
    axes = np.atleast_1d(axes)
    for ax, (P, c), nm in zip(axes, clouds, names):
        if P.shape[0]:
            ix = np.floor(P[:, 0] / cell).astype(int); iy = np.floor(P[:, 1] / cell).astype(int)
            ix -= ix.min(); iy -= iy.min()
            H, W = iy.max() + 1, ix.max() + 1
            img = np.full((H, W), -np.inf)
            np.maximum.at(img, (iy, ix), P[:, 2])
            img[~np.isfinite(img)] = np.nan
            im = ax.imshow(img, cmap="viridis", interpolation="nearest")
            ax.set_title(f"{nm}\n{P.shape[0]:,} pts", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/stage-S2/ieu_conf")
    ap.add_argument("--out", default="results/system/p33_conf")
    ap.add_argument("--thresholds", default="0.2,0.5,1.0,1.5")
    ap.add_argument("--legacy-thr", type=float, default=0.2)
    ap.add_argument("--cell", type=float, default=0.45)
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, C, N, T, has_n = load_keyframes(pathlib.Path(a.run) / "keyframes.npz")
    res = {"run": a.run, "has_N": has_n, "n_keyframes": int(X.shape[0]),
           "N_min": int(N.min()), "N_max": int(N.max()), "clouds": {}}
    made, names = [], []
    for thr in [float(t) for t in a.thresholds.split(",")]:
        P, c = cloud(X, C, N, T, thr, "avg")
        write_ply(out / f"our_avg_{thr}.ply", P, c)
        st = grid_stats(P, a.cell)
        raw = (C / N).reshape(-1)
        kept = raw[np.isfinite(raw) & (raw >= thr)]
        st["conf_p50"] = float(np.median(kept)) if kept.size else 0.0
        st["conf_p90"] = float(np.percentile(kept, 90)) if kept.size else 0.0
        st["kept_frac"] = float(kept.size / max(raw.size, 1))
        res["clouds"][f"avg_{thr}"] = st
        made.append((P, c)); names.append(f"avg (C/N) >= {thr}")
    P0, c0 = cloud(X, C, N, T, a.legacy_thr, "max")
    write_ply(out / f"our_max_{a.legacy_thr}.ply", P0, c0)
    st0 = grid_stats(P0, a.cell)
    st0["conf_p50"] = float(np.median(c0)) if c0.size else 0.0
    st0["conf_p90"] = float(np.percentile(c0, 90)) if c0.size else 0.0
    res["clouds"][f"max_{a.legacy_thr}"] = st0
    made.insert(0, (P0, c0)); names.insert(0, f"legacy max >= {a.legacy_thr}")

    base = res["clouds"][f"max_{a.legacy_thr}"]["floaters"]
    for k, v in res["clouds"].items():
        v["floater_vs_legacy"] = (1.0 - v["floaters"] / base) if base else 0.0
    (out / "conf_compare.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    preview(out / "preview.png", made, names, a.cell)
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in res["clouds"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
