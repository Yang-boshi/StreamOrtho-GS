"""Figures for the E6 directional diagnosis.

    fig_e6d_overlap.png      per-frame overlap vs PC improvement, with quantile bins
    fig_e6d_distance.png     PC improvement by distance-to-build bins
    fig_e6d_trajectory.png   camera track (frame centroids), held frames highlighted
    fig_e6d_ra_subset.png    RA on the whole product vs on the corrected subset
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

COLOURS = {"even_build_odd_held": "#4c72b0", "odd_build_even_held": "#c44e52"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--obs", default="result-new/P0/obs/ieu/pointmap")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pathlib.Path(a.dir)
    figs = d / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    r = json.loads((d / "e6_directional.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    D = r["directions"]

    # ---- 1. overlap ----
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for name, v in D.items():
        f = v["frames"]
        ov = np.array([x["overlap_frac"] for x in f])
        rel = np.array([x["pc_rel_pct"] for x in f])
        ax.scatter(ov, rel, s=14, alpha=0.75, color=COLOURS[name], label=name)
        bins = v["bins"]["overlap"]
        cx = [(b["lo"] + b["hi"]) / 2 for b in bins if b["n"]]
        cy = [b["pc_rel_pct_median"] for b in bins if b["n"]]
        ax.plot(cx, cy, "-o", color=COLOURS[name], lw=1.5, ms=4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhline(20, color="k", ls=":", lw=1, label="20% target")
    ax.set_xlabel("overlap_frac (frame cells ∩ build cells)")
    ax.set_ylabel("per-frame PC improvement (%)")
    ax.set_title("PC improvement vs overlap: the weaker direction does NOT overlap less")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_e6d_overlap.png"); plt.close(fig)

    # ---- 2. distance ----
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w = 0.38
    for i, (name, v) in enumerate(D.items()):
        bins = [b for b in v["bins"]["distance"] if b["n"]]
        x = np.arange(len(bins)) + (i - 0.5) * w
        ax.bar(x, [b["pc_rel_pct_median"] for b in bins], w, color=COLOURS[name], label=name)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["0–50 m", "50–100 m", "100–200 m", ">200 m"], fontsize=7.5)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("PC improvement (%)"); ax.set_title("PC improvement by distance to the nearest build frame")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_e6d_distance.png"); plt.close(fig)

    # ---- 3. trajectory ----
    obs = np.load(pathlib.Path(a.obs) / "obs_merged.npz")
    kf = obs["kf"]
    cx = np.zeros(56); cy = np.zeros(56); has = np.zeros(56, bool)
    for k in np.unique(kf):
        m = kf == k
        cx[k], cy[k] = obs["x"][m].mean(), obs["y"][m].mean()
        has[k] = True
    ks = np.nonzero(has)[0]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(cx[ks], cy[ks], "-", color="0.7", lw=1, label="track")
    for name, v in D.items():
        held = {f["kf"] for f in v["frames"]}
        sel = [k for k in ks if k in held]
        ax.scatter(cx[sel], cy[sel], s=22, color=COLOURS[name], label=f"held: {name.split('_')[0]}")
    ax.set_xlabel("plane x (m)"); ax.set_ylabel("plane y (m)")
    ax.set_title("camera track and the two held-out sets")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_e6d_trajectory.png"); plt.close(fig)

    # ---- 4. RA subset ----
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels, raw_full, fro_full, raw_sub, fro_sub = [], [], [], [], []
    for name, v in D.items():
        ra = v["ra"]
        labels.append(name.split("_")[0])
        raw_full.append(ra["full_raw"]); fro_full.append(ra["full_frozen"])
        raw_sub.append(ra["subset_raw"]); fro_sub.append(ra["subset_frozen"])
    x = np.arange(len(labels)); w = 0.2
    ax.bar(x - 1.5 * w, raw_full, w, color="#8c8c8c", label="full / raw")
    ax.bar(x - 0.5 * w, fro_full, w, color="#55a868", label="full / frozen")
    ax.bar(x + 0.5 * w, raw_sub, w, color="#b0b0b0", label="corrected subset / raw")
    ax.bar(x + 1.5 * w, fro_sub, w, color="#2e7d32", label="corrected subset / frozen")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RA (m)"); ax.set_title("RA does not degrade on the corrected subsets either")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_e6d_ra_subset.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
