"""Figures for the neighbour-consensus objective experiment.

    fig_nf_objectives.png   per-frame PC improvement, one line per objective (both directions)
    fig_nf_gate.png         gate panel: % frames improved / worst frame / LCR per objective
    fig_nf_folds.png        E6-B folds: PC improvement per objective
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

COLOURS = {"raw": "#8c8c8c", "prod": "#4c72b0", "nbr": "#c44e52",
           "hybrid0.1": "#55a868", "hybrid1": "#dd8452", "nbr_top": "#8172b3"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pathlib.Path(a.dir)
    figs = d / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    r = json.loads((d / "e6_neighbor_fit.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    dirs = r["directions"]
    cfgs = [c for c in ("raw", "prod", "nbr", "hybrid0.1", "hybrid1", "nbr_top") if c in dirs[list(dirs)[0]]]

    # ---- 1. metric bars per objective ----
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    x = np.arange(len(cfgs))
    w = 0.38
    for i, (name, v) in enumerate(dirs.items()):
        xx = x + (i - 0.5) * w
        axes[0].bar(xx, [v[c]["pc_rel_median"] for c in cfgs], w, label=name.split("_")[0])
        axes[1].bar(xx, [v[c]["pct_improved"] for c in cfgs], w, label=name.split("_")[0])
        axes[2].bar(xx, [v[c]["min_pc_rel"] for c in cfgs], w, label=name.split("_")[0])
    axes[0].axhline(20, color="k", ls=":", lw=1, label="20% gate")
    axes[0].set_title("median PC improvement (%)"); axes[0].legend(fontsize=7)
    axes[1].axhline(80, color="k", ls=":", lw=1, label="80% gate")
    axes[1].set_title("% frames improved"); axes[1].legend(fontsize=7)
    axes[2].axhline(-10, color="k", ls=":", lw=1, label="-10% gate")
    axes[2].set_title("worst frame (%)"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, fontsize=7)
    fig.suptitle("E6 with a neighbour-consensus objective (+ optional product prior)", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_nf_objectives.png"); plt.close(fig)

    # ---- 2. product metrics ----
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for i, (name, v) in enumerate(dirs.items()):
        xx = x + (i - 0.5) * w
        axes[0].bar(xx, [v[c]["lcr_fixed"] for c in cfgs], w, label=name.split("_")[0])
        axes[1].bar(xx, [v[c]["z_iqr_ratio"] for c in cfgs], w, label=name.split("_")[0])
    axes[0].axhline(dirs[list(dirs)[0]]["raw"]["lcr_fixed"] - 0.002, color="k", ls=":", lw=1,
                    label="raw - 0.2 pp")
    axes[0].set_title("LCR_fixed (held-out rebuild)"); axes[0].legend(fontsize=7)
    axes[1].axhspan(0.95, 1.05, color="g", alpha=0.12, label="allowed")
    axes[1].set_title("z_IQR ratio"); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_nf_product.png"); plt.close(fig)

    # ---- 3. folds ----
    folds = r.get("folds")
    if folds:
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        fc = [c for c in ("prod", "nbr", "hybrid0.1") if c in folds[0]]
        xs = np.arange(len(folds))
        for i, c in enumerate(fc):
            ax.bar(xs + (i - (len(fc) - 1) / 2) * 0.8 / len(fc),
                   [f[c]["pc_rel_median"] for f in folds], 0.8 / len(fc),
                   color=COLOURS.get(c, "#999"), label=c)
        ax.axhline(20, color="k", ls=":", lw=1, label="20% gate")
        ax.set_xticks(xs); ax.set_xticklabels([f"fold {f['fold']}" for f in folds])
        ax.set_ylabel("median PC improvement (%)")
        ax.set_title("E6-B: leave-fold-out, per objective")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figs / "fig_nf_folds.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
