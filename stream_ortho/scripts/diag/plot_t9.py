"""T9 figures.

    fig_nf_selector.png   per-frame selector: median / %improved / worst vs prod and nbr
    fig_nf_fullframe.png  full-frame product metrics per round for prod / nbr / hybrid0.1
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

COLOURS = {"prod": "#4c72b0", "nbr": "#c44e52", "hybrid0.1": "#55a868",
           "selector": "#dd8452", "raw": "#8c8c8c"}


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
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    sel_p = d / "e6_selector.json"
    nf_p = d / "e6_neighbor_fit.json"
    ff_p = d / "fullframe_neighbor.json"

    # ---- selector vs the two fixed objectives ----
    if sel_p.exists() and nf_p.exists():
        sel = json.loads(sel_p.read_text())
        nf = json.loads(nf_p.read_text())
        dirs = list(sel["directions"])
        rows = [("prod", lambda dd: nf["directions"][dd]["prod"]),
                ("nbr", lambda dd: nf["directions"][dd]["nbr"]),
                ("selector", lambda dd: sel["directions"][dd])]
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
        x = np.arange(len(dirs))
        w = 0.8 / len(rows)
        for i, (name, get) in enumerate(rows):
            xx = x + (i - (len(rows) - 1) / 2) * w
            axes[0].bar(xx, [get(dd)["pc_rel_median"] for dd in dirs], w, color=COLOURS[name], label=name)
            axes[1].bar(xx, [get(dd)["pct_improved"] for dd in dirs], w, color=COLOURS[name], label=name)
            axes[2].bar(xx, [get(dd)["min_pc_rel"] for dd in dirs], w, color=COLOURS[name], label=name)
        axes[0].axhline(20, color="k", ls=":", lw=1); axes[0].set_title("median PC improvement (%)")
        axes[1].axhline(90, color="k", ls=":", lw=1); axes[1].set_title("% frames improved (gate 90)")
        axes[2].axhline(-10, color="k", ls=":", lw=1); axes[2].set_title("worst frame (gate -10%)")
        for ax in axes:
            ax.set_xticks(x); ax.set_xticklabels([s.split("_")[0] for s in dirs])
            ax.legend(fontsize=7)
        fig.suptitle("T9-A: half-cell selector (prod/nbr chosen per frame)", fontsize=9)
        fig.tight_layout(); fig.savefig(figs / "fig_nf_selector.png"); plt.close(fig)

    # ---- full-frame ----
    if ff_p.exists():
        ff = json.loads(ff_p.read_text())
        raw = ff["raw"]
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6))
        for ax, (key, title) in zip(axes.ravel(),
                                    (("lcr_fixed", "LCR_fixed (↑)"), ("n_jump", "n_jump (↓)"),
                                     ("ra", "RA (↓)"), ("sei", "SEI (↓)"))):
            for obj, rounds in ff["objectives"].items():
                xs = [r["iter"] for r in rounds]
                ax.plot(xs, [r[key] for r in rounds], "-o", ms=3, color=COLOURS.get(obj), label=obj)
            ax.axhline(raw[key], color="0.5", ls=":", lw=1, label="raw")
            ax.set_xticks([1, 2, 3]); ax.set_xlabel("round"); ax.set_title(title)
            ax.legend(fontsize=7)
        fig.suptitle("T9-B: full-frame product with frozen layers/owner", fontsize=9)
        fig.tight_layout(); fig.savefig(figs / "fig_nf_fullframe.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        for obj, rounds in ff["objectives"].items():
            ax.plot([r["iter"] for r in rounds], [r["b_std"] for r in rounds], "-o", ms=3,
                    color=COLOURS.get(obj), label=f"{obj} b_std")
            ax.plot([r["iter"] for r in rounds], [r["max_step"] for r in rounds], "--s", ms=3,
                    color=COLOURS.get(obj), alpha=0.6, label=f"{obj} max step")
        ax.set_xticks([1, 2, 3]); ax.set_xlabel("round"); ax.set_ylabel("theta statistic (m)")
        ax.set_title("theta drift per round"); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(figs / "fig_nf_theta_fullframe.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
