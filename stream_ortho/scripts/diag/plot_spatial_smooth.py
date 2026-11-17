"""T3.7 figures: spatial smoothing attempts + the owner-boundary diagnosis.

    fig_spatial_smooth.png  LCR_fixed / n_jump / RA / SEI for raw, it3, S1(sigma), S2
    fig_owner_split.png     jump rate and pair counts split by owner (same vs different frame)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


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
    r = json.loads((d / "spatial_smooth.json").read_text())
    res = r["results"]
    labels = [("raw" if x["method"] == "raw" else
               "it3" if x["method"] == "it3" else
               f"S1 σ={x['sigma']:g}" if x["method"] == "S1" else "S2") for x in res]
    cols = ["#8c8c8c", "#c44e52", "#4c72b0", "#4c72b0", "#4c72b0", "#55a868"]
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8))
    for ax, (k, title, ref) in zip(axes.ravel(), (("lcr_fixed", "LCR_fixed (↑)", r["raw_ref"]["lcr_fixed"]),
                                                  ("n_jump", "n_jump on raw pairs (↓)", r["raw_ref"]["n_jump"]),
                                                  ("ra", "RA (↓)", r["raw_ref"]["ra"]),
                                                  ("sei", "SEI (↓)", r["raw_ref"]["sei"]))):
        vals = [x[k] for x in res]
        ax.bar(np.arange(len(res)), vals, color=cols[:len(res)])
        ax.axhline(ref, color="0.4", ls=":", lw=1, label="raw")
        ax.set_xticks(np.arange(len(res))); ax.set_xticklabels(labels, rotation=20, fontsize=7)
        ax.set_title(title); ax.legend(fontsize=7)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v:,}",
                    ha="center", va="bottom", fontsize=6.5)
    fig.suptitle("T3.7: smoothing the correction field makes it WORSE (verdict: %s)" % r["verdict"],
                 fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_spatial_smooth.png"); plt.close(fig)

    # ---- owner split ----
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    x = np.arange(len(res))
    same_n = [o["owner_split"]["same"]["n"] for o in res]
    diff_n = [o["owner_split"]["diff"]["n"] for o in res]
    axes[0].bar(x - 0.2, same_n, 0.4, color="#4c72b0", label="same owner")
    axes[0].bar(x + 0.2, diff_n, 0.4, color="#c44e52", label="different owner")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, fontsize=7)
    axes[0].set_title("neighbour pairs by owner"); axes[0].legend(fontsize=7)
    axes[0].set_ylabel("pairs")

    same_r = [o["owner_split"]["same"]["rate"] for o in res]
    diff_r = [o["owner_split"]["diff"]["rate"] for o in res]
    axes[1].plot(x, same_r, "-o", color="#4c72b0", label="same owner")
    axes[1].plot(x, diff_r, "-o", color="#c44e52", label="different owner")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, fontsize=7)
    axes[1].set_title("jump rate |Δz| > 1 m"); axes[1].set_ylabel("rate"); axes[1].legend(fontsize=7)

    contrib = [o["owner_split"]["same"]["jumps"] + o["owner_split"]["diff"]["jumps"] for o in res]
    axes[2].bar(x, contrib, color="#dd8452")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=20, fontsize=7)
    axes[2].set_title("total jumps"); axes[2].set_ylabel("count")
    fig.suptitle("diagnosis: same-owner jumps are stable ~5%; the regression rides on the "
                 "+46% owner-boundary pairs", fontsize=8.5)
    fig.tight_layout(); fig.savefig(figs / "fig_owner_split.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
