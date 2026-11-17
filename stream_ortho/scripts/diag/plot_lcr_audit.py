"""T3.6 figure: the LCR denominator artefact test.

    fig_lcr_fixed.png      fixed-denominator LCR vs the standard LCR, plus the jump counts
    fig_lcr_denominator.png standard denominator / inconsistent pairs vs the fixed pair set
    fig_theta_frames.png   per-frame b_i and the |db| > 1 m steps (task C)
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
    r = json.loads((d / "lcr_fixed.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    # ---- fixed vs standard LCR + jumps ----
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    ax = axes[0]
    ax.bar([0, 1], [r["lcr_fixed_raw"], r["lcr_fixed_it3"]], color=["#8c8c8c", "#c44e52"])
    for i, v in enumerate([r["lcr_fixed_raw"], r["lcr_fixed_it3"]]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "it3"])
    ax.set_ylim(min(r["lcr_fixed_raw"], r["lcr_fixed_it3"]) - 0.02, 1.0)
    ax.set_title(f"LCR on the FIXED pair set ({r['n_pairs']:,})\n"
                 f"delta {r['delta_pp']:+.2f} pp  ->  {r['verdict']}")

    ax = axes[1]
    ax.bar([0, 1], [r["n_jump_raw"], r["n_jump_it3"]], color=["#8c8c8c", "#c44e52"])
    for i, v in enumerate([r["n_jump_raw"], r["n_jump_it3"]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "it3"])
    ax.set_title("jumps |Δz| > 1 m on the same pairs")

    ax = axes[2]
    ax.bar([0, 1], [r["lcr_std_raw"], r["lcr_std_it3"]], color=["#8c8c8c", "#4c72b0"])
    for i, v in enumerate([r["lcr_std_raw"], r["lcr_std_it3"]]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "it3"])
    ax.set_ylim(min(r["lcr_std_raw"], r["lcr_std_it3"]) - 0.02, 1.0)
    ax.set_title(f"standard LCR (own denominator)\n"
                 f"n_shared {r['n_shared_it3']/r['n_shared_raw']:.0%} of raw")
    fig.suptitle("T3.6-A: the -1.9 pp is NOT a denominator artefact once the pairs are frozen",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_lcr_fixed.png"); plt.close(fig)

    # ---- denominator ----
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(2)
    ax.bar(x - 0.2, [r["n_shared_raw"], r["n_shared_it3"]], 0.4, color="#4c72b0", label="n_shared (denominator)")
    ax.bar(x + 0.2, [r["n_inconsistent_raw"], r["n_inconsistent_it3"]], 0.4,
           color="#dd8452", label="n_inconsistent (numerator)")
    ax.axhline(r["n_pairs"], color="k", ls=":", lw=1, label=f"fixed pair set {r['n_pairs']:,}")
    ax.set_xticks(x); ax.set_xticklabels(["raw", "it3"])
    ax.set_ylabel("pairs"); ax.legend(fontsize=7)
    ax.set_title("why the ratio drops: the denominator shrinks 20%")
    fig.tight_layout(); fig.savefig(figs / "fig_lcr_denominator.png"); plt.close(fig)

    # ---- theta per frame (task C) ----
    th = r.get("theta_final", {})
    if th:
        ks = sorted(int(k) for k in th)
        b = np.array([th[str(k)][0] if str(k) in th else th[k][0] for k in ks])
        tx = np.array([th[str(k)][1] if str(k) in th else th[k][1] for k in ks])
        ty = np.array([th[str(k)][2] if str(k) in th else th[k][2] for k in ks])
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.2), sharex=True)
        axes[0].plot(ks, b, "-o", ms=2.5, color="#c44e52")
        for f in r["theta_stability"]["jumps_at_fold_boundary"]:
            axes[0].axvline(f, color="0.6", ls="--", lw=1)
        for f in r["theta_stability"]["frames_delta_b_gt_1m"]:
            axes[0].axvline(f, color="#dd8452", ls=":", lw=0.6, alpha=0.6)
        axes[0].set_ylabel("b (m)")
        axes[0].set_title(f"per-frame 3-DOF: b_std {r['theta_stability']['b_std']:.2f} m, "
                          f"max step {r['theta_stability']['max_step']:.2f} m, "
                          f"median step {r['theta_stability']['median_step']:.4f} m\n"
                          f"dotted = |Δb|>1 m ({len(r['theta_stability']['frames_delta_b_gt_1m'])} frames), "
                          f"dashed = fold boundary")
        axes[1].plot(ks, tx * 1000, "-o", ms=2, color="#4c72b0", label="tx ×1000")
        axes[1].plot(ks, ty * 1000, "-o", ms=2, color="#55a868", label="ty ×1000")
        axes[1].set_xlabel("keyframe index"); axes[1].set_ylabel("tilt (‰)"); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figs / "fig_theta_frames.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
