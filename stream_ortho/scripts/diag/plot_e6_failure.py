"""Figure for the E6 failure-frame diagnosis (three panels).

    (a) per-frame PC improvement for every held frame (both directions), bad frames marked
    (b) layer_match_frac vs |b| for the probed frames -- is it a layer or a theta problem?
    (c) PC before -> after for the probed frames, plus the flat-only counterfactual
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

COLOURS = {"odd_build_even_held": "#c44e52", "even_build_odd_held": "#4c72b0"}


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
    r = json.loads((d / "failure_frames.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    # (a) all held frames
    ax = axes[0]
    for name, v in r["directions"].items():
        ts = v["theta_series"]
        kf = np.array([t["kf"] for t in ts])
        rel = np.array([t["pc_rel_pct"] for t in ts])
        ax.plot(kf, rel, "-o", ms=3, color=COLOURS[name], label=name.split("_")[0])
        bad = [f["kf"] for f in v["frames"] if f["kind"] == "bad"]
        ax.scatter([k for k in kf if k in bad], [rl for k, rl in zip(kf, rel) if k in bad],
                   s=60, facecolors="none", edgecolors=COLOURS[name], linewidths=1.4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhline(20, color="k", ls=":", lw=1, label="20% target")
    ax.set_xlabel("keyframe"); ax.set_ylabel("per-frame PC improvement (%)")
    ax.set_title("(a) every held frame (circled = the failing ones)")
    ax.legend(fontsize=7)

    # (b) layer match vs |b|
    ax = axes[1]
    for name, v in r["directions"].items():
        for kind, mk in (("bad", "X"), ("good", "o")):
            pts = [f for f in v["frames"] if f["kind"] == kind]
            ax.scatter([f["layer_match_frac"] for f in pts], [abs(f["b"]) for f in pts],
                       marker=mk, s=46, color=COLOURS[name], alpha=0.85,
                       label=f"{name.split('_')[0]} {kind}")
    ax.axhline(3.0, color="k", ls=":", lw=1, label="|b| = 3 m")
    ax.axvline(0.6, color="k", ls="--", lw=1, label="layer match 0.6")
    ax.set_xlabel("layer_match_frac (near layer == product top)")
    ax.set_ylabel("|b| (m)")
    ax.set_title("(b) no layer mismatch, no |b| outlier pattern")
    ax.legend(fontsize=6)

    # (c) before -> after and the flat counterfactual
    ax = axes[2]
    xs = []
    labels = []
    for name, v in r["directions"].items():
        for f in v["frames"]:
            xs.append((f["pc_before"], f["pc_after"], f["pc_after_flat"]))
            labels.append(f"{f['kf']}{'*' if f['kind']=='bad' else ''}")
    xs = np.array(xs)
    idx = np.arange(len(xs))
    ax.plot(idx, xs[:, 0], "o", color="#8c8c8c", label="before")
    ax.plot(idx, xs[:, 1], "o", color="#55a868", label="after (t3)")
    ax.plot(idx, xs[:, 2], "x", color="#c44e52", label="after (flat only)")
    for i in idx:
        ax.plot([i, i], [xs[i, 0], xs[i, 1]], color="0.8", lw=0.8, zorder=0)
    ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("PC (m)"); ax.set_title("(c) '*' = failing frame; flat refit is worse")
    ax.legend(fontsize=7)

    fig.suptitle(f"E6 failure frames — verdict: {r['verdict']}", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_failure_frames.png"); plt.close(fig)
    print("wrote", figs / "fig_failure_frames.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
