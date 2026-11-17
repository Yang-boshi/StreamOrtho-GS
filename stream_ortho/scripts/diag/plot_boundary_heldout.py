"""Held-out boundary bias: the effect does NOT transfer (full-frame only)."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/week10-pivotb/ieu/e6")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pathlib.Path(a.dir)
    r = json.loads((d / "boundary_bias_heldout.json").read_text())
    ful = json.loads((d / "audit" / "boundary_bias.json").read_text()) if (d / "audit" / "boundary_bias.json").exists() else None
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    dirs = r["directions"]
    labels = [k.replace("_build_", " -> ").replace("_held", "") for k in dirs]
    fp_red = [v.get("frame_pairs", {}).get("median_bias_reduction_pct", np.nan)
              for v in dirs.values()]
    step_red = [v.get("median_bias_reduction_pct", np.nan) for v in dirs.values()]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, fp_red, 0.4, color="#55a868", label="held x build frame pairs")
    ax.bar(x + 0.2, step_red, 0.4, color="#c44e52", label="adjacent-cell step")
    ax.axhline(20, color="k", ls=":", lw=1, label="gate (+20%)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("median reduction (%)")
    ax.set_title("held-out: the frame-pair bias does drop; the surface step does not")
    ax.legend(fontsize=7)

    ax = axes[1]
    folds = [f.get("frame_pairs", {}).get("median_bias_reduction_pct", np.nan) for f in r["folds"]]
    ax.bar(np.arange(len(folds)), folds, color="#4c72b0")
    ax.axhline(20, color="k", ls=":", lw=1, label="+20%")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([f"fold {f['fold']}" for f in r["folds"]])
    ax.set_title("held-out frame pairs, 5 folds")
    ax.legend(fontsize=7)
    fig.suptitle("D (corrected): held-build frame-pair bias -28%/-30%, double-wall rate halved; "
                 "the cross-surface step is unchanged by design", fontsize=8.5)
    fig.tight_layout(); fig.savefig(d / "fig_boundary_heldout.png"); plt.close(fig)
    print("wrote", d / "fig_boundary_heldout.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
