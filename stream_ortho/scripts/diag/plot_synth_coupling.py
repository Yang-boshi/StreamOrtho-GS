"""Figures for the synthetic coupling sweep."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/week10-pivotb/synth")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pathlib.Path(a.dir); r = json.loads((d / "synth_coupling.json").read_text())
    g = r["grid"]
    overlaps = sorted({x["overlap"] for x in g}, key=int)
    biases = sorted({x["bias_sigma"] for x in g})
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for key, colour, label in (("raw", "#8c8c8c", "raw"), ("unfrozen", "#c44e52", "unfrozen (re-fuse)"),
                               ("frozen", "#55a868", "frozen")):
        for b in biases:
            xs, ys, ys2 = [], [], []
            for o in overlaps:
                row = next(x for x in g if x["overlap"] == o and x["bias_sigma"] == b)
                xs.append(int(o)); ys.append(row[key]["owner_switch"]); ys2.append(row[key]["boundary_bias_median"])
            axes[0].plot(xs, ys, "-o", ms=3, color=colour, alpha=0.45 + 0.25 * biases.index(b),
                         label=f"{label} b={b}")
            axes[1].plot(xs, ys2, "-o", ms=3, color=colour, alpha=0.45 + 0.25 * biases.index(b))
            axes[2].plot(xs, [next(x for x in g if x["overlap"] == o and x["bias_sigma"] == b)[key]["lcr"] for o in overlaps],
                         "-o", ms=3, color=colour, alpha=0.45 + 0.25 * biases.index(b))
    axes[0].set_title("owner switches"); axes[0].set_yscale("log")
    axes[1].set_title("boundary bias median (m)")
    axes[2].set_title("LCR (frac |dz|<=0.2 m)")
    for ax in axes:
        ax.set_xlabel("frames per cell (overlap)")
    axes[0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Synthetic 2.5D coupling: re-fusing after the correction churns owners; "
                 "freezing keeps them and lowers the boundary bias", fontsize=8.5)
    fig.tight_layout(); fig.savefig(d / "fig_synth_coupling.png"); plt.close(fig)
    print("wrote", d / "fig_synth_coupling.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
