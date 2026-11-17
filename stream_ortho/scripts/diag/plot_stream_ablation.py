"""Streaming vs offline: what the online path costs, and why the window does not fix it."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/week10-stream/ablation_stream.json")
    ap.add_argument("--out", default="results/week10-stream/figs/ablation_stream.png")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.loads(pathlib.Path(a.file).read_text())["rows"]
    order = ["offline_raw", "offline_oracle", "stream_raw", "stream_calib", "stream_window_fp"]
    labels = ["offline raw", "offline oracle\n(3 rounds)", "stream raw", "stream + calib",
              "stream + window FP"]
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    x = np.arange(len(order))
    cols = ["#8c8c8c", "#55a868", "#4c72b0", "#dd8452", "#c44e52"]
    for ax, key, title in ((axes[0], "ra", "RA (lower is better)"),
                           (axes[1], "lcr", "LCR_fixed (higher is better)"),
                           (axes[2], "jump", "n_jump (lower is better)")):
        vals = [r[k][key] for k in order]
        ax.bar(x, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if key != "jump" else f"{v/1000:.0f}k", ha="center",
                    va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, fontsize=7)
        ax.set_title(title)
    axes[0].axhline(0.801, color="k", ls=":", lw=1)
    axes[1].axhline(0.9075, color="k", ls=":", lw=1)
    fig.suptitle("Streaming costs RA +27% via incremental accumulation (not via the calibration); "
                 "the window fixed point makes it worse", fontsize=8.5)
    fig.tight_layout()
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
