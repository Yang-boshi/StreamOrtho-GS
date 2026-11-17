"""Windowed re-calibration: it does not help (RA stays above the offline oracle)."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/week10-stream/ieu_window")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pathlib.Path(a.dir)
    r = json.loads((d / "window_ablation.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
    ax = axes[0]
    names = ["baseline\n(W=0)", "W10/B5/R1", "W10/B5/R2", "W20/B5/R1"]
    ra = [r["baseline"]["quality"]["ra"]] + [s["ra"] for s in r["sweep"]]
    lcr = [r["baseline"]["quality"]["lcr_fixed"]] + [s["lcr"] for s in r["sweep"]]
    x = np.arange(len(names))
    ax.bar(x - 0.2, ra, 0.4, color="#c44e52", label="RA (lower is better)")
    ax.bar(x + 0.2, lcr, 0.4, color="#4c72b0", label="LCR_fixed (higher is better)")
    ax.axhline(0.744, color="k", ls=":", lw=1, label="offline RA 0.744")
    ax.axhline(0.781, color="k", ls="--", lw=1, label="gate 0.781")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_title("windowed re-calibration does not move RA")
    ax.legend(fontsize=6)

    ax = axes[1]
    fr = json.loads((pathlib.Path("results/week10-stream/ieu/stream.json")).read_text())["frames"]
    n = [e["n"] for e in fr]
    tot = [e["calib_ms"] + e["insert_ms"] + e["tile_ms"] for e in fr]
    ax.plot(n, tot, "-", color="#4c72b0", label="baseline per-frame")
    ax.axhline(r["latency_spike_ms"], color="#c44e52", ls="--", lw=1,
               label=f"window pass ~{r['latency_spike_ms']:.0f} ms")
    ax.axhline(500, color="k", ls=":", lw=1, label="P50 budget")
    ax.set_xlabel("frames"); ax.set_ylabel("ms")
    ax.set_title("the pass costs ~1.6 s per trigger")
    ax.legend(fontsize=6)
    fig.suptitle("T19: window re-calibration is a no-op (theta ~ 0) and would break the latency "
                 "budget; RA is a product-accumulation property", fontsize=8.5)
    fig.tight_layout(); fig.savefig(d / "figs" / "window_ablation.png") if (d / "figs").exists() else None
    (d / "figs").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(d / "figs" / "window_ablation.png"); plt.close(fig)
    print("wrote", d / "figs" / "window_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
