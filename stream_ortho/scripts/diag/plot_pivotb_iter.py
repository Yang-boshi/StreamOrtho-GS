"""Convergence figures for the all-frame iterative calibration (round 2).

    fig_iter_convergence.png  LCR / RA / SEI / PC per iteration against the raw product
    fig_iter_theta.png        theta b_std and adjacent-frame step per iteration
    fig_iter_overview.png     2x2 summary with the decision text
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

COLOURS = {"lcr": "#55a868", "ra": "#c44e52", "sei": "#8172b3", "pc": "#4c72b0"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="round-2 directory with preflight_iterative.json")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pathlib.Path(a.dir)
    figs = d / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    r = json.loads((d / "preflight_iterative.json").read_text())
    raw = r["fullframe_iterative"]["raw"]
    its = r["fullframe_iterative"]["iters"]
    dec = r["decision"]
    x = np.arange(len(its) + 1)          # 0 = raw

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.6))
    for ax, (k, label, better) in zip(axes.ravel(), (("lcr", "LCR (↑)", "up"),
                                                     ("ra", "RA m (↓)", "down"),
                                                     ("sei", "SEI (↓)", "down"),
                                                     ("pc", "PC (↓)", "down"))):
        vals = [raw[k]] + [it[k] for it in its] if k in raw else [np.nan] + [it[k] for it in its]
        ax.plot(x, vals, "-o", color=COLOURS[k], lw=1.6, ms=4)
        ax.axhline(vals[0], color="0.5", ls=":", lw=1, label="raw")
        ax.set_xticks(x); ax.set_xticklabels(["raw"] + [f"it{i}" for i in range(1, len(its) + 1)])
        ax.set_title(label)
    fig.tight_layout(); fig.savefig(figs / "fig_iter_convergence.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))
    axes[0].plot(x[1:], [it["b_std"] for it in its], "-o", color="#dd8452")
    axes[0].set_title("theta b_std per iteration"); axes[0].set_xlabel("iteration")
    axes[0].set_xticks(x[1:])
    axes[1].plot(x[1:], [it["max_step"] for it in its], "-o", color="#c44e52", label="max step")
    axes[1].plot(x[1:], [it["median_step"] for it in its], "-o", color="#4c72b0", label="median step")
    axes[1].set_title("adjacent-frame theta step (m)"); axes[1].set_xlabel("iteration")
    axes[1].legend(fontsize=8); axes[1].set_xticks(x[1:])
    fig.tight_layout(); fig.savefig(figs / "fig_iter_theta.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2))
    last = its[-1]
    for ax, (k, label) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), (("lcr", "LCR"), ("ra", "RA m"),
                                                                     ("pc", "PC"))):
        vals = [raw[k]] + [it[k] for it in its] if k in raw else [np.nan] + [it[k] for it in its]
        ax.plot(x, vals, "-o", color=COLOURS[k], lw=1.6, ms=4)
        ax.set_xticks(x); ax.set_xticklabels(["raw"] + [f"it{i}" for i in range(1, len(its) + 1)])
        ax.set_title(label)
    axes[1, 1].axis("off")
    txt = (f"all-frame abs_t3, {len(its)} iterations\n"
           f"raw: LCR {raw['lcr']:.4f}  RA {raw['ra']:.3f}  SEI {raw['sei']:.2f}\n"
           f"it{len(its)}: LCR {last['lcr']:.4f} ({dec['lcr_delta']:+.4f})  "
           f"RA {last['ra']:.3f} ({dec['ra_pct']:+.1f}%)  SEI {last['sei']:.2f} ({dec['sei_pct']:+.1f}%)\n"
           f"PC {raw.get('pc', float('nan')):.3f} -> {last['pc']:.3f}\n"
           f"z_IQR {raw['z_iqr']:.2f} -> {last['z_iqr']:.2f} (no flattening)\n"
           f"coverage {raw['coverage_cells']:.4f} -> {last['coverage_cells']:.4f}\n"
           f"theta b_std {dec['theta_b_std'][0]:.2f} -> {dec['theta_b_std'][-1]:.2f}\n"
           f"max step {dec['theta_max_step'][0]:.2f} -> {dec['theta_max_step'][-1]:.2f}\n"
           f"G-B2: {dec['G_B2_calibration_reaches_product']}")
    axes[1, 1].text(0.02, 0.98, txt, va="top", fontsize=8, family="monospace",
                    bbox=dict(fc="white", ec="0.7"))
    fig.tight_layout(); fig.savefig(figs / "fig_iter_overview.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
