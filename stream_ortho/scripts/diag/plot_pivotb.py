"""Figures for the PIVOT-B pre-check (``results/week10-pivotb/{seq}/figs``).

    fig_preflight_overview.png  2x2: even/odd PC, full-frame LCR/RA, theta stability, SEI
    fig_even_odd_pc.png         PC per method x direction, with the 20% gate
    fig_fullframe_lcr_ra.png    product metrics before/after the full-frame rebuild
    fig_theta_stability.png     per-frame b/tx/ty and adjacent-frame steps
    fig_head_pc.png             head vs pointmap: abs_t3 PC change
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

METHOD_COLOURS = {"raw": "#8c8c8c", "abs_t3": "#4c72b0", "field4x4": "#dd8452"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/week10-pivotb")
    ap.add_argument("--seq", default="ieu")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = pathlib.Path(a.root) / a.seq
    figs = root / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    r = json.loads((root / "preflight.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    eo = r["even_odd"]
    dirs = list(eo)
    methods = [m for m in ("raw", "abs_t3", "field4x4") if m in eo[dirs[0]]["methods"]]

    # ---- E-A ----
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        x = np.arange(len(dirs)) + (i - (len(methods) - 1) / 2) * w
        y = [eo[d_]["methods"][m]["pc"] for d_ in dirs]
        ax.bar(x, y, w, color=METHOD_COLOURS.get(m, "#999"), label=m)
        for xi, yi, d_ in zip(x, y, dirs):
            pct = eo[d_]["methods"][m]["pc_vs_raw_pct"]
            ax.text(xi, yi, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(dirs)))
    ax.set_xticklabels([d_.replace("_", "\n") for d_ in dirs], fontsize=8)
    ax.set_ylabel("held↔build PC (m)")
    ax.set_title("E-A even/odd: PC must drop ≥20% in BOTH directions")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "fig_even_odd_pc.png"); plt.close(fig)

    # ---- E-B ----
    ff = r["fullframe"]
    raw, ab = ff["median"]["raw"], ff["median"]["abs_t3"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, k, better in ((axes[0], "lcr", "up"), (axes[1], "ra", "down"), (axes[2], "sei", "down")):
        vals = [raw.get(k, np.nan), ab.get(k, np.nan)]
        ax.bar(["raw", "abs_t3\n(full rebuild)"], vals, color=["#8c8c8c", "#4c72b0"])
        ax.set_title({"lcr": "LCR (↑)", "ra": "RA m (↓)", "sei": "SEI (↓)"}[k])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("E-B full-frame rebuild: does the calibration reach the product?", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_fullframe_lcr_ra.png"); plt.close(fig)

    # ---- theta stability ----
    ts = r["theta_stability"]
    th = np.asarray(ts["theta"])
    frames = np.asarray(ts["frames"])
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True)
    axes[0].plot(frames, th[:, 0], "-o", ms=2, color="#c44e52")
    axes[0].set_ylabel("b (m)"); axes[0].set_title("abs_t3 field, out-of-fold, per frame")
    axes[1].plot(frames, th[:, 1] * 1000, "-o", ms=2, color="#4c72b0", label="tx ×1000")
    axes[1].plot(frames, th[:, 2] * 1000, "-o", ms=2, color="#55a868", label="ty ×1000")
    axes[1].set_ylabel("tilt (‰)"); axes[1].set_xlabel("keyframe index"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_theta_stability.png"); plt.close(fig)

    # ---- head ----
    if "head" in r:
        h = r["head"]["methods"]
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ms = [m for m in ("raw", "abs_t3") if m in h]
        ax.bar(np.arange(len(ms)), [h[m]["pc"] for m in ms],
               color=[METHOD_COLOURS.get(m, "#999") for m in ms])
        for i, m in enumerate(ms):
            ax.text(i, h[m]["pc"], f"{h[m]['pc']:.3f}  ({h[m]['pc_vs_raw_pct']:+.0f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(ms))); ax.set_xticklabels(ms)
        ax.set_ylabel("held↔build PC (m)")
        ax.set_title("E-C head source: abs_t3 PC")
        fig.tight_layout(); fig.savefig(figs / "fig_head_pc.png"); plt.close(fig)

    # ---- overview ----
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    for i, m in enumerate(methods):
        x = np.arange(len(dirs)) + (i - (len(methods) - 1) / 2) * w
        axes[0, 0].bar(x, [eo[d_]["methods"][m]["pc"] for d_ in dirs], w,
                       color=METHOD_COLOURS.get(m, "#999"), label=m)
    axes[0, 0].set_xticks(np.arange(len(dirs)))
    axes[0, 0].set_xticklabels([d_.split("_")[0] for d_ in dirs])
    axes[0, 0].set_title("E-A PC (held↔build)"); axes[0, 0].legend(fontsize=7)
    axes[0, 1].bar(["raw", "abs_t3"], [raw.get("lcr", np.nan), ab.get("lcr", np.nan)],
                   color=["#8c8c8c", "#4c72b0"])
    axes[0, 1].set_title("E-B LCR after full-frame rebuild")
    axes[1, 0].plot(frames, th[:, 0], "-o", ms=2, color="#c44e52")
    axes[1, 0].set_title("E-C theta b per frame"); axes[1, 0].set_xlabel("keyframe")
    vals = [raw.get("sei", np.nan), ab.get("sei", np.nan)]
    axes[1, 1].bar(["raw", "abs_t3"], vals, color=["#8c8c8c", "#4c72b0"])
    axes[1, 1].set_title("E-B SEI (seam step / interior step)")
    g = r.get("gate", {})
    txt = (f"{r['seq']} / {r['source']}  cell {r['cell']}  {r['n_frames']} kf\n"
           f"G-A both dirs: {g.get('G_A_both_directions')}\n"
           f"  " + "  ".join(f"{k}: {v:+.1f}%" for k, v in g.get("G_A_detail", {}).items()) + "\n"
           f"G-B LCR ok: {g.get('G_B_lcr_not_worse')}   RA ok: {g.get('G_B_ra_not_worse')}\n"
           f"head PC vs raw: {g.get('head_pc_vs_raw_pct')}")
    fig.text(0.012, 0.985, txt, va="top", fontsize=7.5,
             bbox=dict(fc="white", ec="0.6", alpha=0.9))
    fig.tight_layout(rect=(0.22, 0, 1, 1))
    fig.savefig(figs / "fig_preflight_overview.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
