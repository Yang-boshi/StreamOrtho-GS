"""Figures for the T2 observation table.

Reads the JSON/NPZ written by ``build_obs_min`` / ``t2_cross_stats`` and writes PNGs, one
concern per file (plus one overview panel):

    fig_t2_overview.png       4-panel summary (frames/cell, layers, |dz| ladder, spatial map)
    fig_t2_frames_per_cell.png  histogram of keyframes per cell, with the 4.277 line
    fig_t2_layers.png         layer occupancy + within-layer z_std
    fig_t2_frames_map.png     spatial map of frames/cell (where the overlap is)
    fig_t2_dz_ladder.png      |dz| before / after rank-1 / after rank-4
    fig_t2_sources.png        pointmap vs head for the same sequence

Usage::

    python -m stream_ortho.scripts.diag.plot_t2_obs --obs result-new/P0/obs \
        --figs result-new/P0/figs/t2
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _load_stats(obs_root, seq, source):
    p = pathlib.Path(obs_root) / seq / source / "obs_stats.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="result-new/P0/obs")
    ap.add_argument("--figs", default="result-new/P0/figs/t2")
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--source", default="pointmap")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = pathlib.Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    st = _load_stats(a.obs, a.seq, a.source)
    if st is None:
        raise SystemExit(f"missing stats for {a.seq}/{a.source}")
    d = pathlib.Path(a.obs) / a.seq / a.source
    cross = json.loads((d / "cross_stats.json").read_text()) if (d / "cross_stats.json").exists() else None
    hist = np.load(d / "cross_hist.npz") if (d / "cross_hist.npz").exists() else None

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    def frames_bar(ax):
        ch = {int(k): v for k, v in st["counts_hist"].items()}
        ks = sorted(ch)
        ax.bar(ks, [ch[k] for k in ks], color="#4c72b0", width=0.85)
        ax.axvline(st["frames_per_cell"], color="#c44e52", lw=1.4,
                   label=f"mean = {st['frames_per_cell']:.3f}")
        ax.set_xlabel("keyframes per cell"); ax.set_ylabel("cells")
        ax.set_title(f"frames/cell  ({st['seq']}/{st['source']}, cell={st['cell']} m)")
        ax.legend(loc="upper right", fontsize=8)

    # ---- 1. frames per cell ----
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    frames_bar(ax)
    ax.text(0.02, 0.95, f"≥3 frames: {100*st['hist']['>=3']:.2f}%\n"
                        f"1 frame:   {100*st['hist']['1']:.2f}%\n"
                        f"points/cell: {st['points_per_cell']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout(); fig.savefig(figs / "fig_t2_frames_per_cell.png"); plt.close(fig)

    # ---- 2. layers + z_std ----
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    lh = {int(k): v for k, v in st["layer_hist"].items()}
    tot = sum(lh.values())
    axes[0].bar([str(k) for k in sorted(lh)], [100 * v / tot for k, v in sorted(lh.items())],
                color="#55a868")
    axes[0].set_xlabel("layer id (0 = highest)"); axes[0].set_ylabel("% of rows")
    axes[0].set_title("rows per layer")
    axes[1].bar(["p50", "p90"], [st["z_std_p50"], st["z_std_p90"]], color="#8172b3")
    axes[1].set_ylabel("within-layer z std (m)"); axes[1].set_title("layer flatness")
    fig.tight_layout(); fig.savefig(figs / "fig_t2_layers.png"); plt.close(fig)

    # ---- 3. spatial map ----
    if hist is not None:
        g = hist["grid_frames_per_cell"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(g, cmap="viridis", interpolation="nearest")
        ax.set_title(f"frames per cell, {st['seq']}/{st['source']}")
        ax.set_xlabel("x cell"); ax.set_ylabel("y cell"); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("keyframes")
        fig.tight_layout(); fig.savefig(figs / "fig_t2_frames_map.png"); plt.close(fig)

    # ---- 4. |dz| ladder ----
    if hist is not None:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        bins = np.linspace(0, 10, 61)
        for key_name, lbl, c in (("dz_flat", "before", "#c44e52"),
                                 ("dz_rank1", "rank-1", "#dd8452"),
                                 ("dz_rank4", "rank-4", "#55a868")):
            v = hist[key_name]
            ax.hist(v, bins=bins, histtype="step", lw=1.4, color=c, density=True, label=lbl)
        ax.set_xlabel("|Δz| (m)"); ax.set_ylabel("density")
        ax.set_title("cross-frame |Δz| ladder (flat 2-obs columns)")
        if cross:
            ax.axvline(cross["dz_two_obs_median"], color="k", ls=":", lw=1,
                       label=f"all 2-obs median {cross['dz_two_obs_median']:.2f} m")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs / "fig_t2_dz_ladder.png"); plt.close(fig)

    # ---- 5. sources ----
    other = "head" if a.source == "pointmap" else "pointmap"
    st2 = _load_stats(a.obs, a.seq, other)
    if st2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
        for s, c in ((st, "#4c72b0"), (st2, "#c44e52")):
            ch = {int(k): v for k, v in s["counts_hist"].items()}
            ks = sorted(ch)
            axes[0].plot(ks, [ch[k] / sum(ch.values()) for k in ks], "o-", ms=3, color=c,
                         label=f"{s['source']} ({s['frames_per_cell']:.2f})")
            axes[1].hlines(s["z_std_p50"], 0, 1, color=c, label=s["source"])
        axes[0].set_xlabel("keyframes per cell"); axes[0].set_ylabel("fraction of cells")
        axes[0].set_title("frames/cell"); axes[0].legend(fontsize=8)
        axes[1].set_xticks([]); axes[1].set_ylabel("z std p50 (m)")
        axes[1].set_title("layer flatness"); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs / "fig_t2_sources.png"); plt.close(fig)

    # ---- 6. overview ----
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
    frames_bar(axes[0, 0])
    lh = {int(k): v for k, v in st["layer_hist"].items()}
    tot = sum(lh.values())
    axes[0, 1].bar([str(k) for k in sorted(lh)], [100 * v / tot for k, v in sorted(lh.items())],
                   color="#55a868")
    axes[0, 1].set_title("rows per layer"); axes[0, 1].set_ylabel("% of rows")
    if hist is not None:
        bins = np.linspace(0, 10, 61)
        for key_name, lbl, c in (("dz_flat", "before", "#c44e52"),
                                 ("dz_rank1", "rank-1", "#dd8452"),
                                 ("dz_rank4", "rank-4", "#55a868")):
            axes[1, 0].hist(hist[key_name], bins=bins, histtype="step", lw=1.4, color=c,
                            density=True, label=lbl)
        axes[1, 0].set_xlabel("|Δz| (m)"); axes[1, 0].set_ylabel("density")
        axes[1, 0].set_title("cross-frame residual ladder"); axes[1, 0].legend(fontsize=8)
        im = axes[1, 1].imshow(hist["grid_frames_per_cell"], cmap="viridis",
                               interpolation="nearest")
        axes[1, 1].set_title("frames/cell (space)"); axes[1, 1].grid(False)
        fig.colorbar(im, ax=axes[1, 1], shrink=0.75)
    txt = (f"{st['seq']} / {st['source']}\n"
           f"cell {st['cell']} m, {st['n_frames']} kf, rows {st['n_obs_rows']/1e6:.1f} M\n"
           f"frames/cell {st['frames_per_cell']:.3f}\n"
           f"points/cell {st['points_per_cell']:.2f}\n"
           f"≥3 frames {100*st['hist']['>=3']:.2f}%")
    if cross:
        txt += (f"\n|dz| 2-obs {cross['dz_two_obs_median']:.2f} m"
                f"\nrank-1 {cross['rank1_after']:.2f} m"
                f"\nrank-4 {cross['rank4_after']:.2f} m"
                f"\nFCR {cross['fcr']:.2f} m")
    fig.text(0.012, 0.985, txt, va="top", ha="left", fontsize=8,
             bbox=dict(fc="white", ec="0.6", alpha=0.9))
    fig.tight_layout(rect=(0.16, 0, 1, 1))
    fig.savefig(figs / "fig_t2_overview.png"); plt.close(fig)

    print("wrote figures to", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
