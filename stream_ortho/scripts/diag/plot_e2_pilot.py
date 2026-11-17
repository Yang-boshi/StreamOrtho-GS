"""T4-pilot figures.

    fig_lcr_bar.png        LCR_fixed / n_jump / owner_switch per method, both tiles
    fig_pilot_map.png      hillshade of the two sectors: top vs owner_mrf
    fig_owner_map.png      owner image of the same sectors: raw / it3 / mrf
    fig_tradeoff.png       lambda sweep: owner_switch vs LCR_fixed (and RA)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _hs(z, cell=0.45):
    from matplotlib.colors import LightSource, Normalize
    import matplotlib.pyplot as plt
    zz = np.array(z, np.float64)
    live = np.isfinite(zz)
    if not live.any():
        return np.zeros(zz.shape + (3,))
    zz = np.where(live, zz, np.nanmedian(zz))
    lo, hi = np.percentile(zz[live], [2, 98])
    return LightSource(azdeg=315, altdeg=45).shade(
        zz, cmap=plt.get_cmap("terrain"), norm=Normalize(lo, hi),
        blend_mode="soft", vert_exag=2.0, dx=cell, dy=cell)


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
    r = json.loads((d / "pilot.json").read_text())
    m = np.load(d / "pilot_maps.npz")
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    tiles = r["tiles_result"]

    # ---- 1. bars ----
    methods = ["raw", "it3", "quality", "default", "no-owner (ablation)"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ax, key, title in zip(axes, ("lcr_fixed", "n_jump", "owner_switch"),
                              ("LCR_fixed (↑)", "n_jump (↓)", "owner_switch (↓)")):
        w = 0.8 / len(tiles)
        for ti, ot in enumerate(tiles):
            vals = []
            for meth in methods:
                src = ot["baselines"] if meth in ot["baselines"] else ot["mrf"]
                vals.append(src[meth][key])
            x = np.arange(len(methods)) + (ti - 0.5) * w
            ax.bar(x, vals, w, label=f"tile {ot['tile']['id']}")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=20, fontsize=7)
        ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle("E2 pilot: the owner term barely moves the switch count", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_lcr_bar.png"); plt.close(fig)

    # ---- 2. hillshade ----
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
    for ti, ot in enumerate(tiles):
        tid = ot["tile"]["id"]
        axes[ti, 0].imshow(_hs(m[f"t{tid}_it3_z"]), interpolation="nearest")
        axes[ti, 0].set_title(f"tile {tid}: it3 (top)", fontsize=8)
        axes[ti, 1].imshow(_hs(m[f"t{tid}_mrf_z"]), interpolation="nearest")
        axes[ti, 1].set_title(f"tile {tid}: owner_mrf", fontsize=8)
        for ax in axes[ti]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("E2 pilot sectors: top vs owner/layer-coherent selection (hillshade)", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_pilot_map.png"); plt.close(fig)

    # ---- 3. owner maps ----
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6))
    for ti, ot in enumerate(tiles):
        tid = ot["tile"]["id"]
        for ai, key in enumerate(("raw_o", "it3_o", "mrf_o")):
            o = m[f"t{tid}_{key}"]
            ax = axes[ti, ai]
            im = ax.imshow(np.ma.masked_where(o < 0, o), cmap="tab20", interpolation="nearest")
            ax.set_title(f"tile {tid}: {key.replace('_o','')}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="owning keyframe")
    fig.suptitle("owner fragmentation inside the pilot sectors", fontsize=9)
    fig.savefig(figs / "fig_owner_map.png", bbox_inches="tight"); plt.close(fig)

    # ---- 4. tradeoff ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ot in tiles:
        tid = ot["tile"]["id"]
        names = list(ot["mrf"])
        sw = [ot["mrf"][n]["owner_switch"] for n in names]
        lcr = [ot["mrf"][n]["lcr_fixed"] for n in names]
        ra = [ot["mrf"][n]["ra"] for n in names]
        axes[0].scatter(sw, lcr, label=f"tile {tid}")
        axes[1].scatter(sw, ra, label=f"tile {tid}")
        base = ot["baselines"]
        axes[0].axhline(base["raw"]["lcr_fixed"], color="0.5", ls=":", lw=1)
        axes[1].axhline(base["raw"]["ra"], color="0.5", ls=":", lw=1)
    axes[0].set_xlabel("owner_switch"); axes[0].set_ylabel("LCR_fixed")
    axes[0].set_title("lambda sweep: switches vs LCR (dotted = raw)")
    axes[1].set_xlabel("owner_switch"); axes[1].set_ylabel("RA")
    axes[1].set_title("lambda sweep: switches vs RA")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_tradeoff.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
