"""T3.8 figures: owner-pair classes, the E2 feasibility shares, and the map evolution.

    fig_owner_classes.png      the four pair classes (counts + share)
    fig_new_boundary_share.png can E2 fix them? common frame / common layer / coverage-only
    fig_new_boundary_map.png   where the new owner boundaries are (spatial density)
    fig_map_evolution.png      top-surface hillshade after every calibration round (raw→it3)
    fig_owner_map.png          which frame owns each cell, raw vs it3 (the fragmentation)
    fig_map_3d.png             3-D surface of the map, raw vs it3
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _hillshade(z, cell):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize
    zz = np.array(z, np.float64)
    live = np.isfinite(zz)
    if live.sum() == 0:
        return np.zeros(zz.shape + (3,))
    fill = np.nanmedian(zz)
    zz = np.where(live, zz, fill)
    ls = LightSource(azdeg=315, altdeg=45)
    vmin, vmax = np.percentile(zz[live], [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    return ls.shade(zz, cmap=plt.get_cmap("terrain"), norm=norm, blend_mode="soft",
                    vert_exag=2.0, dx=cell, dy=cell)


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
    r = json.loads((d / "owner_switch_report.json").read_text())
    ev = np.load(d / "maps_evolution.npz")
    cell = float(ev["cell"]) * int(ev["down"])

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    # ---- 1. classes ----
    cls = r["classes"]
    names = ["stable_same", "boundary_removed", "new_boundary", "persistent_boundary"]
    vals = [cls[k] for k in names]
    cols = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    b = ax.bar(names, vals, color=cols)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,}\n{100*v/sum(vals):.1f}%",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("neighbour pairs (fixed raw set)")
    ax.set_title(f"owner-pair classes: {sum(vals):,} pairs; new boundaries +{cls['new_boundary']:,}")
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout(); fig.savefig(figs / "fig_owner_classes.png"); plt.close(fig)

    # ---- 2. E2 feasibility ----
    nb, pb = r["new_boundary"], r["persistent_boundary"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(2)
    common = [nb["with_common_layer"], pb["with_common_layer"]]
    noframe = [nb["n"] - nb["with_common_frame"], pb["n"] - pb["with_common_frame"]]
    nolayer = [nb["with_common_frame"] - nb["with_common_layer"],
               pb["with_common_frame"] - pb["with_common_layer"]]
    ax.bar(x, common, 0.5, color="#55a868", label="has a common layer (E2 can fix)")
    ax.bar(x, nolayer, 0.5, bottom=common, color="#dd8452", label="common frame, no common layer")
    ax.bar(x, noframe, 0.5, bottom=np.array(common) + np.array(nolayer),
           color="#c44e52", label="no common frame (coverage-only)")
    for i, (n_, c_) in enumerate(zip([nb["n"], pb["n"]], common)):
        ax.text(i, n_ * 1.02, f"{100*c_/max(n_,1):.1f}%  ({c_:,}/{n_:,})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["new_boundary", "persistent_boundary"])
    ax.set_ylabel("pairs"); ax.legend(fontsize=7)
    ax.set_title(f"E2 feasibility: new boundaries with a common layer = "
                 f"{100*r['share_new_with_common_layer']:.1f}%  ->  {r['decision']}")
    fig.tight_layout(); fig.savefig(figs / "fig_new_boundary_share.png"); plt.close(fig)

    # ---- 3. where are the new boundaries ----
    own_r = ev["owner_raw"]
    z_r = ev["z_raw"]
    own_i = ev["owner_it3"]
    live = np.isfinite(z_r) & np.isfinite(ev["z_it3"])
    diff_r = np.zeros_like(live); diff_i = np.zeros_like(live)
    diff_r[:, :-1] = live[:, :-1] & live[:, 1:] & (own_r[:, :-1] != own_r[:, 1:])
    diff_r[:-1, :] |= live[:-1, :] & live[1:, :] & (own_r[:-1, :] != own_r[1:, :])
    diff_i[:, :-1] = live[:, :-1] & live[:, 1:] & (own_i[:, :-1] != own_i[:, 1:])
    diff_i[:-1, :] |= live[:-1, :] & live[1:, :] & (own_i[:-1, :] != own_i[1:, :])
    new = diff_i & ~diff_r
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.imshow(np.where(live, 1.0, np.nan), cmap="Greys", alpha=0.25, interpolation="nearest")
    ax.imshow(np.where(diff_r, 1.0, np.nan), cmap="Blues", alpha=0.5, interpolation="nearest")
    ax.imshow(np.where(new, 1.0, np.nan), cmap="Reds", alpha=0.8, interpolation="nearest")
    ax.set_title("owner boundaries: blue = raw, red = newly created by the calibration")
    ax.set_xlabel("x (downsampled cells)"); ax.set_ylabel("y"); ax.grid(False)
    fig.tight_layout(); fig.savefig(figs / "fig_new_boundary_map.png"); plt.close(fig)

    # ---- 4. map evolution (hillshade per round) ----
    keys = [k for k in ("raw", "it1", "it2", "it3") if f"z_{k}" in ev.files]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, k in zip(axes.ravel(), keys):
        ax.imshow(_hillshade(ev[f"z_{k}"], cell), interpolation="nearest")
        zz = ev[f"z_{k}"]
        ax.set_title(f"{k}   median z {np.nanmedian(zz):.1f} m   "
                     f"IQR {np.nanpercentile(zz,75)-np.nanpercentile(zz,25):.1f} m", fontsize=8)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("map evolution during the all-frame 3-DOF calibration (hillshade, 4x downsampled)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_map_evolution.png"); plt.close(fig)

    # ---- 5. owner map raw vs it3 ----
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, (o, t) in zip(axes, ((own_r, "raw"), (own_i, "it3"))):
        m = np.ma.masked_where(o < 0, o)
        im = ax.imshow(m, cmap="tab20", interpolation="nearest")
        ax.set_title(f"top-layer owner ({t})"); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="keyframe")
    fig.suptitle("owner fragmentation: which frame owns each cell (downsampled)", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_owner_map.png"); plt.close(fig)

    # ---- 6. 3-D surface ----
    fig = plt.figure(figsize=(11.0, 4.4))
    for i, (k, t) in enumerate(((("raw"), "raw"), ("it3", "it3"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        z = ev[f"z_{k}"]
        step = 2
        zz = z[::step, ::step]
        ys, xs = np.mgrid[0:zz.shape[0], 0:zz.shape[1]]
        ax.plot_surface(xs * cell * step, ys * cell * step, np.nan_to_num(zz, nan=np.nanmedian(zz)),
                        cmap="terrain", linewidth=0, antialiased=False, rstride=2, cstride=2)
        ax.set_title(f"{t}: top surface")
        ax.set_zlim(np.nanpercentile(z, 2), np.nanpercentile(z, 98))
        ax.view_init(elev=35, azim=-60)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(figs / "fig_map_3d.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
