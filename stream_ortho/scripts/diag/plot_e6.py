"""E6 figures.

    fig_e6_even_odd_pc.png   PC per direction (raw vs frozen) with the -20% gate
    fig_e6_heldout_lcr.png   LCR_fixed / RA / SEI per fold and direction
    fig_e6_reference_ncc.png reference: NCC / Rel-RMSE (even vs odd products) [非外部参考]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


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
    r = json.loads((d / "heldout.json").read_text())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    ea = r["even_odd"]
    dirs = list(ea)

    # ---- 1. PC per direction ----
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(dirs))
    raw = [ea[k]["raw"]["pc"] for k in dirs]
    fro = [ea[k]["frozen"]["pc"] for k in dirs]
    ax.bar(x - 0.2, raw, 0.4, color="#8c8c8c", label="raw")
    ax.bar(x + 0.2, fro, 0.4, color="#55a868", label="frozen (held-out)")
    for i, k in enumerate(dirs):
        pct = 100 * (1 - ea[k]["frozen"]["pc"] / ea[k]["raw"]["pc"])
        ax.text(i + 0.2, fro[i], f"{pct:+.1f}%", ha="center", va="bottom", fontsize=8)
        thr = 0.8 * ea[k]["raw"]["pc"]
        ax.plot([i - 0.35, i + 0.35], [thr, thr], "k:", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([k.replace("_", "\n") for k in dirs], fontsize=7.5)
    ax.set_ylabel("held↔build PC (m)")
    ax.set_title("E6-A: held-out PC (dotted = the -20% gate)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "fig_e6_even_odd_pc.png"); plt.close(fig)

    # ---- 2. folds: LCR / RA / SEI ----
    folds = r["folds"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    xs = [f["fold"] for f in folds]
    for ax, (key, title, rel) in zip(axes, (("lcr_fixed", "LCR_fixed (↑)", False),
                                            ("ra", "RA (↓)", True),
                                            ("sei", "SEI (↓)", True))):
        rv = [f["raw"][key] for f in folds]
        fv = [f["frozen"][key] for f in folds]
        ax.plot(xs, rv, "-o", color="#8c8c8c", label="raw")
        ax.plot(xs, fv, "-o", color="#55a868", label="frozen")
        ax.set_xticks(xs); ax.set_xlabel("fold"); ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("E6-B: leave-fold-out (build 44 / held 11), frozen assignment", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_e6_heldout_lcr.png"); plt.close(fig)

    # ---- 3. reference ----
    ref = r.get("reference", {})
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    if ref.get("found"):
        vals = [ref["ncc"], ref["rel_rmse"]]
        ax.bar(["NCC", "Rel-RMSE"], vals, color=["#4c72b0", "#c44e52"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"even vs odd product [{ref.get('label','')}]\n"
                     f"{ref.get('n_cells',0):,} common cells, median |Δz| "
                     f"{ref.get('median_abs_m', float('nan')):.2f} m", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no external reference", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(figs / "fig_e6_reference_ncc.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
