"""Figure for F1 (frozen layer/owner during calibration).

    fig_freeze_owner.png   raw vs it3 (unfrozen) vs frozen: LCR_fixed / n_jump / RA / PC
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = pathlib.Path(a.file)
    r = json.loads(f.read_text())
    figs = f.parent / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    raw = r["raw"]
    it3 = r.get("it3_reference") or {}
    fin = r["iters"][-1]
    it3_pc = 0.234                       # round2/preflight_iterative.json (it3 PC)
    rows = [("raw", raw, None),
            ("it3 (unfrozen)", it3, it3_pc),
            ("frozen", fin, r["pc"])]
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6))
    panels = [("lcr_fixed", "LCR_fixed (↑)", False),
              ("n_jump", "n_jump on fixed raw pairs (↓)", False),
              ("ra", "RA (↓)", False),
              ("pc", "PC frame↔frame (↓)", True)]
    for ax, (key, title, is_pc) in zip(axes.ravel(), panels):
        vals = [(rr[2] if is_pc else rr[1].get(key, np.nan)) for rr in rows]
        vals = [np.nan if v is None else float(v) for v in vals]
        cols = ["#8c8c8c", "#c44e52", "#55a868"]
        b = ax.bar([r_[0] for r_ in rows], vals, color=cols)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v,
                    f"{v:.4f}" if v < 10 else f"{v:,}", ha="center", va="bottom", fontsize=7)
        if is_pc:
            ax.text(0.02, 0.95, "raw PC = 0.531 (E1)", transform=ax.transAxes, fontsize=7,
                    va="top", bbox=dict(fc="white", ec="0.7", alpha=0.9))
        ax.set_title(title); ax.tick_params(axis="x", labelrotation=10)
        finite = [v for v in vals if np.isfinite(v)]
        ax.set_ylim(0, (max(finite) * 1.18) if finite else 1)
        if key == "lcr_fixed":
            ax.axhline(raw["lcr_fixed"] - 0.002, color="k", ls=":", lw=1, label="gate")
            ax.legend(fontsize=7)
        if key == "ra":
            ax.axhline(raw["ra"] * 0.95, color="k", ls=":", lw=1, label="gate (raw×0.95)")
            ax.legend(fontsize=7)
    txt = "\n".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in r["gate"].items())
    fig.suptitle("F1: freezing layer/owner keeps PC & RA and removes the LCR regression\n" + txt,
                 fontsize=8.5)
    fig.tight_layout(); fig.savefig(figs / "fig_freeze_owner.png"); plt.close(fig)
    print("wrote", figs / "fig_freeze_owner.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
