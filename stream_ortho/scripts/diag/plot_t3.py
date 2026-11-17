"""Figures for T3 (``results/week10-sync/{seq}/figs``).

Reads ``metrics.json`` (E1) and ``rank_ladder_{source}.json`` and writes:

    fig_t3_overview.png     4-panel: method bars, DOF ladder, theta drift, graph health
    fig_t3_methods.png      HFR near/top per method with the 25%/40% gates
    fig_t3_dof_ladder.png   residual per rank rung, guard (IQR) annotated
    fig_t3_theta_drift.png  per-frame b_i for every fold (drift / anchor sanity)
    fig_t3_graph.png        edges used, cycle inconsistency, per-fold n_ok
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

METHOD_ORDER = ["raw", "abs_t3", "abs_s7", "field4x4", "sync_t3"]
COLOURS = {"raw": "#8c8c8c", "abs_t3": "#4c72b0", "abs_s7": "#64b5cd",
           "field4x4": "#dd8452", "sync_t3": "#c44e52"}


def _load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/week10-sync")
    ap.add_argument("--seq", default="ieu")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = pathlib.Path(a.root) / a.seq
    figs = root / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    m = _load(root / "metrics.json")
    if m is None:
        raise SystemExit(f"missing {root/'metrics.json'}")
    ladders = {s: _load(root / f"rank_ladder_{s}.json") for s in ("pointmap", "head")}

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    keys = [k for k in METHOD_ORDER if k in m["methods"]]

    def methods_bar(ax):
        x = np.arange(len(keys))
        near = [m["methods"][k].get("hfr_near", np.nan) for k in keys]
        top = [m["methods"][k].get("hfr_top", np.nan) for k in keys]
        ax.bar(x - 0.2, near, 0.4, label="HFR near", color=[COLOURS[k] for k in keys])
        ax.bar(x + 0.2, top, 0.4, label="HFR top", color=[COLOURS[k] for k in keys], alpha=0.55)
        best_abs = min((m["methods"][k].get("hfr_near", np.inf)
                        for k in keys if k.startswith("abs_")), default=np.nan)
        ax.axhline(best_abs, color="#4c72b0", ls="--", lw=1, label="best abs")
        ax.axhline(m["methods"]["raw"]["hfr_near"], color="#8c8c8c", ls=":", lw=1, label="raw")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15)
        ax.set_ylabel("HFR near (m)"); ax.set_title(f"E1 methods, {m['seq']}/{m['source']}")
        ax.legend(fontsize=7)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    methods_bar(ax)
    fig.tight_layout(); fig.savefig(figs / "fig_t3_methods.png"); plt.close(fig)

    # ---- DOF ladder ----
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    widths = 0.38
    for i, (src, c) in enumerate((("pointmap", "#4c72b0"), ("head", "#c44e52"))):
        lad = ladders.get(src)
        if not lad:
            continue
        rungs = [r for r in lad["ladder"] if "ridge" not in r["rung"]]
        y = [r["residual_m"] for r in rungs]
        x = np.arange(len(rungs)) + (i - 0.5) * widths
        bars = ax.bar(x, y, widths, color=c, label=src, alpha=0.9)
        for b, r in zip(bars, rungs):
            if r.get("guard_reason"):
                b.set_hatch("//"); b.set_alpha(0.5)
        ax.axhline(lad["raw_2obs_median_m"], color=c, ls=":", lw=1)
    ax.set_xticks(np.arange(len([r for r in ladders["pointmap"]["ladder"] if "ridge" not in r["rung"]])))
    ax.set_xticklabels([r["rung"] for r in ladders["pointmap"]["ladder"] if "ridge" not in r["rung"]],
                       rotation=15, fontsize=7)
    ax.set_ylabel("2-obs residual (m)")
    ax.set_title("guarded rank ladder (hatched = IQR guard fails)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "fig_t3_dof_ladder.png"); plt.close(fig)

    # ---- theta drift ----
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for d in m.get("drift", []):
        b = np.asarray(d.get("theta_b", []))
        ax.plot(np.arange(b.size), b, "-o", ms=2, lw=1, label=f"fold {d['fold']}")
    ax.set_xlabel("frame in fold's build set"); ax.set_ylabel("b_i (m)")
    ax.set_title("synchronised per-frame offset (gauge-free): drift check")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_t3_theta_drift.png"); plt.close(fig)

    # ---- graph health ----
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    cyc = m.get("cycle", [])
    folds = [d["fold"] for d in m.get("drift", [])]
    ax.bar(np.arange(len(cyc)) - 0.2, [c.get("cycle_median_m", np.nan) for c in cyc], 0.4,
           color="#c44e52", label="cycle median (m)")
    ax.bar(np.arange(len(cyc)) + 0.2, [c.get("cycle_p90_m", np.nan) for c in cyc], 0.4,
           color="#dd8452", alpha=0.8, label="cycle p90 (m)")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="0.5 m: model inconsistent")
    ax.set_xlabel("fold"); ax.set_ylabel("triangle inconsistency (m)")
    ax.set_title("cycle consistency of the pairwise model")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_t3_graph.png"); plt.close(fig)

    # ---- guards (review 11 §4) ----
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    gk = [k for k in keys if "z_iqr_ratio" in m["methods"][k]]
    iqr = [m["methods"][k]["z_iqr_ratio"] for k in gk]
    sh = [m["methods"][k].get("median_z_shift", np.nan) for k in gk]
    axes[0].bar(np.arange(len(gk)), iqr, color=[COLOURS[k] for k in gk])
    axes[0].axhspan(0.95, 1.05, color="g", alpha=0.15, label="allowed [0.95, 1.05]")
    axes[0].set_xticks(np.arange(len(gk))); axes[0].set_xticklabels(gk, rotation=15, fontsize=7)
    axes[0].set_ylabel("IQR(z_corr)/IQR(z_raw)"); axes[0].set_title("held-frame flattening guard")
    axes[0].legend(fontsize=7)
    axes[1].bar(np.arange(len(gk)), sh, color=[COLOURS[k] for k in gk])
    axes[1].axhline(0.5, color="r", ls="--", lw=1, label="0.5 m limit")
    axes[1].set_xticks(np.arange(len(gk))); axes[1].set_xticklabels(gk, rotation=15, fontsize=7)
    axes[1].set_ylabel("|median z shift| (m)"); axes[1].set_title("held-frame shift guard")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "fig_t3_guards.png"); plt.close(fig)

    # ---- overview ----
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    methods_bar(axes[0, 0])
    lad = ladders["pointmap"]
    rungs = [r for r in lad["ladder"] if "ridge" not in r["rung"]]
    axes[0, 1].bar(np.arange(len(rungs)), [r["residual_m"] for r in rungs], color="#4c72b0")
    axes[0, 1].set_xticks(np.arange(len(rungs)))
    axes[0, 1].set_xticklabels([r["rung"].split(" (")[0] for r in rungs], rotation=15, fontsize=7)
    axes[0, 1].set_ylabel("residual (m)"); axes[0, 1].set_title("pointmap rank ladder")
    for d in m.get("drift", []):
        b = np.asarray(d.get("theta_b", []))
        axes[1, 0].plot(np.arange(b.size), b, lw=1, label=f"f{d['fold']}")
    axes[1, 0].set_title("theta drift"); axes[1, 0].set_ylabel("b_i (m)"); axes[1, 0].legend(fontsize=6)
    axes[1, 1].bar(np.arange(len(cyc)), [c.get("cycle_median_m", np.nan) for c in cyc],
                   color="#c44e52")
    axes[1, 1].axhline(0.5, color="k", ls="--", lw=1)
    axes[1, 1].set_title("cycle inconsistency (m)"); axes[1, 1].set_xlabel("fold")
    g = m.get("gate", {})
    txt = (f"{m['seq']} / {m['source']}  cell {m['cell']}  {m['n_frames']} kf\n"
           f"gate passed: {g.get('passed')}\n"
           f"sync vs best abs: {g.get('sync_vs_best_abs_hfr_pct', float('nan')):+.1f}%\n"
           f"sync vs raw: {g.get('sync_vs_raw_hfr_pct', float('nan')):+.1f}%\n"
           + "\n".join(f"{k}: {v}" for k, v in g.get("checks", {}).items()))
    fig.text(0.012, 0.985, txt, va="top", fontsize=7.5,
             bbox=dict(fc="white", ec="0.6", alpha=0.9))
    fig.tight_layout(rect=(0.19, 0, 1, 1))
    fig.savefig(figs / "fig_t3_overview.png"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
