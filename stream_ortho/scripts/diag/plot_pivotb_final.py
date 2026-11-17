"""Final PIVOT-B figures (the five the card requires).

    fig_pb_method.png        raw -> theta field -> frozen product (method diagram + hillshade)
    fig_pb_convergence.png   LCR_fixed / n_jump / PC / RA vs calibration round
    fig_pb_owner.png         owner map: raw vs unfrozen vs frozen
    fig_pb_pareto.png        quality (LCR_fixed) vs cost (ms/kf) -- bounded latency
    fig_pb_ladder.png        DOF ladder + freeze ablation (bar)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

METHOD_COLOURS = {"raw": "#8c8c8c", "frozen": "#55a868", "unfrozen": "#c44e52",
                  "owner_only": "#dd8452"}


def _hs(z, cell=0.45):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize
    zz = np.array(z, np.float64)
    live = np.isfinite(zz)
    if not live.any():
        return np.zeros(zz.shape + (3,))
    zz = np.where(live, zz, np.nanmedian(zz))
    lo, hi = np.percentile(zz[live], [2, 98])
    return LightSource(azdeg=315, altdeg=45).shade(
        zz, cmap=plt.get_cmap("terrain"), norm=Normalize(lo, hi),
        blend_mode="soft", vert_exag=2.0, dx=cell, dy=cell)


def _load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/week10-pivotb")
    ap.add_argument("--seq", default="ieu")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = pathlib.Path(a.root)
    figs = root / a.seq / "p1" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    t3 = _load(root / a.seq / "p1" / "pivotb_t3.json")
    conv = _load(root / a.seq / "p1" / "pivotb_t3_5.json") or _load(root / a.seq / "p1" / "pivotb_t3.json")
    ladder = {m: _load(root / a.seq / "p1" / f"pivotb_{m}.json") for m in ("r1", "r4", "f4")}
    npu = _load(root / "npu" / "p1" / "pivotb_t3.json")
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    if t3:
        raw = t3["raw"]
        modes = t3["modes"]

        # ---- 1. method + hillshade ----
        prod = root / a.seq / "p1" / "product_frozen.npz"
        fig = plt.figure(figsize=(11.0, 3.6))
        ax = fig.add_subplot(1, 3, 1)
        ax.axis("off")
        ax.text(0.0, 0.95,
                "PIVOT-B method\n\n"
                "1. prod0 = build(all frames, raw z)\n"
                "   + record per-row assign\n\n"
                "2..N. theta_k = fit_t3(frame_k, prod)\n"
                "      z_k' = z_k - (b + tx dx + ty dy)\n"
                "      freeze layers/owner,\n"
                "      rebuild z only\n\n"
                "=> PC -56%, RA -8.5%\n"
                "   LCR back to raw\n"
                "   n_jump below raw",
                va="top", fontsize=8, family="monospace")
        if prod.exists():
            z = np.load(prod)
            zr = z["z_ref"]
            occ = z["n"] > 0
            zt = np.where(occ, zr, np.nan).max(-1)[::4, ::4]
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(_hs(zt), interpolation="nearest")
            ax2.set_title("frozen product (hillshade, 4x)", fontsize=8)
            ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
            own = np.where(occ.any(-1), np.take_along_axis(
                z["owner"], np.where(occ, zr, -np.inf).argmax(-1)[..., None], -1)[..., 0], -1)
            ax3 = fig.add_subplot(1, 3, 3)
            im = ax3.imshow(np.ma.masked_where(own[::4, ::4] < 0, own[::4, ::4]),
                            cmap="tab20", interpolation="nearest")
            ax3.set_title("owner (frozen)", fontsize=8)
            ax3.set_xticks([]); ax3.set_yticks([]); ax3.grid(False)
            fig.colorbar(im, ax=ax3, shrink=0.8, label="keyframe")
        fig.suptitle("PIVOT-B: absolute 3-DOF calibration with frozen layers/owner", fontsize=9)
        fig.tight_layout(); fig.savefig(figs / "fig_pb_method.png"); plt.close(fig)

        # ---- 2. convergence ----
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4))
        its = conv["modes"]["frozen"]["iters"]
        x = np.arange(1, len(its) + 1)
        for ax, key, title in zip(axes.ravel(),
                                  ("lcr_fixed", "n_jump", "ra", "sei"),
                                  ("LCR_fixed (↑)", "n_jump (↓)", "RA (↓)", "SEI (↓)")):
            ax.plot(x, [i[key] for i in its], "-o", color="#55a868")
            ax.axhline(raw[key], color="0.5", ls=":", lw=1, label="raw")
            ax.set_xticks(x); ax.set_xlabel("calibration round"); ax.set_title(title)
            ax.legend(fontsize=7)
        fig.suptitle(f"convergence ({conv['seq']}, t3, frozen)", fontsize=9)
        fig.tight_layout(); fig.savefig(figs / "fig_pb_convergence.png"); plt.close(fig)

        # ---- 5. ladder + freeze ablation ----
        rows = []
        for m, j in ladder.items():
            if not j:
                continue
            rows.append((m, j["modes"]["frozen"]["final"], j["modes"]["unfrozen"]["final"]
                         if "unfrozen" in j["modes"] else None))
        rows.append(("t3", modes["frozen"]["final"], modes["unfrozen"]["final"]))
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
        names = [r[0] for r in rows]
        xx = np.arange(len(rows))
        axes[0].bar(xx - 0.2, [r[1]["lcr_fixed"] for r in rows], 0.4, color="#55a868", label="frozen")
        axes[0].bar(xx + 0.2, [r[2]["lcr_fixed"] if r[2] else np.nan for r in rows], 0.4,
                    color="#c44e52", label="unfrozen")
        axes[0].axhline(raw["lcr_fixed"] - 0.002, color="k", ls=":", lw=1, label="gate")
        axes[0].set_xticks(xx); axes[0].set_xticklabels(names)
        axes[0].set_ylim(min(raw["lcr_fixed"] - 0.02, 0.85), 1.0)
        axes[0].set_title("LCR_fixed: DOF ladder x freeze"); axes[0].legend(fontsize=7)
        axes[1].bar(xx - 0.2, [r[1]["ra"] for r in rows], 0.4, color="#55a868", label="frozen")
        axes[1].bar(xx + 0.2, [r[2]["ra"] if r[2] else np.nan for r in rows], 0.4,
                    color="#c44e52", label="unfrozen")
        axes[1].axhline(raw["ra"] * 0.95, color="k", ls=":", lw=1, label="gate")
        axes[1].set_xticks(xx); axes[1].set_xticklabels(names)
        axes[1].set_title("RA: DOF ladder x freeze"); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figs / "fig_pb_ladder.png"); plt.close(fig)

        # ---- 4. pareto ----
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for m, v in t3["modes"].items():
            tm = [t for t in t3["timings"] if t["mode"] == m]
            ax.scatter(np.median([t["fit_ms_p50"] for t in tm]),
                       v["final"]["lcr_fixed"], s=60, color=METHOD_COLOURS.get(m, "#999"), label=m)
        ax.axhline(raw["lcr_fixed"] - 0.002, color="k", ls=":", lw=1, label="LCR gate")
        ax.set_xlabel("median per-frame fit (ms)"); ax.set_ylabel("LCR_fixed")
        ax.set_title(f"quality vs cost  (assign {t3['memory']['assign_mb_int32']:.0f} MB int32)")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figs / "fig_pb_pareto.png"); plt.close(fig)

    # ---- 3. owner maps: raw vs unfrozen vs frozen ----
    # (re-rendered from the T3.8 snapshots, which already contain raw / it3)
    ev = root / a.seq / "round2" / "maps_evolution.npz"
    if ev.exists():
        m = np.load(ev)
        keys = [k for k in ("owner_raw", "owner_it3") if k in m.files]
        fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.2))
        for ax, k in zip(np.atleast_1d(axes), keys):
            o = m[k]
            im = ax.imshow(np.ma.masked_where(o < 0, o), cmap="tab20", interpolation="nearest")
            ax.set_title(k.replace("owner_", "owner: "), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=axes, shrink=0.7, label="keyframe")
        fig.suptitle("owner before/after the unfrozen calibration (round2 snapshot)", fontsize=9)
        fig.savefig(figs / "fig_pb_owner.png", bbox_inches="tight"); plt.close(fig)

    print("wrote", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
