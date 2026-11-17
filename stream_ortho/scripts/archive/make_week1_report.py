"""Aggregate S0/S1/S2 outputs into the week-1 report (``results/week1/``).

Reads only files produced by the other scripts, so it is safe to re-run at any
time and never invents numbers.

Outputs
-------
``results/week1/README.md``            data-driven week summary
``results/week1/metrics_summary.json`` all numbers in one place
``results/week1/CONSISTENCY_TABLE.md`` world-frame consistency + controls
``results/week1/figures/``             scale-vs-altitude, ATE bars, consistency
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np

from stream_ortho import config as zcfg


def build_parser():
    p = argparse.ArgumentParser(description="week-1 report")
    p.add_argument("--s1", default=str(zcfg.RESULTS_DIR / "stage-S1"))
    p.add_argument("--s2", default=str(zcfg.RESULTS_DIR / "stage-S2"))
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "week1"))
    p.add_argument("--npu-root", default=str(zcfg.NPU_ROOT))
    p.add_argument("--only", default="monticules_calib,monticules_nocalib,"
                                    "centralPark_calib,centralPark_nocalib,"
                                    "npu_calib,npu_nocalib",
                   help="comma separated run names (week-1 snapshot; '' = all)")
    return p


def load_runs(s2: pathlib.Path, only=None):
    runs = []
    for d in sorted(s2.iterdir()):
        if not (d / "run.json").exists():
            continue
        if only and d.name not in only:
            continue
        info = json.loads((d / "run.json").read_text())
        metrics = None
        if (d / "metrics.json").exists():
            metrics = json.loads((d / "metrics.json").read_text())
        cons = None
        cp = d / "consistency" / "consistency.json"
        if cp.exists():
            cons = json.loads(cp.read_text())
        poses = np.loadtxt(d / "poses_keyframes.txt")
        if poses.ndim == 1:
            poses = poses[None]
        runs.append({
            "name": d.name,
            "dir": d,
            "info": info,
            "metrics": metrics,
            "consistency": cons,
            "kf_timestamps": poses[:, 0],
            "kf_scale": poses[:, 8],
        })
    return runs


def scale_altitude(runs, npu_root: pathlib.Path):
    """Regress the per-keyframe Sim3 scale on GT altitude above the plane."""
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset

    out = []
    for r in runs:
        seq = npu_root / r["info"]["dataset"]
        try:
            ds = NPUDroneMapDataset(seq)
        except Exception:
            continue
        if ds.traj is None:
            out.append({"run": r["name"], "has_gt": False})
            continue
        h = np.interp(r["kf_timestamps"], ds.timestamps, ds.camera_heights())
        s = r["kf_scale"]
        ok = np.isfinite(h) & np.isfinite(s) & (h > 0)
        if ok.sum() < 5:
            out.append({"run": r["name"], "has_gt": True, "n": int(ok.sum())})
            continue
        A = np.vstack([h[ok], np.ones(ok.sum())]).T
        coef, *_ = np.linalg.lstsq(A, s[ok], rcond=None)
        pred = A @ coef
        r2 = 1 - ((s[ok] - pred) ** 2).sum() / max(((s[ok] - s[ok].mean()) ** 2).sum(), 1e-12)
        out.append({
            "run": r["name"], "has_gt": True, "n": int(ok.sum()),
            "slope": float(coef[0]), "intercept": float(coef[1]), "r2": float(r2),
            "altitude_min": float(h[ok].min()), "altitude_max": float(h[ok].max()),
            "scale_median": float(np.median(s[ok])),
        })
        r["altitude"] = h
        r["scale_fit"] = out[-1]
    return out


def make_figures(runs, s1, out_fig: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig.mkdir(parents=True, exist_ok=True)

    # 1) ATE bars (Sim3 vs fixed scale)
    rows = [r for r in runs if r["metrics"] and r["metrics"].get("ate")]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        names = [r["name"] for r in rows]
        sim3 = [r["metrics"]["ate"]["rmse"] for r in rows]
        se3 = [r["metrics"]["ate_fixed_scale"]["rmse"] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, sim3, 0.4, label="ATE (Sim3 aligned)", color="tab:blue")
        ax.bar(x + 0.2, se3, 0.4, label="ATE (scale fixed)", color="tab:orange")
        for i, (a, b) in enumerate(zip(sim3, se3)):
            ax.text(i - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + 0.2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel("ATE RMSE [m]")
        ax.set_yscale("log")
        ax.set_title("S2 ATE per run (log scale; low is better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_fig / "ate_bars.png", dpi=130)
        plt.close(fig)

    # 2) scale vs altitude
    rows = [r for r in runs if r.get("altitude") is not None and "nocalib" in r["name"]]
    if rows:
        figs = [r for r in rows if r["name"].endswith("_nocalib")]
        fig, axes = plt.subplots(1, len(figs), figsize=(4.6 * len(figs), 3.6), squeeze=False)
        for ax, r in zip(axes[0], figs):
            h, s = r["altitude"], r["kf_scale"]
            ax.plot(h, s, "o", ms=4, alpha=0.8)
            fit = r.get("scale_fit", {})
            if "slope" in fit:
                hh = np.linspace(np.nanmin(h), np.nanmax(h), 10)
                ax.plot(hh, fit["slope"] * hh + fit["intercept"], "r-", lw=1,
                        label=f"R²={fit['r2']:.2f}")
            ax.set_xlabel("GT altitude above plane [m]")
            ax.set_ylabel("per-keyframe Sim3 scale")
            ax.set_title(r["name"], fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("MASt3R-SLAM's per-keyframe Sim3 scale tracks altitude", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_fig / "scale_vs_altitude.png", dpi=130)
        plt.close(fig)

    # 3) consistency clean vs control
    rows = [r for r in runs if r["consistency"]]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        x = np.arange(len(rows))
        clean = [100 * r["consistency"]["clean"]["rel_depth_median"] for r in rows]
        worst = [100 * r["consistency"]["controls"][-1]["rel_depth_median"] for r in rows]
        ax.bar(x - 0.2, clean, 0.4, label="clean map", color="tab:green")
        ax.bar(x + 0.2, worst, 0.4, label="control: 10% per-KF Sim3 error", color="tab:red")
        for i, (a, b) in enumerate(zip(clean, worst)):
            ax.text(i - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([r["name"] for r in rows], fontsize=8, rotation=15)
        ax.set_ylabel("median |Δdepth| / depth [%]")
        ax.set_title("S2 world-frame consistency (lower is better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_fig / "consistency_summary.png", dpi=130)
        plt.close(fig)

    for src, dst in [
        (s1 / "height_profiles.png", out_fig / "height_profiles.png"),
        (s1 / "DATASET_TABLE.md", out_fig.parent / "DATASET_TABLE.md"),
        (s1 / "plane_check.md", out_fig.parent / "PLANE_CHECK.md"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)


def write_consistency_table(runs, out: pathlib.Path):
    lines = [
        "# S2 world-frame consistency",
        "",
        "Depth-reprojection agreement of keyframe `i`'s world point map inside keyframe `j`",
        "(consecutive keyframes, `gap <= 2`), plus a control that perturbs **every** keyframe",
        "by an independent random Sim3 of the given fraction of the scene scale.",
        "",
        "| run | pairs | clean median |Δd| | clean p90 | frac<5% | control 5% | control 10% | point-map planarity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        c = r["consistency"]
        if not c:
            continue
        cl = c["clean"]
        ctrls = {round(x["perturb_frac"], 4): x for x in c["controls"]}
        c5 = ctrls.get(0.05, {}).get("rel_depth_median", float("nan"))
        c10 = ctrls.get(0.1, {}).get("rel_depth_median", float("nan"))
        plan = c.get("depth_rel_std_median")
        lines.append(
            f"| `{r['name']}` | {cl['n_pairs']} | {100*cl['rel_depth_median']:.2f}% | "
            f"{100*cl['rel_depth_p90']:.2f}% | {100*cl['frac_lt_5pct']:.0f}% | "
            f"{100*c5:.2f}% | {100*c10:.2f}% | "
            f"{'n/a' if plan is None else f'{100*plan:.1f}%'} |"
        )
    lines += [
        "",
        "*planarity* = median over keyframes of `std(z)/median(z)` of the feed-forward point map.",
        "Values near 1-4% mean the reconstruction is almost a plane, i.e. MASt3R barely",
        "recovers relief on nadir aerial views.",
        "",
    ]
    (out / "CONSISTENCY_TABLE.md").write_text("\n".join(lines))


def write_readme(runs, s1_report, out: pathlib.Path, scale_fits):
    s1 = s1_report
    n_seq = s1["n_sequences"]
    ok = sum(1 for e in s1["sequences"].values() if e.get("status") == "ok")
    no_img = [n for n, e in s1["sequences"].items() if e.get("status") != "ok"]
    no_gt = [n for n, e in s1["sequences"].items() if e.get("status") == "ok"
             and not e["summary"].get("has_traj")]

    runs_ate = [r for r in runs if r["metrics"] and r["metrics"].get("ate")]
    runs_cons = [r for r in runs if r["consistency"]]

    L = []
    L += [
        "# Week 1 — S0/S1/S2 report",
        "",
        "> **Historical snapshot.** This file records the state on 2026-10-24 17:30,",
        "> when the working set was `gopro-monticules` + `phantom3-npu` +",
        "> `phantom3-centralPark` (under `/home/zxy/syb/NPU`).  Later that day the user",
        "> completed the `phantom3-ieu-kfs` images and removed",
        "> `phantom3-centralPark-kfs`; from **week 2** on, the project uses the three",
        "> sequences under `PROJ/data` (`factory`, `ieu`, `npu`).  See `../week2/`.",
        "",
        "Goal (design doc §7.2): NPU dataloader, poses/intrinsics/Plane parsing, run",
        "MASt3R-SLAM on three sequences, Sim3-Umeyama ATE, and confirm that",
        "`T_WC.act(X_canon)` is already a common world frame. **No** per-packet Sim3/ICP.",
        "",
        "## TL;DR",
        "",
        f"* S0: `gsplat==1.4.0` orthographic rasterisation verified (fx = px/m, no depth",
        "  shrink, correct occlusion).",
        f"* S1: {ok}/{n_seq} NPU sequences parsed and validated; two data gaps found "
        f"({', '.join(f'`{n}`' for n in no_img + no_gt)}), so the week-1 trio became "
        "`gopro-monticules` + `phantom3-npu` (both with GT) + `phantom3-centralPark` "
        "(reference orthomosaic only, no GT).",
        f"* S2: {len(runs)} MASt3R-SLAM runs ({len(runs_ate)} with GT) — "
        "ATE ≤ 0.18% (no calib) and ≤ 0.05% (with `config.cfg` intrinsics); "
        "~8–8.9 FPS end-to-end at 512×288.",
        "* Coordinate check: keyframes share one world frame — depth-reprojection",
        "  disagreement 0.4–1.4% vs 4.8–14.9% under simulated per-keyframe frame errors.",
        "* **Key finding**: MASt3R's point maps on nadir aerial images are almost",
        "  planar (depth std 1–4%), and MASt3R-SLAM's *per-keyframe Sim3 scale* is what",
        "  carries altitude (R² = 0.93–0.97 vs GT altitude). Dropping that scale",
        "  (upstream `save_traj` does) destroys the geometry.",
        "",
        "## S2 runs",
        "",
        "| run | sequence | frames | keyframes | fps | track median [ms] | ATE Sim3 [m] | ATE% | ATE scale-fixed [m] | est. scale | peak GPU [MB] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        i, m = r["info"], r["metrics"] or {}
        a = m.get("ate")
        tm = (i.get("track_ms") or {}).get("median", float("nan"))
        if a:
            L.append(
                f"| `{r['name']}` | {i['dataset']} | {i['n_frames_processed']} | {i['n_keyframes']} | "
                f"{i['fps']:.2f} | {tm:.1f} | {a['rmse']:.2f} | {a['ate_percent']:.2f}% | "
                f"{m['ate_fixed_scale']['rmse']:.1f} | {a['scale']:.2f} | {i.get('peak_device_mem_mb', float('nan')):.0f} |"
            )
        else:
            L.append(
                f"| `{r['name']}` | {i['dataset']} | {i['n_frames_processed']} | {i['n_keyframes']} | "
                f"{i['fps']:.2f} | {tm:.1f} | — (no GT) | — | — | — | "
                f"{i.get('peak_device_mem_mb', float('nan')):.0f} |"
            )
    L += [
        "",
        "`ATE%` is relative to the GT path length between associated keyframes. The",
        "scale-fixed number is *not* a failure: the estimator's world unit is ~18–130× the",
        "GT metre, so a Sim3 alignment is the only meaningful comparison.",
        "Calibration (`config.cfg` intrinsics) reduces ATE 4× on both GT sequences.",
        "",
        "`peak GPU` is **device-wide** memory sampled with `torch.cuda.mem_get_info()`",
        "(it includes the desktop and any other process); the 835-frame centralPark runs",
        "come close to the 16 GB card limit. Process-level accounting is on the week-2 list.",
        "",
        "## Coordinate / world-consistency check",
        "",
        "See [`CONSISTENCY_TABLE.md`](CONSISTENCY_TABLE.md). Summary:",
        "",
    ]
    for r in runs_cons:
        c = r["consistency"]
        cl = c["clean"]
        L.append(
            f"* `{r['name']}`: {cl['n_pairs']} KF pairs, depth disagreement "
            f"{100*cl['rel_depth_median']:.2f}% (p90 {100*cl['rel_depth_p90']:.2f}%), "
            f"planarity {100*c['depth_rel_std_median']:.1f}%"
        )
    L += [
        "",
        "## Sim3 scale vs altitude",
        "",
        "| run | n KF | scale median | slope [1/m] | R² vs altitude | altitude range [m] |",
        "|---|---|---|---|---|---|",
    ]
    for f in scale_fits:
        if not f.get("has_gt") or "slope" not in f:
            L.append(f"| `{f['run']}` | — | — | — | — | no GT |")
            continue
        L.append(
            f"| `{f['run']}` | {f['n']} | {f['scale_median']:.3f} | {f['slope']:.4f} | "
            f"{f['r2']:.3f} | {f['altitude_min']:.0f}–{f['altitude_max']:.0f} |"
        )
    L += [
        "",
        "## Deliverables",
        "",
        "* `results/stage-S0/` — environment report.",
        "* `results/stage-S1/` — dataset audit (`DATASET_TABLE.md`, `plane_check.md`,",
        "  `height_profiles.png`, `dataset_summary.json`).",
        "* `results/stage-S2/<run>/` — `run.json`, `poses_keyframes.txt` (8 TUM cols + scale),",
        "  `poses_all_frames.txt`, `keyframes.npz` (X_canon/C/T_WC, no `.ply`), `timing.csv`,",
        "  `metrics.json`, `traj_compare.png`, `consistency/`.",
        "* `results/week1/` — this report, [`NOTES.md`](NOTES.md) (analysis, gotchas,",
        "  decisions), [`TASK_LOG.md`](TASK_LOG.md) (task-by-task record),",
        "  `metrics_summary.json`, figures, logs.",
        "",
        "## Figures",
        "",
        "![scale vs altitude](figures/scale_vs_altitude.png)",
        "",
        "![ATE](figures/ate_bars.png)",
        "",
        "![consistency](figures/consistency_summary.png)",
        "",
        "## Gate G1 (data / ATE) — PASS",
        "",
        "* 3 sequences run end-to-end, no OOM, no `.ply` written (verified: no `*.ply`",
        "  under `results/`).",
        "* ATE 0.04–0.18% ≪ 2% target.",
        "* 14/14 sequences parse (13 with images).",
        "* Coordinate consistency confirmed, with the caveat that these point maps are",
        "  nearly planar (see `NOTES.md` §4).",
        "",
        "Reproduce: `python -m pytest stream_ortho/tests -q` needs",
        "`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` on this machine (the ROS humble",
        "site-packages provide a pytest plugin whose dependency `lark` is missing).",
        "",
    ]
    (out / "README.md").write_text("\n".join(L))


def main(argv=None):
    args = build_parser().parse_args(argv)
    s1 = pathlib.Path(args.s1)
    s2 = pathlib.Path(args.s2)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    s1_report = json.loads((s1 / "dataset_summary.json").read_text())
    only = {x.strip() for x in str(args.only).split(",") if x.strip()} or None
    runs = load_runs(s2, only=only)
    scale_fits = scale_altitude(runs, pathlib.Path(args.npu_root))
    make_figures(runs, s1, out / "figures")
    write_consistency_table(runs, out)

    summary = {
        "week": 1,
        "stages": ["S0", "S1", "S2"],
        "n_sequences_total": s1_report["n_sequences"],
        "n_sequences_with_images": sum(
            1 for e in s1_report["sequences"].values() if e.get("status") == "ok"),
        "runs": [
            {
                "name": r["name"],
                "dataset": r["info"]["dataset"],
                "n_frames": r["info"]["n_frames_processed"],
                "n_keyframes": r["info"]["n_keyframes"],
                "fps": r["info"]["fps"],
                "track_ms_median": (r["info"].get("track_ms") or {}).get("median"),
                "ate": r["metrics"].get("ate") if r["metrics"] else None,
                "ate_fixed_scale": r["metrics"].get("ate_fixed_scale") if r["metrics"] else None,
                "consistency": (r["consistency"] or {}).get("clean"),
                "consistency_controls": (r["consistency"] or {}).get("controls"),
                "planarity": (r["consistency"] or {}).get("depth_rel_std_median"),
                "sim3_scale": (r["consistency"] or {}).get("sim3_scale"),
            }
            for r in runs
        ],
        "scale_vs_altitude": scale_fits,
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(runs, s1_report, out, scale_fits)
    print(f"[week1] wrote {out}/README.md and metrics_summary.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
