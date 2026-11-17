"""S2 stage: ATE of the MASt3R-SLAM baseline against NPU GT trajectories.

Reads the dumps produced by ``run_mast3r_baseline.py`` and writes, per run:

    metrics.json        ATE with/without scale, rotation error, path length
    traj_compare.png    top view + per-frame error

plus an aggregate ``results/stage-S2/ATE_TABLE.md``.

The GT trajectories are ``timestamp x y z qx qy qz qw`` (camera-to-world), the
same convention as MASt3R-SLAM's ``T_WC``.  Alignment is Sim3/Umeyama; the
scale-free ATE is the headline number, and the scale-fixed (SE3) ATE is
reported separately because the orthographic stage *does* care about metric
scale.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.eval.ate import (
    PoseTrajectory,
    associate_by_time,
    evaluate_ate,
    load_tum,
    save_tum,
    umeyama,
)


def build_parser():
    p = argparse.ArgumentParser(description="S2: ATE evaluation")
    p.add_argument("--runs", nargs="*", default=None, help="run dirs (default: all under stage-S2)")
    p.add_argument("--npu-root", default=str(zcfg.NPU_ROOT))
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "stage-S2"))
    p.add_argument("--max-dt", type=float, default=0.05)
    return p


def run_dirs(args) -> list[pathlib.Path]:
    if args.runs:
        return [pathlib.Path(r) for r in args.runs]
    out = pathlib.Path(args.out)
    return sorted([d for d in out.iterdir() if (d / "run.json").exists()])


def plot_run(est, ref, sim3, res, name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(ref.positions[:, 0], ref.positions[:, 1], "k-", lw=1.2, label="GT")
    aligned = sim3.apply(est.positions)
    ax1.plot(aligned[:, 0], aligned[:, 1], "r-", lw=0.8, alpha=0.85, label="est (Sim3 aligned)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title(f"{name}: top view (XY)")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(ref.positions[:, 0], ref.positions[:, 2], "k-", lw=1.0, label="GT")
    ax2.plot(aligned[:, 0], aligned[:, 2], "r-", lw=0.8, alpha=0.85, label="est")
    ax2.set_title("side view (XZ)")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("z [m]")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(np.arange(res.n_pairs), res.errors, ".-", ms=2, lw=0.6)
    ax3.set_title(f"ATE per frame  RMSE={res.rmse:.2f} m ({res.ate_percent:.2f}%)")
    ax3.set_xlabel("associated frame index")
    ax3.set_ylabel("position error [m]")
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_run(run_dir: pathlib.Path, npu_root: pathlib.Path, out_root: pathlib.Path, max_dt: float):
    info = json.loads((run_dir / "run.json").read_text())
    name = run_dir.name
    metrics = {"run": name, **{k: info.get(k) for k in (
        "dataset", "n_frames_processed", "n_keyframes", "fps", "use_calib",
        "wall_time_s", "peak_device_mem_mb", "track_ms", "track_ms_keyframes",
        "reloc_events", "sim3_scale", "upstream_commit", "device_name",
        "img_downsample", "stride", "frame_hw",
    )}}

    est = load_tum(run_dir / "poses_keyframes.txt")
    metrics["n_est_kf"] = len(est)

    seq_dir = npu_root / info["dataset"]
    gt_path = seq_dir / "trajectory.txt"
    if not gt_path.exists():
        nested = seq_dir / seq_dir.name / "trajectory.txt"
        gt_path = nested if nested.exists() else gt_path
    if not gt_path.exists():
        metrics["ate"] = None
        metrics["note"] = "no GT trajectory for this sequence"
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[S2-ATE] {name}: no GT -> skipped")
        return metrics

    ref = load_tum(gt_path)
    metrics["n_gt"] = len(ref)

    # scale-free (Sim3) headline
    res = evaluate_ate(est, ref, max_dt=max_dt, with_scale=True)
    i_est, i_ref = associate_by_time(est, ref, max_dt=max_dt)
    sim3 = umeyama(est.positions[i_est], ref.positions[i_ref], with_scale=True)
    metrics["ate"] = res.as_dict()
    metrics["ate"]["n_gt_path_length"] = round(ref.path_length, 2)

    # scale-fixed (SE3) -- shows whether the map is metric
    res_fixed = evaluate_ate(est, ref, max_dt=max_dt, with_scale=False)
    metrics["ate_fixed_scale"] = res_fixed.as_dict()

    (run_dir / "sim3_est_to_gt.json").write_text(json.dumps({
        "scale": float(sim3.scale),
        "rotation": np.asarray(sim3.rotation).tolist(),
        "translation": np.asarray(sim3.translation).tolist(),
        "n_pairs": int(res.n_pairs),
        "rmse": float(res.rmse),
        "note": "p_GT = scale * R @ p_est + t  (Umeyama on associated keyframes)",
    }, indent=2))

    save_tum(run_dir / "poses_aligned_sim3.txt", PoseTrajectory(
        est.timestamps, sim3.apply(est.positions), est.quaternions),
        header="estimated keyframe poses aligned to GT (Sim3, scale-free)")
    plot_run(est, ref, sim3, res, name, run_dir / "traj_compare.png")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[S2-ATE] {name:22s} kf={len(est):3d} pairs={res.n_pairs:3d} "
        f"ATE={res.rmse:7.2f} m ({res.ate_percent:5.2f}%)  "
        f"scale={sim3.scale:6.3f}  rot={res.rot_rmse_deg:5.2f} deg  "
        f"fixed-scale ATE={res_fixed.rmse:7.2f} m"
    )
    return metrics


def write_table(all_metrics, out_root: pathlib.Path):
    lines = [
        "# S2 ATE table (MASt3R-SLAM baseline)",
        "",
        "Sim3-aligned (scale-free) ATE is the headline; `ATE(SE3)` keeps the",
        "estimated scale, so it reveals metric-scale error. `ATE%` is relative to",
        "the GT path length over the associated keyframes.",
        "",
        "| run | seq | kf | pairs | ATE Sim3 [m] | ATE% | median | p95 | ATE SE3 [m] | est. scale | rot RMSE [deg] | fps | peak GPU [MB] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        if not m.get("ate"):
            lines.append(f"| `{m['run']}` | {m['dataset']} | {m['n_est_kf']} | — | — | — | — | — | — | — | — | {m.get('fps')} | {m.get('peak_device_mem_mb')} |")
            continue
        a = m["ate"]
        f = m["ate_fixed_scale"]
        lines.append(
            f"| `{m['run']}` | {m['dataset']} | {m['n_est_kf']} | {a['n_pairs']} | "
            f"{a['rmse']:.2f} | {a['ate_percent']:.2f}% | {a['median']:.2f} | {a['p95']:.2f} | "
            f"{f['rmse']:.2f} | {a['scale']:.3f} | {a['rot_rmse_deg']:.2f} | "
            f"{m.get('fps')} | {m.get('peak_device_mem_mb')} |"
        )
    lines.append("")
    (out_root / "ATE_TABLE.md").write_text("\n".join(lines))


def main(argv=None):
    args = build_parser().parse_args(argv)
    npu_root = pathlib.Path(args.npu_root)
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics = []
    for d in run_dirs(args):
        metrics.append(evaluate_run(d, npu_root, out_root, args.max_dt))
    (out_root / "ate_summary.json").write_text(json.dumps(metrics, indent=2))
    write_table(metrics, out_root)
    print(f"[S2-ATE] wrote {out_root/'ATE_TABLE.md'}")


if __name__ == "__main__":
    main()
