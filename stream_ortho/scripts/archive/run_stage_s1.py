"""S1 stage: audit all NPU sequences (data + coordinates), write JSON/Markdown/figures.

Outputs (``results/stage-S1/``):
    dataset_summary.json   per-sequence parsed facts
    DATASET_TABLE.md        human-readable table (also used by the week-1 report)
    plane_check.md          plane/height diagnostics
    height_profiles.png     camera height above the plane, per sequence
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import (
    NPUDroneMapDataset,
    discover_sequences,
)
from stream_ortho.data.plane import (
    camera_heights,
    parse_config_cfg,
    parse_trajectory,
    plane_normal_W,
    world_to_plane,
)


def build_parser():
    p = argparse.ArgumentParser(description="S1: NPU dataset + plane audit")
    p.add_argument("--root", default=str(zcfg.NPU_ROOT))
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "stage-S1"))
    p.add_argument("--no-figures", action="store_true")
    return p


def audit(root: pathlib.Path) -> dict:
    seqs = discover_sequences(root)
    out = {"root": str(root), "n_sequences": len(seqs), "sequences": {}}
    for seq in seqs:
        entry = {
            "dir": str(seq),
            "config": parse_config_cfg(seq / "config.cfg").summary(),
            "n_images_on_disk": len(list((seq / "rgb").glob("*.jpg"))) if (seq / "rgb").is_dir() else 0,
        }
        try:
            ds = NPUDroneMapDataset(seq)
        except FileNotFoundError as exc:
            entry["status"] = "no_images"
            entry["error"] = str(exc)
            out["sequences"][seq.name] = entry
            continue
        ds = NPUDroneMapDataset(seq, use_calib=False)
        entry["status"] = "ok"
        entry["summary"] = ds.summary()
        out["sequences"][seq.name] = entry
    return out


def write_markdown(report: dict, out_dir: pathlib.Path):
    rows = []
    for name, e in report["sequences"].items():
        if e["status"] != "ok":
            rows.append(f"| `{name}` | {e['n_images_on_disk']} | — | — | — | — | **NO IMAGES** |")
            continue
        s = e["summary"]
        h = s["height"]
        hstr = (
            f"{h['min']:.0f} / {h['median']:.0f} / {h['max']:.0f}"
            if h.get("n", 0)
            else "n/a"
        )
        cov = s["gt_coverage"]
        rows.append(
            f"| `{name}` | {s['n_frames']} | {cov['n_matched']}/{cov['n_images']} "
            f"({100*cov['ratio']:.0f}%) | {s['gt_path_length']:.0f} | {hstr} | "
            f"{'flat' if s['layout']=='single' else 'nested'} | ok |"
        )
    md = "\n".join(
        [
            "# S1 dataset audit",
            "",
            f"root: `{report['root']}`   sequences: {report['n_sequences']}",
            "",
            "| sequence | images | GT matched | GT path (m) | height min/med/max (m) | layout | status |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
            "Notes:",
            "- `height` is measured along the `Plane` frame's +Z. The plane *translation*",
            "  is not guaranteed to lie on the ground (see `plane_check.md`).",
            "- `phantom3-centralPark-kfs` ships without `trajectory.txt`; only the",
            "  reference orthomosaic `result.png` is available.",
            "- `phantom3-ieu-kfs` ships `config.cfg` + 467 GT poses but **no** RGB images.",
            "",
        ]
    )
    (out_dir / "DATASET_TABLE.md").write_text(md)


def write_plane_md(report: dict, out_dir: pathlib.Path):
    lines = [
        "# Plane / height diagnostics",
        "",
        "`Plane` = `x y z qx qy qz qw` (scalar-last). `T_WP` maps plane -> world.",
        "A *horizontal* plane has `|n_W . z_W| > 0.99`.",
        "Negative or near-zero median height means the plane origin is above/at the",
        "camera, i.e. the plane is not a ground plane for that sequence.",
        "",
        "| sequence | n_W . z_W | median h (m) | p5..p95 spread (m) | relative spread | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, e in report["sequences"].items():
        if e["status"] != "ok":
            continue
        h = e["summary"]["height"]
        if h.get("n", 0):
            verdict = "ok" if h["median_positive"] and h["relative_spread"] < 0.5 else "plane offset"
            lines.append(
                f"| `{name}` | {h['plane_normal_dot_world_z']:+.4f} | {h['median']:.1f} | "
                f"{h['spread_p5_p95']:.1f} | {h['relative_spread']:.2f} | {verdict} |"
            )
        else:
            lines.append(f"| `{name}` | {h['plane_normal_dot_world_z']:+.4f} | n/a | n/a | n/a | no GT |")
    lines += [
        "",
        "Consequence for later stages: never assume `plane z = 0` is the ground.",
        "Estimate the ground level from the point-map z-quantile in the plane frame.",
        "",
    ]
    (out_dir / "plane_check.md").write_text("\n".join(lines))


def make_figure(report: dict, root: pathlib.Path, out_dir: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = [n for n, e in report["sequences"].items() if e["status"] == "ok"]
    n = len(seqs)
    if n == 0:
        return
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), seqs):
        ds = NPUDroneMapDataset(root / name)
        if ds.traj is None:
            ax.text(0.5, 0.5, "no trajectory", ha="center", va="center")
            ax.set_axis_off()
            continue
        h = camera_heights(ds.T_WP, ds.gt_positions)
        ax.plot(ds.timestamps - ds.timestamps[0], h, lw=0.8, color="tab:blue")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(name.replace("-kfs", ""), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("h above plane (m)", fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle("S1: camera height above the Plane frame (GT trajectory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "height_profiles.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = pathlib.Path(args.root)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = audit(root)
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    write_markdown(report, out_dir)
    write_plane_md(report, out_dir)
    if not args.no_figures:
        make_figure(report, root, out_dir)
    ok = sum(1 for e in report["sequences"].values() if e["status"] == "ok")
    print(f"[S1] {ok}/{report['n_sequences']} sequences ok -> {out_dir}")
    print(f"[S1] wrote dataset_summary.json / DATASET_TABLE.md / plane_check.md")


if __name__ == "__main__":
    main()
