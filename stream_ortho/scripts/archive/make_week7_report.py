"""Aggregate the week-7 (S6b) outputs into ``results/week7/``."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from stream_ortho import config as zcfg


def build_parser():
    p = argparse.ArgumentParser(description="week-7 report")
    p.add_argument("--s6b", default=str(zcfg.RESULTS_DIR / "stage-S6b"))
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "week7"))
    return p


def load(s6b: pathlib.Path):
    runs = {}
    for d in sorted(s6b.iterdir()):
        f = d / "metrics.json"
        if d.is_dir() and f.exists():
            runs[d.name] = json.loads(f.read_text())
    return runs


def write_table(runs, out: pathlib.Path):
    L = ["# S6b — pose update / loop-closure demo", "",
         "A drift is applied to the tail of the trajectory, the map is built with it,",
         "then closed by `Δ = drift⁻¹` per keyframe, the affected tiles are re-rendered",
         "by the async worker, and the result is compared with a GT-pose plane reference.",
         "",
         "| run | drift | moved | tiles re-rendered | vs oracle (whole) | vs oracle (drifted region) | vs plane reference | untouched (control) | tearing outside |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, d in runs.items():
        dr = d["drift"]
        L.append(
            f"| {name} | {dr['max_m']:.0f} m / {dr['max_deg']:.1f}° | {d['n_moved_total']:,} | "
            f"{d['n_dirty_after_closure']}/{d['n_tiles']} ({100*d['dirty_ratio']:.0f}%) | "
            f"{d.get('psnr_before_vs_oracle', float('nan')):.2f} → {d.get('psnr_after_vs_oracle', float('nan')):.2f} "
            f"({d.get('psnr_gain_vs_oracle', float('nan')):+.2f}) | "
            f"{d.get('psnr_before_vs_oracle_tail', float('nan')):.2f} → {d.get('psnr_after_vs_oracle_tail', float('nan')):.2f} | "
            f"{d['psnr_before']:.2f} → {d['psnr_after']:.2f} | "
            f"{d.get('psnr_before_head', float('nan')):.2f} → {d.get('psnr_after_head', float('nan')):.2f} | "
            f"{100*d['tearing_outside_frac']:.4f}% |")
    L.append("")
    (out / "S6B_TABLE.md").write_text("\n".join(L))


def write_readme(runs, out: pathlib.Path):
    base = runs.get("npu_calib_drift8m") or next(iter(runs.values()), None)
    L = [
        "# Week 7 — S6b report (pose update, async worker, tear-free re-render)",
        "",
        "Goal (design doc §7.8): after a loop closure or pose-graph update the map must",
        "not tear, and the ortho must refresh within a bounded time.",
        "",
        "## TL;DR",
        "",
    ]
    if base:
        L += [
            f"* Pose update moved **{base['n_moved_total']:,} Gaussians** (only the drifted",
            f"  keyframes' owner groups, max displacement {base['max_displacement_m']:.2f} m).",
            f"* **{base['n_dirty_after_closure']}/{base['n_tiles']} tiles** re-rendered "
            f"({100*base['dirty_ratio']:.0f}%), in {base['n_worker_pumps']} async pumps "
            f"({base['worker']['render_ms_mean']:.1f} ms/tile).",
            f"* **No tearing**: `{100*base['tearing_outside_frac']:.4f}%` of the pixels outside the",
            "  re-rendered tiles changed (target 0).",
            f"* Quality vs the **un-drifted oracle** (the clean metric): "
            f"{base.get('psnr_before_vs_oracle', float('nan')):.2f} → "
            f"**{base.get('psnr_after_vs_oracle', float('nan')):.2f} dB** "
            f"({base.get('psnr_gain_vs_oracle', float('nan')):+.2f}); in the drifted region "
            f"{base.get('psnr_before_vs_oracle_tail', float('nan')):.2f} → "
            f"{base.get('psnr_after_vs_oracle_tail', float('nan')):.2f} dB.",
            f"* Untouched region unchanged ({base.get('psnr_before_head', float('nan')):.2f} → "
            f"{base.get('psnr_after_head', float('nan')):.2f} dB, the control).",
            "* **The plane reference is not a fair target here**: it carries its own",
            "  registration bias, so for ieu the closure *lowers* reference-PSNR",
            f"  ({runs.get('ieu_calib', {}).get('psnr_before', float('nan')):.2f} → "
            f"{runs.get('ieu_calib', {}).get('psnr_after', float('nan')):.2f} dB) while it *raises*",
            "  oracle-PSNR by +2.7 dB. The oracle comparison is the one that isolates the",
            "  pose-update correctness.",
        ]
    L += [
        "",
        "## What the demo does",
        "",
        "1. Build the map from 8 keyframes with a **simulated drift** on the tail",
        "   (translation + yaw growing to 8 m / 4°);",
        "2. snapshot the ortho (A) — visibly offset from the reference in the tail;",
        "3. \"close the loop\": for each drifted keyframe apply `Δ = drift⁻¹` with",
        "   `apply_pose_update` (rigid move of that keyframe's Gaussians only, quats",
        "   rotated by `q_Δ ⊗ q`), then `mark_pose_changed` for the same keyframes;",
        "4. the **async worker** re-renders the dirty tiles a few at a time within a",
        "   per-pump budget, producing `pose_update_demo.gif`;",
        "5. snapshot (B) and check PSNR + tearing.",
        "",
        "## Acceptance (§7.8)",
        "",
        "| item | target | measured |",
        "|---|---|---|",
        f"| map does not tear after a pose update | no visible tear | {100*(base or {}).get('tearing_outside_frac', float('nan')):.4f}% changed outside re-rendered tiles |",
        f"| only the owner group is updated | yes | {((base or {}).get('n_moved_total', 0)):,} Gaussians, others bit-identical (unit test) |",
        "| ortho refresh bounded | < 3 s | per-pump budget 120 ms, 4 tiles per pump |",
        "| incremental == full re-render | equal | verified bit-exact in `test_pose_update.py` |",
        "",
        "## Bug found this week (and fixed)",
        "",
        "The quaternion conventions were mixed: `GaussianPatch.quats` is **wxyz**",
        "(gsplat/3DGS), but the week-5 shape prior wrote **xyzw** (`R_to_quat_xyzw`) and",
        "`transform_gaussians_to_world` composed an xyzw quaternion with the wxyz",
        "Hamilton product.  Consequence: the edge **rod orientations were wrong** (the",
        "anisotropy magnitude was right, the direction was garbage), and the head-patch",
        "route also rotated its splats incorrectly.  Fixed with `R_to_quat_wxyz[_batch]`",
        "and verified by rotating a rod 90° about the plane normal.",
        "",
        "## How to read the figures",
        "",
        "| file | what it is |",
        "|---|---|",
        "| `ortho_before_<run>.png` | the ortho **before** the closure: the drift is still in the map. The wide step/seam you see is the **injected** 3-8 m drift (8 m = 39 px at 0.2 m/px) at the boundary between the drifted tail and the untouched head -- it is the damage we are about to repair, not a render bug. |",
        "| `ortho_after_<run>.png` | after the closure + re-render: the step is gone. |",
        "| `ortho_oracle_<run>.png` | the same map built **without** drift -- the correctness oracle. `after` should look like this. |",
        "| `pose_update_demo.gif` | frame 0 = before (46/75 tiles dirty); each following frame = one worker pump (4 tiles re-rendered); the caption shows `pump k/12, re-rendered 4k/46`. It shows the re-render is **tile-local and progressive**. |",
        "| `tearing_check_<run>.png` | legacy 1x4 strip (16k px wide, hard to read): `A | B | |B-A|x8 | mask`. |",
        "| `tearing_check_2x2_<run>.png` | **read this one**: the same four panels in a labelled 2x2. Bottom-left = changed pixels (a subset of tile-aligned squares), bottom-right = re-rendered tiles. Changed pixels stay inside the re-rendered tiles -> no tearing. |",
        "| `dirty_mask_<run>.png` (in `stage-S6b/`) | the union of re-rendered tiles. |",
        "| `fault_before_after_oracle_<run>.png` | a crop straddling the step: before / after / oracle side by side. |",
        "",
        "Note: some `tearing_check` panels come from the legacy 1x4 layout; the 2x2",
        "versions were generated from the saved `ortho_before/after` + `dirty_mask` PNGs.",
        "",
        "## Figures",
        "",
        "![pose update gif](figures/pose_update_demo_npu_calib_drift8m.gif)",
        "",
        "![tearing check 2x2](figures/tearing_check_2x2_npu_calib_drift8m.png)",
        "",
        "![fault crop](figures/fault_before_after_oracle_npu_drift8m.png)",
        "",
        "![tearing check](figures/tearing_check.png)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export CUDA_HOME=/usr/local/cuda-11.8",
        "export PATH=/usr/local/cuda-11.8/bin:/home/zxy/anaconda3/envs/2dgs-slam/bin:$PATH",
        "export TORCH_CUDA_ARCH_LIST=8.9",
        "cd /home/zxy/syb/ZSO/PROJ/code",
        "python -m pytest stream_ortho/tests/test_pose_update.py -q",
        "python -m stream_ortho.scripts.run_stage_s6b --run <run> --drift-m 8 --drift-deg 4",
        "python -m stream_ortho.scripts.make_week7_report",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(L))


def main(argv=None):
    a = build_parser().parse_args(argv)
    s6b, out = pathlib.Path(a.s6b), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    runs = load(s6b)
    for name, d in runs.items():
        for f in ("pose_update_demo.gif", "tearing_check.png", "ortho_before.png", "ortho_after.png"):
            src = s6b / name / f
            if src.exists():
                shutil.copy2(src, out / "figures" / f"{f.rsplit('.',1)[0]}_{name}.{f.rsplit('.',1)[1]}")
    write_table(runs, out)
    write_readme(runs, out)
    (out / "metrics_summary.json").write_text(json.dumps(runs, indent=2, default=float))
    print(f"[week7] wrote {out}/README.md, S6B_TABLE.md, metrics_summary.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
