"""Aggregate the week-6 (S6a) outputs into ``results/week6/``."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from stream_ortho import config as zcfg


def build_parser():
    p = argparse.ArgumentParser(description="week-6 report")
    p.add_argument("--s6", default=str(zcfg.RESULTS_DIR / "stage-S6a"))
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "week6"))
    return p


def load(s6: pathlib.Path):
    runs = {}
    for d in sorted(s6.iterdir()):
        f = d / "metrics.json"
        if d.is_dir() and f.exists():
            runs[d.name] = json.loads(f.read_text())
    return runs


def write_table(runs, out: pathlib.Path):
    lines = [
        "# S6a — dirty-tile replay",
        "",
        "Per keyframe: footprint -> dirty tiles (1-tile margin) -> render up to",
        "`max_tiles` within the time budget.  `dirty ratio` uses the number of tiles",
        "covered so far; the grid column repeats it with the final grid size.",
        "",
        "| run | tile | cap | KFs | tiles covered | complete | dirty ratio (mean) | dirty ratio (grid) | e2e mean [ms] | e2e p95 [ms] | refresh mean [ms] | PSNR [dB] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, d in runs.items():
        c = d["coverage_complete"]
        lines.append(
            f"| {name} | {d['tile']} | {d['max_tiles_per_flush']} | {d['n_kfs']} | "
            f"{d['n_tiles_covered']} | {'yes' if c['complete'] else 'NO'} | "
            f"{100*d['dirty_ratio_mean_after_first']:.1f}% | "
            f"{100*d['dirty_ratio_after_first_grid']:.1f}% | "
            f"{d['end_to_end_ms_mean']:.0f} | {d['end_to_end_ms_p95']:.0f} | "
            f"{d['ortho_refresh_ms_mean']:.0f} | {d['psnr_vs_reference']:.2f} |")
    lines.append("")
    (out / "S6A_TABLE.md").write_text("\n".join(lines))


def write_readme(runs, out: pathlib.Path):
    base = runs.get("npu_calib")
    L = [
        "# Week 6 — S6a report (dirty tiles, cache, latency)",
        "",
        "Goal (design doc §7.7): stop re-rendering the whole map for every keyframe.",
        "A keyframe's ground footprint marks tiles dirty; a budgeted flush renders",
        "only the dirty ones; the cache keeps colour/alpha/depth with versions.",
        "",
        "## TL;DR",
        "",
    ]
    if base:
        c = base["coverage_complete"]
        L += [
            f"* Replay of {base['n_kfs']} keyframes: **{base['n_tiles_covered']} tiles covered, "
            f"{base['n_tiles_rendered']} rendered, missing={c['missing']}** "
            f"({'complete' if c['complete'] else 'INCOMPLETE'}).",
            f"* Dirty ratio after the first keyframe: **{100*base['dirty_ratio_mean_after_first']:.1f}%** "
            f"(grid denominator {100*base['dirty_ratio_after_first_grid']:.1f}%) -- target `< 20%`.",
            f"* Per-keyframe end-to-end **{base['end_to_end_ms_mean']:.0f} ms** (p95 "
            f"{base['end_to_end_ms_p95']:.0f} ms) -- target `< 2 s`.",
            f"* Ortho refresh (flush) **{base['ortho_refresh_ms_mean']:.0f} ms** (p95 "
            f"{base['ortho_refresh_ms_p95']:.0f} ms) -- target `< 3 s`.",
            f"* Cache mosaic vs the plane-warp reference over the **whole 78-tile block**: "
            f"**{base['psnr_vs_reference']:.2f} dB** (week 5 measured 21-24 dB on a 2x2 block of",
            "  the same corridor -- the full block includes the strongly stretched strip edges).",
        ]
    L += [
        "",
        "## Mechanism",
        "",
        "```",
        "new keyframe -> footprint (4 image corners -> plane bbox)",
        "              -> tiles_in(bbox) + 1-tile guard ring",
        "              -> cache.mark_dirty(unique)          # idempotent",
        "              -> flush(budget_ms, max_tiles)        # priority: never-rendered, then oldest version",
        "pose update  -> Gaussians with owner in kf_ids -> their tiles + 1 ring -> dirty",
        "```",
        "",
        "* The guard ring is what prevents seams: a Gaussian near a tile border can",
        "  splat into its neighbour, so the neighbour is re-rendered too.",
        "* `mark_dirty` is idempotent -- re-marking a covered tile does not enqueue it",
        "  twice (verified in `test_dirty_tiles.py`).",
        "* The per-flush cap (default 8) bounded a single keyframe: the first frame",
        "  marks ~35 tiles but renders only what fits the budget.",
        "",
        "## Acceptance (§7.7)",
        "",
        "| item | target | measured (npu) |",
        "|---|---|---|",
        f"| dirty ratio per keyframe | < 20% | {100*(base or {}).get('dirty_ratio_mean_after_first', float('nan')):.1f}% |",
        f"| end-to-end per keyframe | < 2 s | {(base or {}).get('end_to_end_ms_mean', float('nan')):.0f} ms |",
        f"| ortho refresh | < 3 s | {(base or {}).get('ortho_refresh_ms_mean', float('nan')):.0f} ms |",
        f"| no missed tiles | 0 | {(base or {}).get('coverage_complete', {}).get('missing', '?')} |",
        "| no duplicate enqueue | 0 | verified by test + `newly_dirty` accounting |",
        "",
        "## Ablations",
        "",
        "See [`S6A_TABLE.md`](S6A_TABLE.md): tile 512 vs 256 (dirty ratio and latency) and",
        "cap 8 vs 4 (latency vs backlog).",
        "",
        "## What the figures show (and what they are NOT)",
        "",
        "| file | meaning |",
        "|---|---|",
        "| `dirty_heatmap_<run>.png` | per-tile **render count** over the replay. Yellow/green = rendered once, red = re-rendered when a later keyframe overlapped it, grey = no tile (outside the union of footprints). It is the QA view of the scheduler, not map content. |",
        "| `mosaic_<run>.png` | the tile cache assembled by grid index. **Black = no keyframe saw that tile** (the flown corridor is S-shaped, the tile-grid bbox is not). Tile order is flipped vertically (`j` grows up, image rows grow down). |",
        "| `compare_<run>.png` | `mosaic \| reference` on the *same* tile block; the reference is the GT-pose plane warp. |",
        "| `blur_stride6_vs_stride2_vs_ref.png` | same crop at 1.2 m/point, 0.4 m/point and the reference: shows that **the blur is the sampling density**, not a broken renderer. |",
        "",
        "Three things must not be read as map errors:",
        "",
        "1. **black areas** = never covered (only 8 of 47 keyframes are replayed);",
        "2. **slanted edges** = each keyframe's perspective footprint is a trapezoid; the",
        "   union of 8 trapezoids is not a rectangle;",
        "3. **blur** = the plane splat samples every 6th pixel (1.2 m) at a 0.2 m GSD,",
        "   so each output pixel averages ~6x6 input pixels. The stride-2 run halves the",
        "   sampling distance and is visibly sharper (see the figure).",
        "",
        "## Figures",
        "",
        "![dirty heatmap](figures/dirty_heatmap_npu_calib.png)",
        "",
        "![stride comparison](figures/blur_stride6_vs_stride2_vs_ref.png)",
        "",
        "![cache mosaic](figures/mosaic_npu_calib.png)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export CUDA_HOME=/usr/local/cuda-11.8",
        "export PATH=/usr/local/cuda-11.8/bin:/home/zxy/anaconda3/envs/2dgs-slam/bin:$PATH",
        "export TORCH_CUDA_ARCH_LIST=8.9",
        "cd /home/zxy/syb/ZSO/PROJ/code",
        "python -m pytest stream_ortho/tests/test_dirty_tiles.py -q",
        "python -m stream_ortho.scripts.run_stage_s6a --run <run> --max-kf 8 --stride 6 --budget-ms 250",
        "python -m stream_ortho.scripts.make_week6_report",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(L))


def main(argv=None):
    a = build_parser().parse_args(argv)
    s6, out = pathlib.Path(a.s6), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    runs = load(s6)
    for name in ("npu_calib", "ieu_calib", "factory_calib"):
        for f, tag in (("dirty_heatmap.png", "dirty_heatmap"), ("mosaic.png", "mosaic"),
                       ("compare.png", "compare")):
            src = s6 / name / f
            if src.exists():
                shutil.copy2(src, out / "figures" / f"{tag}_{name}.png")
    write_table(runs, out)
    write_readme(runs, out)
    (out / "metrics_summary.json").write_text(json.dumps(runs, indent=2, default=float))
    print(f"[week6] wrote {out}/README.md, S6A_TABLE.md, metrics_summary.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
