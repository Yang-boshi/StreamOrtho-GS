# stream-ortho

**Streaming feed-forward 2.5D orthomapping** — images go in, orthophoto tiles come out while
the flight is still running. Per-frame depth bias is removed with an online 3-DOF calibration
whose **layer/owner assignment is frozen**, so the correction does not fragment the product.

## Install

```bash
git clone --recursive <this repo>            # submodules: third_party/*
cd stream-ortho
pip install -e ".[dev]"
```

`torch` with CUDA is only needed by the optional Gaussian head; the replay pipeline runs on
CPU.

## One command

```bash
python -m stream_ortho.app.run \
  --config configs/p2b_async.yaml \
  --images /path/to/rgb \                # or --obs <observation table> for a replay
  --out results/run_001 \
  --max-frames 55
```

Outputs, in `results/run_001/`:

| file | content |
|---|---|
| `mosaic_00010.png …` | preview mosaics, one every `ortho.dump_every` frames (async) |
| `mosaic_final.png` | final mosaic after the last frame |
| `stream.json` | per-frame latency, filter counters, tile coverage, quality vs the oracle |
| `filter_stats.json` | how many points the gate / free-space prior / prune removed |
| `config_resolved.yaml` | the exact configuration of the run |
| `offline/` | post-flight 3 rounds + frozen assignment: `product_frozen.npz`, `pivotb_hybrid.json`, texture ortho |

## Configuration

`configs/default.yaml` documents every switch. The ones that matter:

```yaml
data:
  intrinsics: auto          # auto | exif | provided | none
filter:
  enabled: true             # Module A: gate + free-space prior + isolated prune
  nfp_mode: above           # veto points ABOVE the resolved column top (the original NFP)
map:
  freeze_owner: true        # the method: never re-group, never re-elect an owner
  online_calib: true
ortho:
  render: cell              # cell (fast) | texture | hybrid
  dump_every: 10
  async_dump: true
offline:
  enabled: true             # post-flight refinement
```

Ablations: `configs/ablation_no_filter.yaml`, `ablation_no_freeze.yaml`,
`ablation_no_calib.yaml`, `ablation_offline_only.yaml`.

## P5: 3DGS high-resolution final ortho

`gaussian.final_enabled: true` (see `configs/p5_enabled.yaml`) adds an offline
3DGS final render after the streaming product: every SLAM keyframe's
MASt3r-anchored Gaussians (`X_canon` means, full-resolution SH colour, Splatt3R
scales/quats/opacity) are rendered with an ORTHO gsplat camera onto one anchored
canvas at the auto source-pixel GSD and Laplacian-blended.

```bash
python -m stream_ortho.app.run --config configs/p5_enabled.yaml \
    --e2e --no-calib --images data/phantom3-ieu-kfs --max-frames 200 --seq ieu \
    --out results/run_p5
```

Outputs (in `results/run_p5/p5_3dgs/`):

* `mosaic_final_3dgs.png` (+ `_alpha`) — high-resolution 3DGS ortho
* `ground_3dgs.ply` — standard 3DGS schema (plane frame, Z-up, + `keyframe_id`);
  opens in CloudCompare / SuperSplat / open3d
* `ortho_3dgs.json` — gsd, canvas, timings, coverage
* `ortho_update_3dgs.mp4` + `snaps/` — per-keyframe accumulation clip

The cell-based product is unchanged (`mosaic_final.png`, plus the
`mosaic_final_preview.png` copy). Key switches under `gaussian:`:
`gsd_3dgs` (0 = auto: one source pixel per keyframe), `conf_threshold`,
`blend_levels`, `feather_radius`, `max_kf` (0 = all keyframes). A finished run can
be re-rendered without re-running SLAM:

```bash
python -m stream_ortho.scripts.run_p5 --final \
    --run results/run_p5 --out results/run_p5/p5_3dgs
```

Validation: `results/system/phase2_p5_3dgs/p5_validation_report.md` (ieu + npu,
time/FPS/PLY/roof-retention/ground-RMSE gates all pass).

## Scale and honesty

* **No intrinsics?** `data.intrinsics: none` still runs, but the cell size and GSD are
  *relative* units; `run.json` / `config_resolved.yaml` record `metric_scale: false` and the
  console prints "relative scale". The bundled ieu images have no 35 mm EXIF focal length, so
  `auto` falls back to `default` and reports relative scale as well.
* **Preview ≠ final.** The streaming product is a single pass within a latency budget
  (P50 ≈ 450 ms, P95 < 1 s on this machine). Its ripple amplitude is worse than the offline
  three-round product; `stream.json` reports both and the README of the paper experiment
  keeps them apart.
* **No live front-end is bundled.** `--frontend live` raises a clear error until
  `third_party/MASt3R-SLAM` is added as a submodule.

## Tests

```bash
pytest -q            # 226 tests (in the light env, 20 require the gsplat env and fail there)
```

## Licence

MIT for this repository. Upstream components keep their own licences (see `LICENSES/`).
