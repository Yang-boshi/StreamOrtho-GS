# Ortho-3dgs

**Streaming feed-forward 2.5D orthomapping with a 3DGS high-resolution final
product.** RGB images from a drone flight go in; an orthophoto mosaic is built
*while the flight is still running* (streaming cell/tile product), and a
Gaussian-splatting render produces the high-resolution final ortho plus a
standard 3DGS PLY after the last frame.

This repository is self-contained: the upstream MASt3R-SLAM checkout (with our
2-file patch) is vendored under `third_party/`. Model *weights* are **not**
bundled — see [Setup](#setup).

## End-to-end data flow

```
RGB frames (data/<seq>/*.jpg)
  │
  ├─① stream_ortho/data/npu_dataset.py        read + intrinsics (auto/exif/provided) + undistort
  │
  ├─② SLAM front-end (MASt3R-SLAM loop, mirrored in
  │     stream_ortho/scripts/run_mast3r_baseline.py + app/frontend.py)
  │     · MASt3R inference per frame: pointmap X_canon (288×512×3) + confidence
  │     · keyframe selection, Sim(3) pose optimisation, global BA window
  │     · artefacts: keyframes.npz (X_canon fp16, T_WC, K, frame_id) + run.json
  │
  ├─③ streaming ortho product (app/pipeline.py run_e2e)
  │     · filter/pointmap.py   confidence gate + free-space prior (NFP) + isolated prune
  │     · map/build.py         per-cell owner grid + online 3-DOF calibration + layering
  │     · ortho/               dirty-tile engine: incremental rgb/texture tiles per frame
  │     · artefacts: mosaic_final.png (preview), ortho_update.mp4, stream.json
  │
  └─④ P5 3DGS final render (after final mosaic, when gaussian.final_enabled)
        · adapters/splatt3r.py  per keyframe: Splatt3R head → scales/quats/opacity/SH residual;
        ·                       means anchored to X_canon; SH DC = RGB2SH(full-res image) + residual;
        ·                       Sim(3) → world → plane transform + confidence gate
        · ortho/gsplat_ortho.py gsplat ORTHO camera render onto one anchored canvas (auto GSD)
        · ortho/blend.py        Laplacian-pyramid cross-keyframe blend
        · scripts/export_3dgs_ply.py  standard 3DGS PLY (plane frame, Z-up, keyframe_id)
        · artefacts: mosaic_final_3dgs.png, ground_3dgs.ply, ortho_3dgs.json,
                     ortho_update_3dgs.mp4
```

`splatt3r` the repository is *not* a runtime dependency: the Gaussian-head code
was adapted into `stream_ortho/frontend/gaussian_head.py`; only its checkpoint
is used. The vendored upstream is `third_party/MASt3R-SLAM` (= `mast3r_slam`
package + `thirdparty/mast3r` (dust3r/croco) + `thirdparty/in3d`), carrying two
patches: `dataloader.py` (jpg/jpeg support) and `frame.py` (`pose_version`
hook). See `docs/third_party_notes/`.

## Repository layout

```
Ortho-3dgs/
├── stream_ortho/           the package
│   ├── adapters/           MASt3R/Splatt3R → anchored Gaussians
│   ├── app/                e2e pipeline, CLI (app.run), alignment, artifacts
│   ├── backend/            refinement workers
│   ├── data/               dataset readers
│   ├── eval/               metrics (ATE, grid, latency, …)
│   ├── filter/             pointmap filter (Module A)
│   ├── frontend/           Gaussian head + prediction
│   ├── map/                owner grid, calibration, layering
│   ├── online/             streaming product
│   ├── ortho/              tile engine, gsplat ortho renderer, blending
│   ├── scripts/            active scripts (SLAM baseline, P5, exports, diag/)
│   │   └── archive/        historical experiment scripts (run_s*, make_week*)
│   ├── config.py compat.py geometry.py
├── tests/                  pytest suite
├── configs/                run configs (default, e2e*, p4_*, p5_enabled, ablations)
├── docs/                   legacy README, licences, third-party notes
├── third_party/
│   └── MASt3R-SLAM/        vendored upstream (+ our 2-file patch)
│       └── thirdparty/{mast3r,in3d}
├── pyproject.toml  requirements-lock.txt  README.md
```

## Setup

```bash
git clone <this repo>
cd Ortho-3dgs
pip install -e ".[dev]"        # torch w/ CUDA required for the live front-end
```

Weights (not in the repo) are looked up under `<ORTHO_PROJ>/checkpoints/`:

| file | default path |
|---|---|
| MASt3R | `checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` |
| Splatt3R head | `checkpoints/splatt3r/epoch=19-step=1200.ckpt` |

By default `<ORTHO_PROJ>` is the grand-parent of the repo checkout; set
`ORTHO_PROJ=/path/to/project` to point data (`data/`), checkpoints and results
(`results/`) elsewhere. Datasets are image folders under `$ORTHO_PROJ/data/`.

A ready-made environment (used for development, Python 3.10 + CUDA torch +
gsplat 1.4): activate the `2dgs-slam` conda env, and make sure its `bin/` is on
`PATH` for the gsplat JIT.

## Run end-to-end

From the **repo root's project dir** (`cd $ORTHO_PROJ`), env `2dgs-slam`:

```bash
PYTHONPATH=code/Ortho-3dgs:code/Ortho-3dgs/third_party/MASt3R-SLAM \
python -m stream_ortho.app.run \
  --config code/Ortho-3dgs/configs/p5_enabled.yaml \
  --e2e --no-calib \
  --images data/phantom3-ieu-kfs --seq ieu \
  --max-frames 200 \
  --out results/system/e2e_ieu
```

Calibrated variant (known intrinsics):

```bash
  ... --calib --intrinsics provided --intrinsics-value 1221.15,1221.15,959.5,539.5
```

Outputs in `--out`:

| file | content |
|---|---|
| `mosaic_final.png` | streaming cell preview (P4) |
| `mosaic_final_3dgs.png` (+ `_alpha`) | **3DGS high-resolution ortho** |
| `ground_3dgs.ply` | standard 3DGS PLY (plane frame, Z-up, `keyframe_id`) |
| `ortho_3dgs.json` | GSD, canvas, timings, coverage |
| `ortho_update_3dgs.mp4`, `snaps/` | per-keyframe accumulation clip |
| `keyframes.npz`, `run.json`, `stream.json` | SLAM keyframes + run metadata |

Re-render the 3DGS product from a finished run without re-running SLAM:

```bash
python -m stream_ortho.scripts.run_p5 --final \
    --run results/system/e2e_ieu --out results/system/e2e_ieu/p5_3dgs
```

## Tunable parameters

`configs/default.yaml` documents every switch; `configs/p5_enabled.yaml` is the
e2e + 3DGS preset. The ones that matter:

### data

| key | default | meaning |
|---|---|---|
| `image_size` | 512 | MASt3R input resolution (e2e_768 → 768, sharper but slower) |
| `max_frames` | 0 (all) | cap the number of frames processed |
| `intrinsics` | auto | `auto` \| `exif` \| `provided` \| `none` (relative scale) |
| `intrinsics_value` | null | `[fx,fy,cx,cy]` when `provided` |

### filter (Module A)

| key | default | meaning |
|---|---|---|
| `gate` / `nfp` / `prune` | true | the three filter stages; each can be disabled |
| `min_conf` | 0.0 | extra confidence gate |
| `depth_min_m`/`depth_max_m` | 0.1 / 1500 | depth range kept |
| `height_min_m`/`height_max_m` | −3 / 120 | plane-frame height range kept |
| `iso_radius_m` / `iso_conf_max` | 1.5 / 1.2 | isolated-point prune |
| `nfp_mode` | above | `above` = veto points above the resolved column top |
| `nfp_tau_m` / `nfp_top_pct` / `nfp_min_obs` | 4.0 / 90 / 2 | free-space prior |

### map

| key | default | meaning |
|---|---|---|
| `cell` | 0.45 | mosaic cell size (m) — controls preview resolution |
| `freeze_owner` | true | the method: never re-group / re-elect owners |
| `online_calib` | true | online 3-DOF depth-bias calibration |
| `window` / `window_every` / `window_rounds` | 0 / 5 / 2 | sliding-window recalibration |

### ortho

| key | default | meaning |
|---|---|---|
| `gsd` | 0.30 | preview GSD (m/px); the 3DGS final uses `gaussian.gsd_3dgs` |
| `tile` / `budget_ms` / `max_tiles` | 512 / 500 / 6 | dirty-tile rendering budget |
| `dump_every` / `async_dump` | 1 / true | preview dump frequency (video frames) |

### gaussian (P5 3DGS)

| key | default | meaning |
|---|---|---|
| `final_enabled` | true | run the 3DGS final render after the streaming product |
| `gsd_3dgs` | 0.0 (auto) | 0 = auto: one source pixel per keyframe (metric-safe) |
| `conf_threshold` | 0.5 | Gaussian confidence gate |
| `blend_levels` / `feather_radius` | 4 / 24 | Laplacian blend across keyframes |
| `max_kf` | 0 (all) | cap keyframes used in the final render |
| `apply_filter` | true | filter pointmaps before anchoring |
| `anchor` | head | means source (`head` = X_canon anchoring) |

### offline (post-flight refinement)

| key | default | meaning |
|---|---|---|
| `enabled` / `iters` | true / 3 | post-flight rounds + frozen assignment |
| `render_texture` | true | texture ortho |

CLI: `--config --e2e --calib/--no-calib --images --max-frames --seq --out
--intrinsics provided --intrinsics-value fx,fy,cx,cy`.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q     # light env: 20 optional gsplat tests fail there
```

## Licence

MIT for this repository. Upstream components keep their own licences — see
`docs/LICENSES/` and `docs/third_party_notes/`.
