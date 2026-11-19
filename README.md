# StreamOrtho-GS (Ortho-3dgs)

**True-streaming feed-forward 3DGS orthomapping from monocular drone RGB streams.**
RGB frames go in; MASt3R-SLAM tracks and optimizes keyframe poses while a
feature-shared Splatt3R adapter lifts native-resolution Gaussians per keyframe,
synchronizes backend BA pose updates, maintains a nadir-priority 2.5D column-consensus
3DGS map, and incrementally splats high-resolution orthophoto tiles (`mosaic_XXXXX.png`)
while the flight is still running.

This repository is self-contained: the upstream MASt3R-SLAM checkout (with our
2-file patch) is vendored under `third_party/`. Model *weights* are **not**
bundled — see [Setup](#setup).

## End-to-end streaming data flow

```
RGB frames (plain folder/*.jpg, folder/rgb/*.jpg, or NPU sequence)
  │
  ├─① stream_ortho/data/                     RGBFolderDataset / NPUDroneMapDataset
  │     · intrinsics: auto | provided [fx,fy,cx,cy] | exif | none (no config.cfg required)
  │
  ├─② SLAM front-end + backend BA (MASt3R-SLAM via scripts/run_mast3r_baseline.py + app/frontend.py)
  │     · MASt3R inference per frame: canonical pointmap X_canon (288×512×3) + encoder (feat, pos)
  │     · keyframe selection + async FactorGraph global BA & loop closure
  │     · live POSE_UPDATE_HOOK syncs BA pose refinements via SharedKeyframes.pose_version
  │
  └─③ True-Streaming 3DGS Ortho Pipeline (app/pipeline.py _run_e2e_streaming_3dgs)
        · adapters/splatt3r.py  lift_keyframe_gaussians(): reuses MASt3R (feat, pos) -> Splatt3R head;
        │                       3×3 median depth-spike guard + edge-aware upsampling + GPU Shepperd
        │                       surfel quaternions; full native RGB lifting (`flat_stride: 1`)
        · app/alignment.py      ProgressivePlaneAligner: camera-up bootstrap + per-keyframe ground
        │                       RANSAC leveling + verified ground-inlier Z tie-in (roof-resistant)
        · online/gs_map.py      StreamingGaussianMap: 2D nadir-column Top-1 consensus, post-BA
        │                       Sim(3) rigid pose updates + column re-dedup, spatial tile buckets
        · ortho/gsplat_ortho.py gsplat ORTHO rasterization per dirty tile (+ halo crop & alpha norm)
        · app/viewer.py         non-blocking `--show` live orthophoto OpenCV preview window
        · app/mosaic_writer.py  async per-keyframe frames/mosaic_XXXXX.jpg (0.5× fast dump) without blocking SLAM
        · final outputs (<1s):  mosaic_final.png, p5_3dgs/mosaic_final_3dgs.png,
                                ground_3dgs.ply (Z-up plane frame, log scales, logit opacity),
                                ortho_update.mp4, ortho_fps.json, stream.json
```

`splatt3r` the repository is *not* a runtime dependency: the Gaussian-head code
was adapted into `stream_ortho/frontend/gaussian_head.py`; only its checkpoint
is used. The vendored upstream is `third_party/MASt3R-SLAM` (= `mast3r_slam`
package + `thirdparty/mast3r` (dust3r/croco) + `thirdparty/in3d`).

## Repository layout

```
StreamOrtho-GS/
├── stream_ortho/           the package
│   ├── adapters/           MASt3R/Splatt3R → edge-aware sub-pixel surfels
│   ├── app/                e2e streaming pipeline, progressive aligner, CLI, async writer
│   ├── backend/            refinement workers
│   ├── data/               RGBFolderDataset & NPUDroneMapDataset
│   ├── eval/               metrics (ATE, grid, latency, …)
│   ├── filter/             pointmap filter (Module A)
│   ├── frontend/           Gaussian head + prediction
│   ├── map/                owner grid, calibration, layering
│   ├── online/             StreamingGaussianMap (gs_map.py), StreamingProduct, OnlineCalib
│   ├── ortho/              tile engine, gsplat ortho renderer, blending
│   ├── scripts/            runtime SLAM baseline, PLY export, & legacy fallback helpers
│   ├── config.py compat.py geometry.py
├── tests/                  pytest suite
├── configs/
│   └── base.yaml           default streaming 3DGS configuration
├── docs/                   architecture notes, licences, third-party notes
└── third_party/
    └── MASt3R-SLAM/        vendored upstream (+ checkpoints symlink)
```

## Setup

Requires Linux with an NVIDIA GPU (16 GB+ VRAM recommended for long flights) and matching CUDA toolkit (`nvcc`, tested with PyTorch 2.5.1 + CUDA 11.8/12.1/12.4).

### 1. Create environment & install PyTorch

```bash
git clone https://github.com/Yang-boshi/StreamOrtho-GS.git
cd StreamOrtho-GS

conda create -n ortho-3dgs python=3.10 -y
conda activate ortho-3dgs

# Install PyTorch matching your system CUDA toolkit (`nvcc --version`):
# Example (CUDA 12.1):
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# (Or CUDA 11.8: replace pytorch-cuda=12.1 with pytorch-cuda=11.8)
```

### 2. Build CUDA extensions & install dependencies

```bash
# 1) Core Python dependencies + ninja (for gsplat JIT) & gsplat ortho rasterizer
pip install -e ".[live,dev]"

# 2) Build vendored MASt3R (curope RoPE + asmk), in3d viewer, MASt3R-SLAM backend, and lietorch
pip install --no-build-isolation -e third_party/MASt3R-SLAM/thirdparty/mast3r
pip install --no-build-isolation -e third_party/MASt3R-SLAM/thirdparty/in3d
pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git
pip install --no-build-isolation -e third_party/MASt3R-SLAM
```

### 3. Download pretrained weights

Weights are **not** stored in the repository. Download them into `checkpoints/` at the repository root (or set `export ORTHO_PROJ=/path/to/workspace` if stored externally):

```bash
mkdir -p checkpoints/mast3r checkpoints/splatt3r

# MASt3R + retrieval weights (from Naver Labs / MASt3R-SLAM)
wget -P checkpoints/mast3r https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
wget -P checkpoints/mast3r https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
wget -P checkpoints/mast3r https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

# Splatt3R Gaussian head weight (from HuggingFace brandonsmart/splatt3r_v1.0)
wget -O checkpoints/splatt3r/epoch=19-step=1200.ckpt \
  "https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt"
```

Expected layout:

| file | path under `checkpoints/` |
|---|---|
| MASt3R | `checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` |
| MASt3R retrieval | `checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth` |
| MASt3R codebook | `checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl` |
| Splatt3R head | `checkpoints/splatt3r/epoch=19-step=1200.ckpt` |

`run_mast3r_baseline.py` automatically links `third_party/MASt3R-SLAM/checkpoints`
to `<CKPT_DIR>/mast3r` so backend global BA & loop closure load cleanly.

## Run end-to-end (True-Streaming 3DGS)

Run from the repository root `StreamOrtho-GS/` (`PYTHONPATH=.:third_party/MASt3R-SLAM` ensures the vendored MASt3R-SLAM checkout is always used even if another copy exists in your conda env):

### 1. Auto-detect intrinsics (`config.cfg`, `calib.txt`, or EXIF; fallback to pure RGB)

```bash
PYTHONPATH=.:third_party/MASt3R-SLAM \
python -m stream_ortho.app.run \
  --config configs/base.yaml \
  --e2e --show \
  --images data/phantom3-ieu-kfs \
  --out results/e2e_stream_3dgs_ieu
```

Pass `--show` to open a non-blocking OpenCV window (`Ortho-3DGS Live`) showing
the incremental orthophoto updating as keyframes arrive and BA syncs complete.

### 2. Plain RGB folder with explicit intrinsics (no `config.cfg` needed)

You can set `data.intrinsics: provided` and `data.intrinsics_value: [fx, fy, cx, cy]`
directly in `configs/base.yaml`, or pass via CLI:

```bash
PYTHONPATH=.:third_party/MASt3R-SLAM \
python -m stream_ortho.app.run \
  --config configs/base.yaml \
  --e2e \
  --intrinsics provided --intrinsics-value 1679.71,1679.71,775.465,521.95 \
  --images data/ceshi1 \
  --out results/e2e_stream_ceshi1_calib
```

### 3. Pure RGB only (no intrinsics at all, relative scale)

```bash
PYTHONPATH=.:third_party/MASt3R-SLAM \
python -m stream_ortho.app.run \
  --config configs/base.yaml \
  --e2e --intrinsics none \
  --images data/ceshi1 \
  --out results/e2e_stream_ceshi1_nocalib
```

### Outputs in `--out`

| file | content |
|---|---|
| `frames/mosaic_XXXXX.jpg` | per-keyframe streaming 3DGS orthophoto previews (0.5× async) |
| `mosaic_final.png` | final converged high-resolution 3DGS orthophoto |
| `p5_3dgs/mosaic_final_3dgs.png` (+ `_alpha`) | final 3DGS ortho + raw alpha map |
| `ground_3dgs.ply` | standard 3DGS PLY (leveled Z-up plane frame, `log` scales, `logit` opacity; CloudCompare & SuperSplat ready) |
| `ortho_update.mp4` | aligned per-keyframe incremental accumulation clip |
| `ortho_3dgs.json`, `ortho_fps.json`, `alignment.json`, `stream.json` | timings, GSD, alignment & run metadata |
| `keyframes.npz`, `run.json` | SLAM keyframes + run metadata |

## Key configuration parameters (`configs/base.yaml`)

### `data` & `frontend`

| key | default | meaning |
|---|---|---|
| `data.intrinsics` | `auto` | `auto` \| `exif` \| `provided` \| `none` |
| `data.intrinsics_value` | `null` | `[fx, fy, cx, cy]` when `provided` |
| `frontend.use_calib` | `"auto"` | `"auto"` uses intrinsics whenever available; `"true"` / `"false"` force |

### `gaussian` (Streaming 3DGS)

| key | default | meaning |
|---|---|---|
| `streaming_3dgs` | `true` | `true` = true-streaming per-keyframe 3DGS ortho; `false` = legacy cell preview |
| `adaptive_stride` | `true` | `stride=1` on high-gradient edges/textures, `flat_stride` on smooth areas |
| `flat_stride` | `1` | sampling stride on smooth surfaces (`1` = full dense native RGB) |
| `edge_grad_pct` | `70.0` | image gradient percentile promoted to native `stride=1` |
| `depth_upsample` | `edge` | `edge` (bilinear on smooth, nearest at jumps) \| `nearest` \| `linear` |
| `edge_rel_tau` | `0.03` | relative depth-jump threshold (scale-invariant) |
| `voxel_gsd_mul` | `1.0` | 2D column consensus cell size (`1.0 × gsd` = 1:1 with ortho pixels) |
| `nadir_replace_margin` | `0.15` | replace column winner when `nadir_score` improves by margin |
| `gsd_3dgs` | `0.0` | `0.0` = auto native pixel footprint GSD |
| `max_canvas_px` | `6144` | safety cap on longest mosaic dimension |

### `map`

| key | default | meaning |
|---|---|---|
| `bootstrap_kfs` | `3` | initial frames used to fit global leveled ground plane |
| `progressive_align` | `true` | per-keyframe local ground leveling + verified ground-inlier Z tie-in |

### `ortho`

| key | default | meaning |
|---|---|---|
| `show` | `false` | open real-time preview window (or pass `--show` on CLI) |

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_streaming_3dgs.py tests/test_ply_export.py tests/test_config.py tests/test_pipeline_smoke.py -v
```

## Licence

MIT for this repository. Upstream components keep their own licences — see
`docs/LICENSES/` and `docs/third_party_notes/`.
