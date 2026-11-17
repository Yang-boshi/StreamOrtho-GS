# Architecture

```
RGB frames ──► Frontend (replay now, MASt3R-SLAM live later)
                 │  FramePacket: pose, pointmap, keyframe flag
                 ▼
            PointmapFilter                     Module A
                 │  gate (conf/depth/height) + NFP (free space above the column top)
                 │  + 3-D isolated prune, with unique/low-observation protection
                 ▼
            StreamingProduct + OnlineCalib     Module B
                 │  (cell, layer, owner); a match only updates the weighted mean,
                 │  the owner never changes; one 3-DOF fit per arriving frame
                 ▼
            FOVOrtho + DirtyTiles              Module C
                 │  footprint bbox → mark dirty → flush within a time budget
                 ▼
            AsyncMosaicWriter
                 │  snapshot + bounded queue, worker stitches and writes atomically
                 ▼
            preview mosaics ──► (post-flight) OfflineRefine
                                   3 rounds of frozen calibration → product_frozen.npz
                                   + texture orthophoto
```

## Design rules

1. **The streaming loop never rebuilds the whole product.** It only inserts new observations
   and re-renders the tiles a frame touches.
2. **The owner is a property of the layer, not of the label.** Re-grouping after a per-frame
   correction is what fragments ownership; the product therefore freezes `(cell, slot, owner)`
   and only moves heights.
3. **Every stage is optional and switchable from the config** (`filter.*`, `map.freeze_owner`,
   `map.online_calib`, `offline.enabled`, `ortho.async_dump`), which is what makes the
   ablations in `configs/` single-switch experiments.
4. **Images never block the loop.** Mosaics are snapshotted and written by a background
   worker; a full queue drops the oldest *preview* but never the final.
5. **Scale is explicit.** With `data.intrinsics: none` or missing EXIF the pipeline reports
   `metric_scale=false` and every output says "relative", never metres.

## Module map

| Package | Role |
|---|---|
| `app/` | one-command entry, config schema, pipeline, frontend, callbacks, async writer, offline stage |
| `data/` | `RGBFolderDataset` (plain folder, four intrinsics modes) and the legacy NPU dataset |
| `filter/` | Module A: per-frame pointmap filtering |
| `online/` | Module B: streaming product and online calibration |
| `ortho/` | Module C: grid, dirty-tile scheduler, texture renderer, FOV helpers, mosaic writer |
| `map/`, `eval/` | the offline calibration core and the metrics (shared with the paper) |
| `third_party/` | git submodules for MASt3R-SLAM / splatt3r (not vendored) |
