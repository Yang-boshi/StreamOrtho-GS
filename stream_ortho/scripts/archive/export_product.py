"""P5': export the final ortho **product** from a saved Gaussian cloud.

Renders the whole map at the native GSD into one big image plus a sidecar JSON with the
georeference (plane-frame origin, GSD, ``T_WP``).  A real GeoTIFF needs GDAL/rasterio,
which are not installed here, so the image is written as PNG + TIFF (via ``tifffile``)
with the transform in a JSON sidecar -- enough for QGIS/ArcGIS import by hand and for
reproducible evaluation.

Usage::

    python -m stream_ortho.scripts.export_product --cloud <cloud.npz> --run <run dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.geometry import R_to_quat_xyzw
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.tiles import Tile, stitch


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cloud", required=True)
    p.add_argument("--run", required=True, help="run dir (for T_WP / sim3 / gsd)")
    p.add_argument("--gsd", type=float, default=0.0, help="0 = native GSD from the run")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--max-tiles", type=int, default=400)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    mid = kfs[len(kfs) // 2]
    from stream_ortho.geometry import compose_sim3, sim3_from_vec8

    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd = a.gsd if a.gsd > 0 else depth / float(ds.K_orig[0, 0])

    c = np.load(a.cloud)
    means = c["means"]
    dev = torch.device(zcfg.DEVICE)
    cloud = GaussianCloud(
        means=torch.as_tensor(means, device=dev),
        quats=torch.as_tensor(c["quats"], device=dev),
        scales=torch.as_tensor(c["scales"], device=dev),
        opacity=torch.as_tensor(c["opacity"], device=dev),
        colors=torch.as_tensor(c["colors"], device=dev),
        owner=torch.zeros(len(means), dtype=torch.long, device=dev), meta={})
    P = (means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    grid = OrthoGrid(ds.T_WP, gsd, tile=a.tile, halo=a.halo, z_ground=0.0, z_top=800.0)
    i0, j0 = grid.tile_of(float(P[:, 0].min()), float(P[:, 1].min()))
    i1, j1 = grid.tile_of(float(P[:, 0].max()), float(P[:, 1].max()))
    i0, i1 = min(i0, i1), max(i0, i1)
    j0, j1 = min(j0, j1), max(j0, j1)
    n_tiles = (i1 - i0 + 1) * (j1 - j0 + 1)
    print(f"[exp] gsd {gsd:.4f} m/px, tiles x {i0}..{i1}, y {j0}..{j1} ({n_tiles} tiles)")
    if n_tiles > a.max_tiles:
        raise SystemExit(f"too many tiles ({n_tiles} > {a.max_tiles}); raise --gsd")

    import cv2
    import tifffile

    tiles = []
    nx, ny = i1 - i0 + 1, j1 - j0 + 1
    for jj in range(j0, j1 + 1):
        for ii in range(i0, i1 + 1):
            rgb, alpha, _, _ = render_tile(cloud, grid, ii, jj)
            tiles.append(Tile(ii, jj, rgb, alpha))
    mosaic, cov = stitch(tiles, shape=(ny, nx))
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S29" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    img8 = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "ortho_product.png"), img8[:, :, ::-1])
    tifffile.imwrite(str(out / "ortho_product.tif"), img8)
    meta = {
        "source_run": run_dir.name,
        "n_gaussians": int(len(means)),
        "gsd_m_per_px": float(gsd),
        "origin_plane_m": [float(i0 * a.tile * gsd), float(j0 * a.tile * gsd)],
        "size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
        "extent_plane_m": [float(i0 * a.tile * gsd), float(j0 * a.tile * gsd),
                           float((i1 + 1) * a.tile * gsd), float((j1 + 1) * a.tile * gsd)],
        "T_WP": ds.T_WP.tolist(),
        "coverage": float((cov > 0.5).mean()),
        "note": "ortho in the plane frame defined by T_WP; PNG + TIFF + this sidecar "
                "(a true GeoTIFF needs GDAL/rasterio, not installed)",
    }
    (out / "ortho_product.json").write_text(json.dumps(meta, indent=2))
    print(f"[exp] -> {out}/ortho_product.{{png,tif,json}}  size {mosaic.shape[1]}x{mosaic.shape[0]}, "
          f"coverage {100*meta['coverage']:.1f}%")


if __name__ == "__main__":
    main()
