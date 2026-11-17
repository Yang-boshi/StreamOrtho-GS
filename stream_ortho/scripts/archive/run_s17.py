"""Stage S17: E2 (facade culling) and E1 (point-map fusion) for the ortho.

E2: pixels whose own point-map depth is much smaller than the distance to the
DSM/plane hit are looking at a **vertical surface** (a wall standing in front of
the roof plane).  They are not visible from above, so we drop them instead of
smearing their colour onto the ground next to the building.

E1: fuse all keyframes' point maps into one voxel-averaged consensus cloud
(the relative height stays, the frame-to-frame disagreement goes) and place one
Gaussian per voxel.

Both are compared against the current baseline (plane + piecewise DSM + ray-DSM)
on numpy (`npu`) and on the building-heavy `factory` sequence, using PSNR,
gradient ratio and the roof-edge transition width.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import fuse_pointmaps, plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def edge_width(gray: np.ndarray) -> tuple[float, int]:
    """Median 10-90% rise distance over the strongest edges (px)."""
    import cv2

    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.hypot(gx, gy)
    th = np.percentile(mag, 99.0)
    ys, xs = np.nonzero(mag >= th)
    W = []
    for y, x in zip(ys[::25], xs[::25]):
        if not (12 <= y < g.shape[0] - 12 and 12 <= x < g.shape[1] - 12):
            continue
        prof = g[y, x - 8:x + 9] if abs(gx[y, x]) > abs(gy[y, x]) else g[y - 8:y + 9, x]
        if prof.size != 17 or np.abs(np.diff(prof)).max() < 8:
            continue
        lo, hi = prof.min(), prof.max()
        if hi - lo < 12:
            continue
        f = (prof - lo) / (hi - lo)
        i10 = int(np.argmax(f >= 0.1))
        i90 = int(np.argmax(f >= 0.9))
        if i90 > i10:
            W.append(i90 - i10)
    return (float(np.median(W)) if W else float("nan")), len(W)


def gsharp(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    import cv2

    ga = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor((np.clip(b, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Sobel(ga, cv2.CV_32F, 1, 0, 3))[mask].mean()
                 / max(np.abs(cv2.Sobel(gb, cv2.CV_32F, 1, 0, 3))[mask].mean(), 1e-9))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.3)
    p.add_argument("--tiles", type=int, default=2)
    p.add_argument("--cull", default="0,1.0,3.0")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    ds = NPUDroneMapDataset(pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"])
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0][:8].tolist()
    mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth / float(ds.K_orig[0, 0])
    grid = OrthoGrid(ds.T_WP, gsd, tile=512, halo=32, z_ground=0.0, z_top=800.0)
    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    dsm = build_piecewise_dsm(ds, npz, sg, kfs, gsd=0.5)
    print(f"[s17] {ds.name}: DSM {dsm.shape} {dsm.stats()}")

    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S17" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    cfgs = [("baseline", dict(dsm=dsm, ray_dsm=True, cull_facade=0.0))]
    for c in str(a.cull).split(","):
        c = float(c)
        if c > 0:
            cfgs.append((f"cull{c:g}", dict(dsm=dsm, ray_dsm=True, cull_facade=c)))

    rows = []
    clouds = {}
    for tag, kw in cfgs:
        gmap = GaussianMap(MapConfig(voxel_size=0.3, dedup=True, gate=False,
                                     commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
        for kf in kfs:
            gmap.ingest(plane_patch(ds, npz, kf, sg, stride=a.stride, shape="iso", **kw), kf)
            gmap.commit(kf)
        clouds[tag] = gmap.to_cloud()
        print(f"[s17] {tag}: N={len(clouds[tag]):,}")

    # E1: consensus cloud
    import torch

    fused = fuse_pointmaps(ds, npz, sg, kfs, stride=a.stride, voxel=a.voxel)
    dev = zcfg.DEVICE
    clouds["fused"] = GaussianCloud(
        means=torch.as_tensor(fused.means, device=dev),
        quats=torch.as_tensor(fused.quats, device=dev),
        scales=torch.as_tensor(fused.scales, device=dev),
        opacity=torch.as_tensor(fused.opacity, device=dev),
        colors=torch.as_tensor(fused.colors, device=dev),
        owner=torch.as_tensor(np.asarray(fused.owner).reshape(-1), device=dev),
        meta={})
    print(f"[s17] fused: N={len(clouds['fused']):,} (voxel {a.voxel} m)")

    import cv2

    # pass 1: render everything and keep the masks, so PSNR can be measured on a
    # **common** valid region -- otherwise culling "improves" PSNR simply by hiding
    # pixels (coverage collapse), which is not an improvement at all
    renders = {}
    for n in range(a.tiles):
        i, j = ic, jc + n
        tgt, cov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
        tgt_g = cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ew_ref, _ = edge_width(tgt_g)
        for tag, cloud in clouds.items():
            rgb, alpha, _, _ = render_tile(cloud, grid, i, j)
            renders[(n, tag)] = (rgb, (alpha > 0.5) & (cov > 0.5), cov > 0.5)

    for n in range(a.tiles):
        i, j = ic, jc + n
        tgt, cov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
        common = np.ones_like(cov > 0.5)
        for tag in clouds:
            if (n, tag) in renders:
                common &= renders[(n, tag)][1]
        cov_ref = int((cov > 0.5).sum())
        for tag, cloud in clouds.items():
            if (n, tag) not in renders:
                continue
            rgb, mk, _ = renders[(n, tag)]
            if mk.sum() < 5000:
                continue
            gr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            ew, ne = edge_width(gr)
            rows.append({"config": tag, "tile": [int(i), int(j)], "n": int(len(cloud)),
                         "psnr": float(psnr(rgb[mk], tgt[mk])),
                         "psnr_common": float(psnr(rgb[common], tgt[common])) if common.sum() > 5000 else float("nan"),
                         "coverage": float(mk.sum() / max(cov_ref, 1)),
                         "gsharp": gsharp(rgb, tgt, common) if common.sum() > 5000 else float("nan"),
                         "edge_width": ew, "edge_width_ref": ew_ref, "n_edges": ne,
                         "common_px": int(common.sum())})
            cv2.imwrite(str(out / f"{tag}_tile_{i}_{j}.png"),
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    import collections

    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["config"]].append(r)
    summary = {}
    for k, v in agg.items():
        summary[k] = {"psnr": float(np.nanmean([r["psnr"] for r in v])),
                      "psnr_common": float(np.nanmean([r["psnr_common"] for r in v])),
                      "coverage": float(np.nanmean([r["coverage"] for r in v])),
                      "gsharp": float(np.nanmean([r["gsharp"] for r in v])),
                      "edge_width": float(np.nanmean([r["edge_width"] for r in v])),
                      "edge_width_ref": float(np.nanmean([r["edge_width_ref"] for r in v])),
                      "n": int(np.mean([r["n"] for r in v])), "tiles": len(v)}
    (out / "metrics.json").write_text(json.dumps({"seq": ds.name, "rows": rows,
                                                  "summary": summary}, indent=2, default=float))
    for k, v in summary.items():
        print(f"[s17] {k:9s} PSNR={v['psnr']:5.2f} (common {v['psnr_common']:5.2f}) "
              f"cov={100*v['coverage']:5.1f}% gsharp={v['gsharp']:.3f} "
              f"edge={v['edge_width']:.1f}px (ref {v['edge_width_ref']:.1f}) N={v['n']:,}")
    print(f"[s17] -> {out}")


if __name__ == "__main__":
    main()
