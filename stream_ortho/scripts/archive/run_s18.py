"""Stage S18: per-keyframe 3DGS finetuning and its effect on the ortho.

Compares four builds on the same tiles, with **coverage-aware** metrics
(common-mask PSNR + coverage, see results/DECISIONS.md D11):

* ``base``      : plane + piecewise DSM + ray-DSM (positions fixed, no optimisation)
* ``ft``        : same, then 50 Adam iterations per keyframe against that keyframe's
                  own image (colour / opacity / scale / rotation / tiny mean LR)
* ``s15``       : no optimisation, but splat sigma 1.5x (closes the ortho gaps)
* ``ft+s15``    : both
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.backend.finetune import FinetuneConfig, finetune_patch
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian import GaussianPatch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def edge_width(gray: np.ndarray):
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
    return (float(np.median(W)) if W else float("nan"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--scale-factor", type=float, default=1.5)
    p.add_argument("--tiles", type=int, default=2)
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
    print(f"[s18] {ds.name}: DSM {dsm.shape}")

    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))

    # ---- build the patches ----
    patches = {}
    for kf in kfs:
        patches[kf] = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="iso",
                                  dsm=dsm, ray_dsm=True)

    def scaled(p, f):
        return GaussianPatch(means=p.means, quats=p.quats, scales=p.scales * f,
                             opacity=p.opacity, colors=p.colors, owner=p.owner,
                             conf=p.conf, kind=p.kind, theta=p.theta, weight=p.weight)

    def build(tag_patches):
        gmap = GaussianMap(MapConfig(voxel_size=0.3, dedup=True, gate=False,
                                     commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
        for kf in kfs:
            gmap.ingest(tag_patches[kf], kf)
            gmap.commit(kf)
        return gmap.to_cloud()

    clouds = {"base": build(patches)}
    clouds["s15"] = build({k: scaled(v, a.scale_factor) for k, v in patches.items()})

    if a.iters > 0:
        ft, curves = {}, []
        for kf in kfs:
            r = finetune_patch(patches[kf], ds, npz, kf, sg, FinetuneConfig(iters=a.iters))
            curves.append({"kf": int(kf), "before": r["psnr_before"], "after": r["psnr_after"],
                           "scale_before": r["scale_median_before"],
                           "scale_after": r["scale_median_after"]})
            ft[kf] = GaussianPatch(means=r["means"], quats=r["quats"], scales=r["scales"],
                                   opacity=r["opacity"], colors=r["colors"],
                                   owner=int(npz["frame_id"][kf]), conf=patches[kf].conf,
                                   kind=patches[kf].kind, theta=patches[kf].theta,
                                   weight=patches[kf].weight)
            print(f"[s18] kf{kf:3d} keyframe PSNR {r['psnr_before']:5.2f} -> {r['psnr_after']:5.2f} dB")
        clouds["ft"] = build(ft)
        clouds["ft+s15"] = build({k: scaled(v, a.scale_factor) for k, v in ft.items()})
        print(f"[s18] finetune mean keyframe gain: "
              f"{np.mean([c['after'] - c['before'] for c in curves]):+.2f} dB")

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S18" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    import cv2

    rows = []
    for n in range(a.tiles):
        i, j = ic, jc + n
        tgt, cov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
        cov_ref = int((cov > 0.5).sum())
        ew_ref = edge_width(cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
        rend = {t: render_tile(c, grid, i, j) for t, c in clouds.items()}
        common = np.ones_like(cov > 0.5)
        for t, (rgb, alpha, _, _) in rend.items():
            common &= (alpha > 0.5) & (cov > 0.5)
        for t, (rgb, alpha, _, _) in rend.items():
            mk = (alpha > 0.5) & (cov > 0.5)
            if mk.sum() < 5000:
                continue
            gr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rows.append({"config": t, "tile": [int(i), int(j)],
                         "psnr_common": float(psnr(rgb[common], tgt[common])) if common.sum() > 5000 else float("nan"),
                         "coverage": float(mk.sum() / max(cov_ref, 1)),
                         "edge_width": edge_width(gr), "edge_width_ref": ew_ref,
                         "n": int(len(clouds[t]))})
            cv2.imwrite(str(out / f"{t}_tile_{i}_{j}.png"),
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["config"]].append(r)
    summary = {k: {"psnr_common": float(np.nanmean([r["psnr_common"] for r in v])),
                   "coverage": float(np.nanmean([r["coverage"] for r in v])),
                   "edge_width": float(np.nanmean([r["edge_width"] for r in v])),
                   "n": int(np.mean([r["n"] for r in v]))} for k, v in agg.items()}
    (out / "metrics.json").write_text(json.dumps(
        {"seq": ds.name, "iters": a.iters, "scale_factor": a.scale_factor,
         "rows": rows, "summary": summary}, indent=2, default=float))
    for k, v in summary.items():
        print(f"[s18] {k:8s} PSNR(common)={v['psnr_common']:5.2f} cov={100*v['coverage']:5.1f}% "
              f"edge={v['edge_width']:.1f}px N={v['n']:,}")
    print(f"[s18] -> {out}")


if __name__ == "__main__":
    main()
