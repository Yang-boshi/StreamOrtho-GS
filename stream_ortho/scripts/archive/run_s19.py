"""Stage S19: Gaussian-head 3DGS -- raw vs adapter+cleansed depth.

Three arms, same tiles, coverage-aware metrics (D11):

* ``head_raw``  : the head's own Gaussians as predicted (means/scales/quats/opacity/SH)
* ``head_ad``   : same head depth, but the **ortho adapter** (ground-facing flat discs,
                  scale >= 1 px, full-resolution colours, opacity >= 0.95) + depth cleanser
* ``pm_ad``     : the same adapter/cleanser, but the depth comes from the SLAM point map
                  (which carries the real 20-23 m relief) instead of the flat head depth
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch, head_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def edge_width(gray):
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
    p.add_argument("--tiles", type=int, default=2)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--voxel", type=float, default=0.25)
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
    print(f"[s19] {ds.name}: DSM {dsm.shape}")

    from stream_ortho.frontend.gaussian_head import load_gaussian_head

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE,
                              verbose=False)

    arms = {"head_raw": {}, "head_ad": {}, "pm_ad": {}}
    for kf in kfs:
        arms["head_raw"][kf] = head_patch(ds, npz, kf, sg, model, head, device=zcfg.DEVICE)
        arms["head_ad"][kf] = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                              dsm=dsm, depth_source="head",
                                              sigma_px=a.sigma_px, device=zcfg.DEVICE)
        arms["pm_ad"][kf] = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                            dsm=dsm, depth_source="pointmap",
                                            sigma_px=a.sigma_px, device=zcfg.DEVICE)
        print(f"[s19] kf{kf:3d} built 3 arms")

    clouds = {}
    for name, patches in arms.items():
        gmap = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                     commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
        for kf in kfs:
            gmap.ingest(patches[kf], kf)
            gmap.commit(kf)
        clouds[name] = gmap.to_cloud()
        print(f"[s19] {name:9s} N={len(clouds[name]):,}")

    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S19" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    import cv2

    rows = []
    for n in range(a.tiles):
        i, j = ic + n, jc
        tgt, cov, _, _ = ref.render_tile(grid, i, j, max_frames=1)
        cov_ref = int((cov > 0.5).sum())
        ew_ref = edge_width(cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
        # DSM height per pixel (for the building/ground split)
        uu, vv = np.meshgrid(np.arange(512), np.arange(512))
        P = grid.px_to_plane(uu, vv, i, j, z=0.0)
        zz = dsm.lookup(P[..., 0], P[..., 1], fill=0.0)
        rend = {t: render_tile(c, grid, i, j) for t, c in clouds.items()}
        common = np.ones_like(cov > 0.5)
        for t, (rgb, alpha, _, _) in rend.items():
            common &= (alpha > 0.5) & (cov > 0.5)
        bld = common & (zz > 8.0)
        grd = common & (zz < 2.0)
        for t, (rgb, alpha, _, _) in rend.items():
            mk = (alpha > 0.5) & (cov > 0.5)
            if mk.sum() < 5000:
                continue
            gr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rows.append({"arm": t, "tile": [int(i), int(j)],
                         "psnr_common": float(psnr(rgb[common], tgt[common])),
                         "psnr_building": float(psnr(rgb[bld], tgt[bld])) if bld.sum() > 3000 else float("nan"),
                         "psnr_ground": float(psnr(rgb[grd], tgt[grd])) if grd.sum() > 3000 else float("nan"),
                         "coverage": float(mk.sum() / max(cov_ref, 1)),
                         "edge_width": edge_width(gr), "edge_width_ref": ew_ref,
                         "n": int(len(clouds[t]))})
            cv2.imwrite(str(out / f"{t}_tile_{i}_{j}.png"),
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"reference_tile_{i}_{j}.png"),
                    (np.clip(tgt, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["arm"]].append(r)
    summary = {k: {kk: float(np.nanmean([r[kk] for r in v]))
                   for kk in ("psnr_common", "psnr_building", "psnr_ground", "coverage", "edge_width")}
               | {"n": int(np.mean([r["n"] for r in v]))} for k, v in agg.items()}
    (out / "metrics.json").write_text(json.dumps(
        {"seq": ds.name, "stride": a.stride, "sigma_px": a.sigma_px, "voxel": a.voxel,
         "rows": rows, "summary": summary}, indent=2, default=float))
    for k, v in summary.items():
        print(f"[s19] {k:9s} PSNR={v['psnr_common']:5.2f} (bld {v['psnr_building']:5.2f} / grd "
              f"{v['psnr_ground']:5.2f}) cov={100*v['coverage']:5.1f}% edge={v['edge_width']:.1f}px N={v['n']:,}")
    print(f"[s19] -> {out}")


if __name__ == "__main__":
    main()
