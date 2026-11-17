"""Stage S20: geometry fusion (head detail + DSM prior) and a DSM-orthorectified reference.

Arms (same tiles, coverage-aware metrics, D11):

* ``plane_dsm`` : ray-DSM placement, full-res colours (the current product baseline)
* ``head_raw``  : the head's Gaussians exactly as predicted
* ``head_ad``   : head depth + ortho adapter, per-pixel placement
* ``fused``     : **DSM surface + clipped head deviation** (smooth geometry with detail)

Two references are computed so the metric bias becomes visible:

* ``plane_ref`` : single-frame warp onto the ground plane (what we used before; it
  flattens roofs, so a *correct* 3D ortho is penalised against it),
* ``dsm_ref``   : single-frame warp **through the DSM** (roofs stay at their height).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch, head_patch, plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.compositor import CompositorConfig, OrthoCompositor
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
    p.add_argument("--clip-m", type=float, default=3.0)
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
    plane_ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    dsm = build_piecewise_dsm(ds, npz, sg, kfs, gsd=0.5)
    print(f"[s20] {ds.name}: DSM {dsm.shape}")

    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)

    patches = {k: {"plane_dsm": [], "head_raw": [], "head_ad": [], "fused": []} for k in [0]}
    arms = {"plane_dsm": {}, "head_raw": {}, "head_ad": {}, "fused": {}}
    for kf in kfs:
        arms["plane_dsm"][kf] = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="iso",
                                            dsm=dsm, ray_dsm=True)
        arms["head_raw"][kf] = head_patch(ds, npz, kf, sg, model, head, device=zcfg.DEVICE)
        arms["head_ad"][kf] = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                              dsm=dsm, depth_source="head",
                                              device=zcfg.DEVICE)
        arms["fused"][kf] = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                            dsm=dsm, depth_source="fused", clip_m=a.clip_m,
                                            device=zcfg.DEVICE)
        print(f"[s20] kf{kf:3d} built 4 arms")

    clouds = {}
    for name, ps in arms.items():
        gmap = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                     commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
        for kf in kfs:
            gmap.ingest(ps[kf], kf)
            gmap.commit(kf)
        clouds[name] = gmap.to_cloud()
        print(f"[s20] {name:9s} N={len(clouds[name]):,}")

    # DSM-orthorectified reference: warp the *middle* frame through the DSM
    comp = OrthoCompositor(ds, gsd, z_ground=0.0, dsm=dsm, cfg=CompositorConfig(max_frames=1))
    T_gt = {int(npz["frame_id"][mid]): sim3_matrix(compose_sim3(T[mid], sg))}

    cam = T_gt[int(npz["frame_id"][mid])][:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S20" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    import cv2

    rows = []
    for n in range(a.tiles):
        i, j = ic + n, jc
        tgt_p, cov_p, _, _ = plane_ref.render_tile(grid, i, j, max_frames=1)
        res = comp.compose(grid, i, j, sorted(T_gt.keys()), T_gt)
        if not res.get("ok"):
            print("[s20] no dsm ref"); continue
        tgt_d = res["rgb"]
        cov = cov_p
        rend = {t: render_tile(c, grid, i, j) for t, c in clouds.items()}
        common_p = np.ones_like(cov > 0.5)
        common_d = np.ones_like(cov > 0.5)
        for t, (rgb, alpha, _, _) in rend.items():
            mk = (alpha > 0.5) & (cov > 0.5)
            common_p &= mk
            common_d &= mk
        common_p &= (cov > 0.5)
        common_d &= (cov > 0.5) & np.isfinite(tgt_d).all(-1)
        for t, (rgb, alpha, _, _) in rend.items():
            mk = (alpha > 0.5) & (cov > 0.5)
            if mk.sum() < 5000:
                continue
            gr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rows.append({"arm": t, "tile": [int(i), int(j)],
                         "psnr_plane": float(psnr(rgb[common_p], tgt_p[common_p])),
                         "psnr_dsm": float(psnr(rgb[common_d], tgt_d[common_d])) if common_d.sum() > 5000 else float("nan"),
                         "coverage": float(mk.sum() / max(int((cov > 0.5).sum()), 1)),
                         "edge_width": edge_width(gr), "n": int(len(clouds[t]))})
            cv2.imwrite(str(out / f"{t}_tile_{i}_{j}.png"),
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"plane_ref_tile_{i}_{j}.png"),
                    (np.clip(tgt_p, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"dsm_ref_tile_{i}_{j}.png"),
                    (np.clip(tgt_d, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["arm"]].append(r)
    summary = {k: {kk: float(np.nanmean([r[kk] for r in v]))
                   for kk in ("psnr_plane", "psnr_dsm", "coverage", "edge_width")}
               | {"n": int(np.mean([r["n"] for r in v]))} for k, v in agg.items()}
    (out / "metrics.json").write_text(json.dumps(
        {"seq": ds.name, "stride": a.stride, "clip_m": a.clip_m, "rows": rows,
         "summary": summary}, indent=2, default=float))
    for k, v in summary.items():
        print(f"[s20] {k:9s} PSNR(plane ref)={v['psnr_plane']:5.2f} PSNR(DSM ref)={v['psnr_dsm']:5.2f} "
              f"cov={100*v['coverage']:5.1f}% edge={v['edge_width']:.1f}px N={v['n']:,}")
    print(f"[s20] -> {out}")


if __name__ == "__main__":
    main()
