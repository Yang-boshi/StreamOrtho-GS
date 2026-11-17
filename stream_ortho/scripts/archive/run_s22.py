"""P1: region-level selection (design doc §2.7, decision D12).

Instead of merging two independently predicted sub-maps by voxel averaging -- which
never fires because their overlap is 0-2% -- we compare the **old map content** and
the **new keyframe's content** inside each ortho tile and keep one of them.

Scores (reference-free, i.e. computable online without ground truth):

* ``sharp``  : mean gradient magnitude inside the valid area (detail),
* ``agree``  : mean |render - median(other frames' DSM-warps)| (cross-view consistency),
* ``cover``  : valid fraction of the tile,
* ``layer``  : standard deviation of plane-z inside 1 m cells (lower = less layering).

``score = sharp_norm * cover - w_agree * agree_norm - w_layer * layer_norm``

The script reports, for a set of tiles, the per-keyframe scores, the chosen keyframe,
and the resulting metrics of

* ``stacked`` : all keyframes appended (what today's map does),
* ``selected``: per tile, only the winning keyframe's splats,
* ``plane_dsm``: the ray-DSM baseline for reference.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch, plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.compositor import CompositorConfig, OrthoCompositor
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def grad_mean(rgb, mask):
    import cv2

    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    return float(np.hypot(gx, gy)[mask].mean()) if mask.any() else 0.0


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


def layer_metric(means, T_WP, cell: float = 1.0, z_max: float = 2.0) -> float:
    """Mean per-cell std of plane-z over the **ground** (z < ``z_max``).

    Restricting to the ground matters: inside a 1 m cell a building contributes
    several legitimate heights (wall + roof), which is real 3-D structure, not
    duplicated layers.  Only the ground should be single-layered.
    """
    p = (np.asarray(means, np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
    p = p[(p[:, 2] > -20.0) & (p[:, 2] < z_max)]
    if len(p) == 0:
        return 0.0
    key = np.floor(p[:, :2] / cell).astype(np.int64)
    order = np.lexsort((p[:, 2], key[:, 1], key[:, 0]))
    k_s = key[order]
    z_s = p[order, 2]
    _, start, counts = np.unique(k_s, axis=0, return_index=True, return_counts=True)
    sel = counts >= 2
    if not sel.any():
        return 0.0
    stds = [float(z_s[st:st + c].std()) for st, c in zip(start[sel], counts[sel])][:20000]
    return float(np.mean(stds))


def subset_cloud(patch, device="cuda:0") -> GaussianCloud:
    import torch

    return GaussianCloud(
        means=torch.as_tensor(patch.means, device=device),
        quats=torch.as_tensor(patch.quats, device=device),
        scales=torch.as_tensor(patch.scales, device=device),
        opacity=torch.as_tensor(patch.opacity, device=device),
        colors=torch.as_tensor(patch.colors, device=device),
        owner=torch.as_tensor(np.asarray(patch.owner).reshape(-1), device=device),
        meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--w-agree", type=float, default=0.5)
    p.add_argument("--w-layer", type=float, default=0.5)
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
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()[:8]
    mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(T[mid], sg))
    depth = float(np.median(npz["X_canon"][mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth / float(ds.K_orig[0, 0])
    grid = OrthoGrid(ds.T_WP, gsd, tile=512, halo=32, z_ground=0.0, z_top=800.0)
    plane_ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    dsm = build_piecewise_dsm(ds, npz, sg, kfs[:4], gsd=0.5)
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)

    # per-keyframe patches
    patches = {kf: head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                    dsm=dsm, depth_source="head", device=zcfg.DEVICE)
               for kf in kfs}
    print(f"[p1] built {len(patches)} patches")

    # stacked map (today's behaviour)
    gmap = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    for kf in kfs:
        gmap.ingest(patches[kf], kf)
        gmap.commit(kf)
    stacked = gmap.to_cloud()
    print(f"[p1] stacked: N={len(stacked):,} layer={layer_metric(stacked.means.detach().cpu().numpy(), ds.T_WP):.2f} m")

    # DSM-based candidates for the cross-view agreement score
    comp = OrthoCompositor(ds, gsd, z_ground=0.0, dsm=dsm, cfg=CompositorConfig(max_frames=6))
    T_gt = {int(npz["frame_id"][k]): sim3_matrix(compose_sim3(T[k], sg)) for k in kfs}
    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S22" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2

    rows = []
    for n in range(a.tiles):
        i, j = ic, jc + n
        cands = comp.candidates(grid, i, j, sorted(T_gt.keys()), T_gt)
        if len(cands) < 2:
            continue
        stack = np.stack([c["rgb"] for c in cands], 0)
        valid = np.stack([c["valid"] for c in cands], 0)
        with np.errstate(all="ignore"):
            cons = np.nanmedian(np.where(valid[..., None], stack, np.nan), axis=0)
        ref_ok = valid.any(0) & np.isfinite(cons).all(-1)
        tgt_p, cov_p, _, _ = plane_ref.render_tile(grid, i, j, max_frames=1)

        # score every keyframe's own content in this tile
        scores = []
        for kf in kfs:
            rgb, alpha, _, _ = render_tile(subset_cloud(patches[kf]), grid, i, j)
            msk = (alpha > 0.5) & ref_ok
            if msk.sum() < 5000:
                continue
            sharp = grad_mean(rgb, msk)
            agree = float(np.abs(rgb[msk] - cons[msk]).mean())
            layer = layer_metric(patches[kf].means, ds.T_WP)
            scores.append({"kf": int(kf), "sharp": sharp, "agree": agree,
                           "cover": float(msk.sum() / max(ref_ok.sum(), 1)), "layer": layer})
        if not scores:
            continue
        sh = np.array([s["sharp"] for s in scores]); ag = np.array([s["agree"] for s in scores])
        co = np.array([s["cover"] for s in scores]); la = np.array([s["layer"] for s in scores])
        norm = lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-9)
        sc_arr = norm(sh) * 0.5 + co - a.w_agree * norm(ag) - a.w_layer * norm(la)
        best = int(np.argmax(sc_arr))
        sel_kf = scores[best]["kf"]

        # render the three variants
        rgb_st, al_st, _, _ = render_tile(stacked, grid, i, j)
        rgb_sel, al_sel, _, _ = render_tile(subset_cloud(patches[sel_kf]), grid, i, j)
        pd = plane_patch(ds, npz, sel_kf, sg, stride=a.stride, shape="iso", dsm=dsm, ray_dsm=True)
        rgb_pd, al_pd, _, _ = render_tile(subset_cloud(pd), grid, i, j)

        for tag, rgb, al, means in (("stacked", rgb_st, al_st, stacked.means.detach().cpu().numpy()),
                                    ("selected", rgb_sel, al_sel, patches[sel_kf].means),
                                    ("plane_dsm", rgb_pd, al_pd, pd.means)):
            msk = (al > 0.5) & (cov_p > 0.5)
            if msk.sum() < 5000:
                continue
            gr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rows.append({"variant": tag, "tile": [int(i), int(j)], "kf": int(sel_kf),
                         "psnr_plane": float(psnr(rgb[msk], tgt_p[msk])),
                         "edge_width": edge_width(gr), "coverage": float(al[msk].mean()),
                         "layer": layer_metric(means, ds.T_WP), "n": int(len(means))})
            cv2.imwrite(str(out / f"{tag}_tile_{i}_{j}.png"),
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"reference_tile_{i}_{j}.png"),
                    (np.clip(tgt_p, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        best_row = [s for s in scores if s["kf"] == sel_kf][0]
        print(f"[p1] tile ({i},{j}) chosen kf{sel_kf}  "
              f"sharp={best_row['sharp']:.1f} agree={best_row['agree']:.1f} "
              f"cover={best_row['cover']:.2f} layer={best_row['layer']:.2f}  "
              f"(of {len(scores)} candidates)")

    import collections

    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["variant"]].append(r)
    summary = {k: {"psnr_plane": float(np.nanmean([r["psnr_plane"] for r in v])),
                   "edge_width": float(np.nanmean([r["edge_width"] for r in v])),
                   "coverage": float(np.nanmean([r["coverage"] for r in v])),
                   "layer": float(np.nanmean([r["layer"] for r in v])),
                   "n": int(np.mean([r["n"] for r in v]))} for k, v in agg.items()}
    (out / "metrics.json").write_text(json.dumps({"seq": ds.name, "rows": rows,
                                                  "summary": summary}, indent=2, default=float))
    for k, v in summary.items():
        print(f"[p1] {k:9s} PSNR={v['psnr_plane']:5.2f} edge={v['edge_width']:.1f}px "
              f"cov={100*v['coverage']:5.1f}% layer={v['layer']:.2f} m N={v['n']:,}")
    print(f"[p1] -> {out}")


if __name__ == "__main__":
    main()
