"""S23: streaming region-level selection (P1 wired into the main loop) + metrics.

Per keyframe (flight order):

1. build the keyframe's sub-map patch,
2. ingest it into the streaming map (so the map is complete for the final render),
3. for every tile it covers, score the new patch against the **current owner** and
   take over the tile when the new content wins,

then the final ortho is rendered **per tile from its owner's Gaussians** (strategy C),
and compared against

* ``stacked``  : all keyframes rendered together (today's behaviour),
* ``plane_dsm``: the ray-DSM path for the same owner keyframes.

Reported metrics: PSNR / edge width / coverage / ground-layer std, plus a
**low-frequency PSNR** (both images blurred, sigma=8 px) that is sensitive to the
exposure banding caused by stacking frames with different brightness.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import head_adapt_patch, plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.selection import SelectionConfig, SelectionState, layer_metric
from stream_ortho.ortho.compositor import CompositorConfig, OrthoCompositor
from stream_ortho.ortho.dsm import build_piecewise_dsm
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def grad_mean(rgb, mask):
    import cv2

    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                          cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))[mask].mean()) if mask.any() else 0.0


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


def lowfreq_psnr(a, b, mask, sigma=8.0):
    import cv2

    aa = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
    bb = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigma)
    return float(psnr(aa[mask], bb[mask]))


def cloud_of(patch_list, device="cuda:0"):
    import torch

    def cat(k, dim=1):
        return np.concatenate([getattr(p, k).reshape(len(p), -1) if getattr(p, k).ndim > 1
                               else getattr(p, k).reshape(len(p), 1) for p in patch_list], 0)
    n = sum(len(p) for p in patch_list)
    dev = torch.device(device)
    return GaussianCloud(
        means=torch.as_tensor(cat("means"), device=dev),
        quats=torch.as_tensor(cat("quats"), device=dev),
        scales=torch.as_tensor(cat("scales"), device=dev),
        opacity=torch.as_tensor(cat("opacity").reshape(-1), device=dev),
        colors=torch.as_tensor(cat("colors"), device=dev),
        owner=torch.zeros(n, dtype=torch.long, device=dev), meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.25)
    p.add_argument("--tiles", default="3x3")
    p.add_argument("--w-agree", type=float, default=0.5)
    p.add_argument("--w-layer", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.0)
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

    rows, cols = (int(x) for x in a.tiles.lower().split("x"))
    cam = sim3_matrix(compose_sim3(T[mid], sg))[:3, 3]
    ppl = (cam - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(ppl[0]), float(ppl[1]))
    i0, j0 = ic - cols // 2, jc - rows // 2
    tile_list = [(i0 + di, j0 + dj) for dj in range(rows) for di in range(cols)]
    print(f"[s23] block ({i0},{j0}) {cols}x{rows} -> {len(tile_list)} tiles")

    # cross-view consensus per tile (for the agreement score)
    comp = OrthoCompositor(ds, gsd, z_ground=0.0, dsm=dsm, cfg=CompositorConfig(max_frames=6))
    T_gt = {int(npz["frame_id"][k]): sim3_matrix(compose_sim3(T[k], sg)) for k in kfs}
    cons = {}
    for (i, j) in tile_list:
        cands = comp.candidates(grid, i, j, sorted(T_gt.keys()), T_gt)
        if len(cands) < 2:
            continue
        stack = np.stack([c["rgb"] for c in cands], 0)
        valid = np.stack([c["valid"] for c in cands], 0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(valid[..., None], stack, np.nan), axis=0)
        cons[(i, j)] = (med, valid.any(0) & np.isfinite(med).all(-1))
    print(f"[s23] consensus available for {len(cons)}/{len(tile_list)} tiles")

    cfg = SelectionConfig(w_sharp=0.5, w_agree=a.w_agree, w_layer=a.w_layer,
                          accept_margin=a.margin)
    state = SelectionState()
    gmap = GaussianMap(MapConfig(voxel_size=a.voxel, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2, fusion="weighted"), ds.T_WP)
    patches = {}
    t_loop = time.time()
    for n, kf in enumerate(kfs):
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 dsm=dsm, depth_source="head", device=zcfg.DEVICE)
        patches[kf] = patch
        gmap.ingest(patch, kf)
        gmap.commit(kf)
        # score this keyframe on every tile it can see
        p_plane = (patch.means - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        lay = layer_metric(patch.means, ds.T_WP)
        for (i, j) in tile_list:
            rgb, alpha, _, _ = render_tile(cloud_of([patch]), grid, i, j)
            if (i, j) in cons:
                med, ok = cons[(i, j)]
            else:
                ok = alpha > 0.5
                med = rgb
            msk = (alpha > 0.5) & ok
            if msk.sum() < cfg.min_px:
                continue
            sharp = grad_mean(rgb, msk)
            agree = float(np.abs(rgb[msk] - med[msk]).mean())
            cover = float(msk.sum() / max(int(ok.sum()), 1))
            score = 0.5 * (sharp / 30.0) + cover - cfg.w_agree * agree - cfg.w_layer * lay
            if (i, j) not in state.owner:
                state.owner[(i, j)] = int(kf)
                state.score[(i, j)] = score
                state.first[(i, j)] = int(kf)
            elif score > state.score[(i, j)] + cfg.accept_margin:
                state.swaps.append(((i, j), state.owner[(i, j)], int(kf), float(score)))
                state.owner[(i, j)] = int(kf)
                state.score[(i, j)] = score
        print(f"[s23] kf{n+1}/{len(kfs)} (id {int(npz['frame_id'][kf])})  owners={len(state.owner)}  "
              f"swaps={len(state.swaps)}  N={gmap.n_committed:,}")
    print(f"[s23] streaming loop {time.time()-t_loop:.0f} s; owners={len(state.owner)}, swaps={len(state.swaps)}")

    # ---- final renders ----
    stacked = gmap.to_cloud()
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S23" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    import cv2

    # per-owner patches, and the plane_dsm version of the same owners
    pd_cache = {}
    for kf in sorted(set(state.owner.values())):
        pd_cache[kf] = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="iso",
                                   dsm=dsm, ray_dsm=True)
    results = collections.defaultdict(dict)
    for (i, j) in tile_list:
        if (i, j) not in state.owner:
            continue
        owner = state.owner[(i, j)]
        tgt, cov, _, _ = plane_ref.render_tile(grid, i, j, max_frames=1)
        msk_ref = cov > 0.5
        variants = {
            "stacked": (render_tile(stacked, grid, i, j), stacked.means.detach().cpu().numpy()),
            "selected": (render_tile(cloud_of([patches[owner]]), grid, i, j), patches[owner].means),
            "plane_dsm": (render_tile(cloud_of([pd_cache[owner]]), grid, i, j), pd_cache[owner].means),
        }
        for tag, ((rgb, alpha, _, _), means) in variants.items():
            msk = (alpha > 0.5) & msk_ref
            if msk.sum() < cfg.min_px:
                continue
            gr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            results[tag][(i, j)] = {
                "psnr": float(psnr(rgb[msk], tgt[msk])),
                "psnr_low": lowfreq_psnr(rgb, tgt, msk),
                "edge_width": edge_width(gr),
                "coverage": float(msk.sum() / max(int(msk_ref.sum()), 1)),
                "layer": layer_metric(means, ds.T_WP),
                "owner": int(owner), "n": int(len(means)),
            }
            if tag != "stacked":
                cv2.imwrite(str(out / f"{tag}_tile_{i}_{j}.png"),
                            (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"stacked_tile_{i}_{j}.png"),
                    (np.clip(variants["stacked"][0][0], 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(str(out / f"reference_tile_{i}_{j}.png"),
                    (np.clip(tgt, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    summary = {}
    for tag, per_tile in results.items():
        summary[tag] = {k: float(np.nanmean([v[k] for v in per_tile.values()]))
                        for k in ("psnr", "psnr_low", "edge_width", "coverage", "layer")}
        summary[tag]["n"] = int(np.mean([v["n"] for v in per_tile.values()]))
        summary[tag]["tiles"] = len(per_tile)
    (out / "metrics.json").write_text(json.dumps(
        {"seq": ds.name, "tiles": [[int(i), int(j)] for i, j in tile_list],
         "owner": {f"{i},{j}": int(k) for (i, j), k in state.owner.items()},
         "swaps": [[list(t), int(o), int(nn), float(s)] for t, o, nn, s in state.swaps],
         "summary": summary}, indent=2, default=float))
    for tag, v in summary.items():
        print(f"[s23] {tag:9s} PSNR={v['psnr']:5.2f} (low-freq {v['psnr_low']:5.2f}) "
              f"edge={v['edge_width']:.1f}px cov={100*v['coverage']:5.1f}% "
              f"layer={v['layer']:.2f} m N={v['n']:,} tiles={v['tiles']}")
    print(f"[s23] -> {out}")


if __name__ == "__main__":
    main()
