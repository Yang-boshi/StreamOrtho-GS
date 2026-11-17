"""Where does the ortho blur come from? A 3-way decomposition (diagnostic).

Builds a **single** keyframe with no downsampling and compares it with its own
single-frame plane warp: that isolates the renderer.  Then sweeps

1. ``stride``      (1, 2, 4, 8)   -> quantization / sampling density
2. number of keyframes (1, 2, 4, 8) at a fixed stride -> merging / registration

and reports PSNR / SSIM / sharpness (mean |grad| ratio vs the reference) for
each.  This separates "the renderer is blurry" from "we throw away resolution"
and from "we average misregistered frames".

Usage::

    python -m stream_ortho.scripts.diagnose_blur --run <S2 run dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr, ssim
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def cruise_keyframes(npz, n_max):
    T = npz["T_WC"]
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(len(T))
    cruise = np.where(np.abs(np.log(s / np.median(s))) < np.log(1.3))[0]
    return (cruise if len(cruise) else np.arange(len(T)))[:n_max].tolist()


def sharp_ratio(img, ref, mask):
    import cv2

    def grad(x):
        g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))

    return float(grad(img)[mask].mean() / max(grad(ref)[mask].mean(), 1e-9))


def measure(patch_or_cloud, grid, i, j, ref_rgb, ref_cov, label, out):
    import cv2

    if hasattr(patch_or_cloud, "to_cloud"):
        cloud = patch_or_cloud.to_cloud()
    else:
        cloud = patch_or_cloud
    rgb, alpha, _, _ = render_tile(cloud, grid, i, j)
    m = (alpha > 0.5) & (ref_cov > 0.5)
    ps = psnr(rgb[m], ref_rgb[m]) if m.sum() > 5000 else float("nan")
    if m.sum() > 5000:
        ys, xs = np.nonzero(m)
        ss = ssim(rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                  ref_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    else:
        ss = float("nan")
    sh = sharp_ratio(rgb, ref_rgb, m)
    row = {"label": label, "psnr": float(ps), "ssim": float(ss),
           "sharpness": sh, "coverage": float((alpha > 0.5).mean())}
    print(f"  {label:26s} PSNR={ps:6.2f}  SSIM={ss:.3f}  sharpness={sh:.3f}")
    out.append(row)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description="decompose the ortho blur")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=zcfg.DEVICE)
    a = p.parse_args(argv)

    run = pathlib.Path(a.run)
    info = json.loads((run / "run.json").read_text())
    ds = NPUDroneMapDataset(pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"])
    npz = np.load(run / "keyframes.npz")
    m = json.loads((run / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    kfs = cruise_keyframes(npz, 8)
    kf_mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(npz["T_WC"][kf_mid], sg))
    depth_m = float(np.median(npz["X_canon"][kf_mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth_m / float(ds.K_orig[0, 0])

    grid = OrthoGrid(ds.T_WP, gsd, tile=a.tile, halo=a.halo, z_ground=0.0, z_top=800.0)
    cam_w = sim3_matrix(compose_sim3(npz["T_WC"][kf_mid], sg))[:3, 3]
    ppl = (cam_w - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    i, j = grid.tile_of(float(ppl[0]), float(ppl[1]))

    # (A) reference from exactly ONE keyframe: isolates the renderer from merging
    ref_single = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1,
                                      frame_indices=[int(npz["frame_id"][kf_mid])])
    ref1, cov1, _, _ = ref_single.render_tile(grid, i, j, max_frames=1)
    # (B) reference from the 3 nearest frames (what the rest of the project uses)
    ref3 = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    ref3, cov3, _, _ = ref3.render_tile(grid, i, j, max_frames=3)
    print(f"[diag] tile ({i},{j}) gsd={gsd:.4f} m/px  depth={depth_m:.1f} m  kf={kf_mid}")

    rows = []
    print("\n(A) ONE keyframe, no merging -- isolates the renderer + the splat model")
    for st in (1, 2, 4, 8):
        patch = plane_patch(ds, npz, kf_mid, sg, stride=st, shape="iso", device=a.device)
        gm = GaussianMap(MapConfig(voxel_size=0.05, dedup=False, gate=False,
                                   commit_min_obs=1, commit_max_age=0), ds.T_WP)
        gm.ingest(patch, kf_mid)
        gm.commit(kf_mid)
        measure(gm, grid, i, j, ref1, cov1, f"1 kf, stride {st} ({depth_m / ds.K_orig[0,0]*st:.2f} m/pt)", rows)

    print("\n(B) N keyframes merged at stride 2 -- adds registration + averaging")
    for n in (1, 2, 4, 8):
        gm = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                   commit_min_obs=1, commit_max_age=0), ds.T_WP)
        for kf in kfs[:n]:
            gm.ingest(plane_patch(ds, npz, kf, sg, stride=2, shape="iso", device=a.device), kf)
            gm.commit(kf)
        measure(gm, grid, i, j, ref3, cov3, f"{n} kf merged, stride 2", rows)

    print("\n(D) ONE keyframe, stride 1 -- splat-size sweep (the analytic prior)")
    import stream_ortho.map.build as _build

    for factor in (0.7, 1.0, 1.25, 1.5, 2.0):
        _build.SPLAT_RADIUS_FACTOR = factor
        patch = plane_patch(ds, npz, kf_mid, sg, stride=1, shape="iso", device=a.device)
        gm = GaussianMap(MapConfig(voxel_size=0.05, dedup=False, gate=False,
                                   commit_min_obs=1, commit_max_age=0), ds.T_WP)
        gm.ingest(patch, kf_mid)
        gm.commit(kf_mid)
        measure(gm, grid, i, j, ref1, cov1, f"1 kf, stride 1, radius {factor:.2f}x", rows)
    _build.SPLAT_RADIUS_FACTOR = 1.5

    print("\n(C) reference comparison: single-frame warp vs 3-frame warp")
    msk = (cov1 > 0.5) & (cov3 > 0.5)
    print(f"  single-frame vs 3-frame reference: PSNR={psnr(ref1[msk], ref3[msk]):.2f} dB "
          f"(a proxy for how much the reference itself moves with frame count)")

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S7b" / "blur_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.name}.json").write_text(json.dumps(
        {"run": run.name, "tile": [int(i), int(j)], "gsd": gsd, "depth_m": depth_m,
         "rows": rows}, indent=2))
    print(f"\n[diag] -> {out / (run.name + '.json')}")


if __name__ == "__main__":
    main()
