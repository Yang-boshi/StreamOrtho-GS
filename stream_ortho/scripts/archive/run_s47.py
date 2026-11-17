"""S47 -- product-side parameter sweep for sharpness (build once, render many).

The map is built once (point-map placement, budget, NFP, A'), then the *product* knobs
are swept without rebuilding:

* ``cell``       : 2.5D selection cell (0.30 / 0.45 m) -- must not be coarser than the
                   output GSD or the texture is undersampled,
* ``sigma``      : splat size in metres (0.28 / 0.35 / 0.45),
* ``cell_color`` : average the colours of all samples inside a cell (C).

Sharpness is measured as the mean gradient magnitude on a dense-building tile and on a
ground tile (both relative to the same reference), plus the saved crops for eyeballing.
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
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8
from stream_ortho.map.build import head_adapt_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.selection import (mean_colors_by_cell, quality_score,
                                        select_by_cell)
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic


def cloud_of(M, Q, SC, OP, CO, device="cuda:0"):
    import torch

    d = torch.device(device)
    return GaussianCloud(means=torch.as_tensor(np.ascontiguousarray(M), device=d),
                         quats=torch.as_tensor(np.ascontiguousarray(Q), device=d),
                         scales=torch.as_tensor(np.ascontiguousarray(SC), device=d),
                         opacity=torch.as_tensor(np.ascontiguousarray(OP), device=d),
                         colors=torch.as_tensor(np.ascontiguousarray(CO), device=d),
                         owner=torch.zeros(len(M), dtype=torch.long, device=d), meta={})


def sharp(g):
    import cv2

    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max-kf", type=int, default=30)
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rd = pathlib.Path(a.run)
    info = json.loads((rd / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(rd / "keyframes.npz")
    m = json.loads((rd / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()[: a.max_kf]
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    gm = GaussianMap(MapConfig(voxel_size=0.25, dedup=True, gate=False, commit_min_obs=2,
                               commit_max_age=2, fusion="weighted"), ds.T_WP)
    for kf in kfs:
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=2,
                                 depth_source="pointmap", sigma_m=0.35, conf_min=0.2,
                                 edge_reject=2, edge_opacity=0.0, device=zcfg.DEVICE)
        gm.ingest(patch, kf)
        gm.commit(kf)
    live = gm.committed["dead"] == 0
    c = gm.committed
    M = c["means"][live]
    Q = c["quats"][live]
    SC = c["scales"][live]
    OP = np.clip(c["opacity"][live], 0.95, 1.0)
    CO = c["colors"][live]
    q = quality_score(conf=c["conf"][live], opacity=c["opacity"][live],
                      wsum=c["wsum"][live], n_obs=c["n_obs"][live])
    print(f"[s47] built {gm.n_committed:,} from {len(kfs)} keyframes")
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
    ref = PlaneReferenceMosaic(ds, a.gsd, 0.0, stride=1)
    P = (M - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    st = grid.tile * grid.gsd
    keys = np.floor(P[:, :2] / st).astype(np.int64)
    uq, cn = np.unique(keys, axis=0, return_counts=True)
    order = np.argsort(cn)[::-1]
    tiles = [(int(uq[o][0]), int(uq[o][1])) for o in order[:4]]
    import cv2

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S47" / rd.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in (0.30, 0.45):
        idx = np.where(live)[0]
        sel = idx[select_by_cell(M, ds.T_WP, cell=cell, q=q, prefer="quality")]
        for cc in (False, True):
            CO2 = mean_colors_by_cell(c["means"][sel], c["colors"][sel], ds.T_WP, cell=cell) \
                if cc else c["colors"][sel]
            for sig_scale in (0.62, 0.78, 1.0):
                sc2 = c["scales"][sel].copy()
                # rescale sigma so the *output* footprint is sig_scale x 1 pixel
                base = float(np.median(sc2[:, 0]))
                sig = a.gsd * sig_scale
                sc2[:, :2] *= sig / max(base, 1e-6)
                cloud = cloud_of(c["means"][sel], c["quats"][sel], sc2,
                                 np.clip(c["opacity"][sel], 0.95, 1.0), CO2)
                sh, ps, covs = [], [], []
                for (i, j) in tiles:
                    rgb, al, _, _ = render_tile(cloud, grid, i, j)
                    tgt, cv_, _, _ = ref.render_tile(grid, i, j, max_frames=1)
                    mk = (al > 0.5) & (cv_ > 0.5)
                    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2GRAY).astype(np.float32)
                    sh.append(sharp(g))
                    covs.append(float((al > 0.5).mean()))
                    if mk.sum() > 5000:
                        ps.append(psnr(rgb[mk], tgt[mk]))
                row = {"cell": cell, "cell_color": cc, "sigma": round(sig, 3),
                       "sigma_px": sig_scale, "n": int(len(sel)),
                       "sharp": float(np.mean(sh)), "coverage": float(np.mean(covs)),
                       "psnr": float(np.mean(ps)) if ps else None}
                rows.append(row)
                tag = f"cell{cell}_sig{sig:.2f}_cc{int(cc)}"
                i, j = tiles[0]
                rgb, al, _, _ = render_tile(cloud, grid, i, j)
                cv2.imwrite(str(out / f"{tag}.png"),
                            (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
                print(f"[s47] cell {cell} cc={int(cc)} sigma {sig:.2f} ({sig_scale}px) "
                      f"N={len(sel):,} sharp {row['sharp']:.2f} cov {100*row['coverage']:.1f}% "
                      f"psnr {row['psnr']}")
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
