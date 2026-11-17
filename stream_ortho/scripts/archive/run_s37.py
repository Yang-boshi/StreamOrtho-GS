"""S37 -- NFP veto A/B (the part of NADIR-GS we keep) + throughput (FPS).

Policies:

* ``baseline`` -- no veto (what we had),
* ``nfp``      -- nadir free-space prior veto in ``GaussianMap.ingest``.

Reported together (a single number can always be gamed): floating-new-cell fraction,
per-frame point growth, drift p95 of already-stable cells, coverage, layer, PSNR, the
veto counters, and **FPS** (both the streaming step and the ortho product).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model, psnr
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8
from stream_ortho.map.build import head_adapt_patch
from stream_ortho.map.consistency import (cell_surface, drift_metric, floating_new_cells,
                                          fsc_heldout)
from stream_ortho.map.freespace import NFPConfig, NadirHeightField
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.selection import layer_metric, quality_score, select_by_cell
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.tiles import Tile, stitch


def make_cloud(M, Q, SC, OP, CO, device="cuda:0"):
    import torch

    d = torch.device(device)
    return GaussianCloud(means=torch.as_tensor(np.ascontiguousarray(M), device=d),
                         quats=torch.as_tensor(np.ascontiguousarray(Q), device=d),
                         scales=torch.as_tensor(np.ascontiguousarray(SC), device=d),
                         opacity=torch.as_tensor(np.ascontiguousarray(OP), device=d),
                         colors=torch.as_tensor(np.ascontiguousarray(CO), device=d),
                         owner=torch.zeros(len(M), dtype=torch.long, device=d), meta={})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gsd", type=float, default=0.30)
    p.add_argument("--sigma", type=float, default=0.45)
    p.add_argument("--cell", type=float, default=0.45)
    p.add_argument("--nfp-cell", type=float, default=1.0)
    p.add_argument("--nfp-eps", type=float, default=1.0)
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--max-kf", type=int, default=12)
    p.add_argument("--stable-after", type=int, default=5)
    p.add_argument("--policies", default="baseline,nfp")
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
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()
    kfs_all = list(kfs)
    kfs = kfs[: a.max_kf]
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)

    # a keyframe **not** ingested: used only to measure free-space contamination
    held = [k for k in kfs_all if k not in kfs]
    held_kf = held[len(held) // 2] if held else None

    def sim3_fn(k):
        return sim3_from_vec8(compose_sim3(T[k], sg))
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S37" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in [x.strip() for x in a.policies.split(",") if x.strip()]:
        gm = GaussianMap(MapConfig(voxel_size=0.25, dedup=True, gate=False,
                                   commit_min_obs=2, commit_max_age=2, fusion="weighted"),
                         ds.T_WP)
        if policy == "nfp":
            gm.nfp = NadirHeightField(NFPConfig(cell=a.nfp_cell, eps=a.nfp_eps), ds.T_WP)
        snap = None
        growth, steps = [], []
        t_all = time.time()
        for n, kf in enumerate(kfs):
            t0 = time.perf_counter()
            patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                     depth_source="pointmap", sigma_m=a.sigma,
                                     conf_min=a.conf_min, device=zcfg.DEVICE)
            t_patch = time.perf_counter() - t0
            t1 = time.perf_counter()
            gm.ingest(patch, kf)
            gm.commit(kf)
            t_map = time.perf_counter() - t1
            growth.append(int(gm.n_committed))
            steps.append({"paste": round(1000 * t_patch, 1), "map": round(1000 * t_map, 1)})
            if n + 1 == a.stable_after:
                live = gm.committed["dead"] == 0
                q = quality_score(conf=gm.committed["conf"][live],
                                  opacity=gm.committed["opacity"][live],
                                  wsum=gm.committed["wsum"][live],
                                  n_obs=gm.committed["n_obs"][live])
                k_, z_, _ = cell_surface(gm.committed["means"][live], ds.T_WP, cell=0.5,
                                         q=q, mode="quality")
                snap = {int(x): float(y) for x, y in zip(k_, z_)}
        t_build = time.time() - t_all
        live = gm.committed["dead"] == 0
        M = gm.committed["means"][live]
        q = quality_score(conf=gm.committed["conf"][live], opacity=gm.committed["opacity"][live],
                          wsum=gm.committed["wsum"][live], n_obs=gm.committed["n_obs"][live])
        idx = np.where(live)[0]
        sel = idx[select_by_cell(M, ds.T_WP, cell=a.cell, q=q, prefer="quality")]
        c = gm.committed
        cloud = make_cloud(c["means"][sel], c["quats"][sel], c["scales"][sel],
                           np.clip(c["opacity"][sel], 0.95, 1.0), c["colors"][sel])
        k2, z2, _ = cell_surface(c["means"][sel], ds.T_WP, cell=0.5, mode="quality")
        after = {int(x): float(y) for x, y in zip(k2, z2)}
        drift = drift_metric(snap or {}, after) if snap else {"n": 0, "p95_dz": None}
        flo = floating_new_cells(snap or {}, after)

        Pp = (c["means"][sel] - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        i0, j0 = grid.tile_of(float(Pp[:, 0].min()), float(Pp[:, 1].min()))
        i1, j1 = grid.tile_of(float(Pp[:, 0].max()), float(Pp[:, 1].max()))
        i0, i1, j0, j1 = min(i0, i1), max(i0, i1), min(j0, j1), max(j0, j1)
        t_r = time.perf_counter()
        tiles = []
        for jj in range(j0, j1 + 1):
            for ii in range(i0, i1 + 1):
                rgb, alpha, _, _ = render_tile(cloud, grid, ii, jj)
                tiles.append(Tile(ii, jj, rgb, alpha))
        mosaic, cov = stitch(tiles, shape=(j1 - j0 + 1, i1 - i0 + 1))
        t_render = time.perf_counter() - t_r
        n_tiles = (i1 - i0 + 1) * (j1 - j0 + 1)
        ref = PlaneReferenceMosaic(ds, a.gsd, 0.0, stride=1)
        ps = []
        for jj in range(j0, min(j0 + 3, j1 + 1)):
            for ii in range(i0, min(i0 + 3, i1 + 1)):
                rgb, al, _, _ = render_tile(cloud, grid, ii, jj)
                tgt, cv_, _, _ = ref.render_tile(grid, ii, jj, max_frames=1)
                mk = (al > 0.5) & (cv_ > 0.5)
                if mk.sum() > 5000:
                    ps.append(psnr(rgb[mk], tgt[mk]))
        fsc = {"fsc": None}
        if held_kf is not None:
            fsc = fsc_heldout(c["means"][sel], held_kf, npz,
                              (ds.K_orig, ds.cfg.height, ds.cfg.width), sim3_fn, ds.T_WP)
        import cv2
        cv2.imwrite(str(out / f"{policy}.png"),
                    (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        mean_step = float(np.mean([s["map"] for s in steps]))
        row = {"policy": policy, "n": int(len(sel)), "coverage": float((cov > 0.5).mean()),
               "layer": layer_metric(c["means"][sel], ds.T_WP),
               "psnr": float(np.mean(ps)) if ps else None,
               "drift_p95": drift.get("p95_dz"), "new_cells": flo.get("n_new"),
               "new_median_dz": flo.get("median_dz"), "new_frac_gt1m": flo.get("frac_gt1m"),
               "free_reject": int(gm.n_free_reject),
               "fsc": fsc.get("fsc"), "fsc_visible": fsc.get("n_visible"),
               "fsc_kf": int(npz["frame_id"][held_kf]) if held_kf is not None else None,
               "nfp": gm.nfp.coverage() if gm.nfp is not None else None,
               "growth": growth,
               "ms_patch": float(np.mean([s["paste"] for s in steps])),
               "ms_map": mean_step, "ms_render": 1000 * t_render,
               "tiles": int(n_tiles), "s_build": round(t_build, 1),
               "fps_stream": float(1000.0 / max(mean_step + 0.0, 1e-6)),
               "fps_ortho_full": float(n_tiles / max(t_render, 1e-6)),
               "ms_per_tile": 1000 * t_render / max(n_tiles, 1)}
        rows.append(row)
        print(f"[s37] {policy:8s} N={row['n']:>8,} cov={100*row['coverage']:5.1f}% "
              f"layer={row['layer']:.2f} psnr={row['psnr']:.2f} | drift p95={row['drift_p95']} "
              f"| FSC={100*(row['fsc'] or 0):5.1f}% | new {row['new_cells']:,} "
              f">1m={100*(row['new_frac_gt1m'] or 0):4.1f}% | reject={row['free_reject']:,} "
              f"| map {row['ms_map']:.0f} ms  render {row['ms_render']:.0f} ms / {n_tiles} tiles "
              f"({row['ms_per_tile']:.1f} ms/tile)")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"[s37] -> {out}/metrics.json")


if __name__ == "__main__":
    main()
