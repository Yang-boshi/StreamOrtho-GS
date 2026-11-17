"""S5a: replay a sequence's keyframes through the global map (gate/dedup/commit).

Three modes make the week-4 comparison:

===================  ======  ========  ===============  ==============
mode                 gate    dedup     commit rule      purpose
===================  ======  ========  ===============  ==============
``append``           off     off       immediate        baseline ("只追加")
``gate``             on      off       immediate        gate only
``gatededup``        on      on        n_obs>=2 / age   full pipeline
===================  ======  ========  ===============  ==============

Outputs per mode (``results/stage-S5a/<run>/<mode>/``):

* ``stats.json``      final map stats + gate histogram + timings;
* ``curve.csv``       per-keyframe N / staged / merged / rejected / memory;
* ``tile/``           one 512 tile rendered **from the committed map** plus the
                      plane-warp reference and PSNR/SSIM.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.data.plane import world_to_plane
from stream_ortho.geometry import R_to_quat_xyzw, sim3_from_vec8
from stream_ortho.map.build import head_plane_patch, head_patch, plane_patch
from stream_ortho.map.commit import GateConfig, GateStats
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.frontend.predict import psnr, ssim

MODES = {
    #            gate   dedup  min_obs  max_age
    "append":   (False, False, 1, 0),
    "gate":     (True,  False, 1, 0),
    "gatededup": (True, True,  2, 2),
}


def build_parser():
    p = argparse.ArgumentParser(description="S5a: global map replay")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--mode", default="gatededup", choices=list(MODES))
    p.add_argument("--cloud", default="plane", choices=["plane", "head", "head_plane"])
    p.add_argument("--stride", type=int, default=6, help="pixel stride of the plane patch")
    p.add_argument("--max-kf", type=int, default=20)
    p.add_argument("--voxel", type=float, default=0.4)
    p.add_argument("--budget", type=int, default=0, help="enforce N_max at the end (0 = off)")
    p.add_argument("--budget-strategy", default="conf", choices=["conf", "random"])
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=zcfg.DEVICE)
    return p


def cruise_keyframes(npz, n_max: int):
    T = npz["T_WC"]
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(len(T))
    cruise = np.where(np.abs(np.log(s / np.median(s))) < np.log(1.3))[0]
    if len(cruise) == 0:
        cruise = np.arange(len(T))
    return cruise[:n_max].tolist()


def render_map_tile(gmap, ds, npz, kf_center, out_dir, gsd=None):
    import cv2

    from stream_ortho.geometry import sim3_act

    out_dir.mkdir(parents=True, exist_ok=True)
    cloud = gmap.to_cloud()
    T = npz["T_WC"]
    _, _, s_c = sim3_from_vec8(T[kf_center])
    depth_m = float(np.median(npz["X_canon"][kf_center][..., 2].astype(np.float32))) * s_c
    gsd = gsd or (depth_m / float(ds.K_orig[0, 0]))
    grid = OrthoGrid(ds.T_WP, gsd, tile=512, halo=32, z_ground=0.0, z_top=800.0)

    # centre the tile on the *centre keyframe's nadir* (a well covered spot);
    # the map median can sit at the edge of many keyframes' footprints.
    from stream_ortho.geometry import compose_sim3 as _compose, sim3_matrix as _m4
    cam_w = None
    if getattr(gmap, "sim3_gt", None) is not None:
        cam_w = _m4(_compose(npz["T_WC"][kf_center], np.asarray(gmap.sim3_gt, np.float64)))[:3, 3]
    P = world_to_plane(ds.T_WP, gmap.committed["means"])
    if cam_w is not None:
        cam = world_to_plane(ds.T_WP, cam_w[None])[0]
        cx, cy = float(cam[0]), float(cam[1])
    else:
        cx, cy = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
    i, j = grid.tile_of(cx, cy)
    rgb, alpha, _, _ = render_tile(cloud, grid, i, j)

    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    ref_rgb, ref_cov, n_ref, _ = ref.render_tile(grid, i, j, max_frames=1)
    m = (alpha > 0.5) & (ref_cov > 0.5)
    ps = psnr(rgb[m], ref_rgb[m]) if m.sum() > 1000 else float("nan")
    ss = ssim(rgb[m], ref_rgb[m]) if m.sum() > 1000 else float("nan")

    gap = np.ones((rgb.shape[0], 8, 3), np.float32)
    for name, arr in (("ours.png", rgb), ("reference.png", ref_rgb),
                      ("coverage.png", np.repeat(alpha[..., None], 3, axis=-1)),
                      ("compare.png", np.concatenate([rgb, gap, ref_rgb, gap,
                                                      np.repeat(alpha[..., None], 3, axis=-1)], axis=1))):
        cv2.imwrite(str(out_dir / name), (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
    return {"tile_i": int(i), "tile_j": int(j), "gsd_m_per_px": gsd,
            "coverage": float((alpha > 0.5).mean()),
            "psnr_vs_reference": ps, "ssim_vs_reference": ss,
            "reference_frames": int(n_ref), "n_cloud": len(cloud)}


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"]
    out = pathlib.Path(a.out) if a.out else (
        zcfg.RESULTS_DIR / "stage-S5a" / run_dir.name / a.mode)
    out.mkdir(parents=True, exist_ok=True)

    ds = NPUDroneMapDataset(seq_dir)
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])),
                         [m["scale"]]])

    gate_on, dedup_on, min_obs, max_age = MODES[a.mode]
    cfg = MapConfig(voxel_size=a.voxel, dedup=dedup_on, gate=gate_on,
                    commit_min_obs=min_obs, commit_max_age=max_age,
                    max_points=a.budget, budget_strategy=a.budget_strategy)
    gate_cfg = GateConfig() if gate_on else None
    gmap = GaussianMap(cfg, ds.T_WP, z_ground=0.0, device=a.device, gate_cfg=gate_cfg)
    gmap.sim3_gt = sg                 # 8-vector camera->GT world, for the tile centre

    kfs = cruise_keyframes(npz, a.max_kf)
    gstats = GateStats()
    rows = []
    print(f"[S5a] mode={a.mode} run={run_dir.name} kfs={len(kfs)} voxel={a.voxel} stride={a.stride}")
    model = head = None
    if a.cloud in ("head", "head_plane"):
        from stream_ortho.frontend import predict as P
        from stream_ortho.frontend.gaussian_head import load_gaussian_head

        model = P.load_model(device=a.device)
        head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=a.device, verbose=False)

    for kf in kfs:
        if a.cloud == "plane":
            patch = plane_patch(ds, npz, kf, sg, stride=a.stride, device=a.device)
        elif a.cloud == "head":
            patch = head_patch(ds, npz, kf, sg, model, head, device=a.device)
        else:
            patch = head_plane_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                     device=a.device)
        if patch.meta.get("source") == "plane" and patch.meta.get("depth_m"):
            patch.meta["depth_m"] = float(patch.meta["depth_m"])
        r = gmap.ingest(patch, kf)
        c = gmap.commit(kf)
        if r.get("gate"):
            gstats.update(r["gate"])
        rows.append({
            "kf": int(kf),
            "n_in": r["n_in"],
            "n_new": r.get("n_new", 0),
            "n_merged_committed": r.get("n_merged_committed", 0),
            "n_merged_staged": r.get("n_merged_staged", 0),
            "n_rejected_gate": r["gate"]["n_rejected"] if r.get("gate") else 0,
            "n_committed": gmap.n_committed,
            "n_staged": gmap.n_staged,
            "dup_rate": gmap.duplicate_rate(),
            "mem_mb": round(gmap.memory_mb(), 2),
        })
        if len(rows) % 5 == 0:
            print(f"  kf{kf:3d} N={gmap.n_committed:7d} staged={gmap.n_staged:6d} "
                  f"dup={rows[-1]['dup_rate']:.3f} mem={rows[-1]['mem_mb']:.0f}MB")

    budget = None
    if a.budget:
        budget = gmap.enforce_budget(a.budget, a.budget_strategy)
        print(f"[S5a] budget {a.budget}: {budget}")

    kf_center = kfs[len(kfs) // 2]
    tile = render_map_tile(gmap, ds, npz, kf_center, out / "tile")

    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    stats = gmap.stats(
        mode=a.mode, run=run_dir.name, seq=ds.name, n_kfs=len(kfs),
        stride=a.stride, gate=gstats.as_dict(), budget=budget, tile=tile,
        ingest_rate=sum(r["n_in"] for r in rows) / max(gmap.t_ingest, 1e-6),
        merged_rate=sum(r["n_merged_committed"] + r["n_merged_staged"] for r in rows)
        / max(sum(r["n_in"] for r in rows), 1),
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[S5a] {a.mode}: N={stats['n_committed']} dup={stats['duplicate_rate']:.3f} "
          f"mem={stats['memory_mb']:.1f}MB  PSNR={tile['psnr_vs_reference']:.2f} dB  "
          f"cov={tile['coverage']:.2f}")
    return stats


if __name__ == "__main__":
    main()
