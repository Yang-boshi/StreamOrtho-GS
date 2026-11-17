"""S6a: dirty-tile streaming replay (design doc §7.7).

Replays one sequence's keyframes through the map and the ortho engine, and
records:

* per keyframe: new dirty tiles, tiles rendered, dirty ratio, stage latencies;
* after the replay: coverage completeness (no missed tiles), cache stats,
  per-tile quality vs the GT-plane reference, and a dirty/render heatmap.

Outputs (``results/stage-S6a/<run>/``):

* ``dirty_heatmap.png``  per-tile render-count heatmap (log scale)
* ``latency.json``       per-stage latency + per-keyframe log
* ``mosaic.png``         the assembled cache mosaic
* ``reference.png``      the same area from the plane-warp reference
* ``compare.png``        mosaic | reference
* ``metrics.json``       acceptance numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.eval.latency import StageTimer
from stream_ortho.frontend.keyframe_policy import bbox_from_pose
from stream_ortho.frontend.predict import psnr, ssim
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig


def build_parser():
    p = argparse.ArgumentParser(description="S6a: dirty-tile replay")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=12)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--budget-ms", type=float, default=250.0)
    p.add_argument("--max-tiles", type=int, default=8)
    p.add_argument("--margin-tiles", type=int, default=1)
    p.add_argument("--gsd", type=float, default=0.0)
    p.add_argument("--pose-updates", type=int, default=1,
                   help="how many pose-update events to simulate at the end")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=zcfg.DEVICE)
    return p


def cruise_keyframes(npz, n_max):
    T = npz["T_WC"]
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(len(T))
    cruise = np.where(np.abs(np.log(s / np.median(s))) < np.log(1.3))[0]
    if len(cruise) == 0:
        cruise = np.arange(len(T))
    return cruise[:n_max].tolist()


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"]
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S6a" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    ds = NPUDroneMapDataset(seq_dir)
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])),
                         [m["scale"]]])

    kfs = cruise_keyframes(npz, a.max_kf)
    kf_mid = kfs[len(kfs) // 2]
    # NB: X_canon is in SLAM units, so the metric scale comes from the *composed*
    # camera->GT Sim3 (est->GT scale ~1e2), not from the per-keyframe Sim3 (~1).
    _, _, s_gt = sim3_from_vec8(compose_sim3(npz["T_WC"][kf_mid], sg))
    depth_m = float(np.median(npz["X_canon"][kf_mid][..., 2].astype(np.float32))) * s_gt
    gsd = a.gsd or (depth_m / float(ds.K_orig[0, 0]))
    if gsd < 0.01:
        raise RuntimeError(f"implausible GSD {gsd:.5f} m/px (depth {depth_m:.1f} m)")
    grid = OrthoGrid(ds.T_WP, gsd, tile=a.tile, halo=a.halo, z_ground=0.0, z_top=800.0)

    gmap = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2),
                       ds.T_WP, z_ground=0.0, device=a.device)
    cache = TileCache()

    def render_fn(g, i, j):
        return render_tile(gmap.to_cloud(), g, i, j)

    cfg = SchedulerConfig(margin_tiles=a.margin_tiles, max_tiles_per_flush=a.max_tiles,
                          budget_ms=a.budget_ms, tile=a.tile)
    eng = OrthoEngine(grid, cache, render_fn, cfg)
    timers = StageTimer()

    rows = []
    print(f"[S6a] {run_dir.name}: {len(kfs)} kfs, tile={a.tile}, gsd={gsd:.4f}, "
          f"budget={a.budget_ms}ms cap={a.max_tiles}")
    for n, kf in enumerate(kfs):
        t_kf = StageTimer()
        with t_kf("patch"):
            patch = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="prior",
                                device=a.device)
        with t_kf("ingest"):
            gmap.ingest(patch, kf)
        with t_kf("commit"):
            gmap.commit(kf)
        bbox = bbox_from_pose(sim3_matrix(compose_sim3(npz["T_WC"][kf], sg)),
                              ds.K_orig, int(ds.cfg.width), int(ds.cfg.height),
                              ds.T_WP, z_ground=0.0)
        with t_kf("mark"):
            mk = eng.mark_footprint(bbox, kf_id=kf)
        with t_kf("flush"):
            fl = eng.flush()
        end_to_end = sum(v[0] for v in t_kf.samples.values())
        for k, v in t_kf.samples.items():
            timers.add(k, v[0])
        timers.add("end_to_end", end_to_end)
        dirty_ratio = mk["newly_dirty"] / max(len(eng.covered), 1)
        rows.append({"kf": int(kf), "n_covered": len(eng.covered),
                     "tiles_marked": mk["tiles"], "newly_dirty": mk["newly_dirty"],
                     "rendered": fl["rendered"], "flush_ms": fl["ms"],
                     "dirty_left": fl["dirty_left"],
                     "dirty_ratio": round(dirty_ratio, 4),
                     "end_to_end_ms": round(end_to_end, 1)})
        if n % 3 == 0 or n == len(kfs) - 1:
            print(f"  kf{kf:3d} covered={len(eng.covered):3d} +{mk['newly_dirty']:2d} dirty "
                  f"rendered={fl['rendered']} e2e={end_to_end:.0f}ms")

    bbox_of_mid = bbox_from_pose(sim3_matrix(compose_sim3(npz["T_WC"][kf_mid], sg)),
                                 ds.K_orig, int(ds.cfg.width), int(ds.cfg.height),
                                 ds.T_WP, z_ground=0.0)

    # ---- drain + acceptance -------------------------------------------
    with timers("flush_all"):
        drained = eng.flush_all()
    cov = eng.coverage_complete()

    # ---- simulated pose update (week-7 interface) ---------------------
    pose_log = []
    for k in range(a.pose_updates):
        kf_u = kfs[len(kfs) // 2]
        t0 = timers
        with t0("pose_update"):
            r = eng.mark_pose_changed([kf_u], gmap)
            f = eng.flush()
        pose_log.append({**r, "rendered": f["rendered"], "ms": f["ms"]})

    # ---- quality vs the reference -------------------------------------
    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    keys = sorted(eng.rendered)
    if keys:
        i0 = min(k[0] for k in keys); i1 = max(k[0] for k in keys)
        j0 = min(k[1] for k in keys); j1 = max(k[1] for k in keys)
    else:
        i0 = i1 = j0 = j1 = 0
    n_rows, n_cols = (j1 - j0 + 1), (i1 - i0 + 1)   # NB: `rows` is the per-KF log

    mosaic = cache.mosaic(keys, a.tile)
    # reference rendered on exactly the same tile block (aligned panel, full-coverage metrics)
    ref_mosaic, ref_cov, _ = ref.render_block(grid, i0, j0, n_rows, n_cols, max_frames=1)
    mosaic = mosaic[: ref_mosaic.shape[0], : ref_mosaic.shape[1]]

    both = (ref_cov > 0.5)
    cov_masked = np.zeros_like(both)
    for (i, j) in keys:
        t = cache.tiles[(i, j)]
        if t.renders == 0:
            continue
        y = (j1 - j) * a.tile
        x = (i - i0) * a.tile
        cov_masked[y:y + a.tile, x:x + a.tile] = t.alpha > 0.5
    both &= cov_masked
    if both.sum() > 5000:
        ps = psnr(mosaic[both], ref_mosaic[both])
        ys, xs = np.nonzero(both)
        y0b, y1b = ys.min(), ys.max() + 1
        x0b, x1b = xs.min(), xs.max() + 1
        ss = ssim(mosaic[y0b:y1b, x0b:x1b], ref_mosaic[y0b:y1b, x0b:x1b])
    else:
        ps, ss = float("nan"), float("nan")
    ps_list = [ps]
    ss_list = [ss]

    import cv2

    def save(name, arr):
        cv2.imwrite(str(out / name), (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    save("mosaic.png", mosaic)
    save("reference.png", ref_mosaic)
    gap = np.ones((mosaic.shape[0], 8, 3), np.float32)
    save("compare.png", np.concatenate([mosaic, gap, ref_mosaic], axis=1))

    # heatmap of render counts
    if keys:
        i0 = min(k[0] for k in keys); i1 = max(k[0] for k in keys)
        j0 = min(k[1] for k in keys); j1 = max(k[1] for k in keys)
        H = np.full((j1 - j0 + 1, i1 - i0 + 1), -1.0, np.float32)
        for (i, j) in keys:
            H[j1 - j, i - i0] = cache.tiles[(i, j)].renders
        valid = H >= 0
        Hn = np.zeros(H.shape, np.uint8)
        Hn[valid] = (H[valid] / max(float(H[valid].max()), 1.0) * 255).astype(np.uint8)
        hm = cv2.applyColorMap(Hn, cv2.COLORMAP_TURBO)
        hm[~valid] = (90, 90, 90)          # grey = no tile / outside the union footprint
        scale = 40
        hm = cv2.resize(hm, (hm.shape[1] * scale, hm.shape[0] * scale),
                        interpolation=cv2.INTER_NEAREST)
        for r in range(H.shape[0] + 1):
            cv2.line(hm, (0, r * scale), (hm.shape[1], r * scale), (40, 40, 40), 1)
        for c in range(H.shape[1] + 1):
            cv2.line(hm, (c * scale, 0), (c * scale, hm.shape[0]), (40, 40, 40), 1)
        for r in range(H.shape[0]):
            for c in range(H.shape[1]):
                if H[r, c] > 1:
                    cv2.putText(hm, str(int(H[r, c])), (c * scale + 4, r * scale + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(hm, f"render count per tile (max {int(H.max())}), {H.shape[1]}x{H.shape[0]} tiles",
                    (6, hm.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out / "dirty_heatmap.png"), hm)

    lat = timers.summary()
    lat["per_keyframe"] = rows
    lat["pose_updates"] = pose_log
    (out / "latency.json").write_text(json.dumps(lat, indent=2))

    first = rows[0] if rows else {}
    rest = [r for r in rows[1:]]
    metrics = {
        "run": run_dir.name, "seq": ds.name, "n_kfs": len(kfs),
        "tile": a.tile, "halo": a.halo, "gsd_m_per_px": gsd,
        "budget_ms": a.budget_ms, "max_tiles_per_flush": a.max_tiles,
        "n_tiles_covered": len(eng.covered), "n_tiles_rendered": len(eng.rendered),
        "coverage_complete": cov,
        "drained": drained,
        "dirty_ratio_first_kf": first.get("dirty_ratio", 0.0),
        "dirty_ratio_after_first_grid": float(np.mean([r["newly_dirty"] for r in rest]) / max(len(eng.covered), 1)) if rest else 0.0,
        "dirty_ratio_mean_after_first": float(np.mean([r["dirty_ratio"] for r in rest])) if rest else 0.0,
        "dirty_ratio_max_after_first": float(np.max([r["dirty_ratio"] for r in rest])) if rest else 0.0,
        "end_to_end_ms_mean": lat.get("end_to_end", {}).get("mean_ms", 0.0),
        "end_to_end_ms_p95": lat.get("end_to_end", {}).get("p95_ms", 0.0),
        "ortho_refresh_ms_mean": lat.get("flush", {}).get("mean_ms", 0.0),
        "ortho_refresh_ms_p95": lat.get("flush", {}).get("p95_ms", 0.0),
        "cache_stats": cache.stats(),
        "psnr_vs_reference": float(np.mean(ps_list)) if ps_list else float("nan"),
        "ssim_vs_reference": float(np.mean(ss_list)) if ss_list else float("nan"),
        "n_quality_tiles": len(ps_list),
        "n_committed": gmap.n_committed,
        "memory_mb": round(gmap.memory_mb(), 1),
        "pose_updates": pose_log,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[S6a] covered={len(eng.covered)} rendered={len(eng.rendered)} "
          f"complete={cov['complete']} dirty_mean={100*metrics['dirty_ratio_mean_after_first']:.1f}% "
          f"e2e={metrics['end_to_end_ms_mean']:.0f}ms refresh={metrics['ortho_refresh_ms_mean']:.0f}ms "
          f"PSNR={metrics['psnr_vs_reference']:.2f}dB")
    return metrics


if __name__ == "__main__":
    main()
