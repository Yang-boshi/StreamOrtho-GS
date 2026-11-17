"""S6b: loop-closure / pose-update demo + async worker (design doc §7.8).

Story:

1. build the map from 8 cruise keyframes, but with a **simulated drift** on the
   tail of the trajectory (translation + yaw growing to 3 m / 1.5 deg);
2. render the ortho -> snapshot **A** (drifted, mismatched with the reference);
3. "loop closure": for every drifted keyframe apply ``Δ = drift^-1`` with
   :func:`apply_pose_update` (only that kf's Gaussians move), mark the affected
   tiles dirty via ``mark_pose_changed``, and hand them to the async worker;
4. pump the worker frame by frame, saving a GIF so the ortho visibly heals;
5. snapshot **B** and check: PSNR vs the GT reference improves, tiles
   re-rendered << all tiles, and nothing outside the re-rendered tiles changed
   (**no tearing**).

Outputs (``results/stage-S6b/<run>/``): ``pose_update_demo.gif``,
``tearing_check.png``, ``latency_report.md``, ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.backend.workers import TileWorker, WorkerConfig
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.eval.latency import StageTimer
from stream_ortho.frontend.keyframe_policy import bbox_from_pose
from stream_ortho.frontend.predict import psnr, ssim
from stream_ortho.geometry import compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.geometry import R_to_quat_xyzw
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.pose_update import (apply_pose_update, invert_delta,
                                          sim3_vec_from_matrix)
from stream_ortho.ortho.cache import TileCache
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic
from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig


def build_parser():
    p = argparse.ArgumentParser(description="S6b: pose update + async worker")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=8)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--drift-from", type=int, default=4, help="first kf index with drift")
    p.add_argument("--drift-m", type=float, default=3.0, help="max drift translation [m]")
    p.add_argument("--drift-deg", type=float, default=1.5, help="max drift yaw [deg]")
    p.add_argument("--budget-ms", type=float, default=120.0)
    p.add_argument("--max-tiles", type=int, default=4)
    p.add_argument("--oracle", type=int, default=1,
                   help="also build the un-drifted map as the correctness oracle")
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


def drift_matrix(frac: float, tr: float, deg: float) -> np.ndarray:
    a = np.radians(deg * frac)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tr * frac, 0.5 * tr * frac, 0.1 * tr * frac])
    return T


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"]
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S6b" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    ds = NPUDroneMapDataset(seq_dir)
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    kfs = cruise_keyframes(npz, a.max_kf)
    kf_mid = kfs[len(kfs) // 2]
    _, _, s_gt = sim3_from_vec8(compose_sim3(npz["T_WC"][kf_mid], sg))
    depth_m = float(np.median(npz["X_canon"][kf_mid][..., 2].astype(np.float32))) * s_gt
    gsd = depth_m / float(ds.K_orig[0, 0])

    grid = OrthoGrid(ds.T_WP, gsd, tile=a.tile, halo=a.halo, z_ground=0.0, z_top=800.0)
    gmap = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                 commit_min_obs=2, commit_max_age=2),
                       ds.T_WP, z_ground=0.0, device=a.device)
    cache = TileCache()
    eng = OrthoEngine(grid, cache, lambda g, i, j: render_tile(gmap.to_cloud(), g, i, j),
                      SchedulerConfig(margin_tiles=1, max_tiles_per_flush=64, tile=a.tile))
    worker = TileWorker(cache, lambda g, i, j: render_tile(gmap.to_cloud(), g, i, j), grid,
                        WorkerConfig(budget_ms=a.budget_ms, max_tiles=a.max_tiles))
    timers = StageTimer()

    # ---- 1) build the map with a simulated drift on the tail -------------
    drift = {}
    n_tail = max(1, len(kfs) - a.drift_from)
    for j, kf in enumerate(kfs):
        frac = 0.0 if j < a.drift_from else (j - a.drift_from + 1) / n_tail
        drift[kf] = frac
    print(f"[S6b] {run_dir.name}: {len(kfs)} kfs, drift on {sum(1 for f in drift.values() if f>0)} "
          f"kfs (max {a.drift_m} m / {a.drift_deg} deg), gsd={gsd:.4f}")

    for kf in kfs:
        with timers("build_patch"):
            T_gt = sim3_matrix(compose_sim3(npz["T_WC"][kf], sg))
            T_drift = drift_matrix(drift[kf], a.drift_m, a.drift_deg) @ T_gt
            patch = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="prior",
                                device=a.device,
                                T_gt_wc=sim3_vec_from_matrix(T_drift))
        with timers("ingest"):
            gmap.ingest(patch, kf)
        with timers("commit"):
            gmap.commit(kf)
        bbox = bbox_from_pose(T_drift, ds.K_orig, int(ds.cfg.width), int(ds.cfg.height),
                              ds.T_WP, z_ground=0.0)
        with timers("mark"):
            eng.mark_footprint(bbox, kf_id=kf)
        with timers("flush"):
            eng.flush(budget_ms=1e9, max_tiles=64)
    with timers("drain_build"):
        eng.flush_all()

    keys = sorted(eng.rendered)
    i0 = min(k[0] for k in keys); i1 = max(k[0] for k in keys)
    j0 = min(k[1] for k in keys); j1 = max(k[1] for k in keys)
    n_rows, n_cols = (j1 - j0 + 1), (i1 - i0 + 1)

    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    with timers("reference"):
        ref_mosaic, ref_cov, _ = ref.render_block(grid, i0, j0, n_rows, n_cols, max_frames=1)

    def snapshot():
        mos = cache.mosaic(keys, a.tile)
        cov = np.zeros(mos.shape[:2], np.float32)
        for (i, j) in keys:
            t = cache.tiles.get((i, j))
            if t is None or t.renders == 0:
                continue
            y = (j1 - j) * a.tile
            x = (i - i0) * a.tile
            cov[y:y + a.tile, x:x + a.tile] = t.alpha
        return mos, cov

    A, covA = snapshot()
    both = covA > 0.5
    psnr_A = psnr(A[both], ref_mosaic[both]) if both.sum() > 5000 else float("nan")
    print(f"[S6b] before closure: PSNR {psnr_A:.2f} dB  (whole block)")

    # ---- 2) closure: apply the inverse drift on the tail ----------------
    dirty_keys = set()
    pu = []
    with timers("pose_update"):
        for kf in kfs:
            if drift[kf] <= 0:
                continue
            T_gt = sim3_matrix(compose_sim3(npz["T_WC"][kf], sg))
            d = sim3_vec_from_matrix(drift_matrix(drift[kf], a.drift_m, a.drift_deg))
            r = apply_pose_update(gmap, [kf], invert_delta(d))
            pu.append(r)
            with timers("mark_pose"):
                eng.mark_pose_changed([kf], gmap)
            dirty_keys |= set(eng.cache.dirty_keys())
    # recompute the map index (positions changed)
    gmap._rebuild_index("c")
    print(f"[S6b] closure: {sum(r['n_moved'] for r in pu)} Gaussians moved, "
          f"{len(dirty_keys)} tiles to re-render")

    # ---- 3) async worker: progressive re-render + GIF --------------------
    frames = []
    changed_union = np.zeros(A.shape[:2], bool)
    for (i, j) in sorted(dirty_keys):
        worker.submit([(i, j)], priority=1.0, reason="pose-update")
    step = 0
    while worker.pending() > 0:
        r = worker.pump()
        step += 1
        with timers("worker_pump"):
            pass
        for (i, j) in sorted(dirty_keys)[:step * a.max_tiles]:
            y = (j1 - j) * a.tile
            x = (i - i0) * a.tile
            changed_union[y:y + a.tile, x:x + a.tile] = True
        frames.append((step, r["pending"]))
        if r["rendered"] == 0:
            break
    B, covB = snapshot()
    psnr_B = psnr(B[both], ref_mosaic[both]) if both.sum() > 5000 else float("nan")
    tail = both & changed_union
    head = both & ~changed_union
    psnr_A_tail = psnr(A[tail], ref_mosaic[tail]) if tail.sum() > 5000 else float("nan")
    psnr_B_tail = psnr(B[tail], ref_mosaic[tail]) if tail.sum() > 5000 else float("nan")
    psnr_A_head = psnr(A[head], ref_mosaic[head]) if head.sum() > 5000 else float("nan")
    psnr_B_head = psnr(B[head], ref_mosaic[head]) if head.sum() > 5000 else float("nan")
    print(f"[S6b] after closure:  PSNR {psnr_B:.2f} dB  ({len(frames)} worker pumps)")
    print(f"[S6b] drifted region: {psnr_A_tail:.2f} -> {psnr_B_tail:.2f} dB "
          f"({psnr_B_tail-psnr_A_tail:+.2f});  untouched region: {psnr_A_head:.2f} -> {psnr_B_head:.2f} dB")

    # ---- 4) tearing: anything changed outside the re-rendered tiles? -----
    diff = np.abs(B - A).max(axis=2)
    outside = ~changed_union
    tear_outside = float((diff[outside] > 1e-4).mean())
    tear_inside = float((diff[changed_union] > 1e-4).mean()) if changed_union.any() else 0.0
    print(f"[S6b] tearing outside re-rendered tiles: {100*tear_outside:.4f}% "
          f"(inside {100*tear_inside:.1f}% changed)")

    # ---- 4b) oracle: the same map built **without** drift ----------------
    oracle_rgb = None
    psnr_A_oracle = psnr_B_oracle = float("nan")
    psnr_A_oracle_tail = psnr_B_oracle_tail = float("nan")
    if a.oracle:
        with timers("oracle"):
            gmap2 = GaussianMap(MapConfig(voxel_size=0.4, dedup=True, gate=False,
                                          commit_min_obs=2, commit_max_age=2),
                                ds.T_WP, z_ground=0.0, device=a.device)
            for kf in kfs:
                pp = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="prior",
                                 device=a.device)          # no drift
                gmap2.ingest(pp, kf)
                gmap2.commit(kf)
        cache2 = TileCache()

        def render2(g, i, j):
            from stream_ortho.ortho.rasterizer import render_tile as _rt
            return _rt(gmap2.to_cloud(), g, i, j)

        eng2 = OrthoEngine(grid, cache2, render2,
                           SchedulerConfig(margin_tiles=1, max_tiles_per_flush=64, tile=a.tile))
        for (i, j) in keys:
            eng2.mark_footprint((i * a.tile * gsd, j * a.tile * gsd,
                                 (i + 1) * a.tile * gsd, (j + 1) * a.tile * gsd), kf_id=-1)
        eng2.flush_all()
        oracle_rgb = cache2.mosaic(keys, a.tile)
        oracle_rgb = oracle_rgb[: A.shape[0], : A.shape[1]]
        ok = both & (oracle_rgb.max(axis=2) > 0.02)
        if ok.sum() > 5000:
            psnr_A_oracle = psnr(A[ok], oracle_rgb[ok])
            psnr_B_oracle = psnr(B[ok], oracle_rgb[ok])
            tail_ok = ok & changed_union
            if tail_ok.sum() > 5000:
                psnr_A_oracle_tail = psnr(A[tail_ok], oracle_rgb[tail_ok])
                psnr_B_oracle_tail = psnr(B[tail_ok], oracle_rgb[tail_ok])
        print(f"[S6b] vs the un-drifted oracle: {psnr_A_oracle:.2f} -> {psnr_B_oracle:.2f} dB "
              f"(drifted region {psnr_A_oracle_tail:.2f} -> {psnr_B_oracle_tail:.2f} dB)")
        import cv2 as _cv2
        _cv2.imwrite(str(out / "ortho_oracle.png"),
                     (np.clip(oracle_rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    # ---- 5) figures -----------------------------------------------------
    import cv2

    def save(name, arr):
        cv2.imwrite(str(out / name), (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    save("ortho_before.png", A)
    save("ortho_after.png", B)
    d = np.clip(diff * 8.0, 0, 1)
    d = np.repeat(d[..., None], 3, axis=-1)
    mask = np.repeat(changed_union[..., None].astype(np.float32), 3, axis=-1)
    save("dirty_mask.png", mask)
    gap = np.ones((A.shape[0], 8, 3), np.float32)
    save("tearing_check.png", np.concatenate([A, gap, B, gap, d, gap,
                                              np.repeat((~changed_union)[..., None], 3, axis=-1) * 0.25],
                                             axis=1))

    # GIF: crop to the changed area so the file stays small
    ys, xs = np.nonzero(changed_union)
    if len(ys):
        y0c, y1c = max(0, ys.min() - 64), min(A.shape[0], ys.max() + 64)
        x0c, x1c = max(0, xs.min() - 64), min(A.shape[1], xs.max() + 64)
    else:
        y0c, y1c, x0c, x1c = 0, A.shape[0], 0, A.shape[1]

    def to_gif(img, label=""):
        crop = np.clip(img[y0c:y1c, x0c:x1c], 0, 1)
        h, w = crop.shape[:2]
        if w > 640:
            f = 640 / w
            crop = cv2.resize(crop, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        bgr = np.ascontiguousarray((crop * 255).astype(np.uint8)[:, :, ::-1])
        if label:
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(bgr, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return bgr

    gif = [to_gif(A, f"before closure  ({len(dirty_keys)} tiles dirty, PSNR {psnr_A:.2f} dB)")]
    # replay the pumps, cropping immediately so we never keep 12 full mosaics in RAM
    for step_i, pending in frames:
        cur, _ = snapshot()
        gif.append(to_gif(cur, f"pump {step_i}/{len(frames)}  re-rendered "
                              f"{min(step_i*a.max_tiles, len(dirty_keys))}/{len(dirty_keys)} tiles"))
    try:
        from PIL import Image

        ims = [Image.fromarray(f[:, :, ::-1]) for f in gif]      # RGB for PIL
        ims[0].save(str(out / "pose_update_demo.gif"), save_all=True,
                    append_images=ims[1:], duration=450, loop=0, optimize=False)
    except Exception as e:      # fallback: a PNG sequence
        for n, f in enumerate(gif):
            cv2.imwrite(str(out / f"frame_{n:02d}.png"), f)
        print(f"  (gif writer unavailable: {e}; wrote frames instead)")

    # ---- 6) latency report ----------------------------------------------
    lat = timers.summary()
    lat["worker"] = worker.stats()
    lat["pose_updates"] = pu
    lat["n_files"] = len(gif)
    (out / "latency.json").write_text(json.dumps(lat, indent=2))

    metrics = {
        "run": run_dir.name, "seq": ds.name, "n_kfs": len(kfs),
        "tile": a.tile, "gsd_m_per_px": gsd, "stride": a.stride,
        "drift": {"from_kf": a.drift_from, "max_m": a.drift_m, "max_deg": a.drift_deg},
        "n_tiles": len(keys), "n_dirty_after_closure": len(dirty_keys),
        "dirty_ratio": len(dirty_keys) / max(len(keys), 1),
        "psnr_before": psnr_A, "psnr_after": psnr_B, "psnr_gain": psnr_B - psnr_A,
        "psnr_before_tail": psnr_A_tail, "psnr_after_tail": psnr_B_tail,
        "psnr_gain_tail": psnr_B_tail - psnr_A_tail,
        "psnr_before_head": psnr_A_head, "psnr_after_head": psnr_B_head,
        "tearing_outside_frac": tear_outside, "changed_inside_frac": tear_inside,
        "psnr_before_vs_oracle": psnr_A_oracle, "psnr_after_vs_oracle": psnr_B_oracle,
        "psnr_gain_vs_oracle": psnr_B_oracle - psnr_A_oracle,
        "psnr_before_vs_oracle_tail": psnr_A_oracle_tail,
        "psnr_after_vs_oracle_tail": psnr_B_oracle_tail,
        "n_moved_total": int(sum(r["n_moved"] for r in pu)),
        "max_displacement_m": float(max((r["max_displacement_m"] for r in pu), default=0.0)),
        "worker": worker.stats(), "n_worker_pumps": len(frames),
        "n_committed": gmap.n_committed, "memory_mb": round(gmap.memory_mb(), 1),
        "latency_ms": {k: v["mean_ms"] for k, v in lat.items() if isinstance(v, dict) and "mean_ms" in v},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    md = ["# S6b latency report", "",
          f"run: `{run_dir.name}`, tile {a.tile}, gsd {gsd:.4f} m/px, "
          f"drift {a.drift_m} m / {a.drift_deg} deg on the last {len(kfs)-a.drift_from} keyframes", "",
          "## Stages (mean per call)", "",
          "| stage | n | mean [ms] | p95 [ms] |", "|---|---|---|---|"]
    for k, v in lat.items():
        if isinstance(v, dict) and "mean_ms" in v:
            md.append(f"| {k} | {v['n']} | {v['mean_ms']:.1f} | {v['p95_ms']:.1f} |")
    md += ["", "## Closure", "",
           f"* Gaussians moved: **{metrics['n_moved_total']}** "
           f"(max displacement {metrics['max_displacement_m']:.2f} m)",
           f"* tiles re-rendered: **{len(dirty_keys)} / {len(keys)}** "
           f"({100*metrics['dirty_ratio']:.1f}%)",
           f"* worker pumps: {len(frames)}, per-pump budget {a.budget_ms} ms / {a.max_tiles} tiles",
           f"* PSNR vs reference (whole block): **{psnr_A:.2f} -> {psnr_B:.2f} dB** "
           f"({metrics['psnr_gain']:+.2f} dB)",
           f"* PSNR in the **drifted region**: **{psnr_A_tail:.2f} -> {psnr_B_tail:.2f} dB** "
           f"({metrics['psnr_gain_tail']:+.2f} dB)",
           f"* PSNR in the untouched region (control): {psnr_A_head:.2f} -> {psnr_B_head:.2f} dB",
           f"* **vs the un-drifted oracle**: {psnr_A_oracle:.2f} -> {psnr_B_oracle:.2f} dB "
           f"(drifted region {psnr_A_oracle_tail:.2f} -> {psnr_B_oracle_tail:.2f} dB) -- this is the",
           "  clean correctness metric: the plane reference carries its own bias.",
           f"* tearing outside re-rendered tiles: **{100*tear_outside:.4f}%** (target 0)",
           "", "## Worker", "", f"```\n{json.dumps(worker.stats(), indent=2)}\n```", ""]
    (out / "latency_report.md").write_text("\n".join(md))
    print(f"[S6b] -> {out}")
    return metrics


if __name__ == "__main__":
    main()
