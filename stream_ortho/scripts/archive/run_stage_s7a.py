"""S7a: ortho-aware scoring / pruning ablation (design doc §7.9, P4 partial).

Builds the same map once, then compares five eviction policies **at the same
``N_max``** and reports the metrics the P4 gate asks for:

* ortho PSNR / SSIM against the plane-warp reference (whole tile block),
* no-data (holes) fraction -- must not increase,
* floater ratio before/after -- must drop >= 20%,
* structural-edge retention (tagged edges must not be pruned),
* runtime per policy.

Also writes ``prune_visual.png`` (map coloured by prune reason + the ortho before
vs after pruning) and ``P4_PARTIAL.md``.

Outputs (``results/stage-S7a/<run>/``): ``metrics.json``, ``ablation_budget.png``,
``prune_visual.png``, ``P4_PARTIAL.md``, ``ortho_<strategy>.png``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.predict import psnr, ssim
from stream_ortho.geometry import compose_sim3, sim3_from_vec8, sim3_matrix
from stream_ortho.geometry import R_to_quat_xyzw
from stream_ortho.map.build import plane_patch
from stream_ortho.map.gaussian_map import GaussianMap, MapConfig
from stream_ortho.map.prune import PruneConfig, floater_ratio, prune_mask
from stream_ortho.map.scoring import composition, keep_mask, ortho_score_v2
from stream_ortho.ortho.grid import OrthoGrid
from stream_ortho.ortho.rasterizer import render_tile
from stream_ortho.ortho.reference import PlaneReferenceMosaic

STRATEGIES = ["random", "conf", "voxel", "ortho2", "ortho2_edge", "ortho3", "ortho3_edge", "edge"]


def build_parser():
    p = argparse.ArgumentParser(description="S7a: ortho-aware pruning ablation")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--n-max", type=int, default=250000)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--grid", default="2x2")
    p.add_argument("--cloud", default="plane", choices=["plane", "head_plane"],
                   help="plane = fallback B (uniform attributes); head_plane = learned "
                        "colour/opacity/scale with plane-constrained means (tests the prune rules)")
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


def render_policy(gmap, keys_arr, grid, i0, j0, rows, cols):
    """Render a ``rows x cols`` block from a *temporary* map holding ``keys_arr``."""
    from stream_ortho.map.gaussian import empty_map_arrays
    from stream_ortho.ortho.rasterizer import GaussianCloud
    import torch

    cloud = GaussianCloud(
        torch.as_tensor(keys_arr["means"], device="cuda"),
        torch.as_tensor(keys_arr["quats"], device="cuda"),
        torch.as_tensor(keys_arr["scales"], device="cuda"),
        torch.as_tensor(keys_arr["opacity"], device="cuda"),
        torch.as_tensor(keys_arr["colors"], device="cuda"))
    S = grid.tile
    out = np.zeros((rows * S, cols * S, 3), np.float32)
    cov = np.zeros((rows * S, cols * S), np.float32)
    for dj in range(rows):
        for di in range(cols):
            i, j = i0 + di, j0 + dj
            rgb, alpha, _, _ = render_tile(cloud, grid, i, j)
            r = (rows - 1 - dj) * S
            c = di * S
            out[r:r + S, c:c + S] = rgb
            cov[r:r + S, c:c + S] = alpha
    return out, cov


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"]
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S7a" / run_dir.name
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
    if a.cloud == "head_plane":
        from stream_ortho.frontend import predict as P
        from stream_ortho.frontend.gaussian_head import load_gaussian_head
        from stream_ortho.map.build import head_plane_patch

        model = P.load_model(device=a.device)
        head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=a.device, verbose=False)
    for kf in kfs:
        if a.cloud == "plane":
            patch = plane_patch(ds, npz, kf, sg, stride=a.stride, shape="prior",
                                device=a.device)
        else:
            patch = head_plane_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                     device=a.device)
        gmap.ingest(patch, kf)
        gmap.commit(kf)
    full = {k: v.copy() for k, v in gmap.committed.items()}
    n_full = len(full["means"])
    print(f"[S7a] {run_dir.name}: N={n_full:,} (before budget), N_max={a.n_max:,}, gsd={gsd:.4f}")

    # ---- evaluation block centred on the middle keyframe's nadir ----
    rows, cols = (int(x) for x in a.grid.lower().split("x"))
    cam_w = sim3_matrix(compose_sim3(npz["T_WC"][kf_mid], sg))[:3, 3]
    p_pl = (cam_w - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    ic, jc = grid.tile_of(float(p_pl[0]), float(p_pl[1]))
    i0, j0 = ic - cols // 2, jc - rows // 2
    ref = PlaneReferenceMosaic(ds, gsd, 0.0, stride=1)
    ref_rgb, ref_cov, _ = ref.render_block(grid, i0, j0, rows, cols, max_frames=1)

    base_prune = prune_mask(full, PruneConfig())
    floater_before = floater_ratio(full)
    print(f"[S7a] prune candidates: {base_prune['n_pruned']:,} "
          f"({100*base_prune['n_pruned']/n_full:.1f}%), reasons={base_prune['reasons']}, "
          f"floater_ratio={100*floater_before:.2f}%")

    results = {}
    for strat in STRATEGIES:
        t0 = time.time()
        keep = keep_mask(full, a.n_max, strategy=strat, rng=np.random.default_rng(0))
        dt = (time.time() - t0) * 1000.0
        sub = {k: v[keep] for k, v in full.items()}
        comp = composition(full, keep)
        rgb, cov = render_policy(gmap, sub, grid, i0, j0, rows, cols)
        both = (cov > 0.5) & (ref_cov > 0.5)
        ps = psnr(rgb[both], ref_rgb[both]) if both.sum() > 5000 else float("nan")
        if both.sum() > 5000:
            ys, xs = np.nonzero(both)
            ss = ssim(rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                      ref_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        else:
            ss = float("nan")
        nodata = float(((cov < 0.5) & (ref_cov > 0.5)).sum() / max((ref_cov > 0.5).sum(), 1))
        fr = floater_ratio(sub)
        results[strat] = {
            "n_kept": int(keep.sum()), "psnr": ps, "ssim": ss, "nodata": nodata,
            "floater_ratio": fr, "runtime_ms": round(dt, 1),
            "edge_kept": comp["edge_kept"], "edge_total": comp["edge_total"],
            "edge_keep_ratio": comp["edge_keep_ratio"],
            "area_kept": comp["area_kept"], "area_total": comp["area_total"],
        }
        import cv2

        cv2.imwrite(str(out / f"ortho_{strat}.png"),
                    (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        print(f"[S7a] {strat:11s} N={keep.sum():7d} PSNR={ps:6.2f} SSIM={ss:.3f} "
              f"nodata={100*nodata:.2f}% floater={100*fr:.2f}% "
              f"edge={100*comp['edge_keep_ratio']:.0f}% ({dt:.0f} ms)")

    # ---- ablation figure ------------------------------------------------
    import cv2

    def bar_panel(title, vals, fmt="{:.2f}", best_high=True):
        H, W = 260, 520
        img = np.full((H, W, 3), 255, np.uint8)
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        vmax = max(vals.values()) if vals else 1.0
        vmin = min(vals.values()) if vals else 0.0
        bw = int((W - 40) / max(len(vals), 1))
        for n, (k, v) in enumerate(vals.items()):
            h = int((v - min(vmin, 0)) / max(vmax - min(vmin, 0), 1e-9) * 150)
            x = 20 + n * bw
            col = (60, 60, 200) if (k.startswith("ortho") and best_high) else (150, 150, 150)
            cv2.rectangle(img, (x, 200 - h), (x + bw - 8, 200), col, -1)
            cv2.putText(img, k[:9], (x, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            cv2.putText(img, fmt.format(v), (x, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        return img

    p1 = bar_panel("ortho PSNR [dB] (higher better)", {k: v["psnr"] for k, v in results.items()})
    p2 = bar_panel("SSIM (higher better)", {k: v["ssim"] for k, v in results.items()}, "{:.3f}")
    p3 = bar_panel("no-data vs reference [%] (lower better)",
                   {k: 100 * v["nodata"] for k, v in results.items()}, "{:.2f}", best_high=False)
    p4 = bar_panel("floater ratio [%] (lower better)",
                   {k: 100 * v["floater_ratio"] for k, v in results.items()}, "{:.2f}", best_high=False)
    gap = np.full((p1.shape[0], 8, 3), 255, np.uint8)
    top = np.concatenate([p1, gap, p2], axis=1)
    bot = np.concatenate([p3, gap, p4], axis=1)
    cv2.imwrite(str(out / "ablation_budget.png"), np.concatenate([top, bot], axis=0))

    # ---- prune visual ---------------------------------------------------
    P = (full["means"].astype(np.float64) - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
    lo = P[:, :2].min(axis=0); hi = P[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    Wv = 900
    Hv = int(Wv * span[1] / span[0])
    vis = np.full((Hv, Wv, 3), 255, np.uint8)
    uv = ((P[:, :2] - lo) / span * np.array([Wv - 1, Hv - 1])).astype(int)
    uv[:, 1] = Hv - 1 - uv[:, 1]
    pm = base_prune["mask"]
    reason_img = np.zeros((Hv, Wv), np.uint8)
    for idx, code in ((~pm, 1), (pm, 2)):
        sel = idx
        if sel.sum():
            vis[uv[sel, 1], uv[sel, 0]] = (90, 90, 90) if code == 1 else (40, 40, 220)
            reason_img[uv[sel, 1], uv[sel, 0]] = code
    cv2.putText(vis, "kept (grey) vs pruned (red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(vis, f"pruned {pm.sum():,} / {n_full:,}  reasons={base_prune['reasons']}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(out / "prune_visual.png"), vis)

    # ---- metrics + P4 partial ------------------------------------------
    best = max(results.items(), key=lambda kv: kv[1]["psnr"])
    delta = {k: {"psnr_gain": v["psnr"] - results["conf"]["psnr"],
                 "ssim_gain": v["ssim"] - results["conf"]["ssim"]} for k, v in results.items()}
    metrics = {
        "run": run_dir.name, "seq": ds.name, "n_full": n_full, "n_max": a.n_max,
        "n_kfs": len(kfs), "stride": a.stride, "gsd_m_per_px": gsd, "grid": a.grid,
        "floater_before": floater_before, "floater_after_ortho2": results["ortho2"]["floater_ratio"],
        "prune": {k: v for k, v in base_prune.items() if k != "mask"},
        "results": results, "delta_vs_conf": delta, "best": best[0],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    md = ["# P4 (partial) — ortho-aware scoring & pruning", "",
          f"run `{run_dir.name}`, {len(kfs)} keyframes, stride {a.stride}, "
          f"**N_max = {a.n_max:,}** (full map {n_full:,}), grid {a.grid}", "",
          "| strategy | N kept | PSNR [dB] | ΔPSNR vs conf | SSIM | ΔSSIM | no-data | floater | edge kept |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        md.append(f"| {k} | {v['n_kept']:,} | {v['psnr']:.2f} | {delta[k]['psnr_gain']:+.2f} | "
                  f"{v['ssim']:.3f} | {delta[k]['ssim_gain']:+.3f} | {100*v['nodata']:.2f}% | "
                  f"{100*v['floater_ratio']:.2f}% | {100*v['edge_keep_ratio']:.0f}% |")
    ortho_keys = [k for k in results if k.startswith("ortho")]
    best_key = max(ortho_keys, key=lambda k: results[k]["psnr"]) if ortho_keys else "conf"
    acc_psnr = delta[best_key]["psnr_gain"]
    acc_ssim = delta[best_key]["ssim_gain"]
    md += ["", "## Gate", "",
           f"* Best ortho-aware policy: **`{best_key}`**",
           f"* PSNR gain of `{best_key}` vs `conf`: **{acc_psnr:+.2f} dB** "
           f"(target >= +0.50 dB)",
           f"* SSIM gain: **{acc_ssim:+.3f}** (target >= +0.02)",
           f"* floater ratio: {100*floater_before:.2f}% -> "
           f"{100*results['ortho2']['floater_ratio']:.2f}% "
           f"({100*(results['ortho2']['floater_ratio']/max(floater_before,1e-9)-1):+.0f}%)",
           f"* no-data: " + ", ".join(f"{k} {100*v['nodata']:.2f}%" for k, v in results.items()),
           "", "## Notes", "",
           "* `ortho2` = bounded coverage-gap × structural-detail × footprint cost on top of",
           "  `conf`; every modifier is clamped so the confidence term stays dominant",
           "  (the week-5 `ortho` collapse is documented in week-5 NOTES §4).",
           "* `ortho2_edge` additionally boosts the `kind==1` (structural edge) points.",
           "* `prune_mask` removes low-opacity / oversized / curtain (vertical) / isolated",
           "  floaters, protects structural edges, and a post-pass guarantees one kept point",
           "  per 0.4 m cell so pruning cannot create new holes.", ""]
    (out / "P4_PARTIAL.md").write_text("\n".join(md))
    print(f"[S7a] best={best[0]} PSNR={best[1]['psnr']:.2f}  -> {out}")
    return metrics


if __name__ == "__main__":
    main()
