#!/usr/bin/env python3
"""P5 production pipeline: accumulate keyframes, render ortho tiles, blend, export.

P5.1b (Tasks 5-6): accumulate_keyframes() renders each keyframe's Gaussians
onto ONE anchored canvas at the source-pixel GSD, then laplacian_blend()
fuses the per-keyframe canvases into ``mosaic_p5_1b.png``.

The canvas is anchored to the flight's plane extent -- the same convention as
the P4 cell mosaic -- so the blended result can be compared pixel-to-pixel.

Usage (env 2dgs-slam, from the Ortho-3dgs repo root):
    PYTHONPATH=third_party/MASt3R-SLAM:third_party/MASt3R-SLAM/thirdparty/mast3r \
    python -m stream_ortho.scripts.run_p5 --run results/system/p48_nocalib \
        --kfs 0-9 --out results/system/phase2_p5_3dgs/p5.1b_10kf
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.resolve()))

import argparse
import json
import time

import cv2
import numpy as np
import torch

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.geometry import sim3_from_vec8
from stream_ortho.filter.pointmap import FilterConfig

from stream_ortho.adapters.splatt3r import mast3r_anchored_gaussians
from stream_ortho.ortho.blend import laplacian_blend
from stream_ortho.ortho.gsplat_ortho import render_ortho_tile


def parse_kf_range(spec: str, n_available: int) -> list:
    """\"0-9\" / \"0,3,7\" / \"all\" -> keyframe index list."""
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return list(range(n_available))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < n_available]


def effective_fx(X_canon) -> float:
    from stream_ortho.frontend.predict import fit_effective_intrinsics
    K_guess = np.array([[0.5 * X_canon.shape[1], 0, 0.5 * X_canon.shape[1]],
                        [0, 0.5 * X_canon.shape[0], 0.5 * X_canon.shape[0]],
                        [0, 0, 1.]])
    eff = fit_effective_intrinsics(torch.as_tensor(X_canon, dtype=torch.float32), K_guess)
    return float(eff["x"]["f_eff"])


def source_gsd(X_canon, T_WC) -> float:
    """One MASt3R source pixel in plane units (median depth / fx_eff * Sim3 s)."""
    _, _, s = sim3_from_vec8(T_WC)
    return float(np.median(X_canon[..., 2])) / effective_fx(X_canon) * float(s)


def _load_dataset_for_run(run_dir: pathlib.Path, info: dict):
    ds_dir = pathlib.Path(info.get("dataset_dir", ""))
    if not ds_dir.exists():
        ds_dir = zcfg.DATA_DIR / info["dataset"]
    if (ds_dir / "config.cfg").exists():
        return NPUDroneMapDataset(str(ds_dir))
    from stream_ortho.data.rgb_folder import RGBFolderDataset
    return RGBFolderDataset(str(ds_dir), intrinsics="auto")


def _resolve_plane_aligner(run_dir: pathlib.Path, npz, kf_indices, ds):
    """Load alignment.json if present, or estimate leveled ProgressivePlaneAligner."""
    from stream_ortho.app.alignment import ProgressivePlaneAligner
    align_path = run_dir / "alignment.json"
    if align_path.exists():
        adata = json.loads(align_path.read_text())
        if "T_PW" in adata:
            T_PW = np.asarray(adata["T_PW"], dtype=np.float64)
            T_WP = np.linalg.inv(T_PW)
        elif "T_WP" in adata:
            # Legacy alignment.py saved World->Plane under "T_WP"
            T_PW = np.asarray(adata["T_WP"], dtype=np.float64)
            T_WP = np.linalg.inv(T_PW)
        else:
            T_WP = None
        if T_WP is not None:
            return ProgressivePlaneAligner(mode="plane", T_WP=T_WP)

    aligner = ProgressivePlaneAligner(mode="auto", bootstrap_kfs=min(2, len(kf_indices)))
    for k in kf_indices[:2]:
        R, t, sc = sim3_from_vec8(npz["T_WC"][k])
        pts = npz["X_canon"][k].reshape(-1, 3).astype(np.float64)
        Pw = float(sc) * (pts @ R.T) + t
        if aligner.add(Pw, cam_pos=t):
            break
    return aligner


def accumulate_keyframes(run_dir, kf_indices, ds, tile_pad_px: int = 16,
                         floor_mul: float = 1.0, device: str = "cuda:0",
                         log=print, snap_dir=None, collect_ply: bool = False,
                         conf_threshold: float = 0.5) -> dict:
    """Render every keyframe's Gaussians onto one anchored canvas."""
    from stream_ortho.scripts.export_3dgs_ply import quat_rotate_batch

    run_dir = pathlib.Path(run_dir)
    npz = np.load(run_dir / "keyframes.npz")
    n_kf = len(npz["frame_id"])
    kf_indices = [i for i in kf_indices if 0 <= i < n_kf]
    if not kf_indices:
        raise ValueError("no valid keyframe indices")

    aligner = _resolve_plane_aligner(run_dir, npz, kf_indices, ds)
    all_s = np.asarray(npz["T_WC"][kf_indices, 7], dtype=np.float64)
    med_s = float(np.median(all_s)) if all_s.size else 1.0

    # ---- pass 1: adapter per keyframe, collect leveled plane extents ----------
    per_kf = []
    ply_acc = {"means": [], "sh": [], "opac": [], "scales": [], "quats": [],
               "kf": []} if collect_ply else None
    for k in kf_indices:
        _, _, sc = sim3_from_vec8(npz["T_WC"][k])
        if sc < 0.45 * med_s:
            log(f"[run_p5] kf {k}: skipping scale-collapsed landing frame (s={sc:.3f} vs med {med_s:.3f})")
            continue
        frame_id = int(npz["frame_id"][k])
        img = ds.read_img(frame_id)
        t0 = time.time()
        filt = FilterConfig(enabled=conf_threshold > 0, min_conf=float(conf_threshold),
                            gate_only=True)
        g = mast3r_anchored_gaussians(
            img_u8=img, X_canon=npz["X_canon"][k], sim3_vec=npz["T_WC"][k],
            T_WP=aligner.T_WP_mat, keyframe_id=frame_id, device=device, filter_cfg=filt)
        means_np = g.means.detach().cpu().numpy().astype(np.float64)
        conf_np = g.confidence.detach().cpu().numpy().ravel()
        mp, T_PW_kf, _ = aligner.rectify_keyframe(k, means_np, conf=conf_np)
        ext = mp[:, :2].max(0) - mp[:, :2].min(0)
        per_kf.append({
            "kf": k, "frame_id": frame_id, "g": g, "mp": mp, "ext": ext,
            "T_WP_kf": np.linalg.inv(T_PW_kf),
            "centre": mp[:, :2].mean(0), "z_max": float(mp[:, 2].max()),
            "gsd": source_gsd(npz["X_canon"][k], npz["T_WC"][k]),
            "img": img, "t_adapter": time.time() - t0,
        })
        if ply_acc is not None:
            sc_lin = np.clip(g.scales.detach().cpu().numpy().astype(np.float32), 1e-7, None)
            op_lin = np.clip(g.opacity.detach().cpu().numpy().astype(np.float32), 1e-4, 1.0 - 1e-4)
            ply_acc["means"].append(mp.astype(np.float32))
            ply_acc["sh"].append(g.sh_dc.detach().cpu().numpy().astype(np.float32))
            ply_acc["opac"].append(np.log(op_lin / (1.0 - op_lin)).astype(np.float32))
            ply_acc["scales"].append(np.log(sc_lin).astype(np.float32))
            ply_acc["quats"].append(quat_rotate_batch(
                g.quats.detach().cpu().numpy().astype(np.float64), T_PW_kf[:3, :3]).astype(np.float32))
            ply_acc["kf"].append(np.full(len(means_np), frame_id, np.int32))
        log(f"[run_p5] kf {k} (frame {frame_id}): {g.means.shape[0]} gaussians, "
            f"ext {ext[0]:.2f}x{ext[1]:.2f}m, adapter {per_kf[-1]['t_adapter']:.2f}s")

    # ---- anchored canvas over the union extent (same convention as P4) ------
    gsd = float(np.median([p["gsd"] for p in per_kf]))
    x0 = float(min(p["mp"][:, 0].min() for p in per_kf))
    x1 = float(max(p["mp"][:, 0].max() for p in per_kf))
    y0 = float(min(p["mp"][:, 1].min() for p in per_kf))
    y1 = float(max(p["mp"][:, 1].max() for p in per_kf))
    W = int(np.ceil((x1 - x0) / gsd)) + 2 * tile_pad_px
    H = int(np.ceil((y1 - y0) / gsd)) + 2 * tile_pad_px
    log(f"[run_p5] canvas {H}x{W}px at gsd {gsd:.5f} (extent {x1-x0:.2f}x{y1-y0:.2f}m)")

    # canvas (cu, cv) <-> plane (x0 + cu*gsd, y1 - cv*gsd)
    tiles = []
    coverage = np.zeros((H, W), np.float32)
    for p in per_kf:
        # auto tile size: cover the keyframe's own extent at this gsd
        ext = p["ext"]
        size = int(np.ceil(max(ext[0], ext[1]) / gsd)) + tile_pad_px
        cx, cy = float(p["centre"][0]), float(p["centre"][1])
        t0 = time.time()
        rgb, alpha, _ = render_ortho_tile(
            p["g"], p["T_WP_kf"], cx, cy, p["z_max"], gsd=gsd, tile_size=size,
            scale_floor=float(np.sqrt(float(ext[0] * ext[1]) / p["g"].means.shape[0]) * floor_mul),
            device=device)
        t_render = time.time() - t0
        rgb_np = rgb.clamp(0, 1).cpu().numpy()
        a_np = alpha.squeeze().cpu().numpy()

        # paste position: tile top-left in plane = centre - size*gsd/2
        tl_x = cx - 0.5 * size * gsd
        tl_y = cy + 0.5 * size * gsd          # v axis is flipped
        cu = int(round((tl_x - x0) / gsd)) + tile_pad_px
        cv_ = int(round((y1 - tl_y) / gsd)) + tile_pad_px
        u0, u1 = max(cu, 0), min(cu + size, W)
        v0, v1 = max(cv_, 0), min(cv_ + size, H)
        if u1 <= u0 or v1 <= v0:
            log(f"[run_p5] kf {p['kf']}: tile fully outside canvas, skipped")
            continue
        canvas_rgb = np.zeros((H, W, 3), np.float32)
        canvas_a = np.zeros((H, W), np.float32)
        canvas_rgb[v0:v1, u0:u1] = rgb_np[v0 - cv_:v1 - cv_, u0 - cu:u1 - cu]
        canvas_a[v0:v1, u0:u1] = a_np[v0 - cv_:v1 - cv_, u0 - cu:u1 - cu]
        coverage = np.maximum(coverage, canvas_a)
        tiles.append({"kf": p["kf"], "frame_id": p["frame_id"], "rgb": canvas_rgb,
                      "alpha": canvas_a, "tile_size": size, "centre": [cx, cy],
                      "z_max": p["z_max"], "gsd": gsd, "t_render": t_render,
                      "paste": [u0, v0, u1, v1]})
        if snap_dir is not None:
            # content-cropped snapshot (over black) + box, for the update video
            from stream_ortho.ortho.mosaic import content_crop
            crop_arr, box = content_crop(canvas_rgb, canvas_a)
            snap_dir.mkdir(parents=True, exist_ok=True)
            import cv2 as _cv
            sp = snap_dir / f"snap_kf{p['kf']:04d}.png"
            _cv.imwrite(str(sp), _cv.cvtColor(
                (np.clip(crop_arr, 0, 1) * 255).astype(np.uint8), _cv.COLOR_RGB2BGR))
            tiles[-1]["snap"] = str(sp)
            tiles[-1]["snap_box"] = [int(v) for v in box]
        log(f"[run_p5] kf {p['kf']}: tile {size}px rendered {t_render:.2f}s, "
            f"canvas coverage -> {(coverage > 0.5).mean():.3f}")

    out = {"tiles": tiles, "coverage": coverage, "gsd": gsd,
           "plane_extent": [x0, y0, x1, y1], "canvas_hw": [H, W],
           "tile_pad_px": tile_pad_px}
    if ply_acc is not None:
        out["ply"] = {k: (np.concatenate(v) if v else np.zeros((0, *()), dtype=np.float32))
                      for k, v in ply_acc.items()}
        out["ply"]["kf"] = np.concatenate(ply_acc["kf"]) if ply_acc["kf"] else np.zeros((0,), np.int32)
    return out


def blend_and_save(acc, out_dir, levels: int = 4, feather_px: int = 24,
                   name: str = "mosaic_p5_1b", log=print) -> dict:
    """Task 6: laplacian_blend over the accumulated canvases, save the mosaic."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rgb, wsum = laplacian_blend([t["rgb"] for t in acc["tiles"]],
                                [t["alpha"] for t in acc["tiles"]],
                                levels=levels, feather_px=feather_px)
    t_blend = time.time() - t0
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    a8 = (np.clip(wsum, 0, 1) * 255).astype(np.uint8)
    out_png = out_dir / f"{name}.png"
    cv2.imwrite(str(out_png), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{name}_alpha.png"), a8)

    # content metrics (solid coverage only)
    solid = wsum > 0.95
    ys, xs = np.where(solid)
    metrics = {"n_tiles": len(acc["tiles"]), "blend_levels": levels,
               "feather_px": feather_px, "t_blend_s": t_blend,
               "gsd": acc["gsd"], "canvas_hw": acc["canvas_hw"],
               "coverage_frac": float((acc["coverage"] > 0.5).mean()),
               "solid_frac": float(solid.mean())}
    if len(ys) > 100:
        g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sub = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        msub = solid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        gx = cv2.Sobel(sub, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.abs(gx) + np.abs(gy)
        metrics["content_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        metrics["content_hw"] = [int(sub.shape[0]), int(sub.shape[1])]
        metrics["grad_mean"] = float(grad[msub].mean())
        metrics["grad_p95"] = float(np.percentile(grad[msub], 95))
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"[run_p5] blend {t_blend:.1f}s -> {out_png}")
    if "grad_mean" in metrics:
        log(f"[run_p5] content {metrics['content_hw']} grad {metrics['grad_mean']:.2f} "
            f"(p95 {metrics['grad_p95']:.1f})")
    return metrics


def render_3dgs_final(run_dir, out_dir=None, gsd: float = 0.0, conf_threshold: float = 0.5,
                      blend_levels: int = 4, feather_px: int = 24, floor_mul: float = 1.0,
                      max_kf: int = 0, device: str = "cuda:0", log=print,
                      video: bool = True, ply: bool = True) -> dict:
    """P5.1c entry point: 3DGS final render from a finished run directory.

    Reads ``run_dir/keyframes.npz`` (+ ``run.json`` for the dataset), renders
    every keyframe onto one anchored canvas, blends, and writes:

        mosaic_final_3dgs.png   high-resolution 3DGS ortho
        mosaic_final_3dgs_alpha.png
        ortho_3dgs.json         metadata (gsd, canvas, timings, coverage)
        ground_3dgs.ply         standard 3DGS schema (plane frame, Z up)
        ortho_update_3dgs.mp4   per-keyframe accumulation clip
        snaps/                  per-keyframe content snapshots (video sources)

    Returns the metadata dict (also what the e2e result embeds under "3dgs").
    """
    import cv2

    from stream_ortho.scripts.export_3dgs_ply import write_gaussian_ply

    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir) if out_dir else run_dir
    npz = np.load(run_dir / "keyframes.npz")
    n_available = len(npz["frame_id"])
    kfs = list(range(min(max_kf, n_available)) if max_kf else range(n_available))
    info = json.loads((run_dir / "run.json").read_text())
    ds = _load_dataset_for_run(run_dir, info)

    log(f"[3dgs] {len(kfs)} keyframes from {run_dir}")
    t0 = time.time()
    acc = accumulate_keyframes(run_dir, kfs, ds, floor_mul=floor_mul, device=device,
                               log=log, snap_dir=(out_dir / "snaps") if video else None,
                               collect_ply=ply, conf_threshold=conf_threshold)
    t_accum = time.time() - t0
    metrics = blend_and_save(acc, out_dir, levels=blend_levels, feather_px=feather_px,
                             name="mosaic_final_3dgs", log=log)
    t_blend = metrics.pop("t_blend_s", 0.0)

    # PLY: all keyframes' plane-frame gaussians (no per-kf filtering beyond the
    # adapter's own gates -- keep the export faithful to what was rendered)
    ply_info = None
    if ply and "ply" in acc:
        p = acc["ply"]
        ply_info = write_gaussian_ply(
            out_dir / "ground_3dgs.ply", p["means"], p["sh"], p["opac"], p["scales"],
            p["quats"], p["kf"],
            meta={"coordinate system": "plane (Z up), same frame as the ortho mosaic",
                  "gsd_m_per_px": f"{acc['gsd']:.6f}",
                  "n_keyframes": len(kfs),
                  "scale_reference": info.get("dataset", "")})
        log(f"[3dgs] PLY {ply_info['n']} gaussians, {ply_info['bytes']/1e6:.1f} MB")

    video_info = None
    if video:
        from stream_ortho.app.artifacts import write_update_video
        snaps = [(t["snap"], t.get("snap_box")) for t in acc["tiles"] if "snap" in t]
        video_info = write_update_video(out_dir / "ortho_update_3dgs.mp4",
                                        [s[0] for s in snaps], None, 0,
                                        boxes=[s[1] for s in snaps])
        log(f"[3dgs] video: {video_info}")

    meta = {"n_keyframes": len(kfs), "gsd": acc["gsd"], "canvas_hw": acc["canvas_hw"],
            "plane_extent": acc["plane_extent"], "t_accum_s": t_accum,
            "t_blend_s": t_blend, "t_total_s": time.time() - t0,
            "blend_levels": blend_levels, "feather_px": feather_px,
            "coverage_frac": metrics.get("coverage_frac"),
            "solid_frac": metrics.get("solid_frac"),
            "content_grad": metrics.get("grad_mean"), "content_grad_p95": metrics.get("grad_p95"),
            "content_hw": metrics.get("content_hw"),
            "ply": ply_info, "video": video_info}
    with open(out_dir / "ortho_3dgs.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--kfs", default="all")
    ap.add_argument("--out", default=str(zcfg.RESULTS_DIR / "system" / "phase2_p5_3dgs" / "p5.1b_10kf"))
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--feather-px", type=int, default=24)
    ap.add_argument("--floor-mul", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gsd", type=float, default=0.0)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--max-kf", type=int, default=0)
    ap.add_argument("--final", action="store_true",
                    help="P5.1c production path: render_3dgs_final (mosaic+PLY+json+video)")
    a = ap.parse_args(argv)

    if a.final:
        meta = render_3dgs_final(a.run, out_dir=a.out, gsd=a.gsd, conf_threshold=a.conf,
                                 blend_levels=a.levels, feather_px=a.feather_px,
                                 floor_mul=a.floor_mul, max_kf=a.max_kf, device=a.device)
        print(f"[run_p5] FINAL DONE: {json.dumps(meta, default=str)}")
        return 0

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    ds = _load_dataset_for_run(run_dir, info)
    npz = np.load(run_dir / "keyframes.npz")
    kfs = parse_kf_range(a.kfs, len(npz["frame_id"]))
    print(f"[run_p5] {len(kfs)} keyframes from {run_dir}")

    acc = accumulate_keyframes(run_dir, kfs, ds, floor_mul=a.floor_mul,
                               device=a.device)
    metrics = blend_and_save(acc, a.out, levels=a.levels, feather_px=a.feather_px)
    print(f"[run_p5] DONE: {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
