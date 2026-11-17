"""S3 stage: feed-forward Gaussians on keyframes + perspective re-render.

For each selected keyframe of an S2 run:

1. re-encode the image (deterministic) and run the Gaussian head;
2. render the keyframe with gsplat in pinhole mode (identity view matrix);
3. compare against the resized input image (PSNR / SSIM / coverage).

Variants measured (week-2 questions):

* anchor: ``head`` (head's own point map, what Splatt3R was trained with) vs
  ``mast3r`` (MASt3R-SLAM's fused ``X_canon``);
* colour: ``sh`` (learned SH DC only) vs ``sh+image`` (Splatt3R's inference
  behaviour: SH = sh + RGB2SH(image));
* scale: ``raw`` vs ``pixel`` (multiply by the pixel footprint depth/f).

Outputs go to ``results/stage-S3/<run>/``:
    metrics.json, REPORT.md, kfXX_{target,render,alpha,error}.png,
    gaussians_kfXX.npz (only with --dump), contactsheet.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend import predict as P
from stream_ortho.frontend.gaussian_head import load_gaussian_head


def build_parser():
    p = argparse.ArgumentParser(description="S3: Gaussian head + perspective re-render")
    p.add_argument("--run", required=True, help="stage-S2 run directory (keyframes.npz)")
    p.add_argument("--seq", default=None, help="sequence dir (default: from run.json + data root)")
    p.add_argument("--out", default=None)
    p.add_argument("--n-kf", type=int, default=3, help="number of keyframes to test")
    p.add_argument("--kf", default="auto", help="'auto' or comma-separated keyframe indices")
    p.add_argument("--dump", action="store_true", help="dump Gaussian params of the first keyframe")
    p.add_argument("--device", default=zcfg.DEVICE)
    p.add_argument("--max-points", type=int, default=0, help="stride-subsample Gaussians (0 = all)")
    return p


def select_keyframes(npz, n_kf, spec):
    T = npz["T_WC"]
    s = T[:, 7] if T.shape[1] >= 8 else np.ones(len(T))
    med = float(np.median(s))
    cruise = np.where(np.abs(np.log(s / med)) < np.log(1.3))[0]      # skip take-off/landing
    if len(cruise) < n_kf:
        cruise = np.arange(len(T))
    if spec != "auto":
        return [int(x) for x in str(spec).split(",")]
    pick = np.linspace(0, len(cruise) - 1, n_kf).round().astype(int)
    return [int(cruise[i]) for i in sorted(set(pick))]


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = pathlib.Path(args.run)
    info = json.loads((run_dir / "run.json").read_text())
    seq_dir = pathlib.Path(args.seq) if args.seq else zcfg.dataset_root() / info["dataset"]
    out = pathlib.Path(args.out) if args.out else zcfg.RESULTS_DIR / "stage-S3" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    npz = np.load(run_dir / "keyframes.npz")
    X, C, T = npz["X_canon"], npz["C"], npz["T_WC"]
    fids = npz["frame_id"]
    K = np.asarray(npz["K"], dtype=np.float64)
    H, W = int(npz["hw"][0]), int(npz["hw"][1])

    ds = NPUDroneMapDataset(seq_dir)
    kf_idx = select_keyframes(npz, args.n_kf, args.kf)
    print(f"[S3] run={run_dir.name} seq={ds.name} keyframes={kf_idx} frame={H}x{W}")

    model = P.load_model(device=args.device)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=args.device)

    rows = []
    dumps = {}
    for k in kf_idx:
        fid = int(fids[k])
        img = ds.read_img(fid)
        enc = P.encode_image(model, img, device=args.device)
        if tuple(enc["shape_hw"]) != (H, W):
            print(f"[S3][warn] kf{k}: feature shape {enc['shape_hw']} != npz {(H, W)}")
        g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])

        target = enc["uimg"].astype(np.float32) / 255.0
        # diagnostics: head point map vs MASt3R-SLAM point map
        mu_head = g["mu_cam"].reshape(-1, 3)
        x_slam = torch.as_tensor(X[k].reshape(-1, 3), device=mu_head.device, dtype=torch.float32)
        rel = ((mu_head[:, 2] - x_slam[:, 2]).abs() / x_slam[:, 2].clamp_min(1e-6))
        sc = g["scales"]
        aniso = (sc.max(dim=-1).values / sc.min(dim=-1).values.clamp_min(1e-9))
        qn = g["quats"].norm(dim=-1)
        depth_stats = {
            "head_depth_median": float(mu_head[:, 2].median()),
            "slam_depth_median": float(x_slam[:, 2].median()),
            "rel_depth_diff_median": float(rel.median()),
            "scale_median": float(sc.median()),
            "scale_aniso_median": float(aniso.median()),
            "scale_aniso_p95": float(aniso.flatten().quantile(0.95)),
            "opacity_mean": float(g["opacity"].mean()),
            "opacity_frac_gt_0_99": float((g["opacity"] > 0.99).float().mean()),
            "quat_norm_max_err": float((qn - 1).abs().max()),
            "conf_median": float(g["conf"].median()),
            "reproj_head": P.reprojection_error_px(g["mu_cam"], K),
            "reproj_slam": P.reprojection_error_px(
                torch.as_tensor(X[k], device=g["mu_cam"].device, dtype=torch.float32), K),
            "effective_intrinsics_head": P.fit_effective_intrinsics(g["mu_cam"], K),
        }

        mu_raw = {
            "head": g["mu_cam"],
            "mast3r": torch.as_tensor(X[k], device=g["mu_cam"].device, dtype=torch.float32),
        }
        mu_constrained = {k2: P.constrain_to_rays(v, K) for k2, v in mu_raw.items()}
        for anchor in ("head", "mast3r"):
          for constrain in (True, False):
            mu = mu_constrained[anchor] if constrain else mu_raw[anchor]
            depth = mu[..., 2:3].clamp_min(1e-6)
            for scale_mode in ("raw", "pixel"):
                sc = P.rescale(g["scales"], scale_mode, depth, K)
                for sh_mode in ("sh", "sh+image", "image"):
                    if sh_mode == "image":
                        rgb = torch.as_tensor(target, device=g["sh"].device)
                    elif sh_mode == "sh+image":
                        rgb = P.sh_to_rgb(g["sh"] + P.rgb_to_sh(
                            torch.as_tensor(target, device=g["sh"].device)))
                    else:
                        rgb = P.sh_to_rgb(g["sh"])
                    img_out, alpha, _ = P.render_perspective(
                        mu, sc, g["quats"], g["opacity"], rgb, K, H, W)
                    pred = img_out[0].clamp(0, 1).cpu().numpy()
                    row = {
                        "kf": int(k),
                        "frame_id": fid,
                        "anchor": anchor,
                        "constrain": bool(constrain),
                        "scale": scale_mode,
                        "color": sh_mode,
                        "psnr": P.psnr(pred, target),
                        "ssim": P.ssim(pred, target),
                        "coverage": float(alpha.mean()),
                        **depth_stats,
                    }
                    rows.append(row)
                    if anchor == "head" and sh_mode == "image" and scale_mode == "raw" and constrain:
                        vis = {
                            "target": target,
                            "render": pred,
                            "alpha": alpha[0, :, :, 0].clamp(0, 1).cpu().numpy(),
                            "error": np.abs(pred - target).mean(axis=-1),
                        }

        if args.dump and not dumps:
            dumps["gaussians"] = {
                "mu_cam": g["mu_cam"].cpu().numpy().astype(np.float16),
                "scales": g["scales"].cpu().numpy().astype(np.float16),
                "quats": g["quats"].cpu().numpy().astype(np.float16),
                "opacity": g["opacity"].cpu().numpy().astype(np.float16),
                "sh": g["sh"].cpu().numpy().astype(np.float16),
                "conf": g["conf"].cpu().numpy().astype(np.float16),
                "X_canon": X[k].astype(np.float16),
                "T_WC": T[k],
                "K": K,
                "frame_id": fid,
                "hw": np.array([H, W]),
            }

        save_vis(vis, out / f"kf{int(fid):04d}_compare.png")
        print(f"[S3] kf{k} (frame {fid}): best PSNR "
              f"{max(r['psnr'] for r in rows if r['kf'] == k):.2f} dB  "
              f"head_depth={depth_stats['head_depth_median']:.3f} "
              f"slam_depth={depth_stats['slam_depth_median']:.3f}")

    if dumps:
        np.savez(out / "gaussians_kf.npz", **dumps["gaussians"])
        print(f"[S3] dumped Gaussians -> {out/'gaussians_kf.npz'}")

    metrics = {
        "run": run_dir.name,
        "seq": ds.name,
        "n_keyframes_tested": len(kf_idx),
        "keyframes": kf_idx,
        "hw": [H, W],
        "K": K.tolist(),
        "rows": rows,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics, out)
    make_contact_sheet(out)
    print(f"[S3] -> {out}")


def save_vis(vis, path):
    import cv2

    t, r = vis["target"], vis["render"]
    a = np.repeat(vis["alpha"][..., None], 3, axis=-1)
    e = np.repeat(vis["error"][..., None], 3, axis=-1)
    e = np.clip(e * 4, 0, 1)
    gap = np.ones((t.shape[0], 6, 3), np.float32)
    strip = np.concatenate([t, gap, r, gap, a, gap, e], axis=1)
    cv2.imwrite(str(path), (np.clip(strip, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])


def make_contact_sheet(out: pathlib.Path):
    import cv2

    files = sorted(out.glob("kf*_compare.png"))
    if not files:
        return
    imgs = [cv2.imread(str(f)) for f in files]
    w = min(i.shape[1] for i in imgs)
    imgs = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in imgs]
    cv2.imwrite(str(out / "contactsheet.png"), np.concatenate(imgs, axis=0))


def write_report(metrics, out: pathlib.Path):
    rows = metrics["rows"]
    lines = [
        f"# S3 report — {metrics['run']} ({metrics['seq']})",
        "",
        f"keyframes tested: {metrics['keyframes']}  frame {metrics['hw'][1]}x{metrics['hw'][0]}",
        "",
        "Perspective re-render of the same keyframe (PSNR / SSIM / coverage).",
        "",
        "| kf | anchor | rays | scale | color | PSNR [dB] | SSIM | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["kf"], -x["psnr"])):
        lines.append(
            f"| {r['kf']} | {r['anchor']} | {'yes' if r.get('constrain') else 'no'} | "
            f"{r['scale']} | {r['color']} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['coverage']:.3f} |"
        )
    if rows:
        best = max(rows, key=lambda x: x["psnr"])
        lines += ["", f"best: kf{best['kf']} anchor={best['anchor']} "
                      f"rays={best.get('constrain')} scale={best['scale']} "
                      f"color={best['color']} -> {best['psnr']:.2f} dB"]
        lines += ["", "depth medians (head vs MASt3R-SLAM point map):"]
        for k in sorted({r["kf"] for r in rows}):
            r = min((x for x in rows if x["kf"] == k), key=lambda x: x["anchor"])
            lines.append(f"* kf{k}: head={r['head_depth_median']:.3f}, "
                         f"slam={r['slam_depth_median']:.3f}, "
                         f"|Δz|/z median={r['rel_depth_diff_median']*100:.2f}%")
    lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
