"""Why does the head patch land at the wrong scale? (week-4 parallel investigation)

Compares, per pixel, the head's ray depth with the *geometric* depth of the ray
intersecting the ground plane.  If the ratio is constant, a per-keyframe scale
is enough; if it grows towards the image edges, the head's depth map is too flat
(MASt3R under-estimates the 1/cos(theta) growth) and a **plane-constrained depth**
is required for the ortho path.
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
from stream_ortho.geometry import compose_sim3, sim3_from_vec8


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--kf", type=int, default=-1)
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "stage-S5a" / "head_depth"))
    a = p.parse_args(argv)

    run = pathlib.Path(a.run)
    info = json.loads((run / "run.json").read_text())
    ds = NPUDroneMapDataset(pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"])
    npz = np.load(run / "keyframes.npz")
    m = json.loads((run / "sim3_est_to_gt.json").read_text())
    from stream_ortho.geometry import R_to_quat_xyzw
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    kf = a.kf if a.kf >= 0 else len(npz["frame_id"]) // 2

    model = P.load_model()
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, verbose=False)
    fid = int(npz["frame_id"][kf])
    enc = P.encode_image(model, ds.read_img(fid))
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    K = ds.K_frame()
    H, W = enc["shape_hw"]
    mu = P.constrain_to_rays(g["mu_cam"], K).detach().cpu().numpy()
    d_head = mu[..., 2]

    # geometric depth: ray x ground plane (z_ground = 0 in the plane frame)
    T_gt_cam = compose_sim3(npz["T_WC"][kf], sg)
    R, t, s = sim3_from_vec8(T_gt_cam)
    n = ds.T_WP[:3, 2]
    p0 = ds.T_WP[:3, 3]
    dist_cam = float(((p0 - t) @ n) / ((R[:, 2]) @ n))    # metric distance along the optical axis
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    dirs_w = rays @ R.T
    cosang = np.abs(dirs_w @ n) / np.linalg.norm(dirs_w, axis=-1)
    d_plane = dist_cam / np.maximum(cosang, 1e-6)          # metric depth of the plane along each ray

    # head depth in metres: multiply by the Sim3 scale s of the pose chain
    d_head_m = d_head * float(s)
    ratio = d_plane / d_head_m
    rows = {}
    for name, (uu0, vv0) in {
        "centre": (W // 2, H // 2),
        "right_edge": (W - 8, H // 2),
        "bottom_edge": (W // 2, H - 8),
        "corner": (W - 8, H - 8),
    }.items():
        rows[name] = {
            "ray_angle_deg": float(np.degrees(np.arccos(cosang[vv0, uu0]))),
            "d_head_m": float(d_head_m[vv0, uu0]),
            "d_plane_m": float(d_plane[vv0, uu0]),
            "ratio": float(ratio[vv0, uu0]),
        }
    out = {
        "run": run.name, "seq": ds.name, "kf": int(kf), "frame_id": fid,
        "camera_distance_m": dist_cam,
        "head_depth_m_median": float(np.median(d_head_m)),
        "plane_depth_m_median": float(np.median(d_plane)),
        "ratio_median": float(np.median(ratio)),
        "ratio_p5": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "ratio_range": float(np.percentile(ratio, 95) - np.percentile(ratio, 5)),
        "samples": rows,
        "verdict": None,
    }
    spread = out["ratio_p95"] / max(out["ratio_p5"], 1e-9)
    out["verdict"] = ("constant-scale ok" if spread < 1.10 else
                      "head depth too flat -> use plane-constrained depth")
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{run.name}_kf{kf}.json").write_text(json.dumps(out, indent=2))
    print(f"[diag] {run.name} kf{kf} fid{fid}: camera distance {dist_cam:.1f} m")
    print(f"[diag] ratio plane/head: median {out['ratio_median']:.3f} "
          f"p5 {out['ratio_p5']:.3f} p95 {out['ratio_p95']:.3f} (spread {spread:.2f}x)")
    for name, r in rows.items():
        print(f"   {name:12s} angle {r['ray_angle_deg']:5.1f} deg  "
              f"head {r['d_head_m']:6.1f} m  plane {r['d_plane_m']:6.1f} m  ratio {r['ratio']:.3f}")
    print(f"[diag] verdict: {out['verdict']}")
    return out


if __name__ == "__main__":
    main()
