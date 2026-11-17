"""S31 -- P0 acceptance: roof-placement accuracy, legacy vs fixed.

For a few keyframes we compute, on the *same* rays, four placements:

* ``ground``  : the ray x plane(z=0) hit            (a roof can never be reached),
* ``legacy``  : the old fixed-point DSM iteration   (what the products used),
* ``raycast`` : the new nearest-hit ray march       (P0-2),
* ``truth``   : the point map itself               (upper bound of the metric).

and score each with :func:`stream_ortho.map.placement.roof_placement_accuracy`.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.frontend.gaussian_head import load_gaussian_head
from stream_ortho.frontend.predict import load_model
from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8
from stream_ortho.map.build import raycast_dsm
from stream_ortho.map.placement import roof_placement_accuracy
from stream_ortho.ortho.dsm import build_piecewise_dsm


def head_grid_rays(K, H, W, h, w, R):
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * (W / float(w)),
                         (np.arange(h) + 0.5) * (H / float(h)), indexing="xy")
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    return rays, uu, vv


def legacy_depth(dsm, T_WP, t_cam, dirs):
    """The old implementation: fixed point starting at the z=0 hit."""
    import cv2  # noqa: F401  (kept for parity with build.py)

    n = T_WP[:3, 2]
    p0 = T_WP[:3, 3]
    den = dirs @ n
    den = np.where(np.abs(den) > 1e-9, den, np.nan)
    tt = np.broadcast_to(((p0 - t_cam) @ n) / den, dirs.shape[:-1]).copy()
    for _ in range(4):
        P = t_cam[None, None, :] + tt[..., None] * dirs
        Pp = (P - p0) @ T_WP[:3, :3]
        zz = dsm.lookup(Pp[..., 0], Pp[..., 1], fill=0.0)
        tt_new = ((p0 + n[None, None, :] * zz[..., None] - t_cam) @ n) / den
        tt = np.where(np.isfinite(tt_new) & (tt_new > 0), tt_new, tt)
    return tt


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--kf-frac", default="0.35,0.65")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dsm-kf", type=int, default=32)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    run_dir = pathlib.Path(a.run)
    info = json.loads((run_dir / "run.json").read_text())
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset

    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(run_dir / "keyframes.npz")
    m = json.loads((run_dir / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()
    dsm_kfs = kfs if a.dsm_kf >= len(kfs) else kfs[:: max(1, len(kfs) // a.dsm_kf)]
    dsm = build_piecewise_dsm(ds, npz, sg, dsm_kfs, gsd=0.5)
    print(f"[s31] DSM {dsm.shape} valid {100*float(np.isfinite(dsm.z).mean()):.1f}% "
          f"ground {float(np.median(dsm.z)):.2f} m")
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)

    fr = [float(x) for x in a.kf_frac.split(",")]
    picks = [kfs[min(len(kfs) - 1, int(f * len(kfs)))] for f in fr]
    rows = []
    for k in picks:
        fid = int(npz["frame_id"][k])
        img = ds.read_img(fid)
        H, W = img.shape[:2]
        from stream_ortho.frontend import predict as P

        enc = P.encode_image(model, img, device=zcfg.DEVICE)
        g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
        h, w = int(enc["shape_hw"][0]), int(enc["shape_hw"][1])
        R, t, sv = sim3_from_vec8(compose_sim3(T[k], sg))
        K = ds.K_orig
        rays, _, _ = head_grid_rays(K, H, W, h, w, R)
        dirs = rays @ R.T
        # truth: point map lifted to the plane frame
        X = np.asarray(npz["X_canon"][k], np.float32).reshape(-1, 3) * float(sv)
        P_pm = X @ R.T + t
        P_pm = (P_pm - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3]
        # ground: ray x z=0
        n = ds.T_WP[:3, 2]; p0 = ds.T_WP[:3, 3]
        den = dirs @ n
        t0 = ((p0 - t) @ n) / np.where(np.abs(den) > 1e-9, den, np.nan)
        Pg = t + t0[..., None] * dirs
        Pg = (Pg - p0) @ ds.T_WP[:3, :3]
        # legacy / raycast
        tl = legacy_depth(dsm, ds.T_WP, t, dirs)
        Pl = t + tl[..., None] * dirs
        Pl = (Pl - p0) @ ds.T_WP[:3, :3]
        tr, hit = raycast_dsm(dsm, ds.T_WP, t, dirs)
        Pr = t + tr[..., None] * dirs
        Pr = (Pr - p0) @ ds.T_WP[:3, :3]
        for tag, Pl_ in (("ground", Pg), ("legacy", Pl), ("raycast", Pr), ("truth", P_pm)):
            acc = roof_placement_accuracy(P_pm, Pl_.reshape(-1, 3), window=60.0)
            acc["kf"] = fid
            acc["variant"] = tag
            rows.append(acc)
            print(f"[s31] kf{fid:3d} {tag:8s} roof {acc.get('roof_frac', 0):5.1%} "
                  f"acc_z {acc['acc_z']:5.1%} acc_xy {acc['acc_xy']:5.1%} "
                  f"both {acc['acc_both']:5.1%} med|dz| {acc['med_dz_roof']:.1f} m")
    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S31" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "roof_placement.json").write_text(json.dumps(rows, indent=2))
    print(f"[s31] -> {out}/roof_placement.json")


if __name__ == "__main__":
    main()
