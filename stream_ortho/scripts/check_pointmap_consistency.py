"""S2 diagnostic: is ``T_WC.act(X_canon)`` a *common* world frame?

The design doc's S2 acceptance asks us to "export keyframe point clouds + poses
and confirm the world coordinates of ``T_WC.act(X_canon)`` are consistent".  We
never write a point cloud (hard constraint 2.5); instead we check the geometry
directly from the dumped keyframe tensors.

For a keyframe pair ``(i, j)`` with overlapping views:

1. map keyframe ``i``'s point map into world:  ``p_W = s_i R_i X_i + t_i``
2. map it into camera ``j``:                   ``p_C = R_j^T (p_W - t_j) / s_j``
3. project with ``K_j`` and compare ``p_C.z`` with keyframe ``j``'s own point-map
   depth at the same pixel.

If all keyframes live in one common world frame, the two depths agree up to
MASt3R's noise.  If every packet had been given its own frame (the failure mode
documented in design-doc 2.4), the disagreement grows with the perturbation.

The **control** applies a random Sim3 perturbation (translation + rotation, the
shape of a per-packet alignment error) to every n-th keyframe and re-measures,
sweeping the perturbation magnitude.  A useful diagnostic must degrade
monotonically.

Outputs (``--out``):
    consistency.json
    consistency.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho.data.plane import quat_xyzw_to_matrix


def build_parser():
    p = argparse.ArgumentParser(description="S2: point-map world consistency")
    p.add_argument("--run", required=True, help="stage-S2 run directory")
    p.add_argument("--out", default=None, help="output dir (default: <run>/consistency)")
    p.add_argument("--n-pairs", type=int, default=40, help="max pairs to evaluate")
    p.add_argument("--max-gap", type=int, default=2,
                   help="max keyframe index gap (NPU keyframes are ~8 frames apart, "
                        "so only gap<=2 has real overlap)")
    p.add_argument("--conf", type=float, default=1.5, help="confidence threshold")
    p.add_argument("--nn-samples", type=int, default=4000)
    p.add_argument("--perturb-every", type=int, default=1,
                   help="control: perturb every k-th keyframe (1 = every keyframe, i.e. "
                        "the per-packet-frames failure mode)")
    p.add_argument("--perturb-t", type=float, default=2.0,
                   help="control fallback: max translation [m]")
    p.add_argument("--perturb-deg", type=float, default=1.0,
                   help="control fallback: max rotation [deg]")
    p.add_argument("--perturb-levels", type=str, default="0.005,0.01,0.02,0.05,0.1",
                   help="control sweep: translation perturbation as a FRACTION of the "
                        "scene scale (non-metric units), rotation = frac radians")
    p.add_argument("--seed", type=int, default=0)
    return p


# ---------------------------------------------------------------------------
# Sim3 helpers (lietorch layout: tx ty tz qx qy qz qw s)
# ---------------------------------------------------------------------------

def sim3_decompose(vec8):
    vec8 = np.asarray(vec8, dtype=np.float64).reshape(-1)
    if vec8.size == 7:                       # older dumps: no scale
        t, q, s = vec8[:3], vec8[3:7], 1.0
    else:
        t, q, s = vec8[:3], vec8[3:7], float(vec8[7])
    return quat_xyzw_to_matrix(q), t, s


def act(R, t, s, pts):
    return s * (pts @ R.T) + t


def act_inv(R, t, s, pts):
    return ((pts - t) / s) @ R


def random_perturbation(rng, max_t, max_deg):
    ax = rng.normal(size=3)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    ang = np.deg2rad(rng.uniform(0, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    t = rng.uniform(-max_t, max_t, size=3)
    return R, t


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

def project_pair(Xi, Ci, pose_i, Xj, Cj, pose_j, K, conf, nn_samples, rng):
    """Depth-reprojection agreement + self-normalised NN distance for one pair."""
    h, w = Ci.shape
    Ri, ti, si = pose_i
    Rj, tj, sj = pose_j

    pts_i = Xi.reshape(-1, 3)
    conf_i = Ci.reshape(-1)
    m_i = (conf_i > conf) & np.isfinite(pts_i).all(axis=1) & (pts_i[:, 2] > 1e-3)
    if m_i.sum() < 100:
        return None
    pts_i = pts_i[m_i]

    p_w = act(Ri, ti, si, pts_i)
    p_c = act_inv(Rj, tj, sj, p_w)

    z = p_c[:, 2]
    front = z > 1e-3
    if front.sum() < 100:
        return None
    p_c, z = p_c[front], z[front]

    uv = p_c @ K.T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inside.sum() < 100:
        return None
    u_i = np.clip(np.round(u[inside]).astype(np.int64), 0, w - 1)
    v_i = np.clip(np.round(v[inside]).astype(np.int64), 0, h - 1)
    z = z[inside]

    Xj_flat = Xj.reshape(-1, 3)
    Cj_flat = Cj.reshape(-1)
    idx = v_i * w + u_i
    z_j = Xj_flat[idx, 2]
    conf_j = Cj_flat[idx]
    ok = (conf_j > conf) & np.isfinite(z_j) & (z_j > 1e-3)
    if ok.sum() < 50:
        return None
    z, z_j = z[ok], z_j[ok]

    rel = np.abs(z - z_j) / np.maximum(z_j, 1e-9)
    depth_med = float(np.median(z_j))

    # --- point-set agreement in world coordinates ----------------------
    from scipy.spatial import cKDTree

    m_j = (Cj_flat > conf) & np.isfinite(Xj_flat).all(axis=1)
    p_wj = act(Rj, tj, sj, Xj_flat[m_j])
    if len(p_wj) < 200:
        return None
    sel_j = rng.choice(len(p_wj), size=min(4 * nn_samples, len(p_wj)), replace=False)
    p_wj_s = p_wj[sel_j]
    tree = cKDTree(p_wj_s)
    sel_i = rng.choice(len(p_w), size=min(nn_samples, len(p_w)), replace=False)
    d, _ = tree.query(p_w[sel_i], k=1)
    sel_self = rng.choice(len(p_wj_s), size=min(nn_samples, len(p_wj_s)), replace=False)
    d_self, _ = tree.query(p_wj_s[sel_self], k=2)
    spacing = float(np.median(d_self[:, 1]))

    return {
        "n": int(ok.sum()),
        "rel_depth_median": float(np.median(rel)),
        "rel_depth_p90": float(np.percentile(rel, 90)),
        "rel_depth_mean": float(rel.mean()),
        "frac_lt_2pct": float((rel < 0.02).mean()),
        "frac_lt_5pct": float((rel < 0.05).mean()),
        "nn_median_m": float(np.median(d)),
        "nn_ratio": float(np.median(d) / max(spacing, 1e-9)),
        "self_spacing_m": spacing,
        "depth_median": depth_med,
    }


def _aggregate(rows):
    if not rows:
        return {}
    return {
        "n_pairs": len(rows),
        "n_pixels": int(sum(r["n"] for r in rows)),
        "rel_depth_median": float(np.median([r["rel_depth_median"] for r in rows])),
        "rel_depth_p90": float(np.median([r["rel_depth_p90"] for r in rows])),
        "frac_lt_5pct": float(np.median([r["frac_lt_5pct"] for r in rows])),
        "nn_ratio": float(np.median([r["nn_ratio"] for r in rows])),
        "nn_median_m": float(np.median([r["nn_median_m"] for r in rows])),
        "self_spacing_m": float(np.median([r["self_spacing_m"] for r in rows])),
    }


def evaluate_run(run_dir, args):
    npz = np.load(run_dir / "keyframes.npz")
    X = npz["X_canon"]
    C = npz["C"]
    T = npz["T_WC"]
    K = np.asarray(npz["K"], dtype=np.float64)
    n, h, w, _ = X.shape
    poses = [sim3_decompose(T[k]) for k in range(n)]
    scales = [p[2] for p in poses]

    rng = np.random.default_rng(args.seed)
    pairs = [(i, i + g) for g in range(1, args.max_gap + 1) for i in range(n - g)]
    if len(pairs) > args.n_pairs:
        sel = np.linspace(0, len(pairs) - 1, args.n_pairs).astype(int)
        pairs = [pairs[k] for k in sel]

    Xf = X.astype(np.float32)
    Cf = C[..., 0].astype(np.float32)

    # scene scale in the estimator's own units: median distance camera->points
    radii = []
    for k in range(n):
        R, t, s = poses[k]
        m = Cf[k].reshape(-1) > args.conf
        if m.sum() == 0:
            continue
        pw = act(R, t, s, Xf[k].reshape(-1, 3)[m])
        radii.append(float(np.median(np.linalg.norm(pw - t, axis=1))))
    scene_scale = float(np.median(radii)) if radii else 1.0

    def run_pairs(pose_list):
        out = []
        for i, j in pairs:
            r = project_pair(Xf[i], Cf[i], pose_list[i], Xf[j], Cf[j], pose_list[j],
                             K, args.conf, args.nn_samples, rng)
            if r is not None:
                out.append({"i": i, "j": j, "gap": j - i, **r})
        return out

    clean = run_pairs(poses)

    # perturbations are expressed as a FRACTION of the scene scale, because the
    # estimator works in its own (non-metric) units.
    levels = [float(x) for x in str(args.perturb_levels).split(",") if x.strip()]
    controls = []
    for frac in levels:
        t_abs = frac * scene_scale
        deg = np.degrees(frac)          # equal angular error at unit distance
        poses_pert = list(poses)
        for k in range(1, n, max(1, args.perturb_every)):
            R, tr, s = poses_pert[k]
            Rp, tp = random_perturbation(rng, t_abs, deg)
            poses_pert[k] = (Rp @ R, Rp @ tr + tp, s)
        agg = _aggregate(run_pairs(poses_pert))
        if agg:
            agg.update({"perturb_frac": frac, "perturb_t_units": t_abs,
                        "perturb_deg": deg})
            controls.append(agg)

    if not controls:
        agg = _aggregate(clean)
        if agg:
            controls.append(agg)

    # --- per-keyframe geometry: how planar is the feed-forward point map? ---
    kf_geom = []
    for k in range(n):
        m = Cf[k].reshape(-1) > max(args.conf, 5.0)
        if m.sum() < 1000:
            continue
        z = Xf[k].reshape(-1, 3)[m][:, 2]
        kf_geom.append({
            "kf": k,
            "depth_median": float(np.median(z)),
            "depth_rel_std": float(z.std() / np.median(z)),
            "depth_p5_p95_rel": float((np.percentile(z, 95) - np.percentile(z, 5)) / np.median(z)),
        })

    result = {
        "run": run_dir.name,
        "n_keyframes": int(n),
        "hw": [int(h), int(w)],
        "K": K.tolist(),
        "sim3_scale": {"min": float(min(scales)), "max": float(max(scales)),
                       "median": float(np.median(scales))},
        "conf_threshold": args.conf,
        "max_gap": args.max_gap,
        "perturb_every": args.perturb_every,
        "scene_scale_units": scene_scale,
        "clean": _aggregate(clean),
        "controls": controls,
        "keyframe_geometry": kf_geom,
        "depth_rel_std_median": float(np.median([g["depth_rel_std"] for g in kf_geom]))
        if kf_geom else None,
        "pairs_clean": clean,
    }
    return result, clean


def make_figure(clean, result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    if clean:
        rel = np.array([r["rel_depth_median"] for r in clean]) * 100
        ax.hist(rel, bins=20, alpha=0.8, color="tab:blue")
    ax.set_xlabel("per-pair median |Δdepth| / depth  [%]")
    ax.set_ylabel("# pairs")
    ax.set_title("clean: world-frame depth agreement")
    ax.grid(alpha=0.3)

    ax = axes[1]
    c = result["clean"]
    ctrl = result["controls"]
    if c:
        ax.axhline(c["rel_depth_median"] * 100, color="tab:blue", ls="--",
                   label=f"clean {100*c['rel_depth_median']:.2f}%")
    if ctrl:
        ts = [x["perturb_frac"] for x in ctrl]
        ys = [x["rel_depth_median"] * 100 for x in ctrl]
        ax.plot(ts, ys, "o-", color="tab:red", label="perturbed (control)")
        for t, y in zip(ts, ys):
            ax.annotate(f"{y:.2f}%", (t, y), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("per-keyframe Sim3 translation perturbation / scene scale")
    ax.set_ylabel("median depth disagreement [%]")
    ax.set_title("diagnostic sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    labels, vals = [], []
    if c:
        labels.append("clean")
        vals.append(c["nn_ratio"])
    if ctrl:
        labels += [f"{100*x['perturb_frac']:.1f}%" for x in ctrl]
        vals += [x["nn_ratio"] for x in ctrl]
    colors = ["tab:blue"] + ["tab:red"] * len(ctrl)
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("NN distance / self spacing")
    ax.set_title("point-set agreement (1.0 = same surface)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    if c:
        fig.suptitle(
            f"{result['run']}: {c.get('n_pairs', 0)} KF pairs, "
            f"clean median depth disagreement {100*c['rel_depth_median']:.2f}%",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = pathlib.Path(args.run)
    out = pathlib.Path(args.out) if args.out else run_dir / "consistency"
    out.mkdir(parents=True, exist_ok=True)
    result, clean = evaluate_run(run_dir, args)
    result["pairs_clean"] = clean
    (out / "consistency.json").write_text(json.dumps(result, indent=2))
    make_figure(clean, result, out / "consistency.png")

    c = result["clean"]
    print(f"[S2-cons] {run_dir.name}: {result['n_keyframes']} kfs, "
          f"{c.get('n_pairs', 0)} KF pairs (gap<={args.max_gap})")
    print(f"  clean     : depth disagreement median {100*c['rel_depth_median']:.2f}% "
          f"(p90 {100*c['rel_depth_p90']:.2f}%), frac<5% = {100*c['frac_lt_5pct']:.1f}%, "
          f"NN/self-spacing = {c['nn_ratio']:.2f}")
    for x in result["controls"]:
        print(f"  perturb {100*x['perturb_frac']:5.1f}% of scene ({x['perturb_deg']:4.1f} deg): "
              f"median {100*x['rel_depth_median']:5.2f}%, p90 {100*x['rel_depth_p90']:5.2f}%, "
              f"NN/self-spacing {x['nn_ratio']:5.2f}")
    if result.get("depth_rel_std_median") is not None:
        print(f"  feed-forward point map planarity: median depth std/median = "
              f"{100*result['depth_rel_std_median']:.1f}%")
    print(f"[S2-cons] -> {out}")


if __name__ == "__main__":
    main()
