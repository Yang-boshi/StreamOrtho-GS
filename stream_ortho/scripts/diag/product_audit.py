"""T13 -- is the 3-DOF correction relevant to the *product*, or only to global consistency?

T12 showed the calibration does not reduce the local seam step.  Before writing that up as a
negative result, this audit answers two questions:

A. **Where is the tail?**  For the 6.6% of cells that move by more than 1 m: how far are they
   from an owner boundary, are they roof / ground / edge cells, and are they inside the
   ortho-visible area?
B. **Does the frame-pair bias at owner boundaries actually drop?**  For every owner pair
   (A, B) that meets at a boundary, the bias is ``median(z_A - z_B)`` over the cells the two
   frames *share* (a same-surface comparison, unlike the seam step which mixes surfaces).
   ``bias_reduction = 1 - |d_hybrid| / |d_raw|``.

C. Product-level metrics replacing the seam step: boundary bias p95, double-wall rate
   (|bias| > 1 m), and the per-frame intercept spread.

Outputs (``<out>/audit/``): ``tail_map.png``, ``tail_stats.json``, ``boundary_bias.json``,
``fig_boundary_bias.png``, ``profile_top<k>.png``.

Usage::

    python -m stream_ortho.scripts.diag.product_audit --obs result-new/P0/obs/ieu/pointmap \
        --out results/week10-pivotb/ieu --seq ieu
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from stream_ortho.eval import pivotb_preflight as P
from stream_ortho.eval import sync_eval as S
from stream_ortho.map.calib import CalibConfig, FrozenProduct
from stream_ortho.scripts.diag.render_ortho import (calibrate_hybrid, rgb_grid, shade,
                                                    top_surface)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--lam-p", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--min-shared", type=int, default=500)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt, median_filter, uniform_filter

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    cfg = CalibConfig()
    d, ix, iy = P.load(a.obs)
    slices = S.frame_slices(d["kf"])
    kfs = sorted(slices)
    allidx = np.sort(np.concatenate([slices[k] for k in kfs]))
    bbox = P._bbox(ix[allidx], iy[allidx], d["cell"])
    origin = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    prod_raw, _, rows = P.product(d, ix, iy, kfs, slices, bbox, return_assign=True)
    fp = FrozenProduct.from_product(prod_raw)
    rgb_g, _ = rgb_grid(prod_raw["assign"], rows["rgb"], rows["conf"] * rows["n"],
                        *prod_raw["n"].shape)
    prod, z_cur = calibrate_hybrid(d, slices, kfs, prod_raw, fp, rows, origin, cfg,
                                   a.iters, a.lam_p, a.k, return_z=True)

    z_raw, c_raw, live = top_surface(prod_raw, rgb_g)
    z_hyb, c_hyb, _ = top_surface(prod, rgb_g)
    dz = np.where(live, z_hyb - z_raw, np.nan)
    own = np.where(prod_raw["n"] > 0, prod_raw["owner"], -1).max(-1)

    out = pathlib.Path(a.out) / "audit"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- A. where is the tail ----------------
    b = np.zeros_like(live)
    dd = own[:, :-1] != own[:, 1:]
    b[:, :-1] |= dd; b[:, 1:] |= dd
    dd = own[:-1, :] != own[1:, :]
    b[:-1, :] |= dd; b[1:, :] |= dd
    b &= live
    dist = distance_transform_edt(~b)
    tail = live & (np.abs(dz) > 1.0)
    ground = median_filter(np.where(live, z_raw, np.nanmedian(z_raw[live])), size=41, mode="nearest")
    height_above_local = z_raw - ground
    roof = live & (height_above_local > 2.0)
    veg = live & ((height_above_local > 0.3) & (height_above_local <= 2.0))
    stats = {
        "n_live": int(live.sum()), "n_tail": int(tail.sum()),
        "tail_frac": float(tail.sum() / max(live.sum(), 1)),
        "tail_dist_to_boundary_p50": float(np.median(dist[tail])) if tail.any() else None,
        "tail_dist_to_boundary_p90": float(np.percentile(dist[tail], 90)) if tail.any() else None,
        "all_dist_to_boundary_p50": float(np.median(dist[live])),
        "tail_on_roof_frac": float((tail & roof).sum() / max(tail.sum(), 1)),
        "tail_on_veg_frac": float((tail & veg).sum() / max(tail.sum(), 1)),
        "tail_on_ground_frac": float((tail & ~(roof | veg)).sum() / max(tail.sum(), 1)),
        "tail_within_2cells_of_boundary_frac": float((tail & (dist <= 2)).sum() / max(tail.sum(), 1)),
        "roof_frac_of_live": float(roof.sum() / max(live.sum(), 1)),
    }
    (out / "tail_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    ds = 2
    img = np.clip(c_raw, 0, 255).astype(np.uint8)
    axes[0].imshow(img[::ds, ::ds]); axes[0].set_title("ortho (raw)", fontsize=8)
    axes[1].imshow(np.where(tail, 1.0, 0.0)[::ds, ::ds], cmap="inferno", interpolation="nearest")
    axes[1].set_title(f"|dz|>1 m: {100*stats['tail_frac']:.1f}% of live cells", fontsize=8)
    axes[2].imshow(np.clip(dist, 0, 20)[::ds, ::ds], cmap="viridis", interpolation="nearest")
    axes[2].set_title("distance to nearest owner boundary (cells)", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"{a.seq}: where does the correction happen?", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "tail_map.png"); plt.close(fig)

    # ---------------- B. frame-pair bias at owner boundaries ----------------
    # owner pairs that actually meet at a boundary
    pairs = {}
    for ax_, sl_a, sl_b in ((1, (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                            (0, (slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        m = live[sl_a] & live[sl_b] & (own[sl_a] != own[sl_b]) & (own[sl_a] >= 0) & (own[sl_b] >= 0)
        if not m.any():
            continue
        ia, ib = own[sl_a][m], own[sl_b][m]
        for u, v in zip(ia, ib):
            key = (int(min(u, v)), int(max(u, v)))
            pairs[key] = pairs.get(key, 0) + 1

    singles = {k: S.single_layer_cells(d["key"][slices[k]]) for k in kfs}
    recs = []
    for (A, B), nb in sorted(pairs.items(), key=lambda kv: -kv[1]):
        common = S._intersect_sorted(singles[A], singles[B])
        if common.size < a.min_shared:
            continue
        rA = S._rows(d, slices[A])
        rB = S._rows(d, slices[B])
        oA = np.argsort(rA["key"], kind="stable")
        oB = np.argsort(rB["key"], kind="stable")
        za_raw = S._lookup(rA["key"][oA], rA["z"][oA], common)
        zb_raw = S._lookup(rB["key"][oB], rB["z"][oB], common)
        za_hyb = S._lookup(rA["key"][oA], z_cur[A][oA], common)
        zb_hyb = S._lookup(rB["key"][oB], z_cur[B][oB], common)
        d_raw = float(np.median(za_raw - zb_raw))
        d_hyb = float(np.median(za_hyb - zb_hyb))
        recs.append({"A": int(A), "B": int(B), "n_boundary_pairs": int(nb),
                     "n_shared": int(common.size), "bias_raw": d_raw, "bias_hybrid": d_hyb,
                     "bias_reduction_pct": float(100 * (1 - abs(d_hyb) / abs(d_raw)))
                     if d_raw else float("nan")})

    if recs:
        red = np.array([r["bias_reduction_pct"] for r in recs])
        wr = np.array([r["n_boundary_pairs"] for r in recs], float)
        bias_p95_raw = float(np.percentile([abs(r["bias_raw"]) for r in recs], 95))
        bias_p95_hyb = float(np.percentile([abs(r["bias_hybrid"]) for r in recs], 95))
        b_metrics = {
            "n_owner_pairs": len(recs),
            "weighted_bias_reduction_pct": float(np.average(red, weights=wr)),
            "median_bias_reduction_pct": float(np.median(red)),
            "frac_pairs_improved": float(np.mean(red > 0)),
            "frac_pairs_reduction_ge20": float(np.mean(red >= 20)),
            "boundary_bias_p95_raw": bias_p95_raw,
            "boundary_bias_p95_hybrid": bias_p95_hyb,
            "double_wall_rate_raw": float(np.mean([abs(r["bias_raw"]) > 1.0 for r in recs])),
            "double_wall_rate_hybrid": float(np.mean([abs(r["bias_hybrid"]) > 1.0 for r in recs])),
            "weighted_bias_reduction_pct_robust": float(np.average(
                [r["bias_reduction_pct"] for r in recs if abs(r["bias_raw"]) >= 0.2],
                weights=[r["n_boundary_pairs"] for r in recs if abs(r["bias_raw"]) >= 0.2]))
            if any(abs(r["bias_raw"]) >= 0.2 for r in recs) else float("nan"),
            "n_pairs_bias_ge_0p2m": int(sum(abs(r["bias_raw"]) >= 0.2 for r in recs)),
            "frame_bias_std_raw": float(np.std([r["bias_raw"] for r in recs])),
            "frame_bias_std_hybrid": float(np.std([r["bias_hybrid"] for r in recs])),
        }
    else:
        b_metrics = {"n_owner_pairs": 0}

    # ---------------- C. figures ----------------
    if recs:
        recs_sorted = sorted(recs, key=lambda r: -abs(r["bias_raw"]))[:20]
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        x = np.arange(len(recs_sorted))
        ax.bar(x - 0.2, [abs(r["bias_raw"]) for r in recs_sorted], 0.4, label="raw",
               color="#8c8c8c")
        ax.bar(x + 0.2, [abs(r["bias_hybrid"]) for r in recs_sorted], 0.4, label="hybrid",
               color="#c44e52")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['A']}-{r['B']}" for r in recs_sorted], rotation=90, fontsize=6)
        ax.set_ylabel("|median(z_A - z_B)| (m)")
        ax.set_title(f"frame-pair bias on shared cells, top 20 owner pairs\n"
                     f"weighted reduction {b_metrics['weighted_bias_reduction_pct']:+.1f}%", fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig_boundary_bias.png"); plt.close(fig)

        # profiles across the worst boundary of the top-3 pairs
        for rank, r in enumerate(sorted(recs, key=lambda q: -q["n_boundary_pairs"])[:3]):
            A_, B_ = r["A"], r["B"]
            m = live[:, :-1] & live[:, 1:] & (own[:, :-1] != own[:, 1:]) & \
                (((own[:, :-1] == A_) & (own[:, 1:] == B_)) |
                 ((own[:, :-1] == B_) & (own[:, 1:] == A_)))
            if not m.any():
                continue
            ii, jj = np.nonzero(m)
            k = len(ii) // 2
            i0, j0 = ii[k], jj[k]
            j0a, j1a = max(j0 - 40, 0), min(j0 + 41, z_raw.shape[1])
            fig, ax = plt.subplots(figsize=(7.6, 3.0))
            ax.plot(np.arange(j0a, j1a) * prod_raw["cell"], z_raw[i0, j0a:j1a], "-o", ms=3,
                    color="#8c8c8c", label="raw")
            ax.plot(np.arange(j0a, j1a) * prod_raw["cell"], z_hyb[i0, j0a:j1a], "-o", ms=3,
                    color="#c44e52", label="hybrid")
            ax.axvline(j0 * prod_raw["cell"], color="k", ls=":", lw=1)
            ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
            ax.set_title(f"owner boundary {A_}|{B_} at row {i0}: "
                         f"frame bias {r['bias_raw']:+.2f} -> {r['bias_hybrid']:+.2f} m", fontsize=8)
            ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(out / f"profile_top{rank}.png"); plt.close(fig)

    res = {"seq": a.seq, "cell": prod_raw["cell"], "tail": stats, "boundary_bias": b_metrics,
           "pairs": recs}
    (out / "boundary_bias.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({"tail": {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in stats.items()}}, indent=1))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in b_metrics.items()}, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
