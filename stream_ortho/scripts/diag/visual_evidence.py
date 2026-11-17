"""T12 -- visual evidence for a correction that is invisible at full-image scale.

The hybrid calibration moves half the cells by 2.5 cm (median) and only 5% by more than
1.24 m, so a full-image raw-vs-hybrid comparison cannot show it.  This script produces the
evidence that can:

* ``fig_diff_map.png``      dz per cell with the p50/p95 annotated and owner boundaries
* ``fig_diff_hist.png``     |dz| histogram with the median / p95 / tail-5% markers
* ``fig_owner_boundary.png`` owner boundaries, raw vs hybrid
* ``crop_<i>_compare.png``  raw | hybrid | dz, on the densest |dz| tail crossing an owner seam
* ``profile_<i>.png``       a profile through the seam, raw vs hybrid, with the step height

Gate: at least 3 crops must show a step reduction >= 20%.

Usage::

    python -m stream_ortho.scripts.diag.visual_evidence --obs result-new/P0/obs/ieu/pointmap \
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


def owner_boundary(owner, live):
    """Pixels where a 4-neighbour has a different owner."""
    b = np.zeros_like(live)
    d1 = owner[:, :-1] != owner[:, 1:]
    b[:, :-1] |= d1
    b[:, 1:] |= d1
    d2 = owner[:-1, :] != owner[1:, :]
    b[:-1, :] |= d2
    b[1:, :] |= d2
    return b & live


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--lam-p", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--half", type=int, default=110)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

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
    H, W, L = prod_raw["n"].shape
    rgb_g, _ = rgb_grid(prod_raw["assign"], rows["rgb"], rows["conf"] * rows["n"], H, W, L)
    prod = calibrate_hybrid(d, slices, kfs, prod_raw, fp, rows, origin, cfg,
                            a.iters, a.lam_p, a.k)

    z_raw, c_raw, live = top_surface(prod_raw, rgb_g)
    z_hyb, c_hyb, _ = top_surface(prod, rgb_g)
    dz = np.where(live, z_hyb - z_raw, np.nan)
    own_raw = np.where(prod_raw["n"] > 0, prod_raw["owner"],
                       -1).max(-1)
    own_hyb = np.where(prod["n"] > 0, prod["owner"], -1).max(-1)

    stats = {"seq": a.seq, "cell": prod_raw["cell"],
             "dz_median": float(np.nanmedian(dz)), "dz_p05": float(np.nanpercentile(dz, 5)),
             "dz_p95": float(np.nanpercentile(dz, 95)),
             "abs_p50": float(np.nanmedian(np.abs(dz))),
             "abs_p95": float(np.nanpercentile(np.abs(dz), 95)),
             # NOTE: these are fractions of *live* cells (the earlier version divided by the
             # whole grid, which under-reported the tail)
             "frac_abs_gt_1m": float(np.sum(np.abs(dz[live]) > 1.0) / max(live.sum(), 1)),
             "frac_abs_gt_0p1m": float(np.sum(np.abs(dz[live]) > 0.1) / max(live.sum(), 1))}

    out = pathlib.Path(a.out) / "visual"
    out.mkdir(parents=True, exist_ok=True)

    # ---- diff map ----
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ds = 2
    vmax = float(np.nanpercentile(np.abs(dz), 99))
    im = ax.imshow(dz[::ds, ::ds], cmap="coolwarm", interpolation="nearest",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    fig.colorbar(im, ax=ax, shrink=0.75, label="z_hybrid - z_raw (m)")
    ax.set_title(f"{a.seq}: per-cell height change  (median {stats['dz_median']:+.3f} m, "
                 f"|dz| p95 {stats['abs_p95']:.2f} m, {100*stats['frac_abs_gt_1m']:.1f}% >1 m)",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(out / "fig_diff_map.png"); plt.close(fig)

    # ---- histogram ----
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    v = np.abs(dz[live])
    ax.hist(np.clip(v, 0, 3), bins=120, color="#4c72b0")
    for val, lab, col in ((stats["abs_p50"], "p50", "k"), (stats["abs_p95"], "p95", "k")):
        ax.axvline(val, color=col, ls=":", lw=1)
        ax.text(val, ax.get_ylim()[1] * 0.9, f" {lab} {val:.2f} m", fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("|dz| (m, clipped at 3)")
    ax.set_ylabel("cells"); ax.set_title("the correction lives in the tail, not the bulk")
    fig.tight_layout(); fig.savefig(out / "fig_diff_hist.png"); plt.close(fig)

    # ---- owner boundary ----
    b_raw = owner_boundary(own_raw, live)
    b_hyb = owner_boundary(own_hyb, live)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, (b, ttl) in zip(axes, ((b_raw, "raw"), (b_hyb, "hybrid"))):
        img = np.zeros(b.shape + (3,))
        img[..., 0] = np.clip(c_raw[..., 0] / 255.0, 0, 1) * 0.55
        img[..., 1] = np.clip(c_raw[..., 1] / 255.0, 0, 1) * 0.55
        img[..., 2] = np.clip(c_raw[..., 2] / 255.0, 0, 1) * 0.55
        img[b] = [1.0, 0.15, 0.15]
        ax.imshow(img[::ds, ::ds], interpolation="nearest")
        ax.set_title(f"{ttl}: {int(b.sum()):,} boundary cells", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(out / "fig_owner_boundary.png"); plt.close(fig)

    # ---- crops: densest |dz| tail that also crosses an owner seam ----
    from scipy.ndimage import uniform_filter
    dens = uniform_filter((np.abs(dz) > 1.0).astype(np.float32), size=2 * a.half + 1)
    dens[~live] = -1
    dens_seam = dens * (uniform_filter(b_raw.astype(np.float32), size=2 * a.half + 1) > 0)
    crops, used = [], []
    work = dens_seam.copy()
    for i in range(a.crops):
        ci, cj = np.unravel_index(np.nanargmax(work), work.shape)
        if not np.isfinite(work[ci, cj]) or work[ci, cj] < 0:
            break
        crop = {"i": int(ci), "j": int(cj), "tail_density": float(work[ci, cj])}
        i0, i1 = max(ci - a.half, 1), min(ci + a.half, H - 1)
        j0, j1 = max(cj - a.half, 1), min(cj + a.half, W - 1)
        crop.update({"i0": int(i0), "i1": int(i1), "j0": int(j0), "j1": int(j1)})
        # three-panel
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
        for ax, (z, c, ttl) in zip(axes[:2], ((z_raw, c_raw, "raw"), (z_hyb, c_hyb, "hybrid"))):
            img = np.clip(c[i0:i1, j0:j1], 0, 255).astype(np.uint8)
            hs = shade(z[i0:i1, j0:j1], prod_raw["cell"], live[i0:i1, j0:j1])[..., :3]
            ax.imshow(np.clip(0.75 * img / 255.0 + 0.35 * hs, 0, 1), interpolation="nearest")
            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        im = axes[2].imshow(dz[i0:i1, j0:j1], cmap="coolwarm",
                            norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        axes[2].set_title("dz (m)", fontsize=9)
        axes[2].set_xticks([]); axes[2].set_yticks([]); axes[2].grid(False)
        fig.colorbar(im, ax=axes[2], shrink=0.8)
        fig.suptitle(f"crop {i}: i={ci}, j={cj}  ({a.half*2} cells = "
                     f"{a.half*2*prod_raw['cell']:.0f} m)", fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"crop_{i}_compare.png"); plt.close(fig)

        # profile: the row through the crop centre, crossing the seam
        row = int(np.clip(ci, i0, i1 - 1))
        s_idx = np.arange(j0, j1)
        zr = z_raw[row, j0:j1]
        zh = z_hyb[row, j0:j1]
        lv = live[row, j0:j1]
        def step_of(z):
            dif = np.abs(np.diff(z[lv])) if lv.sum() > 3 else np.array([np.nan])
            return float(np.nanpercentile(dif, 95)), float(np.nanmax(dif))
        s_r, m_r = step_of(zr)
        s_h, m_h = step_of(zh)

        # the metric that matters: the jump measured **across owner boundaries** on this row
        own_row = own_raw[row, j0:j1]
        cross = np.nonzero(np.diff(own_row) != 0)[0]
        cross = cross[(cross >= 0) & (cross + 1 < zr.size)]
        if cross.size >= 3:
            jr = np.abs(np.diff(zr)[cross])
            jh = np.abs(np.diff(zh)[cross])
            seam_r = float(np.median(jr))
            seam_h = float(np.median(jh))
        else:
            seam_r = seam_h = float("nan")
        crop.update({"profile_row": int(row), "step_p95_raw": s_r, "step_p95_hyb": s_h,
                     "step_max_raw": m_r, "step_max_hyb": m_h,
                     "n_seam_crossings": int(cross.size),
                     "seam_step_raw": seam_r, "seam_step_hyb": seam_h,
                     "seam_step_reduction_pct": (float(100 * (1 - seam_h / seam_r))
                                                 if seam_r and np.isfinite(seam_r) else float("nan")),
                     "step_reduction_pct": float(100 * (1 - s_h / s_r)) if s_r else float("nan")})
        fig, ax = plt.subplots(figsize=(8.6, 3.0))
        ax.plot(s_idx * prod_raw["cell"], zr, lw=1.0, color="#8c8c8c", label="raw")
        ax.plot(s_idx * prod_raw["cell"], zh, lw=1.0, color="#c44e52", label="hybrid")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(f"profile row {row}: seam step {seam_r:.2f} -> {seam_h:.2f} m "
                     f"({crop['seam_step_reduction_pct']:+.1f}%), p95 diff {s_r:.2f} -> {s_h:.2f}",
                     fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / f"profile_{i}.png"); plt.close(fig)

        work[max(ci - 2 * a.half, 0):ci + 2 * a.half, max(cj - 2 * a.half, 0):cj + 2 * a.half] = -1
        crops.append(crop)
        print(f"crop {i}: i={ci} j={cj} tail {crop['tail_density']:.3f}  "
              f"seam step {seam_r:.2f} -> {seam_h:.2f} m "
              f"({crop['seam_step_reduction_pct']:+.1f}%) over {cross.size} crossings")

    stats["crops"] = crops
    stats["crops_with_step_reduction_20"] = int(sum(
        1 for c in crops if np.isfinite(c.get("seam_step_reduction_pct", np.nan))
        and c["seam_step_reduction_pct"] >= 20))
    stats["visual_gate_passed"] = bool(stats["crops_with_step_reduction_20"] >= 3)
    (out / "visual_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "crops"}, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
