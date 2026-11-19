"""Quantitative geometric quality metrics for streaming 3DGS maps (Stage 0 diagnostics)."""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np


def _group_z_stats(
    xy: np.ndarray,
    z: np.ndarray,
    cell_size: float,
    min_pts: int = 4,
    band_gap: float = 0.04,
    range_thresh: float = 0.06,
    gnd_ref: float = 0.0,
    elev_offset: float = 0.08,
    overhang_gnd_tol: float = 0.04,
    overhang_elev_tol: float = 0.12,
) -> dict:
    """Vectorized 2D spatial-cell vertical distribution statistics."""
    if xy.shape[0] == 0:
        return {
            "n_cells": 0,
            "n_elev_cells": 0,
            "intra_cell_z_std_mean": 0.0,
            "intra_cell_z_std_med": 0.0,
            "intra_cell_z_std_p95": 0.0,
            "elev_z_std_mean": 0.0,
            "elev_z_std_p95": 0.0,
            "multi_band_frac": 0.0,
            "wide_range_frac": 0.0,
            "elev_multi_band_frac": 0.0,
            "elev_wide_range_frac": 0.0,
            "overhang_cell_frac": 0.0,
            "overhang_elev_pt_frac": 0.0,
        }

    cs = max(float(cell_size), 1e-4)
    ix = np.floor(xy[:, 0].astype(np.float64) / cs).astype(np.int64)
    iy = np.floor(xy[:, 1].astype(np.float64) / cs).astype(np.int64)
    ck = (ix + (1 << 20)) * (1 << 21) + (iy + (1 << 20))

    zf = z.astype(np.float64)
    ord_idx = np.lexsort((zf, ck))
    ck_s = ck[ord_idx]
    z_s = zf[ord_idx]

    starts = np.r_[0, np.nonzero(np.diff(ck_s))[0] + 1]
    ends = np.r_[starts[1:], ck_s.size]
    counts = ends - starts

    valid = counts >= int(min_pts)
    if not valid.any():
        return {
            "n_cells": 0,
            "n_elev_cells": 0,
            "intra_cell_z_std_mean": 0.0,
            "intra_cell_z_std_med": 0.0,
            "intra_cell_z_std_p95": 0.0,
            "elev_z_std_mean": 0.0,
            "elev_z_std_p95": 0.0,
            "multi_band_frac": 0.0,
            "wide_range_frac": 0.0,
            "elev_multi_band_frac": 0.0,
            "elev_wide_range_frac": 0.0,
            "overhang_cell_frac": 0.0,
            "overhang_elev_pt_frac": 0.0,
        }

    sum_z = np.add.reduceat(z_s, starts)[valid]
    sum_z2 = np.add.reduceat(z_s * z_s, starts)[valid]
    cnt_v = counts[valid].astype(np.float64)
    mean_z = sum_z / cnt_v
    var_z = np.maximum(sum_z2 / cnt_v - mean_z * mean_z, 0.0)
    std_z = np.sqrt(var_z)

    z_min = np.minimum.reduceat(z_s, starts)[valid]
    z_max = np.maximum.reduceat(z_s, starts)[valid]
    z_rng = z_max - z_min

    # Max consecutive vertical gap inside each sorted cell
    dz = np.diff(z_s)
    same_cell = ck_s[1:] == ck_s[:-1]
    dz_in = np.where(same_cell, dz, 0.0)
    # Pad one zero so reduceat works on starts[:-1] and last group safely
    dz_pad = np.r_[dz_in, 0.0]
    # Note: last element of cell k's internal diff sits before ends[k]-1; excluding the boundary
    # diff via same_cell guarantees max_gap[k] is strictly within cell k.
    max_gap_all = np.maximum.reduceat(dz_pad, starts)
    max_gap = max_gap_all[valid]

    has_gap = max_gap > float(band_gap)
    wide_rng = z_rng > float(range_thresh)

    elev_cell = z_max > (gnd_ref + float(elev_offset))
    n_elev_cells = int(elev_cell.sum())

    # Detached overhang cells: ground visible underneath AND elevated surface above an air gap
    overhang_cell = (
        (z_min <= gnd_ref + float(overhang_gnd_tol))
        & (z_max >= gnd_ref + float(overhang_elev_tol))
        & (max_gap >= float(band_gap))
    )
    elev_high_mask = z_s >= (gnd_ref + float(overhang_elev_tol))
    n_elev_pts_total = int(elev_high_mask.sum())
    if n_elev_pts_total > 0 and overhang_cell.any():
        # Map cell-level overhang flag back to sorted points via group spans
        valid_starts = starts[valid][overhang_cell]
        valid_ends = ends[valid][overhang_cell]
        pt_in_overhang = np.zeros(z_s.size, dtype=bool)
        for s0, s1 in zip(valid_starts.tolist(), valid_ends.tolist()):
            pt_in_overhang[s0:s1] = True
        overhang_elev_pt_frac = float((pt_in_overhang & elev_high_mask).sum()) / float(n_elev_pts_total)
    else:
        overhang_elev_pt_frac = 0.0

    res = {
        "n_cells": int(valid.sum()),
        "n_elev_cells": n_elev_cells,
        "intra_cell_z_std_mean": float(std_z.mean()),
        "intra_cell_z_std_med": float(np.median(std_z)),
        "intra_cell_z_std_p95": float(np.percentile(std_z, 95)),
        "multi_band_frac": float(has_gap.mean()),
        "wide_range_frac": float(wide_rng.mean()),
        "overhang_cell_frac": float(overhang_cell[elev_cell].mean()) if n_elev_cells > 0 else 0.0,
        "overhang_elev_pt_frac": overhang_elev_pt_frac,
    }
    if n_elev_cells > 0:
        res["elev_z_std_mean"] = float(std_z[elev_cell].mean())
        res["elev_z_std_p95"] = float(np.percentile(std_z[elev_cell], 95))
        res["elev_multi_band_frac"] = float(has_gap[elev_cell].mean())
        res["elev_wide_range_frac"] = float(wide_rng[elev_cell].mean())
    else:
        res["elev_z_std_mean"] = 0.0
        res["elev_z_std_p95"] = 0.0
        res["elev_multi_band_frac"] = 0.0
        res["elev_wide_range_frac"] = 0.0
    return res


def compute_gs_quality_metrics(
    means_p: np.ndarray,
    owner_kf: Optional[np.ndarray] = None,
    gsd: Optional[float] = None,
    cell_size: Optional[float] = None,
    band_gap: float = 0.04,
) -> dict:
    """Compute Stage-0 quantitative geometry, layering, and ghosting metrics."""
    m = np.asarray(means_p, dtype=np.float64).reshape(-1, 3)
    fin = np.isfinite(m).all(1)
    m = m[fin]
    if m.shape[0] == 0:
        return {"n_points": 0}

    z = m[:, 2]
    lo_cut = np.percentile(z, 50)
    gnd_ref = float(np.median(z[z <= lo_cut]))

    cs_fine = float(cell_size) if cell_size is not None else max(8.0 * float(gsd or 0.0025), 0.02)
    cs_coarse = 2.0 * cs_fine

    fine_stats = _group_z_stats(
        m[:, :2], z, cell_size=cs_fine, band_gap=band_gap, range_thresh=1.5 * band_gap, gnd_ref=gnd_ref
    )
    coarse_stats = _group_z_stats(
        m[:, :2], z, cell_size=cs_coarse, band_gap=band_gap, range_thresh=1.5 * band_gap, gnd_ref=gnd_ref
    )

    out = {
        "n_points": int(m.shape[0]),
        "gnd_ref_z": gnd_ref,
        "z_percentiles": {
            "p1": float(np.percentile(z, 1)),
            "p5": float(np.percentile(z, 5)),
            "p50": float(np.percentile(z, 50)),
            "p95": float(np.percentile(z, 95)),
            "p99": float(np.percentile(z, 99)),
            "max": float(z.max()),
        },
        "fine_cell_size": cs_fine,
        "fine": fine_stats,
        "coarse_cell_size": cs_coarse,
        "coarse": coarse_stats,
    }

    if owner_kf is not None:
        kf = np.asarray(owner_kf, dtype=np.int32).ravel()[fin]
        ord_k = np.argsort(kf)
        kf_s = kf[ord_k]
        z_ks = z[ord_k]
        ks, starts, counts = np.unique(kf_s, return_index=True, return_counts=True)
        p15s, lo50s = [], []
        per_kf = []
        for k_val, s0, cnt in zip(ks.tolist(), starts.tolist(), counts.tolist()):
            if cnt < 200:
                continue
            zk = z_ks[s0:s0 + cnt]
            p15 = float(np.percentile(zk, 15))
            lo50 = float(np.median(zk[zk <= np.percentile(zk, 50)]))
            p15s.append(p15)
            lo50s.append(lo50)
            per_kf.append({
                "kf": int(k_val),
                "n": int(cnt),
                "p15": p15,
                "lo50_med": lo50,
                "max": float(zk.max()),
            })
        if p15s:
            p15_arr = np.asarray(p15s, dtype=np.float64)
            lo_arr = np.asarray(lo50s, dtype=np.float64)
            out["inter_kf_ground"] = {
                "n_kfs": len(p15s),
                "p15_mean": float(p15_arr.mean()),
                "p15_std": float(p15_arr.std()),
                "p15_iqr": float(np.percentile(p15_arr, 75) - np.percentile(p15_arr, 25)),
                "p15_span": float(p15_arr.max() - p15_arr.min()),
                "lo50_std": float(lo_arr.std()),
                "lo50_span": float(lo_arr.max() - lo_arr.min()),
                "per_kf": per_kf,
            }
    return out


def compute_metrics_from_ply(ply_path, gsd: Optional[float] = None, cell_size: float = 0.02) -> dict:
    """Load a 3DGS PLY and compute Stage-0 geometric quality metrics."""
    from stream_ortho.scripts.export_3dgs_ply import read_gaussian_ply

    d = read_gaussian_ply(pathlib.Path(ply_path))["data"]
    means = np.stack([d["x"], d["y"], d["z"]], axis=1)
    kf = d["keyframe_id"] if "keyframe_id" in d.dtype.names else None
    return compute_gs_quality_metrics(means, owner_kf=kf, gsd=gsd, cell_size=cell_size)
