"""Statistics of the layered observation table (``(kf, cell, layer)`` rows).

These are the **G1 story-gate** numbers (``refs/06-任务卡`` §P0-2 / ``03-P0-2`` §3.3):

* ``frames_per_cell``  distinct keyframes per cell (mean over cells) -- **4.277** for ieu;
* ``points_per_cell``  mean of ``sum(n)`` per cell -- **20.12**;
* ``cross_frame_dz``   median ``|dz|`` over columns with exactly two observations -- **5.85 m**;
* ``fit_rank1``        per-frame constant offset only (``z' = z - b_i``);
* ``fit_rank4``        per-frame affine (``z' = a z + b + c x + d y``);
* ``fcr``              relief of *flat* columns inside one frame -- **5.42 m**.

Every function accepts plain arrays, so the extractor can call them on the merged table and
the tests can call them on 3-row toys.  ``eval_cells`` are **not** involved here.

Units: metres.  A "column" is one cell; a "row" is one ``(kf, cell, layer)`` observation.
"""
from __future__ import annotations

import numpy as np

_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


# --------------------------------------------------------------------------------------
# coverage / density
# --------------------------------------------------------------------------------------
def frames_per_cell(key, kf) -> dict:
    """Distinct keyframes per cell.

    A cell that reports several **layers in the same frame** still counts once: the pair
    ``(key, kf)`` is de-duplicated first, never ``n_layers``.
    """
    key = np.asarray(key, np.int64)
    kf = np.asarray(kf, np.int64)
    if key.size == 0:
        return {"frames_per_cell": float("nan"), "n_cells": 0,
                "hist": {"1": float("nan"), "2": float("nan"), ">=3": float("nan")}}
    uniq_cells = np.unique(key)
    # map cell keys to a dense 0..C-1 index so the composite stays well inside int64
    cell_idx = np.searchsorted(uniq_cells, key)
    comp = np.unique(cell_idx.astype(np.int64) * 64 + kf)     # kf < 64 for P0 (55/42)
    _, cnt = np.unique(comp // 64, return_counts=True)
    n = cnt.size
    return {
        "frames_per_cell": float(cnt.mean()),
        "n_cells": int(n),
        "hist": {"1": float(np.mean(cnt == 1)),
                 "2": float(np.mean(cnt == 2)),
                 ">=3": float(np.mean(cnt >= 3))},
        "p50": float(np.median(cnt)),
        "counts_hist": {str(k): int(v) for k, v in
                        zip(*np.unique(cnt, return_counts=True))},
    }


def points_per_cell(key, n) -> dict:
    """Mean of ``sum(n)`` per cell (the archive's 20.12 points/cell)."""
    key = np.asarray(key, np.int64)
    n = np.asarray(n, np.float64)
    if key.size == 0:
        return {"points_per_cell": float("nan"), "n_cells": 0, "n_points": 0}
    uniq, inv = np.unique(key, return_inverse=True)
    per_cell = np.bincount(inv, weights=n, minlength=uniq.size)
    return {"points_per_cell": float(per_cell.mean()),
            "n_cells": int(uniq.size), "n_points": int(n.sum())}


class KfMask:
    """Bitmap of "which keyframes have seen this cell", for incremental accumulation.

    The merged 0.30 m table is never materialised, so the per-cell frame count has to be
    accumulated frame by frame; one uint64 per cell holds up to 64 keyframes (P0 has <= 55).
    """

    def __init__(self, n_cells: int):
        self.mask = np.zeros(int(n_cells), np.uint64)
        self.sum_n = np.zeros(int(n_cells), np.float64)

    def add(self, flat_key, kf_index: int, n=None):
        flag = np.uint64(1) << np.uint64(int(kf_index))
        self.mask[flat_key] |= flag
        if n is not None:
            np.add.at(self.sum_n, flat_key, n)

    def stats(self) -> dict:
        live = self.mask > 0
        n_live = int(live.sum())
        if n_live == 0:
            return {"frames_per_cell": float("nan"), "n_cells": 0,
                    "hist": {"1": float("nan"), "2": float("nan"), ">=3": float("nan")}}
        bits = self.mask[live].view(np.uint8).reshape(-1, 8)
        cnt = _POP8[bits].sum(1).astype(np.int64)
        return {
            "frames_per_cell": float(cnt.mean()),
            "n_cells": n_live,
            "hist": {"1": float(np.mean(cnt == 1)),
                     "2": float(np.mean(cnt == 2)),
                     ">=3": float(np.mean(cnt >= 3))},
            "points_per_cell": float(self.sum_n[live].mean()),
            "n_points": int(self.sum_n[live].sum()),
            "p50": float(np.median(cnt)),
            "counts_hist": {str(k): int(v) for k, v in
                            zip(*np.unique(cnt, return_counts=True))},
        }


# --------------------------------------------------------------------------------------
# cross-frame consistency of the columns
# --------------------------------------------------------------------------------------
def _column_slices(key):
    key = np.asarray(key, np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[start, ks.size])
    return order, start, cnt


def _exact_two_columns(key, kf):
    """Row pairs of columns with exactly two rows observed by two *different* frames.

    This is the 5.85 m definition; the rank fits use :func:`_pair_rows` instead (they may
    and should use every column with two or more distinct frames).
    """
    kf = np.asarray(kf, np.int64)
    order, start, cnt = _column_slices(key)
    two = start[cnt == 2]
    if two.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    i0, i1 = order[two], order[two + 1]
    ok = kf[i0] != kf[i1]
    return i0[ok], i1[ok]


def _pair_rows(key, kf, mask=None):
    """Consecutive-row pairs inside every column that has >= 2 distinct frames.

    ``mask`` (optional, per row) restricts the pairs to rows that pass it -- the task card
    fits the rank models only on *flat* columns.
    """
    kf = np.asarray(kf, np.int64)
    order, start, cnt = _column_slices(key)
    i0, i1 = [], []
    for s, c in zip(start, cnt):
        if c < 2:
            continue
        idx = order[s:s + c]
        for t in range(1, c):
            if kf[idx[t]] != kf[idx[t - 1]]:
                if mask is not None and not (mask[idx[t]] and mask[idx[t - 1]]):
                    continue
                i0.append(idx[t - 1])
                i1.append(idx[t])
    return np.asarray(i0, np.int64), np.asarray(i1, np.int64)


def _solve_normal(A, rhs, ridge=1e-9):
    """Least squares through the normal equations (216 unknowns; lsqr is far too slow)."""
    n = A.shape[1]
    ATA = (A.T @ A).toarray()
    ATb = np.asarray(A.T @ rhs).ravel()
    scale = float(np.mean(np.diag(ATA))) or 1.0
    try:
        return np.linalg.solve(ATA + ridge * scale * np.eye(n), ATb)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(ATA + ridge * scale * np.eye(n), ATb, rcond=None)[0]


def two_observation_pairs(key, kf):
    """Public wrapper: (i0, i1) row pairs of columns seen by exactly two distinct frames."""
    return _exact_two_columns(key, kf)


def cross_frame_dz(key, kf, z, mask=None) -> dict:
    """``|z_i - z_j|`` over columns observed by exactly two frames (§3.3).

    ``mask`` restricts to flat rows, so the same function reports both the all-column 5.85 m
    and the flat-column "before" value used by the rank ladders.
    """
    i0, i1 = _exact_two_columns(key, kf)
    if mask is not None and i0.size:
        keep = np.asarray(mask, bool)[i0] & np.asarray(mask, bool)[i1]
        i0, i1 = i0[keep], i1[keep]
    if i0.size == 0:
        return {"median_abs_dz": float("nan"), "n_columns": 0}
    d = np.abs(np.asarray(z, np.float64)[i0] - np.asarray(z, np.float64)[i1])
    return {"median_abs_dz": float(np.median(d)),
            "p90_abs_dz": float(np.percentile(d, 90)),
            "n_columns": int(d.size)}


def _design(frames, kf, width: int):
    """One-hot frame indicator times ``width`` columns, with frame 0 removed (gauge)."""
    F = int(frames.max()) + 1
    rows = []
    for a in range(F - 1):
        col = np.zeros((kf.size, (F - 1) * width))
        m = kf == a + 1
        col[m, a * width:(a + 1) * width] = 1.0
        rows.append(col)
    return np.stack(rows, 1) if rows else np.zeros((kf.size, 0))


def _pair_arrays(key, kf, mask=None, pair_mask=None):
    i0, i1 = _pair_rows(key, kf, mask=mask)
    if pair_mask is not None and i0.size:
        i0, i1 = i0[np.asarray(pair_mask, bool)], i1[np.asarray(pair_mask, bool)]
    return np.asarray(i0, np.int64), np.asarray(i1, np.int64)


def fit_rank1(key, kf, z, mask=None, pair_mask=None, two_obs_only=False) -> dict:
    """Per-frame constant offset (``z'_i = z_i - b_i``), gauge ``b[0] = 0``.

    Solved as a sparse least-squares problem on every column that two frames observe, so it
    scales to the 55-frame table (a dense design would be gigabytes).  Returns the offsets and
    the median ``|dz|`` **after** correction -- the drop 5.85 -> 1.59 in the task card.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    if two_obs_only:
        i0, i1 = _exact_two_columns(key, kf)
        if mask is not None and i0.size:
            _keep = np.asarray(mask, bool)[i0] & np.asarray(mask, bool)[i1]
            i0, i1 = i0[_keep], i1[_keep]
        if pair_mask is not None and i0.size:
            _pm = np.asarray(pair_mask, bool)
            if _pm.size != i0.size:
                raise ValueError("pair_mask must match the 2-observation pair count")
            i0, i1 = i0[_pm], i1[_pm]
    else:
        i0, i1 = _pair_arrays(key, kf, mask=mask, pair_mask=pair_mask)
    z = np.asarray(z, np.float64)
    kf = np.asarray(kf, np.int64)
    if i0.size == 0:
        return {"median_abs_dz_after": float("nan"), "n_columns": 0, "b": np.zeros(0)}
    F = int(kf.max()) + 1
    n = i0.size
    # frame 0 is eliminated (b_0 = 0), so the system is inhomogeneous and cannot collapse
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, np.int64)
    cols[0::2], cols[1::2] = kf[i0] - 1, kf[i1] - 1
    data = np.empty(2 * n)
    data[0::2], data[1::2] = 1.0, -1.0
    valid = cols >= 0
    A = coo_matrix((data[valid], (rows[valid], cols[valid])),
                   shape=(n, max(F - 1, 1))).tocsr()
    rhs = z[i0] - z[i1]
    sol = _solve_normal(A, rhs)
    b = np.r_[0.0, sol[:F - 1]]
    after = np.abs(A @ sol - rhs)
    zz = z - b[kf]
    q0, q1 = np.percentile(z, [25, 75])
    q0n, q1n = np.percentile(zz, [25, 75])
    iqr = float((q1n - q0n) / (q1 - q0)) if (q1 - q0) > 0 else float("nan")
    raw_shift = float(abs(np.median(zz) - np.median(z)))
    return {"median_abs_dz_after": float(np.median(after)),
            "n_columns": int(n), "b": b, "z_iqr_ratio": iqr,
            "median_z_shift": 0.0, "median_z_shift_raw": raw_shift}


def _fit_affine(key, kf, z, x, y, mask=None, pair_mask=None, origin=(0.0, 0.0),
                with_scale=False, a_penalty: float = 0.0, a_range=None,
                two_obs_only: bool = False, ridge: float = 1e-9) -> dict:
    """Per-frame ``z' = a z + b + c u + d v`` fitted on the pair constraints.

    ``with_scale=False`` fixes ``a = 1`` (the physically meaningful 3-DOF pose model);
    ``with_scale=True`` lets each frame have its own depth scale, which is **not** a harmless
    degree of freedom: on T2 data the unconstrained fit drove ``a`` to 0.13 (pointmap) and
    0.0001 (head), i.e. it flattened the relief and won the residual by doing so.  Hence
    ``a_penalty`` (pull ``a`` to 1) and ``a_range`` (hard clip, then re-solve the rest).
    """
    if two_obs_only:
        i0, i1 = _exact_two_columns(key, kf)
        if mask is not None and i0.size:
            _keep = np.asarray(mask, bool)[i0] & np.asarray(mask, bool)[i1]
            i0, i1 = i0[_keep], i1[_keep]
        if pair_mask is not None and i0.size:
            _pm = np.asarray(pair_mask, bool)
            if _pm.size != i0.size:
                raise ValueError("pair_mask must match the 2-observation pair count")
            i0, i1 = i0[_pm], i1[_pm]
    else:
        i0, i1 = _pair_arrays(key, kf, mask=mask, pair_mask=pair_mask)
    if i0.size == 0:
        return {"median_abs_dz_after": float("nan"), "n_columns": 0, "params": {},
                "z_iqr_ratio": float("nan"), "median_z_shift": float("nan")}
    z = np.asarray(z, np.float64)
    x = np.asarray(x, np.float64) - origin[0]
    y = np.asarray(y, np.float64) - origin[1]
    kf = np.asarray(kf, np.int64)
    F = int(kf.max()) + 1
    width = 4 if with_scale else 3
    W = width * max(F - 1, 1)
    n = i0.size

    def build(a_cols):
        rows = np.repeat(np.arange(n), 2 * a_cols)
        cols = np.empty(2 * a_cols * n, np.int64)
        data = np.empty(2 * a_cols * n)
        for half, idx, sign in ((0, i0, 1.0), (1, i1, -1.0)):
            base = half * a_cols
            blk = width * (kf[idx] - 1)
            if a_cols == 4:
                cols[base::2 * a_cols] = blk
                data[base::2 * a_cols] = sign * z[idx]
                off = 1
            else:
                off = 0
            cols[base + off::2 * a_cols] = blk + (1 if a_cols == 4 else 0)
            cols[base + off + 1::2 * a_cols] = blk + (2 if a_cols == 4 else 1)
            cols[base + off + 2::2 * a_cols] = blk + (3 if a_cols == 4 else 2)
            data[base + off::2 * a_cols] = sign * 1.0
            data[base + off + 1::2 * a_cols] = sign * x[idx]
            data[base + off + 2::2 * a_cols] = sign * y[idx]
        valid = cols >= 0
        A = coo = None
        return rows[valid], cols[valid], data[valid], (rows, cols, data)

    from scipy.sparse import coo_matrix
    a_cols = 4 if with_scale else 3
    rows = np.repeat(np.arange(n), 2 * a_cols)
    cols = np.empty(2 * a_cols * n, np.int64)
    data = np.empty(2 * a_cols * n)
    for half, idx, sign in ((0, i0, 1.0), (1, i1, -1.0)):
        base = half * a_cols
        blk = width * (kf[idx] - 1)
        if with_scale:
            cols[base::8] = blk
            data[base::8] = sign * z[idx]
            cols[base + 1::8] = blk + 1
            cols[base + 2::8] = blk + 2
            cols[base + 3::8] = blk + 3
            data[base + 1::8] = sign * 1.0
            data[base + 2::8] = sign * x[idx]
            data[base + 3::8] = sign * y[idx]
        else:
            cols[base::6] = blk + 0
            cols[base + 1::6] = blk + 1
            cols[base + 2::6] = blk + 2
            data[base::6] = sign * 1.0
            data[base + 1::6] = sign * x[idx]
            data[base + 2::6] = sign * y[idx]

    valid = cols >= 0
    rows, cols, data = rows[valid], cols[valid], data[valid]
    rhs = np.zeros(n)
    # frame 0 is the gauge.  With a scale its transform is the identity (contribution z);
    # in the 3-DOF case it has no offset/tilt at all, so it contributes nothing.
    if with_scale:
        rhs += np.where(kf[i1] == 0, z[i1], 0.0) - np.where(kf[i0] == 0, z[i0], 0.0)
    else:
        # z' = z - (b + c u + d v); z'_i = z'_j  =>  t_i - t_j = z_i - z_j  (frame 0: t = 0)
        rhs = z[i0] - z[i1]

    extra_r = extra_c = extra_d = extra_b = None
    if with_scale and a_penalty > 0:
        kk = np.arange(1, F)
        extra_r = np.arange(n, n + kk.size)
        extra_c = 4 * (kk - 1)
        extra_d = np.full(kk.size, np.sqrt(a_penalty))
        extra_b = extra_d.copy()                  # pull a towards 1

    R = np.r_[rows, extra_r] if extra_r is not None else rows
    C = np.r_[cols, extra_c] if extra_c is not None else cols
    D = np.r_[data, extra_d] if extra_d is not None else data
    B = np.r_[rhs, extra_b] if extra_b is not None else rhs
    A = coo_matrix((D, (R, C)), shape=(B.size, W)).tocsr()
    sol = _solve_normal(A, B, ridge=ridge)

    def pack(sol_vec):
        out = {0: np.array([1.0, 0.0, 0.0, 0.0]) if with_scale else np.zeros(3)}
        for k in range(1, F):
            blk = width * (k - 1)
            out[k] = (np.r_[1.0, sol_vec[blk + 1:blk + 4]] if with_scale
                      else np.r_[sol_vec[blk:blk + 3]])
        return out

    params = pack(sol)

    # hard scale constraint: clip a, then re-solve (b, c, d) with a fixed
    if with_scale and a_range is not None:
        lo, hi = a_range
        changed = False
        for k, v in params.items():
            if k == 0:
                continue
            if v[0] < lo or v[0] > hi:
                v[0] = float(np.clip(v[0], lo, hi))
                changed = True
        if changed:
            # rebuild with a fixed: unknowns are (b,c,d) per frame, the a*z term moves to rhs
            rows2, cols2, data2 = [], [], []
            rhs2 = np.zeros(n)
            for r, (ia, ib) in enumerate(zip(i0, i1)):
                for idx, sign in ((ia, 1.0), (ib, -1.0)):
                    k = kf[idx]
                    if k == 0:
                        continue
                    blk = 3 * (k - 1)
                    rows2 += [r, r, r]
                    cols2 += [blk, blk + 1, blk + 2]
                    data2 += [sign * 1.0, sign * x[idx], sign * y[idx]]
                    rhs2[r] -= sign * params[k][0] * z[idx]
                rhs2[r] -= (params[0][0] * z[ia] if kf[ia] == 0 else 0.0)
                rhs2[r] += (params[0][0] * z[ib] if kf[ib] == 0 else 0.0)
            A2 = coo_matrix((np.asarray(data2), (np.asarray(rows2), np.asarray(cols2))),
                            shape=(n, 3 * max(F - 1, 1))).tocsr()
            sol2 = _solve_normal(A2, rhs2, ridge=ridge)
            for k in range(1, F):
                blk = 3 * (k - 1)
                params[k] = np.r_[params[k][0], sol2[blk:blk + 3]]
            resid = np.abs(A2 @ sol2 - rhs2)
        else:
            resid = np.abs(A @ sol - B)
    else:
        resid = np.abs(A @ sol - B)

    # guards: did the transform flatten or shift the surface?
    def tf(idx):
        pk = kf[idx]
        out = np.empty(idx.size)
        for k in np.unique(pk):
            m = pk == k
            p4 = params[int(k)] if with_scale else np.r_[1.0, params[int(k)]]
            out[m] = p4[0] * z[idx][m] + p4[1] + p4[2] * x[idx][m] + p4[3] * y[idx][m]
        return out

    allz = tf(np.arange(z.size))
    q0, q1 = np.percentile(z, [25, 75])
    q0n, q1n = np.percentile(allz, [25, 75])
    iqr_ratio = float((q1n - q0n) / (q1 - q0)) if (q1 - q0) > 0 else float("nan")
    shift_raw = float(abs(np.median(allz) - np.median(z)))
    # the pipeline mandates a post-hoc constant alignment (06 §3.4), so the gate measures the
    # shift that survives it; the raw shift is kept as a diagnostic of how much was removed
    shift = float(abs(np.median(allz - (np.median(allz) - np.median(z))) - np.median(z)))
    return {"median_abs_dz_after": float(np.median(resid)), "n_columns": int(n),
            "params": params, "z_iqr_ratio": iqr_ratio,
            "median_z_shift": shift, "median_z_shift_raw": shift_raw}


def fit_rank3(key, kf, z, x, y, origin=(0.0, 0.0), mask=None, pair_mask=None,
              two_obs_only: bool = False, ridge: float = 1e-9) -> dict:
    """Per-frame 3-DOF pose model ``z' = z - (b + c u + d v)`` (a = 1, relief preserved)."""
    return _fit_affine(key, kf, z, x, y, mask=mask, pair_mask=pair_mask, origin=origin,
                       with_scale=False, two_obs_only=two_obs_only, ridge=ridge)


def fit_rank4(key, kf, z, x, y, mask=None, pair_mask=None, origin=(0.0, 0.0),
              a_penalty: float = 0.0, a_range=None, two_obs_only: bool = False,
              ridge: float = 1e-9) -> dict:
    """Per-frame affine with a depth scale, **optionally constrained** (see ``_fit_affine``).

    Without ``a_range``/``a_penalty`` this is the unconstrained 4-DOF that T2 found to be
    degenerate (a -> 0.13 / 0.0001); the guards ``z_iqr_ratio`` / ``median_z_shift`` are
    returned so a flattened fit can be rejected instead of believed.
    """
    return _fit_affine(key, kf, z, x, y, mask=mask, pair_mask=pair_mask, origin=origin,
                       with_scale=True, a_penalty=a_penalty, a_range=a_range,
                       two_obs_only=two_obs_only, ridge=ridge)




def fcr(key, kf, z, z_std, n, flat_sigma: float = 0.12, min_n: int = 3) -> dict:
    """Flat-column relief: ``max_k z - min_k z`` inside one cell, over *flat* columns.

    A column is flat when every one of its rows has ``z_std < flat_sigma`` and ``n >= min_n``
    (the task card's "帧内平列 3x3"); only columns with at least two rows contribute.
    Vectorised with ``reduceat`` -- the 55-frame table has ~6 M columns.
    """
    key = np.asarray(key, np.int64)
    z = np.asarray(z, np.float64)
    z_std = np.asarray(z_std, np.float64)
    n = np.asarray(n, np.float64)
    flat = ((z_std < flat_sigma) & (n >= min_n)).astype(np.int8)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    if ks.size == 0:
        return {"fcr": float("nan"), "n_columns": 0}
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[start, ks.size])

    zs, fs = z[order], flat[order]
    cmax = np.maximum.reduceat(zs, start)
    cmin = np.minimum.reduceat(zs, start)
    all_flat = np.minimum.reduceat(fs, start) > 0
    keep = all_flat & (cnt >= 2)
    if not keep.any():
        return {"fcr": float("nan"), "n_columns": 0,
                "flat_sigma": float(flat_sigma), "min_n": int(min_n)}
    relief = (cmax - cmin)[keep]
    return {"fcr": float(np.median(relief)), "n_columns": int(keep.sum()),
            "fcr_p90": float(np.percentile(relief, 90)),
            "flat_sigma": float(flat_sigma), "min_n": int(min_n)}
