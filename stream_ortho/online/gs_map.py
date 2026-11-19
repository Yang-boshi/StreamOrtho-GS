"""Streaming column-hashed 3DGS map with nadir-priority consensus & pose sync.

Designed for true streaming orthophoto generation:

* Stores all Gaussians in the leveled **Plane frame** (``+Z`` up, ground near ``z=0``).
* Uses **2D Nadir-Column Top-1 Band Consensus**: each ``(ix, iy)`` column holds
  at most ONE visible surface Gaussian:
    - within ``z_band`` (same surface despite minor SLAM Z-jitter): winner-take-best
      replacement by ``nadir_score`` (eliminates vertical building layering & bloat);
    - well below existing surface (``z_new < z_old - z_band``): NFP occlusion veto;
    - well above existing surface (``z_new > z_old + z_band``): accepted only for
      high-confidence near-nadir observations (roofs replace earlier oblique ground).
* Maintains an owner ``keyframe_id`` per Gaussian so MASt3R-SLAM backend BA
  pose updates move owned Gaussians rigidly via ``ΔT = T_new ∘ T_old⁻¹``.
* Maintains a coarse 2D tile-bucket spatial index for fast ``gsplat`` rendering.
* Exports standard 3DGS PLY files with ``log(scales)`` and ``logit(opacity)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _hash_2d(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Lossless 62-bit bijective packing for (ix, iy) grid cells (zero collisions)."""
    ux = (ix.astype(np.int64) + np.int64(1 << 30)) & np.int64((1 << 31) - 1)
    uy = (iy.astype(np.int64) + np.int64(1 << 30)) & np.int64((1 << 31) - 1)
    return (ux << np.int64(31)) | uy


def _hash_2d_torch(ix, iy):
    """GPU bit-identical lossless 62-bit bijective packing for (ix, iy) grid cells."""
    import torch

    off = int(1 << 30)
    mask31 = int((1 << 31) - 1)
    ux = (ix.to(torch.int64) + off) & mask31
    uy = (iy.to(torch.int64) + off) & mask31
    return (ux << 31) | uy


def _weighted_median_4(z4: np.ndarray, w4: np.ndarray) -> np.ndarray:
    """Vectorized median-centered robust elevation fusion over up to 4 viewpoint votes."""
    w_pos = np.maximum(w4, 0.0)
    z_eff = np.where(w_pos > 0.0, z4, np.float32(1e9))
    ord_i = np.argsort(z_eff, axis=1)
    zs = np.take_along_axis(z_eff, ord_i, axis=1)
    ws = np.take_along_axis(w_pos, ord_i, axis=1)
    cw = np.cumsum(ws, axis=1)
    half = 0.5 * cw[:, 3:4]
    pick = np.argmax(cw >= half, axis=1)
    z_med = zs[np.arange(z4.shape[0]), pick]
    # Smooth tent-weighted average within +-4cm of the weighted median prevents 2-view step jumps
    # on roofs while giving zero weight to outliers > 4cm from the median.
    w_tent = w_pos * np.clip(1.0 - np.abs(z4 - z_med[:, None]) / 0.04, 0.0, 1.0)
    w_sum = w_tent.sum(axis=1)
    z_fused = np.where(w_sum > 1e-8, (w_tent * z4).sum(axis=1) / np.maximum(w_sum, 1e-8), z_med)
    return z_fused.astype(np.float32)


def _fuse_z_slots_torch(
    z4: np.ndarray,
    w4: np.ndarray,
    kf4: np.ndarray,
    z_new: np.ndarray,
    w_new: np.ndarray,
    kf_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated top-4 viewpoint height fusion matching ``_fuse_z_slots``."""
    import torch

    with torch.inference_mode():
        dev = "cuda"
        m = z4.shape[0]
        z4_t = torch.from_numpy(z4).to(device=dev, dtype=torch.float32)
        w4_t = torch.from_numpy(w4).to(device=dev, dtype=torch.float32)
        kf4_t = torch.from_numpy(kf4).to(device=dev, dtype=torch.int32)
        zn_t = torch.from_numpy(z_new).to(device=dev, dtype=torch.float32)
        wn_t = torch.from_numpy(w_new).to(device=dev, dtype=torch.float32)
        rows = torch.arange(m, device=dev, dtype=torch.int64)

        same_kf = kf4_t == int(kf_id)
        has_kf = same_kf.any(dim=1)
        if bool(has_kf.any()):
            slot_k = torch.argmax(same_kf[has_kf].to(torch.int8), dim=1)
            rk = rows[has_kf]
            z4_t[rk, slot_k] = zn_t[has_kf]
            w4_t[rk, slot_k] = torch.maximum(w4_t[rk, slot_k], wn_t[has_kf])

        rem = ~has_kf
        if bool(rem.any()):
            empty = w4_t[rem] <= 0.0
            has_emp = empty.any(dim=1)
            if bool(has_emp.any()):
                re = rows[rem][has_emp]
                se = torch.argmax(empty[has_emp].to(torch.int8), dim=1)
                z4_t[re, se] = zn_t[re]
                w4_t[re, se] = wn_t[re]
                kf4_t[re, se] = int(kf_id)
            full_mask = ~has_emp
            if bool(full_mask.any()):
                rf = rows[rem][full_mask]
                s_min = torch.argmin(w4_t[rf], dim=1)
                w_min = w4_t[rf, s_min]
                better = wn_t[rf] > w_min
                if bool(better.any()):
                    rb = rf[better]
                    sb = s_min[better]
                    z4_t[rb, sb] = zn_t[rb]
                    w4_t[rb, sb] = wn_t[rb]
                    kf4_t[rb, sb] = int(kf_id)

        w_pos = w4_t.clamp_min(0.0)
        z_eff = torch.where(w_pos > 0.0, z4_t, torch.full_like(z4_t, 1e9))
        zs, ord_i = torch.sort(z_eff, dim=1)
        ws = torch.gather(w_pos, 1, ord_i)
        cw = torch.cumsum(ws, dim=1)
        half = 0.5 * cw[:, 3:4]
        pick = torch.argmax((cw >= half).to(torch.int8), dim=1)
        z_med = zs[rows, pick]
        w_tent = w_pos * (1.0 - (z4_t - z_med.unsqueeze(1)).abs() / 0.04).clamp(0.0, 1.0)
        w_sum = w_tent.sum(dim=1)
        z_fused = torch.where(w_sum > 1e-8, (w_tent * z4_t).sum(dim=1) / w_sum.clamp_min(1e-8), z_med)

        return (
            z4_t.cpu().numpy(),
            w4_t.cpu().numpy(),
            kf4_t.cpu().numpy(),
            z_fused.to(torch.float32).cpu().numpy(),
        )


def _fuse_z_slots(
    z4: np.ndarray,
    w4: np.ndarray,
    kf4: np.ndarray,
    z_new: np.ndarray,
    w_new: np.ndarray,
    kf_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update top-4 viewpoint height slots per column and return fused weighted median."""
    m = z4.shape[0]
    if m == 0:
        return z4, w4, kf4, np.zeros((0,), dtype=np.float32)
    if m >= 50_000:
        try:
            import torch

            if torch.cuda.is_available():
                return _fuse_z_slots_torch(z4, w4, kf4, z_new, w_new, kf_id)
        except Exception:
            pass

    rows = np.arange(m)
    same_kf = kf4 == int(kf_id)
    has_kf = same_kf.any(axis=1)
    if has_kf.any():
        slot_k = np.argmax(same_kf[has_kf], axis=1)
        rk = rows[has_kf]
        z4[rk, slot_k] = z_new[has_kf]
        w4[rk, slot_k] = np.maximum(w4[rk, slot_k], w_new[has_kf])

    rem = ~has_kf
    if rem.any():
        empty = w4[rem] <= 0.0
        has_emp = empty.any(axis=1)
        if has_emp.any():
            re = rows[rem][has_emp]
            se = np.argmax(empty[has_emp], axis=1)
            z4[re, se] = z_new[re]
            w4[re, se] = w_new[re]
            kf4[re, se] = int(kf_id)
        full_mask = ~has_emp
        if full_mask.any():
            rf = rows[rem][full_mask]
            s_min = np.argmin(w4[rf], axis=1)
            w_min = w4[rf, s_min]
            better = w_new[rf] > w_min
            if better.any():
                rb = rf[better]
                sb = s_min[better]
                z4[rb, sb] = z_new[rb]
                w4[rb, sb] = w_new[rb]
                kf4[rb, sb] = int(kf_id)

    z_med = _weighted_median_4(z4, w4)
    return z4, w4, kf4, z_med


class StreamingGaussianMap:
    """Incremental 3DGS map in leveled Plane frame with fast spatial tile lookup."""

    def __init__(
        self,
        voxel_size: float,
        tile_world_size: float,
        nadir_replace_margin: float = 0.08,
        z_band: float | None = None,
        min_cos_nadir: float = 0.35,
    ):
        self.voxel_size = max(float(voxel_size), 1e-6)
        self.cell_2d = self.voxel_size
        self.tile_world_size = max(float(tile_world_size), 1e-6)
        self.nadir_replace_margin = float(nadir_replace_margin)
        self.z_band = float(z_band) if z_band is not None else self.voxel_size * 10.0
        self.min_cos_nadir = float(min_cos_nadir)

        # Flat committed arrays (Plane frame)
        self.means = np.zeros((0, 3), dtype=np.float32)
        self.quats = np.zeros((0, 4), dtype=np.float32)      # wxyz
        self.scales = np.zeros((0, 3), dtype=np.float32)     # linear (>0)
        self.opacity = np.zeros((0,), dtype=np.float32)      # linear in (0, 1]
        self.sh_dc = np.zeros((0, 3), dtype=np.float32)
        self.owner_kf = np.zeros((0,), dtype=np.int32)
        self.nadir_score = np.zeros((0,), dtype=np.float32)
        self.conf = np.zeros((0,), dtype=np.float32)

        # Decoupled 2.5D top-4 viewpoint height votes + GPU Z-buffer free-space carve votes
        self.z_buf = np.zeros((0, 4), dtype=np.float32)
        self.w_buf = np.zeros((0, 4), dtype=np.float32)
        self.kf_buf = np.full((0, 4), -1, dtype=np.int32)
        self.carve_votes = np.zeros((0,), dtype=np.uint8)

        # Sorted 2D column hash -> committed row index (1 Gaussian per column)
        self._col_h = np.zeros((0,), dtype=np.int64)
        self._col_i = np.zeros((0,), dtype=np.int64)

        # Pose bookkeeping per keyframe for BA rigid sync
        self.kf_T_WC: Dict[int, np.ndarray] = {}
        self.kf_T_PW: Dict[int, np.ndarray] = {}

        # Spatial 2D tile bucket cache (ti, tj) -> row indices
        self._buckets: Dict[Tuple[int, int], np.ndarray] = {}
        self._buckets_dirty: bool = False
        self.version: int = 0
        self.last_max_shift_px: float = 0.0

    @property
    def n_points(self) -> int:
        return int(self.means.shape[0])

    def _rebuild_column_index(self):
        if self.n_points == 0:
            self._col_h = np.zeros((0,), dtype=np.int64)
            self._col_i = np.zeros((0,), dtype=np.int64)
            return
        c2 = np.floor(self.means[:, :2] / self.cell_2d).astype(np.int64)
        h2 = _hash_2d(c2[:, 0], c2[:, 1])
        order = np.argsort(h2, kind="mergesort")
        self._col_h = h2[order]
        self._col_i = order.astype(np.int64)

    def _rebuild_buckets(self):
        self._buckets.clear()
        if self.n_points == 0:
            self._buckets_dirty = False
            return
        ti = np.floor(self.means[:, 0] / self.tile_world_size).astype(np.int64)
        tj = np.floor(self.means[:, 1] / self.tile_world_size).astype(np.int64)
        key = (ti + (1 << 20)) * (1 << 21) + (tj + (1 << 20))
        order = None
        k_s = None
        if self.n_points >= 200_000:
            try:
                import torch

                if torch.cuda.is_available():
                    k_t = torch.from_numpy(key).cuda()
                    k_st, ord_t = torch.sort(k_t)
                    order = ord_t.cpu().numpy()
                    k_s = k_st.cpu().numpy()
                    del k_t, k_st, ord_t
            except Exception:
                order = None
                k_s = None
        if order is None:
            order = np.argsort(key)
            k_s = key[order]
        starts = np.r_[0, np.nonzero(np.diff(k_s))[0] + 1]
        ends = np.r_[starts[1:], k_s.size]
        for s0, s1 in zip(starts, ends):
            idx = order[s0:s1].astype(np.int32)
            kval = int(k_s[s0])
            i_val = int(kval // (1 << 21)) - (1 << 20)
            j_val = int(kval % (1 << 21)) - (1 << 20)
            self._buckets[(i_val, j_val)] = idx
        self._buckets_dirty = False

    def _query_columns(
        self, x: np.ndarray, y: np.ndarray, exclude_kf: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(x)
        if self._col_h.size == 0 or n == 0:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)
        ix = np.floor(np.asarray(x, np.float64) / self.cell_2d).astype(np.int64)
        iy = np.floor(np.asarray(y, np.float64) / self.cell_2d).astype(np.int64)
        q = _hash_2d(ix, iy)
        pos = np.searchsorted(self._col_h, q)
        pos_c = np.clip(pos, 0, self._col_h.size - 1)
        hit = (pos < self._col_h.size) & (self._col_h[pos_c] == q)
        rows = self._col_i[pos_c]
        if exclude_kf is not None:
            hit &= self.owner_kf[rows] != int(exclude_kf)
        z_out = np.where(hit, self.means[rows, 2], 0.0).astype(np.float32)
        return z_out, hit

    def query_elevation(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Query committed surface elevation at plane ``(x, y)`` (snaps to cell or 3x3)."""
        return self._query_columns(x, y, exclude_kf=None)

    def query_elevation_excluding(
        self, x: np.ndarray, y: np.ndarray, exclude_kf: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query committed surface elevation excluding points owned by ``exclude_kf``."""
        return self._query_columns(x, y, exclude_kf=int(exclude_kf))

    def _slice_rows(self, keep: np.ndarray):
        """Slice all per-Gaussian arrays consistently."""
        self.means = self.means[keep]
        self.quats = self.quats[keep]
        self.scales = self.scales[keep]
        self.opacity = self.opacity[keep]
        self.sh_dc = self.sh_dc[keep]
        self.owner_kf = self.owner_kf[keep]
        self.nadir_score = self.nadir_score[keep]
        self.conf = self.conf[keep]
        self.z_buf = self.z_buf[keep]
        self.w_buf = self.w_buf[keep]
        self.kf_buf = self.kf_buf[keep]
        self.carve_votes = self.carve_votes[keep]

    def _prune_keep_mask(self, keep_mask: np.ndarray):
        """Remove rows where ``keep_mask`` is False in O(N) without sorting."""
        n_old = self.n_points
        n_keep = int(keep_mask.sum())
        if n_keep == n_old:
            return
        new_row = np.full(n_old, -1, dtype=np.int64)
        new_row[keep_mask] = np.arange(n_keep, dtype=np.int64)
        self._slice_rows(keep_mask)

        if self._col_h.size > 0:
            mapped = new_row[self._col_i]
            valid_c = mapped >= 0
            self._col_h = self._col_h[valid_c]
            self._col_i = mapped[valid_c]

        if not self._buckets_dirty and self._buckets:
            remapped = {}
            for bkey, arr in self._buckets.items():
                mb = new_row[arr]
                kb = mb[mb >= 0].astype(np.int32)
                if kb.size > 0:
                    remapped[bkey] = kb
            self._buckets = remapped

    def _rescue_same_kf_collisions(
        self,
        dup_idx: np.ndarray,
        c2_dup: np.ndarray,
        keep_mask: np.ndarray,
        occ_sorted: np.ndarray,
        occ_extra: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """4-neighbor Moiré rescue for same-keyframe BA column collisions.

        Returns ``(rescued_hashes, rescued_rows)`` sorted by ``rescued_hashes``.
        """
        if dup_idx.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        frac = (self.means[dup_idx, :2] / self.cell_2d) - (c2_dup.astype(np.float32) + 0.5)
        use_x = np.abs(frac[:, 0]) >= np.abs(frac[:, 1])
        sx = np.where(use_x, np.where(frac[:, 0] >= 0.0, 1, -1), 0)
        sy = np.where(~use_x, np.where(frac[:, 1] >= 0.0, 1, -1), 0)
        rescued_h_parts = []
        rescued_r_parts = []
        occ_a = occ_sorted
        occ_b = occ_extra if (occ_extra is not None and occ_extra.size > 0) else None
        for dir_mul in (1, -1):
            if dup_idx.size == 0:
                break
            nx_c = c2_dup[:, 0] + dir_mul * sx
            ny_c = c2_dup[:, 1] + dir_mul * sy
            nh = _hash_2d(nx_c, ny_c)
            is_occ = np.zeros(nh.size, dtype=bool)
            for occ_arr in (occ_a, occ_b):
                if occ_arr is not None and occ_arr.size > 0:
                    pos = np.searchsorted(occ_arr, nh)
                    pos_c = np.clip(pos, 0, occ_arr.size - 1)
                    is_occ |= (pos < occ_arr.size) & (occ_arr[pos_c] == nh)
            free_m = ~is_occ
            if free_m.any():
                cand_rows = dup_idx[free_m]
                cand_nh = nh[free_m]
                cand_nx = nx_c[free_m]
                cand_ny = ny_c[free_m]
                u_nh, uf = np.unique(cand_nh, return_index=True)
                rescued = cand_rows[uf]
                self.means[rescued, 0] = (cand_nx[uf].astype(np.float32) + 0.5) * float(self.cell_2d)
                self.means[rescued, 1] = (cand_ny[uf].astype(np.float32) + 0.5) * float(self.cell_2d)
                keep_mask[rescued] = True
                rescued_h_parts.append(u_nh)
                rescued_r_parts.append(rescued.astype(np.int64))
                occ_a = np.sort(np.concatenate([occ_a, u_nh]))
                still_dup = ~keep_mask[dup_idx]
                dup_idx = dup_idx[still_dup]
                c2_dup = c2_dup[still_dup]
                sx = sx[still_dup]
                sy = sy[still_dup]
        if not rescued_h_parts:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        rh = np.concatenate(rescued_h_parts)
        rr = np.concatenate(rescued_r_parts)
        ord_r = np.argsort(rh)
        return rh[ord_r], rr[ord_r]

    def _merge_sorted_columns(
        self, h_a: np.ndarray, i_a: np.ndarray, h_b: np.ndarray, i_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """O(N+M) merge of two strictly sorted disjoint ``(hash, row_idx)`` arrays."""
        if h_a.size == 0:
            return h_b, i_b
        if h_b.size == 0:
            return h_a, i_a
        p_b = np.searchsorted(h_a, h_b, side="left") + np.arange(h_b.size, dtype=np.int64)
        p_a = np.arange(h_a.size, dtype=np.int64) + np.searchsorted(h_b, h_a, side="left")
        mh = np.empty(h_a.size + h_b.size, dtype=np.int64)
        mi = np.empty_like(mh)
        mh[p_a], mi[p_a] = h_a, i_a
        mh[p_b], mi[p_b] = h_b, i_b
        return mh, mi

    def _deduplicate_moved_subset(self, moved_mask: np.ndarray):
        """Incremental 2D column dedup when BA moved only a subset of points (<35%)."""
        moved_idx = np.nonzero(moved_mask)[0]
        if moved_idx.size == 0:
            return
        if self._col_h.size != self.n_points:
            self.deduplicate_columns()
            return

        c2_m = np.floor(self.means[moved_idx, :2] / self.cell_2d).astype(np.int64)
        h2_m = _hash_2d(c2_m[:, 0], c2_m[:, 1])

        unm_col = ~moved_mask[self._col_i]
        h_unm = self._col_h[unm_col]
        i_unm = self._col_i[unm_col]

        ord_m = np.argsort(-self.nadir_score[moved_idx], kind="mergesort")
        h2_ms = h2_m[ord_m]
        u_hm, first_m, inv_m = np.unique(h2_ms, return_index=True, return_inverse=True)
        win_m = moved_idx[ord_m[first_m]]
        win_for_ord_m = win_m[inv_m]

        keep_mask = ~moved_mask
        keep_mask[win_m] = True

        # Compete moved-frame winners against unmoved committed columns
        if h_unm.size > 0 and u_hm.size > 0:
            pos_u = np.searchsorted(h_unm, u_hm)
            pos_uc = np.clip(pos_u, 0, h_unm.size - 1)
            hit_u = (pos_u < h_unm.size) & (h_unm[pos_uc] == u_hm)
            if hit_u.any():
                r_unm = i_unm[pos_uc[hit_u]]
                r_mov = win_m[hit_u]
                mov_wins = self.nadir_score[r_mov] > self.nadir_score[r_unm]
                if mov_wins.any():
                    keep_mask[r_unm[mov_wins]] = False
                if (~mov_wins).any():
                    keep_mask[r_mov[~mov_wins]] = False
            win_surv = keep_mask[win_m]
        else:
            win_surv = np.ones(win_m.size, dtype=bool)

        # Rescue same-keyframe self-collisions within moved points
        ord_rows_m = moved_idx[ord_m]
        is_dup_m = np.ones(ord_rows_m.size, dtype=bool)
        is_dup_m[first_m] = False
        same_kf_m = self.owner_kf[ord_rows_m] == self.owner_kf[win_for_ord_m]
        dup_rows_m = ord_rows_m[is_dup_m & same_kf_m]
        rh, rr = np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        if dup_rows_m.size > 0:
            c2_dup = np.floor(self.means[dup_rows_m, :2] / self.cell_2d).astype(np.int64)
            rh, rr = self._rescue_same_kf_collisions(
                dup_rows_m, c2_dup, keep_mask, u_hm, occ_extra=h_unm
            )

        n_old = self.n_points
        keep = np.nonzero(keep_mask)[0]
        if keep.size < n_old:
            new_row = np.full(n_old, -1, dtype=np.int64)
            new_row[keep] = np.arange(keep.size, dtype=np.int64)
            self._slice_rows(keep)
            unm_keep = new_row[i_unm] >= 0
            h_unm = h_unm[unm_keep]
            i_unm = new_row[i_unm[unm_keep]]
            h_mov = u_hm[win_surv]
            i_mov = new_row[win_m[win_surv]]
            if rh.size > 0:
                h_mov, i_mov = self._merge_sorted_columns(h_mov, i_mov, rh, new_row[rr])
        else:
            h_mov = u_hm[win_surv]
            i_mov = win_m[win_surv].astype(np.int64)
            if rh.size > 0:
                h_mov, i_mov = self._merge_sorted_columns(h_mov, i_mov, rh, rr)

        self._col_h, self._col_i = self._merge_sorted_columns(h_unm, i_unm, h_mov, i_mov)

    def deduplicate_columns(self):
        """Enforce strictly 1 highest-nadir Gaussian per 2D column (used after BA moves poses)."""
        if self.n_points == 0:
            return
        c2 = np.floor(self.means[:, :2] / self.cell_2d).astype(np.int64)
        h2 = _hash_2d(c2[:, 0], c2[:, 1])
        # Sort by descending nadir_score so unique keeps the best-scoring view per column
        ord_s = np.argsort(-self.nadir_score, kind="mergesort")
        h2_s = h2[ord_s]
        u_h, first, inv_idx = np.unique(h2_s, return_index=True, return_inverse=True)
        win_rows_for_u = ord_s[first]
        win_for_ord = win_rows_for_u[inv_idx]
        keep_mask = np.zeros(self.n_points, dtype=bool)
        keep_mask[win_rows_for_u] = True

        # Rescue BA-shifted points that rounded into a same-keyframe neighbor's cell
        is_dup_ord = ~keep_mask[ord_s]
        same_kf_ord = self.owner_kf[ord_s] == self.owner_kf[win_for_ord]
        dup_idx = ord_s[is_dup_ord & same_kf_ord]
        rh, rr = np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        if dup_idx.size > 0:
            rh, rr = self._rescue_same_kf_collisions(dup_idx, c2[dup_idx], keep_mask, u_h)

        n_old = self.n_points
        keep = np.nonzero(keep_mask)[0]
        if keep.size < n_old:
            new_row = np.full(n_old, -1, dtype=np.int64)
            new_row[keep] = np.arange(keep.size, dtype=np.int64)
            self._slice_rows(keep)
            i_u = new_row[win_rows_for_u]
            i_r = new_row[rr] if rr.size > 0 else rr
        else:
            i_u = win_rows_for_u.astype(np.int64)
            i_r = rr
        self._col_h, self._col_i = self._merge_sorted_columns(u_h, i_u, rh, i_r)

    def carve_from_depth(
        self,
        kf_id: int,
        depth_map: np.ndarray,
        K_depth: tuple,
        T_WC: np.ndarray,
        T_PW: np.ndarray,
        bbox_p: tuple,
        carve_tol: float = 0.08,
        min_votes: int = 2,
        min_cos_view: float = 0.75,
        device: str = "cuda:0",
    ) -> dict:
        """GPU Z-buffer multi-view free-space carving against committed map Gaussians.

        Projects existing Gaussians inside ``bbox_p`` into keyframe ``kf_id``'s
        canonical depth map (protected by a 3x3 min-depth filter at roof edges).
        Points sitting strictly in front of open background surface accumulate
        ``carve_votes`` and are removed once ``carve_votes >= min_votes``.
        """
        if self.n_points == 0:
            return {"checked": 0, "voted": 0, "carved": 0}

        from stream_ortho.geometry import sim3_from_vec8, sim3_matrix

        x0, y0, x1, y1 = [float(v) for v in bbox_p]
        cand = self.query_bbox(x0, y0, x1, y1, pad=0.05)
        if cand.size == 0:
            return {"checked": 0, "voted": 0, "carved": 0}
        cand = cand[(self.owner_kf[cand] != int(kf_id)) & (self.owner_kf[cand] >= 0)]
        if cand.size == 0:
            return {"checked": 0, "voted": 0, "carved": 0}

        d_arr = np.asarray(depth_map, dtype=np.float32)
        Hd, Wd = d_arr.shape[:2]
        fx, fy, cx, cy = [float(v) for v in K_depth]
        T_wc_arr = np.asarray(T_WC, dtype=np.float64)
        T_pw_arr = np.asarray(T_PW, dtype=np.float64)
        _, t_wc, s_sim3 = sim3_from_vec8(T_wc_arr)
        M_WC_arr = sim3_matrix(T_wc_arr)
        M_CP = np.linalg.inv(M_WC_arr) @ np.linalg.inv(T_pw_arr)
        M_PC = T_pw_arr @ M_WC_arr
        cam_p = (T_pw_arr[:3, :3] @ t_wc + T_pw_arr[:3, 3]).astype(np.float32)
        s_scale = max(float(s_sim3), 1e-6)
        tol_cam = float(carve_tol) / s_scale

        means_c = self.means[cand]
        try:
            import torch
            import torch.nn.functional as F

            use_dev = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
            pts_t = torch.from_numpy(means_c).to(device=use_dev, dtype=torch.float32)
            R_cp = torch.as_tensor(M_CP[:3, :3], dtype=torch.float32, device=use_dev)
            t_cp = torch.as_tensor(M_CP[:3, 3], dtype=torch.float32, device=use_dev)
            mpc_r2 = torch.as_tensor(M_PC[2, :3], dtype=torch.float32, device=use_dev)
            mpc_t2 = float(M_PC[2, 3])
            cam_t = torch.as_tensor(cam_p, dtype=torch.float32, device=use_dev)

            dp = pts_t - cam_t.unsqueeze(0)
            dist_p = dp.norm(dim=-1).clamp_min(1e-6)
            cos_p = -dp[:, 2] / dist_p

            Xc = pts_t @ R_cp.T + t_cp.unsqueeze(0)
            zc = Xc[:, 2]
            inv_z = 1.0 / zc.clamp_min(1e-6)
            uu = torch.round(fx * (Xc[:, 0] * inv_z) + cx).to(torch.int64)
            vv = torch.round(fy * (Xc[:, 1] * inv_z) + cy).to(torch.int64)

            in_f = (
                (zc > 1e-3)
                & (cos_p >= float(min_cos_view))
                & (uu >= 2) & (uu < Wd - 2)
                & (vv >= 2) & (vv < Hd - 2)
            )
            if not bool(in_f.any()):
                return {"checked": int(cand.size), "voted": 0, "carved": 0}

            z_ref_gnd = min(0.0, float(np.percentile(means_c[:, 2], 10)))
            d_t = torch.from_numpy(d_arr).to(device=use_dev, dtype=torch.float32)
            d_4d = d_t.unsqueeze(0).unsqueeze(0)
            d_min5 = -F.max_pool2d(-d_4d, kernel_size=5, stride=1, padding=2)[0, 0]
            d_max5 = F.max_pool2d(d_4d, kernel_size=5, stride=1, padding=2)[0, 0]
            uu_in = uu[in_f]
            vv_in = vv[in_f]
            d_bg = d_min5[vv_in, uu_in]
            d_span5 = d_max5[vv_in, uu_in] - d_bg
            # Only carve along flat interior rays away from roof/tree depth boundaries
            xb_c = ((uu_in.to(torch.float32) - float(cx)) / max(float(fx), 1.0)) * d_bg
            yb_c = ((vv_in.to(torch.float32) - float(cy)) / max(float(fy), 1.0)) * d_bg
            z_bg_plane = mpc_r2[0] * xb_c + mpc_r2[1] * yb_c + mpc_r2[2] * d_bg + mpc_t2
            pts_z = pts_t[in_f, 2]
            req_gap = torch.clamp(0.65 * (pts_z - z_ref_gnd), min=float(carve_tol))
            can_carve_elev = (
                (d_span5 <= float(carve_tol))
                & (z_bg_plane >= (z_ref_gnd - 0.15))
                & ((z_bg_plane < (z_ref_gnd + 0.10)) | ((pts_z - z_bg_plane) > req_gap))
            )
            viol_sub = ((d_bg - zc[in_f]) > tol_cam) & can_carve_elev
            idx_in = torch.nonzero(in_f, as_tuple=False).squeeze(-1)[viol_sub]
            viol_idx = idx_in.cpu().numpy()
        except ImportError:
            from scipy.ndimage import maximum_filter, minimum_filter

            dp = means_c.astype(np.float32) - cam_p[None, :]
            dist_p = np.maximum(np.linalg.norm(dp, axis=1), 1e-6)
            cos_p = -dp[:, 2] / dist_p
            Xc = means_c.astype(np.float64) @ M_CP[:3, :3].T + M_CP[:3, 3]
            zc = Xc[:, 2]
            inv_z = 1.0 / np.maximum(zc, 1e-6)
            uu = np.round(fx * (Xc[:, 0] * inv_z) + cx).astype(np.int64)
            vv = np.round(fy * (Xc[:, 1] * inv_z) + cy).astype(np.int64)
            in_f = (
                (zc > 1e-3)
                & (cos_p >= float(min_cos_view))
                & (uu >= 2) & (uu < Wd - 2)
                & (vv >= 2) & (vv < Hd - 2)
            )
            if not in_f.any():
                return {"checked": int(cand.size), "voted": 0, "carved": 0}
            z_ref_gnd = min(0.0, float(np.percentile(means_c[:, 2], 10)))
            d_min5 = minimum_filter(d_arr, size=5, mode="nearest")
            d_max5 = maximum_filter(d_arr, size=5, mode="nearest")
            uu_in = uu[in_f]
            vv_in = vv[in_f]
            d_bg = d_min5[vv_in, uu_in]
            d_span5 = d_max5[vv_in, uu_in] - d_bg
            xb_c = ((uu_in.astype(np.float64) - float(cx)) / max(float(fx), 1.0)) * d_bg
            yb_c = ((vv_in.astype(np.float64) - float(cy)) / max(float(fy), 1.0)) * d_bg
            z_bg_plane = M_PC[2, 0] * xb_c + M_PC[2, 1] * yb_c + M_PC[2, 2] * d_bg + M_PC[2, 3]
            pts_z = means_c[in_f, 2]
            req_gap = np.maximum(float(carve_tol), 0.65 * (pts_z - z_ref_gnd))
            can_carve_elev = (
                (d_span5 <= float(carve_tol))
                & (z_bg_plane >= (z_ref_gnd - 0.15))
                & ((z_bg_plane < (z_ref_gnd + 0.10)) | ((pts_z - z_bg_plane) > req_gap))
            )
            viol_sub = ((d_bg - zc[in_f]) > tol_cam) & can_carve_elev
            viol_idx = np.nonzero(in_f)[0][viol_sub]

        n_voted = int(viol_idx.size)
        n_carved = 0
        if n_voted > 0:
            rows_hit = cand[viol_idx]
            self.carve_votes[rows_hit] = np.minimum(
                self.carve_votes[rows_hit].astype(np.int32) + 1, 255
            ).astype(np.uint8)
            kill_rows = rows_hit[self.carve_votes[rows_hit] >= int(min_votes)]
            n_carved = int(kill_rows.size)
            if n_carved > 0:
                if self.n_points < 200_000 or n_carved >= max(8192, self.n_points // 50):
                    keep = self.owner_kf >= 0
                    keep[kill_rows] = False
                    self._prune_keep_mask(keep)
                else:
                    self.opacity[kill_rows] = 0.0
                    self.owner_kf[kill_rows] = -2
                    self.carve_votes[kill_rows] = 0
                    if self._col_h.size > 0:
                        keep_col = self.owner_kf[self._col_i] >= 0
                        self._col_h = self._col_h[keep_col]
                        self._col_i = self._col_i[keep_col]
                self.version += 1

        return {"checked": int(cand.size), "voted": n_voted, "carved": n_carved}

    def _rows_changed_bucket(self, rows: np.ndarray, new_xy: np.ndarray) -> bool:
        if self._buckets_dirty or not self._buckets:
            return self._buckets_dirty
        ti_o = np.floor(self.means[rows, 0] / self.tile_world_size).astype(np.int64)
        tj_o = np.floor(self.means[rows, 1] / self.tile_world_size).astype(np.int64)
        ti_n = np.floor(new_xy[:, 0] / self.tile_world_size).astype(np.int64)
        tj_n = np.floor(new_xy[:, 1] / self.tile_world_size).astype(np.int64)
        return bool((ti_o != ti_n).any() or (tj_o != tj_n).any())

    def ingest_keyframe(
        self,
        kf_id: int,
        means_p,
        quats_p,
        scales_p,
        opacity,
        sh_dc,
        nadir_score,
        conf,
        T_WC: np.ndarray,
        T_PW: np.ndarray,
        cam_rays_p=None,
    ) -> dict:
        """Filter, deduplicate per 2D column by nadir quality, and commit one keyframe."""
        self.kf_T_WC[int(kf_id)] = np.asarray(T_WC, dtype=np.float64).copy()
        self.kf_T_PW[int(kf_id)] = np.asarray(T_PW, dtype=np.float64).copy()

        if hasattr(means_p, "detach"):
            import torch

            with torch.inference_mode():
                m_t = means_p.detach().reshape(-1, 3).to(torch.float32)
                q_t = quats_p.detach().reshape(-1, 4).to(torch.float32)
                sc_t = scales_p.detach().reshape(-1, 3).to(torch.float32)
                op_t = opacity.detach().reshape(-1).to(torch.float32)
                sh_t = sh_dc.detach().reshape(-1, 3).to(torch.float32)
                ns_t = nadir_score.detach().reshape(-1).to(torch.float32)
                cf_t = conf.detach().reshape(-1).to(torch.float32)

                ok_t = torch.isfinite(m_t).all(dim=1) & torch.isfinite(sc_t).all(dim=1) & torch.isfinite(ns_t)
                if cam_rays_p is not None:
                    r_t = cam_rays_p.detach().reshape(-1, 3).to(device=m_t.device, dtype=torch.float32)
                    cn_t = (-r_t[:, 2]).clamp(0.0, 1.0)
                    ns_t = ns_t * cn_t.pow(6)
                    ok_t &= (cn_t >= float(self.min_cos_nadir))
                else:
                    cn_t = torch.ones(m_t.shape[0], dtype=torch.float32, device=m_t.device)

                if not bool(ok_t.any()):
                    return {"in": 0, "added": 0, "replaced": 0, "culled": int(m_t.shape[0]), "total": self.n_points}

                m_t, q_t, sc_t, op_t, sh_t, ns_t, cf_t, cn_t = (
                    m_t[ok_t], q_t[ok_t], sc_t[ok_t], op_t[ok_t],
                    sh_t[ok_t], ns_t[ok_t], cf_t[ok_t], cn_t[ok_t],
                )
                n_in = int(m_t.shape[0])

                c2_t = torch.floor(m_t[:, :2] / float(self.cell_2d)).to(torch.int64)
                h2_t = _hash_2d_torch(c2_t[:, 0], c2_t[:, 1])
                ord_in = torch.argsort(ns_t, descending=True)
                h2_s = h2_t[ord_in]
                hs_val, hs_idx = torch.sort(h2_s, stable=True)
                first_m = torch.ones_like(hs_val, dtype=torch.bool)
                if hs_val.numel() > 1:
                    first_m[1:] = hs_val[1:] != hs_val[:-1]
                sel_in = ord_in[hs_idx[first_m]]

                c2_sel_t = c2_t[sel_in]
                m_t = m_t[sel_in]
                if int(sel_in.numel()) < n_in:
                    m_t = m_t.clone()
                    m_t[:, 0] = (c2_sel_t[:, 0].to(torch.float32) + 0.5) * float(self.cell_2d)
                    m_t[:, 1] = (c2_sel_t[:, 1].to(torch.float32) + 0.5) * float(self.cell_2d)

                m = m_t.cpu().numpy()
                q = q_t[sel_in].cpu().numpy()
                sc = sc_t[sel_in].cpu().numpy()
                op = op_t[sel_in].cpu().numpy()
                sh = sh_t[sel_in].cpu().numpy()
                ns = ns_t[sel_in].cpu().numpy()
                cf = cf_t[sel_in].cpu().numpy()
                cn = cn_t[sel_in].cpu().numpy()
                c2_sel = c2_sel_t.cpu().numpy()
                h2 = hs_val[first_m].cpu().numpy()
        else:
            m = np.asarray(means_p, dtype=np.float32).reshape(-1, 3)
            q = np.asarray(quats_p, dtype=np.float32).reshape(-1, 4)
            sc = np.asarray(scales_p, dtype=np.float32).reshape(-1, 3)
            op = np.asarray(opacity, dtype=np.float32).reshape(-1)
            sh = np.asarray(sh_dc, dtype=np.float32).reshape(-1, 3)
            ns = np.asarray(nadir_score, dtype=np.float32).reshape(-1)
            cf = np.asarray(conf, dtype=np.float32).reshape(-1)

            ok = np.isfinite(m).all(1) & np.isfinite(sc).all(1) & np.isfinite(ns)
            if cam_rays_p is not None:
                cos_nadir = np.clip(-np.asarray(cam_rays_p, np.float32)[:, 2], 0.0, 1.0)
                ns = ns * (cos_nadir ** 6)
                ok &= (cos_nadir >= self.min_cos_nadir)
            else:
                cos_nadir = np.ones(m.shape[0], dtype=np.float32)

            if not ok.any():
                return {"in": 0, "added": 0, "replaced": 0, "culled": int(m.shape[0]), "total": self.n_points}

            m, q, sc, op, sh, ns, cf, cn = (
                m[ok], q[ok], sc[ok], op[ok], sh[ok], ns[ok], cf[ok], cos_nadir[ok]
            )
            n_in = int(m.shape[0])

            # 1. Intra-frame 2D column dedup: keep highest nadir_score per (ix, iy) column
            # Keep h2 strictly ascending so downstream searchsorted hits CPU L1/L2/L3 cache.
            c2 = np.floor(m[:, :2] / self.cell_2d).astype(np.int64)
            h2 = _hash_2d(c2[:, 0], c2[:, 1])
            ord_in = np.argsort(-ns, kind="mergesort")
            h2_s = h2[ord_in]
            _, uniq_idx = np.unique(h2_s, return_index=True)
            sel_in = ord_in[uniq_idx]
            c2_sel = c2[sel_in]
            m, q, sc, op, sh, ns, cf, cn, h2 = (
                m[sel_in], q[sel_in], sc[sel_in], op[sel_in],
                sh[sel_in], ns[sel_in], cf[sel_in], cn[sel_in], h2[sel_in]
            )
            if sel_in.size < n_in:
                # Snap deduplicated dense columns to regular column centers so BA translations
                # shift all columns in phase instead of causing Moiré self-collision holes.
                m = m.copy()
                m[:, 0] = (c2_sel[:, 0].astype(np.float32) + 0.5) * float(self.cell_2d)
                m[:, 1] = (c2_sel[:, 1].astype(np.float32) + 0.5) * float(self.cell_2d)

        # 2. Cross-frame decoupled 2.5D geometry fusion + nadir appearance consensus
        n_rep = 0
        n_fused = 0
        if self._col_h.size > 0:
            pos = np.searchsorted(self._col_h, h2)
            pos_c = np.clip(pos, 0, self._col_h.size - 1)
            hit = (pos < self._col_h.size) & (self._col_h[pos_c] == h2)
            row_old = self._col_i[pos_c]

            z_old = self.means[row_old, 2]
            s_old = self.nadir_score[row_old]
            dz = m[:, 2] - z_old
            z_ref = np.maximum(np.minimum(z_old, m[:, 2]), 0.0)
            z_band_eff = self.z_band + 0.18 * z_ref

            same_band = hit & (np.abs(dz) <= z_band_eff)
            if same_band.any():
                r_sb = row_old[same_band]
                z4_u, w4_u, kf4_u, z_med = _fuse_z_slots(
                    self.z_buf[r_sb],
                    self.w_buf[r_sb],
                    self.kf_buf[r_sb],
                    m[same_band, 2],
                    ns[same_band],
                    int(kf_id),
                )
                self.z_buf[r_sb] = z4_u
                self.w_buf[r_sb] = w4_u
                self.kf_buf[r_sb] = kf4_u
                self.means[r_sb, 2] = z_med
                n_fused = int(r_sb.size)

            rep_same = same_band & (ns > s_old * (1.0 + self.nadir_replace_margin))
            # Higher roof replaces lower ground only when strictly higher nadir quality
            rep_above = (
                hit
                & (dz > z_band_eff)
                & (cn >= 0.80)
                & (ns > s_old * (1.0 + self.nadir_replace_margin))
            )
            # Straight-down nadir view or high-confidence interior view can clear an earlier edge/veil floater
            rep_below = (
                hit
                & (dz < -z_band_eff)
                & (((cn >= 0.88) & (ns > s_old * 2.5)) | (ns > s_old * 5.0))
            )

            if rep_same.any():
                r_s = row_old[rep_same]
                if self._rows_changed_bucket(r_s, m[rep_same, :2]):
                    self._buckets_dirty = True
                # Keep fused weighted-median Z while updating XY and appearance to best nadir view
                self.means[r_s, :2] = m[rep_same, :2]
                self.quats[r_s] = q[rep_same]
                self.scales[r_s] = sc[rep_same]
                self.opacity[r_s] = op[rep_same]
                self.sh_dc[r_s] = sh[rep_same]
                self.owner_kf[r_s] = int(kf_id)
                self.nadir_score[r_s] = ns[rep_same]
                self.conf[r_s] = cf[rep_same]
                self.carve_votes[r_s] = 0
                n_rep += int(r_s.size)

            rep_band = rep_above | rep_below
            if rep_band.any():
                r_b = row_old[rep_band]
                if self._rows_changed_bucket(r_b, m[rep_band, :2]):
                    self._buckets_dirty = True
                self.means[r_b] = m[rep_band]
                self.quats[r_b] = q[rep_band]
                self.scales[r_b] = sc[rep_band]
                self.opacity[r_b] = op[rep_band]
                self.sh_dc[r_b] = sh[rep_band]
                self.owner_kf[r_b] = int(kf_id)
                self.nadir_score[r_b] = ns[rep_band]
                self.conf[r_b] = cf[rep_band]
                self.z_buf[r_b] = 0.0
                self.w_buf[r_b] = 0.0
                self.kf_buf[r_b] = -1
                self.z_buf[r_b, 0] = m[rep_band, 2]
                self.w_buf[r_b, 0] = ns[rep_band]
                self.kf_buf[r_b, 0] = int(kf_id)
                self.carve_votes[r_b] = 0
                n_rep += int(r_b.size)

            add_mask = ~hit
            if add_mask.any():
                # 4-neighbor interior-gap guard: prevent oblique floaters, ground shadows, or
                # edge-bleed pixels from infiltrating 1-cell seam gaps inside existing roofs/ground.
                cand_idx = np.nonzero(add_mask)[0]
                cx_a = c2_sel[cand_idx, 0]
                cy_a = c2_sel[cand_idx, 1]
                nb_cnt = np.zeros(cand_idx.size, dtype=np.int32)
                nb_z_sum = np.zeros(cand_idx.size, dtype=np.float32)
                nb_s_max = np.zeros(cand_idx.size, dtype=np.float32)
                for dx_n, dy_n in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nh = _hash_2d(cx_a + dx_n, cy_a + dy_n)
                    pn = np.searchsorted(self._col_h, nh)
                    pnc = np.clip(pn, 0, self._col_h.size - 1)
                    h_nb = (pn < self._col_h.size) & (self._col_h[pnc] == nh)
                    if h_nb.any():
                        r_nb = self._col_i[pnc[h_nb]]
                        nb_cnt[h_nb] += 1
                        nb_z_sum[h_nb] += self.means[r_nb, 2]
                        nb_s_max[h_nb] = np.maximum(nb_s_max[h_nb], self.nadir_score[r_nb])
                in_seam = nb_cnt >= 2
                if in_seam.any():
                    si = cand_idx[in_seam]
                    z_nb = nb_z_sum[in_seam] / nb_cnt[in_seam].astype(np.float32)
                    s_nb = nb_s_max[in_seam]
                    z_c = m[si, 2]
                    ns_c = ns[si]
                    zb_nb = self.z_band + 0.18 * np.maximum(np.maximum(z_nb, z_c), 0.0)
                    dz_nb = np.abs(z_c - z_nb)
                    veto = (ns_c < 0.10 * s_nb) | (
                        (dz_nb > zb_nb) & (ns_c <= s_nb * (1.0 + self.nadir_replace_margin))
                    )
                    if veto.any():
                        add_mask[si[veto]] = False
                    blend = ~veto & (dz_nb <= zb_nb)
                    if blend.any():
                        bi = si[blend]
                        zb_b = z_nb[blend]
                        sb_b = s_nb[blend]
                        m[bi, 2] = (ns[bi] * m[bi, 2] + sb_b * zb_b) / np.maximum(ns[bi] + sb_b, 1e-8)
        else:
            add_mask = np.ones(m.shape[0], dtype=bool)

        n_add = int(add_mask.sum())
        if n_add > 0:
            n_old = self.n_points
            m_new = m[add_mask]
            ns_new = ns[add_mask]
            self.means = np.concatenate([self.means, m_new], axis=0)
            self.quats = np.concatenate([self.quats, q[add_mask]], axis=0)
            self.scales = np.concatenate([self.scales, sc[add_mask]], axis=0)
            self.opacity = np.concatenate([self.opacity, op[add_mask]], axis=0)
            self.sh_dc = np.concatenate([self.sh_dc, sh[add_mask]], axis=0)
            self.owner_kf = np.concatenate(
                [self.owner_kf, np.full(n_add, int(kf_id), dtype=np.int32)], axis=0
            )
            self.nadir_score = np.concatenate([self.nadir_score, ns_new], axis=0)
            self.conf = np.concatenate([self.conf, cf[add_mask]], axis=0)

            zb_add = np.zeros((n_add, 4), dtype=np.float32)
            wb_add = np.zeros((n_add, 4), dtype=np.float32)
            kb_add = np.full((n_add, 4), -1, dtype=np.int32)
            zb_add[:, 0] = m_new[:, 2]
            wb_add[:, 0] = ns_new
            kb_add[:, 0] = int(kf_id)
            self.z_buf = np.concatenate([self.z_buf, zb_add], axis=0)
            self.w_buf = np.concatenate([self.w_buf, wb_add], axis=0)
            self.kf_buf = np.concatenate([self.kf_buf, kb_add], axis=0)
            self.carve_votes = np.concatenate(
                [self.carve_votes, np.zeros((n_add,), dtype=np.uint8)], axis=0
            )

            # Merge new column hashes into sorted _col_h in O(N + M); h2 is already strictly sorted.
            ha = h2[add_mask]
            ia = np.arange(n_add, dtype=np.int64) + n_old
            self._col_h, self._col_i = self._merge_sorted_columns(self._col_h, self._col_i, ha, ia)

            if not self._buckets_dirty:
                m_add = m[add_mask]
                ti = np.floor(m_add[:, 0] / self.tile_world_size).astype(np.int64)
                tj = np.floor(m_add[:, 1] / self.tile_world_size).astype(np.int64)
                key = (ti + (1 << 20)) * (1 << 21) + (tj + (1 << 20))
                order = np.argsort(key)
                k_s = key[order]
                starts = np.r_[0, np.nonzero(np.diff(k_s))[0] + 1]
                ends = np.r_[starts[1:], k_s.size]
                for s0, s1 in zip(starts, ends):
                    idx = (order[s0:s1] + n_old).astype(np.int32)
                    kval = int(k_s[s0])
                    i_val = int(kval // (1 << 21)) - (1 << 20)
                    j_val = int(kval % (1 << 21)) - (1 << 20)
                    prev = self._buckets.get((i_val, j_val))
                    self._buckets[(i_val, j_val)] = idx if prev is None else np.concatenate([prev, idx])

        if n_add > 0 or n_rep > 0 or n_fused > 0:
            self.version += 1

        return {
            "in": n_in,
            "added": n_add,
            "replaced": n_rep,
            "fused": n_fused,
            "culled": int(n_in - n_add - n_rep),
            "total": self.n_points,
        }

    def update_poses(
        self,
        pose_updates: Dict[int, np.ndarray],
        new_T_PW: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Rigidly update Gaussians whose owner keyframe was refined by backend BA.

        When ``new_T_PW`` is provided for a keyframe, the update replaces the stale
        pre-BA local rectification ``T_PW_old`` cleanly via
        ``T_PW_new @ dM_W @ inv(T_PW_old)`` rather than stacking stale offsets.
        """
        from stream_ortho.geometry import quat_rotate_batch_torch, sim3_from_vec8, sim3_matrix
        from stream_ortho.map.pose_update import sim3_vec_from_matrix
        from stream_ortho.scripts.export_3dgs_ply import quat_rotate_batch

        self.last_max_shift_px = 0.0
        if not pose_updates or self.n_points == 0:
            return []

        use_gpu_q = False
        try:
            import torch

            use_gpu_q = torch.cuda.is_available()
        except Exception:
            pass

        dirty_boxes: List[Tuple[float, float, float, float]] = []
        moved_kfs: List[int] = []
        max_shift_m = 0.0
        for kf_id, T_WC_new in pose_updates.items():
            kf_id = int(kf_id)
            T_WC_old = self.kf_T_WC.get(kf_id)
            T_PW = self.kf_T_PW.get(kf_id)
            if T_WC_old is None or T_PW is None:
                continue
            T_PW_new = np.asarray(new_T_PW[kf_id], np.float64) if (new_T_PW and kf_id in new_T_PW) else T_PW
            sel = self.owner_kf == kf_id
            if not sel.any():
                self.kf_T_WC[kf_id] = np.asarray(T_WC_new, np.float64).copy()
                self.kf_T_PW[kf_id] = T_PW_new.copy()
                continue

            M_old = sim3_matrix(T_WC_old)
            M_new = sim3_matrix(T_WC_new)
            dM_W = M_new @ np.linalg.inv(M_old)
            dM_P = T_PW_new @ dM_W @ np.linalg.inv(T_PW)
            dvec = sim3_vec_from_matrix(dM_P)
            R_d, t_d, s_d = sim3_from_vec8(dvec)

            old_pts = self.means[sel]
            moved = float(s_d) * (old_pts.astype(np.float64) @ R_d.T) + t_d
            # Skip sub-cell micro-jitter BA updates (< 0.25 * cell_2d); keeping kf_T_WC/kf_T_PW
            # at the last applied pose ensures cumulative small updates still apply cleanly.
            max_xy_shift = float(np.max(np.abs(moved[:, :2] - old_pts[:, :2])))
            max_z_shift = float(np.max(np.abs(moved[:, 2] - old_pts[:, 2])))
            if (
                max_xy_shift < 0.25 * float(self.cell_2d)
                and max_z_shift < 0.002
                and abs(float(s_d) - 1.0) < 5e-4
            ):
                continue

            old_bb = (
                float(old_pts[:, 0].min()), float(old_pts[:, 1].min()),
                float(old_pts[:, 0].max()), float(old_pts[:, 1].max()),
            )
            dz_pt = (moved[:, 2] - old_pts[:, 2]).astype(np.float32)
            self.means[sel] = moved.astype(np.float32)
            zb_s = self.z_buf[sel]
            wb_s = self.w_buf[sel]
            zb_s = np.where(wb_s > 0.0, zb_s + dz_pt[:, None], zb_s)
            self.z_buf[sel] = zb_s
            self.means[sel, 2] = _weighted_median_4(zb_s, wb_s)
            q_sel = self.quats[sel]
            if use_gpu_q and q_sel.shape[0] >= 20_000:
                import torch

                with torch.inference_mode():
                    qt_gpu = torch.from_numpy(q_sel).cuda()
                    self.quats[sel] = quat_rotate_batch_torch(qt_gpu, R_d).cpu().numpy()
            else:
                self.quats[sel] = quat_rotate_batch(q_sel.astype(np.float64), R_d).astype(np.float32)
            if abs(float(s_d) - 1.0) > 1e-6:
                self.scales[sel] = (self.scales[sel] * float(s_d)).astype(np.float32)

            new_bb = (
                float(moved[:, 0].min()), float(moved[:, 1].min()),
                float(moved[:, 0].max()), float(moved[:, 1].max()),
            )
            dirty_boxes.append((
                min(old_bb[0], new_bb[0]), min(old_bb[1], new_bb[1]),
                max(old_bb[2], new_bb[2]), max(old_bb[3], new_bb[3]),
            ))
            self.kf_T_PW[kf_id] = T_PW_new.copy()
            self.kf_T_WC[kf_id] = np.asarray(T_WC_new, np.float64).copy()
            moved_kfs.append(kf_id)
            if max_xy_shift > max_shift_m:
                max_shift_m = max_xy_shift

        self.last_max_shift_px = max_shift_m / max(float(self.cell_2d), 1e-9)

        if moved_kfs:
            moved_mask = np.isin(self.owner_kf, np.asarray(moved_kfs, dtype=np.int32))
            n_moved = int(moved_mask.sum())
            if self.n_points >= 50_000 and 0 < n_moved < int(0.35 * self.n_points):
                self._deduplicate_moved_subset(moved_mask)
            else:
                self.deduplicate_columns()
            self._buckets_dirty = True
            self.version += 1
        return dirty_boxes

    def apply_plane_transform(self, dT_P: np.ndarray) -> None:
        """Apply a global rigid transform ``dT_P`` across all committed Plane-frame Gaussians."""
        if self.n_points == 0:
            return
        from stream_ortho.geometry import quat_rotate_batch_torch
        from stream_ortho.scripts.export_3dgs_ply import quat_rotate_batch

        dT = np.asarray(dT_P, dtype=np.float64)
        R_d = dT[:3, :3]
        t_d = dT[:3, 3]
        old_z = self.means[:, 2].copy()
        moved = self.means.astype(np.float64) @ R_d.T + t_d
        self.means = moved.astype(np.float32)
        dz_pt = (self.means[:, 2] - old_z).astype(np.float32)
        self.z_buf = np.where(self.w_buf > 0.0, self.z_buf + dz_pt[:, None], self.z_buf)
        has_w = (self.w_buf > 0.0).any(axis=1)
        if has_w.any():
            self.means[has_w, 2] = _weighted_median_4(self.z_buf[has_w], self.w_buf[has_w])

        rotated_q = False
        try:
            import torch

            if torch.cuda.is_available() and self.n_points >= 20_000:
                with torch.inference_mode():
                    out_q = np.empty_like(self.quats)
                    chunk = 1_000_000
                    for s0 in range(0, self.n_points, chunk):
                        s1 = min(s0 + chunk, self.n_points)
                        qt = torch.from_numpy(self.quats[s0:s1]).cuda()
                        out_q[s0:s1] = quat_rotate_batch_torch(qt, R_d).cpu().numpy()
                    self.quats = out_q
                    rotated_q = True
        except Exception:
            rotated_q = False
        if not rotated_q:
            self.quats = quat_rotate_batch(self.quats.astype(np.float64), R_d).astype(np.float32)

        for kf_k in list(self.kf_T_PW.keys()):
            self.kf_T_PW[kf_k] = dT @ self.kf_T_PW[kf_k]

        self.deduplicate_columns()
        self._buckets_dirty = True
        self.version += 1

    def query_bbox(
        self, x0: float, y0: float, x1: float, y1: float, pad: float = 0.0
    ) -> np.ndarray:
        """Return indices of committed Gaussians inside ``[x0-pad, x1+pad] x [y0-pad, y1+pad]``."""
        if self.n_points == 0:
            return np.zeros((0,), dtype=np.int32)
        if self._buckets_dirty:
            self._rebuild_buckets()

        qx0, qy0 = x0 - pad, y0 - pad
        qx1, qy1 = x1 + pad, y1 + pad
        i0 = int(np.floor(qx0 / self.tile_world_size))
        i1 = int(np.floor(qx1 / self.tile_world_size))
        j0 = int(np.floor(qy0 / self.tile_world_size))
        j1 = int(np.floor(qy1 / self.tile_world_size))

        chunks = []
        for ti in range(i0, i1 + 1):
            for tj in range(j0, j1 + 1):
                arr = self._buckets.get((ti, tj))
                if arr is not None and arr.size > 0:
                    chunks.append(arr)
        if not chunks:
            return np.zeros((0,), dtype=np.int32)
        cand = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        xy = self.means[cand, :2]
        in_box = (xy[:, 0] >= qx0) & (xy[:, 0] <= qx1) & (xy[:, 1] >= qy0) & (xy[:, 1] <= qy1)
        return cand[in_box]

    def compact(self) -> int:
        """Compact any soft-carved rows (``owner_kf < 0``) out of active storage."""
        if self.n_points == 0:
            return 0
        keep = self.owner_kf >= 0
        n_dead = int((~keep).sum())
        if n_dead > 0:
            self._prune_keep_mask(keep)
            self.version += 1
        return n_dead

    def export_ply(self, path, meta: Optional[dict] = None) -> dict:
        """Write standard 3DGS PLY in Plane frame (``log(scales)``, ``logit(opacity)``)."""
        from stream_ortho.scripts.export_3dgs_ply import write_gaussian_ply

        self.compact()

        scales_log = np.log(np.clip(self.scales, 1e-7, None)).astype(np.float32)
        op_c = np.clip(self.opacity, 1e-4, 1.0 - 1e-4)
        op_logit = np.log(op_c / (1.0 - op_c)).astype(np.float32)
        return write_gaussian_ply(
            path,
            self.means,
            self.sh_dc,
            op_logit,
            scales_log,
            self.quats,
            self.owner_kf,
            meta=meta,
        )
