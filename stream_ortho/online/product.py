"""Streaming (online) 2.5D product: incremental ``(cell, layer, owner)`` with frozen assignment.

The offline product is built from **all** frames at once and then re-grouped every round.  The
streaming product instead consumes one frame at a time and never re-groups:

* a new observation is merged into the **nearest existing layer** of its cell when
  ``|z - layer_z| <= z_gap`` -- the layer's weighted mean moves, but its **owner is frozen**;
* only when no layer matches does the observation open a **new layer** (up to ``L``), owned by
  the arriving frame;
* nothing is ever deleted, re-ordered or re-owned, which is exactly the invariant that the
  offline experiments showed to matter (owner switches explode when re-fusing).

Queries: :meth:`top` (highest layer per cell) and :meth:`near` (nearest layer to a height,
used by the online calibration).
"""
from __future__ import annotations

import numpy as np


class StreamingProduct:
    def __init__(self, cell: float, ox: int, oy: int, H: int, W: int, L: int = 3,
                 z_gap: float = 2.0):
        self.cell, self.ox, self.oy = float(cell), int(ox), int(oy)
        self.H, self.W, self.L = int(H), int(W), int(L)
        self.z_gap = float(z_gap)
        n = self.H * self.W * self.L
        # float32 keeps the whole product under ~300 MB (24.3 M rows, 3 layers)
        self.z_sum = np.zeros(n, np.float32)
        self.w_sum = np.zeros(n, np.float32)
        self.n = np.zeros(n, np.uint16)
        self.owner = np.full(n, -1, np.int32)
        # colour per slot: the orthophoto needs real RGB, not a height shading
        self.rgb_sum = np.zeros((n, 3), np.float32)
        self.frames = 0
        #: bumped on every insert; the tile renderer caches the top-surface window on it
        self.version = 0
        #: points more than this above a slot's running mean are treated as floaters once the
        #: slot has at least two observations (0 or None disables the check)
        self.top_tau = 2.0

    # ------------------------------------------------------------------
    def _flat(self, ix, iy):
        ix = np.clip(np.asarray(ix, np.int64) - self.ox, 0, self.W - 1)
        iy = np.clip(np.asarray(iy, np.int64) - self.oy, 0, self.H - 1)
        return iy * self.W + ix, ix, iy

    @staticmethod
    def _acc(base, idx, values):
        """``base[idx] += values`` with duplicate-safe bincount (much faster than add.at)."""
        if idx.size == 0:
            return
        uniq, inv = np.unique(idx, return_inverse=True)
        base[uniq] += np.bincount(inv, weights=values, minlength=uniq.size).astype(base.dtype)

    def insert(self, kf: int, ix, iy, z, w, rgb=None, top_tau=None) -> dict:
        """Merge one frame's observations; existing owners never change."""
        z = np.asarray(z, np.float64)
        w = np.asarray(w, np.float64)
        ok = np.isfinite(z) & np.isfinite(w) & (w > 0)
        tau = self.top_tau if top_tau is None else float(top_tau)
        if rgb is not None:
            rgb = np.asarray(rgb, np.float32)[ok]
        z, w, ix, iy = z[ok], w[ok], ix[ok], iy[ok]
        if z.size == 0:
            return {"inserted": 0, "new_layers": 0}
        flat, cx, cy = self._flat(ix, iy)
        M = z.size
        L = self.L
        base = flat * L                                   # (M,)
        slots = base[:, None] + np.arange(L)[None, :]     # (M, L)
        lz = self.z_sum[slots] / np.maximum(self.w_sum[slots], 1e-12)
        occ = self.n[slots] > 0
        d = np.where(occ, np.abs(z[:, None] - lz), np.inf)
        best = d.argmin(1)                                # (M,)
        dist = d[np.arange(M), best]
        match = np.isfinite(dist) & (dist <= self.z_gap)
        empty = ~occ.any(1)

        # (a) merge into the matched layer
        sel = match
        if sel.any():
            idx = slots[np.arange(M), best][sel]
            self._acc(self.z_sum, idx, z[sel] * w[sel])
            self._acc(self.w_sum, idx, w[sel])
            self.n[np.unique(idx)] += 1
            if rgb is not None:
                for c in range(3):
                    self._acc(self.rgb_sum[:, c], idx, (rgb[sel, c] * w[sel]).astype(np.float32))
        # (b) open a new layer in a free slot (owner = arriving frame)
        new = ~match & empty
        if new.any():
            # one frame can put several layers in the same cell: give them consecutive free
            # slots instead of letting them overwrite each other at slot 0
            rows_new = np.nonzero(new)[0]
            order = np.argsort(flat[rows_new], kind="stable")
            rows_new = rows_new[order]
            cells_sorted = flat[rows_new]
            starts = np.r_[0, np.nonzero(np.diff(cells_sorted))[0] + 1]
            seq = np.arange(rows_new.size) - np.repeat(
                starts, np.diff(np.r_[starts, rows_new.size]))
            slot = np.minimum(seq, L - 1)
            idx = flat[rows_new] * L + slot      # slot index in the product, not the row index
            self._acc(self.z_sum, idx, z[rows_new] * w[rows_new])
            self._acc(self.w_sum, idx, w[rows_new])
            self.n[np.unique(idx)] += 1
            # the first row of a cell sets the owner; later rows in the same cell keep it
            first_rows = rows_new[starts]
            self.owner[idx[starts]] = int(kf)
            if rgb is not None:
                for c in range(3):
                    self._acc(self.rgb_sum[:, c], idx,
                              (rgb[rows_new, c] * w[rows_new]).astype(np.float32))
        # (c) a non-empty cell whose nearest layer is too far and has a free slot
        spill = ~match & ~empty
        n_new = 0
        if spill.any():
            free = ~occ[spill]
            rows = np.nonzero(spill)[0]
            has_free = free.any(1)
            rows = rows[has_free]
            if rows.size:
                slot = free[has_free].argmax(1)
                idx = slots[rows, slot]
                self.z_sum[idx] = z[rows] * w[rows]
                self.w_sum[idx] = w[rows]
                self.n[idx] = 1
                self.owner[idx] = int(kf)
                if rgb is not None:
                    self.rgb_sum[idx] = rgb[rows] * w[rows][:, None]
                n_new = int(rows.size)
        # (d) nothing free: merge into the nearest layer to keep the product bounded
        lost = ~match & ~empty
        if lost.any():
            rows = np.nonzero(lost)[0]
            if n_new:
                free_all = (~occ[lost]).any(1)
                rows = rows[~free_all]
            if rows.size == 0:
                # everything was handled by the spill branch above; keep its result
                self.frames += 1
                if n_new:
                    return {"inserted": n_new, "new_layers": n_new, "slots": n_new,
                            "ok": True}
                return {"inserted": 0, "new_layers": 0, "slots": 0, "ok": True}
            idx = slots[rows, best[rows]]
            self._acc(self.z_sum, idx, z[rows] * w[rows])
            self._acc(self.w_sum, idx, w[rows])
            self.n[np.unique(idx)] += 1
            if rgb is not None:
                for c in range(3):
                    self._acc(self.rgb_sum[:, c], idx,
                              (rgb[rows, c] * w[rows]).astype(np.float32))
        self.frames += 1
        self.version += 1
        # per-row slot (in the same order as the input rows) so a windowed re-calibration
        # can move a frame's heights without re-grouping
        slot_rows = np.where(match, slots[np.arange(M), best], 0)
        spill_rows = np.zeros(M, np.int64)
        if spill.any():
            free = ~occ[spill]
            rows_sp = np.nonzero(spill)[0]
            has_free = free.any(1)
            slot_of_sp = np.zeros(M, np.int64)
            if rows_sp.size:
                slot_of_sp[rows_sp[has_free]] = slots[rows_sp[has_free],
                                                      free[has_free].argmax(1)]
            slot_rows = np.where(match, slot_rows, slot_of_sp)
        return {"inserted": int(M), "new_layers": int(new.sum()) + n_new,
                "slots": slot_rows, "ok": ok}

    # ------------------------------------------------------------------
    def layers(self):
        """(H, W, L) views of the layer heights, support and owner."""
        n = self.n.reshape(self.H, self.W, self.L)
        z = np.where(n > 0, self.z_sum.reshape(self.H, self.W, self.L) /
                     np.maximum(self.w_sum.reshape(self.H, self.W, self.L), 1e-12), np.nan)
        return z, n, self.owner.reshape(self.H, self.W, self.L)

    def top_zr_window(self, r0: int, r1: int, c0: int, c1: int):
        """Top height and colour over a cell window (a view, not a copy of the whole grid).

        The streaming renderer only ever needs the tiles it is about to draw; computing the
        argmax over the full 21 M-cell product costs ~200 ms per frame and grows with the
        flight.
        """
        shp = (self.H, self.W, self.L)
        z3 = self.z_sum.reshape(shp)[r0:r1, c0:c1]
        w3 = self.w_sum.reshape(shp)[r0:r1, c0:c1]
        n3 = self.n.reshape(shp)[r0:r1, c0:c1]
        idx = np.where(n3 > 0, z3, -np.inf).argmax(-1)
        wsel = np.take_along_axis(w3, idx[..., None], 2)[..., 0]
        z = np.take_along_axis(z3, idx[..., None], 2)[..., 0]
        z = np.where(wsel > 0, z / np.maximum(wsel, 1e-9), np.nan)
        rgb = np.take_along_axis(self.rgb_sum.reshape(self.H, self.W, self.L, 3)[r0:r1, c0:c1],
                                 idx[..., None, None], 2)[..., 0, :]
        rgb = np.where(wsel[..., None] > 0, rgb / np.maximum(wsel[..., None], 1e-9), 0.0)
        return z.astype(np.float32), np.clip(rgb, 0.0, 255.0).astype(np.float32)

    def top_zr(self):
        """Top height and its colour from a single argmax (one pass, not two)."""
        shp = (self.H, self.W, self.L)
        occ = self.n.reshape(shp) > 0
        idx = np.where(occ, self.z_sum.reshape(shp), -np.inf).argmax(-1)
        w = self.w_sum.reshape(shp)
        wsel = np.take_along_axis(w, idx[..., None], 2)[..., 0]
        z = np.take_along_axis(self.z_sum.reshape(shp), idx[..., None], 2)[..., 0]
        z = np.where(wsel > 0, z / np.maximum(wsel, 1e-9), np.nan)
        rgb = np.take_along_axis(self.rgb_sum.reshape(self.H, self.W, self.L, 3),
                                 idx[..., None, None], 2)[..., 0, :]
        rgb = np.where(wsel[..., None] > 0, rgb / np.maximum(wsel[..., None], 1e-9), 0.0)
        return z.astype(np.float32), np.clip(rgb, 0.0, 255.0).astype(np.float32)

    def top_rgb(self):
        """Confidence-weighted colour of each cell's top layer, float32 in [0, 255]."""
        w = self.w_sum.reshape(self.H, self.W, self.L)
        occ = self.n.reshape(self.H, self.W, self.L) > 0
        idx = np.where(occ, self.z_sum.reshape(self.H, self.W, self.L), -np.inf).argmax(-1)
        rgb = self.rgb_sum.reshape(self.H, self.W, self.L, 3)
        top_rgb = np.take_along_axis(rgb, idx[..., None, None], 2)[..., 0, :]
        wsel = np.take_along_axis(w, idx[..., None], 2)[..., 0]
        out = np.where(wsel[..., None] > 0, top_rgb / np.maximum(wsel[..., None], 1e-9), 0.0)
        return np.clip(out, 0.0, 255.0).astype(np.float32)

    def top(self):
        z, n, own = self.layers()
        occ = n > 0
        idx = np.where(occ, z, -np.inf).argmax(-1)
        top_z = np.take_along_axis(z, idx[..., None], -1)[..., 0]
        top_owner = np.take_along_axis(own, idx[..., None], -1)[..., 0]
        return np.where(occ.any(-1), top_z, np.nan), top_owner

    def near(self, ix, iy, z):
        """Height of the layer nearest to ``z`` at the given cells (NaN when none)."""
        flat, _, _ = self._flat(ix, iy)
        base = flat * self.L
        slots = base[:, None] + np.arange(self.L)[None, :]
        n = self.n[slots]
        lz = self.z_sum[slots] / np.maximum(self.w_sum[slots], 1e-12)
        occ = n > 0
        d = np.where(occ, np.abs(z[:, None] - lz), np.inf)
        j = d.argmin(1)
        out = lz[np.arange(z.size), j]
        return np.where(occ.any(1), out, np.nan)

    def detach_rows(self, slots, w, z):
        """Temporarily remove a frame's contribution (leave-one-out for re-calibration)."""
        np.subtract.at(self.z_sum, slots, (w * z).astype(self.z_sum.dtype))
        np.subtract.at(self.w_sum, slots, w.astype(self.w_sum.dtype))
        np.subtract.at(self.n, slots, 1)

    def attach_rows(self, slots, w, z):
        """Put a frame's (possibly re-calibrated) contribution back into its own slots."""
        np.add.at(self.z_sum, slots, (w * z).astype(self.z_sum.dtype))
        np.add.at(self.w_sum, slots, w.astype(self.w_sum.dtype))
        np.add.at(self.n, slots, 1)

    def shift_rows(self, slots, w, dz):
        """Move the heights of already-inserted rows by ``dz`` (same slots, same owners)."""
        np.add.at(self.z_sum, slots, (w * dz).astype(self.z_sum.dtype))

    def as_product(self) -> dict:
        """Dict in the same layout as the offline product (for the shared metrics)."""
        L = self.L
        z = np.where(self.n > 0, self.z_sum / np.maximum(self.w_sum, 1e-12), 0.0)
        return {"n": self.n.reshape(self.H, self.W, L).astype(np.float32),
                "z_ref": z.reshape(self.H, self.W, L).astype(np.float32),
                "owner": self.owner.reshape(self.H, self.W, L),
                "sum_w": self.w_sum.reshape(self.H, self.W, L).astype(np.float32),
                "sum_wz": self.z_sum.reshape(self.H, self.W, L).astype(np.float32),
                "sum_wz2": np.zeros_like(self.z_sum, np.float32).reshape(self.H, self.W, L),
                "cell": self.cell, "H": self.H, "W": self.W, "L": L,
                "ox": self.ox, "oy": self.oy,
                "x0": self.ox * self.cell, "y0": self.oy * self.cell}

    def memory_mb(self, itemsize=4):
        return float(self.z_sum.nbytes + self.w_sum.nbytes + self.n.nbytes +
                     self.owner.nbytes + self.rgb_sum.nbytes) / 1e6
