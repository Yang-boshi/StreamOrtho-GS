"""P0-2 (v2): **global dense grid** column store with L layers.

The tile-blocked version was abandoned (week9-04/05): both failures came from the same
class of bug -- tile-local indices and NaN-filled empty layers.  A single global dense
grid removes that class entirely:

* one index expression: ``flat = (iy - oy) * W + (ix - ox)`` (no tile origin, no local
  index, no cross-tile collision);
* empty layers are ``n = 0`` (masked), **never NaN**;
* one ``argsort`` + ``reduceat`` per frame aggregates the observations of a cell, so the
  layer decision is made **once per (frame, cell)** -- that is what made
  ``n_new_layer`` 38.7 M instead of ~8 M.

A low observation arriving under a high surface is kept as **another layer** (never
filtered by the top surface -- the mistake behind the rejected B_cache run).
Memory: ``H × W × L × (4+4+4+4+3*2+2+4+4+2)`` bytes; the ieu bbox at 0.30 m with L=3 is
~1.6-2.0 GB, inside the 4 GB budget.
"""

from __future__ import annotations

import numpy as np


class ColumnStore:
    def __init__(self, cell: float, T_WP, L: int = 3, z_gap: float = 2.0,
                 bbox: tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0)):
        self.cell = float(cell)
        self.T_WP = np.asarray(T_WP, np.float64)
        self.L = int(L)
        self.z_gap = float(z_gap)
        x0, y0, x1, y1 = (float(v) for v in bbox)
        self.ox = int(np.floor(x0 / self.cell))
        self.oy = int(np.floor(y0 / self.cell))
        self.W = int(np.ceil(x1 / self.cell)) - self.ox + 1
        self.H = int(np.ceil(y1 / self.cell)) - self.oy + 1
        H, W, L = self.H, self.W, self.L
        f, f16, u16, i32 = np.float32, np.float16, np.uint16, np.int32
        self.z_ref = np.zeros((H, W, L), f)
        self.sum_w = np.zeros((H, W, L), f)
        self.sum_wz = np.zeros((H, W, L), f)
        self.sum_wz2 = np.zeros((H, W, L), f)
        self.rgb = np.zeros((H, W, L, 3), f16)
        self.n = np.zeros((H, W, L), u16)
        self.last_kf = np.full((H, W, L), -1, i32)
        self.owner = np.full((H, W, L), -1, i32)
        self.best_conf = np.zeros((H, W, L), f16)
        # representative observation colour (highest conf in the layer) -- keeps the
        # original texture instead of the 0.45 m box average that cost 10% sharpness
        self.rep_rgb = np.zeros((H, W, L, 3), f16)
        self.rep_conf = np.zeros((H, W, L), f16)
        self.n_frames = 0
        self.n_obs = 0
        self.n_new_layer = 0
        self.n_overflow_points = 0
        self.n_overflow_cells = 0
        self.overflow_samples: list[dict] = []

    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.H * self.W

    def flat_index(self, ix, iy):
        """Global flat index; raises if the cell is outside the allocated bbox."""
        ix = np.asarray(ix, np.int64)
        iy = np.asarray(iy, np.int64)
        flat = (iy - self.oy) * self.W + (ix - self.ox)
        if flat.size and (flat.min() < 0 or flat.max() >= self.n_cells):
            raise IndexError(f"cell outside bbox: flat {flat.min()}..{flat.max()} "
                             f"of {self.n_cells}")
        return flat

    # ------------------------------------------------------------------
    def merge_frame(self, frame, weight: np.ndarray | None = None) -> dict:
        from stream_ortho.map.selection import unpack_cell

        ix, iy = unpack_cell(np.asarray(frame.cell_key, np.int64))
        flat = self.flat_index(ix, iy)
        z = np.asarray(frame.z, np.float32)
        rgb = np.asarray(frame.rgb, np.float32)
        conf = np.asarray(frame.conf, np.float32)
        w = np.ones(len(z), np.float32) if weight is None else np.asarray(weight, np.float32)
        self.n_frames += 1
        self.n_obs += len(z)
        if len(z) == 0:
            return self.stats()

        # ---- aggregate this frame's points per cell (one decision per frame,cell) ----
        # sort by (cell, conf): the **last** row of each group is its highest-conf
        # observation, which becomes the layer's representative colour
        order = np.lexsort((conf, flat))
        fs = flat[order]
        uniq, start = np.unique(fs, return_index=True)
        cnt = np.diff(np.append(start, len(fs)))
        rep = start + cnt - 1                      # index of the max-conf row per group
        w_s, z_s, c_s = w[order], z[order], conf[order]
        rgb_s = rgb[order]
        sw = np.add.reduceat(w_s, start)
        sz = np.add.reduceat(w_s * z_s, start)
        sz2 = np.add.reduceat(w_s * z_s * z_s, start)
        srgbf = np.add.reduceat(w_s[:, None] * rgb_s, start, axis=0)
        cmax = c_s[rep]
        rep_rgb = rgb_s[rep]
        zbar = sz / np.maximum(sw, 1e-6)

        # ---- pick a layer per cell (occ mask, no NaN) ----
        occ = self.n.reshape(-1, self.L)[uniq] > 0
        zr = self.z_ref.reshape(-1, self.L)[uniq]
        d = np.where(occ, np.abs(zbar[:, None] - zr), np.inf)
        best = np.argmin(d, axis=1)
        fits = d[np.arange(uniq.size), best] <= self.z_gap
        # ``argmax(~occ)`` already yields 0..L-1 for an **empty** cell (the previous
        # ``np.where(occ.any(1), ...)`` turned that into -1 and miscounted every first
        # observation as overflow)
        free_any = (~occ).any(1)
        free = np.argmax(~occ, 1)
        layer = np.where(fits, best, np.where(free_any, free, best))
        over = (~fits) & (~free_any)
        self.n_new_layer += int(((~fits) & free_any).sum())
        self.n_overflow_points += int(cnt[over].sum())
        self.n_overflow_cells += int(over.sum())
        if over.any() and len(self.overflow_samples) < 2000:
            for j in np.nonzero(over)[0][:5]:
                zs = zr[j][occ[j]]
                self.overflow_samples.append({
                    "kf": int(frame.kf), "z_new": float(zbar[j]),
                    "z_layers": [float(v) for v in zs],
                    "gap_to_nearest": float(np.min(np.abs(zs - zbar[j]))) if zs.size else None,
                    "n_layers": int(occ[j].sum())})

        target = uniq * self.L + layer          # unique per frame -> plain += is safe
        self.sum_w.reshape(-1)[target] += sw
        self.sum_wz.reshape(-1)[target] += sz
        self.sum_wz2.reshape(-1)[target] += sz2
        self.rgb.reshape(-1, 3)[target] = (
            self.rgb.reshape(-1, 3)[target].astype(np.float32) + srgbf).astype(np.float16)
        self.n.reshape(-1)[target] += cnt.astype(np.uint16)
        self.last_kf.reshape(-1)[target] = int(frame.kf)
        # representative colour: replace only where this frame's best observation beats
        # the stored one (vectorised, no Python loop)
        rbetter = cmax > self.rep_conf.reshape(-1)[target].astype(np.float32)
        self.rep_rgb.reshape(-1, 3)[target[rbetter]] = rep_rgb[rbetter].astype(np.float16)
        self.rep_conf.reshape(-1)[target[rbetter]] = cmax[rbetter].astype(np.float16)
        better = cmax > self.best_conf.reshape(-1)[target].astype(np.float32)
        self.owner.reshape(-1)[target[better]] = int(frame.kf)
        self.best_conf.reshape(-1)[target[better]] = cmax[better].astype(np.float16)
        self.z_ref.reshape(-1)[target] = (
            self.sum_wz.reshape(-1)[target]
            / np.maximum(self.sum_w.reshape(-1)[target], 1e-6))
        return None          # stats() is O(N) and must stay out of the per-frame path

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        occ = (self.n > 0)
        per_cell = occ.sum(-1)
        live = int((per_cell > 0).sum())
        hist = {"1": int((per_cell == 1).sum()), "2": int((per_cell == 2).sum()),
                ">=3": int((per_cell >= 3).sum())}
        mem = int(sum(a.nbytes for a in (self.z_ref, self.sum_w, self.sum_wz,
                                         self.sum_wz2, self.rgb, self.n, self.last_kf,
                                         self.owner, self.best_conf, self.rep_rgb,
                                         self.rep_conf))) / 1024 / 1024
        return {
            "n_frames": self.n_frames, "n_obs": self.n_obs,
            "bbox_cells": int(self.n_cells), "n_cells_live": live,
            "obs_per_cell": float(self.n_obs / max(live, 1)),
            "layers_hist": hist,
            "n_layers_total": int(occ.sum()),
            "n_new_layer": self.n_new_layer,
            "n_overflow_points": self.n_overflow_points,
            "n_overflow_cells": self.n_overflow_cells,
            "mem_mb": mem,
        }

    def sigma_z(self):
        """Per-(cell,layer) standard deviation of z, from the weighted sums."""
        sw = np.maximum(self.sum_w, 1e-6)
        mz = self.sum_wz / sw
        return np.sqrt(np.maximum(self.sum_wz2 / sw - mz * mz, 0.0))

    def to_cloud(self, layer: str = "top", tile_ids=None, gate: str | None = None,
                 occlusion: str | None = None) -> dict:
        """One point per live cell; ``layer='top'`` takes the highest, ``'bottom'`` lowest."""
        occ = (self.n > 0)
        any_occ = occ.any(-1)
        if layer == "top":
            pick = np.where(occ, self.z_ref, -np.inf).argmax(-1)
        else:
            pick = np.where(occ, self.z_ref, np.inf).argmin(-1)
        if gate in ("flat", "flat+support"):
            # facade discriminator (week9-09): a horizontal surface has a small in-layer
            # z spread, a wall does not.  Prefer the highest layer that is supported AND
            # flat (sigma_z <= 0.5 m); fall back to the plain top where none exists.
            # Observations are not deleted -- this only changes what gets rendered.
            flat = occ & (self.n >= 2) & (self.sigma_z() <= 0.5)
            has = flat.any(-1)
            alt = np.where(flat, self.z_ref, -np.inf).argmax(-1)
            pick = np.where(has, alt, pick)
        if gate == "flat+support":
            # then remove isolated unsupported cells (product-level; observations kept)
            from scipy.ndimage import median_filter

            occ2 = occ
            sel_n = np.take_along_axis(self.n, pick[..., None], -1)[..., 0]
            ztop = np.take_along_axis(self.z_ref, pick[..., None], -1)[..., 0]
            sup = occ2 & (self.n >= 2)
            zs = np.where(sup, self.z_ref, np.nan).max(-1)
            fill = np.nanmedian(zs) if np.isfinite(zs).any() else 0.0
            med = median_filter(np.where(np.isfinite(zs), zs, fill), size=5, mode="nearest")
            bad = occ2.any(-1) & (sel_n < 2) & (np.abs(ztop - med) > 1.5)
            any_occ = any_occ & ~bad
        if occlusion in ("drop", "expand"):
            # Nadir visibility (week9-12): the roof mask is computed on the **top height
            # field**, not per layer.  Per-layer sigma_z made "roof" mean "this layer is
            # flat", which only caught 26% of walls.  Here:
            #   ground = 21x21 minimum of the height field
            #   roof   = height above ground + 1 m, locally flat (<0.5 m per cell), support>=2
            #   roof_d = roof dilated by 2 cells -> the occlusion halo
            #   wall   = selected layer sigma_z > 0.5, inside roof_d, not roof itself
            from scipy.ndimage import (binary_dilation, distance_transform_edt,
                                       gaussian_filter, minimum_filter)
            sig = self.sigma_z()
            sel_sig = np.take_along_axis(sig, pick[..., None], -1)[..., 0]
            sel_n = np.take_along_axis(self.n, pick[..., None], -1)[..., 0]
            ztop = np.take_along_axis(self.z_ref, pick[..., None], -1)[..., 0]
            visit = np.asarray(any_occ)
            zf = np.where(visit, ztop, np.nan)
            med = np.nanmedian(zf) if np.isfinite(zf).any() else 0.0
            zfill0 = np.where(np.isfinite(zf), zf, med)
            ground = minimum_filter(zfill0, size=21, mode="nearest")
            zs = gaussian_filter(zfill0, 2.0)
            gy, gx = np.gradient(zs)
            grad = np.hypot(gx, gy)
            roof = visit & (ztop - ground > 1.0) & (grad < 0.5) & (sel_n >= 2)
            roof_d = binary_dilation(roof, iterations=2)
            wall = visit & (sel_sig > 0.5) & roof_d & ~roof
            # nearest roof cell for every cell (used by the "expand" variant)
            _, idx = distance_transform_edt(~roof, return_distances=True,
                                            return_indices=True)
            self._occ_stats = {"roof_cells": int(roof.sum()), "wall_cells": int(wall.sum()),
                               "wall_frac_of_visited": float(wall.sum() / max(visit.sum(), 1))}
            if occlusion == "drop":
                any_occ = visit & ~wall
                self._occ_stats["handled"] = int(wall.sum())
            else:
                sy, sx = idx[0], idx[1]
                zroof = np.where(roof, ztop, np.nan)[sy, sx]
                ok = wall & np.isfinite(zroof)
                ztop = np.where(ok, zroof, ztop)
                self._occ_stats["handled"] = int(ok.sum())
                self._occ_stats["wall_cells"] = int(wall.sum())
                wall = ok
        jj, ii = np.nonzero(any_occ)
        jj, ii = np.nonzero(any_occ)
        li = pick[jj, ii]
        zz = np.take_along_axis(self.z_ref, pick[..., None], -1)[..., 0][jj, ii]
        if occlusion == "expand":
            zz = np.where(wall[jj, ii], ztop[jj, ii], zz)
        sw = np.maximum(self.sum_w[jj, ii, li], 1e-6)
        rep = self.rep_rgb[jj, ii, li].astype(np.float32) / 255.0
        if occlusion == "expand" and "sy" in dir():
            pass
        mean = self.rgb[jj, ii, li].astype(np.float32) / sw[:, None] / 255.0
        has_rep = self.rep_conf[jj, ii, li] > 0
        cc = np.where(has_rep[:, None], rep, mean)
        ix = ii + self.ox
        iy = jj + self.oy
        px = (ix.astype(np.float64) + 0.5) * self.cell
        py = (iy.astype(np.float64) + 0.5) * self.cell
        P = np.stack([px, py, zz.astype(np.float64)], -1)
        W = (P @ self.T_WP[:3, :3].T + self.T_WP[:3, 3]).astype(np.float32)
        return {"means": W, "colors": np.clip(cc, 0, 1).astype(np.float32),
                "n": self.n[jj, ii, li].astype(np.int32)}

    # ------------------------------------------------------------------
    def to_gaussian_cloud(self, tile_ids=None, layer: str = "top", sigma_m: float = 0.18,
                          gate: str | None = None, occlusion: str | None = None):
        """``to_cloud`` -> a ``GaussianCloud`` ready for ``render_tile`` (P0-3).

        Splats are oriented with the ground normal (the plane frame's z axis rotated into
        the world), flattened to 0.1 sigma, and sized ``sigma_m`` in **metres** (0.6 x cell
        by default, i.e. slightly under one output pixel at 0.30 m GSD).  Opacity is
        ``min(1, n/2)``: a cell seen once is drawn faintly, a cell seen by two or more
        frames is opaque -- deliberately *not* forced to 1 like the old
        ``--opacity-floor 0.95`` path, because the observation count is now a real signal.
        """
        import torch

        from stream_ortho.geometry import R_to_quat_wxyz
        from stream_ortho.ortho.rasterizer import GaussianCloud

        d = self.to_cloud(layer=layer, tile_ids=tile_ids, gate=gate,
                          occlusion=occlusion)
        n = len(d["means"])
        q = R_to_quat_wxyz(self.T_WP[:3, :3])
        quats = np.repeat(np.asarray(q, np.float32)[None], n, 0)
        scales = np.tile(np.array([sigma_m, sigma_m, 0.1 * sigma_m], np.float32), (n, 1))
        opacity = np.minimum(1.0, d["n"].astype(np.float32) / 2.0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return GaussianCloud(
            means=torch.as_tensor(d["means"], device=dev),
            quats=torch.as_tensor(quats, device=dev),
            scales=torch.as_tensor(scales, device=dev),
            opacity=torch.as_tensor(opacity, device=dev),
            colors=torch.as_tensor(d["colors"], device=dev),
            owner=torch.zeros(n, dtype=torch.long, device=dev), meta={})

    def gate_diagnostics(self) -> dict:
        """How much of the facade can the flat gate actually replace? (week9-10)"""
        occ = self.n > 0
        sig = self.sigma_z()
        flat = occ & (self.n >= 2) & (sig <= 0.5)
        has_flat = flat.any(-1)
        # the **rendered** layer: flat pick where available, else the plain top
        alt = np.where(flat, self.z_ref, -np.inf).argmax(-1)
        top = np.where(occ, self.z_ref, -np.inf).argmax(-1)
        pick = np.where(has_flat, alt, top)
        sel_sig = np.take_along_axis(sig, pick[..., None], -1)[..., 0]
        sel_n = np.take_along_axis(self.n, pick[..., None], -1)[..., 0]
        facade = occ.any(-1) & (sel_n >= 2) & (sel_sig > 0.5)
        no_fb = facade & ~has_flat
        return {"n_cells_out": int(occ.any(-1).sum()),
                "n_facade_kept": int(facade.sum()),
                "n_facade_no_fallback": int(no_fb.sum()),
                "facade_no_fallback_frac": float(no_fb.sum() / max(facade.sum(), 1))}

    def occlusion_stats(self) -> dict:
        return getattr(self, "_occ_stats", {})
