"""Global Gaussian map: staging -> gate -> voxel dedup/merge -> commit -> budget.

Implements the week-4 part of the design doc (§7.5):

* ``GaussianPatch`` (one keyframe) -> **staging**;
* voxel dedup (voxel = ``2 x GSD``) with a *conservative* merge (same voxel,
  close position/colour/scale -> running average, ``n_obs += 1``);
* **commit** when a staged point is re-observed (``n_obs >= 2``) or old enough
  (``age >= tau``) -- the multi-view consistency idea of §2.2;
* a *minimal* budget enforcer (``enforce_budget``) so the week-5 policy can be
  plugged in without touching the data structures.

Everything is numpy/CPU; ``to_cloud`` converts to the torch cloud only for
rendering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stream_ortho.map.gaussian import (
    GaussianPatch,
    array_bytes,
    concat_arrays,
    empty_map_arrays,
    take_rows,
)


@dataclass
class MapConfig:
    voxel_size: float = 0.4          # metres; design doc: 2 x GSD
    merge_dist_frac: float = 0.75    # of voxel_size
    color_tol: float = 0.35          # L1 over RGB
    scale_ratio_tol: float = 3.0
    commit_min_obs: int = 2          # multi-view confirmation
    commit_max_age: int = 2          # ... or age in keyframes
    dedup: bool = True
    gate: bool = True
    fusion: str = "weighted"        # mean | weighted | best | robust
    exposure_norm: bool = False      # per-patch gain/offset matched to the map
    merge_radius: float = 0.5        # "robust": observations further apart than this
                                     # are NOT averaged, the better one wins
    max_points: int = 0              # 0 = no budget (week 5 sets it)
    budget_strategy: str = "conf"    # random | conf
    seed: int = 0


class GaussianMap:
    def __init__(self, cfg: MapConfig, T_WP: np.ndarray, z_ground: float = 0.0,
                 device: str = "cuda:0", gate_cfg=None):
        self.cfg = cfg
        self.T_WP = np.asarray(T_WP, np.float64)
        self.z_ground = float(z_ground)
        self.device = device
        self.gate_cfg = gate_cfg
        self.committed = empty_map_arrays()
        self.staged = empty_map_arrays()
        self._vox_c: Dict[int, int] = {}
        self._vox_s: Dict[int, int] = {}
        self._sc_h = np.zeros(0, np.int64)      # sorted hashes of committed
        self._sc_i = np.zeros(0, np.int64)      # row index for each sorted hash
        self._ss_h = np.zeros(0, np.int64)
        self._ss_i = np.zeros(0, np.int64)
        self.kf_id = -1
        self.rng = np.random.default_rng(cfg.seed)
        self.log: List[dict] = []
        self.t_ingest = 0.0
        self.t_commit = 0.0
        self.n_evicted = 0               # points removed by tile-level selection
        self.n_free_reject = 0           # points vetoed by the nadir free-space prior
        self.nfp = None                  # optional NadirHeightField (map/freespace.py)
        self._nfp_last = None
        self.n_exposure_fix = 0          # patches whose exposure was normalised
        self.n_merge_hits = 0            # observations that hit an existing voxel
        self.n_merge_conflicts = 0       # ... of which disagreed by > merge_radius

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _hashes(self, means: np.ndarray) -> np.ndarray:
        """Vectorised voxel hash (int64) for ``means``."""
        from stream_ortho.map.gaussian import key_hashes, voxel_keys

        return key_hashes(voxel_keys(means, self.cfg.voxel_size))

    @property
    def n_committed(self) -> int:
        return int(len(self.committed["means"]))

    @property
    def n_staged(self) -> int:
        return int(len(self.staged["means"]))

    def n_total(self) -> int:
        return self.n_committed + self.n_staged

    def memory_mb(self) -> float:
        return (array_bytes(self.committed) + array_bytes(self.staged)) / 1e6

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    # -- index maintenance ------------------------------------------------
    def _merge_index(self, which: str = "c", n_old: int = 0):
        """Insert the rows ``[n_old:]`` into the existing sorted index in O(N+M).

        ``commit`` used to rebuild the whole index with an ``argsort`` over all N rows,
        which dominated the frame time (measured: 1.05 s at N=9.4 M, 3.08 s before the
        per-commit dict was removed).  Merging two sorted arrays is enough, because
        hashes only ever get *appended*.
        """
        h_all = self._hashes(self.committed["means"])
        if n_old <= 0 or getattr(self, "_sc_h", None) is None or len(self._sc_h) != n_old:
            order = np.argsort(h_all) if len(h_all) else np.zeros(0, np.int64)
            self._sc_h, self._sc_i = h_all[order], order.astype(np.int64)
            return
        new_h = h_all[n_old:]
        if len(new_h) == 0:
            return
        order = np.argsort(new_h)
        nh = new_h[order]
        new_idx = (np.arange(len(new_h), dtype=np.int64) + n_old)[order]
        old_h, old_i = self._sc_h, self._sc_i
        # positions of the new rows among the old ones, and of the old ones among the new
        pos_new = np.searchsorted(old_h, nh, side="left") + np.arange(len(nh), dtype=np.int64)
        pos_old = np.arange(len(old_h), dtype=np.int64) + \
            np.searchsorted(nh, old_h, side="left")
        mh = np.empty(len(old_h) + len(nh), np.int64)
        mi = np.empty_like(mh)
        mh[pos_old] = old_h
        mi[pos_old] = old_i
        mh[pos_new] = nh
        mi[pos_new] = new_idx
        self._sc_h, self._sc_i = mh, mi

    def _rebuild_index(self, which: str = "c", n_old: int = 0):
        if which == "c":
            if n_old > 0:
                self._merge_index("c", n_old)
                return
            h = self._hashes(self.committed["means"]) if self.n_committed else np.zeros(0, np.int64)
            order = np.argsort(h)
            self._sc_h, self._sc_i = h[order], order.astype(np.int64)
            self._vox_c = None          # dropped: an 8M-entry Python dict per commit
                                        # cost seconds; _match uses the sorted arrays
        else:
            h = self._hashes(self.staged["means"]) if self.n_staged else np.zeros(0, np.int64)
            order = np.argsort(h)
            self._ss_h, self._ss_i = h[order], order.astype(np.int64)
            self._vox_s = None

    def _match(self, h: np.ndarray, which: str = "c"):
        """Rows (in the target arrays) whose voxel hash equals ``h`` (or -1)."""
        sh, si = (self._sc_h, self._sc_i) if which == "c" else (self._ss_h, self._ss_i)
        if len(sh) == 0:
            return np.full(len(h), -1, np.int64)
        pos = np.clip(np.searchsorted(sh, h), 0, len(sh) - 1)
        hit = sh[pos] == h
        out = np.full(len(h), -1, np.int64)
        out[hit] = si[pos[hit]]
        return out

    def _geom_ok(self, arr: dict, idx: np.ndarray, patch: GaussianPatch,
                 sel: np.ndarray) -> np.ndarray:
        """Vectorised merge test (distance / colour / scale ratio) for matched rows."""
        m = float(self.cfg.merge_dist_frac) * float(self.cfg.voxel_size)
        d = np.linalg.norm(arr["means"][idx] - patch.means[sel], axis=1)
        dc = np.abs(arr["colors"][idx] - patch.colors[sel]).sum(axis=1)
        smax = np.maximum(arr["scales"][idx].max(axis=1), patch.scales[sel].max(axis=1))
        smin = np.minimum(arr["scales"][idx].min(axis=1), patch.scales[sel].min(axis=1))
        return (d <= m) & (dc <= self.cfg.color_tol) & (smax / np.maximum(smin, 1e-9) <= self.cfg.scale_ratio_tol)

    def _bulk_merge(self, arr: dict, idx: np.ndarray, patch: GaussianPatch, sel: np.ndarray):
        """Merge a re-observation into an existing Gaussian.

        ``fusion="weighted"`` (default): quality-weighted average, where the
        quality is the view weight stored per observation (near-nadir and
        image-centre observations dominate) -- borrowed from Map2DFusion's
        per-pixel weight image (``w = 1 - r/r_max``).
        ``fusion="best"``: winner-take-all, replace with the better observation.
        ``fusion="mean"``: the old behaviour (plain running average).
        """
        mode = getattr(self.cfg, "fusion", "weighted")
        w_new = np.asarray(patch.weight[sel], np.float64)
        w_old = arr["wsum"][idx].astype(np.float64)
        # diagnostic: how often a new observation lands on an existing voxel at all
        self.n_merge_hits += int(len(idx))
        if mode == "robust":
            # two observations that disagree by more than merge_radius are *different
            # surfaces* (e.g. one frame sees the roof, another the ground below it);
            # averaging them is what produced the "multi-layer" buildings.  Keep the
            # better observation instead, and count the conflict.
            d = np.linalg.norm(patch.means[sel] - arr["means"][idx], axis=1)
            conflict = d > float(getattr(self.cfg, "merge_radius", 0.5))
            self.n_merge_conflicts += int(conflict.sum())
            if conflict.any():
                better = w_new[conflict] >= w_old[conflict]
                j = idx[conflict][better]
                s_sel = np.asarray(sel)[conflict][better]
                if len(j):
                    for key, src in (("means", patch.means), ("colors", patch.colors),
                                     ("scales", patch.scales)):
                        arr[key][j] = src[s_sel]
                    arr["wsum"][j] = w_new[conflict][better]
                    arr["opacity"][j] = patch.opacity[s_sel]
            keep = ~conflict
            idx, sel = idx[keep], np.asarray(sel)[keep]
            w_new = w_new[keep]
            w_old = w_old[keep]
            if len(idx) == 0:
                return
        w_opa = w_new  # opacity follows the same weighting
        if mode == "best":
            better = w_new >= w_old
            if better.any():
                j = idx[better]
                s_sel = np.asarray(sel)[better]
                for key, src in (("means", patch.means), ("colors", patch.colors),
                                 ("scales", patch.scales)):
                    arr[key][j] = src[s_sel]
                arr["wsum"][j] = w_new[better]
        else:
            if mode == "mean":
                w_new = np.ones_like(w_new)
                w_old = arr["n_obs"][idx].astype(np.float64)
            tot = np.maximum(w_old + w_new, 1e-9)
            wv = (w_new / tot).astype(np.float32)[:, None]
            for key, src in (("means", patch.means), ("colors", patch.colors),
                             ("scales", patch.scales)):
                arr[key][idx] = (1 - wv) * arr[key][idx] + wv * src[sel]
            arr["wsum"][idx] = (w_old + w_new).astype(np.float32)
        if mode == "best":
            opa_w = np.asarray(better, np.float64)
        else:
            opa_w = np.asarray(w_opa, np.float64) / np.maximum(
                np.asarray(arr["n_obs"][idx], np.float64) + 1.0, 1e-9)
        arr["opacity"][idx] = ((1 - opa_w) * arr["opacity"][idx]
                               + opa_w * patch.opacity[sel]).astype(np.float32)
        arr["conf"][idx] = np.maximum(arr["conf"][idx], patch.conf[sel])
        arr["n_obs"][idx] = np.minimum(arr["n_obs"][idx].astype(np.int32) + 1, 32767).astype(np.int16)
        arr["last_seen"][idx] = self.kf_id
        arr["kind"][idx] = np.maximum(arr["kind"][idx], patch.kind[sel])

    def ingest(self, patch: GaussianPatch, kf_id: int) -> dict:
        """Add one keyframe's patch (gate -> dedup -> merge -> stage), vectorised."""
        t0 = time.time()
        self.kf_id = int(kf_id)
        patch.kf_id = int(kf_id)
        if patch.owner < 0:
            patch.owner = int(kf_id)

        if self.nfp is not None:
            # NFP veto (map/freespace.py): drop candidates floating in space the map has
            # already seen through.  Admission only -- nothing is averaged or rewritten.
            keep, _info = self.nfp.veto(patch.means)
            n_rej = int((~keep).sum())
            if n_rej:
                patch = patch.subset(keep)
                self.n_free_reject += n_rej
        self._nfp_last = patch

        if getattr(self.cfg, "exposure_norm", False) and self.n_committed > 5000 \
                and patch.colors.ndim == 2 and patch.colors.shape[1] >= 3:
            # Match the patch's colours to the map **on the ground cells they share**.
            # The naive bbox-median version *hurt* (PSNR 19.34 -> 19.21) because a
            # frame's bbox and the map's bbox contain different content; cells that
            # both sides actually observe are the right place to compare.
            Tw = self.T_WP
            cell = 1.0

            def _ground(means):
                P = (np.asarray(means, np.float64) - Tw[:3, 3]) @ Tw[:3, :3]
                m = (P[:, 2] > -20.0) & (P[:, 2] < 2.0)
                return P[m][:, :2], m

            def _cell_mean(keys, cols):
                order = np.lexsort((keys[:, 1], keys[:, 0]))
                ks, cs = keys[order], cols.astype(np.float64)[order]
                uniq, start_i, counts = np.unique(ks, axis=0, return_index=True,
                                                  return_counts=True)
                sums = np.add.reduceat(cs, start_i, axis=0)
                return uniq, sums / np.maximum(counts, 1)[:, None]

            pc, pmask = _ground(patch.means)
            mc, mmask = _ground(self.committed["means"])
            if len(pc) > 200 and len(mc) > 200:
                up, cp = _cell_mean(np.floor(pc / cell).astype(np.int64),
                                    patch.colors[pmask][:, :3])
                um, cm = _cell_mean(np.floor(mc / cell).astype(np.int64),
                                    self.committed["colors"][mmask][:, :3])
                from stream_ortho.map.selection import pack_cell

                key_p = pack_cell(up[:, 0], up[:, 1])
                key_m = pack_cell(um[:, 0], um[:, 1])
                common = np.intersect1d(key_p, key_m)
                if len(common) > 50:
                    a = np.median(cp[np.isin(key_p, common)], axis=0)
                    b = np.median(cm[np.isin(key_m, common)], axis=0)
                    if np.all(a > 1e-3):
                        gain = np.clip(b / a, 0.8, 1.2).astype(np.float32)
                        patch.colors = np.clip(patch.colors * gain, 0.0, 1.0)
                        self.n_exposure_fix += 1

        gate_res = None
        if self.cfg.gate and self.gate_cfg is not None:
            from stream_ortho.map.commit import commit_gate

            gate_res = commit_gate(patch, self.gate_cfg, self.T_WP, self.z_ground)
            patch = patch.subset(gate_res["valid"])
        n_in = len(patch)
        if n_in == 0:
            return {"n_in": 0, "n_merged_committed": 0, "n_merged_staged": 0,
                    "n_new": 0, "gate": gate_res, "dt": time.time() - t0}

        staged_before = self.n_staged
        h = self._hashes(patch.means)
        if self.cfg.dedup:
            _, first = np.unique(h, return_index=True)
            keep_idx = np.sort(first)
            h, patch = h[keep_idx], patch.subset(keep_idx)

        alive = np.ones(len(h), bool)
        merged_c = merged_s = 0

        # 1) committed
        idx_c = self._match(h, "c")
        hit = (idx_c >= 0) & alive
        if self.cfg.dedup and hit.any():
            sel = np.where(hit)[0]
            ok = self._geom_ok(self.committed, idx_c[sel], patch, sel)
            sel, idx_c_ok = sel[ok], idx_c[sel][ok]
            if len(sel):
                self._bulk_merge(self.committed, idx_c_ok, patch, sel)
                alive[sel] = False
                merged_c = len(sel)

        # 2) staged
        idx_s = self._match(h, "s")
        hit = (idx_s >= 0) & alive
        if self.cfg.dedup and hit.any():
            sel = np.where(hit)[0]
            ok = self._geom_ok(self.staged, idx_s[sel], patch, sel)
            sel, idx_s_ok = sel[ok], idx_s[sel][ok]
            if len(sel):
                # staged merge counts as a re-observation
                n_old = self.staged["n_obs"][idx_s_ok].astype(np.float32)
                w = 1.0 / (n_old + 1.0)
                for key, src in (("means", patch.means), ("colors", patch.colors),
                                 ("scales", patch.scales)):
                    self.staged[key][idx_s_ok] = (1 - w)[:, None] * self.staged[key][idx_s_ok] + w[:, None] * src[sel]
                self.staged["opacity"][idx_s_ok] = (1 - w) * self.staged["opacity"][idx_s_ok] + w * patch.opacity[sel]
                self.staged["n_obs"][idx_s_ok] = np.minimum(self.staged["n_obs"][idx_s_ok].astype(np.int32) + 1, 32767).astype(np.int16)
                self.staged["last_seen"][idx_s_ok] = self.kf_id
                self.staged["kind"][idx_s_ok] = np.maximum(self.staged["kind"][idx_s_ok], patch.kind[sel])
                alive[sel] = False
                merged_s = len(sel)
        # remaining points that collide with an existing voxel but fail the merge test
        # are dropped (dedup semantics: one Gaussian per voxel)
        if self.cfg.dedup:
            alive &= (idx_c < 0) & (idx_s < 0)

        new_sel = np.where(alive)[0]
        if len(new_sel):
            sub = patch.subset(new_sel)
            self._append_rows(self.staged, sub, n_obs=1, born=self.kf_id)
            self._rebuild_index("s")

        dt = time.time() - t0
        self.t_ingest += dt
        return {"n_in": int(n_in), "n_unique_voxels": int(len(h)),
                "n_merged_committed": int(merged_c), "n_merged_staged": int(merged_s),
                "n_new": int(self.n_staged - staged_before), "gate": gate_res, "dt": dt}

    def _append_rows(self, arrays: dict, patch: GaussianPatch, n_obs: int, born: int):
        n = len(patch)
        if n == 0:
            return
        add = {
            "means": patch.means, "quats": patch.quats, "scales": patch.scales,
            "opacity": patch.opacity, "colors": patch.colors, "conf": patch.conf,
            "owner": np.full(n, patch.owner, np.int32),
            "n_obs": np.full(n, n_obs, np.int16),
            "born_kf": np.full(n, born, np.int32),
            "last_seen": np.full(n, self.kf_id, np.int32),
            "kind": patch.kind.astype(np.int8), "theta": patch.theta.astype(np.float32),
            "wsum": patch.weight.astype(np.float32),
            "dead": np.zeros(n, np.int8),
        }
        for k, v in add.items():
            arrays[k] = np.concatenate([arrays[k], np.asarray(v, dtype=arrays[k].dtype)], axis=0)

    def _merge_into(self, arrays: dict, j: int, patch: GaussianPatch, i: int):
        n_old = float(arrays["n_obs"][j])
        w = 1.0 / (n_old + 1.0)
        arrays["means"][j] = (1 - w) * arrays["means"][j] + w * patch.means[i]
        arrays["colors"][j] = (1 - w) * arrays["colors"][j] + w * patch.colors[i]
        arrays["scales"][j] = (1 - w) * arrays["scales"][j] + w * patch.scales[i]
        arrays["opacity"][j] = (1 - w) * arrays["opacity"][j] + w * patch.opacity[i]
        arrays["conf"][j] = max(float(arrays["conf"][j]), float(patch.conf[i]))
        arrays["n_obs"][j] = min(int(arrays["n_obs"][j]) + 1, 32767)
        arrays["last_seen"][j] = self.kf_id
        arrays["kind"][j] = max(int(arrays["kind"][j]), int(patch.kind[i]))

    def _append_row(self, arrays: dict, patch: GaussianPatch, i: int,
                    n_obs: int, born: int, owner: int):
        row = {
            "means": patch.means[i], "quats": patch.quats[i], "scales": patch.scales[i],
            "opacity": np.float32(patch.opacity[i]), "colors": patch.colors[i],
            "conf": np.float32(patch.conf[i]), "owner": np.int32(owner),
            "n_obs": np.int16(n_obs), "born_kf": np.int32(born),
            "last_seen": np.int32(self.kf_id), "kind": np.int8(patch.kind[i]),
            "theta": np.float32(patch.theta[i]),
        }
        for k, v in row.items():
            a = np.asarray(v)
            a = a.reshape((1,) + a.shape) if arrays[k].ndim > 1 else a.reshape(1)
            arrays[k] = np.concatenate([arrays[k], a.astype(arrays[k].dtype)], axis=0)

    def _append_staged(self, patch: GaussianPatch, i: int, key: tuple):
        j = len(self.staged["means"])
        self._append_row(self.staged, patch, i, n_obs=1, born=self.kf_id,
                         owner=int(patch.owner))
        # (dict index removed -- see _rebuild_index)

    # ------------------------------------------------------------------
    # commit
    # ------------------------------------------------------------------
    def commit(self, kf_id: Optional[int] = None) -> dict:
        """Move confirmed/aged staged points into the committed map (vectorised)."""
        t0 = time.time()
        n_before = self.n_committed
        if kf_id is not None:
            self.kf_id = int(kf_id)
        if self.nfp is not None and self._nfp_last is not None:
            self.nfp.update(self._nfp_last.means, int(self.kf_id))
            self._nfp_last = None
        n_st = self.n_staged
        if n_st == 0:
            return {"committed": 0, "kept_staged": 0, "n_committed": self.n_committed}

        obs = self.staged["n_obs"]
        age = self.kf_id - self.staged["born_kf"]
        ready = (obs >= self.cfg.commit_min_obs) | (age >= self.cfg.commit_max_age)
        if not ready.any():
            return {"committed": 0, "kept_staged": int(n_st), "n_committed": self.n_committed}

        moved = take_rows(self.staged, np.where(ready)[0])
        keep = take_rows(self.staged, np.where(~ready)[0])
        self.staged = keep
        self._rebuild_index("s")

        h = self._hashes(moved["means"])
        idx_c = self._match(h, "c")
        hit = idx_c >= 0
        n_merged = 0
        if hit.any():
            sel = np.where(hit)[0]
            idx_ok = idx_c[sel]
            # weighted merge by n_obs
            n_old = self.committed["n_obs"][idx_ok].astype(np.float64)
            n_new = moved["n_obs"][sel].astype(np.float64)
            w = (n_new / np.maximum(n_old + n_new, 1e-9)).astype(np.float32)[:, None]
            for key in ("means", "colors", "scales", "opacity"):
                src = moved[key][sel]
                if src.ndim == 1:
                    src = src[:, None]
                    self.committed[key][idx_ok] = ((1 - w[:, 0]) * self.committed[key][idx_ok]
                                                   + w[:, 0] * src[:, 0])
                else:
                    self.committed[key][idx_ok] = (1 - w) * self.committed[key][idx_ok] + w * src
            self.committed["conf"][idx_ok] = np.maximum(self.committed["conf"][idx_ok], moved["conf"][sel])
            self.committed["n_obs"][idx_ok] = np.minimum(
                self.committed["n_obs"][idx_ok].astype(np.int32) + moved["n_obs"][sel], 32767).astype(np.int16)
            self.committed["last_seen"][idx_ok] = self.kf_id
            self.committed["kind"][idx_ok] = np.maximum(self.committed["kind"][idx_ok], moved["kind"][sel])
            n_merged = len(sel)

        add_sel = np.where(~hit)[0]
        if len(add_sel):
            sub = take_rows(moved, add_sel)          # moved is a dict of arrays
            for k in self.committed.keys():
                self.committed[k] = np.concatenate(
                    [self.committed[k], np.asarray(sub[k], dtype=self.committed[k].dtype)], axis=0)
            self._rebuild_index("c", n_old=n_before)

        self.t_commit += time.time() - t0
        return {"committed": int(len(moved)), "kept_staged": int(len(keep["means"])),
                "n_committed": self.n_committed, "n_merged_on_commit": int(n_merged)}

    # ------------------------------------------------------------------
    # budget (week 5 will replace the score)
    # ------------------------------------------------------------------
    def evict_tile(self, grid, i: int, j: int, owner=None, halo: float = 0.0) -> int:
        """Mark committed points inside tile ``(i, j)`` as dead (P1 selection).

        When ``owner`` is given only that keyframe's points are evicted, so a tile can
        switch content without touching the neighbouring tiles.
        """
        if self.n_committed == 0:
            return 0
        c = self.committed
        P = (c["means"] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
        s_tile = grid.tile * grid.gsd
        inside = ((P[:, 0] >= i * s_tile - halo) & (P[:, 0] < (i + 1) * s_tile + halo) &
                  (P[:, 1] >= j * s_tile - halo) & (P[:, 1] < (j + 1) * s_tile + halo))
        if owner is not None:
            inside &= (c["owner"] == int(owner))
        n = int(inside.sum())
        if n:
            c["dead"][inside] = 1
            self.n_evicted += n
        return n

    def collapse_cells(self, grid, cell: float, keep_n: int = 1,
                       prefer: str = "quality", locked=None, avg_color: bool = True) -> int:
        """Apply the 2.5D cell selection **in place** (streaming P1'').

        Keeps the best (or highest) ``keep_n`` observation per ``cell``-metre cell and
        marks the rest dead.  Cost is O(N log N) (measured: 2 M points in ~0.35 s), so
        it can run every few keyframes inside the streaming loop instead of only at the
        very end.
        """
        from stream_ortho.map.selection import quality_score, select_by_cell

        if self.n_committed == 0:
            return 0
        c = self.committed
        live = (c["dead"] == 0) if "dead" in c else np.ones(self.n_committed, bool)
        idx = np.where(live)[0]
        if len(idx) < 1000:
            return 0
        q = quality_score(conf=c["conf"][idx], opacity=c["opacity"][idx],
                          wsum=c["wsum"][idx], n_obs=c["n_obs"][idx])
        if locked:
            # **freeze**: for a locked cell keep the observation closest to the locked
            # height instead of the best-quality one, so a later frame cannot replace an
            # already-stable surface (the "new frame pulls the building" symptom)
            P = (c["means"][idx] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
            ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
            from stream_ortho.map.selection import pack_cell

            key = pack_cell(ij[:, 0], ij[:, 1])
            qq = q.copy()
            for k, zlock in locked.items():
                m = key == int(k)
                if not m.any():
                    continue
                qq[m] = 1.0 - np.abs(P[m, 2] - float(zlock)) / float(max(cell, 1e-3))
            keep = idx[select_by_cell(c["means"][idx], self.T_WP, cell=cell, q=qq,
                                      keep_n=keep_n, prefer="quality")]
        else:
            if prefer == "column":
                from stream_ortho.map.selection import select_column_top
                ki, col_out = select_column_top(c["means"][idx], c["colors"][idx],
                                                self.T_WP, cell=cell)
                keep = idx[ki]
                if len(keep):
                    c["colors"][keep] = col_out
            else:
                keep = idx[select_by_cell(c["means"][idx], self.T_WP, cell=cell, q=q,
                                          keep_n=keep_n, prefer=prefer)]
        if avg_color and len(keep):
            # P0-3a (week8-end-48): this block was missing entirely -- ``avg_color`` used
            # to exist only in the signature, so the earlier "C is active" claim was
            # wrong.  The mean must be taken over **all live observations of the cell**
            # (not over the survivors, which are already 1/cell and would give the mean
            # of a single sample).
            from stream_ortho.map.selection import mean_colors_by_cell, pack_cell

            live = (c["dead"] == 0) if "dead" in c else np.ones(self.n_committed, bool)
            P_all = (c["means"][live] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
            ij_all = np.floor(P_all[:, :2] / float(cell)).astype(np.int64)
            key_all = pack_cell(ij_all[:, 0], ij_all[:, 1])
            mean_all = mean_colors_by_cell(c["means"][live], c["colors"][live],
                                           self.T_WP, cell=cell)
            uniq, first = np.unique(key_all, return_index=True)
            lut = {int(k): mean_all[f] for k, f in zip(uniq, first)}
            P_k = (c["means"][keep] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
            ijk = np.floor(P_k[:, :2] / float(cell)).astype(np.int64)
            keys_k = pack_cell(ijk[:, 0], ijk[:, 1])
            for n_i, kk in enumerate(keys_k):
                v = lut.get(int(kk))
                if v is not None:
                    c["colors"][keep[n_i]] = v
        drop = np.ones(self.n_committed, bool)
        drop[keep] = False
        drop &= live
        n = int(drop.sum())
        if n:
            c["dead"][drop] = 1
            self.n_evicted += n
        return n

    def evict_tiles(self, grid, tiles) -> int:
        """Batch version of :meth:`evict_tile`: one O(N) pass for many tiles."""
        if self.n_committed == 0 or not tiles:
            return 0
        c = self.committed
        P = (c["means"] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
        s_tile = grid.tile * grid.gsd
        ii = np.floor(P[:, 0] / s_tile).astype(np.int64)
        jj = np.floor(P[:, 1] / s_tile).astype(np.int64)
        from stream_ortho.map.selection import pack_cell

        ids = pack_cell(ii, jj)
        wt = np.asarray(tiles, np.int64)
        want = pack_cell(wt[:, 0], wt[:, 1])
        inside = np.isin(ids, want)
        n = int(inside.sum())
        if n:
            c["dead"][inside] = 1
            self.n_evicted += n
        return n

    def enforce_budget(self, n_max: int, strategy: str = "conf") -> dict:
        """Evict down to ``n_max`` with the chosen scoring policy (week-5)."""
        from stream_ortho.map.scoring import composition, keep_mask

        n = self.n_committed
        if n_max <= 0 or n <= n_max:
            return {"evicted": 0, "n_committed": n,
                    "composition": composition(self.committed, np.ones(n, bool))}

        if strategy.startswith("cell"):
            # (1) budget must never fall below the number of occupied cells, otherwise
            # whole cells lose their only splat and the ortho gets *salt-and-pepper*
            # holes (measured in S49: N oscillated 1.8M-12M while there were ~1.9M
            # occupied 0.30 m cells).  Here we keep one splat per cell and, if we still
            # exceed the budget, drop **whole cells** (lowest score first) so the holes
            # that remain are contiguous instead of scattered.
            from stream_ortho.map.selection import pack_cell, quality_score, select_by_cell

            c = self.committed
            cell = float(strategy.split(":")[1]) if ":" in strategy else 0.45
            q = quality_score(conf=c["conf"], opacity=c["opacity"], wsum=c["wsum"],
                              n_obs=c["n_obs"])
            keep1 = select_by_cell(c["means"], self.T_WP, cell=cell, q=q, keep_n=1)
            P = (c["means"][keep1] - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
            ij = np.floor(P[:, :2] / cell).astype(np.int64)
            key = pack_cell(ij[:, 0], ij[:, 1])
            if len(keep1) > n_max:                     # drop whole cells
                order = np.argsort(-q[keep1])
                _, first = np.unique(key[order], return_index=True)
                cell_best = order[first]
                keep_cells = cell_best[np.argsort(-q[keep1][cell_best])][: int(n_max)]
                keep = np.zeros(n, bool)
                keep[keep1[keep_cells]] = True
            else:
                keep = np.zeros(n, bool)
                keep[keep1] = True
            comp = composition(c, keep)
            self.committed = take_rows(c, np.where(keep)[0])
            self.staged = empty_map_arrays()
            self._rebuild_index("c")
            self._rebuild_index("s")
            return {"evicted": int(n - keep.sum()), "n_committed": int(keep.sum()),
                    "strategy": strategy, "composition": comp,
                    "n_cells": int(len(keep1))}

        keep = keep_mask(self.committed, n_max, strategy=strategy, rng=self.rng)
        comp = composition(self.committed, keep)
        self.committed = take_rows(self.committed, np.where(keep)[0])
        self.staged = empty_map_arrays()
        self._rebuild_index("c")
        self._rebuild_index("s")
        return {"evicted": int(n - len(keep)), "n_committed": int(len(keep)),
                "strategy": strategy, "composition": comp}

    # ------------------------------------------------------------------
    def duplicate_rate(self, voxel: Optional[float] = None) -> float:
        """Fraction of committed Gaussians that share a voxel with another one."""
        n = self.n_committed
        if n == 0:
            return 0.0
        if voxel is None:
            h = self._hashes(self.committed["means"])
        else:
            from stream_ortho.map.gaussian import key_hashes, voxel_keys

            h = key_hashes(voxel_keys(self.committed["means"], float(voxel)))
        uniq = int(np.unique(h).size)
        return float(1.0 - uniq / n)

    def to_cloud(self, device: Optional[str] = None):
        """Convert the committed map to a renderable GaussianCloud."""
        import torch

        from stream_ortho.ortho.rasterizer import GaussianCloud

        dev = device or self.device
        c = self.committed
        if "dead" in c and bool(np.any(c["dead"])):
            live = c["dead"] == 0
            c = {k: v[live] for k, v in c.items()}
        return GaussianCloud(
            torch.as_tensor(c["means"], device=dev),
            torch.as_tensor(c["quats"], device=dev),
            torch.as_tensor(c["scales"], device=dev),
            torch.as_tensor(c["opacity"], device=dev),
            torch.as_tensor(c["colors"], device=dev),
            owner=torch.as_tensor(c["owner"], device=dev),
        )

    def stats(self, **extra) -> dict:
        c = self.committed
        out = {
            "n_committed": self.n_committed,
            "n_staged": self.n_staged,
            "n_total": self.n_total(),
            "memory_mb": round(self.memory_mb(), 2),
            "duplicate_rate": round(self.duplicate_rate(), 4),
            "n_obs_median": float(np.median(c["n_obs"])) if len(c["n_obs"]) else 0.0,
            "n_obs_p95": float(np.percentile(c["n_obs"], 95)) if len(c["n_obs"]) else 0.0,
            "t_ingest_s": round(self.t_ingest, 2),
            "t_commit_s": round(self.t_commit, 2),
            "cfg": {
                "voxel_size": self.cfg.voxel_size, "merge_dist_frac": self.cfg.merge_dist_frac,
                "color_tol": self.cfg.color_tol, "commit_min_obs": self.cfg.commit_min_obs,
                "commit_max_age": self.cfg.commit_max_age, "dedup": self.cfg.dedup,
                "gate": self.cfg.gate, "max_points": self.cfg.max_points,
                "budget_strategy": self.cfg.budget_strategy,
            },
        }
        out.update(extra)
        return out
