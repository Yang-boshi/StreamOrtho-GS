"""NFP -- Nadir Free-Space Prior (the only part of "NADIR-GS" we keep).

**What it is.** A 1 m *column* height field over the plane frame.  For every column it
stores the observed height range ``(H_lo, H_hi)``, how many independent keyframes
support it (``n``) and a state:

* ``resolved`` : ``n >= 2`` and ``H_hi - H_lo <= tau_mix``  (the column has one surface),
* ``mixed``    : observed but with a large height range (roof + ground, trees, noise),
* ``unknown``  : never observed (or only once -> bootstrap).

**What it does.** A candidate splat is *vetoed* when its column is ``resolved`` and the
splat sits **well above the known surface** (``z > H_hi + eps``) -- i.e. it is floating
in space that the map has already seen through.  This is a discrete, 2.5D version of
free-space carving.

**Why it is needed (measured).** ``admission_mask`` in ``map/consistency.py`` uses a
3x3 neighbourhood of *already existing* points, so at a building border (1-3 m outside
the outline) the neighbourhood is empty (``coverage < 0.2``) and every floating point is
let through: 77% of the new cells there were more than 1 m off the local surface
(median 2.43 m, p90 8.23 m).  The NFP has a value in exactly those columns -- it saw the
**ground** there -- so ``z=20 > 0 + 1`` is a decidable veto.

**What it is not.** It averages nothing and it never rewrites an existing surface, so it
does not violate the "select, do not fuse" rule of the design doc (§2.7).  It only
removes candidates *before* they enter the map.

**Honest limit.** A column can only express "above the top surface is free space"; it
cannot reject points under an overhang or eave.  That is a reasonable approximation for
near-nadir aerial imagery and must be stated as a limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NFPConfig:
    cell: float = 1.0            # metres per column
    tau_mix: float = 1.5         # H_hi - H_lo above this -> "mixed" (cannot judge)
    eps: float = 1.0             # veto margin above H_hi (metres)
    min_support: int = 2         # independent keyframes needed for "resolved"
    bootstrap: int = 3           # first K keyframes only build, never veto
    smooth_px: int = 5           # median filter on the per-column top, kills noise spikes
    lo_pct: float = 10.0
    hi_pct: float = 90.0
    z_lo: float = -30.0          # ignore absurd heights when building
    z_hi: float = 80.0


@dataclass
class NadirHeightField:
    cfg: NFPConfig = field(default_factory=NFPConfig)
    T_WP: np.ndarray = field(default_factory=lambda: np.eye(4))
    # per-cell accumulators
    _acc: dict = field(default_factory=dict)     # key -> list of z samples (subsampled)
    _frames: dict = field(default_factory=dict)  # key -> last kf id that saw it
    _nframes: dict = field(default_factory=dict)  # key -> number of distinct keyframes
    _cache: dict = field(default_factory=dict)   # key -> (lo, hi, n, state)
    _dense_cache: tuple = None                   # (H_hi, n, state, origin)
    n_built: int = 0
    n_vetoed: int = 0
    n_frames: int = 0

    # ------------------------------------------------------------------
    def _keys(self, means):
        P = (np.asarray(means, np.float64) - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
        from stream_ortho.map.selection import pack_cell

        ij = np.floor(P[:, :2] / float(self.cfg.cell)).astype(np.int64)
        return pack_cell(ij[:, 0], ij[:, 1]), P[:, 2]

    def update(self, means, kf_id: int, max_per_cell: int = 64) -> None:
        """Accumulate one keyframe: per-cell (lo, hi) of **this frame** only.

        D (week8-end-38): the old version concatenated a numpy array per cell (~500k
        cells per keyframe), which cost 1-2 s/frame and dominated the streaming loop.
        Here each frame is grouped once with a lexsort, and the running per-column
        statistics are updated with plain dict operations (~100 ms).
        """
        if len(means) == 0:
            return
        key, z = self._keys(means)
        ok = (z > self.cfg.z_lo) & (z < self.cfg.z_hi)
        key, z = key[ok], z[ok]
        if len(key) == 0:
            return
        order = np.lexsort((z, key))
        ks, zs = key[order], z[order]
        uniq, start, counts = np.unique(ks, return_index=True, return_counts=True)
        lo_i = start + np.floor(counts * self.cfg.lo_pct / 100.0).astype(np.int64)
        hi_i = start + np.minimum(np.floor(counts * self.cfg.hi_pct / 100.0).astype(np.int64),
                                 counts - 1)
        lo_f, hi_f = zs[lo_i], zs[hi_i]
        frames = self._frames
        kf = int(kf_id)
        for u, a, bb in zip(uniq.tolist(), lo_f.tolist(), hi_f.tolist()):
            e = self._acc.get(u)
            if e is None:
                self._acc[u] = [a, bb]
                frames[u] = kf
                continue
            # widen the observed height range; count a column once per keyframe
            if a < e[0]:
                e[0] = a
            if bb > e[1]:
                e[1] = bb
            f = frames.get(u)
            if f != kf:
                frames[u] = kf
                self._nframes[u] = self._nframes.get(u, 1) + 1
        self.n_built += int(len(uniq))
        self.n_frames += 1
        self._dense_cache = None

    def _stats(self, u: int):
        c = self._cache.get(u)
        if c is not None:
            return c
        e = self._acc.get(u)
        if e is None:
            out = (np.nan, np.nan, 0, 0)
        else:
            lo, hi = float(e[0]), float(e[1])
            n = int(self._nframes.get(u, 1))
            if n < self.cfg.min_support:
                state = 0
            elif hi - lo <= self.cfg.tau_mix:
                state = 1
            else:
                state = 2
            out = (lo, hi, n, state)
        self._cache[u] = out
        return out

    # ------------------------------------------------------------------
    def _dense(self):
        """Dense (H_hi, n, state) grid, rebuilt lazily after each ``update``.

        Vectorised: one lexsort over all (cell, z) samples gives every cell's height
        quantiles without a per-cell Python loop (a per-cell loop over 44k cells cost
        2.3 s, which is unacceptable in a streaming loop).
        """
        if self._dense_cache is not None:
            return self._dense_cache
        if not self._acc:
            self._dense_cache = (np.zeros((0, 0), np.float32), np.zeros((0, 0), np.int32),
                                 np.zeros((0, 0), np.int8), np.zeros(2, np.int64))
            return self._dense_cache
        from stream_ortho.map.selection import unpack_cell

        keys = np.fromiter(self._acc.keys(), np.int64, len(self._acc))
        pairs = np.array([self._acc[int(k)] for k in keys], np.float64)
        lo_v, hi_v = pairs[:, 0], pairs[:, 1]
        uniq = keys
        nfr = np.fromiter((self._nframes.get(int(k), 1) for k in keys), np.int32, len(keys))
        state = np.zeros(len(uniq), np.int8)
        resolved = (nfr >= self.cfg.min_support) & ((hi_v - lo_v) <= self.cfg.tau_mix)
        mixed = (nfr >= self.cfg.min_support) & ~resolved
        state[resolved] = 1
        state[mixed] = 2
        ix, iy = unpack_cell(uniq)
        x0, y0 = int(ix.min()), int(iy.min())
        nx, ny = int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1
        Hraw = np.full((ny, nx), np.nan, np.float32)
        Hraw[iy - y0, ix - x0] = hi_v.astype(np.float32)
        # **smooth the top**: a single noisy point would otherwise raise H_hi by metres
        # and make its own column un-vetoable -- the prior would inherit exactly the
        # contamination it is supposed to remove.  A real roof is a large connected
        # region, so a median filter keeps it while removing isolated spikes.
        if self.cfg.smooth_px and self.cfg.smooth_px > 1 and np.isfinite(Hraw).any():
            from scipy.ndimage import median_filter
            base = np.where(np.isfinite(Hraw), Hraw, float(np.nanmedian(Hraw)))
            Hs = median_filter(base, size=int(self.cfg.smooth_px)).astype(np.float32)
            H = np.where(np.isfinite(Hraw), Hs, np.nan).astype(np.float32)
        else:
            H = Hraw
        N = np.zeros((ny, nx), np.int32)
        S = np.zeros((ny, nx), np.int8)
        N[iy - y0, ix - x0] = nfr
        S[iy - y0, ix - x0] = state
        self._dense_cache = (H, N, S, np.array([x0, y0], np.int64))
        return self._dense_cache

    def veto(self, means, building: bool = True):
        """Vectorised: drop candidates floating above a ``resolved`` column.

        Returns ``(keep_mask, info)``.  Nothing is averaged or rewritten, so this is an
        *admission* filter, not a fusion step.
        """
        means = np.asarray(means, np.float64)
        if len(means) == 0:
            return np.ones(0, bool), {"vetoed": 0, "judged": 0, "resolved": 0}
        if self.n_frames < self.cfg.bootstrap:
            return np.ones(len(means), bool), {"vetoed": 0, "judged": 0,
                                               "resolved": 0, "bootstrap": True}
        hi, nn, st, org = self._dense()
        if hi.size == 0:
            return np.ones(len(means), bool), {"vetoed": 0, "judged": 0, "resolved": 0}
        P = (means - self.T_WP[:3, 3]) @ self.T_WP[:3, :3]
        ix = np.floor(P[:, 0] / float(self.cfg.cell)).astype(np.int64) - org[0]
        iy = np.floor(P[:, 1] / float(self.cfg.cell)).astype(np.int64) - org[1]
        inside = (ix >= 0) & (ix < hi.shape[1]) & (iy >= 0) & (iy < hi.shape[0])
        keep = np.ones(len(means), bool)
        if inside.any():
            h = hi[iy[inside], ix[inside]]
            s_ = st[iy[inside], ix[inside]]
            z = P[inside, 2]
            judged = np.isfinite(h) & ((s_ == 1) | (s_ == 2))
            # resolved column: anything above the known top is floating.
            # **mixed** column (roof+ground+noise, H_hi-H_lo > tau_mix): the observed
            # range itself is uncertain, so use the wider margin ``tau_mix + eps``
            # above the *ground* level -- otherwise the noisiest border columns, which
            # are exactly where the floating points are, would escape judgement.
            bad = (np.isfinite(h) & (s_ == 1) & (z > h + self.cfg.eps)) | \
                  (np.isfinite(h) & (s_ == 2) &
                   (z > h + self.cfg.tau_mix + self.cfg.eps))
            idx = np.where(inside)[0]
            keep[idx[bad]] = False
            n_judged = int(judged.sum())
        else:
            n_judged = 0
        nv = int((~keep).sum())
        self.n_vetoed += nv
        return keep, {"vetoed": nv, "judged": n_judged, "resolved": n_judged}

    def coverage(self) -> dict:
        n = len(self._acc)
        if n == 0:
            return {"cells": 0}
        st = [self._stats(u)[3] for u in self._acc]
        st = np.asarray(st)
        return {"cells": int(n), "resolved": float((st == 1).mean()),
                "mixed": float((st == 2).mean()), "unknown": float((st == 0).mean())}
