"""P0-1: observation store (keep 1..L observations per cell instead of deleting them).

The streaming loop used to collapse every keyframe to *one point per cell* immediately,
which throws away the redundancy the paper needs (measured: 1.42 obs/cell before
collapsing, 1.00 after).  ``ObservationStore`` keeps the last ``window`` frames **whole**
and hands older frames to a sink (``on_evict``) that merges them into the column store
(P0-2, ``map/columns.py``).

Contract (the point of the module):

* ``append_frame`` is **O(patch)**: it stores one array bundle per keyframe and never
  concatenates across frames (asserted by a test that monkeypatches ``np.concatenate``);
* the ring keeps exactly the last ``window`` frames; each evicted frame is reported once
  (via ``on_evict`` **and** through ``pop_oldest`` so a caller can drain it either way);
* layers are preserved: two frames seeing the same cell at z=1 and z=9 both survive.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameObs:
    kf: int
    cell_key: np.ndarray      # (N,) int64, pack_cell(floor(x/cell), floor(y/cell))
    z: np.ndarray             # (N,) float32, plane-frame height
    rgb: np.ndarray           # (N,3) uint8
    conf: np.ndarray          # (N,) float16
    cos_inc: np.ndarray       # (N,) float16, |cos(incidence)|
    owner: np.ndarray = None  # (N,) int32, = kf

    @staticmethod
    def from_patch(patch, kf: int, T_WP, cell: float, copy: bool = True) -> "FrameObs":
        """Build one frame's observations from a ``GaussianPatch`` (P0-1b).

        ``copy=True`` makes the store **own** the arrays: the pipeline reuses buffers
        (``head_adapt_patch`` returns fresh ones today, but nothing guarantees it), and a
        silent aliasing bug would poison the history.

        ``cos_inc`` is filled with **1.0** at this stage: the patch carries no reliable
        camera ray, so an incidence angle cannot be computed yet; P0-3 computes it once the
        poses and the per-cell normal are available.  It is therefore *not* a quality
        signal in P0-1/P0-2 — the weighted mean must not treat it as one.
        """
        from stream_ortho.map.selection import pack_cell

        T_WP = np.asarray(T_WP, np.float64)
        P = (np.asarray(patch.means, np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
        ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
        key = pack_cell(ij[:, 0], ij[:, 1])
        rgb = (np.asarray(patch.colors, np.float32) * 255.0).astype(np.uint8)
        conf = np.asarray(patch.conf, np.float32).astype(np.float16)
        z = P[:, 2].astype(np.float32)
        cos = np.ones(len(P), np.float16)
        if copy:
            key, z, rgb, conf, cos = (np.array(a, copy=True) for a in (key, z, rgb, conf, cos))
        return FrameObs(kf=int(kf), cell_key=key, z=z, rgb=rgb, conf=conf, cos_inc=cos)

    def __post_init__(self):
        n = len(self.cell_key)
        for name in ("z", "rgb", "conf", "cos_inc"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"FrameObs.{name}: {len(getattr(self, name))} != {n}")
        if self.owner is None:
            self.owner = np.full(n, int(self.kf), np.int32)


class ObservationStore:
    def __init__(self, cell: float, T_WP, window: int = 15, on_evict=None):
        self.cell = float(cell)
        self.T_WP = np.asarray(T_WP, np.float64)
        self.window = int(window)
        self.on_evict = on_evict
        self._ring: deque[FrameObs] = deque()
        self._pending: deque[FrameObs] = deque()   # evicted, not yet drained
        self.n_appended = 0
        self._evicted_kfs: list[int] = []

    # ------------------------------------------------------------------
    def append_frame(self, kf=None, cell_key=None, z=None, rgb=None, conf=None,
                     cos_inc=None, owner=None, frame: FrameObs | None = None) -> FrameObs:
        """Store one keyframe.  O(patch): a single append, no cross-frame copy."""
        if frame is not None:
            f = frame
        else:
            # B3: the store takes ownership (copies) so a caller reusing its buffers
            # cannot silently rewrite history.
            f = FrameObs(
                kf=int(kf), cell_key=np.array(cell_key, np.int64, copy=True),
                z=np.array(z, np.float32, copy=True), rgb=np.array(rgb, np.uint8, copy=True),
                conf=np.array(conf, np.float16, copy=True),
                cos_inc=np.array(cos_inc, np.float16, copy=True),
                owner=None if owner is None else np.array(owner, np.int32, copy=True))
        self._ring.append(f)
        self.n_appended += 1
        while len(self._ring) > self.window:
            old = self._ring.popleft()
            self._evicted_kfs.append(old.kf)
            # B2: exactly one delivery path -- a sink, or the pending queue (never both,
            # which leaked ~46 MB/frame whenever a sink was installed).
            if self.on_evict is not None:
                self.on_evict(old)
            else:
                self._pending.append(old)
                # leak guard: with no sink the caller **must** drain, otherwise the
                # evicted frames accumulate (measured: 40 frames = 1.90 GB on top of the
                # 714 MB ring, which is what the 3.3x RSS report actually was).
                if len(self._pending) > self.window:
                    warnings.warn(
                        f"ObservationStore: {len(self._pending)} evicted frames are "
                        f"queued (window={self.window}) -- call pop_oldest() or pass "
                        f"on_evict, otherwise memory grows linearly.",
                        RuntimeWarning, stacklevel=2)
        return f

    def pop_oldest(self) -> FrameObs | None:
        """Return an evicted frame (oldest first); None if nothing was evicted."""
        # B1: only drain the pending queue -- popping from the ring would silently
        # shrink the retained window, which is the whole point of the store.
        if self._pending:
            return self._pending.popleft()
        return None

    # ------------------------------------------------------------------
    def query(self, keys) -> list[FrameObs]:
        """Every stored observation whose cell is in ``keys`` (layers included)."""
        keys = np.asarray(keys, np.int64)
        out = []
        for f in self._ring:
            m = np.isin(f.cell_key, keys)
            if not m.any():
                continue
            out.append(FrameObs(kf=f.kf, cell_key=f.cell_key[m], z=f.z[m], rgb=f.rgb[m],
                                conf=f.conf[m], cos_inc=f.cos_inc[m], owner=f.owner[m]))
        return out

    def stats(self) -> dict:
        n_obs = int(sum(len(f.cell_key) for f in self._ring))
        keys = np.concatenate([f.cell_key for f in self._ring]) if self._ring else \
            np.zeros(0, np.int64)
        n_cells = int(len(np.unique(keys))) if keys.size else 0
        return {
            "window": self.window,
            "n_frames": len(self._ring),
            "n_appended": self.n_appended,
            "n_evicted_frames": len(self._evicted_kfs),
            "n_pending": len(self._pending),
            "evicted_kfs": list(self._evicted_kfs),
            "n_obs": n_obs,
            "n_cells": n_cells,
            "obs_per_cell": float(n_obs / max(n_cells, 1)),
        }


def frames_per_cell(per_frame_keys, chunk: int = 4_000_000):
    """Distinct **frames** per cell (not points per cell).

    ``per_frame_keys`` is a sequence of per-frame ``cell_key`` arrays.  A frame that puts
    three points into the same cell counts **once**, which is the quantity the paper needs
    ("how many keyframes see this product cell"), as opposed to ``points/cell`` which also
    grows with the sampling density.  Implemented as a two-pass unique + searchsorted so
    memory stays O(#cells) instead of O(#observations).
    """
    allu = np.zeros(0, np.int64)
    for k in per_frame_keys:
        allu = np.union1d(allu, np.unique(np.asarray(k, np.int64)))
    counts = np.zeros(len(allu), np.int64)
    for k in per_frame_keys:
        ku = np.unique(np.asarray(k, np.int64))
        if ku.size == 0:
            continue
        pos = np.searchsorted(allu, ku)
        np.add.at(counts, pos, 1)
    return allu, counts
