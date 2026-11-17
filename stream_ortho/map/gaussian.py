"""Gaussian patch / map data structures (design doc §11.8, week-4 deliverable).

Storage is **numpy on the CPU** (float32) with a voxel hash for dedup; the map is
converted to a :class:`stream_ortho.ortho.rasterizer.GaussianCloud` only when
rendering.  Rationale:

* the map must survive thousands of commits (per-keyframe) without torch
  allocator churn;
* voxel hashing / merging is much simpler with numpy and a dict;
* the GPU only needs the *current* tile's Gaussians (week 6 dirty tiles).

Fields follow §11.8 but note the extra bookkeeping the gate needs:
``conf``, ``n_obs``, ``born_kf``, ``last_seen``, ``owner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _f32(x, shape_hint=None):
    a = np.asarray(x, dtype=np.float32)
    if shape_hint is not None and a.shape != shape_hint:
        a = a.reshape(shape_hint)
    return np.ascontiguousarray(a)


@dataclass
class GaussianPatch:
    """One keyframe's Gaussians in the **world** frame (metric or map units)."""

    means: np.ndarray            # (N,3)
    quats: np.ndarray            # (N,4) wxyz
    scales: np.ndarray           # (N,3) > 0
    opacity: np.ndarray          # (N,) or (N,1)
    colors: np.ndarray           # (N,3)
    conf: Optional[np.ndarray] = None      # (N,) or None
    kind: Optional[np.ndarray] = None      # (N,) 0=area, 1=edge (adaptive retention)
    weight: Optional[np.ndarray] = None    # (N,) observation quality (view weighting)
    theta: Optional[np.ndarray] = None     # (N,) edge orientation in the plane [rad]
    owner: int = -1
    kf_id: int = -1
    T_WC: Optional[np.ndarray] = None      # 8-vector Sim3 (owner-anchor, week 7)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.means = _f32(self.means, (-1, 3))
        n = self.means.shape[0]
        self.quats = _f32(self.quats, (n, 4))
        self.scales = _f32(self.scales, (n, 3))
        self.opacity = _f32(self.opacity).reshape(n)
        self.colors = _f32(self.colors, (n, 3))
        self.conf = (np.ones(n, np.float32) if self.conf is None
                     else _f32(self.conf).reshape(n))
        self.kind = (np.zeros(n, np.int8) if self.kind is None
                     else np.asarray(self.kind, np.int8).reshape(n))
        self.theta = (np.full(n, np.nan, np.float32) if self.theta is None
                      else np.asarray(self.theta, np.float32).reshape(n))
        self.weight = (np.ones(n, np.float32) if self.weight is None
                       else _f32(self.weight).reshape(n))
        if self.owner < 0:
            self.owner = self.kf_id

    def __len__(self):
        return self.means.shape[0]

    def subset(self, mask: np.ndarray) -> "GaussianPatch":
        m = np.asarray(mask)
        if m.dtype != bool:
            sel = np.zeros(len(self), bool)
            sel[m.reshape(-1).astype(np.int64)] = True
            m = sel
        mask = m.reshape(-1)
        return GaussianPatch(
            means=self.means[mask], quats=self.quats[mask], scales=self.scales[mask],
            opacity=self.opacity[mask], colors=self.colors[mask], conf=self.conf[mask],
            kind=self.kind[mask], theta=self.theta[mask], weight=self.weight[mask],
            owner=self.owner, kf_id=self.kf_id,
            T_WC=None if self.T_WC is None else np.asarray(self.T_WC),
            meta=dict(self.meta),
        )

    def stats(self) -> dict:
        s = self.scales
        return {
            "n": len(self),
            "scale_median": float(np.median(s)),
            "opacity_median": float(np.median(self.opacity)),
            "conf_median": float(np.median(self.conf)),
            "edge_ratio": float(self.kind.mean()) if len(self) else 0.0,
            "points": self.means.mean(axis=0).round(2).tolist() if len(self) else None,
        }


def empty_map_arrays() -> dict:
    return {
        "means": np.zeros((0, 3), np.float32),
        "quats": np.zeros((0, 4), np.float32),
        "scales": np.zeros((0, 3), np.float32),
        "opacity": np.zeros((0,), np.float32),
        "colors": np.zeros((0, 3), np.float32),
        "conf": np.zeros((0,), np.float32),
        "owner": np.zeros((0,), np.int32),
        "n_obs": np.zeros((0,), np.int16),
        "born_kf": np.zeros((0,), np.int32),
        "last_seen": np.zeros((0,), np.int32),
        "kind": np.zeros((0,), np.int8),
        "theta": np.zeros((0,), np.float32),
        "wsum": np.zeros((0,), np.float32),
        "dead": np.zeros((0,), np.int8),
    }


def concat_arrays(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        out[k] = np.concatenate([a[k], b[k]], axis=0) if len(b[k]) else a[k]
    return out


def take_rows(a: dict, idx) -> dict:
    return {k: v[idx] for k, v in a.items()}


def array_bytes(a: dict) -> int:
    return int(sum(v.nbytes for v in a.values()))


def voxel_keys(means: np.ndarray, voxel: float) -> np.ndarray:
    """Integer voxel coordinates (N,3) at ``voxel`` metres."""
    return np.floor(np.asarray(means, np.float64) / float(voxel)).astype(np.int64)


def key_hashes(keys: np.ndarray) -> np.ndarray:
    """Stable 64-bit hash of integer voxel coordinates."""
    k = np.asarray(keys, np.int64)
    h = (k[:, 0] * np.int64(73856093)) ^ (k[:, 1] * np.int64(19349663)) ^ (k[:, 2] * np.int64(83492791))
    return h.astype(np.int64)
