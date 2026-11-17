"""Commit gate for incoming Gaussian patches (design doc §7.5 item 3).

The gate is deliberately simple, cheap and *explainable*; every rejected point
carries a reason so the week-4 report can show a histogram instead of a single
number.  Thresholds are expressed in metres (the map is metric after the
est->GT Sim3) and can be overridden per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from stream_ortho.map.gaussian import GaussianPatch


@dataclass
class GateConfig:
    conf_min: float = 0.5
    depth_min_m: float = 5.0
    depth_max_m: float = 1500.0
    opacity_min: float = 0.05
    scale_max_m: float = 5.0           # a splat larger than this is a floater/blur
    scale_min_m: float = 1e-4
    height_min_m: float = -3.0         # below the ground plane -> reject
    height_max_m: float = 120.0        # above -> sky / cable / artefact
    note: str = "S5a default"


def _height_above_plane(means: np.ndarray, T_WP: np.ndarray, z_ground: float) -> np.ndarray:
    """Signed height of world points above the ground plane (metres)."""
    P = (np.asarray(means, np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
    return P[:, 2] - float(z_ground)


def commit_gate(patch: GaussianPatch, cfg: GateConfig, T_WP: np.ndarray,
                z_ground: float = 0.0) -> Dict[str, object]:
    """Return ``valid_mask`` plus per-reason rejection counts.

    ``reasons`` counts are *independent* (a point can fail several), while
    ``n_valid`` is the size of the final mask, so the numbers are not additive
    by design -- the report shows both.
    """
    n = len(patch)
    h = _height_above_plane(patch.means, T_WP, z_ground) if n else np.zeros(0)
    depth = patch.meta.get("depth_m")
    if depth is None:
        depth = np.full(n, np.nan, np.float32)
    depth = np.asarray(depth, np.float32).reshape(-1) if np.size(depth) else np.full(n, np.nan, np.float32)
    if depth.size == 1 and n > 1:
        depth = np.full(n, float(depth), np.float32)

    reasons = {
        "conf": patch.conf < cfg.conf_min,
        "opacity": patch.opacity < cfg.opacity_min,
        "scale_max": patch.scales.max(axis=1) > cfg.scale_max_m,
        "scale_min": patch.scales.min(axis=1) < cfg.scale_min_m,
        "height_low": h < cfg.height_min_m,
        "height_high": h > cfg.height_max_m,
        "depth": np.isfinite(depth) & ((depth < cfg.depth_min_m) | (depth > cfg.depth_max_m)),
    }
    reject = np.zeros(n, bool)
    for v in reasons.values():
        reject |= v
    valid = ~reject
    return {
        "valid": valid,
        "n": int(n),
        "n_valid": int(valid.sum()),
        "n_rejected": int(reject.sum()),
        "rejected_ratio": float(reject.mean()) if n else 0.0,
        "reasons": {k: int(v.sum()) for k, v in reasons.items()},
    }


@dataclass
class GateStats:
    n_seen: int = 0
    n_passed: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def update(self, res: Dict[str, object]):
        self.n_seen += int(res["n"])
        self.n_passed += int(res["n_valid"])
        for k, v in res["reasons"].items():
            self.reasons[k] = self.reasons.get(k, 0) + int(v)

    def as_dict(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "n_passed": self.n_passed,
            "pass_ratio": self.n_passed / max(1, self.n_seen),
            "reasons": dict(self.reasons),
        }
