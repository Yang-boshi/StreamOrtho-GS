"""Online 3-DOF calibration: one fit per arriving frame against the streaming product.

The offline method iterates three rounds over all frames.  Online there is no such luxury:
each frame is fitted **once**, against whatever the product already holds, and the fit is
skipped (bootstrap) or downgraded (fallback) when the product is too sparse or the solution
looks implausible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stream_ortho.map.calib import CalibConfig, apply_t3, fit_t3


@dataclass
class OnlineCalibConfig:
    min_frames: int = 10          # bootstrap: no fit until the product holds this many frames
    min_obs: int = 500            # below this many usable observations, keep the raw heights
    max_abs_b: float = 3.0        # a larger offset is treated as a failed fit
    max_resid: float = 1.0        # median |residual| above this is treated as a failed fit
    fallback: str = "identity"    # identity | rank1
    ridge: float = 0.0


@dataclass
class OnlineCalibStats:
    n_fit: int = 0
    n_bootstrap: int = 0
    n_fallback_low_obs: int = 0
    n_fallback_bad: int = 0
    b_history: list = field(default_factory=list)

    @property
    def fallback_frac(self) -> float:
        tot = self.n_fit + self.n_bootstrap + self.n_fallback_low_obs + self.n_fallback_bad
        return float((self.n_fallback_low_obs + self.n_fallback_bad) / tot) if tot else 0.0


class OnlineCalib:
    def __init__(self, cfg: OnlineCalibConfig | None = None, origin=(0.0, 0.0)):
        self.cfg = cfg or OnlineCalibConfig()
        self.calib = CalibConfig(ridge=self.cfg.ridge)
        self.origin = origin
        self.stats = OnlineCalibStats()

    def process(self, product, frames_done: int, kf: int, x, y, z, w):
        """Return (z_corrected, info) for one arriving frame."""
        info = {"kf": int(kf), "mode": "bootstrap", "theta": None, "resid": float("nan")}
        if frames_done < self.cfg.min_frames:
            self.stats.n_bootstrap += 1
            return z, info

        z_near = product.near(np.floor(np.asarray(x) / product.cell).astype(np.int64),
                              np.floor(np.asarray(y) / product.cell).astype(np.int64),
                              np.asarray(z, np.float64))
        ok = np.isfinite(z_near)
        if ok.sum() < self.cfg.min_obs:
            self.stats.n_fallback_low_obs += 1
            info["mode"] = "fallback_low_obs"
            return z, info

        th = fit_t3(z, z_near, x, y, w, self.calib, self.origin)
        if th is None:
            self.stats.n_fallback_low_obs += 1
            info["mode"] = "fallback_low_obs"
            return z, info
        r = (np.asarray(z, np.float64) - z_near)[ok] - (
            th[0] + th[1] * (np.asarray(x)[ok] - self.origin[0]) +
            th[2] * (np.asarray(y)[ok] - self.origin[1]))
        info["theta"] = [float(v) for v in th]
        info["resid"] = float(np.median(np.abs(r)))
        if abs(th[0]) > self.cfg.max_abs_b or info["resid"] > self.cfg.max_resid:
            self.stats.n_fallback_bad += 1
            info["mode"] = "fallback_bad_fit"
            if self.cfg.fallback == "rank1":
                b = float(np.median((np.asarray(z, np.float64) - z_near)[ok]))
                b = float(np.clip(b, -self.cfg.max_abs_b, self.cfg.max_abs_b))
                self.stats.b_history.append(b)
                return np.asarray(z, np.float64) - b, info
            return z, info
        self.stats.n_fit += 1
        self.stats.b_history.append(float(th[0]))
        info["mode"] = "fit"
        return apply_t3(z, x, y, th, self.origin), info
