"""PIVOT-B calibration core: per-frame 3-DOF fit, application, and a **frozen** product.

The method, in one sentence: every keyframe carries its own low-order deformation
``z' = z - (b + tx (x-x0) + ty (y-y0))``; fitting it to the product removes the frame-to-frame
inconsistency, but **rebuilding the layers from the corrected observations destroys the
ownership structure** (the number of owner-boundary pairs grows ~+46%, LCR drops ~2 pp).
Freezing ``(cell, layer-slot, owner)`` and updating only the heights keeps the consistency
gain and removes that regression.

This module is the only place that implements the three primitives:

``fit_t3``          robust (Huber IRLS) 3-DOF fit of one frame against the product's near layer
``apply_t3``        subtract the fitted field from a frame's heights
``FrozenProduct``   the frozen assignment: rebuild heights over it, never re-group

Units are metres; the plane frame and the global origin ``(x0, y0)`` are the caller's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibConfig:
    """Fitting parameters (frozen by the PIVOT-B card; do not tune per sequence)."""

    huber_iters: int = 4
    huber_k: float = 1.345
    min_obs: int = 50
    #: Optional Tikhonov term.  Off by default: a per-frame 3-DOF fit is well posed, and even
    #: a 1e-9 relative ridge measurably biases the intercept on exact data (the augmented rows
    #: widen the spectrum).  Enable only for a degenerate support, and check the residual.
    ridge: float = 0.0


def _solve_ls(A, rhs, ridge=0.0):
    """Weighted least squares by **QR/SVD** (``lstsq``), optionally with a ridge term.

    Do *not* use the normal equations here: the design is ``[1, x, y]`` with plane
    coordinates of order 300 m, so ``A^T A`` squares the condition number.  In float64 that
    costs ~1.6e-5 m on the intercept once a ridge is added (measured on synthetic data at
    ±300 m), which is the same order as the metric differences this project cares about.
    ``lstsq`` keeps full precision and, with three columns, is just as fast.
    """
    if ridge > 0:
        scale = float(np.sqrt(np.mean(A * A))) or 1.0
        m = A.shape[1]
        A = np.vstack([A, np.sqrt(ridge) * scale * np.eye(m)])
        rhs = np.concatenate([rhs, np.zeros(m)])
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


def fit_t3(z, z_near, x, y, w=None, cfg: CalibConfig | None = None,
           origin=(0.0, 0.0)) -> np.ndarray | None:
    """Robust fit of ``z - z_near = b + tx (x-x0) + ty (y-y0)`` for one frame.

    Parameters
    ----------
    z        : (N,) observed heights
    z_near   : (N,) height of the nearest occupied product layer (NaN where none)
    x, y     : (N,) plane coordinates
    w        : (N,) weights (``conf * |cos|``), optional
    cfg      : :class:`CalibConfig`
    origin   : global ``(x0, y0)`` -- one origin for the whole experiment

    Returns the 3-vector ``(b, tx, ty)`` or ``None`` when there are too few usable rows.
    Nothing is hard-dropped: outliers are down-weighted by Huber.
    """
    cfg = cfg or CalibConfig()
    z = np.asarray(z, np.float64)
    z_near = np.asarray(z_near, np.float64)
    ok = np.isfinite(z_near) & np.isfinite(z)
    if ok.sum() < cfg.min_obs:
        return None
    d = z[ok] - z_near[ok]
    u = np.asarray(x, np.float64)[ok] - origin[0]
    v = np.asarray(y, np.float64)[ok] - origin[1]
    ww = (np.ones(ok.sum()) if w is None
          else np.clip(np.nan_to_num(np.asarray(w, np.float64)[ok], nan=1.0), 1e-3, None))
    A = np.stack([np.ones_like(u), u, v], 1)
    # start from the plain weighted LS: starting the Huber loop from p = 0 would let the
    # first scale estimate be dominated by the deformation itself and over-down-weight
    p = _solve_ls(A * ww[:, None], d * ww, cfg.ridge)
    for _ in range(max(1, cfg.huber_iters)):
        r = d - A @ p
        # floor the scale at 1 cm: on (near-)exact data the MAD collapses to ~0 and the Huber
        # weights would then shrink *every* row, biasing an otherwise exact fit
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-2)
        h = np.minimum(1.0, cfg.huber_k * s / np.maximum(np.abs(r), 1e-9))
        wh = ww * h
        p = _solve_ls(A * wh[:, None], d * wh, cfg.ridge)
    return p


def apply_t3(z, x, y, theta, origin=(0.0, 0.0)) -> np.ndarray:
    """Return ``z - (b + tx (x-x0) + ty (y-y0))``; the input array is not modified."""
    theta = np.asarray(theta, np.float64).reshape(3)
    return np.asarray(z, np.float64) - (theta[0] +
                                        theta[1] * (np.asarray(x, np.float64) - origin[0]) +
                                        theta[2] * (np.asarray(y, np.float64) - origin[1]))


class FrozenProduct:
    """The raw product's layer/owner assignment, frozen so heights can be re-estimated.

    ``from_product`` keeps the per-observation ``assign`` (which ``(cell, slot)`` each row was
    merged into), the owner grid, the support and the geometry.  ``rebuild`` then recomputes
    ``z_ref`` (and the weighted sums) from new heights **without re-grouping or re-electing
    owners** -- that is the PIVOT-B fix for the owner fragmentation.
    """

    def __init__(self, assign, owner, n, shape, cell, ox, oy, L):
        self.assign = assign
        self.owner = owner
        self.n = n
        self.H, self.W = shape
        self.cell = cell
        self.ox, self.oy = ox, oy
        self.L = L
        self.n_cells = self.H * self.W

    @classmethod
    def from_product(cls, prod) -> "FrozenProduct":
        assign = prod.get("assign")
        if assign is None:
            raise ValueError("product has no per-row assignment; build it with "
                             "build_product(..., return_assign=True)")
        return cls(np.asarray(assign, np.int64), np.asarray(prod["owner"]).copy(),
                   np.asarray(prod["n"]).copy(), (prod["H"], prod["W"]), prod["cell"],
                   prod["ox"], prod["oy"], prod["L"])

    def rebuild(self, z_rows, w_rows, n_rows=None) -> dict:
        """Weighted-mean heights over the frozen assignment; owner/support unchanged."""
        z = np.asarray(z_rows, np.float64)
        w = np.asarray(w_rows, np.float64)
        size = self.n_cells * self.L
        ok = self.assign >= 0
        a = self.assign[ok]
        zz, ww = z[ok], w[ok]
        num = np.bincount(a, weights=ww * zz, minlength=size)
        den = np.bincount(a, weights=ww, minlength=size)
        swz2 = np.bincount(a, weights=ww * zz * zz, minlength=size)
        z_ref = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        return {
            "n": self.n, "owner": self.owner,
            "z_ref": z_ref.reshape(self.H, self.W, self.L),
            "sum_w": den.reshape(self.H, self.W, self.L),
            "sum_wz": num.reshape(self.H, self.W, self.L),
            "sum_wz2": swz2.reshape(self.H, self.W, self.L),
            "cell": self.cell, "W": self.W, "H": self.H, "L": self.L,
            "ox": self.ox, "oy": self.oy,
            "x0": self.ox * self.cell, "y0": self.oy * self.cell,
        }

    def memory_mb(self, dtype=np.int64) -> float:
        return float(self.assign.size * np.dtype(dtype).itemsize) / 1e6
