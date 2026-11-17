"""Tests for the texture-preserving orthophoto renderer.

A synthetic world is built with an *exact* camera model so that the renderer's geometry can be
checked without any real data:

* a flat plane painted with a checkerboard -> the backward warp must reproduce the pattern and
  the edge width must match the input sampling (GSD check);
* a point hidden behind a nearer surface must be rejected by the depth test (occlusion);
* a pixel no frame sees must be a hole, never a black fill.
"""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.ortho.texture import FrameIO, edge_width_10_90, render_texture_tile


def _make_world(h=120, w=160, origin=(-20.0, -15.0), gsd=0.25, k=0, plane_z=0.0,
                checker=(4, 4)):
    """A padding-free synthetic world: identity Sim3, camera looking straight down."""
    # camera 40 m above the plane, looking down -z, image plane aligned with x/y
    R = np.eye(3)
    t = np.array([0.0, 0.0, 40.0])
    s = 1.0
    K = np.array([[100.0, 0.0, w / 2.0], [0.0, 100.0, h / 2.0], [0.0, 0.0, 1.0]])
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    chk = ((xx // checker[1]) + (yy // checker[0])) % 2
    img[chk == 0] = [0.9, 0.9, 0.9]
    img[chk == 1] = [0.1, 0.1, 0.1]
    return R, t, s, K, img


class FakeIO:
    """Minimal stand-in for FrameIO with an analytic camera and a depth buffer."""

    def __init__(self, R, t, s, K, img, depth):
        self.R = {0: R}
        self.t = {0: t}
        self.s = {0: s}
        self.K = K
        self.img = img
        self.depth = {0: depth}
        self.kfs = [0]
        self.img_scale = 1.0
        self.centre = {0: np.array([0.0, 0.0, 40.0])}

    def image(self, k):
        return self.img

    def project(self, k, P):
        X = (P - self.t[k]) @ self.R[k]
        d = np.linalg.norm(X, axis=1)
        Xs = X * self.s[k]
        u = self.K[0, 0] * Xs[:, 0] / Xs[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * Xs[:, 1] / Xs[:, 2] + self.K[1, 2]
        return u, v, d

    def depth_at(self, k, u, v):
        H, W = self.depth[k].shape
        px = np.clip(u.astype(np.int64), 0, W - 1)
        py = np.clip(v.astype(np.int64), 0, H - 1)
        return self.depth[k][py, px]


def _flat_product(x0=-20.0, y0=-15.0, cell=0.5, H=40, W=40, z=0.0):
    n = np.ones((H, W, 3), np.float32)
    z_ref = np.zeros((H, W, 3), np.float32)
    z_ref[..., 0] = z
    owner = np.zeros((H, W, 3), np.int32)
    return {"cell": cell, "ox": int(np.floor(x0 / cell)), "oy": int(np.floor(y0 / cell)),
            "H": H, "W": W, "L": 3, "n": n, "z_ref": z_ref, "owner": owner}


def test_texture_tile_reproduces_checkerboard():
    R, t, s, K, img = _make_world(checker=(16, 16))
    # the depth buffer agrees with the plane everywhere
    depth = np.full(img.shape[:2], 40.0, np.float32)
    fio = FakeIO(R, t, s, K, img, depth)
    prod = _flat_product()
    grid = {"x0": -20.0, "y0": -15.0, "h": 40, "w": 40}
    # tau must cover the perspective distance variation across this synthetic plane (~5 m)
    rgb, conf, hole, info = render_texture_tile(prod, fio, grid, 0, 0, gsd=1.0, tau=10.0)

    assert hole.mean() < 0.05
    assert info["frac_hole"] < 0.05
    # the rendered pattern must have the two checker levels (not a flat grey)
    vals = rgb.reshape(-1, 3).mean(-1)
    assert vals.max() - vals.min() > 0.5
    # and the colours must be bimodal around the input palette (bilinear edge pixels are
    # expected in between)
    vis = vals[~hole.ravel()]
    assert np.mean(vis > 0.7) > 0.2 and np.mean(vis < 0.3) > 0.2


def test_occlusion_rejects_the_hidden_surface():
    """A point 10 m *below* the depth buffer must fail the tau test and stay a hole."""
    R, t, s, K, img = _make_world()
    depth = np.full(img.shape[:2], 40.0, np.float32)      # buffer says the surface is at 40 m
    fio = FakeIO(R, t, s, K, img, depth)
    prod = _flat_product(z=-10.0)                          # but the product is 10 m lower
    grid = {"x0": -20.0, "y0": -15.0, "h": 12, "w": 12}
    _, _, hole, info = render_texture_tile(prod, fio, grid, 0, 0, gsd=1.0, tau=0.5)
    assert info["frac_hole"] > 0.95


def test_no_visible_frame_is_a_hole_not_black():
    R, t, s, K, img = _make_world()
    depth = np.full(img.shape[:2], 40.0, np.float32)
    fio = FakeIO(R, t, s, K, img, depth)
    prod = _flat_product()
    # a grid entirely outside the camera frustum
    grid = {"x0": 500.0, "y0": 500.0, "h": 8, "w": 8}
    rgb, _, hole, info = render_texture_tile(prod, fio, grid, 0, 0, gsd=1.0, tau=0.5)
    assert info["frac_hole"] == 1.0
    assert np.all(rgb == 0)                                # zeros are "no data", not black paint


def test_edge_width_recovers_the_input_sampling():
    """A 1-px step measured on a checkerboard must come back as ~1-2 px."""
    img = np.zeros((64, 64, 3), np.float32)
    img[:, 32:] = 1.0
    w, n = edge_width_10_90(img, min_grad_pct=99.0, max_px=4)
    assert n > 0
    assert 0.5 <= w <= 2.0
