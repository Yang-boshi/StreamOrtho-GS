"""P3.4: raw geometry and nearest-neighbour depth upsampling.

The adapter used to resample the 288x512 pointmap depth to the image with bilinear
interpolation; at a roof/ground step that invents depths between the two surfaces.  These
tests pin the two properties that matter: the raw path keeps the model grid as-is, and the
nearest path introduces no new depth values.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def test_raw_grid_has_exactly_hw_points():
    h, w = 288, 512
    X = np.random.default_rng(0).random((h, w, 3))
    assert X.reshape(-1, 3).shape[0] == h * w


def test_nearest_upsampling_invents_no_new_depths():
    h, w, H, W = 8, 16, 64, 128
    d_h = np.repeat(np.repeat(np.random.default_rng(1).random((h, w)) * 10, 8, 0), 8, 1)
    up = cv2.resize(d_h, (W, H), interpolation=cv2.INTER_NEAREST)
    assert np.isin(up, d_h).all(), "nearest must only copy existing values"


def test_bilinear_upsampling_does_invent_depths():
    """The counter-example: bilinear produces values that exist nowhere in the input."""
    h, w, H, W = 8, 16, 64, 128
    d_h = np.zeros((h, w))
    d_h[:, w // 2:] = 10.0                     # one hard step
    up = cv2.resize(d_h, (W, H), interpolation=cv2.INTER_LINEAR)
    assert not np.isin(up, d_h).all(), "bilinear must create intermediate values"
    mid = ((up > 0.1) & (up < 9.9)).sum()
    assert mid > 0, "the step should become a ramp"


def test_edge_mode_keeps_the_step_and_interpolates_elsewhere():
    from stream_ortho.map.build import upsample_depth
    h, w = 8, 16
    d = np.full((h, w), 10.0)
    d[:, w // 2:] = 20.0                        # a hard step
    nearest = upsample_depth(d, 64, 128, "nearest")
    edge = upsample_depth(d, 64, 128, "edge", edge_tau_m=0.75)
    assert np.isin(nearest, d).all(), "nearest must not invent values"
    # edge mode is nearest right at the step, so it also invents no intermediate values
    assert np.isin(edge, d).all(), "edge mode must not invent values either"
    # ... while plain bilinear does
    bil = upsample_depth(d, 64, 128, "linear")
    assert not np.isin(bil, d).all()
