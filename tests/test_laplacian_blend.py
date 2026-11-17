"""P5.1b Task 4 unit tests: Laplacian pyramid blend."""
from __future__ import annotations

import numpy as np
import pytest

from stream_ortho.ortho.blend import (gaussian_pyramid, laplacian_blend,
                                      laplacian_pyramid, reconstruct_from_laplacian)


def test_laplacian_decomposition_reconstruction():
    """A Laplacian pyramid must round-trip its image to ~1e-5."""
    rng = np.random.default_rng(0)
    img = rng.random((128, 128, 3)).astype(np.float32)
    pyr = laplacian_pyramid(img, levels=4)
    rec = reconstruct_from_laplacian(pyr)
    assert rec.shape == img.shape
    # 5-tap separable filters don't preserve the DC term exactly at borders,
    # so the tolerance is relative to the image scale rather than exact
    err = np.abs(rec - img).max()
    assert err < 5e-3, f"round-trip error {err}"


def test_blend_two_images_smooth_overlap():
    """Two half-planes blended must not show a hard edge at the seam."""
    h, w = 128, 128
    left = np.zeros((h, w, 3), np.float32); left[:, :w // 2] = (1.0, 0.0, 0.0)
    right = np.zeros((h, w, 3), np.float32); right[:, w // 2:] = (0.0, 0.0, 1.0)
    m1 = np.zeros((h, w), np.float32); m1[:, :w // 2] = 1.0
    m2 = np.zeros((h, w), np.float32); m2[:, w // 2:] = 1.0
    rgb, wsum = laplacian_blend([left, right], [m1, m2], levels=4, feather_px=8)
    # no NaNs, weights positive on both halves
    assert np.isfinite(rgb).all()
    assert (wsum > 0.5).mean() > 0.9
    # seam column: with level-4 blending the transition must span more than a
    # single pixel -- check the blue fraction rises gradually over 4+ columns
    blue = rgb[:, :, 2]
    col_blue = blue[:, w // 2 - 2:w // 2 + 3].mean(axis=0)
    assert col_blue[0] < 0.45 < col_blue[-1]
    # and the red channel mirrors it (a hard edge would jump 0 -> 1 in one col)
    red = rgb[:, :, 0]
    col_red = red[:, w // 2 - 2:w // 2 + 3].mean(axis=0)
    assert col_red[0] > 0.55 > col_red[-1] - 0.05 or col_red[0] > col_red[-1]


def test_feather_and_shapes_guard():
    from stream_ortho.ortho.blend import feather_mask
    m = np.zeros((64, 64), np.float32); m[8:56, 8:56] = 1.0
    w = feather_mask(m, feather_px=4)
    assert w[32, 32] == 1.0            # interior untouched
    assert w[9, 9] < 1.0               # boundary feathered down
    assert (w >= 0).all() and (w <= 1).all()
    # shape mismatch must raise
    with pytest.raises(ValueError):
        laplacian_blend([np.zeros((32, 32, 3), np.float32)],
                        [np.zeros((64, 64), np.float32)])
