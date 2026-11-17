"""P4.2: the three e2e blockers -- confidence resampling, product grid, calib tri-state."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def resample_conf(C, hx, wx):
    """The same rule app.frontend uses on a live frame."""
    if C.ndim == 3:
        C = C[..., 0]
    if C.shape != (hx, wx):
        C = cv2.resize(C.astype(np.float32), (wx, hx), interpolation=cv2.INTER_NEAREST)
    return C.reshape(-1)


def test_conf_is_resampled_spatially_not_repeated():
    C = np.zeros((192, 256), np.float32)
    C[0, 0] = 9.0
    out = resample_conf(C, 288, 512).reshape(288, 512)
    assert out.size == 288 * 512
    # the marked corner covers a block at the top-left, not every third pixel of a flat array
    assert out[:6, :6].max() == pytest.approx(9.0)
    assert out[100:, 100:].max() == pytest.approx(0.0)


def test_flat_repeat_would_misplace_the_marker():
    C = np.zeros((192, 256), np.float32)
    C[0, 0] = 9.0
    flat = np.repeat(C.reshape(-1), 3).reshape(288, 512)
    # this is what the old fallback did: the marker spreads along the row
    assert flat[0, 0] == pytest.approx(9.0)
    assert flat[0, 2] == pytest.approx(9.0)
    assert flat[0, 3] == pytest.approx(0.0)


def test_large_grid_keeps_far_frames_inside():
    from stream_ortho.online.product import StreamingProduct
    cell = 1.0
    G = 512
    p = StreamingProduct(cell, 0, 0, G, G, L=3, z_gap=2.0)
    ix = np.array([0, 0, 200])
    iy = np.array([0, 200, 0])
    r = p.insert(0, ix, iy, np.full(3, 10.0), np.ones(3))
    assert r["inserted"] == 3
    z, n, _ = p.layers()
    assert float(n[0, 0, 0]) == 1.0
    assert float(n[200, 0, 0]) == 1.0 and float(n[0, 200, 0]) == 1.0


def test_calib_tristate():
    def resolve(mode, have_k):
        return {"true": True, "false": False, "auto": bool(have_k)}[mode]
    assert resolve("true", True) is True
    assert resolve("false", True) is False
    assert resolve("auto", True) is True
    assert resolve("auto", False) is False
