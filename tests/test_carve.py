"""P3.2-A: multi-view support carving keeps supported layers and drops one-frame splashes."""
from __future__ import annotations

import numpy as np

from stream_ortho.filter.carve import CarveConfig, carve_rows


def _rows(kfs, layers, zs, keys=None):
    return {"kf": np.asarray(kfs), "key": np.asarray(keys if keys is not None else [5] * len(kfs)),
            "layer": np.asarray(layers), "z": np.asarray(zs, float)}


def test_one_frame_splash_above_a_supported_layer_is_removed():
    kf = np.r_[np.arange(6), np.full(4, 7)]
    d = _rows(kf, [0] * 6 + [1] * 4, [10.0] * 6 + [14.0] * 4)
    keep, stats = carve_rows(d, CarveConfig(min_frames=2))
    assert keep[:6].all() and not keep[6:].any()
    assert stats["no_support"] == 4


def test_supported_second_layer_survives():
    kf = np.r_[np.arange(6), np.arange(4)]
    d = _rows(kf, [0] * 6 + [1] * 4, [10.0] * 6 + [14.0] * 4)
    keep, stats = carve_rows(d, CarveConfig(min_frames=2))
    assert keep.all() and stats["no_support"] == 0


def test_single_layer_cell_is_never_touched():
    d = _rows([0], [0], [10.0])
    keep, stats = carve_rows(d, CarveConfig(min_frames=2))
    assert keep.all(), "a cell observed once must keep its only point (coverage)"


def test_small_gap_is_not_carved():
    """A one-frame layer that is only 0.3 m above the surface is not a splash."""
    kf = np.r_[np.arange(6), np.full(4, 7)]
    d = _rows(kf, [0] * 6 + [1] * 4, [10.0] * 6 + [10.3] * 4)
    keep, stats = carve_rows(d, CarveConfig(min_frames=2, min_gap_m=1.0))
    assert keep.all()


def test_disabled_returns_everything():
    d = _rows(np.r_[np.arange(6), np.full(4, 7)], [0] * 6 + [1] * 4, [10.0] * 6 + [14.0] * 4)
    keep, stats = carve_rows(d, CarveConfig(enabled=False))
    assert keep.all() and stats["kept"] == stats["total"]
