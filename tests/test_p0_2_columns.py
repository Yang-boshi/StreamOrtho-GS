"""P0-2 (v2) tests: global dense-grid column store.

Expectations are hand-computed first (cell=0.45, x = 0.1 + 0.45 k -> floor(x/0.45) = k,
so ten samples give ten distinct cells -- checked in the test itself before asserting).
"""

import numpy as np

from stream_ortho.map.columns import ColumnStore
from stream_ortho.map.obs import FrameObs
from stream_ortho.map.selection import pack_cell

CELL = 0.45
BBOX = (-1.0, -1.0, 6.0, 6.0)


def _frame(kf, zs, xy=None, rgb=None, conf=0.5):
    zs = np.atleast_1d(np.asarray(zs, np.float32))
    n = len(zs)
    xy = np.zeros((n, 2)) + 0.1 if xy is None else np.asarray(xy, np.float64).reshape(n, 2)
    ij = np.floor(xy / CELL).astype(np.int64)
    rgb = np.full((n, 3), 100, np.uint8) if rgb is None else np.asarray(rgb, np.uint8)
    return FrameObs(kf=kf, cell_key=pack_cell(ij[:, 0], ij[:, 1]), z=zs, rgb=rgb,
                    conf=np.full(n, conf, np.float16), cos_inc=np.ones(n, np.float16))


def _store(L=3, z_gap=2.0, bbox=BBOX):
    return ColumnStore(CELL, np.eye(4), L=L, z_gap=z_gap, bbox=bbox)


def test_columns_never_filter_by_top():
    """A low observation after a high one becomes a **second layer** (never dropped)."""
    s = _store()
    s.merge_frame(_frame(0, [9.0]))
    s.merge_frame(_frame(1, [1.0]))
    st = s.stats()
    assert st["layers_hist"]["2"] == 1, st["layers_hist"]
    assert st["n_overflow_cells"] == 0
    assert float(s.to_cloud(layer="top")["means"][0, 2]) > 8.0
    assert float(s.to_cloud(layer="bottom")["means"][0, 2]) < 2.0


def test_columns_merge_within_z_gap():
    s = _store(z_gap=2.0)
    s.merge_frame(_frame(0, [1.0]))
    s.merge_frame(_frame(1, [1.4]))
    st = s.stats()
    assert st["layers_hist"]["1"] == 1 and st["layers_hist"]["2"] == 0
    z = float(s.to_cloud()["means"][0, 2])
    assert abs(z - 1.2) < 0.05, z


def test_columns_overflow_is_counted_not_dropped():
    s = _store(L=2)
    for kf, z in enumerate((0.0, 5.0, 10.0)):
        s.merge_frame(_frame(kf, [z]))
    st = s.stats()
    assert st["n_overflow_cells"] == 1, st
    assert st["n_overflow_points"] == 1, st
    assert st["n_obs"] == 3                      # nothing discarded
    assert st["layers_hist"]["2"] == 1


def test_columns_small_bbox_allocates_only_bbox():
    """The grid is exactly the requested bbox; only touched cells are live."""
    s = _store(bbox=(-1.0, -1.0, 5.0, 5.0))
    assert s.n_cells == s.W * s.H
    s.merge_frame(_frame(0, [0.0]))
    assert s.stats()["n_cells_live"] == 1
    assert s.stats()["bbox_cells"] == s.W * s.H


def test_global_index_negative_and_boundary():
    """ix,iy in {-1,0,1}: nine distinct flat indices, inside the grid."""
    s = _store(bbox=(-2.0, -2.0, 2.0, 2.0))
    flats = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            flats.append(int(s.flat_index(np.array([ix]), np.array([iy]))[0]))
    assert len(set(flats)) == 9, flats
    assert min(flats) >= 0 and max(flats) < s.n_cells


def test_layer_accounting():
    """10 cells, 5 points per cell per frame -> one layer decision per (frame, cell)."""
    xs = np.arange(10) * CELL + 0.1
    assert len(set(np.floor(xs / CELL).astype(int).tolist())) == 10   # hand-checked
    s = _store()
    xy = np.c_[np.repeat(xs, 5), np.zeros(50)]
    for kf in range(3):
        s.merge_frame(_frame(kf, np.full(50, 1.0 + 0.1 * kf, np.float32), xy=xy))
    st = s.stats()
    assert st["n_cells_live"] == 10, st
    assert st["layers_hist"]["1"] == 10, st["layers_hist"]
    assert st["n_new_layer"] == st["n_layers_total"] == 10, st
    assert st["n_overflow_points"] == 0 and st["n_overflow_cells"] == 0
    assert st["n_obs"] == 150


def test_top_bottom_no_nan():
    """Empty layers must never leak into top/bottom (no NaN anywhere)."""
    s = _store()
    s.merge_frame(_frame(0, [3.0]))
    for lay in ("top", "bottom"):
        m = s.to_cloud(layer=lay)["means"]
        assert np.isfinite(m).all(), (lay, m)
        assert abs(float(m[0, 2]) - 3.0) < 1e-4


def test_columns_representative_colour():
    """Colour is the **highest-conf observation** of the layer (not the weighted mean).

    Reason: the layer mean over a 0.45 m cell is a 1.5 px box average, which cost 10% of
    the product sharpness; one representative sample keeps the texture.  The higher-conf
    observation must win even when a lower-conf one arrives later.
    """
    s = _store()
    s.merge_frame(_frame(0, [1.0], rgb=[[200, 100, 50]], conf=0.9))
    s.merge_frame(_frame(1, [1.0], rgb=[[0, 0, 0]], conf=0.3))
    c = s.to_cloud()["colors"][0]
    assert abs(c[0] - 200 / 255) < 0.05 and abs(c[2] - 50 / 255) < 0.05, c
    # and a later, more confident observation replaces it
    s.merge_frame(_frame(2, [1.0], rgb=[[0, 0, 255]], conf=0.95))
    c = s.to_cloud()["colors"][0]
    assert abs(c[2] - 1.0) < 0.05, c


def test_columns_to_cloud_applies_T_WP():
    T = np.eye(4)
    T[:3, 3] = [100.0, -50.0, 7.0]
    s = ColumnStore(CELL, T, L=3, bbox=BBOX)
    s.merge_frame(_frame(0, [1.0]))
    m = s.to_cloud()["means"][0]
    assert abs(m[0] - (0.225 + 100.0)) < 1e-3
    assert abs(m[1] - (0.225 - 50.0)) < 1e-3
    assert abs(m[2] - (1.0 + 7.0)) < 1e-3
