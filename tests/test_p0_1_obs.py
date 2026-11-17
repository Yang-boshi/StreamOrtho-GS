"""P0-1 red tests: the observation store must keep 1..L observations per cell.

Written BEFORE the implementation (rule R4): every test below must fail with
``ModuleNotFoundError: stream_ortho.map.obs`` until ``map/obs.py`` exists.
"""

import numpy as np
import pytest


def _frame(kf, n, xy_range=(0.0, 4.0), z=0.0, cell=0.45, grid=False):
    rng = np.random.default_rng(kf)
    if grid:
        # a fixed lattice -> the same physical cells appear in every frame
        g = np.arange(n)
        xy = np.stack([(xy_range[0] + (g % 8) * 0.05), (xy_range[0] + (g // 8) * 0.05)], 1)
    else:
        xy = rng.uniform(*xy_range, (n, 2))
    from stream_ortho.map.selection import pack_cell

    ij = np.floor(xy / cell).astype(np.int64)
    return dict(
        kf=kf,
        cell_key=pack_cell(ij[:, 0], ij[:, 1]),
        z=np.full(n, float(z), np.float32) + rng.normal(0, 0.01, n).astype(np.float32),
        rgb=(rng.uniform(0, 255, (n, 3))).astype(np.uint8),
        conf=np.full(n, 0.5, np.float16),
        cos_inc=np.full(n, 0.9, np.float16),
    )


def _store(window=15, cell=0.45):
    from stream_ortho.map.obs import ObservationStore

    return ObservationStore(cell=cell, T_WP=np.eye(4), window=window)


def test_pack_cell_roundtrip_negative():
    """pack/unpack must be a bijection for negative and large indices."""
    from stream_ortho.map.selection import pack_cell, unpack_cell

    rng = np.random.default_rng(0)
    ix = rng.integers(-500_000, 500_000, 5000).astype(np.int64)
    iy = rng.integers(-500_000, 500_000, 5000).astype(np.int64)
    k = pack_cell(ix, iy)
    assert len(np.unique(k)) == len(k)          # injective
    rx, ry = unpack_cell(k)
    assert np.array_equal(rx, ix) and np.array_equal(ry, iy)


def test_obs_ring_keeps_last_W_frames():
    s = _store(window=5)
    for kf in range(8):
        s.append_frame(**{k: v for k, v in _frame(kf, 50).items()})
    st = s.stats()
    assert st["n_frames"] == 5
    assert st["n_evicted_frames"] == 3
    assert st["evicted_kfs"] == [0, 1, 2]


def test_obs_append_does_not_concatenate_full_map(monkeypatch):
    calls = {"n": 0, "max_rows": 0}
    real = np.concatenate

    def spy(seq, *a, **k):
        calls["n"] += 1
        try:
            rows = max(len(x) for x in seq)
        except Exception:
            rows = 0
        calls["max_rows"] = max(calls["max_rows"], rows)
        return real(seq, *a, **k)

    monkeypatch.setattr(np, "concatenate", spy)
    s = _store(window=15)
    n = 2000
    for kf in range(5):
        s.append_frame(**_frame(kf, n))
    # the contract: never concatenate across frames (a full-map copy would be >= 2 * n)
    assert calls["max_rows"] < 2 * n, (calls["n"], calls["max_rows"])


def test_obs_preserves_two_layers_in_same_cell():
    s = _store(window=15)
    a = _frame(0, 20, xy_range=(0.0, 0.4), z=1.0, grid=True)
    b = _frame(1, 20, xy_range=(0.0, 0.4), z=9.0, grid=True)
    s.append_frame(**a)
    s.append_frame(**b)
    key = a["cell_key"][0]
    got = s.query(np.array([key]))
    assert len(got) == 2, len(got)                      # both frames keep the cell
    zs = sorted(np.round(np.concatenate([g.z for g in got]).astype(float), 1).tolist())
    assert min(zs) < 2.0 and max(zs) > 8.0, zs          # both layers survive


def test_obs_overflow_returns_oldest_frame():
    s = _store(window=3)
    for kf in range(5):
        s.append_frame(**_frame(kf, 10))
    seen = [s.pop_oldest().kf for _ in range(2)]
    assert seen == [0, 1], seen


def test_obs_stats_obs_per_cell():
    s = _store(window=15)
    s.append_frame(**_frame(0, 10, xy_range=(0.0, 2.0)))
    st = s.stats()
    assert st["n_obs"] > 0
    assert st["obs_per_cell"] == pytest.approx(st["n_obs"] / max(st["n_cells"], 1))


def test_obs_keys_stable_across_frames():
    """The same physical cell must map to the same key from different frames."""
    from stream_ortho.map.selection import pack_cell

    s = _store(window=15)
    a = _frame(0, 200, xy_range=(1.0, 1.2))
    b = _frame(1, 200, xy_range=(1.0, 1.2))
    s.append_frame(**a)
    s.append_frame(**b)
    k0 = set(a["cell_key"].tolist())
    k1 = set(b["cell_key"].tolist())
    assert k0 & k1, "same xy range must produce overlapping cell keys"
    assert s.stats()["n_cells"] == len(k0 | k1)


def test_obs_eviction_preserves_layers():
    """Evicting a frame must not delete the layers of the remaining frames."""
    evicted = []

    def sink(frame):
        evicted.append(frame.kf)

    from stream_ortho.map.obs import ObservationStore

    s = ObservationStore(cell=0.45, T_WP=np.eye(4), window=2, on_evict=sink)
    s.append_frame(**_frame(0, 20, xy_range=(0.0, 0.4), z=1.0, grid=True))
    s.append_frame(**_frame(1, 20, xy_range=(0.0, 0.4), z=9.0, grid=True))
    s.append_frame(**_frame(2, 20, xy_range=(0.0, 0.4), z=1.2, grid=True))
    assert evicted == [0], evicted
    key = _frame(1, 1, xy_range=(0.0, 0.4), grid=True)["cell_key"][0]
    got = s.query(np.array([key]))
    zs = np.concatenate([g.z for g in got]).astype(float)
    # frame 1's roof layer must survive the eviction of frame 0, and the newer frame 2
    # (ground) must be there too -- i.e. eviction removes a frame, not a layer
    assert zs.max() > 8.0 and zs.min() < 2.0, (zs.min(), zs.max())


# ---------------------------------------------------------------- P0-1b (bug fixes)
def test_pop_oldest_only_drains_pending():
    """B1: pop_oldest must never shrink the retained window."""
    s = _store(window=3)
    for kf in range(5):
        s.append_frame(**_frame(kf, 10))
    assert [s.pop_oldest().kf for _ in range(2)] == [0, 1]
    assert s.pop_oldest() is None            # pending empty -> None, ring untouched
    assert s.stats()["n_frames"] == 3


def test_on_evict_is_delivered_once_and_not_leaked():
    """B2: with a sink, *_pending must stay empty* and each kf is delivered exactly once."""
    got = []
    from stream_ortho.map.obs import ObservationStore

    s = ObservationStore(cell=0.45, T_WP=np.eye(4), window=2, on_evict=lambda f: got.append(f.kf))
    for kf in range(5):
        s.append_frame(**_frame(kf, 10))
    assert got == [0, 1, 2], got
    assert len(s._pending) == 0, s._pending      # no leak when a sink is used


def test_obs_owns_its_arrays():
    """B3: mutating the caller's array after append must not change the store."""
    s = _store(window=15)
    f = _frame(0, 64)
    key = f["cell_key"][0]
    sel = f["cell_key"] == key                  # only the points in that cell
    z_src = f["z"][sel].copy()
    s.append_frame(**f)
    f["z"][:] = -999.0                          # caller reuses/mutates its buffer
    got = s.query(np.array([key]))[0]
    assert len(got.z) == len(z_src)
    assert np.allclose(np.sort(got.z), np.sort(z_src))


def test_from_patch_matches_patch_fields():
    """B3: from_patch must reproduce cell_key/z/rgb/conf, including negative coordinates."""
    from stream_ortho.map.gaussian import GaussianPatch
    from stream_ortho.map.obs import FrameObs

    n = 500
    rng = np.random.default_rng(0)
    T = np.eye(4)
    means = np.c_[rng.uniform(-50, 50, (n, 2)), rng.uniform(0, 20, n)].astype(np.float32)
    cols = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    conf = rng.uniform(0, 1, n).astype(np.float32)
    patch = GaussianPatch(means=means, quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                          scales=np.full((n, 3), 0.2, np.float32),
                          opacity=np.ones(n, np.float32), colors=cols,
                          owner=3, conf=conf, weight=np.ones(n, np.float32))
    f = FrameObs.from_patch(patch, kf=7, T_WP=T, cell=0.45)
    from stream_ortho.map.selection import pack_cell

    ij = np.floor(means[:, :2] / 0.45).astype(np.int64)
    assert np.array_equal(f.cell_key, pack_cell(ij[:, 0], ij[:, 1]))
    assert np.allclose(f.z, means[:, 2])
    assert np.array_equal(f.rgb, (cols * 255).astype(np.uint8))
    assert np.allclose(f.conf.astype(np.float32), conf, atol=1e-3)
    assert np.allclose(f.cos_inc, 1.0) and f.kf == 7
    assert np.all(f.owner == 7)


def test_frame_count_per_cell():
    """frames/cell counts distinct keyframes: 3 points of one frame in a cell = 1."""
    from stream_ortho.map.obs import frames_per_cell

    f0 = np.array([7, 7, 7, 9], np.int64)      # 3 points in cell 7, 1 in cell 9
    f1 = np.array([7, 7, 11], np.int64)        # 2 points in cell 7, 1 in cell 11
    uniq, cnt = frames_per_cell([f0, f1])
    d = dict(zip(uniq.tolist(), cnt.tolist()))
    assert d[7] == 2          # two frames saw cell 7 (not 5 points)
    assert d[9] == 1 and d[11] == 1
    # a cell seen by one frame with many points still counts once
    _, c2 = frames_per_cell([np.full(100, 5, np.int64)])
    assert list(c2) == [1]


def test_pending_does_not_grow_when_drained():
    """The leak guard: draining after every append keeps n_pending at 0."""
    s = _store(window=3)
    for kf in range(10):
        s.append_frame(**_frame(kf, 5))
        while s.pop_oldest() is not None:
            pass                                   # consume immediately (drop the frame)
    st = s.stats()
    assert st["n_pending"] == 0, st["n_pending"]
    assert st["n_frames"] == 3                    # the retained window is intact
    assert st["n_evicted_frames"] == 7


def test_pending_leak_guard_warns():
    """Without draining, the store must warn instead of growing silently."""
    import warnings as _w

    s = _store(window=2)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        for kf in range(6):
            s.append_frame(**_frame(kf, 5))
    assert any("evicted frames are queued" in str(w.message) for w in rec), rec
