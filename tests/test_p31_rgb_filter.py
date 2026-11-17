"""P3.1: the mosaic must be colour, the final must be re-rendered, offline must be filtered."""
from __future__ import annotations

import numpy as np

from stream_ortho.online.product import StreamingProduct


def test_product_carries_colour():
    p = StreamingProduct(cell=0.45, ox=0, oy=0, H=4, W=4, L=3, z_gap=2.0)
    p.insert(0, np.array([0]), np.array([0]), np.array([10.0]), np.array([1.0]),
             rgb=np.array([[200.0, 10.0, 10.0]], np.float32))
    rgb = p.top_rgb()
    assert rgb.shape == (4, 4, 3)
    assert np.allclose(rgb[0, 0], [200.0, 10.0, 10.0])
    assert rgb[2, 2].max() == 0.0                  # untouched cells stay black


def test_colour_is_a_weighted_mean():
    p = StreamingProduct(cell=0.45, ox=0, oy=0, H=2, W=2, L=3, z_gap=5.0)
    p.insert(0, np.array([0]), np.array([0]), np.array([10.0]), np.array([1.0]),
             rgb=np.array([[0.0, 0.0, 0.0]], np.float32))
    p.insert(1, np.array([0]), np.array([0]), np.array([10.1]), np.array([3.0]),
             rgb=np.array([[200.0, 0.0, 0.0]], np.float32))
    assert np.allclose(p.top_rgb()[0, 0], [150.0, 0.0, 0.0], atol=1.0)


def test_same_frame_multiple_layers_do_not_overwrite():
    """Two layers of one cell in one frame must land in two slots, not fight for slot 0."""
    p = StreamingProduct(cell=0.45, ox=0, oy=0, H=2, W=2, L=3, z_gap=2.0)
    p.insert(0, np.array([0, 0]), np.array([0, 0]), np.array([0.0, 10.0]),
             np.array([1.0, 1.0]),
             rgb=np.array([[10.0, 0, 0], [200.0, 0, 0]], np.float32))
    z, n, _ = p.layers()
    assert float(n[0, 0, 0]) == 1.0 and float(n[0, 0, 1]) == 1.0
    assert np.allclose(p.top_rgb()[0, 0], [200.0, 0, 0])   # the top layer's colour


def test_dsm_mode_still_available():
    """The renderer keeps the old hillshade as an explicit diagnostic mode."""
    import inspect
    from stream_ortho.scripts.run_stream import tile_render_fn_factory
    assert "mode" in inspect.signature(tile_render_fn_factory).parameters


def test_offline_filter_rewrites_the_table(tmp_path):
    """The offline stage must run Module A and write its own statistics."""
    import pathlib
    import numpy as np
    from stream_ortho.app.offline import filter_obs
    from stream_ortho.filter.pointmap import FilterConfig

    src = tmp_path / "obs"
    src.mkdir()
    rng = np.random.default_rng(0)
    n = 400
    kf = np.repeat(np.arange(4), n // 4).astype(np.int32)
    x = rng.uniform(0, 20, n); y = rng.uniform(0, 20, n)
    z = np.full(n, 10.0); conf = np.ones(n)
    np.savez(src / "obs_merged.npz", T_WP=np.eye(4), cell=np.float32(0.45), kf=kf,
             frame_id=kf.copy(), key=np.zeros(n, np.int64), x=x, y=y, z=z, conf=conf,
             cos_inc=np.ones(n), D=np.full(n, 30.0), layer=np.zeros(n, np.int32),
             rgb=np.full((n, 3), 100.0, np.float32), n=np.ones(n, np.int32))
    out, stats = filter_obs(src, tmp_path, FilterConfig(cell=0.45))
    assert (pathlib.Path(out) / "obs_merged.npz").exists()
    assert (tmp_path / "filter_stats_offline.json").exists()
    assert stats["kept"] > 0 and stats["total"] > 0
    kept = np.load(pathlib.Path(out) / "obs_merged.npz")
    assert kept["kf"].size == stats["kept"]
    assert "T_WP" in kept.files and "rgb" in kept.files
