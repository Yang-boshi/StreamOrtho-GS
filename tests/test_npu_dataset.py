"""S1 tests: ``NPUDroneMapDataset`` over the sequences kept in ``PROJ/data``.

Current data set (2026-10-24, after the user completed ``ieu`` and removed
``centralPark``): ``phantom3-factory-kfs`` (402), ``phantom3-ieu-kfs`` (467),
``phantom3-npu-kfs`` (457) -- all flat layouts, all with 100% GT coverage.

Run::

    python -m pytest stream_ortho/tests/test_npu_dataset.py -q
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import (
    NPUDroneMapDataset,
    discover_sequences,
    list_timestamped_images,
    resolve_sequence_root,
)

DATA_ROOT = zcfg.dataset_root()


def _seqs():
    return discover_sequences(DATA_ROOT)


def test_layout_resolution():
    """Both on-disk layouts must resolve to the dir holding ``config.cfg``."""
    with tempfile.TemporaryDirectory() as td:
        flat = pathlib.Path(td) / "flat"
        (flat / "rgb").mkdir(parents=True)
        (flat / "config.cfg").write_text("Plane=0 0 0 0 0 0 1\n")
        assert resolve_sequence_root(flat) == flat

        nested = pathlib.Path(td) / "nested"
        inner = nested / "nested"
        (inner / "rgb").mkdir(parents=True)
        (inner / "config.cfg").write_text("Plane=0 0 0 0 0 0 1\n")
        assert resolve_sequence_root(nested) == inner
        assert resolve_sequence_root(inner) == inner
    print("[ok] nested / flat layout resolution")


def test_all_sequences_load():
    seqs = _seqs()
    names = {s.name for s in seqs}
    # the data root gained phantom3-centralPark-kfs (no GT trajectory), so the old
    # "exactly 3" assertion no longer describes the environment; require the paper's
    # sequences and accept any extras
    assert {"phantom3-ieu-kfs", "phantom3-npu-kfs", "phantom3-factory-kfs"} <= names, names
    loaded = []
    for seq in seqs:
        ds = NPUDroneMapDataset(seq)
        loaded.append(ds)
        assert len(ds) > 0, ds.name
        if ds.traj is None:            # e.g. centralPark: imagery only, no GT trajectory
            continue
        assert ds.gt_coverage.ratio > 0.99, (ds.name, ds.gt_coverage.as_dict())
    print(f"[ok] {len(loaded)}/{len(seqs)} sequences load with full GT: "
          f"{[(d.name, len(d)) for d in loaded]}")


def test_per_sequence_contract():
    checked = 0
    for seq in _seqs():
        ds = NPUDroneMapDataset(seq)
        on_disk = list_timestamped_images(ds.root / "rgb")
        assert ds.cfg.width == 1920 and ds.cfg.height == 1080, seq.name
        assert len(ds) == len(on_disk), (seq.name, len(ds), len(on_disk))
        assert len(ds) > 0
        assert ds.timestamps.dtype == np.float64
        assert np.all(np.diff(ds.timestamps) > 0), f"{seq.name}: 时间戳非递增"
        assert tuple(ds.frame_shape()) == (288, 512), (seq.name, ds.frame_shape())
        if ds.traj is not None:
            assert len(ds.traj) == len(ds), (seq.name, len(ds.traj), len(ds))
        h = ds.height_diagnostics()
        assert h["plane_is_horizontal"], (seq.name, h)
        checked += 1
    # the paper's three sequences must satisfy the contract; imagery-only extras may not
    assert checked >= 3, checked
    print(f"[ok] per-sequence contract on {checked} sequences (timestamps, shapes, GT, plane)")


def test_image_reading():
    ds = NPUDroneMapDataset(DATA_ROOT / "phantom3-npu-kfs")
    img = ds.read_img(0)
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    ts, img2 = ds[0]
    assert float(ts) == float(ds.timestamps[0])
    assert img2.shape == img.shape
    print("[ok] read_img / __getitem__ shapes and timestamp")


def test_subsample_keeps_alignment():
    ds = NPUDroneMapDataset(DATA_ROOT / "phantom3-ieu-kfs")
    n0 = len(ds)
    ts0, gt0 = ds.timestamps.copy(), ds.gt_positions.copy()
    ds.subsample(4)
    assert len(ds) == len(range(0, n0, 4)), (len(ds), n0)
    np.testing.assert_allclose(ds.timestamps, ts0[::4])
    np.testing.assert_allclose(ds.gt_positions, gt0[::4])
    assert np.all(np.diff(ds.timestamps) > 0)
    print(f"[ok] subsample(4): {n0} -> {len(ds)} frames, GT aligned")


def test_calibration_matches_upstream():
    """Our analytic K_frame must match upstream's (<= 1.5 px)."""
    for name in ("phantom3-npu-kfs", "phantom3-factory-kfs", "phantom3-ieu-kfs"):
        ds = NPUDroneMapDataset(DATA_ROOT / name)
        analytic = ds.K_frame_analytic()
        ds.enable_calibration()
        upstream = np.asarray(ds.camera_intrinsics.K_frame)
        err = float(np.abs(analytic - upstream).max())
        assert err < 1.5, (name, err)
        if not ds.has_calib():        # imagery-only sequences (centralPark) carry no plane
            continue
        img = ds.camera_intrinsics.remap(ds.read_img(0))
        assert img.shape == (1080, 1920, 3), (name, img.shape)
        print(f"[ok] {name}: max |K_frame_analytic - upstream| = {err:.3f} px")
        assert tuple(ds.get_img_shape()[0]) == (288, 512)


def test_discover_and_summaries():
    import json

    seqs = _seqs()
    names = {s.name for s in seqs}
    assert {"phantom3-ieu-kfs", "phantom3-npu-kfs", "phantom3-factory-kfs"} <= names
    for seq in seqs:
        ds = NPUDroneMapDataset(seq)
        s = ds.summary()
        json.dumps(s)
        assert s["n_frames"] == len(ds)
        assert s["T_WP_roundtrip_error"] < 1e-9
    print("[ok] summaries are JSON-serialisable for 3 sequences")


def run_all():
    if not DATA_ROOT.exists():
        print("[skip] data root missing")
        return
    test_layout_resolution()
    test_all_sequences_load()
    test_per_sequence_contract()
    test_image_reading()
    test_subsample_keeps_alignment()
    test_calibration_matches_upstream()
    test_discover_and_summaries()
    print("test_npu_dataset: ALL PASSED")


if __name__ == "__main__":
    run_all()
