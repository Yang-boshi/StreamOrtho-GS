"""RGBFolderDataset: numeric names, no numeric names, intrinsics modes, no config.cfg."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from stream_ortho.data.rgb_folder import RGBFolderDataset


def _make_dir(tmp_path, names=("10.png", "2.png", "1.png"), size=(64, 96)):
    for n in names:
        Image.fromarray((np.random.default_rng(0).random((size[0], size[1], 3)) * 255)
                        .astype(np.uint8)).save(tmp_path / n)
    return tmp_path


def test_numeric_names_are_sorted_by_value(tmp_path):
    d = RGBFolderDataset(_make_dir(tmp_path), intrinsics="none")
    assert [p.name for p in d.files] == ["1.png", "2.png", "10.png"]
    assert len(d) == 3


def test_non_numeric_names_keep_name_order(tmp_path):
    d = RGBFolderDataset(_make_dir(tmp_path, names=("b.png", "a.png", "c.png")),
                         intrinsics="none")
    assert [p.name for p in d.files] == ["a.png", "b.png", "c.png"]


def test_provided_intrinsics_are_metric(tmp_path):
    d = RGBFolderDataset(_make_dir(tmp_path), intrinsics="provided",
                         intrinsics_value=[100.0, 100.0, 48.0, 32.0])
    assert d.metric_scale is True
    assert d.K_frame[0, 0] == 100.0


def test_none_intrinsics_is_relative_scale(tmp_path):
    d = RGBFolderDataset(_make_dir(tmp_path), intrinsics="none")
    assert d.metric_scale is False
    assert d.intrinsics_source == "none"
    assert d.K_frame[0, 0] == pytest.approx(0.8 * 96)


def test_auto_without_exif_falls_back_and_flags(tmp_path):
    d = RGBFolderDataset(_make_dir(tmp_path), intrinsics="auto")
    assert d.intrinsics_source in ("default", "exif")
    if d.intrinsics_source == "default":
        assert d.metric_scale is False


def test_item_is_a_normalised_tensor(tmp_path):
    d = RGBFolderDataset(_make_dir(tmp_path), image_size=48, intrinsics="none")
    t, img = d[0]
    assert img.shape[0] == 3 and max(img.shape[1:]) == 48   # long side resized
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RGBFolderDataset(tmp_path / "nope", intrinsics="none")
