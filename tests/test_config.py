"""Config: defaults, unknown keys, ablations, CLI overrides."""
from __future__ import annotations

import pathlib

import pytest

from stream_ortho.app.config import Config, ConfigError, load_config, override

CFG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.map.freeze_owner is True
    assert cfg.ortho.render == "rgb"          # P3.1: a mosaic is a colour orthophoto
    assert cfg.filter.enabled is True


def test_load_default_yaml():
    cfg = load_config(CFG_DIR / "default.yaml")
    assert cfg.map.cell == 0.45
    assert cfg.ortho.gsd == 0.30


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("map:\n  cell: 0.5\n  typo_key: 1\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_unknown_section_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("maps:\n  cell: 0.5\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_ablation_configs_flip_one_switch():
    a = load_config(CFG_DIR / "ablation_no_filter.yaml")
    b = load_config(CFG_DIR / "ablation_no_freeze.yaml")
    c = load_config(CFG_DIR / "ablation_offline_only.yaml")
    assert a.filter.enabled is False and a.map.freeze_owner is True
    assert b.map.freeze_owner is False and b.filter.enabled is True
    assert c.map.online_calib is False


def test_override_changes_and_validates():
    cfg = override(Config(), data__images="/tmp/x", ortho__gsd=0.5)
    assert cfg.data.images == "/tmp/x" and cfg.ortho.gsd == 0.5
    with pytest.raises(ConfigError):
        override(Config(), ortho__gsd=-1)


def test_provided_intrinsics_requires_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  intrinsics: provided\n")
    with pytest.raises(ConfigError):
        load_config(p)
