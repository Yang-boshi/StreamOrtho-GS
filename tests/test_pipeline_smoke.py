"""P0 acceptance: one command, 10 frames, exit 0, config resolved and written."""
from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

from stream_ortho.app.run import main


def _images(tmp_path, n=10, size=(72, 96)):
    root = tmp_path / "rgb"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray((rng.random((size[0], size[1], 3)) * 255).astype(np.uint8)).save(
            root / f"{i:04d}.png")
    return root


def test_one_command_runs_and_writes_config(tmp_path):
    root = _images(tmp_path)
    out = tmp_path / "out"
    base_cfg = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    assert main(["--config", str(base_cfg), "--dry-run"]) == 0
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text("frontend:\n  backend: replay\n")
    rc = main(["--config", str(cfg), "--images", str(root), "--out", str(out),
               "--max-frames", "10", "--intrinsics", "none"])
    assert rc == 0
    assert (out / "run.json").exists()
    assert (out / "config_resolved.yaml").exists()
    res = json.loads((out / "run.json").read_text())
    assert res["n_frames"] == 10
    assert res["metric_scale"] is False            # intrinsics: none
    assert [s["name"] for s in res["stages"]][:3] == ["data", "frontend", "filter"]


def test_unknown_key_exits_nonzero(tmp_path):
    root = _images(tmp_path, n=2)
    bad = tmp_path / "bad.yaml"
    bad.write_text("ortho:\n  gsd: 0.3\n  nope: 1\n")
    rc = main(["--config", str(bad), "--images", str(root), "--out", str(tmp_path / "o")])
    assert rc == 2
