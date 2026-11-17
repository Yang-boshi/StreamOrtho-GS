"""Small compatibility layer for the upstream MASt3R-SLAM checkout.

The upstream repository is treated as read-only (design doc 11.1).  We only
add its root to ``sys.path`` and expose the few helpers we need.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys

from stream_ortho import config as zcfg


def ensure_upstream_on_path() -> pathlib.Path:
    """Put the local MASt3R-SLAM checkout first on ``sys.path``.

    This matters because the conda env may have an *editable* install of another
    copy of ``mast3r_slam``.  We want every run of this project to use the
    vendored ``third_party/MASt3R-SLAM`` checkout instead.
    """
    root = zcfg.MAST3R_SLAM_DIR
    root_s = str(root)
    if not (root / "mast3r_slam").is_dir():
        raise FileNotFoundError(f"MASt3R-SLAM 代码不存在: {root}")
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    for sub in ("thirdparty/mast3r", "thirdparty/in3d"):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.append(p)
    return root


def load_upstream_config(path=None):
    """Load the upstream YAML config into the global ``mast3r_slam.config.config``."""
    ensure_upstream_on_path()
    from mast3r_slam.config import config, load_config  # noqa: WPS433

    if not config:
        p = pathlib.Path(path) if path else zcfg.MAST3R_SLAM_DIR / "config" / "base.yaml"
        cwd = os.getcwd()
        # upstream default paths are relative ("config/base.yaml")
        os.chdir(zcfg.MAST3R_SLAM_DIR)
        try:
            load_config(str(p) if pathlib.Path(p).is_absolute() else str(p))
        finally:
            os.chdir(cwd)
    return config


@contextlib.contextmanager
def upstream_cwd():
    """Run inside the MASt3R-SLAM repo (upstream uses relative checkpoint paths)."""
    cwd = os.getcwd()
    os.chdir(zcfg.MAST3R_SLAM_DIR)
    try:
        yield zcfg.MAST3R_SLAM_DIR
    finally:
        os.chdir(cwd)


def upstream_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=zcfg.MAST3R_SLAM_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"
