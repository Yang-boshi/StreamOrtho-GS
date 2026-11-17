"""Project paths and default runtime settings.

Importing this module must never import torch, so that light-weight tooling
(dataset audits, parsing) stays fast.
"""

from __future__ import annotations

import os
import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PKG_DIR = pathlib.Path(__file__).resolve().parent          # .../Ortho-3dgs/stream_ortho
REPO_DIR = PKG_DIR.parent                                   # .../Ortho-3dgs (repo root)

# Project root holding external data / checkpoints / results.  Defaults to the
# parent of the repo checkout; override with the ``ORTHO_PROJ`` environment
# variable when the repo lives elsewhere (e.g. a plain GitHub clone).
PROJ = pathlib.Path(os.environ.get("ORTHO_PROJ", REPO_DIR.parent.parent))

# Vendored upstream checkout (carries our 2-file patch: jpg loader + pose hook).
MAST3R_SLAM_DIR = REPO_DIR / "third_party" / "MASt3R-SLAM"

DATA_DIR = PROJ / "data"
# Canonical dataset root: the sequences kept directly under ``PROJ/data``.
# (``PROJ/data/NPU`` is an optional symlink to the full 13-sequence DroneMap
#  release; it is *not* the default because the project works on the three
#  sequences under ``PROJ/data``.)
NPU_ROOT = pathlib.Path(os.environ.get("ZSO_NPU_ROOT", DATA_DIR))
CKPT_DIR = PROJ / "checkpoints"
RESULTS_DIR = PROJ / "results"


def results_dir(stage: str) -> pathlib.Path:
    """Return (and create) ``results/<stage>``."""
    p = RESULTS_DIR / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_root() -> pathlib.Path:
    """Resolve the NPU root, following a symlink if needed."""
    return NPU_ROOT.resolve()


# ---------------------------------------------------------------------------
# Defaults (16 GB card; the design doc's table is for an 8 GB card)
# ---------------------------------------------------------------------------

IMG_SIZE = 512          # MASt3R fixed input resolution
DEVICE = "cuda:0"

MAST3R_CKPT = CKPT_DIR / "mast3r" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
MAST3R_RETRIEVAL_CKPT = (
    CKPT_DIR / "mast3r"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
)
MAST3R_CODEBOOK = (
    CKPT_DIR / "mast3r"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
)
SPLATT3R_HEAD_CKPT = CKPT_DIR / "splatt3r" / "epoch=19-step=1200.ckpt"


def checkpoints_status() -> dict:
    """Return a dict of expected checkpoint paths -> (exists, size_in_MB)."""
    out = {}
    for name, p in {
        "mast3r": MAST3R_CKPT,
        "mast3r_retrieval": MAST3R_RETRIEVAL_CKPT,
        "mast3r_codebook": MAST3R_CODEBOOK,
        "splatt3r_head": SPLATT3R_HEAD_CKPT,
    }.items():
        out[name] = {
            "path": str(p),
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0,
        }
    return out
