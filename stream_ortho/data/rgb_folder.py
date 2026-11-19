"""A plain RGB-folder dataset that does not need ``config.cfg``.

The upstream loop expects a ``MonocularDataset``: an indexable sequence of
``(timestamp, image_tensor)`` plus ``read_img``/``frame_shape``/``K_frame``/``T_WP``.  This
class provides exactly that for a directory of images, with four intrinsics modes:

``provided``  use ``intrinsics_value`` (metric)
``exif``      read the 35 mm equivalent focal length from EXIF (metric, approximate)
``auto``      EXIF when available, otherwise the default ``fx = fy = 0.8 W`` (flagged)
``none``      no calibration at all: default ``K``, ``metric_scale = False`` -- the pipeline
              still runs, but the cell size and GSD are *relative* units and the run record
              says so.
"""
from __future__ import annotations

import pathlib
from typing import List, Optional, Tuple

import numpy as np


def _list_images(root: pathlib.Path) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files and (root / "rgb").is_dir():
        files = [p for p in (root / "rgb").iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"no images in {root}")

    def key(p: pathlib.Path):
        stem = p.stem
        try:
            return (0, float(stem), "")
        except ValueError:
            return (1, 0.0, p.name)

    files.sort(key=key)
    return files


def exif_focal_px(path: pathlib.Path, width: int) -> Optional[float]:
    """35 mm-equivalent focal length -> pixels (``f35 * W / 36``); ``None`` when absent."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except Exception:
        return None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            data = {TAGS.get(k, k): v for k, v in exif.items()}
            # 35 mm equivalent lives in the Exif IFD under PhotographicSensitivity neighbours
            f35 = data.get("FocalLengthIn35mmFilm")
            if not f35:
                ifd = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
                f35 = {TAGS.get(k, k): v for k, v in ifd.items()}.get("FocalLengthIn35mmFilm")
            if not f35:
                return None
            return float(f35) * width / 36.0
    except Exception:
        return None


class RGBFolderDataset:
    def __init__(self, root, image_size: int = 512, stride: int = 1,
                 intrinsics: str = "auto", intrinsics_value=None, max_frames: int = 0,
                 T_WP=None, load_images: bool = True):
        self.root = pathlib.Path(root)
        self.files = _list_images(self.root)[:: max(int(stride), 1)]
        if max_frames:
            self.files = self.files[: int(max_frames)]
        self.image_size = int(image_size)
        self.intrinsics_mode = intrinsics
        self.T_WP = np.eye(4) if T_WP is None else np.asarray(T_WP, np.float64)
        self._imgs = {}
        self.timestamps = np.arange(len(self.files), dtype=np.float64)
        self.frame_ids = list(range(len(self.files)))
        self._load_first(load_images)
        self._set_intrinsics(intrinsics_value)

    # ------------------------------------------------------------------
    def _load_first(self, load_images: bool):
        from PIL import Image
        with Image.open(self.files[0]) as im:
            self._hw = (im.height, im.width)
        self._f0 = self.files[0]
        if load_images:
            self._imgs[0] = self.read_img(0)

    def _calib_txt_intrinsics(self):
        """Parse fx, fy, cx, cy from calib.txt in root or parent dir if present."""
        import math
        import re
        for cand in (self.root / "calib.txt", self.root.parent / "calib.txt"):
            if not cand.is_file():
                continue
            try:
                vals = []
                for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    for tok in re.split(r"[,\s]+", line):
                        if tok:
                            vals.append(float(tok))
                if len(vals) >= 4 and all(math.isfinite(v) for v in vals[:4]):
                    fx, fy, cx, cy = vals[:4]
                    if fx > 10.0 and fy > 10.0:
                        return (fx, fy, cx, cy)
            except Exception:
                continue
        return None

    def _set_intrinsics(self, intrinsics_value):
        H, W = self._hw
        self.intrinsics_source = "none"
        self.metric_scale = False
        fx = 0.8 * W
        if self.intrinsics_mode == "provided" and intrinsics_value:
            fx, fy, cx, cy = [float(v) for v in intrinsics_value]
            self.intrinsics_source, self.metric_scale = "provided", True
        elif self.intrinsics_mode in ("exif", "auto"):
            calib = self._calib_txt_intrinsics() if self.intrinsics_mode == "auto" else None
            if calib is not None:
                fx, fy, cx, cy = calib
                self.intrinsics_source = "calib.txt"
                self.metric_scale = True
            else:
                f = exif_focal_px(self._f0, W)
                if f:
                    fx = fy = f
                    cx, cy = W / 2.0, H / 2.0
                    self.intrinsics_source = "exif"
                    self.metric_scale = True
                else:
                    fy, cx, cy = fx, W / 2.0, H / 2.0
                    self.intrinsics_source = "default"
                    self.metric_scale = False
        else:                                   # none
            fy, cx, cy = fx, W / 2.0, H / 2.0
            self.intrinsics_source = "none"
        self._K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch
        arr = self.read_img(i)
        # (3, H, W) float in [0, 1], resized to image_size on the long side
        h, w = arr.shape[:2]
        scale = self.image_size / max(h, w) if self.image_size else 1.0
        if scale < 1.0:
            from PIL import Image
            arr = np.asarray(Image.fromarray(arr).resize(
                (max(int(w * scale), 1), max(int(h * scale), 1)), Image.BILINEAR))
        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return float(self.timestamps[i]), t

    def read_img(self, fid: int) -> np.ndarray:
        if fid not in self._imgs:
            from PIL import Image
            with Image.open(self.files[int(fid)]) as im:
                self._imgs[int(fid)] = np.asarray(im.convert("RGB"))
        return self._imgs[int(fid)]

    def frame_shape(self) -> Tuple[int, int]:
        H, W = self._hw
        if self.image_size:
            scale = float(self.image_size) / float(max(H, W))
            w_r = max(int(round(W * scale)), 1)
            h_r = max(int(round(H * scale)), 1)
            # Match MASt3R-SLAM resize_img multiple-of-16 crop
            w_f = ((w_r // 16) * 16) or w_r
            h_f = ((h_r // 16) * 16) or h_r
            return (h_f, w_f)
        return self._hw

    def get_img_shape(self):
        return self.frame_shape()

    def has_calib(self) -> bool:
        return bool(self.metric_scale)

    @property
    def K_orig(self) -> np.ndarray:
        return self._K

    @property
    def K_frame(self) -> np.ndarray:
        return self._K

    def K_slam_frame(self) -> np.ndarray:
        """Intrinsics scaled to the MASt3R resized frame shape."""
        H, W = self._hw
        hf, wf = self.frame_shape()
        sx, sy = wf / float(W), hf / float(H)
        K = self._K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def name(self) -> str:
        return f"rgb:{self.root.name}"

    @property
    def img_size(self):
        return (int(self.image_size), int(self.image_size))

    def report(self) -> dict:
        return {"n_frames": len(self), "root": str(self.root),
                "frame_shape": list(self._hw), "intrinsics_source": self.intrinsics_source,
                "metric_scale": bool(self.metric_scale),
                "scale_note": ("metric" if self.metric_scale else
                               "relative: cell/GSD are arbitrary units")}
