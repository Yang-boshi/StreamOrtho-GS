"""Frontends: a replay frontend (works from an observation table) and a live stub.

The live MASt3R-SLAM frontend would import ``mast3r_slam`` from ``third_party/`` and hand
over one ``FramePacket`` per tracked frame.  That package is not vendored in this checkout, so
the live path raises a clear error instead of silently falling back; every experiment in this
repository runs through the replay frontend, which is the same interface.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    kf: int
    frame_id: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    conf: np.ndarray
    depth: np.ndarray
    cos_inc: np.ndarray
    rgb: np.ndarray | None = None
    is_keyframe: bool = True
    #: present on the live path so the offline stage / carving / 3DGS can reuse
    T_WC: np.ndarray | None = None
    K: np.ndarray | None = None
    C: np.ndarray | None = None
    N: float = 1.0
    x_hw: tuple | None = None
    c_hw: tuple | None = None
    use_calib: bool | None = None
    img_u8: np.ndarray | None = None
    K_orig: np.ndarray | None = None
    feat: object | None = None
    pos: object | None = None
    shape_hw: tuple | None = None
    model: object | None = None


class ReplayFrontend:
    """Feeds the frames of an observation table in temporal order."""

    def __init__(self, obs_dir, max_frames: int = 0, cell: float = 0.45):
        from stream_ortho.eval import sync_eval as S
        d = np.load(pathlib.Path(obs_dir) / "obs_merged.npz")
        self.slices = S.frame_slices(d["kf"])
        self.kfs = sorted(self.slices)[: int(max_frames) if max_frames else None]
        self.d = d
        self.cell = cell
        self.n_frames = len(self.kfs)

    def __iter__(self):
        for k in self.kfs:
            idx = self.slices[k]
            yield FramePacket(kf=int(k), frame_id=int(self.d["frame_id"][idx][0]),
                              x=self.d["x"][idx].astype(np.float64),
                              y=self.d["y"][idx].astype(np.float64),
                              z=self.d["z"][idx].astype(np.float64),
                              conf=self.d["conf"][idx].astype(np.float64),
                              depth=self.d["D"][idx].astype(np.float64),
                              cos_inc=self.d["cos_inc"][idx].astype(np.float64),
                              rgb=(self.d["rgb"][idx].astype(np.float32)
                                   if "rgb" in self.d else None))

    def __len__(self):
        return len(self.kfs)


class LiveFrontend:
    """Run the upstream MASt3R-SLAM loop in-process and hand every keyframe to a callback.

    The loop is the one in ``scripts/run_mast3r_baseline`` (a faithful mirror of upstream's
    ``main.py``); this class only attaches a hook, so there is no fork and no second copy of
    the SLAM code.
    """

    def __init__(self, dataset_dir, out_dir, calib=False, max_frames=0, stride=1,
                 single_thread=False, hook=None, intrinsics="auto", intrinsics_value=None):
        # the SLAM loop changes the process cwd, so keep an absolute path
        import pathlib as _pl
        self.dataset_dir = str(_pl.Path(dataset_dir).resolve())
        self.out_dir = str(out_dir)
        self.calib = bool(calib)
        self.max_frames = int(max_frames)
        self.stride = int(stride)
        self.single_thread = bool(single_thread)
        self.hook = hook
        self.intrinsics = str(intrinsics)
        self.intrinsics_value = intrinsics_value
        self.packets = 0

    def _make_hook(self, callback):
        import numpy as np
        from stream_ortho.scripts.run_mast3r_baseline import _frame_pose

        def _hook(k, frame):
            Xg = frame.X_canon.detach().cpu().numpy()
            X = Xg.reshape(-1, 3).astype(np.float64)
            N = X.shape[0]
            # the pointmap grid must come from the frame, not from the array shape: on this
            # build X_canon is already (N, 3), so shape[:2] would give (N, 3) and the colours
            # would be resized to a 3-pixel column (the grey stripes)
            hx = wx = None
            for attr in ("img_true_shape", "img_shape"):
                if hasattr(frame, attr):
                    v = getattr(frame, attr)
                    try:
                        v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).reshape(-1)
                        if v.size >= 2:
                            hx, wx = int(v[0]), int(v[1])
                            break
                    except Exception:
                        continue
            if getattr(self, 'shp_dbg', 0) < 2:
                self.shp_dbg = getattr(self, 'shp_dbg', 0) + 1
                print(f"[kf-shape] Xg={Xg.shape} hw=({hx},{wx}) C={Cg.shape if 'Cg' in dir() else None}", flush=True)
            if hx is None or hx * wx != N:
                raise RuntimeError(f"cannot infer the pointmap grid: N={N} shape={Xg.shape} "
                                   f"hw=({hx},{wx})")

            Cg = frame.C.detach().cpu().numpy().astype(np.float32)
            if Cg.ndim == 3:
                Cg = Cg[..., 0]
            if Cg.size == N:
                C = Cg.reshape(-1).astype(np.float64)
            elif Cg.ndim == 2:
                import cv2
                C = cv2.resize(Cg, (wx, hx), interpolation=cv2.INTER_NEAREST).reshape(-1)
                C = C.astype(np.float64)
            else:
                raise RuntimeError(f"confidence shape {Cg.shape} cannot be aligned to N={N}")
            assert C.size == N

            img_full = None
            rgb = None
            try:
                img = self._reader.read_img(int(frame.frame_id))
                # the pointmap is inferred from the *remapped* image, so the colours must come
                # from the same remap; sampling the raw frame misaligns them (noise, no structure)
                remap = getattr(getattr(self._reader, "camera_intrinsics", None), "remap", None)
                if remap is not None:
                    img = remap(img)
                img_full = img
                import cv2 as _cv
                small = _cv.resize(img, (wx, hx), interpolation=_cv.INTER_AREA)
                rgb = np.asarray(small, np.float32).reshape(-1, 3)
                assert rgb.shape[0] == N
            except Exception as exc:
                self.rgb_errors = getattr(self, "rgb_errors", 0) + 1
                if self.rgb_errors <= 2:
                    print(f"[kf-colour] {exc!r}", flush=True)
                rgb = None

            from stream_ortho.scripts import run_mast3r_baseline as _RMB
            K_orig = getattr(self._reader, "K_orig", None)
            if K_orig is None and hasattr(self._reader, "K"):
                K_orig = self._reader.K
            fk = getattr(frame, "K", None)
            if fk is not None and hasattr(fk, "detach"):
                fk = fk.detach().cpu().numpy()
            packet = FramePacket(
                kf=int(k), frame_id=int(frame.frame_id), x=X[:, 0], y=X[:, 1], z=X[:, 2],
                conf=C, depth=X[:, 2], cos_inc=np.ones(N), is_keyframe=True, rgb=rgb,
                T_WC=_frame_pose(frame),
                K=np.asarray(fk, np.float64) if fk is not None else None,
                C=C, N=float(getattr(frame, "N", 1)),
                x_hw=(hx, wx), c_hw=tuple(int(v) for v in Cg.shape), use_calib=self.calib,
                img_u8=img_full,
                K_orig=(None if K_orig is None else np.asarray(K_orig, np.float64)),
                feat=getattr(frame, "feat", None),
                pos=getattr(frame, "pos", None),
                shape_hw=(hx, wx),
                model=_RMB.ACTIVE_MODEL)
            self.packets += 1
            callback(packet)
        return _hook

    def _open_reader(self):
        """A dataset used only to fetch RGB colours for the packets."""
        from stream_ortho.data.npu_dataset import NPUDroneMapDataset
        try:
            return NPUDroneMapDataset(self.dataset_dir, stride=self.stride, use_calib=self.calib)
        except Exception:
            from stream_ortho.data.rgb_folder import RGBFolderDataset
            return RGBFolderDataset(
                self.dataset_dir,
                intrinsics=self.intrinsics,
                intrinsics_value=self.intrinsics_value,
                stride=self.stride,
            )

    def run(self, callback, pose_callback=None):
        from stream_ortho.scripts import run_mast3r_baseline as R
        self._reader = self._open_reader()
        R.KEYFRAME_HOOK = self._make_hook(callback)
        R.POSE_UPDATE_HOOK = pose_callback
        import pathlib as _pl
        fmt = "npu" if (_pl.Path(self.dataset_dir) / "config.cfg").exists() else "rgb"
        argv = ["--dataset", self.dataset_dir, "--out", self.out_dir,
                "--dataset-format", fmt, "--intrinsics", self.intrinsics]
        if self.intrinsics_value:
            argv += ["--intrinsics-value", ",".join(str(v) for v in self.intrinsics_value)]
        if self.calib:
            argv.append("--calib")
        if self.max_frames:
            argv += ["--max-frames", str(self.max_frames)]
        if self.stride != 1:
            argv += ["--stride", str(self.stride)]
        if self.single_thread:
            argv.append("--single-thread")
        try:
            R.main(argv)
        finally:
            R.KEYFRAME_HOOK = None
            R.POSE_UPDATE_HOOK = None
            R.ACTIVE_MODEL = None
        return self.packets
