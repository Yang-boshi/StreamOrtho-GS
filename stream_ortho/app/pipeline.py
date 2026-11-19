"""Pipeline skeleton: the stages the full system runs, in order.

P0 only wires the data path (config -> dataset -> stage plan -> run record); the streaming
product, the per-frame filter and the FOV tiles are attached in P1/P2.  The stage list is
printed and written to ``run.json`` so a run is self-describing even before it produces maps.
"""
from __future__ import annotations

import collections
import contextlib
import json
import pathlib
import threading
import time

import numpy as np

from stream_ortho.app.config import Config


class _AsyncOrthoWorker:
    """Background worker decoupling SLAM tracking from 3DGS mapping/tile flush.

    * Keyframes are queued FIFO (bounded for backpressure) and drained BEFORE
      any pending BA pose updates so BA updates never overtake queued keyframes.
    * Pose updates arriving while the worker is busy coalesce in-place under
      the condition lock, eliminating redundant intermediate deduplications.
    """

    def __init__(self, on_kf, on_pose, max_queue: int = 64, ba_interval: int = 8):
        self._on_kf = on_kf
        self._on_pose = on_pose
        self._max_queue = max(int(max_queue), 1)
        self._ba_interval = max(int(ba_interval), 1)
        self._cv = threading.Condition()
        self._kf_q: collections.deque = collections.deque()
        self._pending_poses: dict[int, np.ndarray] = {}
        self._closed = False
        self._error: BaseException | None = None
        self._kfs_since_ba = 0
        self.coalesced_pose_batches = 0
        self.executed_pose_batches = 0
        self._thread = threading.Thread(target=self._run, name="Ortho3DGSWorker", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            import torch

            ctx = torch.inference_mode()
        except Exception:
            ctx = contextlib.nullcontext()
        with ctx:
            while True:
                pkt = None
                poses = None
                with self._cv:
                    while not self._kf_q and not self._pending_poses and not self._closed:
                        self._cv.wait()
                    if self._pending_poses and (not self._kf_q or self._kfs_since_ba >= self._ba_interval):
                        queued_ids = {int(p.kf) for p in self._kf_q}
                        ready_keys = [k for k in self._pending_poses if k not in queued_ids]
                        if ready_keys:
                            poses = {k: self._pending_poses.pop(k) for k in ready_keys}
                            self._kfs_since_ba = 0
                    if poses is None and self._kf_q:
                        pkt = self._kf_q.popleft()
                        if int(pkt.kf) in self._pending_poses:
                            pkt.T_WC = self._pending_poses.pop(int(pkt.kf))
                        self._kfs_since_ba += 1
                        self._cv.notify_all()
                    elif poses is None and not self._kf_q and not self._pending_poses and self._closed:
                        break
                if pkt is None and poses is None:
                    continue
                try:
                    if pkt is not None:
                        self._on_kf(pkt)
                    elif poses is not None:
                        self.executed_pose_batches += 1
                        self._on_pose(poses)
                except BaseException as exc:
                    with self._cv:
                        self._error = exc
                        self._closed = True
                        self._cv.notify_all()
                    break

    def submit_keyframe(self, pkt):
        with self._cv:
            if self._error is not None:
                raise self._error
            while len(self._kf_q) >= self._max_queue and not self._closed:
                self._cv.wait(timeout=0.05)
            if self._error is not None:
                raise self._error
            self._kf_q.append(pkt)
            self._cv.notify_all()

    def submit_pose_updates(self, pose_updates: dict):
        if not pose_updates:
            return
        with self._cv:
            if self._error is not None:
                return
            if self._pending_poses:
                self.coalesced_pose_batches += 1
            for k, v in pose_updates.items():
                self._pending_poses[int(k)] = np.asarray(v, dtype=np.float64).copy()
            self._cv.notify_all()

    def drain(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join()
        if self._error is not None:
            raise self._error


class Pipeline:
    def __init__(self, cfg: Config, out_dir):
        self.cfg = cfg
        # resolve: upstream changes the process cwd, and a relative output path would land
        # inside the SLAM checkout instead of the directory the user asked for
        self.out = pathlib.Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.stages = []
        self.t0 = time.time()

    # ------------------------------------------------------------------
    def build_dataset(self):
        from stream_ortho.data.rgb_folder import RGBFolderDataset
        c = self.cfg.data
        if c.config_cfg:
            from stream_ortho.data.npu_dataset import NPUDroneMapDataset
            self.dataset = NPUDroneMapDataset(c.config_cfg)
        else:
            self.dataset = RGBFolderDataset(
                c.images, image_size=c.image_size, stride=c.stride,
                intrinsics=c.intrinsics, intrinsics_value=c.intrinsics_value,
                max_frames=c.max_frames)
        return self.dataset

    def plan(self) -> list:
        c = self.cfg
        stages = [{"name": "data", "n_frames": len(self.dataset),
                   **self.dataset.report()},
                  {"name": "frontend", "backend": c.frontend.backend,
                   "use_calib": c.frontend.use_calib},
                  {"name": "filter", "enabled": c.filter.enabled,
                   "gate": c.filter.gate, "nfp": c.filter.nfp, "prune": c.filter.prune},
                  {"name": "gaussian", "enabled": c.gaussian.enabled,
                   "anchor": c.gaussian.anchor, "keyframes_only": c.gaussian.keyframes_only},
                  {"name": "map", "cell": c.map.cell, "z_gap": c.map.z_gap,
                   "freeze_owner": c.map.freeze_owner, "online_calib": c.map.online_calib,
                   "window": c.map.window},
                  {"name": "ortho", "gsd": c.ortho.gsd, "tile": c.ortho.tile,
                   "render": c.ortho.render, "fov_only": c.ortho.fov_only,
                   "budget_ms": c.ortho.budget_ms},
                  {"name": "offline", "enabled": c.offline.enabled, "iters": c.offline.iters,
                   "model": c.offline.model}]
        self.stages = stages
        return stages

    def run_streaming(self, obs_dir, frontend="replay", oracle=None) -> dict:
        """P1: replay the observation table through filter -> frozen insert -> dirty tiles."""
        import numpy as np
        from stream_ortho.app.callbacks import FrameCallbacks
        from stream_ortho.app.frontend import LiveFrontend, ReplayFrontend
        from stream_ortho.filter.pointmap import FilterConfig, PointmapFilter
        from stream_ortho.online.calib import OnlineCalib, OnlineCalibConfig
        from stream_ortho.online.product import StreamingProduct
        from stream_ortho.ortho.cache import TileCache
        from stream_ortho.ortho.grid import OrthoGrid
        from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig
        from stream_ortho.scripts.run_stream import tile_render_fn_factory

        c = self.cfg
        fe = (ReplayFrontend(obs_dir, max_frames=c.data.max_frames, cell=c.map.cell)
              if frontend == "replay" else LiveFrontend())
        d = np.load(pathlib.Path(obs_dir) / "obs_merged.npz")
        T_WP = np.asarray(d["T_WP"], np.float64)
        first = next(iter(fe))
        ox = int(np.floor(first.x.min() / c.map.cell))
        oy = int(np.floor(first.y.min() / c.map.cell))
        W = int(np.ceil((first.x.max() - first.x.min()) / c.map.cell)) + 2
        H = int(np.ceil((first.y.max() - first.y.min()) / c.map.cell)) + 2
        if oracle and pathlib.Path(oracle).exists():
            _op = np.load(oracle)
            ox, oy = int(_op["ox"]), int(_op["oy"])
            H, W = int(_op["z_ref"].shape[0]), int(_op["z_ref"].shape[1])
        product = StreamingProduct(c.map.cell, ox, oy, H, W, L=c.map.max_layers,
                                   z_gap=c.map.z_gap)
        calib = OnlineCalib(OnlineCalibConfig(min_frames=10)) if c.map.online_calib else None
        filt = PointmapFilter(FilterConfig(enabled=c.filter.enabled, gate=c.filter.gate,
                                           nfp=c.filter.nfp, prune=c.filter.prune,
                                           min_conf=c.filter.min_conf,
                                           iso_radius_m=c.filter.iso_radius_m,
                                           curtain_ratio=c.filter.curtain_ratio,
                                           max_prune_frac=c.filter.max_prune_frac,
                                           cell=c.map.cell,
                                           origin=(ox * c.map.cell, oy * c.map.cell),
                                           nfp_tau_m=c.filter.nfp_tau_m,
                                           nfp_min_obs=c.filter.nfp_min_obs,
                                           nfp_mode=c.filter.nfp_mode,
                                           nfp_top_pct=c.filter.nfp_top_pct,
                                           nfp_below_min_gap_m=c.filter.nfp_below_min_gap_m,
                                           protect_unique=c.filter.protect_unique,
                                           protect_low_obs=c.filter.protect_low_obs))
        grid = OrthoGrid(T_WP=T_WP, gsd=c.ortho.gsd, tile=c.ortho.tile)
        cache = TileCache()
        render_mode = "dsm" if c.ortho.render == "dsm" else "rgb"
        render_fn = tile_render_fn_factory(product, c.ortho.gsd, c.ortho.tile, mode=render_mode)
        engine = OrthoEngine(grid, cache, render_fn,
                             SchedulerConfig(tile=c.ortho.tile, budget_ms=c.ortho.budget_ms,
                                             max_tiles_per_flush=c.ortho.max_tiles))
        origin = (0.5 * (first.x.min() + first.x.max()), 0.5 * (first.y.min() + first.y.max()))
        z_ground = float(np.percentile(first.z[np.isfinite(first.z)], 5))
        writer = None
        fmt = "png" if c.ortho.dump_format in ("png",) else "jpg"
        if c.ortho.async_dump and c.ortho.dump_every:
            from stream_ortho.app.mosaic_writer import AsyncMosaicWriter
            writer = AsyncMosaicWriter(maxsize=max(1, int(c.ortho.dump_queue)),
                                       fmt=fmt, scale=float(c.ortho.dump_scale))
        cb = FrameCallbacks(product, engine, calib, filt, c.map.cell, origin, c.ortho.gsd,
                            z_ground=z_ground, out_dir=self.out,
                            dump_every=c.ortho.dump_every, tile_size=c.ortho.tile,
                            writer=writer, dump_fmt=fmt)
        for pkt in fe:
            cb.on_frame(pkt)
        engine.flush_all()
        # the cached tiles were rendered at different product versions, so stitching them
        # leaves seams and wrong roofs; re-render every covered tile from the FINAL product
        from stream_ortho.ortho.cache import TileCache as _TC
        from stream_ortho.ortho.mosaic import save_mosaic
        final_cache = _TC()
        for (i, j) in sorted(engine.covered):
            r, al, dep, _ = render_fn(grid, i, j)
            final_cache.put((i, j), r, al, dep, ms=0.0)
        final_path = self.out / f"mosaic_final.{fmt}"
        if writer is not None:
            from stream_ortho.app.mosaic_writer import snapshot_tiles
            writer.submit(final_path, snapshot_tiles(final_cache), final=True, scale=1.0)
            writer.close()                       # waits for the final to land
            final = {"path": str(final_path), "async": True,
                     "tiles": writer.stats()["written"]}
        else:
            final = save_mosaic(final_cache, final_path, tile_size=c.ortho.tile, feather=8)
        res = cb.summary(time.time() - self.t0)
        if final:
            res["final_mosaic"] = final
        z_top, own = product.top()
        live = np.isfinite(z_top)
        res["quality"] = {"live_cells": int(live.sum()),
                          "coverage_of_grid": float(live.mean()),
                          "product_mb": product.memory_mb()}
        if oracle and pathlib.Path(oracle).exists():
            from stream_ortho.scripts.diag.lcr_fixed_audit import shared2_pairs, top_surface
            from stream_ortho.scripts.diag.spatial_smooth import metrics as surf_metrics
            op = np.load(oracle)
            raw_n = op["n"].astype(np.float32)
            raw_z = op["z_ref"].astype(np.float32)
            Hc, Wc = min(H, raw_n.shape[0]), min(W, raw_n.shape[1])
            sums = {"sum_w": product.w_sum.reshape(H, W, product.L)[:Hc, :Wc],
                    "sum_wz": product.z_sum.reshape(H, W, product.L)[:Hc, :Wc],
                    "sum_wz2": np.zeros((Hc, Wc, product.L), np.float32)}
            sp = {**sums,
                  "n": product.n.reshape(H, W, product.L)[:Hc, :Wc],
                  "z_ref": (product.z_sum / np.maximum(product.w_sum, 1e-12)).reshape(
                      H, W, product.L)[:Hc, :Wc],
                  "owner": product.owner.reshape(H, W, product.L)[:Hc, :Wc],
                  "cell": c.map.cell, "H": Hc, "W": Wc, "L": product.L, "ox": ox, "oy": oy}
            raw_c = {"n": raw_n[:Hc, :Wc], "z_ref": raw_z[:Hc, :Wc],
                     "owner": op["owner"][:Hc, :Wc].astype(np.int32), "cell": c.map.cell,
                     "H": Hc, "W": Wc, "L": raw_n.shape[2], "ox": int(op["ox"]),
                     "oy": int(op["oy"])}
            masks = shared2_pairs(raw_c)
            z_top_raw, _ = top_surface(raw_c)
            lz = raw_z[raw_n > 0]
            iqr = float(np.percentile(lz, 75) - np.percentile(lz, 25)) if lz.size else 1.0
            m = surf_metrics(sp, masks, z_top_raw, iqr)
            res["quality"].update({k: float(m[k]) for k in
                                   ("lcr_fixed", "n_jump", "ra", "sei", "coverage_cells",
                                    "z_iqr_ratio")})
        res.update({"status": "p1_streaming", "frontend": frontend,
                    "obs": str(obs_dir), "metric_scale": None,
                    "product_mb": product.memory_mb(),
                    "tiles": {"rendered": len(engine.rendered),
                              "covered": len(engine.covered)},
                    "config": self.cfg.to_dict()})
        if getattr(c, "offline", None) is not None and c.offline.enabled and obs_dir:
            from stream_ortho.app.offline import run_offline
            try:
                res["offline"] = run_offline(obs_dir, self.out, seq=c.data.seq,
                                             iters=c.offline.iters, model=c.offline.model,
                                             lam_p=c.offline.lam_p,
                                             texture=c.offline.render_texture,
                                             apply_filter=c.offline.filter,
                                             apply_carve=c.offline.carve,
                                             cell_hint=c.map.cell)
            except Exception as exc:          # the preview must survive a failed refine
                res["offline_error"] = repr(exc)
        (self.out / "filter_stats.json").write_text(
            json.dumps(res["filter"], indent=2, ensure_ascii=False))
        (self.out / "stream.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        self.cfg.save(self.out / "config_resolved.yaml")
        return res

    def _resolve_use_calib(self) -> bool:
        c = self.cfg
        if c.frontend.use_calib == "true":
            if c.data.intrinsics not in ("provided", "exif") and not (
                pathlib.Path(c.data.images) / "config.cfg"
            ).exists():
                raise RuntimeError("--calib needs intrinsics; pass --intrinsics provided or use --no-calib")
            return True
        if c.frontend.use_calib == "false":
            return False
        # "auto": enable calibration whenever dataset has config.cfg, provided K, or EXIF focal
        if (pathlib.Path(c.data.images) / "config.cfg").exists():
            return True
        if c.data.intrinsics in ("provided", "exif"):
            return True
        if c.data.intrinsics == "auto" and c.data.images:
            try:
                from stream_ortho.data.rgb_folder import RGBFolderDataset
                ds = RGBFolderDataset(c.data.images, intrinsics="auto", load_images=False)
                return bool(ds.has_calib())
            except Exception:
                return False
        return False

    def _run_e2e_streaming_3dgs(self, use_calib: bool) -> dict:
        """True-streaming feed-forward 3DGS orthomapping (per-keyframe + BA pose sync)."""
        from PIL import Image
        from stream_ortho.adapters.splatt3r import lift_keyframe_gaussians
        from stream_ortho.app.alignment import ProgressivePlaneAligner
        from stream_ortho.app.artifacts import write_update_video
        from stream_ortho.app.frontend import LiveFrontend
        from stream_ortho.app.mosaic_writer import AsyncMosaicWriter, snapshot_tiles, stitch_arrays
        from stream_ortho.frontend.gaussian_head import load_gaussian_head
        from stream_ortho.frontend.predict import load_model
        from stream_ortho.geometry import quat_rotate_batch_torch, sim3_from_vec8
        from stream_ortho.online.gs_map import StreamingGaussianMap
        from stream_ortho.ortho.cache import TileCache
        from stream_ortho.ortho.grid import OrthoGrid
        from stream_ortho.ortho.gsplat_ortho import make_streaming_tile_render_fn
        from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig
        from stream_ortho import config as zcfg

        c = self.cfg
        aligner = ProgressivePlaneAligner(
            mode=getattr(c.data, "plane_mode", "auto"),
            bootstrap_kfs=max(1, int(getattr(c.map, "bootstrap_kfs", 1))),
            T_WP=None,
        )
        fmt = "png" if c.ortho.dump_format in ("png",) else "jpg"
        dump_every = int(getattr(c.ortho, "dump_every", 1) or 0)
        dump_scale = float(getattr(c.ortho, "dump_scale", 1.0))
        viewer = None
        if bool(getattr(c.ortho, "show", False)):
            from stream_ortho.app.viewer import LiveOrthoViewer
            viewer = LiveOrthoViewer(window_name=f"Ortho-3DGS Live ({self.out.name})")

        state = {
            "n": 0,
            "skipped": 0,
            "skipped_scale": 0,
            "boot_pkts": [],
            "sim3_scales": [],
            "cam_heights": [],
            "global_relevels": 0,
            "head": None,
            "model": None,
            "n_tiles": 0,
            "frames": [],
            "frame_tile_bounds": [],
            "viewer": viewer,
            "kf_diag": [],
            "ba_updates": [],
            "kf_cache": {},
            "kf_bboxes": {},
            "align_ms": [],
            "quat_rot_ms": [],
            "ins_only_ms": [],
            "lift_only_ms": [],
            "ba_ms": [],
        }

        def _world_512(pkt):
            R, t, sc = sim3_from_vec8(np.asarray(pkt.T_WC, np.float64))
            P = np.stack([pkt.x, pkt.y, pkt.z], axis=1).astype(np.float64)
            return float(sc) * (P @ R.T) + t, float(sc), t

        def _init_streaming(pkt, Pw):
            Pp0 = aligner.to_plane(Pw)
            ext = float(max(Pp0[:, 0].ptp(), Pp0[:, 1].ptp(), 1e-3))
            _, _, sc = sim3_from_vec8(np.asarray(pkt.T_WC, np.float64))
            med_z = float(np.median(np.asarray(pkt.z, np.float64)))
            if float(c.gaussian.gsd_3dgs) > 0.0:
                gsd = float(c.gaussian.gsd_3dgs)
            else:
                if pkt.K_orig is not None and float(pkt.K_orig[0, 0]) > 1.0:
                    fx0 = float(pkt.K_orig[0, 0])
                elif pkt.img_u8 is not None:
                    w0 = float(pkt.img_u8.shape[1])
                    fx0 = 0.8 * w0
                else:
                    fx0 = 0.8 * 1920.0
                native_gsd = (med_z / max(fx0, 10.0)) * max(float(sc), 1e-6) * float(c.gaussian.gsd_mul)
                # Safety floor so huge flights do not exceed max_canvas_px
                gsd = max(native_gsd, ext / max(float(c.gaussian.max_canvas_px) / 4.0, 256.0))

            tile_px = int(c.ortho.tile)
            halo = int(getattr(c.gaussian, "tile_halo", 16))
            voxel_size = float(getattr(c.gaussian, "voxel_gsd_mul", 2.0)) * gsd
            z_band = max(voxel_size * 10.0, 0.04, 0.025 * med_z * max(float(sc), 1e-6))
            gs_map = StreamingGaussianMap(
                voxel_size=voxel_size,
                tile_world_size=tile_px * gsd,
                nadir_replace_margin=float(getattr(c.gaussian, "nadir_replace_margin", 0.08)),
                z_band=z_band,
            )
            grid = OrthoGrid(T_WP=np.eye(4), gsd=gsd, tile=tile_px, halo=halo)
            render_fn = make_streaming_tile_render_fn(
                gs_map, gsd=gsd, tile_size=tile_px, halo=halo, device=zcfg.DEVICE
            )
            engine = OrthoEngine(
                grid,
                TileCache(),
                render_fn,
                SchedulerConfig(
                    tile=tile_px,
                    margin_tiles=0,
                    priority="newest",
                    budget_ms=float(c.ortho.budget_ms),
                    max_tiles_per_flush=int(c.ortho.max_tiles),
                ),
            )
            writer = None
            if dump_every > 0 and c.ortho.async_dump:
                writer = AsyncMosaicWriter(
                    maxsize=max(1, int(c.ortho.dump_queue)),
                    fmt=fmt,
                    scale=dump_scale,
                )
            state.update(
                gs_map=gs_map,
                grid=grid,
                engine=engine,
                writer=writer,
                gsd=gsd,
                voxel_size=voxel_size,
                relative_scale=(not use_calib),
            )
            print(
                f"[e2e-3dgs] initialized streaming 3DGS: gsd={gsd:.5f}, "
                f"voxel={voxel_size:.5f}, tile={tile_px}px, use_calib={use_calib}",
                flush=True,
            )

        def _cache_kf_pkt(pkt, img_u8, X_canon, hx, wx):
            import cv2
            from types import SimpleNamespace

            ok_jpg, buf = cv2.imencode(".jpg", img_u8[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            feat_cpu = pkt.feat.detach().cpu() if hasattr(pkt.feat, "detach") else pkt.feat
            pos_cpu = pkt.pos.detach().cpu() if hasattr(pkt.pos, "detach") else pkt.pos
            state["kf_cache"][int(pkt.kf)] = SimpleNamespace(
                kf=int(pkt.kf),
                frame_id=int(pkt.frame_id),
                img_jpg=(buf if ok_jpg else None),
                img_u8=(None if ok_jpg else img_u8),
                X_canon=np.asarray(X_canon, np.float32),
                C=(None if pkt.C is None else np.asarray(pkt.C, np.float32).reshape(hx, wx)),
                x_hw=(int(hx), int(wx)),
                K_orig=pkt.K_orig,
                feat=feat_cpu,
                pos=pos_cpu,
                shape_hw=pkt.shape_hw,
                T_WC=np.asarray(pkt.T_WC, np.float64).copy(),
            )

        def _process_one_kf(pkt, reproject_only: bool = False):
            import cv2
            import torch

            t_kf0 = time.perf_counter()
            if state["model"] is None:
                state["model"] = pkt.model if pkt.model is not None else load_model(device=zcfg.DEVICE)
            if state["head"] is None:
                ckpt = c.gaussian.head_ckpt or zcfg.SPLATT3R_HEAD_CKPT
                state["head"] = load_gaussian_head(state["model"], ckpt, device=zcfg.DEVICE, verbose=False)

            hx, wx = pkt.x_hw
            if hasattr(pkt, "X_canon"):
                X_canon = pkt.X_canon
            else:
                X_canon = np.stack([pkt.x, pkt.y, pkt.z], axis=1).reshape(hx, wx, 3)

            R_chk, _, sc = sim3_from_vec8(np.asarray(pkt.T_WC, np.float64))
            if not reproject_only:
                # Reject diverged tracking frames (flipped camera optical axis or abrupt height/scale jump)
                fwd_pz = float(aligner.T_PW[2, :3] @ R_chk[:, 2])
                zc_arr = np.asarray(X_canon[..., 2], dtype=np.float64)
                pos_z = zc_arr[np.isfinite(zc_arr) & (zc_arr > 0.0)]
                med_zc = float(np.median(pos_z)) if pos_z.size else 1.0
                h_cam = float(sc) * max(med_zc, 1e-3)
                if fwd_pz > -0.45:
                    state["skipped_scale"] += 1
                    return
                if len(state["sim3_scales"]) >= 5:
                    med_s = float(np.median(state["sim3_scales"][-10:]))
                    med_h = float(np.median(state["cam_heights"][-10:]))
                    n_obs = float(getattr(pkt, "N", 4.0) or 4.0)
                    h_lo = 0.50 if n_obs <= 2.5 else 0.35
                    if (
                        sc < 0.35 * med_s
                        or sc > 3.0 * med_s
                        or h_cam < h_lo * med_h
                        or h_cam > 2.8 * med_h
                    ):
                        state["skipped_scale"] += 1
                        return
                state["sim3_scales"].append(float(sc))
                state["cam_heights"].append(float(h_cam))
            img_u8 = getattr(pkt, "img_u8", None)
            if img_u8 is None and getattr(pkt, "img_jpg", None) is not None:
                bgr = cv2.imdecode(pkt.img_jpg, cv2.IMREAD_COLOR)
                img_u8 = bgr[:, :, ::-1].copy()
            if img_u8 is None:
                rgb_arr = np.asarray(pkt.rgb, np.float32).reshape(hx, wx, 3)
                img_u8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
            if not reproject_only:
                _cache_kf_pkt(pkt, img_u8, X_canon, hx, wx)

            t_lift0 = time.perf_counter()
            slam_conf_arr = None
            if pkt.C is not None:
                slam_conf_arr = np.asarray(pkt.C, np.float32).reshape(hx, wx)
            feat_dev = pkt.feat.to(zcfg.DEVICE) if hasattr(pkt.feat, "to") else pkt.feat
            pos_dev = pkt.pos.to(zcfg.DEVICE) if hasattr(pkt.pos, "to") else pkt.pos
            g = lift_keyframe_gaussians(
                img_u8=img_u8,
                X_canon=X_canon,
                sim3_vec=np.asarray(pkt.T_WC, np.float64),
                keyframe_id=int(pkt.kf),
                feat=feat_dev,
                pos=pos_dev,
                shape_hw=pkt.shape_hw,
                K_orig=pkt.K_orig if use_calib else None,
                model=state["model"],
                head=state["head"],
                conf_threshold=float(c.gaussian.conf_threshold),
                adaptive_stride=bool(getattr(c.gaussian, "adaptive_stride", True)),
                flat_stride=int(getattr(c.gaussian, "flat_stride", 2)),
                edge_grad_pct=float(getattr(c.gaussian, "edge_grad_pct", 70.0)),
                depth_upsample=str(getattr(c.gaussian, "depth_upsample", "edge")),
                edge_rel_tau=float(getattr(c.gaussian, "edge_rel_tau", 0.03)),
                edge_reject=int(getattr(c.gaussian, "edge_reject", 1)),
                edge_pct=float(getattr(c.gaussian, "edge_pct", 95.0)),
                slam_conf=slam_conf_arr,
                device=zcfg.DEVICE,
            )
            del feat_dev, pos_dev

            t_lift_ms = (time.perf_counter() - t_lift0) * 1000.0
            if g.means.shape[0] == 0:
                return

            n_lifted = int(g.means.shape[0])
            dev = g.means.device
            means_w_t = g.means.detach()
            conf_t = g.confidence.detach().reshape(-1)
            align_step = max(1, n_lifted // 16384)

            if reproject_only:
                T_PW_kf = aligner.rect_history.get(int(pkt.kf), aligner.T_PW)
                t_align_ms = 0.0
                rect_info = {"tilt_deg": 0.0, "dz": 0.0, "inlier_frac": 0.0, "rectified": False}
            else:
                R_init, t_init, sc_init = sim3_from_vec8(np.asarray(pkt.T_WC, np.float64))
                canon_w = float(sc_init) * (X_canon.reshape(-1, 3).astype(np.float64) @ R_init.T) + t_init
                Pp_canon_raw = aligner.to_plane(canon_w)

                # Subsample on GPU for plane alignment (rectify_keyframe subsamples <=4096 anyway)
                means_sub_np = means_w_t[::align_step].cpu().numpy().astype(np.float64)
                conf_sub_np = conf_t[::align_step].cpu().numpy().astype(np.float64)
                Pp_lifted_raw_sub = aligner.to_plane(means_sub_np)
                T_PW_base = aligner.T_PW
                R_base_t = torch.as_tensor(T_PW_base[:3, :3], dtype=torch.float32, device=dev)
                t_base_t = torch.as_tensor(T_PW_base[:3, 3], dtype=torch.float32, device=dev)
                z_raw_t = (means_w_t @ R_base_t[2].unsqueeze(-1)).squeeze(-1) + t_base_t[2]
                imax_raw = int(torch.argmax(z_raw_t).item())
                raw_max_xy_z = (means_w_t[imax_raw:imax_raw + 1] @ R_base_t.T + t_base_t)[0].cpu().numpy()

                t_align0 = time.perf_counter()
                _, T_PW_kf, rect_info = aligner.rectify_keyframe(
                    pkt.kf,
                    means_sub_np,
                    conf=conf_sub_np,
                    surface_ref=state["gs_map"],
                    enabled=bool(getattr(c.map, "progressive_align", True)),
                    return_points=False,
                )
                t_align_ms = (time.perf_counter() - t_align0) * 1000.0
                if rect_info.get("tie_rejected", False):
                    state["skipped_scale"] += 1
                    state["kf_cache"].pop(int(pkt.kf), None)
                    return

            t_q0 = time.perf_counter()
            R_PW = T_PW_kf[:3, :3]
            R_pw_t = torch.as_tensor(R_PW, dtype=torch.float32, device=dev)
            t_pw_t = torch.as_tensor(T_PW_kf[:3, 3], dtype=torch.float32, device=dev)
            means_p_t = means_w_t @ R_pw_t.T + t_pw_t
            quats_p_t = quat_rotate_batch_torch(g.quats.detach(), R_PW)
            cam_rays_p_t = g.cam_rays_w.detach() @ R_pw_t.T

            # Ray-constrained roof elevation tie-in: align residual height bias on
            # tall roofs along exact viewing rays to avoid lateral parallax steps.
            if (
                bool(getattr(c.map, "progressive_align", True))
                and state["gs_map"].n_points > 0
            ):
                elev_cur_t = means_p_t[:, 2] > 0.12
                n_elev = int(elev_cur_t.sum().item())
                if n_elev >= 200:
                    elev_step = max(1, n_elev // 16384)
                    elev_sub_np = means_p_t[elev_cur_t][::elev_step].cpu().numpy()
                    z_ref_e, hit_e = state["gs_map"].query_elevation(
                        elev_sub_np[:, 0], elev_sub_np[:, 1]
                    )
                    ov_e = hit_e & (z_ref_e > 0.12) & (np.abs(z_ref_e - elev_sub_np[:, 2]) < 0.18)
                    min_ov = max(16, 200 // elev_step)
                    if int(ov_e.sum()) >= min_ov:
                        diffs_e = z_ref_e[ov_e] - elev_sub_np[ov_e, 2]
                        dz_roof = float(np.median(diffs_e))
                        mad_e = float(np.median(np.abs(diffs_e - dz_roof)))
                        if mad_e < 0.06 and 0.003 < abs(dz_roof) < 0.16:
                            w_elev_t = ((means_p_t[:, 2] - 0.06) / 0.08).clamp(0.0, 1.0)
                            rz_p_t = cam_rays_p_t[:, 2].clamp(max=-0.35)
                            ray_step_t = (float(dz_roof) * w_elev_t) / rz_p_t
                            means_p_t = means_p_t + ray_step_t.unsqueeze(-1) * cam_rays_p_t

            t_q_ms = (time.perf_counter() - t_q0) * 1000.0

            t_ins0 = time.perf_counter()
            p_min = means_p_t[:, :2].min(dim=0).values.cpu().numpy()
            p_max = means_p_t[:, :2].max(dim=0).values.cpu().numpy()
            fb_pre = (float(p_min[0]), float(p_min[1]), float(p_max[0]), float(p_max[1]))

            carve_info = {"checked": 0, "voted": 0, "carved": 0}
            if (
                not reproject_only
                and g.depth_map is not None
                and g.K_depth is not None
                and state["gs_map"].n_points > 0
            ):
                carve_tol = max(float(state["gs_map"].z_band) * 2.0, 0.10)
                carve_info = state["gs_map"].carve_from_depth(
                    kf_id=int(pkt.kf),
                    depth_map=g.depth_map,
                    K_depth=g.K_depth,
                    T_WC=np.asarray(pkt.T_WC, np.float64),
                    T_PW=T_PW_kf,
                    bbox_p=fb_pre,
                    carve_tol=carve_tol,
                    min_votes=2,
                    min_cos_view=0.85,
                    device=zcfg.DEVICE,
                )
                state["total_carved"] = state.get("total_carved", 0) + int(carve_info["carved"])

            scales_p_t = g.scales.detach().clone()
            min_s_xy = 0.85 * float(state["gs_map"].cell_2d)
            scales_p_t[:, :2] = scales_p_t[:, :2].clamp_min(min_s_xy)

            if not reproject_only:
                imax_rect = int(torch.argmax(means_p_t[:, 2]).item())
                rect_max_xyz = means_p_t[imax_rect].cpu().numpy()
                rect_z_p15 = float(np.percentile(means_p_t[::align_step, 2].cpu().numpy(), 15))

            ins_info = state["gs_map"].ingest_keyframe(
                kf_id=int(pkt.kf),
                means_p=means_p_t,
                quats_p=quats_p_t,
                scales_p=scales_p_t,
                opacity=g.opacity.detach().reshape(-1),
                sh_dc=g.sh_dc.detach(),
                nadir_score=g.nadir_score.detach(),
                conf=conf_t,
                T_WC=np.asarray(pkt.T_WC, np.float64),
                T_PW=T_PW_kf,
                cam_rays_p=cam_rays_p_t,
            )
            # Release large transient GPU tensors before gsplat tile rasterization
            del g, means_w_t, conf_t, means_p_t, quats_p_t, cam_rays_p_t, scales_p_t
            state["kf_bboxes"][int(pkt.kf)] = fb_pre
            if reproject_only:
                halo_m = float(getattr(c.gaussian, "tile_halo", 16)) * float(state["gsd"])
                state["engine"].mark_footprint(
                    (fb_pre[0] - halo_m, fb_pre[1] - halo_m, fb_pre[2] + halo_m, fb_pre[3] + halo_m),
                    kf_id=int(pkt.kf),
                    reason="ba_heal",
                )
                return

            t_ins_ms = (time.perf_counter() - t_ins0) * 1000.0
            state.setdefault("insert_ms", []).append(t_lift_ms + t_align_ms + t_q_ms + t_ins_ms)
            state["lift_only_ms"].append(t_lift_ms)
            state["align_ms"].append(t_align_ms)
            state["quat_rot_ms"].append(t_q_ms)
            state["ins_only_ms"].append(t_ins_ms)

            state["kf_diag"].append({
                "kf": int(pkt.kf),
                "frame_id": int(pkt.frame_id),
                "n_pts_lifted": n_lifted,
                "sc_init": float(sc_init),
                "t_wc_init": [float(v) for v in t_init],
                "tilt_deg": float(rect_info["tilt_deg"]),
                "dz": float(rect_info["dz"]),
                "inlier_frac": float(rect_info["inlier_frac"]),
                "rectified": bool(rect_info["rectified"]),
                "canon_raw_z_p15": float(np.percentile(Pp_canon_raw[:, 2], 15)),
                "canon_raw_z_max": float(Pp_canon_raw[:, 2].max()),
                "lifted_raw_z_p15": float(np.percentile(Pp_lifted_raw_sub[:, 2], 15)),
                "lifted_raw_z_max": float(raw_max_xy_z[2]),
                "lifted_raw_max_xy": [float(raw_max_xy_z[0]), float(raw_max_xy_z[1])],
                "lifted_rect_z_p15": rect_z_p15,
                "lifted_rect_z_max": float(rect_max_xyz[2]),
                "lifted_rect_max_xy": [float(rect_max_xyz[0]), float(rect_max_xyz[1])],
                "added": int(ins_info["added"]),
                "replaced": int(ins_info["replaced"]),
                "fused": int(ins_info.get("fused", 0)),
                "carved": int(carve_info.get("carved", 0)),
                "carve_voted": int(carve_info.get("voted", 0)),
                "lift_ms": float(t_lift_ms),
                "align_ms": float(t_align_ms),
                "quat_rot_ms": float(t_q_ms),
                "ins_ms": float(t_ins_ms),
            })

            halo_m = float(getattr(c.gaussian, "tile_halo", 16)) * float(state["gsd"])
            fb = (
                fb_pre[0] - halo_m, fb_pre[1] - halo_m,
                fb_pre[2] + halo_m, fb_pre[3] + halo_m,
            )
            t_tile0 = time.perf_counter()
            state["engine"].mark_footprint(fb, kf_id=int(pkt.kf))

            # Once wide 2D multi-strip trajectory coverage is achieved, re-level global T_PW
            # cleanly so the whole map rests flat on z=0 (removes bootstrap plane tilt).
            if (
                bool(getattr(c.map, "progressive_align", True))
                and state["global_relevels"] < 2
                and ((state["global_relevels"] == 0 and state["n"] >= 12) or state["n"] == 48)
            ):
                cam_pts = []
                for kf_id_c, T_wc_c in state["gs_map"].kf_T_WC.items():
                    T_pw_c = state["gs_map"].kf_T_PW.get(kf_id_c, aligner.T_PW)
                    _, t_wc_c, _ = sim3_from_vec8(T_wc_c)
                    cam_pts.append(T_pw_c[:3, :3] @ t_wc_c + T_pw_c[:3, 3])
                cam_pos_p = np.asarray(cam_pts, dtype=np.float64) if cam_pts else None
                min_t = 0.5 if state["global_relevels"] == 0 else 1.0
                dT_global = aligner.estimate_global_relevel(
                    state["gs_map"].means, min_tilt_deg=min_t, cam_pos_p=cam_pos_p
                )
                if dT_global is not None:
                    state["gs_map"].apply_plane_transform(dT_global)
                    aligner.apply_global_plane_transform(dT_global, map_pts_p=state["gs_map"].means)
                    eng = state["engine"]
                    eng.cache.tiles.clear()
                    eng.covered.clear()
                    eng.footprints.clear()
                    R_g2 = dT_global[:2, :2]
                    t_g2 = dT_global[:2, 3]
                    for kb_k, (bx0, by0, bx1, by1) in list(state["kf_bboxes"].items()):
                        cs = np.array([[bx0, by0], [bx0, by1], [bx1, by0], [bx1, by1]], dtype=np.float64)
                        cs2 = cs @ R_g2.T + t_g2
                        nb = (
                            float(cs2[:, 0].min()), float(cs2[:, 1].min()),
                            float(cs2[:, 0].max()), float(cs2[:, 1].max()),
                        )
                        state["kf_bboxes"][kb_k] = nb
                        eng.mark_footprint(
                            (nb[0] - halo_m, nb[1] - halo_m, nb[2] + halo_m, nb[3] + halo_m),
                            kf_id=int(kb_k),
                            reason="global_relevel",
                        )
                    state["global_relevels"] += 1

            fl = state["engine"].flush()
            if torch.cuda.is_available() and (
                torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 256 * 1024 * 1024
            ):
                torch.cuda.empty_cache()
            t_tile_ms = (time.perf_counter() - t_tile0) * 1000.0
            state.setdefault("tile_ms", []).append(t_tile_ms)
            state["n_tiles"] += int(fl.get("rendered", 0))

            t_dump0 = time.perf_counter()
            tiles_snap = None
            need_dump = dump_every > 0 and (state["n"] % dump_every == 0)
            need_view = state["viewer"] is not None and state["viewer"].active
            if need_dump or need_view:
                tiles_snap = snapshot_tiles(state["engine"].cache)
            if tiles_snap and need_view:
                state["viewer"].submit(
                    tiles_snap,
                    f"kf={pkt.kf} ({state['n'] + 1}) | gs={state['gs_map'].n_points:,} | tiles={len(tiles_snap)}",
                )
            if need_dump:
                fpath = self.out / "frames" / f"mosaic_{state['n']:05d}.{fmt}"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                if tiles_snap:
                    imin = min(k[0] for k, _, _ in tiles_snap)
                    imax = max(k[0] for k, _, _ in tiles_snap)
                    jmin = min(k[1] for k, _, _ in tiles_snap)
                    jmax = max(k[1] for k, _, _ in tiles_snap)
                    state["frame_tile_bounds"].append((imin, imax, jmin, jmax))
                    state["frames"].append(str(fpath))
                    if state["writer"] is not None:
                        state["writer"].submit(fpath, tiles_snap, final=False, scale=dump_scale)
                    else:
                        from stream_ortho.app.mosaic_writer import _save_rgba
                        rgb_s, a_s = stitch_arrays(tiles_snap)
                        if rgb_s is not None:
                            _save_rgba(fpath, rgb_s, a_s, fmt, dump_scale)
            state.setdefault("dump_ms", []).append((time.perf_counter() - t_dump0) * 1000.0)

            if state["n"] < 5 or state["n"] % 10 == 0:
                print(
                    f"[e2e-3dgs] kf={pkt.kf:3d} added={ins_info['added']:6d} "
                    f"rep={ins_info['replaced']:5d} total={state['gs_map'].n_points:7d} "
                    f"tilt={rect_info['tilt_deg']:.2f}° dz={rect_info['dz']:+.3f} "
                    f"lift={t_lift_ms:.0f}ms ins={t_ins_ms:.0f}ms tile={t_tile_ms:.0f}ms",
                    flush=True,
                )
            state["n"] += 1

        def on_keyframe(pkt):
            Pw, _, cam_t = _world_512(pkt)
            if "gs_map" not in state:
                state["boot_pkts"].append(pkt)
                if not aligner.add(Pw, cam_pos=cam_t):
                    state["skipped"] += 1
                    return
                _init_streaming(pkt, Pw)
                for bp in state["boot_pkts"]:
                    _process_one_kf(bp)
                state["boot_pkts"] = []
            else:
                _process_one_kf(pkt)

        def on_pose_update(pose_updates):
            if not pose_updates:
                return
            if "gs_map" not in state:
                for bp in state["boot_pkts"]:
                    if int(bp.kf) in pose_updates:
                        bp.T_WC = pose_updates[int(bp.kf)]
                return
            t_ba0 = time.perf_counter()
            pre_stats = {}
            new_T_PW_map = {}
            prog_en = bool(getattr(c.map, "progressive_align", True))
            for k_u, T_new in pose_updates.items():
                k_i = int(k_u)
                T_old = state["gs_map"].kf_T_WC.get(k_i)
                if T_old is None:
                    continue
                T_new_arr = np.asarray(T_new, np.float64)
                T_pw_new = aligner.realign_after_ba(
                    k_i, T_old, T_new_arr, enabled=prog_en, surface_ref=state["gs_map"]
                )
                if T_pw_new is not None:
                    new_T_PW_map[k_i] = T_pw_new
                sel_u = state["gs_map"].owner_kf == k_i
                if not sel_u.any():
                    continue
                _, t_old, s_old = sim3_from_vec8(T_old)
                _, t_new, s_new = sim3_from_vec8(T_new_arr)
                zu = state["gs_map"].means[sel_u, 2]
                pre_stats[k_i] = {
                    "s_old": float(s_old),
                    "s_new": float(s_new),
                    "ds": float(s_new / max(float(s_old), 1e-9)),
                    "dt_w": float(np.linalg.norm(t_new - t_old)),
                    "z_p15_before": float(np.percentile(zu, 15)),
                    "z_max_before": float(zu.max()),
                }

            dirty_boxes = state["gs_map"].update_poses(
                pose_updates, new_T_PW=(new_T_PW_map or None)
            )
            if state["gs_map"].last_max_shift_px >= 3.0 and dirty_boxes:
                # Heal Voronoi seam cracks opened when BA pulls adjacent keyframes apart
                overlaps = []
                for k_c, kb in state["kf_bboxes"].items():
                    area = 0.0
                    for bb in dirty_boxes:
                        ix0, iy0 = max(kb[0], bb[0]), max(kb[1], bb[1])
                        ix1, iy1 = min(kb[2], bb[2]), min(kb[3], bb[3])
                        if ix1 > ix0 and iy1 > iy0:
                            area += (ix1 - ix0) * (iy1 - iy0)
                    if area > 0.0 and k_c in state["kf_cache"]:
                        overlaps.append((area, k_c))
                overlaps.sort(reverse=True)
                for _, k_c in overlaps[:16]:
                    cpkt = state["kf_cache"][k_c]
                    latest_T = state["gs_map"].kf_T_WC.get(k_c)
                    if latest_T is not None:
                        cpkt.T_WC = latest_T.copy()
                    _process_one_kf(cpkt, reproject_only=True)

            for k_i, ps in pre_stats.items():
                sel_a = state["gs_map"].owner_kf == k_i
                if sel_a.any():
                    pts_a = state["gs_map"].means[sel_a]
                    za = pts_a[:, 2]
                    res_r = (
                        aligner.estimate_residual_rect(pts_a)
                        if pts_a.shape[0] >= 200
                        else {"res_tilt_deg": 0.0, "res_dz": 0.0, "inlier_frac": 0.0}
                    )
                    ps.update({
                        "kf": k_i,
                        "n_pts_after": int(sel_a.sum()),
                        "z_p15_after": float(np.percentile(za, 15)),
                        "z_max_after": float(za.max()),
                        "d_z_p15": float(np.percentile(za, 15)) - ps["z_p15_before"],
                        "d_z_max": float(za.max()) - ps["z_max_before"],
                        "post_ba_res_tilt_deg": float(res_r["res_tilt_deg"]),
                        "post_ba_res_dz": float(res_r["res_dz"]),
                    })
                    state["ba_updates"].append(ps)
            state["ba_ms"].append((time.perf_counter() - t_ba0) * 1000.0)
            halo_m = float(getattr(c.gaussian, "tile_halo", 16)) * float(state["gsd"])
            for bb in dirty_boxes:
                state["engine"].mark_footprint(
                    (bb[0] - halo_m, bb[1] - halo_m, bb[2] + halo_m, bb[3] + halo_m),
                    reason="ba_sync",
                )
            if dirty_boxes:
                fl = state["engine"].flush()
                state["n_tiles"] += int(fl.get("rendered", 0))
                if state["viewer"] is not None and state["viewer"].active:
                    snap_ba = snapshot_tiles(state["engine"].cache)
                    if snap_ba:
                        state["viewer"].submit(
                            snap_ba,
                            f"BA sync | gs={state['gs_map'].n_points:,} | tiles={len(snap_ba)}",
                        )

        fe = LiveFrontend(
            c.data.images,
            self.out,
            calib=use_calib,
            max_frames=c.data.max_frames,
            stride=c.data.stride,
            single_thread=c.frontend.single_thread,
            intrinsics=c.data.intrinsics,
            intrinsics_value=c.data.intrinsics_value,
        )
        use_async_worker = bool(getattr(c.ortho, "async_worker", True))
        worker = (
            _AsyncOrthoWorker(on_keyframe, on_pose_update, max_queue=64, ba_interval=8)
            if use_async_worker
            else None
        )
        t0 = time.time()
        try:
            n_kf = fe.run(
                worker.submit_keyframe if worker is not None else on_keyframe,
                pose_callback=(worker.submit_pose_updates if worker is not None else on_pose_update),
            )
            if worker is not None:
                worker.drain()
            if "gs_map" not in state:
                raise RuntimeError(
                    f"e2e: {n_kf} keyframes arrived but the plane was never estimated"
                )

            # Final BA sync + compact soft-carved rows + column dedup + flush dirty tiles cleanly
            state["kf_cache"].clear()
            state["gs_map"].compact()
            state["gs_map"].deduplicate_columns()
            state["engine"].flush_all()
            if state["writer"] is not None:
                state["writer"].close(wait=True)

            # Final full-resolution 3DGS orthophoto directly from live TileCache (<1s)
            final_tiles = snapshot_tiles(state["engine"].cache)
            if state["viewer"] is not None and state["viewer"].active and final_tiles:
                state["viewer"].submit(
                    final_tiles,
                    f"Final | gs={state['gs_map'].n_points:,} | tiles={len(final_tiles)}",
                )
        finally:
            if worker is not None:
                try:
                    worker.drain()
                except Exception:
                    pass
            if viewer is not None:
                viewer.close()
        rgb_full, a_full = stitch_arrays(final_tiles)
        final_info = None
        if rgb_full is not None:
            occ = np.nonzero(a_full > 0.02)
            if occ[0].size:
                r0, r1 = int(occ[0].min()), int(occ[0].max()) + 1
                c0, c1 = int(occ[1].min()), int(occ[1].max()) + 1
                rgb_c = np.clip(rgb_full[r0:r1, c0:c1], 0.0, 1.0)
                a_c = np.clip(a_full[r0:r1, c0:c1], 0.0, 1.0)
            else:
                rgb_c = np.clip(rgb_full, 0.0, 1.0)
                a_c = np.clip(a_full, 0.0, 1.0)

            # Covered pixels are already un-premultiplied in gsplat_ortho; keep alpha opaque
            # in the final PNG so viewers do not double-darken sparse regions.
            a_opaque = np.where(a_c > 0.02, 1.0, 0.0).astype(np.float32)
            final_path = self.out / "mosaic_final.png"
            rgba_u8 = (np.concatenate([rgb_c, a_opaque[..., None]], axis=-1) * 255.0).astype(np.uint8)
            Image.fromarray(rgba_u8, mode="RGBA").save(final_path)

            p5_dir = self.out / "p5_3dgs"
            p5_dir.mkdir(parents=True, exist_ok=True)
            final_3dgs_path = p5_dir / "mosaic_final_3dgs.png"
            Image.fromarray(rgba_u8, mode="RGBA").save(final_3dgs_path)
            Image.fromarray((a_c * 255.0).astype(np.uint8), mode="L").save(
                p5_dir / "mosaic_final_3dgs_alpha.png"
            )
            final_info = {
                "tiles": len(final_tiles),
                "shape": [int(rgb_c.shape[0]), int(rgb_c.shape[1])],
                "covered_frac": float((a_c > 0.02).mean()),
                "path": str(final_path),
                "path_3dgs": str(final_3dgs_path),
            }

        # Export standard 3DGS PLY (Plane frame Z-up, log scales, logit opacity)
        ply_path = self.out / "ground_3dgs.ply"
        ply_info = state["gs_map"].export_ply(
            ply_path,
            meta={
                "coordinate system": "leveled plane (Z up), same frame as ortho mosaic",
                "gsd_per_px": f"{state['gsd']:.6f}",
                "n_keyframes": int(n_kf),
                "use_calib": str(use_calib),
            },
        )
        # Keep copy/link under p5_3dgs/ for backwards path compatibility
        p5_ply = self.out / "p5_3dgs" / "ground_3dgs.ply"
        p5_ply.parent.mkdir(parents=True, exist_ok=True)
        try:
            if p5_ply.exists() or p5_ply.is_symlink():
                p5_ply.unlink()
            p5_ply.symlink_to(ply_path)
        except Exception:
            import shutil
            shutil.copyfile(ply_path, p5_ply)

        # Build aligned update video across growing tile bounds
        video = None
        if state["frames"]:
            bounds = state["frame_tile_bounds"]
            existing = [(fp, bd) for fp, bd in zip(state["frames"], bounds)
                        if pathlib.Path(fp).exists()]
            if existing:
                ex_frames = [e[0] for e in existing]
                ex_bounds = [e[1] for e in existing]
                imin_all = min(b[0] for b in ex_bounds)
                jmax_all = max(b[3] for b in ex_bounds)
                tile_s_scaled = int(round(int(c.ortho.tile) * dump_scale))
                boxes = []
                for (bi0, bi1, bj0, bj1) in ex_bounds:
                    r0_b = (jmax_all - bj1) * tile_s_scaled
                    c0_b = (bi0 - imin_all) * tile_s_scaled
                    hh_b = (bj1 - bj0 + 1) * tile_s_scaled
                    ww_b = (bi1 - bi0 + 1) * tile_s_scaled
                    boxes.append((r0_b, r0_b + hh_b, c0_b, c0_b + ww_b))
                video = write_update_video(
                    self.out / "ortho_update.mp4",
                    ex_frames,
                    state["engine"].cache,
                    c.ortho.tile,
                    boxes=boxes,
                )

        align = aligner.save(self.out / "alignment.json")

        # Stage-0 quantitative 3DGS quality metrics & post-BA residual alignment audit
        from stream_ortho.eval.gs_metrics import compute_gs_quality_metrics
        diag_metrics = compute_gs_quality_metrics(
            state["gs_map"].means, owner_kf=state["gs_map"].owner_kf, gsd=state["gsd"]
        )
        post_ba_kfs = []
        full_gnd_meds = []
        for k_val in np.unique(state["gs_map"].owner_kf).tolist():
            k_int = int(k_val)
            sel_k = state["gs_map"].owner_kf == k_int
            if int(sel_k.sum()) < 200:
                continue
            pts_k = state["gs_map"].means[sel_k]
            rr = aligner.estimate_residual_rect(pts_k)
            entry = {
                "kf": k_int,
                "n_pts": int(sel_k.sum()),
                "z_p15": float(np.percentile(pts_k[:, 2], 15)),
                "z_max": float(pts_k[:, 2].max()),
                **rr,
            }
            if k_int in aligner._kf_gnd_w and k_int in aligner.rect_history:
                T_pw_k = aligner.rect_history[k_int]
                gp_k = aligner._kf_gnd_w[k_int] @ T_pw_k[:3, :3].T + T_pw_k[:3, 3]
                g_med = float(np.median(gp_k[:, 2]))
                rr_full = aligner.estimate_residual_rect(gp_k)
                entry["full_gnd_med"] = g_med
                entry["full_gnd_res_dz"] = float(rr_full["res_dz"])
                entry["full_gnd_res_tilt_deg"] = float(rr_full["res_tilt_deg"])
                full_gnd_meds.append(g_med)
            post_ba_kfs.append(entry)
        fg_arr = np.asarray(full_gnd_meds, dtype=np.float64) if full_gnd_meds else np.zeros(0)
        diag_metrics["post_ba_residual_alignment"] = {
            "n_kfs": len(post_ba_kfs),
            "max_abs_res_dz": float(max((abs(x["res_dz"]) for x in post_ba_kfs), default=0.0)),
            "p95_abs_res_dz": float(np.percentile([abs(x["res_dz"]) for x in post_ba_kfs], 95)) if post_ba_kfs else 0.0,
            "max_res_tilt_deg": float(max((x["res_tilt_deg"] for x in post_ba_kfs), default=0.0)),
            "full_gnd_med_std": float(fg_arr.std()) if fg_arr.size else 0.0,
            "full_gnd_med_span": float(fg_arr.max() - fg_arr.min()) if fg_arr.size else 0.0,
            "full_gnd_max_abs_res_dz": float(max((abs(x.get("full_gnd_res_dz", 0.0)) for x in post_ba_kfs), default=0.0)),
            "per_kf": post_ba_kfs,
        }
        diag_metrics["latency_ms"] = {
            "lift_p50": float(np.percentile(state["lift_only_ms"] or [0.0], 50)),
            "align_p50": float(np.percentile(state["align_ms"] or [0.0], 50)),
            "quat_rot_p50": float(np.percentile(state["quat_rot_ms"] or [0.0], 50)),
            "ingest_p50": float(np.percentile(state["ins_only_ms"] or [0.0], 50)),
            "tile_p50": float(np.percentile(state.get("tile_ms") or [0.0], 50)),
            "ba_sync_total_ms": float(sum(state["ba_ms"])),
            "n_ba_events": len(state["ba_updates"]),
            "async_worker": bool(worker is not None),
            "ba_batches_executed": (worker.executed_pose_batches if worker is not None else len(state["ba_ms"])),
            "ba_batches_coalesced": (worker.coalesced_pose_batches if worker is not None else 0),
        }
        diag_metrics["total_carved"] = int(state.get("total_carved", 0))
        (self.out / "diag_metrics.json").write_text(
            json.dumps(diag_metrics, indent=2, ensure_ascii=False)
        )
        (self.out / "kf_diag.json").write_text(
            json.dumps({"keyframes": state["kf_diag"], "ba_updates": state["ba_updates"]},
                       indent=2, ensure_ascii=False)
        )
        fine_m = diag_metrics.get("fine", {})
        ig_m = diag_metrics.get("inter_kf_ground", {})
        pba_m = diag_metrics["post_ba_residual_alignment"]
        lat_m = diag_metrics["latency_ms"]
        print(
            f"[e2e-diag] cell={diag_metrics.get('fine_cell_size', 0):.3f} "
            f"elev_std={fine_m.get('elev_z_std_mean', 0):.4f} "
            f"multi_band={fine_m.get('elev_multi_band_frac', 0):.3f} "
            f"ghost_pts={fine_m.get('overhang_elev_pt_frac', 0):.3f} "
            f"kf_gnd_std={ig_m.get('p15_std', 0):.4f} "
            f"full_gnd_std={pba_m.get('full_gnd_med_std', 0):.4f} "
            f"post_ba_dz_max={pba_m['max_abs_res_dz']:.4f} "
            f"full_dz_max={pba_m.get('full_gnd_max_abs_res_dz', 0):.4f} "
            f"carved={diag_metrics.get('total_carved', 0)} "
            f"(lift={lat_m['lift_p50']:.0f}ms align={lat_m['align_p50']:.0f}ms "
            f"qrot={lat_m['quat_rot_p50']:.0f}ms ins={lat_m['ingest_p50']:.0f}ms "
            f"tile={lat_m['tile_p50']:.0f}ms)",
            flush=True,
        )
        ins = np.asarray(state.get("insert_ms", [0.0]), float)
        til = np.asarray(state.get("tile_ms", [0.0]), float)
        upd = ins + til
        ortho = {
            "insert_ms": {"p50": float(np.percentile(ins, 50)), "p95": float(np.percentile(ins, 95))},
            "tile_ms": {"p50": float(np.percentile(til, 50)), "p95": float(np.percentile(til, 95))},
            "ortho_update_ms": {"p50": float(np.percentile(upd, 50)), "p95": float(np.percentile(upd, 95))},
            "ortho_fps": float(1000.0 / max(np.percentile(upd, 50), 1e-6)),
            "tiles_per_s": float(1000.0 * state.get("n_tiles", 0) / max(upd.sum(), 1e-6)),
            "n_updates": int(upd.size),
        }
        (self.out / "ortho_fps.json").write_text(json.dumps(ortho, indent=2, ensure_ascii=False))
        ortho_3dgs = {
            "streaming": True,
            "n_keyframes": int(n_kf),
            "n_inserted": int(state["n"]),
            "skipped_scale": int(state["skipped_scale"]),
            "gsd": float(state["gsd"]),
            "voxel_size": float(state["voxel_size"]),
            "n_gaussians": int(state["gs_map"].n_points),
            "ply": ply_info,
            "final": final_info,
            "t_total_s": time.time() - t0,
        }
        (self.out / "p5_3dgs" / "ortho_3dgs.json").write_text(
            json.dumps(ortho_3dgs, indent=2, ensure_ascii=False)
        )
        res = {
            "status": "e2e",
            "streaming_3dgs": True,
            "n_keyframes": int(n_kf),
            "n_inserted": int(state["n"]),
            "skipped_scale": int(state["skipped_scale"]),
            "ortho": ortho,
            "video": video,
            "gsd_used": float(state["gsd"]),
            "n_gaussians": int(state["gs_map"].n_points),
            "ply": ply_info,
            "wall_s": time.time() - t0,
            "alignment": align,
            "metric_scale": bool(use_calib),
            "use_calib": bool(use_calib),
            "relative_scale": bool(state.get("relative_scale", False)),
            "tiles": {
                "covered": len(state["engine"].covered),
                "rendered": len(state["engine"].rendered),
            },
            "coverage_complete": state["engine"].coverage_complete(),
            "final_mosaic": final_info,
            "3dgs": ortho_3dgs,
            "config": self.cfg.to_dict(),
        }
        (self.out / "stream.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        self.cfg.save(self.out / "config_resolved.yaml")
        return res

    def run_e2e(self) -> dict:
        """One command, no external tables: RGB -> SLAM -> plane -> filter -> product -> tiles."""
        use_calib = self._resolve_use_calib()
        c = self.cfg
        if getattr(c.gaussian, "enabled", False) and getattr(c.gaussian, "streaming_3dgs", False):
            return self._run_e2e_streaming_3dgs(use_calib=use_calib)

        import numpy as np
        from stream_ortho.app.alignment import PlaneAligner
        from stream_ortho.app.frontend import LiveFrontend
        from stream_ortho.filter.pointmap import FilterConfig, PointmapFilter
        from stream_ortho.geometry import sim3_from_vec8
        from stream_ortho.online.product import StreamingProduct
        from stream_ortho.ortho.cache import TileCache
        from stream_ortho.ortho.grid import OrthoGrid
        from stream_ortho.ortho.scheduler import OrthoEngine, SchedulerConfig
        from stream_ortho.scripts.run_stream import tile_render_fn_factory

        aligner = PlaneAligner(mode=getattr(c.data, "plane_mode", "auto"),
                               bootstrap_kfs=int(getattr(c.map, "bootstrap_kfs", 5)),
                               T_WP=None)
        state = {"n": 0, "skipped": 0}

        def _world(pkt):
            R, t, sc = sim3_from_vec8(np.asarray(pkt.T_WC, np.float64))
            P = np.stack([pkt.x, pkt.y, pkt.z], axis=1).astype(np.float64)
            assert P.shape[1] == 3 and P.shape[0] == np.asarray(pkt.x).size, \
                f"_world: bad point array {P.shape}"
            state["cam_t"] = t
            state["n_pts"] = P.shape[0]
            return float(sc) * (P @ R.T) + t

        def _build(Pp):
            # without a GT Sim3 the cloud is in SLAM units; fix the *relative* resolution from
            # its extent instead of pretending 0.45 m / 0.30 m mean anything there
            ext = float(max(Pp[:, 0].ptp(), Pp[:, 1].ptp()))
            cell = c.map.cell
            G0 = int(getattr(c.map, "grid_cells", 2048) or 2048)
            if c.data.plane_mode != "gt":
                cell = max(ext / 1024.0, 1e-6)
                state["gsd"] = cell
                state["relative_scale"] = True
            else:
                state["gsd"] = c.ortho.gsd
                state["relative_scale"] = False
            G = int(getattr(c.map, "grid_cells", 0) or 0)   # 0 -> fallback bbox
            if G > 0:
                # preallocate: a later keyframe can be far from frame 0, and a grid built from
                # the first frame alone would clip every other footprint to the border
                cx = 0.5 * (Pp[:, 0].min() + Pp[:, 0].max())
                cy = 0.5 * (Pp[:, 1].min() + Pp[:, 1].max())
                ox = int(np.floor(cx / cell)) - G // 2
                oy = int(np.floor(cy / cell)) - G // 2
                H = W = G
            else:
                ox = int(np.floor(Pp[:, 0].min() / cell)) - 8
                oy = int(np.floor(Pp[:, 1].min() / cell)) - 8
                W = int(np.ceil((Pp[:, 0].max() - Pp[:, 0].min()) / cell)) + 16
                H = int(np.ceil((Pp[:, 1].max() - Pp[:, 1].min()) / cell)) + 16
            prod = StreamingProduct(cell, ox, oy, H, W, L=c.map.max_layers, z_gap=c.map.z_gap)
            filt = PointmapFilter(FilterConfig(
                enabled=c.filter.enabled, gate=c.filter.gate, nfp=c.filter.nfp,
                prune=c.filter.prune, min_conf=c.filter.min_conf, cell=cell,
                origin=(ox * cell, oy * cell), nfp_mode=c.filter.nfp_mode,
                nfp_tau_m=c.filter.nfp_tau_m, protect_unique=c.filter.protect_unique,
                depth_min_m=c.filter.depth_min_m, depth_max_m=c.filter.depth_max_m))
            gsd = state.get("gsd", c.ortho.gsd)
            grid = OrthoGrid(T_WP=np.eye(4), gsd=gsd, tile=c.ortho.tile)
            eng = OrthoEngine(grid, TileCache(),
                              tile_render_fn_factory(prod, gsd, c.ortho.tile, mode="rgb"),
                              SchedulerConfig(tile=c.ortho.tile, budget_ms=c.ortho.budget_ms,
                                              max_tiles_per_flush=c.ortho.max_tiles))
            # anchored canvas covering the whole preallocated product: a still taken at any
            # moment lands on the same pixel grid (per-frame crops broke the video writer).
            # The anchor must be the product's tile range, NOT (0,0): the plane origin sits
            # inside the flight area, so most tiles carry negative i/j and an (0,0)-anchored
            # stitch silently skipped them -- the early video frames were black.
            tile_m = max(float(gsd) * int(c.ortho.tile), 1e-9)
            i0 = int(np.floor((ox * cell) / tile_m))
            i1 = int(np.floor((ox + G) * cell / tile_m - 1e-9))
            j0 = int(np.floor((oy * cell) / tile_m))
            j1 = int(np.floor((oy + G) * cell / tile_m - 1e-9))
            state.update(product=prod, filter=filt, grid=grid, engine=eng, ox=ox, oy=oy,
                         cell=cell, gsd=gsd, canvas_origin=(i0, j0),
                         canvas_shape=(j1 - j0 + 1, i1 - i0 + 1),
                         canvas_jmax=j1)

        def on_keyframe(pkt):
            Pw = _world(pkt)
            state["dbg"] = state.get("dbg", 0) + 1
            if state["dbg"] <= 6:
                print(f"[e2e] kf={pkt.kf} n={Pw.shape[0]} "
                      f"world_xy=({Pw[:,0].min():.1f},{Pw[:,0].max():.1f}) "
                      f"world_z=({Pw[:,2].min():.1f},{Pw[:,2].max():.1f}) "
                      f"align_ready={aligner.ready}", flush=True)
            if "product" not in state:
                if not aligner.add(Pw):
                    state["skipped"] += 1
                    return
                Pp = aligner.to_plane(Pw)
                _build(Pp)
            else:
                Pp = aligner.to_plane(Pw)
            prod, filt = state["product"], state["filter"]
            # a live frame can carry mismatched array lengths (confidence on a coarser grid,
            # pointmap on the image grid); align on the shortest so it still contributes
            n = min(Pp.shape[0], np.asarray(pkt.conf).size)
            if n != Pp.shape[0]:
                state["mismatch"] = state.get("mismatch", 0) + 1
            Pp = Pp[:n]
            conf = np.asarray(pkt.conf, np.float64)[:n]
            # depth must be metric: the camera-frame z is in SLAM units (centimetres here),
            # so use the world distance from the camera centre instead
            depth = np.linalg.norm(Pp[:n] - state.get("cam_t", np.zeros(3)), axis=1)
            keep, creasons = filt.apply(Pp[:, 0], Pp[:, 1], Pp[:, 2], conf, depth=depth,
                                        height=Pp[:, 2] - float(np.percentile(Pp[:, 2], 5)))
            if state["dbg"] <= 6:
                print(f"[e2e]   keep={int(keep.sum())}/{keep.size} reasons={ {k: v for k, v in creasons.items() if k in ('gate','nfp','prune')} }", flush=True)
            if not keep.any():
                return
            cell, ox, oy = float(prod.cell), int(prod.ox), int(prod.oy)
            # ``StreamingProduct.insert`` takes ABSOLUTE cell indices and subtracts the origin
            # itself (``_flat``).  Subtracting it here as well shifted every point by ox/oy a
            # second time, and with a 2048-cell preallocated grid the huge negative origin
            # clipped the whole footprint onto the last row (the grey stripes).
            cix_abs = np.floor(Pp[keep, 0] / cell).astype(np.int64)
            ciy_abs = np.floor(Pp[keep, 1] / cell).astype(np.int64)
            inside = ((cix_abs >= ox) & (cix_abs < ox + prod.W) &
                      (ciy_abs >= oy) & (ciy_abs < oy + prod.H))
            state.setdefault("inside_ratio", []).append(
                float(inside.mean()) if inside.size else 0.0)
            cix_abs, ciy_abs = cix_abs[inside], ciy_abs[inside]
            if cix_abs.size == 0:
                return
            Pp = Pp[keep][inside]
            conf = conf[keep][inside]
            rgbk = (None if getattr(pkt, "rgb", None) is None
                    else np.asarray(pkt.rgb, np.float32)[:n][keep][inside])
            t_ins = time.perf_counter()
            prod.insert(pkt.kf, cix_abs, ciy_abs, Pp[:, 2], conf, rgb=rgbk)
            # the occupancy must match the point count, not collapse onto one row
            assert int((prod.n > 0).sum()) > 0.1 * Pp.shape[0], "product occupancy collapsed"
            if state.setdefault("cloud_chunks", []).__len__() < 400:
                state["cloud_chunks"].append(np.c_[Pp, rgbk if rgbk is not None
                                                  else np.zeros((Pp.shape[0], 3), np.float32),
                                                  np.full(Pp.shape[0], pkt.kf, np.float32)])
                state["cloud_kept"] = state.get("cloud_kept", 0) + int(Pp.shape[0])
            state.setdefault("insert_ms", []).append((time.perf_counter() - t_ins) * 1000)
            fb = (float(Pp[:, 0].min()), float(Pp[:, 1].min()),
                  float(Pp[:, 0].max()), float(Pp[:, 1].max()))
            if state["n"] < 3:
                print(f"[e2e] fb={tuple(round(v,1) for v in fb)} cix_abs=({cix_abs.min()},{cix_abs.max()}) ciy_abs=({ciy_abs.min()},{ciy_abs.max()}) origin=({ox},{oy})", flush=True)
            t_tile = time.perf_counter()
            state["engine"].mark_footprint(fb, kf_id=pkt.kf)
            fl = state["engine"].flush()
            t_tile_done = time.perf_counter()
            state.setdefault("tile_ms", []).append((t_tile_done - t_tile) * 1000)
            state["n_tiles"] = state.get("n_tiles", 0) + int(fl.get("rendered", 0))
            upd_ms = float(state["insert_ms"][-1] + state["tile_ms"][-1])
            dump_ms = 0.0
            if state["n"] % max(int(getattr(c.ortho, "dump_every", 1) or 1), 1) == 0:
                from stream_ortho.app.mosaic_writer import snapshot_tiles
                from stream_ortho.ortho.mosaic import save_mosaic
                fpath = self.out / "frames" / f"mosaic_{state['n']:05d}.png"
                # constant pixel grid: anchored origin + fixed shape; the content box is
                # cropped per frame (the product only grows, so boxes are monotone) and the
                # video writer places every frame into the union box at the right offset
                info = save_mosaic(state["engine"].cache, fpath, tile_size=c.ortho.tile,
                                   crop=True, scale=1.0, origin=state["canvas_origin"],
                                   shape=state["canvas_shape"],
                                   jmax_anchor=state["canvas_jmax"])
                dump_ms = (time.perf_counter() - t_tile_done) * 1000
                if info:
                    state.setdefault("frames", []).append(str(fpath))
                    state.setdefault("frame_boxes", []).append(info.get("crop_box"))
            state.setdefault("dump_ms", []).append(dump_ms)
            state["n"] += 1

        fe = LiveFrontend(c.data.images, self.out, calib=use_calib,
                          max_frames=c.data.max_frames, stride=c.data.stride,
                          single_thread=c.frontend.single_thread)
        t0 = time.time()
        n_kf = fe.run(on_keyframe)
        if "product" not in state:
            raise RuntimeError(
                f"e2e: {n_kf} keyframes arrived but the plane was never estimated "
                f"(bootstrap needs {aligner.bootstrap_kfs}); increase --max-frames")
        state["engine"].flush_all()
        from stream_ortho.ortho.mosaic import save_mosaic
        fn = tile_render_fn_factory(state["product"], state.get("gsd", c.ortho.gsd),
                                    c.ortho.tile, mode="rgb")
        final_cache = TileCache()
        for (i, j) in sorted(state["engine"].covered):
            r, a, d, _ = fn(state["grid"], i, j)
            final_cache.put((i, j), r, a, d, ms=0.0)
        final = save_mosaic(final_cache, self.out / f"mosaic_final.{c.ortho.dump_format}",
                            tile_size=c.ortho.tile, feather=8, crop=True,
                            origin=state.get("canvas_origin"),
                            shape=state.get("canvas_shape"),
                            jmax_anchor=state.get("canvas_jmax"))
        # P5.1c: offline 3DGS final (high-res) render from the keyframes this run saved
        final3d = None
        if c.gaussian.final_enabled:
            from stream_ortho.scripts.run_p5 import render_3dgs_final
            import shutil
            if final:
                # tasks.md convention: preview copy under the explicit name
                shutil.copyfile(self.out / f"mosaic_final.{c.ortho.dump_format}",
                                self.out / f"mosaic_final_preview.{c.ortho.dump_format}")
            try:
                final3d = render_3dgs_final(
                    self.out, out_dir=self.out / "p5_3dgs",
                    gsd=c.gaussian.gsd_3dgs, conf_threshold=c.gaussian.conf_threshold,
                    blend_levels=c.gaussian.blend_levels,
                    feather_px=c.gaussian.feather_radius, max_kf=c.gaussian.max_kf,
                    video=True, ply=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                final3d = {"error": repr(exc)}
        self.out.mkdir(parents=True, exist_ok=True)
        align = aligner.save(self.out / "alignment.json")
        self.out.mkdir(parents=True, exist_ok=True)
        ins = np.asarray(state.get("insert_ms", [0.0]), float)
        til = np.asarray(state.get("tile_ms", [0.0]), float)
        upd = ins + til
        from stream_ortho.app.artifacts import write_points_ply, write_update_video
        video = None
        if state.get("frames"):
            # every still was cropped to its own content box on the anchored canvas; the
            # product only grows, so the union of the boxes covers every frame's content
            boxes = list(state.get("frame_boxes") or [])
            video = write_update_video(self.out / "ortho_update.mp4", state["frames"],
                                       final_cache, c.ortho.tile, boxes=boxes)
        chunks = state.get("cloud_chunks") or []
        raw_pts = np.vstack(chunks) if chunks else np.zeros((0, 7), np.float32)
        cloud = write_points_ply(self.out / "ground_points.ply", raw_pts)
        ortho = {"insert_ms": {"p50": float(np.percentile(ins, 50)),
                               "p95": float(np.percentile(ins, 95))},
                 "tile_ms": {"p50": float(np.percentile(til, 50)),
                             "p95": float(np.percentile(til, 95))},
                 "ortho_update_ms": {"p50": float(np.percentile(upd, 50)),
                                     "p95": float(np.percentile(upd, 95))},
                 "ortho_fps": float(1000.0 / max(np.percentile(upd, 50), 1e-6)),
                 "ortho_fps_no_dump": float(1000.0 / max(np.percentile(upd, 50), 1e-6)),
                 "ortho_fps_with_dump": float(1000.0 / max(
                     np.percentile(upd + np.asarray(state.get("dump_ms", [0.0]), float), 50),
                     1e-6)),
                 "tiles_per_s": float(1000.0 * state.get("n_tiles", 0) / max(upd.sum(), 1e-6)),
                 "n_updates": int(upd.size)}
        (self.out / "ortho_fps.json").write_text(json.dumps(ortho, indent=2, ensure_ascii=False))
        res = {"status": "e2e", "n_keyframes": int(n_kf), "n_inserted": state["n"],
               "ortho": ortho, "video": video, "cloud": cloud,
               "inside_ratio_p50": float(np.median(state.get("inside_ratio", [0.0]))),
               "product_mb": float(state["product"].memory_mb()),
               "cell_used": float(state.get("cell", 0.0)),
               "gsd_used": float(state.get("gsd", 0.0)),
               "n_points_per_frame": int(state.get("n_pts", 0)),
               "skipped_bootstrap": state["skipped"],
               "packets_with_length_mismatch": int(state.get("mismatch", 0)), "wall_s": time.time() - t0,
               "alignment": align,
               "metric_scale": align.get("source") == "config",
               "use_calib": use_calib, "cell_used": cell if "product" not in state else
               float(state["product"].cell),
               "gsd_used": state.get("gsd", c.ortho.gsd),
               "relative_scale": bool(state.get("relative_scale", False)),
               "tiles": {"covered": len(state["engine"].covered),
                         "rendered": len(state["engine"].rendered)},
               "coverage_complete": state["engine"].coverage_complete(),
               "final_mosaic": final, "3dgs": final3d, "config": self.cfg.to_dict()}
        (self.out / "stream.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        self.cfg.save(self.out / "config_resolved.yaml")
        return res

    def run(self) -> dict:
        self.build_dataset()
        stages = self.plan()
        for st in stages:
            print(f"[stage] {st['name']}: "
                  + ", ".join(f"{k}={v}" for k, v in st.items() if k != "name")[:180])
        res = {"status": "p0_skeleton", "n_frames": len(self.dataset),
               "metric_scale": bool(self.dataset.metric_scale),
               "intrinsics_source": self.dataset.intrinsics_source,
               "stages": stages, "wall_s": time.time() - self.t0,
               "config": self.cfg.to_dict()}
        (self.out / "run.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        self.cfg.save(self.out / "config_resolved.yaml")
        return res
