"""Pipeline skeleton: the stages the full system runs, in order.

P0 only wires the data path (config -> dataset -> stage plan -> run record); the streaming
product, the per-frame filter and the FOV tiles are attached in P1/P2.  The stage list is
printed and written to ``run.json`` so a run is self-describing even before it produces maps.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from stream_ortho.app.config import Config


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

    def run_e2e(self) -> dict:
        """One command, no external tables: RGB -> SLAM -> plane -> filter -> product -> tiles."""
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

        c = self.cfg
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
                # the grid must span the whole flight, not just the first frame: size one cell
                # so that the preallocated grid covers ~8x the first cloud
                cell = max(ext / 256.0, 1e-6)   # 1:1 with the 288x512 colour source (sharpest)
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

        if c.frontend.use_calib == "true" and c.data.intrinsics not in ("provided", "exif"):
            raise RuntimeError("--calib needs intrinsics; pass --intrinsics provided or "
                               "use --no-calib")
        use_calib = {"true": True, "false": False,
                     "auto": c.data.intrinsics in ("provided", "exif")}[c.frontend.use_calib]
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
