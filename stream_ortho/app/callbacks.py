"""Per-frame callbacks: filter -> frozen insert -> dirty tiles."""
from __future__ import annotations

import pathlib
import time

import numpy as np


class FrameCallbacks:
    def __init__(self, product, engine, calib, filt, cell, origin, gsd, tile_renderer=None,
                 z_ground=0.0, out_dir=None, dump_every=0, tile_size=512, writer=None,
                 dump_fmt="png"):
        self.product = product
        self.engine = engine
        self.calib = calib
        self.filter = filt
        self.cell = cell
        self.origin = origin
        self.z_ground = float(z_ground)
        self.out_dir = pathlib.Path(out_dir) if out_dir else None
        self.dump_every = int(dump_every)
        self.tile_size = int(tile_size)
        self.mosaics = []
        self.writer = writer
        self.dump_fmt = dump_fmt
        self.log = []
        self.filter_stats = {"frames": 0, "total": 0, "gate": 0, "nfp": 0,
                             "nfp_above": 0, "nfp_below": 0, "prune": 0,
                             "protected_unique": 0, "protected_low_obs": 0, "kept": 0}
        self.n = 0

    def on_frame(self, pkt):
        t0 = time.perf_counter()
        # the height gate is about metres *above the ground*, but plane z carries an arbitrary
        # offset (the ieu frame runs from -472 m to -257 m), so subtract the ground reference
        keep, reasons = self.filter.apply(pkt.x, pkt.y, pkt.z, pkt.conf, depth=pkt.depth,
                                          height=pkt.z - self.z_ground)
        t_filter = (time.perf_counter() - t0) * 1000
        for k in ("gate", "nfp", "nfp_above", "nfp_below", "prune", "protected_unique",
                  "protected_low_obs", "kept", "total"):
            self.filter_stats[k] += int(reasons.get(k, 0))
        self.filter_stats["frames"] += 1

        self.n += 1
        t0 = time.perf_counter()
        z = pkt.z if not self.calib else None
        z_cor, info = (self.calib.process(self.product, self.n - 1, pkt.kf, pkt.x[keep],
                                          pkt.y[keep], pkt.z[keep],
                                          pkt.conf[keep] * pkt.cos_inc[keep])
                       if self.calib else (pkt.z[keep], {"mode": "none"}))
        t_calib = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cix = np.floor(pkt.x[keep] / self.cell).astype(np.int64)
        ciy = np.floor(pkt.y[keep] / self.cell).astype(np.int64)
        rgb = None
        if getattr(pkt, "rgb", None) is not None:
            rgb = np.asarray(pkt.rgb, np.float32)[keep]
        ins = self.product.insert(pkt.kf, cix, ciy, z_cor, pkt.conf[keep] * pkt.cos_inc[keep],
                                  rgb=rgb)
        t_insert = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fb = None
        if keep.any():
            from stream_ortho.ortho.fov import footprint_bbox
            fb = footprint_bbox(np.stack([pkt.x[keep], pkt.y[keep]], 1))
        if fb is not None:
            self.engine.mark_footprint(fb, kf_id=pkt.kf)
            fl = self.engine.flush()
        else:
            fl = {"rendered": 0, "dirty_left": 0}
        if (self.dump_every and self.out_dir is not None
                and self.n % self.dump_every == 0):
            path = self.out_dir / f"mosaic_{self.n:05d}.{self.dump_fmt}"
            if self.writer is not None:
                from stream_ortho.app.mosaic_writer import snapshot_tiles
                tiles = snapshot_tiles(self.engine.cache)
                self.writer.submit(path, tiles, final=False)     # non-blocking
                self.mosaics.append({"path": str(path), "tiles": len(tiles), "async": True})
            else:
                from stream_ortho.ortho.mosaic import save_mosaic
                info = save_mosaic(self.engine.cache, path, tile_size=self.tile_size)
                if info:
                    self.mosaics.append(info)
        t_tile = (time.perf_counter() - t0) * 1000

        self.log.append({"n": self.n, "kf": pkt.kf, "filter_ms": t_filter, "calib_ms": t_calib,
                         "insert_ms": t_insert, "tile_ms": t_tile,
                         "kept": int(keep.sum()), "total": int(keep.size),
                         "removed_frac": reasons.get("removed_frac", 0.0),
                         "mode": info.get("mode"), "rendered": fl.get("rendered", 0),
                         "dirty_left": fl.get("dirty_left", 0)})
        return self.log[-1]

    def summary(self, wall_s: float) -> dict:
        ms = np.array([e["filter_ms"] + e["calib_ms"] + e["insert_ms"] + e["tile_ms"]
                       for e in self.log]) if self.log else np.zeros(1)
        fs = dict(self.filter_stats)
        if fs["total"]:
            fs["removed_frac"] = 1.0 - fs["kept"] / fs["total"]
        return {"n_frames": len(self.log), "wall_s": wall_s,
                "latency": {"p50": float(np.percentile(ms, 50)),
                            "p95": float(np.percentile(ms, 95))},
                "filter": fs, "mosaics": self.mosaics,
                "writer": self.writer.stats() if self.writer is not None else None,
                "coverage_complete": self.engine.coverage_complete(),
                "filter_ms": {"p50": float(np.median([e["filter_ms"] for e in self.log]))
                              if self.log else 0.0},
                "frames": self.log}
