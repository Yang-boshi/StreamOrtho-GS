"""Post-flight refinement: three rounds of frozen calibration over the whole capture.

The streaming product is a single pass with bounded latency; this stage is the "final
product" the system promises: the same 3-DOF model, three rounds, layers and owners frozen,
followed by a texture orthophoto.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time


def filter_obs(obs_dir, out_dir, fcfg, cell: float = 0.45, z_ground: float | None = None):
    """Apply Module A frame by frame and write ``obs_filtered/obs_merged.npz``.

    The offline path used to read the raw table, so the final orthophoto still contained every
    floater the streaming filter had removed -- that is why raw and hybrid looked identical.
    """
    import numpy as np
    from stream_ortho.eval import sync_eval as S
    from stream_ortho.filter.pointmap import PointmapFilter

    src = pathlib.Path(obs_dir) / "obs_merged.npz"
    d = dict(np.load(src))
    T_WP = np.load(src)["T_WP"]
    kf = np.asarray(d["kf"])
    if z_ground is None:
        z_ground = float(np.percentile(d["z"][np.isfinite(d["z"])], 5))
    ox = int(np.floor(np.min(d["x"]) / cell))
    oy = int(np.floor(np.min(d["y"]) / cell))
    import dataclasses
    fcfg = dataclasses.replace(fcfg, origin=(ox * cell, oy * cell), cell=cell)
    filt = PointmapFilter(fcfg)
    keep = np.zeros(kf.size, bool)
    stats = {"gate": 0, "nfp_above": 0, "nfp_below": 0, "prune": 0,
             "protected_unique": 0, "protected_low_obs": 0, "total": 0, "kept": 0}
    for k in np.unique(kf):
        idx = np.nonzero(kf == k)[0]
        keep[idx], reasons = filt.apply(d["x"][idx], d["y"][idx], d["z"][idx],
                                        np.asarray(d["conf"][idx], np.float64),
                                        depth=d["D"][idx], height=d["z"][idx] - z_ground)
        for key in ("gate", "nfp_above", "nfp_below", "prune", "protected_unique",
                    "protected_low_obs"):
            stats[key] += int(reasons.get(key, 0))
        stats["total"] += int(reasons["total"])
        stats["kept"] += int(reasons["kept"])
    stats["removed_frac"] = 1.0 - stats["kept"] / max(stats["total"], 1)

    out = pathlib.Path(out_dir) / "obs_filtered"
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v[keep] for k, v in d.items() if hasattr(v, "shape") and v.shape[:1] == kf.shape}
    np.savez(out / "obs_merged.npz", T_WP=T_WP, cell=np.float32(cell), **payload)
    (pathlib.Path(out_dir) / "filter_stats_offline.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    return str(out), stats


def carve_obs(obs_dir, out_dir, cfg=None):
    """Apply multi-view support carving and write ``obs_carved/obs_merged.npz``."""
    import numpy as np
    from stream_ortho.filter.carve import CarveConfig, carve_rows

    src = pathlib.Path(obs_dir) / "obs_merged.npz"
    d = dict(np.load(src))
    T_WP = np.load(src)["T_WP"]
    cell = float(np.load(src)["cell"])
    keep, stats = carve_rows(d, cfg or CarveConfig())
    out = pathlib.Path(out_dir) / "obs_carved"
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v[keep] for k, v in d.items() if hasattr(v, "shape") and v.shape[:1] == keep.shape}
    np.savez(out / "obs_merged.npz", T_WP=T_WP, cell=np.float32(cell), **payload)
    (pathlib.Path(out_dir) / "carve_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    return str(out), stats


def suppress_top_obs(obs_dir, out_dir, tau: float = 2.0):
    """Write ``obs_top`` with per-frame splashes above each cell's median removed."""
    import numpy as np
    from stream_ortho.filter.carve import suppress_top_rows

    src = pathlib.Path(obs_dir) / "obs_merged.npz"
    d = dict(np.load(src))
    root = np.load(src)
    keep, stats = suppress_top_rows(d, tau)
    out = pathlib.Path(out_dir) / "obs_top"
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v[keep] for k, v in d.items()
               if hasattr(v, "shape") and v.shape[:1] == keep.shape}
    np.savez(out / "obs_merged.npz", T_WP=root["T_WP"], cell=root["cell"], **payload)
    (pathlib.Path(out_dir) / "top_suppress.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    return str(out), stats


def run_offline(obs_dir, out_dir, seq="ieu", iters=3, model="hybrid", lam_p=0.1,
                texture=True, cell_hint=None, apply_filter: bool = True,
                filter_config=None, apply_carve: bool = True):
    """Run the offline calibration (and optional texture ortho); returns a small summary."""
    from stream_ortho.scripts.run_pivotb import run as run_pivotb

    t0 = time.time()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub = out / "offline"
    filt_stats = None
    obs_raw = str(obs_dir)                       # keep the unfiltered table for the raw panel
    if apply_filter:
        from stream_ortho.filter.pointmap import FilterConfig
        fcfg = filter_config or FilterConfig()
        obs_dir, filt_stats = filter_obs(obs_dir, out, fcfg, cell=float(cell_hint or 0.45))
    top_stats = None
    if getattr(run_offline, "suppress_top", True):
        obs_dir, top_stats = suppress_top_obs(obs_dir, out, tau=2.0)
    carve_stats = None
    if apply_carve:
        from stream_ortho.filter.carve import CarveConfig
        obs_dir, carve_stats = carve_obs(obs_dir, out, CarveConfig())
    res, path = run_pivotb(obs_dir, sub, seq, ["frozen"], iters, model, 0.0, "",
                           lam_p=lam_p, k=6)
    summary = {"json": str(path), "iters": iters, "model": model,
               "obs_used": str(obs_dir), "filter": filt_stats, "carve": carve_stats,
               "top_suppress": top_stats,
               "product": str(sub / "product_frozen.npz"),
               "quality": {k: float(v) for k, v in res["modes"]["frozen"]["final"].items()
                           if isinstance(v, (int, float))},
               "gate": res.get("gate", {}).get("frozen", {})}
    if texture:
        try:
            from stream_ortho.scripts.diag.render_ortho import main as render_main
            # render_ortho builds both the raw and the frozen product from the observation
            # table, so it only needs --obs/--out
            # both panels must see their own table: filtered for panels 2/3, raw for panel 1
            render_main(["--obs", str(obs_dir), "--obs-raw", obs_raw, "--out", str(sub),
                         "--seq", seq, "--iters", str(iters), "--lam-p", str(lam_p)])
            summary["ortho"] = str(sub / "ortho" / "ortho_compare.png")
        except Exception as exc:                  # texture is optional; never fail the run
            summary["ortho_error"] = repr(exc)
    summary["wall_s"] = time.time() - t0
    (out / "offline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
