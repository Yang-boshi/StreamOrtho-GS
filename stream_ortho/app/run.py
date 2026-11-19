"""One command: ``python -m stream_ortho.app.run --config ... --images ... --out ...``."""
from __future__ import annotations

import argparse
import sys

from stream_ortho.app.config import ConfigError, load_config, override


def build_parser():
    ap = argparse.ArgumentParser(prog="stream_ortho.app.run",
                                 description="streaming feed-forward orthomapping")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--images", default=None, help="RGB folder (overrides data.images)")
    ap.add_argument("--out", default="results/run_001")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--intrinsics", default=None,
                    choices=("auto", "exif", "provided", "none"))
    ap.add_argument("--intrinsics-value", default=None,
                    help="fx,fy,cx,cy (required with --intrinsics provided)")
    ap.add_argument("--gsd", type=float, default=None)
    ap.add_argument("--cell", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate the config and exit")
    ap.add_argument("--obs", default=None,
                    help="observation table dir; switches to the replay streaming pipeline")
    ap.add_argument("--frontend", default="replay", choices=("replay", "live"))
    ap.add_argument("--calib", action="store_true", help="use calibrated intrinsics (live)")
    ap.add_argument("--no-calib", action="store_true", help="no intrinsics: relative scale")
    ap.add_argument("--e2e", action="store_true",
                    help="images only: run the SLAM frontend in-process (no obs table)")
    ap.add_argument("--no-filter", action="store_true", help="ablation: disable Module A")
    ap.add_argument("--seq", default=None, help="sequence name (data.seq)")
    ap.add_argument("--oracle", default=None,
                    help="offline product npz used only for the quality comparison")
    ap.add_argument("--show", action="store_true",
                    help="open a live preview window updating as orthophoto tiles change")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.calib and a.no_calib:
        print("config error: --calib and --no-calib are mutually exclusive", file=sys.stderr)
        return 2
    try:
        cfg = load_config(a.config)
        iv = [float(v) for v in a.intrinsics_value.split(",")] if a.intrinsics_value else None
        override(cfg, data__images=a.images, data__max_frames=a.max_frames,
                 filter__enabled=(False if a.no_filter else None),
                 data__intrinsics=a.intrinsics, data__intrinsics_value=iv,
                 data__seq=a.seq,
                 frontend__use_calib=("true" if a.calib else ("false" if a.no_calib else None)),
                 ortho__gsd=a.gsd, ortho__show=(True if a.show else None), map__cell=a.cell)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if a.dry_run:
        print(cfg.to_yaml())
        return 0
    if not cfg.data.images and not a.obs:
        print("config error: data.images is empty (pass --images, or --obs for replay)",
              file=sys.stderr)
        return 2

    from stream_ortho.app.pipeline import Pipeline
    pipe = Pipeline(cfg, a.out)
    if a.obs:
        res = pipe.run_streaming(a.obs, a.frontend, a.oracle)
    elif a.e2e or a.frontend == "live" or cfg.frontend.backend == "live":
        res = pipe.run_e2e()
    else:
        res = pipe.run()
    if res.get("status") == "e2e":
        print(f"done: {res['n_keyframes']} keyframes, tiles {res['tiles']} -> {a.out}")
    elif res.get("status") == "p1_streaming":
        lat = res["latency"]; fs = res["filter"]
        print(f"done: {res['n_frames']} frames  P50 {lat['p50']:.0f} ms  "
              f"filtered {100*fs.get('removed_frac', 0):.1f}% -> {a.out}")
    else:
        note = "" if res["metric_scale"] else "  [relative scale: no metric intrinsics]"
        print(f"done: {res['n_frames']} frames -> {a.out}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
