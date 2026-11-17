"""A/B comparison with hard guards (week9-00, Day 0).

Refuses to compare runs that are not comparable: the GSD, extent, sequence and frame
count must match, and both runs must carry a ``run_config.json`` with its git hash.
"""

from __future__ import annotations

import json
import pathlib
import sys

from stream_ortho.eval.metrics import measure

KEYS = [("coverage", "覆盖率", 100.0), ("count", "孔洞数", 1.0),
        ("lap", "锐度", 1.0), ("grad_p90", "梯度p90", 1.0),
        ("patchwork", "邻域色差", 1.0), ("der", "DER", 1.0), ("lsr_px", "LSR(px)", 1.0)]


def _cfg(run: pathlib.Path) -> dict:
    p = run / "run_config.json"
    if not p.exists():
        raise SystemExit(f"{run}: 缺少 run_config.json，这次 run 不算数（铁律 R5）")
    return json.loads(p.read_text())


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit("usage: compare.py RUN_A RUN_B")
    a, b = pathlib.Path(argv[0]), pathlib.Path(argv[1])
    ca, cb = _cfg(a), _cfg(b)
    for k in ("gsd", "seq", "n_frames"):
        if ca.get(k) != cb.get(k):
            raise SystemExit(f"不可比：{k} {ca.get(k)} != {cb.get(k)}")
    ma, mb = measure(a, ca.get("seq")), measure(b, cb.get("seq"))
    print(f"| 指标 | A {a.name} | B {b.name} | Δ |")
    print("|---|---|---|---|")
    for key, label, scale in KEYS:
        va, vb = ma.get(key), mb.get(key)
        if va is None or vb is None or (isinstance(va, float) and va != va):
            continue
        dv = (vb - va) * scale
        print(f"| {label} | {va*scale:.2f} | {vb*scale:.2f} | {dv:+.2f} |")
    for key, label in (("ms_patch", "ms patch"), ("ms_map", "ms map"), ("ms_render", "ms render")):
        va = ma.get("ms", {}).get(key, {}).get("p50")
        vb = mb.get("ms", {}).get(key, {}).get("p50")
        if va and vb:
            print(f"| {label} P50 (ms) | {va:.0f} | {vb:.0f} | {vb-va:+.0f} |")
    json.dump({"A": ma, "B": mb}, open(a.parent / "compare.json", "w"), indent=2)


if __name__ == "__main__":
    main()
