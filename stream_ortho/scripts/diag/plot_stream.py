"""Figures for the streaming replay: latency vs N, coverage vs N, quality vs oracle."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/week10-stream/ieu")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pathlib.Path(a.dir); r = json.loads((d / "stream.json").read_text())
    fr = r["frames"]
    n = np.array([e["n"] for e in fr]); tot = np.array([e["calib_ms"] + e["insert_ms"] + e["tile_ms"] for e in fr])
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    ax = axes[0]
    ax.plot(n, tot, "-", color="#4c72b0", lw=1.2, label="total / frame")
    ax.plot(n, [e["calib_ms"] for e in fr], "-", color="#55a868", lw=0.9, label="calib")
    ax.plot(n, [e["insert_ms"] for e in fr], "-", color="#dd8452", lw=0.9, label="insert")
    ax.plot(n, [e["tile_ms"] for e in fr], "-", color="#c44e52", lw=0.9, label="tiles")
    ax.axhline(500, color="k", ls=":", lw=1, label="P50 budget 500 ms")
    ax.axhline(1000, color="k", ls="--", lw=1, label="P95 budget 1 s")
    ax.set_xlabel("frames consumed"); ax.set_ylabel("ms")
    ax.set_title(f"latency (P50 {r['latency']['p50']:.0f} ms, P95 {r['latency']['p95']:.0f} ms)")
    ax.legend(fontsize=6)

    ax = axes[1]
    ck = r["checkpoints"]
    ks = sorted(int(k) for k in ck)
    ax.plot(ks, [ck[str(k)]["live_cells"] / 1e6 for k in ks], "-o", color="#4c72b0", label="live cells (M)")
    ax.plot(ks, [ck[str(k)]["coverage_of_bbox"] for k in ks], "-s", color="#55a868", label="coverage of bbox")
    ax2 = ax.twinx()
    ax2.plot(ks, [ck[str(k)]["rendered_tiles"] for k in ks], "-^", color="#c44e52", label="rendered tiles")
    ax2.set_ylabel("rendered tiles"); ax2.grid(False)
    ax.set_xlabel("frames"); ax.set_title("coverage grows with the stream")
    ax.legend(fontsize=6, loc="upper left"); ax2.legend(fontsize=6, loc="lower right")

    ax = axes[2]
    off = {"lcr_fixed": 0.9077, "n_jump": 197932, "ra": 0.744}
    qs = r.get("quality_stream") or {}
    names = ["LCR_fixed (↑)", "n_jump (↓)", "RA (↓)"]
    ours = [qs.get("lcr_fixed", np.nan), qs.get("n_jump", np.nan), qs.get("ra", np.nan)]
    ref = [off["lcr_fixed"], off["n_jump"], off["ra"]]
    x = np.arange(3)
    ax.bar(x - 0.2, [o / rr * 100 for o, rr in zip(ours, ref)], 0.4, color="#4c72b0", label="streaming / offline (%)")
    ax.axhline(100, color="k", ls=":", lw=1, label="offline oracle = 100")
    ax.axhline(80, color="r", ls="--", lw=1, label="80% gate")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("% of offline")
    ax.set_title("quality vs the offline oracle at N=55")
    ax.legend(fontsize=6)
    fig.suptitle("Streaming replay on ieu: real-time-ish latency, LCR close to offline, RA lags", fontsize=8.5)
    fig.tight_layout(); fig.savefig(d / "figs" / "stream_summary.png"); plt.close(fig)
    print("wrote", d / "figs" / "stream_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
