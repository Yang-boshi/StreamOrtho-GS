"""P3.5: geometry x cell grid -- one figure for the numbers that decide it."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/system")
    ap.add_argument("--out", default="results/system/p35_summary.png")
    ap.add_argument("--tag", default="p35")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(p for p in pathlib.Path(a.dir).glob(f"{a.tag}_*") if p.is_dir())
    rows = []
    for d in runs:
        st = d / "stream.json"
        dg = d / "p35_diag.json"
        if not st.exists():
            continue
        r = json.loads(st.read_text())
        q = r["quality"]
        g = (r.get("offline") or {}).get("gate", {})
        v = g.get("values", {})
        row = {"name": d.name.replace(f"{a.tag}_", ""),
               "coverage": q["coverage_cells"], "coverage_cells_seen": None, "lcr": v.get("lcr_fixed"),
               "jump": v.get("n_jump"), "ra": v.get("ra"), "sei": v.get("sei"),
               "p50": r["latency"]["p50"],
               "roof_ret": None, "ground_ret": None, "strip": None, "floater": None}
        if dg.exists():
            dd = json.loads(dg.read_text())
            row.update({"coverage_cells_seen": dd.get("coverage"),
                        "roof_ret": dd.get("roof_retention_median"),
                        "ground_ret": dd.get("ground_retention"),
                        "strip": dd.get("edge_strip_score"),
                        "floater": dd.get("floater_frac_of_live")})
        rows.append(row)
    if not rows:
        raise SystemExit("no runs found")
    names = [r["name"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    x = np.arange(len(rows))
    panels = (("coverage", "coverage (≥0.75)", 0.75),
              ("roof_ret", "roof retention (≥0.90)", 0.90),
              ("strip", "edge strip score (lower better)", None),
              ("lcr", "LCR_fixed (≥0.90)", 0.90))
    for ax, (key, title, gate) in zip(axes, panels):
        if key == "coverage":
            vals = [r["coverage_cells_seen"] if r.get("coverage_cells_seen") is not None
                    else (r[key] or 0.0) for r in rows]
        else:
            vals = [r[key] if r[key] is not None else 0.0 for r in rows]
        ax.bar(x, vals, color="#4c72b0")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if isinstance(v, float) else str(v), ha="center",
                    va="bottom", fontsize=7)
        if gate is not None:
            ax.axhline(gate, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.suptitle("P3.5: geometry x cell grid (raw cannot reach the coverage gate; "
                 "adapter+nearest is the only one that does)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out); plt.close(fig)
    (pathlib.Path(a.out).with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
