"""P4.2 metric closure: the three definitions that had to be corrected.

1. edge duplication -- how many *observations of the same frame* land on the same cell.  This is
   what nearest-neighbour upsampling multiplies (the old "strip score" counted multi-view
   redundancy instead, which is a feature, not a defect).
2. floater -- a point more than ``tau`` above its cell's cross-frame median *and* supported by
   fewer than ``min_frames`` frames.  The old test (above the local p10 by 2.5 m) counted real
   buildings.
3. top-surface error distribution -- replacing the +-1.5 m roof-retention binary.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def edge_duplication(d, cell_is_edge=None):
    """Mean and p90 of (rows per (frame, cell)) -- 1.0 means every frame votes once."""
    kf = np.asarray(d["kf"], np.int64)
    key = np.asarray(d["key"], np.int64)
    pair = key * 1000003 + kf
    uniq, counts = np.unique(pair, return_counts=True)
    key_of = (uniq // 1000003)
    return {"mean_obs_per_frame_cell": float(counts.mean()),
            "p90": float(np.percentile(counts, 90)),
            "frac_multi": float(np.mean(counts > 1)),
            "n_pairs": int(uniq.size)}


def floaters(d, tau=2.0, min_frames=2):
    z = np.asarray(d["z"], np.float64)
    key = np.asarray(d["key"], np.int64)
    kf = np.asarray(d["kf"], np.int64)
    med = {}
    order = np.argsort(key, kind="stable")
    ks, zs, z_order_kf = key[order], z[order], kf[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[start, ks.size])
    medv = np.array([np.median(zs[s:s + c]) for s, c in zip(start, cnt)])
    medarr = np.repeat(medv, cnt)
    # distinct frames per cell, not the number of rows
    p2 = ks * 1000003 + z_order_kf
    uniq, inv = np.unique(p2, return_inverse=True)
    dframes = np.bincount(np.r_[0, np.cumsum(ks[1:] != ks[:-1])], minlength=ks.size)
    # count distinct kf per key directly
    kf_s = z_order_kf
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], ks.size]
    df = np.array([np.unique(kf_s[a:b]).size for a, b in zip(starts, ends)])
    nframes = np.repeat(df, cnt)
    bad = (zs > medarr + tau) & (nframes < min_frames)
    return {"n_floaters": int(bad.sum()), "floater_frac": float(bad.mean()),
            "tau": tau, "min_frames": min_frames}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", default="results/system/p42_metrics.json")
    a = ap.parse_args(argv)
    d = np.load(pathlib.Path(a.obs) / "obs_merged.npz")
    res = {"obs": a.obs, "edge": edge_duplication(d), "floaters": floaters(d)}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
