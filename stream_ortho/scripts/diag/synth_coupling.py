"""T37-A -- is the calibration x assignment coupling a property of 2.5D layered maps?

A controlled world is generated: a ground plane, rectangular buildings, and N aerial frames
that each observe the *top surface* of a window of cells.  Every frame carries its own 3-DOF
bias ``b + tx u + ty v`` plus noise, which is exactly the failure mode the real method fixes.

Three pipelines are compared:

``raw``       no correction, product from the biased frames
``unfrozen``  estimate theta per frame against the product, correct, then **re-fuse**
              (layers and owners re-elected) -- the "strip adjustment then merge" proxy
``frozen``    the same theta, but the product is rebuilt over the **frozen** assignment

Metrics: owner switches, boundary bias (median |z_A - z_B| between adjacent cells with
different owners, on the cells the two frames share), LCR (fraction of adjacent cell pairs
whose top differs by <= 0.2 m), and the frame-to-product PC.

Sweep: frames-per-cell (overlap) x bias magnitude.

Usage::

    python -m stream_ortho.scripts.diag.synth_coupling --out results/week10-pivotb/synth
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np


def make_world(rng, n_cells=160, cell=0.45, n_buildings=24, max_h=14.0):
    """Ground + rectangular buildings on an (n_cells, n_cells) grid."""
    z = np.zeros((n_cells, n_cells), np.float32)
    for _ in range(n_buildings):
        w = rng.integers(6, 22)
        h = rng.integers(6, 22)
        i = rng.integers(0, n_cells - w)
        j = rng.integers(0, n_cells - h)
        z[i:i + w, j:j + h] = rng.uniform(4.0, max_h)
    return z


def observe(z, cell, rng, n_frames, frame_cells, bias_sigma, noise=0.05):
    """Per-frame observations of the top surface with a 3-DOF bias (vectorised)."""
    n = z.shape[0]
    x0 = rng.uniform(0, 20)
    ks, is_, js, zs = [], [], [], []
    for k in range(n_frames):
        i0 = int(rng.integers(0, max(n - frame_cells, 1)))
        j0 = int(rng.integers(0, max(n - frame_cells, 1)))
        ii, jj = np.mgrid[i0:min(i0 + frame_cells, n), j0:min(j0 + frame_cells, n)]
        x = (jj + 0.5) * cell
        y = (ii + 0.5) * cell
        b = rng.normal(0, bias_sigma)
        tx = rng.normal(0, bias_sigma / (frame_cells * cell)) * 0.5
        ty = rng.normal(0, bias_sigma / (frame_cells * cell)) * 0.5
        zz = z[ii, jj] + b + tx * (x - x0) + ty * (y - x0) + rng.normal(0, noise, x.shape)
        ks.append(np.full(ii.size, k, np.int32))
        is_.append(ii.ravel().astype(np.int32))
        js.append(jj.ravel().astype(np.int32))
        zs.append(zz.ravel())
    return (np.concatenate(ks), np.concatenate(is_), np.concatenate(js),
            np.concatenate(zs).astype(np.float64))


def build(kf, ii, jj, zz, n_cells, cell, owner_by_weight=True):
    """Top surface per cell (max z) and its owner (the frame that saw that maximum)."""
    flat = ii * n_cells + jj
    top = np.full(n_cells * n_cells, -np.inf)
    own = np.full(n_cells * n_cells, -1, np.int32)
    order = np.lexsort((-zz, flat))            # per cell, highest z first
    f2, i2, j2, z2, k2 = flat[order], ii[order], jj[order], zz[order], kf[order]
    first = np.r_[0, np.nonzero(np.diff(f2))[0] + 1]
    top[f2[first]] = z2[first]
    own[f2[first]] = k2[first]
    if not owner_by_weight:
        cnt = np.bincount(f2, minlength=n_cells * n_cells)
        own[f2[first]] = k2[first]
    return top.reshape(n_cells, n_cells), own.reshape(n_cells, n_cells)


def metrics(top, own, cell, step=0.2, bnd_thr=0.2):
    live = np.isfinite(top)
    n_switch = 0
    diffs = []
    # horizontal and vertical neighbours
    for axis in (0, 1):
        if axis == 1:
            a = (slice(None), slice(0, -1)); b = (slice(None), slice(1, None))
        else:
            a = (slice(0, -1), slice(None)); b = (slice(1, None), slice(None))
        m = live[a] & live[b]
        n_switch += int((m & (own[a] != own[b])).sum())
        d = np.abs(top[a] - top[b])[m]
        diffs.append(d)
    d = np.concatenate(diffs)
    return {"owner_switch": n_switch,
            "boundary_bias_median": float(np.median(d)),
            "boundary_bias_p95": float(np.percentile(d, 95)),
            "lcr": float(np.mean(d <= step))}


def fit_frames(kf, ii, jj, zz, top, cell, origin, n_frames, huber=4):
    """Per-frame 3-DOF fit against the current product top."""
    thetas = np.zeros((n_frames, 3))
    for k in range(n_frames):
        m = kf == k
        if m.sum() < 50:
            continue
        x = (jj[m] + 0.5) * cell - origin
        y = (ii[m] + 0.5) * cell - origin
        d = zz[m] - top[ii[m], jj[m]]
        ok = np.isfinite(d)
        A = np.stack([np.ones(ok.sum()), x[ok], y[ok]], 1)
        dd = d[ok]
        p = np.linalg.lstsq(A, dd, rcond=None)[0]
        for _ in range(huber):
            r = dd - A @ p
            s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-2)
            h = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
            p = np.linalg.lstsq(A * h[:, None], dd * h, rcond=None)[0]
        thetas[k] = p
    return thetas


def run_case(seed, n_frames, frame_cells, bias_sigma, n_cells=160, cell=0.45, iters=2,
             freeze=True):
    rng = np.random.default_rng(seed)
    z = make_world(rng, n_cells=n_cells, cell=cell)
    kf, ii, jj, zz = observe(z, cell, rng, n_frames, frame_cells, bias_sigma)
    origin = (n_cells * cell) / 2
    top, own = build(kf, ii, jj, zz, n_cells, cell)
    m_raw = metrics(top, own, cell)
    for _ in range(iters):
        th = fit_frames(kf, ii, jj, zz, np.where(np.isfinite(top), top, 0.0), cell, origin,
                        n_frames)
        zc = zz - (th[kf, 0] + th[kf, 1] * ((jj + 0.5) * cell - origin) +
                   th[kf, 2] * ((ii + 0.5) * cell - origin))
        if freeze:
            # keep the original assignment, only move the heights
            flat = ii * n_cells + jj
            acc = np.zeros(n_cells * n_cells); cnt = np.zeros(n_cells * n_cells)
            np.add.at(acc, flat, zc); np.add.at(cnt, flat, 1.0)
            top = (acc / np.maximum(cnt, 1)).reshape(n_cells, n_cells)
        else:
            top, own = build(kf, ii, jj, zc, n_cells, cell)
    m = metrics(top, own, cell)
    return {"raw": m_raw, **({"frozen": m} if freeze else {"unfrozen": m})}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/week10-pivotb/synth")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)

    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    overlaps = {"2": 220, "3": 160, "4": 120, "6": 80}       # frame size -> frames per cell
    biases = [0.2, 0.5, 1.0]
    res = {"settings": {"overlap_frames_per_cell": list(overlaps), "bias_sigma": biases,
                        "n_seeds": a.seeds, "cell": 0.45},
           "grid": []}
    for label, fc in overlaps.items():
        for bs in biases:
            accs = {}
            for s in range(a.seeds):
                for fr in (True, False):
                    r = run_case(seed=1000 + s, n_frames=24, frame_cells=fc, bias_sigma=bs,
                                 freeze=fr)
                    for key in ("raw", "frozen", "unfrozen"):
                        if key in r:
                            accs.setdefault(key, []).append(r[key])
            row = {"overlap": label, "bias_sigma": bs}
            for key, vals in accs.items():
                row[key] = {m: float(np.median([v[m] for v in vals])) for m in vals[0]}
            res["grid"].append(row)
            print(f"overlap {label} bias {bs}: raw sw {row['raw']['owner_switch']:.0f} "
                  f"lcr {row['raw']['lcr']:.3f} bias {row['raw']['boundary_bias_median']:.3f} | "
                  f"unfrozen sw {row.get('unfrozen', {}).get('owner_switch', float('nan')):.0f} "
                  f"lcr {row.get('unfrozen', {}).get('lcr', float('nan')):.3f} | "
                  f"frozen lcr {row.get('frozen', {}).get('lcr', float('nan')):.3f} "
                  f"bias {row.get('frozen', {}).get('boundary_bias_median', float('nan')):.3f}")
    res["wall_s"] = time.time() - t0
    (out / "synth_coupling.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("->", out / "synth_coupling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
