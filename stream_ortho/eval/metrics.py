"""Frozen ortho metrics (week9-00, Day 0).

Every number in the paper must come from this file — no hand-copied values.

Definitions (frozen, see docs/METRICS.md):

* ``valid``      alpha >= 0.5 (from ``coverage.npy`` if the exporter saved it, otherwise
                 the black-pixel mask ``gray >= 8`` — reported as ``valid_source``).
* ``coverage``   valid.sum() / extent_px, extent = the fixed union bbox at this GSD.
* ``holes``      connected components of ``~valid`` with area < 60 px.
* ``sharpness``  mean(|Laplacian|) inside ``valid`` eroded by 3 px; plus gradient p90.
* ``patchwork``  on valid pixels in the lowest 20% of the gradient, mean(|Ix|+|Iy|).
* ``DER``        two-edge ratio on 15 hand-annotated roof edges (eval/edges_<seq>.json):
                 fraction of edges whose profile shows >= 2 significant gradient peaks.
* ``LSR``        RMS distance (px) of the profile maxima from a fitted straight line.
* ``ms_*``       P50/P95/max read from stats.json (never sum log lines).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def _load_gray(png: pathlib.Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(png))
    if im is None:
        raise FileNotFoundError(png)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def valid_mask(run: pathlib.Path, mode: str = "nonblack"):
    """(valid, source).  ``mode='nonblack'`` is the **frozen** definition (week9-07):
    it is what the S58 B baseline can reproduce (its alpha was not saved), so every
    product is measured with it; alpha is kept as a diagnostic only."""
    g = _load_gray(run / "ortho_product.png")
    if mode == "alpha":
        cov = run / "coverage.npy"
        if cov.exists():
            return (np.load(cov) >= 0.5), "alpha"
    return (g >= 8), "nonblack"


def holes(mask: np.ndarray, max_area: int = 60):
    import cv2

    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    small = areas[(areas > 0) & (areas < max_area)]
    return {"count": int(small.size), "area_px": int(small.sum())}


def sharpness(gray: np.ndarray, mask: np.ndarray):
    import cv2

    m = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if m.sum() < 1000:
        return {"lap": float("nan"), "grad_p90": float("nan")}
    g = gray.astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    grad = np.hypot(gx, gy)
    flat = m & (grad < np.percentile(grad[m], 20))
    return {"lap": float(lap[m].mean()), "grad_p90": float(np.percentile(grad[m], 90)),
            # median is useless here: |Laplacian| of 8-bit grey is integer-valued, so the
            # median is 1 for every product.  The **mean** over flat pixels is what
            # separates smooth from grainy.
            "grain": float(lap[flat].mean()) if flat.any() else float("nan")}


def patchwork(gray: np.ndarray, mask: np.ndarray):
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1))            # (H, W-1)
    gy = np.abs(np.diff(g, axis=0))            # (H-1, W)
    m = mask[:, :-1] & mask[:, 1:]             # same-shape mask for gx
    m2 = mask[:-1, :] & mask[1:, :]            # same-shape mask for gy
    if m.sum() < 1000:
        return float("nan")
    thr = np.percentile(gx[m], 20)
    sel = m & (gx < thr)
    thr2 = np.percentile(gy[m2], 20)
    sel2 = m2 & (gy < thr2)
    vals = np.concatenate([gx[sel], gy[sel2]])
    return float(vals.mean()) if vals.size else float("nan")


def _profile_peaks(prof: np.ndarray, prominence: float = 0.2, min_gap: int = 2):
    from scipy.signal import find_peaks

    if prof.size < 5:
        return np.zeros(0, int)
    p = prof - np.median(prof)
    if p.max() <= 0:
        return np.zeros(0, int)
    idx, _ = find_peaks(p, prominence=prominence * p.max(), distance=min_gap)
    return idx


def edges(gray: np.ndarray, edges_json: pathlib.Path):
    """DER (>=2 peaks) and LSR (RMS of the maxima about a fitted line), in pixels."""
    if not edges_json.exists():
        return {"der": float("nan"), "lsr_px": float("nan"), "n_edges": 0}
    spec = json.loads(edges_json.read_text())
    two, rms = 0, []
    for e in spec:
        x0, y0, x1, y1 = e["p0"][0], e["p0"][1], e["p1"][0], e["p1"][1]
        n = int(max(abs(x1 - x0), abs(y1 - y0)))
        if n < 10:
            continue
        ts = np.linspace(0, 1, n)
        xs = np.clip((x0 + (x1 - x0) * ts).astype(int), 0, gray.shape[1] - 1)
        ys = np.clip((y0 + (y1 - y0) * ts).astype(int), 0, gray.shape[0] - 1)
        prof = gray[ys, xs].astype(np.float32)
        pk = _profile_peaks(np.abs(np.diff(prof)))
        if len(pk) >= 2:
            two += 1
            pk = pk.astype(np.float32)
            # fit a straight line along the edge and measure the peak deviation
            t = np.linspace(0, 1, len(prof))
            A = np.stack([t, np.ones_like(t)], 1)
            coef, *_ = np.linalg.lstsq(A, prof, rcond=None)
            resid = prof[pk] - (A[pk] @ coef)
            rms.append(float(np.sqrt((resid ** 2).mean())))
    return {"der": two / max(len(spec), 1), "lsr_px": float(np.mean(rms)) if rms else float("nan"),
            "n_edges": len(spec)}


def ms_stats(run: pathlib.Path):
    st = run / "stats.json"
    if not st.exists():
        return {}
    frames = json.loads(st.read_text())["frames"]
    out = {}
    for k in ("ms_patch", "ms_map", "ms_render"):
        v = np.array([f.get(k, np.nan) for f in frames], float)
        v = v[np.isfinite(v)]
        if v.size:
            out[k] = {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                      "max": float(v.max()), "sum_s": float(v.sum() / 1000.0)}
    out["n_frames"] = len(frames)
    if frames and "n_gauss" in frames[-1]:
        out["n_gauss_last"] = int(frames[-1]["n_gauss"])
    return out


def measure(run: str | pathlib.Path, seq: str | None = None):
    run = pathlib.Path(run)
    gray = _load_gray(run / "ortho_product.png")
    mask, src = valid_mask(run, "nonblack")
    meta = json.loads((run / "ortho_product.json").read_text()) if (run / "ortho_product.json").exists() else {}
    extent_px = int(meta.get("size_px", [gray.shape[1], gray.shape[0]])[0]) * \
        int(meta.get("size_px", [gray.shape[1], gray.shape[0]])[1])
    out = {
        "run": str(run), "valid_source": src,
        "coverage": float(mask.sum() / max(extent_px, 1)),
        "coverage_px": int(mask.sum()), "extent_px": extent_px,
        **holes(mask),
        **sharpness(gray, mask),
        "patchwork": patchwork(gray, mask),
        "ms": ms_stats(run),
        "gsd": meta.get("gsd_m_per_px"),
    }
    if seq:
        ej = pathlib.Path(__file__).with_name(f"edges_{seq}.json")
        out.update(edges(gray, ej))
    return out


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seq", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = [measure(r, a.seq) for r in a.runs]
    txt = json.dumps(res, indent=2)
    if a.out:
        pathlib.Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
