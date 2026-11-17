"""Stage timers for the streaming pipeline (design doc §7.7 item 3).

Usage::

    timers = StageTimer()
    with timers("head"):
        patch = plane_patch(...)
    with timers("commit"):
        gmap.commit(kf)
    ...
    timers.summary()   # {'head': {'n':.., 'mean_ms':.., 'p50_ms':.., 'p95_ms':..}, ...}

The timers are cheap (perf_counter) and deliberately allocation-free so they can
stay in the streaming loop of week 7/8.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List


class StageTimer:
    def __init__(self):
        self.samples: Dict[str, List[float]] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.samples.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)

    def add(self, name: str, ms: float):
        self.samples.setdefault(name, []).append(float(ms))

    def get(self, name: str) -> List[float]:
        return self.samples.get(name, [])

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, dict]:
        out = {}
        for name, xs in self.samples.items():
            if not xs:
                continue
            import numpy as np

            a = np.asarray(xs, dtype=np.float64)
            out[name] = {
                "n": int(a.size),
                "mean_ms": round(float(a.mean()), 3),
                "p50_ms": round(float(np.percentile(a, 50)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "max_ms": round(float(a.max()), 3),
                "total_ms": round(float(a.sum()), 2),
            }
        return out

    def add_totals(self, name: str, ms: float):
        """Record an aggregate (e.g. end-to-end per keyframe) as one sample."""
        self.add(name, ms)
