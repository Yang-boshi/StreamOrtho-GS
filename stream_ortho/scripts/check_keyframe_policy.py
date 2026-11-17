"""Evaluate the keyframe policy skeleton on a sequence's GT trajectory.

Reports how many frames the policy would keep vs MASt3R-SLAM's keyframe count,
and the reason histogram.  No GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from stream_ortho import config as zcfg
from stream_ortho.data.npu_dataset import NPUDroneMapDataset
from stream_ortho.frontend.keyframe_policy import KeyframePolicy, PolicyConfig


def main(argv=None):
    p = argparse.ArgumentParser(description="keyframe policy skeleton evaluation")
    p.add_argument("--seq", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-new-area", type=float, default=0.05)
    p.add_argument("--min-baseline", type=float, default=2.0)
    p.add_argument("--min-overlap", type=float, default=0.30)
    p.add_argument("--max-overlap", type=float, default=0.92)
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "stage-S4" / "keyframe_policy"))
    a = p.parse_args(argv)

    ds = NPUDroneMapDataset(a.seq)
    cfg = PolicyConfig(min_new_area=a.min_new_area, min_baseline_m=a.min_baseline,
                       min_overlap=a.min_overlap, max_overlap=a.max_overlap)
    policy = KeyframePolicy(cfg, ds.K_orig, ds.cfg.width, ds.cfg.height,
                            ds.T_WP, z_ground=0.0)
    reasons = {}
    n = 0
    for i in range(0, len(ds), a.stride):
        T = ds.gt_pose(i)
        if T is None:
            continue
        d = policy.decide(T)
        reasons[d.reason] = reasons.get(d.reason, 0) + 1
        n += 1
    st = policy.stats()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"seq": ds.name, "n_frames": n, "stride": a.stride,
           "cfg": {"min_new_area": a.min_new_area, "min_baseline": a.min_baseline,
                   "min_overlap": a.min_overlap, "max_overlap": a.max_overlap},
           "reasons": reasons, **st,
           "footprint_area_mean": st["mean_footprint_area_m2"]}
    tag = f"{ds.name}_n{a.min_new_area}_b{a.min_baseline}"
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[policy] {ds.name}: {n} frames -> {st['n_added']} keyframes "
          f"({100*st['accept_ratio']:.1f}%), mean footprint {st['mean_footprint_area_m2']:.0f} m^2")
    print(f"[policy] reasons: {reasons}")


if __name__ == "__main__":
    main()
