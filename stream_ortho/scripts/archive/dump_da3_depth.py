"""Dump depth maps from Depth-Anything-3 (metric) for a sequence's keyframes.

The DA3 metric checkpoint is available locally under
``SLAM+GS/Depth-Anything-3/checkpoints/DA3METRIC-LARGE`` (1.3 GB).  We run it on
the *full-resolution* keyframes to get a per-pixel metric depth that is
independent of the (0.78 m/point) MASt3R point map — this is the candidate
"dense geometry source" for the DSM (week8-end-7).

Outputs ``results/stage-S12/<run>/da3_depth.npz`` with
``depth[k]`` (float16), ``conf[k]``, ``frame_id[k]`` and ``K``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np


def _stub_optional_deps():
    """DA3's ``utils.export.gs`` imports ``moviepy`` (video export only).

    We never export a video, so shim it with a no-op module instead of adding a
    dependency to the (working, pinned) ``2dgs-slam`` environment.
    """
    import types

    stub = pathlib.Path("/tmp/dastub")
    if stub.is_dir() and str(stub) not in sys.path:
        sys.path.insert(0, str(stub))


def build_parser():
    p = argparse.ArgumentParser(description="dump DA3 metric depth for keyframes")
    p.add_argument("--run", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--max-kf", type=int, default=8)
    p.add_argument("--ckpt", default="SLAM+GS/Depth-Anything-3/checkpoints/DA3METRIC-LARGE")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    from stream_ortho import config as zcfg
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset

    root = pathlib.Path(__file__).resolve().parents[2]          # .../PROJ/code
    sys.path.insert(0, str(root / "SLAM+GS" / "Depth-Anything-3" / "src"))
    _stub_optional_deps()
    from depth_anything_3.api import DepthAnything3

    run = pathlib.Path(a.run)
    info = json.loads((run / "run.json").read_text())
    ds = NPUDroneMapDataset(pathlib.Path(a.seq) if a.seq else zcfg.dataset_root() / info["dataset"])
    npz = np.load(run / "keyframes.npz")
    T = npz["T_WC"]
    sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0][: a.max_kf].tolist()

    ckpt = (root / a.ckpt) if not pathlib.Path(a.ckpt).is_absolute() else pathlib.Path(a.ckpt)
    print(f"[da3] loading {ckpt}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(str(ckpt))
    model = model.to(a.device).eval()
    print(f"[da3] loaded in {time.time()-t0:.1f}s")

    depths, confs, fids = [], [], []
    for k in kfs:
        fid = int(npz["frame_id"][k])
        img = ds.read_img(fid)                       # HxWx3 uint8
        t1 = time.time()
        pred = model.inference([img])                 # list of RGB -> prediction
        d = pred.depth[0] if hasattr(pred, "depth") else pred["depth"][0]
        c = pred.conf[0] if hasattr(pred, "conf") and pred.conf is not None else np.ones_like(d)
        d = np.asarray(d, np.float32)
        c = np.asarray(c, np.float32)
        if d.shape != (img.shape[0], img.shape[1]):
            import cv2

            d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            c = cv2.resize(c, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        depths.append(d.astype(np.float16))
        confs.append(c.astype(np.float16))
        fids.append(fid)
        print(f"  [da3] kf{k:3d} fid{fid:4d} depth med={np.median(d):8.2f} m "
              f"p5={np.percentile(d,5):8.2f} p95={np.percentile(d,95):8.2f}  {time.time()-t1:.2f}s")

    out = pathlib.Path(a.out) if a.out else zcfg.RESULTS_DIR / "stage-S12" / run.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "da3_depth.npz",
                        depth=np.stack(depths), conf=np.stack(confs),
                        frame_id=np.asarray(fids, np.int32), K=ds.K_orig)
    print(f"[da3] -> {out/'da3_depth.npz'}  ({len(fids)} frames)")


if __name__ == "__main__":
    main()
