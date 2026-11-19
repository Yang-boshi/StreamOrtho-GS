"""S2: MASt3R-SLAM baseline on one NPU sequence, with dumps for evaluation.

This is a *copy* of the upstream ``main.py`` loop (design doc 11.7: main.py is
read-only).  Differences, all deliberate:

* no visualisation, no ``save_reconstruction`` -> the main path never writes a
  ``.ply`` (hard constraint 2.5);
* no ``.ply``/point-cloud intermediate at all: we dump the keyframe tensors
  (``X_canon``, ``C``, ``T_WC``) so the S2 consistency check can use
  ``T_WC.act(X_canon)`` directly;
* per-frame and per-keyframe timing, device memory sampling;
* a RELOC frame is never appended as a new keyframe (upstream keeps the stale
  ``add_new_kf`` value across iterations, which can duplicate a relocalised
  keyframe).

Usage::

    python -m stream_ortho.scripts.run_mast3r_baseline \
        --dataset /home/zxy/syb/NPU/gopro-monticules-kfs \
        --out results/stage-S2/gopro-monticules --max-frames 60   # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np

from stream_ortho import config as zcfg
from stream_ortho import compat
from stream_ortho.data.npu_dataset import NPUDroneMapDataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MASt3R-SLAM baseline (S2)")
    p.add_argument("--dataset", required=True, help="NPU sequence or RGB folder")
    p.add_argument("--dataset-format", choices=("npu", "rgb"), default="npu",
                   help="rgb: a plain image folder without config.cfg")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--config", default=None, help="upstream yaml (default config/base.yaml)")
    p.add_argument("--calib", action="store_true", help="use config.cfg / provided / exif intrinsics")
    p.add_argument("--intrinsics", default="auto", choices=("auto", "exif", "provided", "none"),
                   help="intrinsics source when dataset_format=rgb")
    p.add_argument("--intrinsics-value", default=None,
                   help="fx,fy,cx,cy when --intrinsics provided")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1, help="image stride (after sorting)")
    p.add_argument("--img-downsample", type=int, default=1, help="upstream img_downsample")
    p.add_argument("--subsample", type=int, default=1, help="upstream dataset subsample")
    p.add_argument("--single-thread", action="store_true")
    p.add_argument("--kf-buffer", type=int, default=1024)
    p.add_argument("--no-dump-tensors", action="store_true", help="skip keyframe npz")
    p.add_argument("--save-frames", type=int, default=0, help="also save N jpgs (debug)")
    return p


def _sim3_vec(T_WC) -> np.ndarray:
    """Full lietorch Sim3 vector: (tx,ty,tz,qx,qy,qz,qw,s) -- 8 numbers.

    The upstream SE3 conversion (``as_SE3``) drops the scale; we keep it so the
    world transform is exact and the ``s == 1`` assumption is *verified* rather
    than assumed.
    """
    data = T_WC.data.detach().cpu().numpy().reshape(-1)
    return np.asarray(data, dtype=np.float64)


#: set by ``app.frontend.LiveFrontend`` to receive every committed keyframe in-process
KEYFRAME_HOOK = None

#: set by ``app.frontend.LiveFrontend`` to receive backend BA pose updates: fn(pose_dict)
#: where pose_dict maps keyframe index k -> (8,) Sim3 vector.
POSE_UPDATE_HOOK = None

#: active MASt3R model reference for in-process feature/decoder reuse
ACTIVE_MODEL = None


def _frame_pose(frame) -> np.ndarray:
    return _sim3_vec(frame.T_WC)


def _emit_pose_updates(keyframes, last_poses: dict, last_ver: int, force: bool = False) -> int:
    """Check ``keyframes.pose_version`` and emit changed keyframe poses to POSE_UPDATE_HOOK."""
    hook = POSE_UPDATE_HOOK
    if hook is None:
        return last_ver
    cur_ver = keyframes.get_pose_version()
    if not force and cur_ver == last_ver:
        return cur_ver
    n_kf = len(keyframes)
    if n_kf == 0:
        return cur_ver
    with keyframes.lock:
        T_all = keyframes.T_WC[:n_kf].detach().cpu().numpy().reshape(n_kf, -1).astype(np.float64)
    changed = {}
    for k in range(n_kf):
        prev = last_poses.get(k)
        if prev is None:
            continue
        vec = T_all[k]
        dt = float(np.linalg.norm(vec[:3] - prev[:3]))
        dq = float(np.linalg.norm(vec[3:7] - prev[3:7]))
        ds = float(abs(vec[7] - prev[7]))
        # During live streaming, ignore sub-pixel BA jitter (<0.005) to avoid
        # redundant tile re-renders; force=True at end-of-stream syncs everything.
        thresh = 1e-4 if force else 5e-3
        if dt > thresh or dq > thresh or ds > thresh:
            changed[k] = vec.copy()
            last_poses[k] = vec.copy()
    if changed:
        try:
            hook(changed)
        except Exception as exc:
            print(f"[pose-hook] {exc!r}", flush=True)
    return cur_ver


def run(args) -> dict:
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = pathlib.Path(args.dataset).expanduser().resolve()

    # --- import upstream (read-only) ------------------------------------
    repo = compat.ensure_upstream_on_path()
    # Ensure third_party/MASt3R-SLAM/checkpoints resolves so backend retrieval_database loads
    up_ckpt = zcfg.MAST3R_SLAM_DIR / "checkpoints"
    if (zcfg.CKPT_DIR / "mast3r").is_dir():
        if up_ckpt.is_symlink() and not up_ckpt.exists():
            up_ckpt.unlink()
        if not up_ckpt.exists():
            try:
                up_ckpt.symlink_to(zcfg.CKPT_DIR / "mast3r")
            except Exception:
                pass
    compat.load_upstream_config(args.config)
    import main as ms_main  # upstream main.py, imported as a module

    from mast3r_slam.config import config
    from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
    from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
    from mast3r_slam.tracker import FrameTracker
    import lietorch
    import torch
    import torch.multiprocessing as mp

    if args.config:
        compat.load_upstream_config(args.config)
    config["single_thread"] = bool(args.single_thread)
    config["dataset"]["img_downsample"] = int(args.img_downsample)
    config["dataset"]["subsample"] = int(args.subsample)
    if args.calib:
        config["use_calib"] = True
    print(f"[S2] config: use_calib={config['use_calib']} img_downsample={config['dataset']['img_downsample']}")

    if args.dataset_format == "rgb":
        # a plain folder of images: RGBFolderDataset provides the same contract
        from stream_ortho.data.rgb_folder import RGBFolderDataset
        iv = ([float(v) for v in args.intrinsics_value.split(",")]
              if getattr(args, "intrinsics_value", None) else None)
        intr_mode = args.intrinsics if config["use_calib"] else "none"
        dataset = RGBFolderDataset(
            dataset_dir, image_size=zcfg.IMG_SIZE, stride=args.stride,
            intrinsics=intr_mode, intrinsics_value=iv,
            max_frames=args.max_frames)
    else:
        dataset = NPUDroneMapDataset(
            dataset_dir,
            img_size=zcfg.IMG_SIZE,
            stride=args.stride,
            use_calib=bool(config["use_calib"]),
            max_frames=args.max_frames,
        )
    if args.subsample > 1:
        dataset.subsample(args.subsample)
    print(f"[S2] dataset {dataset.name}: {len(dataset)} frames, shape={dataset.frame_shape()}")

    device = zcfg.DEVICE
    _shape = dataset.get_img_shape()
    h, w = (_shape[0] if isinstance(_shape[0], (tuple, list)) else _shape)
    print(f"[S2] keyframe frame size h,w = {int(h)},{int(w)}")

    kf_buf = min(int(args.kf_buffer), max(64, len(dataset) + 8))
    print(f"[S2] SharedKeyframes buffer={kf_buf} (requested={args.kf_buffer}, frames={len(dataset)})")
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    keyframes = SharedKeyframes(manager, h, w, buffer=kf_buf)
    states = SharedStates(manager, h, w)

    global ACTIVE_MODEL
    model = load_mast3r(str(zcfg.MAST3R_CKPT), device=device)
    model.share_memory()
    ACTIVE_MODEL = model

    use_calib = bool(config["use_calib"])
    K = None
    if use_calib:
        if hasattr(dataset, "has_calib") and not dataset.has_calib():
            print("[S2] warning: use_calib=True requested, but dataset has no metric calib; "
                  "falling back to use_calib=False", flush=True)
            use_calib = False
            config["use_calib"] = False
        else:
            if getattr(dataset, "camera_intrinsics", None) is not None:
                K_np = dataset.camera_intrinsics.K_frame
            elif hasattr(dataset, "K_slam_frame"):
                K_np = dataset.K_slam_frame()
            else:
                K_np = dataset.K_frame()
            K = torch.from_numpy(np.asarray(K_np)).to(device, dtype=torch.float32)
            keyframes.set_intrinsics(K)

    tracker = FrameTracker(model, keyframes, device)
    backend = ctx.Process(
        target=ms_main.run_backend, args=(config, model, states, keyframes, K)
    )
    backend.start()

    # --- main loop (mirrors upstream main.py) ---------------------------
    frame_rows = []          # all processed frames
    timing_rows = []         # per-frame latency
    device_mem = []          # sampled device memory (MB)
    n_kf_seen = 0
    i = 0
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    frames_to_save = []
    stats = {"n_reloc": 0, "n_kf": 0}
    last_poses = {}
    seen_pose_ver = 0
    emitted_kf_idx = 0
    ds_img_size = (int(dataset.img_size[0])
                   if isinstance(dataset.img_size, (tuple, list))
                   else int(dataset.img_size))

    while True:
        mode = states.get_mode()
        if mode == Mode.TERMINATED:
            break
        if i >= len(dataset):
            break

        timestamp, img = dataset[i]
        if isinstance(img, torch.Tensor) and img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).cpu().numpy()
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=ds_img_size, device=device)

        if args.save_frames and i < args.save_frames:
            frames_to_save.append((i, np.asarray(img)))

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            stats["n_kf"] += 1
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            add_new_kf = False
        elif mode == Mode.TRACKING:
            t0 = time.time()
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            dt = time.time() - t0
            if try_reloc:
                stats["n_reloc"] += 1
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            if add_new_kf:
                keyframes.append(frame)
                kf_idx = len(keyframes) - 1
                stats["n_kf"] += 1
                states.queue_global_optimization(kf_idx)
                # Emit previous keyframe(s) now that their multi-view tracking window
                # has completed and X_canon has fused N=5..24 stereo observations.
                while emitted_kf_idx < kf_idx:
                    kf_ready = keyframes[emitted_kf_idx]
                    last_poses[emitted_kf_idx] = _frame_pose(kf_ready)
                    _emit_keyframe(emitted_kf_idx, kf_ready)
                    emitted_kf_idx += 1
            if config["single_thread"]:
                with states.lock:
                    while len(states.global_optimizer_tasks) > 0:
                        pass
            seen_pose_ver = _emit_pose_updates(keyframes, last_poses, seen_pose_ver)
            timing_rows.append((i, float(timestamp), dt, int(bool(add_new_kf))))
        elif mode == Mode.RELOC:
            t0 = time.time()
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            if config["single_thread"]:
                while True:
                    with states.lock:
                        if states.reloc_sem.value == 0:
                            break
                    time.sleep(0.005)
            # NOTE: a relocalisation frame is NOT appended here; the backend's
            # relocalization() appends it if the loop closure succeeds.
            add_new_kf = False
            timing_rows.append((i, float(timestamp), time.time() - t0, 0))
        else:
            raise RuntimeError(f"invalid mode {mode}")

        frame_rows.append(
            (i, float(timestamp), *_frame_pose(frame), int(bool(add_new_kf)),
             len(keyframes) - 1 if add_new_kf else -1)
        )

        if i % 10 == 0:
            free, total = torch.cuda.mem_get_info()
            device_mem.append((i, (total - free) / 1e6))
        if i % 25 == 0:
            if torch.cuda.is_available():
                if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > 192 * 1024 * 1024:
                    torch.cuda.empty_cache()
            elapsed = time.time() - t_start
            print(
                f"[S2] frame {i:4d}/{len(dataset)}  kf={len(keyframes):3d}  "
                f"mode={mode}  {i / max(elapsed, 1e-9):5.2f} fps"
            )
        i += 1

    # --- flush final keyframe(s) and drain backend BA before terminating ---
    while emitted_kf_idx < len(keyframes):
        kf_ready = keyframes[emitted_kf_idx]
        last_poses[emitted_kf_idx] = _frame_pose(kf_ready)
        _emit_keyframe(emitted_kf_idx, kf_ready)
        emitted_kf_idx += 1
    drain_t0 = time.time()
    while time.time() - drain_t0 < 15.0:
        with states.lock:
            pending = len(states.global_optimizer_tasks)
        if pending == 0:
            break
        time.sleep(0.02)
    _emit_pose_updates(keyframes, last_poses, seen_pose_ver, force=True)

    # --- stop backend ---------------------------------------------------
    states.set_mode(Mode.TERMINATED)
    backend.join(timeout=120)
    if backend.is_alive():
        backend.terminate()
        backend.join(timeout=30)
    total_time = time.time() - t_start
    n_kf_final = len(keyframes)
    print(f"[S2] done: {i} frames, {n_kf_final} keyframes, {total_time:.1f}s "
          f"({i / max(total_time,1e-9):.2f} fps)")

    # --- dumps ----------------------------------------------------------
    stats.update(
        {
            "dataset": dataset.name,
            "dataset_dir": str(dataset_dir),
            "n_frames_requested": len(dataset),
            "n_frames_processed": i,
            "n_keyframes": n_kf_final,
            "n_keyframes_shared": args.kf_buffer,
            "wall_time_s": round(total_time, 3),
            "fps": round(i / max(total_time, 1e-9), 3),
            "use_calib": use_calib,
            "img_downsample": int(args.img_downsample),
            "stride": int(args.stride),
            "single_thread": bool(args.single_thread),
            "frame_hw": [int(h), int(w)],
            "reloc_events": stats["n_reloc"],
            "peak_device_mem_mb": round(max([m for _, m in device_mem], default=0.0), 1),
            "upstream_commit": compat.upstream_commit(),
            "mast3r_ckpt": str(zcfg.MAST3R_CKPT),
            "torch_version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
        }
    )
    if timing_rows:
        td = np.asarray([r[2] for r in timing_rows])
        stats["track_ms"] = {
            "mean": round(1000 * float(td.mean()), 2),
            "median": round(1000 * float(np.median(td)), 2),
            "p95": round(1000 * float(np.percentile(td, 95)), 2),
            "max": round(1000 * float(td.max()), 2),
        }
        kf_mask = np.asarray([r[3] for r in timing_rows], dtype=bool)
        if kf_mask.any():
            stats["track_ms_keyframes"] = round(1000 * float(td[kf_mask].mean()), 2)
        stats["n_tracking_frames"] = int(len(timing_rows))

    # per-frame poses (pre-global-optimisation for keyframes)
    with open(out_dir / "poses_all_frames.txt", "w") as f:
        f.write("# timestamp frame_id tx ty tz qx qy qz qw s is_keyframe keyframe_idx\n")
        for row in frame_rows:
            f.write(" ".join(str(v) for v in row) + "\n")

    # final optimised keyframe poses (comparable to upstream save_traj)
    kf_timestamps = []
    kf_poses = []
    with open(out_dir / "poses_keyframes.txt", "w") as f:
        # 8 TUM columns + scale (lietorch Sim3 scale, expected exactly 1.0)
        f.write("# timestamp tx ty tz qx qy qz qw s\n")
        for k in range(n_kf_final):
            kf = keyframes[k]
            ts = float(dataset.timestamps[kf.frame_id])
            pose = _frame_pose(kf)
            kf_timestamps.append(ts)
            kf_poses.append(pose)
            f.write(f"{ts:.9f} " + " ".join(f"{v:.9f}" for v in pose) + "\n")

    # keyframe tensors (the ONLY geometry dump; no .ply)
    if not args.no_dump_tensors and n_kf_final > 0:
        kh, kw = int(h), int(w)
        X = np.zeros((n_kf_final, kh, kw, 3), dtype=np.float16)
        C = np.zeros((n_kf_final, kh, kw, 1), dtype=np.float16)
        T = np.zeros((n_kf_final, 8), dtype=np.float64)
        # upstream's ``get_average_conf`` divides the accumulated confidence by the number of
        # pointmap updates; without N a per-frame cloud cannot reproduce its ``C/N`` filter
        Nupd = np.zeros((n_kf_final,), dtype=np.int32)
        fids = np.zeros((n_kf_final,), dtype=np.int32)
        tss = np.zeros((n_kf_final,), dtype=np.float64)
        for k in range(n_kf_final):
            kf = keyframes[k]
            X[k] = kf.X_canon.detach().cpu().numpy().astype(np.float16).reshape(kh, kw, 3)
            C[k] = kf.C.detach().cpu().numpy().astype(np.float16).reshape(kh, kw, 1)
            T[k] = _frame_pose(kf)
            try:
                Nupd[k] = max(int(getattr(kf, "N", 1)), 1)
            except Exception:
                Nupd[k] = 1
            fids[k] = int(kf.frame_id)
            tss[k] = float(dataset.timestamps[kf.frame_id])
        k_save = (dataset.K_slam_frame() if hasattr(dataset, "K_slam_frame")
                  else (dataset.K_frame() if callable(dataset.K_frame) else dataset.K_frame))
        np.savez(
            out_dir / "keyframes.npz",
            X_canon=X, C=C, N=Nupd, T_WC=T, frame_id=fids, timestamp=tss,
            K=k_save, hw=np.array([kh, kw]),
            use_calib=np.array([use_calib]),
        )
        stats["sim3_scale"] = {
            "min": float(np.min(T[:, 7])),
            "max": float(np.max(T[:, 7])),
            "median": float(np.median(T[:, 7])),
        }
        print(f"[S2] Sim3 scale: min={T[:,7].min():.6f} max={T[:,7].max():.6f} (expect 1.0)")
        print(f"[S2] keyframe tensors -> {out_dir/'keyframes.npz'} "
              f"({(out_dir/'keyframes.npz').stat().st_size/1e6:.1f} MB)")

    with open(out_dir / "timing.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame_id", "timestamp", "dt_s", "is_keyframe"])
        wtr.writerows(timing_rows)

    with open(out_dir / "device_mem_mb.csv", "w") as f:
        f.write("frame_id,mem_mb\n")
        for fid, m in device_mem:
            f.write(f"{fid},{m:.1f}\n")

    if frames_to_save:
        import cv2

        for fid, img in frames_to_save:
            cv2.imwrite(str(out_dir / f"frame_{fid:05d}.jpg"), img[:, :, ::-1])

    with open(out_dir / "run.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[S2] wrote {out_dir/'run.json'}")
    return stats


def _emit_keyframe(k: int, frame):
    """Hand a freshly committed keyframe to the in-process pipeline, if one is attached."""
    hook = KEYFRAME_HOOK
    if hook is None:
        return
    try:
        hook(k, frame)
    except Exception as exc:                      # never break the SLAM loop
        print(f"[kf-hook] {exc!r}", flush=True)
        import traceback; traceback.print_exc()


def main(argv=None):
    args = build_parser().parse_args(argv)
    # upstream uses relative checkpoint paths; run inside the repo
    out = pathlib.Path(args.out).expanduser().resolve()
    ds = pathlib.Path(args.dataset).expanduser().resolve()
    args.out, args.dataset = str(out), str(ds)
    compat.ensure_upstream_on_path()
    os.chdir(zcfg.MAST3R_SLAM_DIR)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    run(args)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    main()
