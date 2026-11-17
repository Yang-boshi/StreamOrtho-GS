"""Two-process snapshot rendering: map worker (A) publishes, renderer (B) consumes.

Why two *processes*: the map step (ingest/commit) and the render step both touch CUDA,
and calling CUDA from a second thread in the same process hangs (measured twice).  A
process boundary is the only clean way to overlap them.

Protocol (``/dev/shm/zso_snap`` by default):

* worker writes ``means/quats/scales/opacity/colors.npy`` then bumps ``version.txt``
  (written last, via tmp+rename, so a reader never sees a half-written snapshot);
* renderer polls ``version.txt``; when it changes it reloads the arrays **once** and
  keeps rendering that frozen snapshot until the next version appears.  It therefore
  never blocks on the map step: its frame time is pure render time.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np


def _write(p: pathlib.Path, arr):
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, p)


def publish(dirn: pathlib.Path, c: dict, version: int):
    dirn.mkdir(parents=True, exist_ok=True)
    for k, n in (("means", "means"), ("quats", "quats"), ("scales", "scales"),
                 ("opacity", "opacity"), ("colors", "colors")):
        _write(dirn / f"{n}.npy", np.ascontiguousarray(c[k], np.float32))
    tmp = dirn / "version.tmp"
    tmp.write_text(str(version))
    os.replace(tmp, dirn / "version.txt")


def read_version(dirn: pathlib.Path) -> int:
    try:
        return int((dirn / "version.txt").read_text().strip())
    except Exception:
        return -1


def load_snapshot(dirn: pathlib.Path):
    d = {n: np.load(dirn / f"{n}.npy") for n in
         ("means", "quats", "scales", "opacity", "colors")}
    return d


def worker(a):
    from stream_ortho import config as zcfg
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset
    from stream_ortho.frontend.gaussian_head import load_gaussian_head
    from stream_ortho.frontend.predict import load_model
    from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3
    from stream_ortho.map.build import head_adapt_patch
    from stream_ortho.map.freespace import NFPConfig, NadirHeightField
    from stream_ortho.map.gaussian_map import GaussianMap, MapConfig

    rd = pathlib.Path(a.run)
    info = json.loads((rd / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(rd / "keyframes.npz")
    m = json.loads((rd / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]; sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()[: a.max_kf]
    model = load_model(device=zcfg.DEVICE)
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, device=zcfg.DEVICE, verbose=False)
    gm = GaussianMap(MapConfig(voxel_size=0.25, dedup=True, gate=False, commit_min_obs=2,
                               commit_max_age=2, fusion="weighted"), ds.T_WP)
    nfp = NadirHeightField(NFPConfig(cell=1.0, eps=1.0), ds.T_WP) if a.nfp else None
    gm.nfp = nfp
    out = pathlib.Path(a.shm); ver = 0; t0 = time.time(); pub_t = []
    for kf in kfs:
        patch = head_adapt_patch(ds, npz, kf, sg, model, head, stride=a.stride,
                                 depth_source="pointmap", sigma_m=a.sigma,
                                 conf_min=0.2, edge_reject=a.edge_reject,
                                 edge_opacity=a.edge_opacity, device=zcfg.DEVICE)
        gm.ingest(patch, kf); gm.commit(kf)
        if a.collapse_every and (len(gm.committed["means"]) % 1 == 0) and \
                (kf % a.collapse_every == 0):
            pass
        keep = np.ones(len(gm.committed["means"]), bool)
        if a.collapse_k and (kfs.index(kf) + 1) % a.collapse_k == 0:
            from stream_ortho.map.selection import pack_cell, quality_score, select_by_cell
            c = gm.committed
            q = quality_score(conf=c["conf"], opacity=c["opacity"], wsum=c["wsum"],
                              n_obs=c["n_obs"])
            k1 = select_by_cell(c["means"], ds.T_WP, cell=a.cell, q=q, keep_n=1)
            kk = np.zeros(len(c["means"]), bool); kk[k1] = True
            keep = kk
        if (kfs.index(kf) + 1) % a.publish_every == 0 or kf == kfs[-1]:
            c = gm.committed
            ver += 1
            t1 = time.time()
            publish(out, {k: c[k][keep] for k in ("means", "quats", "scales",
                                                  "opacity", "colors")},
                    ver)
            pub_t.append(time.time() - t1)
            print(f"[A] kf {kf} N={int(keep.sum()):,} published v{ver} in "
                  f"{1000*(time.time()-t1):.0f} ms", flush=True)
    print(f"[A] done: {len(kfs)} keyframes, {ver} snapshots, {time.time()-t0:.0f}s", flush=True)


def render(a):
    import cv2
    import torch
    from stream_ortho import config as zcfg
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset
    from stream_ortho.ortho.grid import OrthoGrid
    from stream_ortho.ortho.rasterizer import GaussianCloud, render_tile
    from stream_ortho.ortho.tiles import Tile, stitch
    from stream_ortho.geometry import R_to_quat_xyzw, compose_sim3, sim3_from_vec8

    rd = pathlib.Path(a.run)
    info = json.loads((rd / "run.json").read_text())
    ds = NPUDroneMapDataset(zcfg.dataset_root() / info["dataset"])
    npz = np.load(rd / "keyframes.npz")
    m = json.loads((rd / "sim3_est_to_gt.json").read_text())
    sg = np.concatenate([m["translation"], R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])
    T = npz["T_WC"]; sc = T[:, 7]
    kfs = np.where(np.abs(np.log(sc / np.median(sc))) < np.log(1.3))[0]
    kfs = kfs[np.argsort(npz["frame_id"][kfs])].tolist()
    # fixed extent from the whole sequence (same as --fixed-extent)
    XP = []
    for kf in kfs:
        R_, t_, s_ = sim3_from_vec8(compose_sim3(T[kf], sg))
        X = np.asarray(npz["X_canon"][kf], np.float32)[::8, ::8].reshape(-1, 3)
        Pw = float(s_) * (X @ R_.T) + t_
        XP.append((Pw - ds.T_WP[:3, 3]) @ ds.T_WP[:3, :3])
    XP = np.concatenate(XP); XP = XP[(XP[:, 2] > -20) & (XP[:, 2] < 2)]
    grid = OrthoGrid(ds.T_WP, a.gsd, tile=512, halo=16, z_ground=0.0, z_top=800.0)
    i0, j0 = grid.tile_of(float(XP[:, 0].min()), float(XP[:, 1].min()))
    i1, j1 = grid.tile_of(float(XP[:, 0].max()), float(XP[:, 1].max()))
    i0, i1 = min(i0, i1), max(i0, i1); j0, j1 = min(j0, j1), max(j0, j1)
    tiles_ij = [(ii, jj) for jj in range(j0, j1 + 1) for ii in range(i0, i1 + 1)]
    out = pathlib.Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    shm = pathlib.Path(a.shm)
    d = torch.device(zcfg.DEVICE)
    ver = -1; cur = None; frames = 0; times = []; reloads = 0; t_end = time.time() + a.seconds
    while time.time() < t_end:
        v = read_version(shm)
        if v != ver and v > 0:
            try:
                d2 = load_snapshot(shm); ver = v; reloads += 1
                cur = GaussianCloud(
                    means=torch.as_tensor(d2["means"], device=d),
                    quats=torch.as_tensor(d2["quats"], device=d),
                    scales=torch.as_tensor(d2["scales"], device=d),
                    opacity=torch.as_tensor(np.clip(d2["opacity"], 0.95, 1.0), device=d),
                    colors=torch.as_tensor(d2["colors"], device=d),
                    owner=torch.zeros(len(d2["means"]), dtype=torch.long, device=d),
                    meta={})
            except Exception as e:
                print("[B] snapshot read failed:", e, flush=True)
        if cur is None:
            time.sleep(0.02); continue
        t0 = time.perf_counter()
        tiles = []
        for (ii, jj) in tiles_ij:
            rgb, alpha, _, _ = render_tile(cur, grid, ii, jj)
            tiles.append(Tile(ii, jj, rgb, alpha))
        mosaic, cov = stitch(tiles, shape=(j1 - j0 + 1, i1 - i0 + 1))
        img = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        f = a.size / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)),
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "frames" / f"{frames:04d}.png"), img)
        times.append(time.perf_counter() - t0)
        frames += 1
    t = np.array(times)
    meta = {"frames": frames, "reloads": reloads, "final_version": ver,
            "render_ms_mean": float(1000 * t.mean()), "render_ms_std": float(1000 * t.std()),
            "fps_mean": float(1.0 / t.mean()), "fps_min": float(1.0 / t.max())}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"[B] {frames} frames, {reloads} snapshot reloads | "
          f"{1000*t.mean():.0f}±{1000*t.std():.0f} ms -> {1/t.mean():.2f} FPS "
          f"(min {1/t.max():.2f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["worker", "render"])
    p.add_argument("--run", required=True)
    p.add_argument("--shm", default="/dev/shm/zso_snap")
    p.add_argument("--out", default="/home/zxy/syb/ZSO/PROJ/results/stage-S54/two_proc")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.39)
    p.add_argument("--gsd", type=float, default=0.45)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--cell", type=float, default=0.45)
    p.add_argument("--collapse-k", type=int, default=5)
    p.add_argument("--collapse-every", type=int, default=0)
    p.add_argument("--publish-every", type=int, default=5)
    p.add_argument("--max-kf", type=int, default=55)
    p.add_argument("--edge-reject", type=int, default=2)
    p.add_argument("--edge-opacity", type=float, default=0.25)
    p.add_argument("--nfp", action="store_true", default=True)
    p.add_argument("--seconds", type=float, default=200.0)
    a = p.parse_args()
    (worker if a.role == "worker" else render)(a)


if __name__ == "__main__":
    main()
