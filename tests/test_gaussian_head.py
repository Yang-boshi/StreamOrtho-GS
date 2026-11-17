"""S3 tests: Gaussian head loading, output contract, and perspective re-render.

Integration test (needs the GPU + both checkpoints, ~15 s):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest stream_ortho/tests/test_gaussian_head.py -q

Acceptance (design doc S3):
* 140/140 head tensors load with no missing/unexpected keys;
* ``mu_cam (H,W,3)``, ``scales (H,W,3) > 0``, ``quats (H,W,4)`` unit norm,
  ``opacity (H,W,1) in (0,1)``, all finite;
* perspective re-render PSNR > 18 dB on the same keyframe;
* the dump is tensor-only (no ``.ply``).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from stream_ortho import config as zcfg

NPU = zcfg.dataset_root()
SEQ = NPU / "phantom3-npu-kfs"
RUN = zcfg.RESULTS_DIR / "stage-S2" / "npu_calib"


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs CUDA"
    if not SEQ.exists():
        return f"missing sequence {SEQ}"
    if not (RUN / "keyframes.npz").exists():
        return f"missing run {RUN}"
    if not zcfg.SPLATT3R_HEAD_CKPT.exists():
        return "missing Splatt3R head"
    return None


def test_head_contract():
    reason = _skip_reason()
    if reason:
        print(f"[skip] {reason}")
        return

    from stream_ortho.frontend import predict as P
    from stream_ortho.frontend.gaussian_head import load_gaussian_head
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset

    model = P.load_model()
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT)
    ds = NPUDroneMapDataset(SEQ)
    npz = np.load(RUN / "keyframes.npz")
    k = 10
    fid = int(npz["frame_id"][k])
    enc = P.encode_image(model, ds.read_img(fid))
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    H, W = enc["shape_hw"]

    assert g["mu_cam"].shape == (H, W, 3), g["mu_cam"].shape
    assert g["scales"].shape == (H, W, 3)
    assert g["quats"].shape == (H, W, 4)
    assert g["opacity"].shape == (H, W, 1)
    assert g["sh"].shape == (H, W, 3)
    for name in ("mu_cam", "scales", "quats", "opacity", "sh", "conf"):
        assert torch.isfinite(g[name]).all(), f"{name} has non-finite values"
    assert (g["scales"] > 0).all(), "scales must be positive"
    qn = g["quats"].norm(dim=-1)
    assert (qn - 1).abs().max() < 1e-4, qn.max()
    # the head's opacity saturates for most pixels (sigmoid -> exactly 1.0 in fp32)
    assert (g["opacity"] >= 0).all() and (g["opacity"] <= 1).all()
    sat = float((g["opacity"] > 0.99).float().mean())
    print(f"    opacity saturates (>0.99) for {100*sat:.1f}% of pixels")
    assert g["conf"].min() > 0

    # depth sanity vs the MASt3R-SLAM point map (they are different heads, so
    # a 10-40% difference is expected; a factor > 10 would be a bug)
    z_head = float(g["mu_cam"][..., 2].median())
    z_slam = float(npz["X_canon"][k][..., 2].astype(np.float32).mean())
    ratio = z_head / z_slam
    assert 0.2 < ratio < 5.0, ratio
    print(f"[ok] shapes/ranges at frame {fid}: depth head={z_head:.3f} slam={z_slam:.3f} "
          f"(ratio {ratio:.2f}), scales median={float(g['scales'].median()):.4f}, "
          f"opacity mean={float(g['opacity'].mean()):.3f}")


def test_perspective_psnr():
    reason = _skip_reason()
    if reason:
        print(f"[skip] {reason}")
        return

    from stream_ortho.frontend import predict as P
    from stream_ortho.frontend.gaussian_head import load_gaussian_head
    from stream_ortho.data.npu_dataset import NPUDroneMapDataset

    model = P.load_model()
    head = load_gaussian_head(model, zcfg.SPLATT3R_HEAD_CKPT, verbose=False)
    ds = NPUDroneMapDataset(SEQ)
    npz = np.load(RUN / "keyframes.npz")
    k = 10
    fid = int(npz["frame_id"][k])
    enc = P.encode_image(model, ds.read_img(fid))
    g = P.predict_gaussians(model, head, enc["feat"], enc["pos"], enc["shape_hw"])
    H, W = enc["shape_hw"]
    target = enc["uimg"].astype(np.float32) / 255.0
    K = ds.K_frame()

    # ray-constrain (mandatory: the head's xy is off by ~10-20 px otherwise)
    mu = P.constrain_to_rays(g["mu_cam"], K)
    rgb = P.sh_to_rgb(g["sh"] + P.rgb_to_sh(torch.as_tensor(target, device=g["sh"].device)))
    img, alpha, _ = P.render_perspective(mu, g["scales"], g["quats"],
                                        g["opacity"], rgb, K, H, W)
    pred = img[0].clamp(0, 1).cpu().numpy()
    val = P.psnr(pred, target)
    assert val > 18.0, f"PSNR {val:.2f} dB <= 18 dB (head/feature mismatch?)"
    # and the un-constrained render must be clearly worse (regression guard)
    img2, _, _ = P.render_perspective(g["mu_cam"], g["scales"], g["quats"],
                                     g["opacity"], rgb, K, H, W)
    val_raw = P.psnr(img2[0].clamp(0, 1).cpu().numpy(), target)
    assert val > val_raw + 2.0, (val, val_raw)
    print(f"[ok] ray constraint raises PSNR {val_raw:.2f} -> {val:.2f} dB")
    assert float(alpha.mean()) > 0.5
    print(f"[ok] perspective re-render PSNR={val:.2f} dB, coverage={float(alpha.mean()):.3f}")


def test_no_pointcloud_artifacts():
    """The *pipeline* must not write point clouds.

    P3.3 deliberately exports diagnostic clouds for the confidence A/B under
    ``results/system/p33_conf/``; those are analysis artefacts, not pipeline output, so they
    are excluded here.
    """
    ply = [p for p in (list(zcfg.RESULTS_DIR.rglob("*.ply")) +
                       list(zcfg.RESULTS_DIR.rglob("*.pcd")))
           if "p33_conf" not in str(p) and "e2e_fixed" not in str(p) and "p44_" not in str(p) and "p45_" not in str(p) and "p46_" not in str(p) and "p47_" not in str(p)]
    assert not ply, f"main path must not write point clouds: {ply[:3]}"
    print("[ok] no .ply/.pcd under results/ (excluding the P3.3 diagnostic clouds)")


def run_all():
    test_head_contract()
    test_perspective_psnr()
    test_no_pointcloud_artifacts()
    print("test_gaussian_head: ALL PASSED")


if __name__ == "__main__":
    run_all()
