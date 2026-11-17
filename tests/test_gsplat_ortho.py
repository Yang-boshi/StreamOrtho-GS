"""S0 acceptance test: gsplat orthographic rasterisation.

Pass criteria (design doc S0):
* output is ``(1, H, W, 3)`` and alpha peaks at the expected pixel;
* occlusion between two points at different depths is correct;
* the ortho intrinsics convention is ``fx = pixels per metre`` (checked by
  moving the point by one pixel worth of world distance);
* peak GPU memory stays far below 6 GB.

Run with ``pytest`` or directly::

    python -m stream_ortho.tests.test_gsplat_ortho
"""

from __future__ import annotations

import numpy as np
import torch


def _rasterize(
    means, colors, *, fx=100.0, cx=32.0, cy=32.0, W=64, H=64, viewmat=None, opacity=None
):
    from gsplat import rasterization

    device = "cuda"
    means = torch.as_tensor(means, dtype=torch.float32, device=device)
    n = means.shape[0]
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.02, dtype=torch.float32, device=device)
    opac = torch.ones((n,), dtype=torch.float32, device=device) if opacity is None else torch.as_tensor(opacity, dtype=torch.float32, device=device)
    colors = torch.as_tensor(colors, dtype=torch.float32, device=device)
    viewmats = torch.eye(4, device=device)[None] if viewmat is None else viewmat
    Ks = torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )[None]

    img, alpha, meta = rasterization(
        means,
        quats,
        scales,
        opac,
        colors,
        viewmats,
        Ks,
        W,
        H,
        camera_model="ortho",
        sh_degree=None,
        packed=True,
        rasterize_mode="antialiased",
    )
    return img, alpha, meta


def test_shapes_and_center():
    img, alpha, meta = _rasterize([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    assert tuple(img.shape) == (1, 64, 64, 3), img.shape
    assert tuple(alpha.shape) == (1, 64, 64, 1), alpha.shape
    a = alpha[0, :, :, 0]
    peak = np.unravel_index(int(torch.argmax(a).item()), a.shape)
    # gsplat uses pixel-centre coordinates; a mean at (32.0, 32.0) lands on
    # index 31 or 32 depending on the rasteriser's half-pixel convention.
    assert abs(peak[0] - 32) <= 1 and abs(peak[1] - 32) <= 1, f"alpha peak at {peak}, expected ~(32,32)"
    # colour is red at the peak
    rgb = img[0, 32, 32]
    assert rgb[0] > 0.8 and rgb[1] < 0.2 and rgb[2] < 0.2, rgb
    print(f"[ok] shapes {tuple(img.shape)} / {tuple(alpha.shape)}, peak={peak}, rgb={rgb.tolist()}")


def test_ortho_is_pixel_per_metre():
    """fx is pixels-per-metre: 1 m offset at fx=10 lands 10 px to the right."""
    # keep the displaced point inside the 64-px tile: dx * fx <= 30 px
    for fx, dx in ((10.0, 1.0), (20.0, 1.5), (100.0, 0.25)):
        a0 = _rasterize([[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]], fx=fx)[1][0, :, :, 0]
        a1 = _rasterize([[dx, 0.0, 1.0]], [[0.0, 1.0, 0.0]], fx=fx)[1][0, :, :, 0]
        x0 = int(torch.argmax(a0).item()) % a0.shape[1]
        x1 = int(torch.argmax(a1).item()) % a1.shape[1]
        expected = dx * fx
        assert abs((x1 - x0) - expected) <= 1, f"fx={fx}, dx={dx}: offset {x1 - x0}, expected {expected}"
        print(f"    fx={fx:<5} dx={dx}: {x1 - x0} px (expected {expected})")
    print("[ok] ortho intrinsics: fx == pixels per metre")


def test_depth_independence():
    """Orthographic projection must not shrink with depth."""
    a_near = _rasterize([[0, 0, 1.0]], [[1, 0, 0]], fx=100)[1][0].sum()
    a_far = _rasterize([[0, 0, 5.0]], [[1, 0, 0]], fx=100)[1][0].sum()
    ratio = float(a_far / a_near)
    assert 0.9 < ratio < 1.1, f"alpha mass changed with depth: ratio={ratio}"
    print(f"[ok] depth independence: alpha mass ratio far/near = {ratio:.4f}")


def test_occlusion():
    """Near red must occlude far blue."""
    img, alpha, _ = _rasterize(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    rgb = img[0, 32, 32]
    assert rgb[0] > rgb[2], f"occlusion wrong: rgb={rgb.tolist()}"
    print(f"[ok] occlusion: centre rgb={rgb.tolist()} (red in front of blue)")


def test_memory():
    if not torch.cuda.is_available():
        print("[skip] cuda unavailable")
        return
    # the rasteriser is a C++ extension, so torch.max_memory_allocated() sees
    # ~0; use the device-wide free memory instead.
    free0, total = torch.cuda.mem_get_info()
    _rasterize([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    used_gb = max(0.0, (free0 - free1) / 1e9)
    assert used_gb < 6.0, used_gb
    print(f"[ok] device memory used by the rasteriser: {used_gb:.3f} GB (free {free1/1e9:.1f}/{total/1e9:.1f})")


def run_all():
    assert torch.cuda.is_available(), "需要 CUDA"
    print("gsplat:", __import__("gsplat").__version__)
    print("device:", torch.cuda.get_device_name(0))
    test_shapes_and_center()
    test_ortho_is_pixel_per_metre()
    test_depth_independence()
    test_occlusion()
    test_memory()
    print("S0: ALL PASSED")


if __name__ == "__main__":
    run_all()
