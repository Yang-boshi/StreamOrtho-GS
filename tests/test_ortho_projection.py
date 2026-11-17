"""S3/S4 test: **synthetic** orthographic projection (analytic ground truth).

Validates the plane-frame ortho camera without any network:

* 4 Gaussians are placed at known plane coordinates ``(±2, ±1, 0)``;
* the tile is rendered at a known ``gsd`` / tile origin;
* the rendered blob centroids must match the analytic pixel positions to < 1 px
  (design-doc S4 gate: "已知地面点像素误差 < 1 px");
* the image is not black / not empty, and the plane is not flipped.

Run::

    python -m stream_ortho.tests.test_ortho_projection
"""

from __future__ import annotations

import numpy as np
import torch

from stream_ortho.geometry import ortho_K, ortho_viewmat, plane_point_to_pixel


def _render(points_xy, gsd=0.01, tile=512, halo=32, z_top=5.0):
    from stream_ortho.frontend.predict import render_ortho

    W = H = tile + 2 * halo
    T_WP = np.eye(4)                     # plane == world
    cx = cy = 0.0
    viewmat = torch.as_tensor(ortho_viewmat(T_WP, cx, cy, z_top),
                              dtype=torch.float32, device="cuda")[None]
    K = ortho_K(gsd, W, H)

    n = len(points_xy)
    means = torch.tensor([[x, y, 0.0] for x, y in points_xy], dtype=torch.float32, device="cuda")
    quats = torch.zeros((n, 4), device="cuda")
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.15, dtype=torch.float32, device="cuda")
    opac = torch.full((n, 1), 1.0, device="cuda")
    colors = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]][:n], dtype=torch.float32, device="cuda")
    img, alpha, _ = render_ortho(means, scales, quats, opac, colors, viewmat, K, H, W)
    return img[0].clamp(0, 1).cpu().numpy(), alpha[0, :, :, 0].cpu().numpy(), (cx, cy, gsd, W, H, halo)


def test_synthetic_positions():
    """One Gaussian per render (avoids colour crosstalk), 4 known ground points."""
    pts = [(-2.0, 1.0), (2.0, 1.0), (2.0, -1.0), (-2.0, -1.0)]
    errs = []
    for k, (x, y) in enumerate(pts):
        img, alpha, (cx, cy, gsd, W, H, halo) = _render([(x, y)])
        exp_uv = plane_point_to_pixel(np.eye(4), cx, cy, gsd, W, H,
                                      np.array([[x, y, 0.0]]))[0]
        m = img[:, :, 0] > 0.5 * img[:, :, 0].max()
        ys, xs = np.nonzero(m)
        got = np.array([xs.mean(), ys.mean()])
        err = float(np.linalg.norm(got - exp_uv))
        errs.append(err)
        print(f"    point {k} plane=({x:+.1f},{y:+.1f})  expected={exp_uv.round(2)}  "
              f"got={got.round(2)}  err={err:.3f} px")
    assert max(errs) < 1.0, f"pixel error {max(errs):.3f} px > 1 px"
    print(f"[ok] synthetic ortho projection: max error {max(errs):.3f} px")


def test_orientation_not_flipped():
    """+x_P must go right, +y_P must go up (image y is down)."""
    img, alpha, (cx, cy, gsd, W, H, halo) = _render([(1.0, 0.0)], gsd=0.01, tile=256)
    ys, xs = np.nonzero(img[:, :, 0] > 0.3)
    assert xs.mean() > W / 2, "a point at +x_P must land right of the tile centre"
    ys2, xs2 = np.nonzero(alpha > 0.5)
    print(f"[ok] +x_P -> right (u={xs.mean():.1f} > {W/2}), alpha bbox x=[{xs2.min()},{xs2.max()}]")

    img, alpha, (cx, cy, gsd, W, H, halo) = _render([(0.0, 1.0)], gsd=0.01, tile=256)
    ys, xs = np.nonzero(img[:, :, 0] > 0.3)
    assert ys.mean() < H / 2, "a point at +y_P (up) must land above the tile centre"
    print(f"[ok] +y_P -> up (v={ys.mean():.1f} < {H/2})")


def test_gsd_scaling():
    """Doubling the GSD halves the pixel distance from the centre."""
    for gsd in (0.005, 0.01, 0.02):
        img, alpha, (cx, cy, g, W, H, halo) = _render([(1.0, 0.0)], gsd=gsd, tile=512)
        ys, xs = np.nonzero(img[:, :, 0] > 0.3)
        d = xs.mean() - W / 2
        assert abs(d - 1.0 / gsd) < 1.0, (gsd, d)
        print(f"    gsd={gsd}: 1 m offset -> {d:.1f} px (expected {1.0/gsd:.0f})")
    print("[ok] GSD scaling: fx = 1/gsd")


def _textured_plane_cloud(x0, y0, extent, gsd, z=0.0, seed=0, tex=None):
    """Gaussians on a plane with a random checker texture (world frame)."""
    from stream_ortho.ortho.rasterizer import GaussianCloud

    rng = np.random.default_rng(seed)
    n = int(round(extent / gsd))
    xs = x0 + (np.arange(n) + 0.5) * gsd
    ys = y0 + (np.arange(n) + 0.5) * gsd
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], -1)
    if tex is None:      # blocky random texture (good for seam tests)
        t = rng.random((16, 16, 3)).astype(np.float32)
        xi = (np.clip((X.ravel() - x0) / extent, 0, 0.999) * 16).astype(int)
        yi = (np.clip((Y.ravel() - y0) / extent, 0, 0.999) * 16).astype(int)
        col = t[yi, xi]
    else:                # continuous texture evaluated at the Gaussian centres
        col = np.asarray(tex(X.ravel(), Y.ravel(), extent), dtype=np.float32)
    q = torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(len(pts), 1)
    return GaussianCloud(
        torch.tensor(pts, dtype=torch.float32, device="cuda"), q,
        torch.full((len(pts), 3), 1.5 * gsd, device="cuda"),
        torch.ones(len(pts), device="cuda"),
        torch.tensor(col, device="cuda"))


def test_two_by_two_seam():
    """2x2 tiles must stitch into the same image as one big tile (P1 seam gate)."""
    from stream_ortho.ortho.grid import OrthoGrid
    from stream_ortho.ortho.rasterizer import render_tile
    from stream_ortho.ortho.tiles import Tile, seam_gradient_ratio, stitch

    gsd, tile, halo = 0.05, 256, 32
    grid = OrthoGrid(T_WP=np.eye(4), gsd=gsd, tile=tile, halo=halo, z_ground=0.0, z_top=50.0)
    cloud = _textured_plane_cloud(-20.0, -20.0, 40.0, gsd / 2)

    tiles = []
    for dj in (0, 1):
        for di in (0, 1):
            rgb, alpha, _, _ = render_tile(cloud, grid, di, dj)
            tiles.append(Tile(di, dj, rgb, alpha))
    mosaic, cov = stitch(tiles, shape=(2, 2))
    assert mosaic.shape == (2 * tile, 2 * tile, 3), mosaic.shape

    seam = seam_gradient_ratio(mosaic, tile)
    assert seam["ratio"] < 1.5, seam
    print(f"[ok] 2x2 seam ratio {seam['ratio']:.2f} (< 1.5), "
          f"seam grad {seam['seam_gradient']:.4f}, interior {seam['interior_gradient']:.4f}")

    # compare against one big tile covering the same ground area
    big = OrthoGrid(T_WP=np.eye(4), gsd=gsd, tile=2 * tile, halo=halo, z_ground=0.0, z_top=50.0)
    rgb_big, alpha_big, _, _ = render_tile(cloud, big, 0, 0)
    m = (cov > 0.5) & (alpha_big > 0.5)
    from stream_ortho.frontend.predict import psnr

    val = psnr(mosaic[m], rgb_big[m])
    assert val > 30.0, f"stitched vs single tile PSNR {val:.2f} dB <= 30 dB"
    print(f"[ok] stitched 2x2 vs single 512 tile: PSNR {val:.2f} dB (> 30 dB)")


def _smooth_texture(seed=3, n=5):
    """Bilinear random texture: band-limited w.r.t. the GSD, so interpolation is not a factor."""
    rng = np.random.default_rng(seed)
    g = rng.random((n, n, 3)).astype(np.float32)

    def f(x, y, extent=5.12):
        u = np.clip(np.asarray(x) / extent, 0, 1) * (n - 1)
        v = np.clip(np.asarray(y) / extent, 0, 1) * (n - 1)
        i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int)
        i1 = np.minimum(i0 + 1, n - 1); j1 = np.minimum(j0 + 1, n - 1)
        du = (u - i0)[..., None]; dv = (v - j0)[..., None]
        return ((g[j0, i0] * (1 - du) + g[j0, i1] * du) * (1 - dv)
                + (g[j1, i0] * (1 - du) + g[j1, i1] * du) * dv)
    return f


def test_synthetic_ortho_psnr():
    """Render a textured plane and compare with the analytic ortho (P1 gate > 30 dB)."""
    from stream_ortho.ortho.grid import OrthoGrid
    from stream_ortho.ortho.rasterizer import render_tile
    from stream_ortho.frontend.predict import psnr

    gsd, tile, halo = 0.02, 256, 32
    grid = OrthoGrid(T_WP=np.eye(4), gsd=gsd, tile=tile, halo=halo, z_ground=0.0, z_top=50.0)
    x0, y0, extent = 0.0, 0.0, 5.12      # tile (0,0) covers x,y in [0, 5.12]
    tex = _smooth_texture(3, 5)
    cloud = _textured_plane_cloud(x0, y0, extent, gsd, seed=3, tex=tex)
    rgb, alpha, _, _ = render_tile(cloud, grid, 0, 0)

    # analytic expectation: the same continuous texture sampled on the tile grid
    u, v = np.meshgrid(np.arange(tile), np.arange(tile))
    pP = grid.px_to_plane(u, v, 0, 0, z=0.0)
    expected = tex(pP[..., 0], pP[..., 1], extent)
    m = alpha > 0.5
    assert m.sum() > 0.3 * m.size, f"only {m.mean()*100:.1f}% of the tile is covered"
    val = psnr(rgb[m], expected[m])
    assert val > 30.0, f"synthetic ortho PSNR {val:.2f} dB <= 30 dB"
    print(f"[ok] synthetic ortho PSNR {val:.2f} dB (> 30 dB) over {m.mean()*100:.0f}% of the tile")


def run_all():
    assert torch.cuda.is_available()
    print("device:", torch.cuda.get_device_name(0))
    test_synthetic_positions()
    test_orientation_not_flipped()
    test_gsd_scaling()
    test_two_by_two_seam()
    test_synthetic_ortho_psnr()
    print("test_ortho_projection: ALL PASSED")


if __name__ == "__main__":
    run_all()
