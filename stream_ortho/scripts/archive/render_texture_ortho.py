"""CLI: texture-preserving orthophoto from the frozen product + the original frames.

Renders crops (and optionally the full extent) at a finer GSD than the 0.45 m product by
backward-warping the source images onto the calibrated top surface (see ``ortho/texture.py``).

    python -m stream_ortho.scripts.render_texture_ortho \
        --obs result-new/P0/obs/ieu/pointmap --product results/week10-pivotb/ieu/p1/product_frozen.npz \
        --run results/stage-S2/ieu_calib --images data/phantom3-ieu-kfs/rgb \
        --out results/week10-pivotb/ieu/texture --gsd 0.21 --crops 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from stream_ortho.ortho.texture import FrameIO, edge_width_10_90, render_texture_tile, top_surface_at


def sim3_reference(run_dir):
    """The Sim3 reference recorded when the observation table was built."""
    import json as _json
    from stream_ortho.geometry import R_to_quat_xyzw
    m = _json.loads((pathlib.Path(run_dir) / "sim3_est_to_gt.json").read_text())
    return np.concatenate([m["translation"],
                           R_to_quat_xyzw(np.asarray(m["rotation"])), [m["scale"]]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--run", required=True, help="dir with keyframes.npz")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ieu")
    ap.add_argument("--gsd", type=float, default=0.21)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=3)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--half", type=int, default=256)
    ap.add_argument("--full", action="store_true", help="also render the whole extent")
    ap.add_argument("--depth-hw", default="540,960", help="depth buffer resolution H,W")
    ap.add_argument("--candidate", default="owner3", choices=("owner", "owner1", "owner3"))
    ap.add_argument("--hybrid", default="none", choices=("none", "A", "B", "C"),
                    help="A: cell colour on the edge band; B: owner-only texture there; "
                         "C: same-layer fill on the band")
    ap.add_argument("--grad-tau", type=float, default=0.4, help="edge threshold, m per 0.45 m cell")
    ap.add_argument("--dilate", type=int, default=1, help="edge band dilation in cells")
    ap.add_argument("--grad-pct", type=float, default=90.0,
                    help="gradient percentile that defines the edge band")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    obs = dict(np.load(pathlib.Path(a.obs) / "obs_merged.npz"))
    prod_npz = np.load(a.product)
    product = {"cell": float(prod_npz["cell"]), "ox": int(prod_npz["ox"]),
               "oy": int(prod_npz["oy"]),
               "H": prod_npz["z_ref"].shape[0], "W": prod_npz["z_ref"].shape[1],
               "L": prod_npz["z_ref"].shape[2],
               "n": prod_npz["n"].astype(np.float32),
               "z_ref": prod_npz["z_ref"].astype(np.float32),
               "owner": prod_npz["owner"].astype(np.int32)}
    npz = np.load(pathlib.Path(a.run) / "keyframes.npz")
    sg = sim3_reference(a.run)
    dh, dw = (int(v) for v in a.depth_hw.split(","))
    fio = FrameIO(npz, obs, a.images, sg, depth_hw=(dh, dw))

    # per-cell colour from the observation table (the product's own 0.45 m colour), used to
    # fill pixels where no frame passes the depth test
    cell_rgb = np.zeros((product["H"], product["W"], 3), np.float32)
    wts = np.zeros((product["H"], product["W"]), np.float32)
    _ix = np.clip(np.asarray(obs["ix"], np.int64) - product["ox"], 0, product["W"] - 1)
    _iy = np.clip(np.asarray(obs["iy"], np.int64) - product["oy"], 0, product["H"] - 1)
    _w = np.asarray(obs["conf"], np.float32) * np.asarray(obs["n"], np.float32)
    _flat = _iy * product["W"] + _ix
    _size = product["H"] * product["W"]
    _den = np.bincount(_flat, weights=_w, minlength=_size)
    for ch in range(3):
        _num = np.bincount(_flat, weights=_w * np.asarray(obs["rgb"][:, ch], np.float32),
                           minlength=_size)
        cell_rgb[..., ch] = np.where(_den > 0, _num / np.maximum(_den, 1e-9), 0.0).reshape(
            product["H"], product["W"]) / 255.0

    def fill_from_cells(rgb, hole, grid):
        ci_x = np.clip(((grid["x0"] + (np.arange(grid["w"]) + 0.5) * a.gsd) //
                        product["cell"]).astype(np.int64) - product["ox"], 0, product["W"] - 1)
        ci_y = np.clip(((grid["y0"] + (np.arange(grid["h"]) + 0.5) * a.gsd) //
                        product["cell"]).astype(np.int64) - product["oy"], 0, product["H"] - 1)
        fb = cell_rgb[np.ix_(ci_y, ci_x)]
        out = rgb.copy()
        out[hole] = fb[hole]
        return out

    # top-layer index per cell (0..L-1); used to keep the fill on the same surface
    _occ = product["n"] > 0
    layer_id = np.where(_occ, product["z_ref"], -np.inf).argmax(-1).astype(np.int16)

    def _edge_mask():
        """Edge band from the top-surface height gradient and layer-id changes."""
        from scipy.ndimage import binary_dilation
        occ = product["n"] > 0
        z = np.where(occ, product["z_ref"], np.nan).max(-1)
        live = occ.any(-1)
        gx = np.zeros_like(z); gy = np.zeros_like(z)
        gx[:, :-1] = np.abs(np.diff(z, axis=1)); gx[:, -1] = gx[:, -2]
        gy[:-1, :] = np.abs(np.diff(z, axis=0)); gy[-1, :] = gy[-2, :]
        lid = np.where(occ, product["z_ref"], -np.inf).argmax(-1)
        ldx = np.zeros_like(live); ldy = np.zeros_like(live)
        ldx[:, :-1] = lid[:, :-1] != lid[:, 1:]
        ldy[:-1, :] = lid[:-1, :] != lid[1:, :]
        g = np.maximum(gx, gy)[live]
        thr = max(a.grad_tau, float(np.nanpercentile(g, a.grad_pct))) if g.size else a.grad_tau
        e = live & ((np.maximum(gx, gy) > thr) | ldx | ldy)
        if a.dilate > 0:
            e = binary_dilation(e, iterations=int(a.dilate))
        info_edge = {"grad_threshold": float(thr), "edge_frac_cells": float(e.sum() / max(live.sum(), 1))}
        return e & live, info_edge

    edge_cell, edge_info = _edge_mask() if a.hybrid != "none" else (None, {})

    def edge_alpha(grid):
        """Feathered alpha of the edge band on the output grid (1 = use the stable colour)."""
        from scipy.ndimage import distance_transform_edt
        h, w = grid["h"], grid["w"]
        xg = grid["x0"] + (np.arange(w) + 0.5) * a.gsd
        yg = grid["y0"] + (np.arange(h) + 0.5) * a.gsd
        ci = np.clip((np.tile(xg, h) // product["cell"]).astype(np.int64) - product["ox"], 0,
                     product["W"] - 1)
        cj = np.clip((np.repeat(yg, w) // product["cell"]).astype(np.int64) - product["oy"], 0,
                     product["H"] - 1)
        m = edge_cell[cj, ci].reshape(h, w)
        if not m.any():
            return np.zeros((h, w), np.float32), m
        # distance to the nearest edge pixel for the *outside* pixels; 1 inside the band,
        # feathering outward over ~2 px (the previous version computed it the other way round
        # and made every off-edge pixel alpha=1)
        d = distance_transform_edt(~m)
        alpha = np.where(m, 1.0, np.clip(1.0 - d / 2.0, 0.0, 1.0))
        return alpha.astype(np.float32), m

    def _layer_at(xs, ys):
        """Top-layer index at per-pixel coordinates (elementwise, not a cross product)."""
        ci = np.clip((np.asarray(xs) // product["cell"]).astype(np.int64) - product["ox"],
                     0, product["W"] - 1)
        cj = np.clip((np.asarray(ys) // product["cell"]).astype(np.int64) - product["oy"],
                     0, product["H"] - 1)
        return layer_id[cj, ci]

    def surface_fill(rgb, hole, grid):
        """Fill each hole from the nearest valid pixel **on the same top layer**.

        Returns (filled_rgb, filled_from_cell_mask, still_hole_mask).
        """
        from scipy.ndimage import distance_transform_edt
        h, w = hole.shape
        xg = grid["x0"] + (np.arange(w) + 0.5) * a.gsd
        yg = grid["y0"] + (np.arange(h) + 0.5) * a.gsd
        lay = _layer_at(np.tile(xg, h), np.repeat(yg, w)).reshape(h, w)
        out = rgb.copy()
        from_cell = np.zeros_like(hole)
        still = hole.copy()
        if not hole.any():
            return out, from_cell, still
        for lid in np.unique(lay[hole]):
            m_l = (lay == lid)
            valid_l = m_l & ~hole
            if not valid_l.any():
                continue
            idx = distance_transform_edt(~valid_l, return_distances=False, return_indices=True)
            sel = hole & m_l
            out[sel] = rgb[tuple(idx[:, sel])]
            still[sel] = False
        # anything left has no same-layer source: fall back to the cell colour
        if still.any():
            fb = cell_rgb[np.ix_(
                np.clip(((grid["y0"] + (np.arange(h) + 0.5) * a.gsd) // product["cell"]).astype(np.int64)
                        - product["oy"], 0, product["H"] - 1),
                np.clip(((grid["x0"] + (np.arange(w) + 0.5) * a.gsd) // product["cell"]).astype(np.int64)
                        - product["ox"], 0, product["W"] - 1))]
            out[still] = fb[still]
            from_cell[still] = True
            still[:] = False
        return out, from_cell, still

    _mode = {"owner": dict(max_frames=1, tau=a.tau, owner_only=True),
             "owner1": dict(max_frames=2, tau=min(a.tau, 0.5), owner_only=False),
             "owner3": dict(max_frames=4, tau=a.tau, owner_only=False)}[a.candidate]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"seq": a.seq, "gsd": a.gsd, "tau": a.tau, "max_frames": a.max_frames,
           "product_cell": product["cell"], "crops": [], "wall_s": None}

    # ---- crops: pick the cells with the most layers (buildings) ----
    n_layers = product["n"]
    score = (n_layers >= 2).sum(-1).astype(np.float32)
    from scipy.ndimage import uniform_filter
    dens = uniform_filter(score, size=41)
    # live-coverage map so a crop is never taken at the survey edge
    from scipy.ndimage import uniform_filter as _uf
    live_cov = _uf(((product["n"] > 0).any(-1)).astype(np.float32), size=41)
    used = []
    for ci in range(a.crops):
        cand = np.argsort(dens.ravel())[::-1][:400]
        pick = None
        for k in cand:
            i, j = np.unravel_index(int(k), dens.shape)
            if live_cov[i, j] >= 0.70 and all(abs(i - u[0]) > 60 or abs(j - u[1]) > 60 for u in used):
                pick = (i, j)
                break
        if pick is None:
            i, j = np.unravel_index(int(np.argmax(dens)), dens.shape)
        else:
            i, j = pick
        used.append((int(i), int(j)))
        dens[max(i - 60, 0):i + 60, max(j - 60, 0):j + 60] = -1
        x0 = (product["ox"] + j - a.half) * product["cell"]
        y0 = (product["oy"] + i - a.half) * product["cell"]
        grid = {"x0": float(x0), "y0": float(y0),
                "h": int(2 * a.half * product["cell"] / a.gsd),
                "w": int(2 * a.half * product["cell"] / a.gsd)}
        t1 = time.time()
        rgb, conf, hole, info = render_texture_tile(
            product, fio, grid, i, j, gsd=a.gsd, tau=_mode["tau"],
            max_frames=_mode["max_frames"], owner_only=_mode["owner_only"])
        dur = time.time() - t1
        # the 0.45 m cell render on the same grid, for the comparison
        xs = grid["x0"] + (np.arange(grid["w"]) + 0.5) * a.gsd
        ys = grid["y0"] + (np.arange(grid["h"]) + 0.5) * a.gsd
        Xg, Yg = np.meshgrid(xs, ys)
        zt, okg = top_surface_at(product, Xg.ravel(), Yg.ravel())
        rgb_cell = np.zeros((Xg.size, 3), np.float32)
        ci_x = np.clip((Xg.ravel() // product["cell"]).astype(np.int64) - product["ox"], 0,
                       product["W"] - 1)
        ci_y = np.clip((Yg.ravel() // product["cell"]).astype(np.int64) - product["oy"], 0,
                       product["H"] - 1)
        cmean = np.zeros_like(rgb_cell)
        rgb_cell[okg] = cmean[okg]                     # placeholder, filled below if rgb available
        e_tex, n_tex = edge_width_10_90(rgb)
        frac_hole_raw = float(hole.mean())
        rgb_filled, from_cell, still = surface_fill(rgb, hole, grid)
        if a.hybrid != "none":
            cell_img_h = fill_from_cells(np.zeros_like(rgb), np.ones(hole.shape, bool), grid)
            if a.hybrid == "C":
                # the whole band is re-filled from the nearest same-layer pixel outside it,
                # so the band gets a low-pass, layer-consistent colour
                _a_b, _m_b = edge_alpha(grid)
                rgb_band, _, _ = surface_fill(rgb_filled, _m_b | hole, grid)
                base = rgb_band
            elif a.hybrid == "B":
                rgb_b, _, hole_b, _ = render_texture_tile(
                    product, fio, grid, i, j, gsd=a.gsd, tau=0.5, max_frames=1,
                    owner_only=True)
                base = np.where(hole_b[..., None], rgb_filled, rgb_b)
            else:
                base = rgb_filled
            alpha, _m = edge_alpha(grid)
            rgb_filled = (alpha[..., None] * cell_img_h + (1 - alpha[..., None]) * base)
            edge_frac = float((alpha > 0.5).mean())
        # baseline: the same grid painted with the 0.45 m per-cell colours
        cell_img = fill_from_cells(np.zeros_like(rgb), np.ones(rgb.shape[:2], bool), grid)
        e_cell, n_cell = edge_width_10_90(cell_img)
        e_tex, n_tex = edge_width_10_90(rgb_filled)
        e_flat = n_flat = float("nan")
        e_edge = float("nan")
        if a.hybrid != "none":
            alpha_m, _mm = edge_alpha(grid)
            flat_m = alpha_m < 0.1
            e_flat, n_flat = edge_width_10_90(rgb_filled, mask=flat_m)
            e_edge, _ = edge_width_10_90(rgb_filled, mask=~flat_m)
        res["crops"].append({"i": int(i), "j": int(j), "n_pixels": info["n_pixels"],
                             "frac_hole_raw": frac_hole_raw,
                             "frac_filled_from_cells": float(from_cell.mean()),
                             "frac_still_hole": float(still.mean()),
                             "frac_multi_frame": info["frac_multi_frame"],
                             "edge_width_px": e_tex, "n_edges": n_tex,
                             "edge_frac": float(edge_frac) if a.hybrid != "none" else 0.0,
                             "edge_width_flat_px": e_flat, "edge_width_edge_px": e_edge,
                             "edge_width_cell_px": e_cell, "n_edges_cell": n_cell,
                             "edge_reduction_pct": (float(100 * (1 - e_tex / e_cell))
                                                    if e_cell and np.isfinite(e_cell) else None),
                             "edge_width_m": e_tex * a.gsd, "render_s": dur})

        rgb_cellfill = fill_from_cells(rgb, hole, grid)
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
        axes[0].imshow(np.clip(rgb, 0, 1))
        axes[0].set_title(f"texture @ {a.gsd:.2f} m ({a.candidate})", fontsize=8)
        axes[1].imshow(np.clip(rgb_cellfill, 0, 1))
        axes[1].set_title(f"cell-colour fill ({100*frac_hole_raw:.1f}% holes)", fontsize=8)
        axes[2].imshow(np.clip(rgb_filled, 0, 1))
        axes[2].set_title("same-layer surface fill", fontsize=8)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.suptitle(f"crop {ci}: i={i} j={j}  ({2*a.half*product['cell']:.0f} m)", fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"tex_crop_{ci}_{a.candidate}.png"); plt.close(fig)
        np.savez_compressed(out / f"tex_crop_{ci}_{a.candidate}.npz", rgb=rgb_filled.astype(np.float16),
                            conf=conf.astype(np.float16), hole=hole,
                            x0=grid["x0"], y0=grid["y0"], gsd=a.gsd)
        print(f"crop {ci}: i={i} j={j} holes(raw) {100*frac_hole_raw:.1f}% "
              f"edge cell {e_cell:.2f}px -> tex {e_tex:.2f}px "
              f"edge {e_tex:.2f} px = {e_tex*a.gsd:.2f} m ({dur:.1f}s)")

    # ---- full extent (optional) ----
    if a.full:
        gsd = a.gsd
        W_out = int(product["W"] * product["cell"] / gsd)
        H_out = int(product["H"] * product["cell"] / gsd)
        rgb = np.zeros((H_out, W_out, 3), np.float32)
        hole = np.ones((H_out, W_out), bool)
        tile = 512
        for i0 in range(0, H_out, tile):
            for j0 in range(0, W_out, tile):
                hh, ww = min(tile, H_out - i0), min(tile, W_out - j0)
                grid = {"x0": product["ox"] * product["cell"] + j0 * gsd,
                        "y0": product["oy"] * product["cell"] + i0 * gsd,
                        "h": hh, "w": ww}
                r, c, ho, _ = render_texture_tile(product, fio, grid, i0, j0, gsd=gsd,
                                                  tau=_mode["tau"],
                                                  max_frames=_mode["max_frames"],
                                                  owner_only=_mode["owner_only"])
                rgb[i0:i0 + hh, j0:j0 + ww] = r
                hole[i0:i0 + hh, j0:j0 + ww] = ho
        # outside = beyond the product's live area; hole_inside = live but no candidate
        live_full = np.zeros((H_out, W_out), bool)
        for i0 in range(0, H_out, tile):
            for j0 in range(0, W_out, tile):
                hh, ww = min(tile, H_out - i0), min(tile, W_out - j0)
                xg = (product["ox"] * product["cell"] + j0 * gsd) + (np.arange(ww) + 0.5) * gsd
                yg = (product["oy"] * product["cell"] + i0 * gsd) + (np.arange(hh) + 0.5) * gsd
                _, ok = top_surface_at(product, np.tile(xg, hh), np.repeat(yg, ww))
                live_full[i0:i0 + hh, j0:j0 + ww] = ok.reshape(hh, ww)
        outside = ~live_full
        hole_inside = hole & live_full
        rgb_full, from_cell, still = surface_fill(
            rgb, hole_inside, {"x0": product["ox"] * product["cell"],
                               "y0": product["oy"] * product["cell"],
                               "h": H_out, "w": W_out})
        if a.hybrid != "none":
            full_grid = {"x0": product["ox"] * product["cell"],
                         "y0": product["oy"] * product["cell"],
                         "h": H_out, "w": W_out}
            alpha_full, _mf = edge_alpha(full_grid)
            cell_full = fill_from_cells(np.zeros((H_out, W_out, 3), np.float32),
                                        np.ones((H_out, W_out), bool), full_grid)
            rgb_full = alpha_full[..., None] * cell_full + (1 - alpha_full[..., None]) * rgb_full
            res["full"] = {**(res.get("full") or {}), "edge_frac_full": float((alpha_full > 0.5).mean())}
        rgba = np.zeros((H_out, W_out, 4), np.uint8)
        rgba[..., :3] = (np.clip(rgb_full, 0, 1) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(live_full, 255, 0).astype(np.uint8)
        from PIL import Image as _Im
        _Im.fromarray(rgba, mode="RGBA").save(out / "ortho_texture_full.png")
        _Im.fromarray((live_full * 255).astype(np.uint8), mode="L").save(out / "valid_mask.png")
        res["full"] = {"h": H_out, "w": W_out,
                       "frac_outside": float(outside.mean()),
                       "frac_hole_inside": float(hole_inside.sum() / max(live_full.sum(), 1)),
                       "frac_filled_from_cell": float(from_cell.sum() / max(live_full.sum(), 1)),
                       "frac_filled_same_layer": float((hole_inside.sum() - from_cell.sum())
                                                       / max(live_full.sum(), 1)),
                       "frac_still_hole": float(still.mean())}

    res["wall_s"] = time.time() - t0
    (out / "texture_ortho.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"wall {res['wall_s']:.1f}s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
