"""Region-level selection for streaming sub-map fusion (design doc v3.2 §2.7).

Decision D12: two independently predicted sub-maps overlap in only 0-2% of voxels,
so averaging never fires and the map ends up stacking one sub-map per keyframe.
Instead we keep, per ortho tile, **one owner keyframe** and render that tile from the
owner's Gaussians only; a newly arriving keyframe takes over a tile when its content
scores better on an ortho-quality criterion (detail + coverage - cross-view
disagreement - layering).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectionConfig:
    w_sharp: float = 0.5
    w_agree: float = 0.5
    w_layer: float = 0.5
    min_px: int = 5000
    accept_margin: float = 0.0     # the new keyframe must beat the owner by this margin


@dataclass
class SelectionState:
    owner: dict = field(default_factory=dict)      # (i, j) -> kf index
    score: dict = field(default_factory=dict)      # (i, j) -> best score
    swaps: list = field(default_factory=list)      # (i, j, old_kf, new_kf, score)
    first: dict = field(default_factory=dict)      # (i, j) -> kf that claimed it


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-9)


def layer_metric(means, T_WP, cell: float = 1.0, z_max: float = 2.0) -> float:
    """Mean per-cell std of plane-z over the **ground** (z < ``z_max``).

    Buildings contribute several legitimate heights inside a 1 m cell (wall + roof),
    which is real structure rather than duplicated layers, so only the ground counts.
    """
    p = (np.asarray(means, np.float64) - T_WP[:3, 3]) @ T_WP[:3, :3]
    p = p[(p[:, 2] > -20.0) & (p[:, 2] < z_max)]
    if len(p) < 10:
        return 0.0
    key = np.floor(p[:, :2] / cell).astype(np.int64)
    order = np.lexsort((p[:, 2], key[:, 1], key[:, 0]))
    k_s, z_s = key[order], p[order, 2]
    _, start, counts = np.unique(k_s, axis=0, return_index=True, return_counts=True)
    sel = counts >= 2
    if not sel.any():
        return 0.0
    stds = [float(z_s[st:st + c].std()) for st, c in zip(start[sel], counts[sel])][:20000]
    return float(np.mean(stds))


def quality_score(conf=None, opacity=None, wsum=None, n_obs=None) -> np.ndarray:
    """Per-observation quality used by the 2.5D cell selection.

    ``q = conf * opacity * wsum`` (all bounded, all already normalised in the pipeline);
    ``n_obs`` gives a small bonus to repeatedly observed points, capped at 2x, so a
    single lucky observation cannot outrank a well-observed one.
    """
    n = None
    for a in (conf, opacity, wsum, n_obs):
        if a is not None:
            n = len(a)
            break
    if n is None:
        return np.zeros(0, np.float32)
    q = np.ones(n, np.float32)
    if conf is not None:
        q = q * np.clip(np.asarray(conf, np.float32), 0.0, 1.0)
    if opacity is not None:
        q = q * np.clip(np.asarray(opacity, np.float32), 0.0, 1.0)
    if wsum is not None:
        q = q * np.clip(np.asarray(wsum, np.float32), 1e-3, None)
    if n_obs is not None:
        q = q * np.minimum(np.asarray(n_obs, np.float32), 2.0) / 2.0
    return q.astype(np.float32)


def select_by_cell(means, T_WP, cell: float = 0.5, q=None, mask=None,
                   keep_n: int = 1, prefer: str = "quality") -> np.ndarray:
    """**2.5D cell selection** (P1''): every ``(x, y)`` cell keeps its best observation.

    This replaces the tile-level replacement, which punched tile-shaped holes when the
    winning keyframe did not cover the whole tile.  Here nothing is deleted per tile:
    for each ``cell``-metre cell of the plane frame we keep the ``keep_n`` observations
    with the highest quality ``q`` (default: higher z, then index), so

    * coverage is preserved (every observed cell survives with one point),
    * layers collapse to a single surface (this is what an ortho product shows),
    * no neighbourhood/voxel-hash match is needed (which is why the voxel merge never
      fired: two frames' points for the same ground rarely fall in the same 0.25 m voxel).

    ``keep_n > 1`` keeps the top ``n`` per cell (needed if a cell really has a roof and
    the ground below it and both matter); the ortho product uses ``keep_n=1``.

    Returns the sorted indices of the kept observations.
    """
    means = np.asarray(means, np.float64)
    if len(means) == 0:
        return np.zeros(0, np.int64)
    P = (means - T_WP[:3, 3]) @ T_WP[:3, :3]
    ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
    # pack the 2-D cell id into one key (works for any extent)
    key = (ij[:, 0] - ij[:, 0].min()) * (int(ij[:, 1].max() - ij[:, 1].min()) + 1) \
        + (ij[:, 1] - ij[:, 1].min())
    if q is None:
        q = np.ones(len(means), np.float32)
    q = np.asarray(q, np.float32)
    if mask is not None:
        m = np.asarray(mask, bool)
    else:
        m = np.ones(len(means), bool)
    # ``prefer="top"`` ranks by plane-z instead of quality: an orthographic product
    # shows the **highest** surface of a cell, and a well-observed ground point under a
    # roof would otherwise win over the roof itself.
    if prefer == "top":
        Pz = P[:, 2].astype(np.float32)
        # np.lexsort uses the **last** key as primary: cell, then highest z, then quality
        order = np.lexsort((-q, -Pz, key))
    else:
        order = np.lexsort((-q, key))          # by cell, then best quality first
    ks = key[order]
    # fully vectorised top-keep_n per cell (a Python loop over millions of cells was
    # far too slow: it dominated the whole product run)
    _, first_idx, counts = np.unique(ks, return_index=True, return_counts=True)
    rank = np.arange(len(ks), dtype=np.int64) - np.repeat(first_idx, counts)
    keep = rank < int(keep_n)
    if mask is not None:
        keep &= m[order]
    sel = np.sort(order[keep]).astype(np.int64)
    return sel


CELL_KEY_OFF = 1 << 20      # supports +-500 km at a 0.5 m cell
CELL_KEY_W = 1 << 21


def pack_cell(ix, iy) -> np.ndarray:
    """Exact 2-D cell key.

    The earlier ``ix * 100003 + iy`` packing silently wrapped for **negative** ``iy``
    (``key % 100003`` re-maps ``-5`` to ``99998``), which inflated dense grids to
    ~100k rows and made half the queries miss their cell.  Offsetting first keeps the
    packing a bijection for ``|cell index| < 2^20``.
    """
    return (np.asarray(ix, np.int64) + CELL_KEY_OFF) * CELL_KEY_W + \
        (np.asarray(iy, np.int64) + CELL_KEY_OFF)


def unpack_cell(key):
    k = np.asarray(key, np.int64)
    return k // CELL_KEY_W - CELL_KEY_OFF, k % CELL_KEY_W - CELL_KEY_OFF


def mean_colors_by_cell(means, colors, T_WP, cell: float = 0.30):
    """Replace every splat's colour by its cell's mean colour (C).

    Geometry is unaffected -- it is still the single selected point per cell -- but the
    1-3 colour samples that the selection discarded are now averaged back in, which is a
    cheap ~2x supersampling of the texture (the roof grid lines and road markings are
    otherwise lost, because keeping one sample of a 0.5 m-period pattern is aliasing).
    """
    means = np.asarray(means, np.float64)
    cols = np.asarray(colors, np.float64)
    if len(means) == 0:
        return cols.astype(np.float32)
    P = (means - T_WP[:3, 3]) @ T_WP[:3, :3]
    ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
    key = pack_cell(ij[:, 0], ij[:, 1])
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq, start, counts = np.unique(ks, return_index=True, return_counts=True)
    cs = cols[order]
    sums = np.add.reduceat(cs, start, axis=0)
    mean = sums / counts[:, None]
    out = np.empty_like(cols)
    out[order] = np.repeat(mean, counts, axis=0)
    return out.astype(np.float32)


def mv_band_filter(means, opacity, T_WP, cell: float = 0.45, min_support: int = 2,
                   band_op: float = 0.5):
    """Multi-view band replacement: drop a band splat only where the cell is covered.

    Band splats (the depth-discontinuity ones) carry a reduced opacity (A'), so
    ``opacity < band_op`` identifies them without any extra column.  A band splat is
    deleted **only if** its cell already holds ``min_support`` clean observations
    (``opacity >= band_op``), i.e. another view saw that spot as an *interior* pixel.
    Otherwise it is kept (attenuated) so no black hole opens up.
    """
    means = np.asarray(means, np.float64)
    op = np.asarray(opacity, np.float32)
    if len(means) == 0:
        return np.ones(0, bool)
    P = (means - T_WP[:3, 3]) @ T_WP[:3, :3]
    ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
    key = pack_cell(ij[:, 0], ij[:, 1])
    band = op < float(band_op)
    uniq, inv = np.unique(key, return_inverse=True)
    clean = (~band).astype(np.int64)
    n_clean = np.bincount(inv, weights=clean, minlength=len(uniq))
    supported = n_clean[inv] >= int(min_support)
    return ~(band & supported)


def select_column_top(means, colors, T_WP, cell: float = 0.45, z_gap: float = 2.0,
                      q=None):
    """P0-2: robust **top surface per column** (one splat per cell).

    For every cell the observations are sorted by height, split into layers where the
    height jump exceeds ``z_gap`` (a roof vs the ground below it), the **topmost layer**
    is kept, and the survivor takes that layer's **median height and median colour**.

    This replaces the old ``argmax z`` rule, which picked whichever *frame* happened to
    reconstruct that cell highest: because each frame's depth bias is a smooth low
    frequency field, the winner formed large spatial patches -> the blocky colour
    patches (one frame per patch, 12-17% exposure differences between patches) and the
    segmented lateral shift that bent straight lines.

    Returns ``(keep_idx, colors_out)`` where ``colors_out`` is aligned to ``keep_idx``.
    """
    means = np.asarray(means, np.float64)
    cols = np.asarray(colors, np.float64)
    if len(means) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    P = (means - T_WP[:3, 3]) @ T_WP[:3, :3]
    ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
    key = pack_cell(ij[:, 0], ij[:, 1])
    order = np.lexsort((-P[:, 2], key))          # by cell, then highest first
    ks, zs = key[order], P[order, 2]
    # layer break inside a cell: either the cell changed, or the height jumped > z_gap
    new_cell = np.empty(len(ks), bool)
    new_cell[0] = True
    new_cell[1:] = ks[1:] != ks[:-1]
    jumped = np.zeros(len(ks), bool)
    jumped[1:] = (zs[:-1] - zs[1:]) > float(z_gap)
    layer_start = new_cell | jumped
    # the first layer of each cell is the top surface: mark its members
    layer_id = np.cumsum(layer_start)
    # the top layer of a cell = the rows before the first in-cell jump.  Vectorised:
    # propagate each cell's ``layer_id`` at its start forward and compare.
    jumped = jumped & ~new_cell
    base = np.maximum.accumulate(np.where(new_cell, layer_id, -1))
    top_layer = layer_id == base
    sel = order[top_layer]
    if len(sel) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    # one representative + that layer's median height / median colour per cell
    k2 = pack_cell(np.floor(((means[sel] - T_WP[:3, 3]) @ T_WP[:3, :3])[:, 0] / cell).astype(np.int64),
                   np.floor(((means[sel] - T_WP[:3, 3]) @ T_WP[:3, :3])[:, 1] / cell).astype(np.int64))
    z2 = ((means[sel] - T_WP[:3, 3]) @ T_WP[:3, :3])[:, 2]
    o2 = np.lexsort((z2, k2))
    k2s, z2s = k2[o2], z2[o2]
    c2s = cols[sel][o2]
    _, st, cnt = np.unique(k2s, return_index=True, return_counts=True)
    # fully vectorised (a Python loop over every cell cost ~10 s per collapse at 2 M
    # cells): median-by-rank for the height, **mean** for the colour (this is what
    # averages the 12-17% per-frame exposure differences), representative = median-rank.
    mid = st + (cnt - 1) // 2
    rep = mid
    sums = np.add.reduceat(c2s, st, axis=0)
    med_c = sums / cnt[:, None]
    idx_out = sel[rep]
    # BUG FIX (week8-end-49): the indices were sorted but the colours were left in
    # **cell order**, so collapse assigned cell j's colour to some unrelated point ->
    # "the whole map suddenly turns into a smear" at every collapse frame, then the new
    # keyframe looks fine until the next collapse scrambles it again.  Sort both with the
    # same permutation.
    perm = np.argsort(idx_out)
    return idx_out[perm], med_c[perm].astype(np.float32)


def column_top_filter(means, T_WP, reference_means=None, cell: float = 0.45,
                      h_tol: float = 2.0):
    """P1-4: drop points that sit far **below** their column's top surface.

    Facade / side-wall splats land in columns whose top is the roof, so they are metres
    below it.  Rejecting ``z < z_max - h_tol`` removes them without the "dig a band in
    every frame" trick of ``edge_reject`` (which left a black moat and whose holes other
    frames refilled with their own bleed points).

    ``reference_means`` lets the caller pass the committed map as well, so a column's top
    can come from an earlier (well-observed) frame rather than only the current patch.
    """
    means = np.asarray(means, np.float64)
    if len(means) == 0:
        return np.ones(0, bool)
    ref = means if reference_means is None else np.concatenate(
        [np.asarray(reference_means, np.float64), means], axis=0)
    Pr = (ref - T_WP[:3, 3]) @ T_WP[:3, :3]
    ijr = np.floor(Pr[:, :2] / float(cell)).astype(np.int64)
    kr = pack_cell(ijr[:, 0], ijr[:, 1])
    # per-column max z (vectorised with reduceat over a key-sorted array)
    o = np.lexsort((-Pr[:, 2], kr))
    ks = kr[o]
    uniq, st = np.unique(ks, return_index=True)
    zmax = Pr[o][st, 2]                      # first entry of each column = its max z
    P = (means - T_WP[:3, 3]) @ T_WP[:3, :3]
    ij = np.floor(P[:, :2] / float(cell)).astype(np.int64)
    k = pack_cell(ij[:, 0], ij[:, 1])
    pos = np.clip(np.searchsorted(uniq, k), 0, len(uniq) - 1)
    hit = uniq[pos] == k
    ztop = np.where(hit, zmax[pos], np.inf)
    return ~(P[:, 2] < ztop - float(h_tol))
