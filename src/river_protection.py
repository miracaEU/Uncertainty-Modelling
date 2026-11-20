"""River flood protection per asset: FLOPROS as 5 km block means, gaps filled nearby.

MIRACA's AssetRisk_PanEU (hazard_river.load_protection_standards) coarsens the
500 m FLOPROS raster 10 x 10 before the overlay - xarray's
coarsen(x=10, y=10, boundary="trim").mean() on the whole raster, so no-data
pixels are skipped and the blocks are counted from the raster's top-left corner -
and gives each feature the block under its centroid, 0 where the block has no data.
block_means() reproduces that block grid exactly; only the window around the
assets is read, which gives the same blocks because the window starts on a block
boundary of the full raster.

On top of that, a feature whose block is 0 takes the value of the nearest block
with a positive standard, if that block's nearest edge lies within FILL_KM of the
centroid. Distance is to the block's edge, not its centre: the block is the area
that carries the value, so this is the distance to the nearest point with a
standard. Every block touching the feature's own block is therefore always in
reach; a tie is resolved towards the higher standard.

FLOPROS uses -1 for no data and has no zeros (values 2-10000 years), so a block is
0 exactly when all 100 of its pixels are no data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

BLOCK = 10       # 500 m pixels per block side: 5 km, as AssetRisk_PanEU coarsens
FILL_KM = 5.0

# source codes stored per feature
NONE, FROM_BLOCK, FROM_FILL = 0, 1, 2
SOURCE_LABEL = {NONE: "none", FROM_BLOCK: "block", FROM_FILL: "fill"}


def block_means(path: str | Path, bounds: tuple[float, float, float, float] | None = None):
    """FLOPROS 5 km block means covering bounds (EPSG:3035), on the full raster's block grid.

    Returns (mean, x_left, y_top, size): mean[i, j] is NaN where the block has no valid
    pixel, block (i, j) spans x_left + j*size .. +size and y_top - i*size .. -size.
    The raster's last rows and columns that do not fill a whole block are dropped,
    as boundary="trim" does.
    """
    with rasterio.open(path) as src:
        t = src.transform
        if abs(t.b) > 0 or abs(t.d) > 0 or not np.isclose(t.a, -t.e):
            raise ValueError(f"{path}: expected a north-up raster with square pixels")
        res = t.a
        size = res * BLOCK
        nby, nbx = src.height // BLOCK, src.width // BLOCK
        if bounds is None:
            i0, i1, j0, j1 = 0, nby, 0, nbx
        else:
            minx, miny, maxx, maxy = bounds
            j0 = int(np.clip(np.floor((minx - t.c) / size), 0, nbx))
            j1 = int(np.clip(np.floor((maxx - t.c) / size) + 1, 0, nbx))
            i0 = int(np.clip(np.floor((t.f - maxy) / size), 0, nby))
            i1 = int(np.clip(np.floor((t.f - miny) / size) + 1, 0, nby))
        if i1 <= i0 or j1 <= j0:
            return np.full((0, 0), np.nan), t.c + j0 * size, t.f - i0 * size, size
        win = Window(j0 * BLOCK, i0 * BLOCK, (j1 - j0) * BLOCK, (i1 - i0) * BLOCK)
        a = src.read(1, window=win).astype(np.float64)
        nodata = src.nodata
    valid = np.isfinite(a) & (a != nodata) if nodata is not None else np.isfinite(a)
    a = np.where(valid, a, 0.0).reshape(i1 - i0, BLOCK, j1 - j0, BLOCK)
    n = valid.reshape(a.shape).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = a.sum(axis=(1, 3)) / n
    mean[n == 0] = np.nan
    return mean, t.c + j0 * size, t.f - i0 * size, size


def assign_river_protection(x: np.ndarray, y: np.ndarray, path: str | Path,
                            fill_km: float = FILL_KM) -> pd.DataFrame:
    """Protection standard for points x, y (EPSG:3035 metres, feature centroids).

    Columns: prot_rp (the standard used), block_rp (the 5 km block under the point,
    0 if none: MIRACA's value), source (0 none, 1 block, 2 fill), fill_km (edge
    distance to the filling block, NaN unless filled), fill_x / fill_y (nearest
    point of that block).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = pd.DataFrame({"prot_rp": np.zeros(len(x)), "block_rp": np.zeros(len(x)),
                        "source": np.full(len(x), NONE, np.int8), "fill_km": np.nan,
                        "fill_x": np.nan, "fill_y": np.nan})
    if not len(x):
        return out
    reach = fill_km * 1000.0
    margin = reach + BLOCK * 500.0 * 2
    mean, x_left, y_top, size = block_means(
        path, (x.min() - margin, y.min() - margin, x.max() + margin, y.max() + margin))
    if mean.size == 0:
        return out

    j = np.floor((x - x_left) / size).astype(np.int64)
    i = np.floor((y_top - y) / size).astype(np.int64)
    inside = (i >= 0) & (i < mean.shape[0]) & (j >= 0) & (j < mean.shape[1])
    own = np.full(len(x), np.nan)
    own[inside] = mean[i[inside], j[inside]]
    block_rp = np.nan_to_num(own, nan=0.0).clip(min=0.0)
    out["block_rp"] = block_rp
    out["prot_rp"] = block_rp
    out.loc[block_rp > 0, "source"] = FROM_BLOCK

    todo = np.flatnonzero(block_rp == 0)
    pos_i, pos_j = np.nonzero(np.nan_to_num(mean, nan=0.0) > 0)
    if not todo.size or not pos_i.size or reach <= 0:
        return out
    cx = x_left + (pos_j + 0.5) * size
    cy = y_top - (pos_i + 0.5) * size
    vals = mean[pos_i, pos_j]
    tree = cKDTree(np.column_stack([cx, cy]))
    half = size / 2
    # A block's edge is within reach only if its centre is within reach + half-diagonal.
    cands = tree.query_ball_point(np.column_stack([x[todo], y[todo]]), r=reach + half * np.sqrt(2))
    prot, src, dist, fx, fy = (out[c].to_numpy().copy() for c in ("prot_rp", "source", "fill_km",
                                                                    "fill_x", "fill_y"))
    for k, c in zip(todo, cands):
        if not c:
            continue
        c = np.asarray(c)
        px = np.clip(x[k], cx[c] - half, cx[c] + half)
        py = np.clip(y[k], cy[c] - half, cy[c] + half)
        d = np.hypot(px - x[k], py - y[k])
        ok = d <= reach
        if not ok.any():
            continue
        c, d, px, py = c[ok], d[ok], px[ok], py[ok]
        best = np.lexsort((-vals[c], d))[0]       # nearest; on a tie, the higher standard
        prot[k], src[k], dist[k] = vals[c[best]], FROM_FILL, d[best] / 1000.0
        fx[k], fy[k] = px[best], py[best]
    out["prot_rp"], out["source"], out["fill_km"], out["fill_x"], out["fill_y"] = prot, src, dist, fx, fy
    return out
