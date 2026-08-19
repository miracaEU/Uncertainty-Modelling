"""Coastal flood Stage-1 extraction, following AssetRisk_PanEU/hazard_coastal.py.

Coastal flooding is modelled exactly like river flooding downstream - it uses
the SAME depth-damage curves (the F_Vuln_Depth sheet, via the asset's flood
curve groups) and the same EAD integration - so once Stage 1 has cached its
exposure fragments, Stage 2 (src/risk_model.py) treats it just like river.
The ONE thing that is different, and the reason this needs its own module, is
where the hazard rasters come from: unlike river/earthquake/windstorm (local
GeoTIFFs), the CoCLiCo coastal flood maps are streamed at runtime from a remote
STAC catalogue on Google Cloud, one tile at a time. That is a network- and
package-dependency (pystac_client), imported lazily below so the rest of the
pipeline runs without it.

Two Stage-1 products, mirroring the local-raster hazards:
  extract_coastal_profiles()   -> the {country}_{asset}_coastal_profiles.parquet
                                   fragment table (seg, rp, intensity, quantity),
                                   identical schema to the other hazards, built
                                   by running damagescanner's VectorExposure on
                                   each streamed tile and accumulating fragments
                                   per (feature, return period).
  sample_coastal_protection()  -> per-feature coastal design return period, from
                                   the local COASTPROS-EU table joined to NUTS2
                                   regions (no network needed for this part).

Baseline only: the 2010 / present-day LOW_DEFENDED scenario is used (the
reference's 2050/2100 SSP sea-level-rise horizons are the coastal analogue of
the river 'warming' factor and are left as a future extension - coastal
scenarios here carry no warming factor).

Faithful to hazard_coastal.py: same STAC catalogue/collection, the same tile
filtering (LOW_DEFENDED, present horizon, skip 'static'), and the same
COASTPROS-EU -> NUTS2 protection assignment (modelled RP, country-mean then 0
fallback). It is simplified in one place: we key fragments by our own integer
feature index rather than (osm_id, LAU), because our exposure tables are one
row per feature already, so the reference's cross-LAU aggregation is unneeded.
"""

import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from damagescanner.core import VectorExposure

# geom kind codes match preprocess.py: 0 = line (m), 1 = polygon (m^2), 2 = point (count)


def _lazy_stac():
    """Import the STAC client lazily with a clear message if it's missing."""
    try:
        import pystac_client  # noqa: F401
        from pystac.extensions.projection import ProjectionExtension  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Coastal flooding needs the STAC client packages, which aren't installed:\n"
            f"    {e}\n"
            "Install them into the pipeline venv:\n"
            "    uv pip install pystac-client pystac\n"
            "(Only the coastal hazard needs these; the rest of the pipeline does not.)"
        )
    import pystac_client
    from pystac.extensions.projection import ProjectionExtension

    return pystac_client, ProjectionExtension


def _read_tile(href: str) -> xr.Dataset | None:
    """Open one remote GeoTIFF tile as an xarray Dataset (band_data). None on failure.

    Transcribed from hazard_coastal.py::_read_tile - builds explicit x/y coords
    from the raster bounds so the tile carries a CRS damagescanner can use.
    """
    import rasterio
    import rioxarray  # noqa: F401 - registers the .rio accessor

    try:
        with rasterio.open(href) as src:
            data = src.read(1).astype(np.float32)
            xs = np.linspace(src.bounds.left, src.bounds.right, src.width, endpoint=False)
            ys = np.linspace(src.bounds.top, src.bounds.bottom, src.height, endpoint=False)
            crs = src.crs.to_string() if src.crs else "EPSG:3035"
        da = xr.DataArray(
            data[np.newaxis],
            dims=("band", "y", "x"),
            coords={"band": [1], "x": xs, "y": ys, "spatial_ref": 0},
        )
        ds = xr.Dataset({"band_data": da})
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        ds = ds.rio.write_crs(crs, inplace=False)
        return ds
    except Exception as e:  # noqa: BLE001 - a bad tile shouldn't abort the whole stream
        print(f"  [coastal] WARNING: could not read tile {href}: {e}")
        return None


def _stream_tiles(features_3035: gpd.GeoDataFrame, coastal_cfg: dict):
    """Yield (return_period, tile_dataset) for the baseline coastal scenario.

    Faithful to hazard_coastal.py::stream_coastal_tiles: connect to the STAC
    catalogue, keep only the configured horizon/scenario/defence tiles (skip
    'static'), parse the RP from the item name, and skip tiles that don't
    intersect any feature or are all-zero. Tiles are read (and the caller is
    expected to discard them) one at a time to bound memory.
    """
    import shapely

    pystac_client, ProjectionExtension = _lazy_stac()

    horizon = str(coastal_cfg.get("time_horizon", "2010"))
    climate = str(coastal_cfg.get("climate_scenario", "None"))
    defence = str(coastal_cfg.get("defence", "LOW_DEFENDED"))
    url = coastal_cfg["stac_url"]
    collection_id = coastal_cfg["collection"]

    print(f"  [coastal] connecting to STAC {url} (collection {collection_id})")
    catalog = pystac_client.Client.open(url)
    collection = catalog.get_child(id=collection_id)
    if collection is None:
        raise SystemExit(f"[coastal] STAC collection '{collection_id}' not found at {url}")

    feature_tree = shapely.STRtree(features_3035.geometry.values)

    n_yielded = 0
    for item in collection.get_items():
        name = "_".join(item.id.split("\\")).split(".")[0]
        if "static" in name or horizon not in name or climate not in name or defence not in name:
            continue
        rp = None
        for part in name.split("_")[:-1]:
            try:
                rp = int(part)
                break
            except ValueError:
                continue
        if rp is None:
            continue

        for i, asset_key in enumerate(item.assets):
            if i == 0:
                continue  # first asset is metadata
            asset = item.assets[asset_key]
            try:
                proj = ProjectionExtension.ext(asset)
                [ring] = proj.geometry["coordinates"]
                tile_geom = shapely.Polygon(ring)
            except Exception:  # noqa: BLE001
                continue
            if len(feature_tree.query(tile_geom)) == 0:
                continue  # no features under this tile
            tile = _read_tile(asset.href)
            if tile is None:
                continue
            if float(tile.band_data.max()) == 0:
                del tile
                continue
            n_yielded += 1
            yield rp, tile
            del tile
    if n_yielded == 0:
        print("  [coastal] NOTE: no intersecting coastal tiles found (landlocked / no coastal exposure).")


def extract_coastal_profiles(
    features: gpd.GeoDataFrame, geom_kind: np.ndarray, coastal_cfg: dict
) -> tuple[pd.DataFrame, dict, float | None]:
    """Stream coastal tiles and build the fragment table (seg, rp, intensity, quantity).

    Same return signature as preprocess.extract_hazard_profiles:
    (profiles, exposed_quantity_per_rp, cell_area_m2). Features are worked in
    EPSG:3035 (the CoCLiCo tiles' CRS); each tile is overlaid with
    VectorExposure and its exposed cell fragments accumulated per (feature, RP),
    with the same geometry-correct quantity conversion as the local-raster path
    (metres for lines, m^2 for polygons via cell_area, 1 for points).
    """
    features_3035 = features.to_crs(3035)

    seg_parts, rp_parts, val_parts, qty_parts = [], [], [], []
    exposed_qty_per_rp: dict[int, float] = {}
    cell_area_m2 = None
    tile_count = 0
    t0 = time.time()

    for rp, tile in _stream_tiles(features_3035, coastal_cfg):
        tile_count += 1
        # Restrict to features under this tile's bounds, keeping their global
        # (seg) index so VectorExposure's output index maps straight back.
        minx, miny, maxx, maxy = tile.rio.bounds()
        in_tile = features_3035.cx[minx:maxx, miny:maxy]
        if in_tile.empty:
            del tile
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exposed, _, _, rp_cell_area_m2 = VectorExposure(
                hazard_file=tile,
                feature_file=in_tile[["object_type", "geometry"]],
                hazard_value_col="band_data",
                disable_progress=True,
                return_full=False,
            )
        del tile
        if rp_cell_area_m2 is not None:
            cell_area_m2 = rp_cell_area_m2

        for seg_idx, values, coverage in zip(exposed.index, exposed["values"], exposed["coverage"]):
            v = np.asarray(values, dtype=np.float64)
            c = np.asarray(coverage, dtype=np.float64)
            keep = np.isfinite(v) & (v > 0) & (c > 0)
            if not keep.any():
                continue
            v, c = v[keep], c[keep]
            q = c * cell_area_m2 if geom_kind[seg_idx] == 1 else c
            seg_parts.append(np.full(len(v), seg_idx, dtype=np.int32))
            val_parts.append(v.astype(np.float32))
            qty_parts.append(q.astype(np.float32))
            rp_parts.append(np.full(len(v), rp, dtype=np.int32))
            exposed_qty_per_rp[rp] = exposed_qty_per_rp.get(rp, 0.0) + float(q.sum())

    if seg_parts:
        profiles = pd.DataFrame(
            {
                "seg": np.concatenate(seg_parts),
                "rp": np.concatenate(rp_parts),
                "intensity": np.concatenate(val_parts),
                "quantity": np.concatenate(qty_parts),
            }
        ).sort_values(["seg", "rp"], ignore_index=True)
    else:
        profiles = pd.DataFrame(
            {
                "seg": pd.Series(dtype=np.int32),
                "rp": pd.Series(dtype=np.int32),
                "intensity": pd.Series(dtype=np.float32),
                "quantity": pd.Series(dtype=np.float32),
            }
        )
    print(
        f"  [coastal] streamed {tile_count} tiles, {len(profiles)} fragments across "
        f"RPs {sorted(exposed_qty_per_rp)} ({time.time() - t0:.1f}s)"
    )
    configured = sorted(int(r) for r in coastal_cfg.get("return_periods", []))
    off_grid = [rp for rp in sorted(exposed_qty_per_rp) if rp not in configured]
    if off_grid:
        print(
            f"  [coastal] NOTE: streamed RPs {off_grid} are not in the configured "
            f"coastal return_periods {configured}. They are still integrated (the "
            f"risk model folds streamed RPs into the grid), but verify the STAC "
            f"item-name RP parsing in _stream_tiles if this looks wrong."
        )
    return profiles, exposed_qty_per_rp, cell_area_m2


def sample_coastal_protection(features: gpd.GeoDataFrame, coastal_cfg: dict) -> np.ndarray:
    """Per-feature coastal design return period from COASTPROS-EU joined to NUTS2.

    Transcribed (and simplified) from hazard_coastal.py::
    load_coastal_protection_standards: use each NUTS2 region's modelled RP,
    falling back to the country mean, then 0 (unprotected). Feature centroids
    are matched to NUTS2 by 'within', with a nearest-region fallback for
    coastal features whose centroid falls just outside the land polygons. No
    network needed - both inputs are local files.
    """
    coastpros_path = coastal_cfg["coastpros_path"]
    nuts2_path = coastal_cfg["nuts2_path"]
    sheet = coastal_cfg.get("protection_sheet", "COASTPROS-EU")
    rp_col = coastal_cfg.get("protection_rp_col", "MODELLED RETURN PERIOD")
    print("  [coastal] assigning coastal protection standards (COASTPROS-EU x NUTS2)...")

    df = pd.read_excel(coastpros_path, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    df[rp_col] = pd.to_numeric(df[rp_col], errors="coerce")
    country_mean = df.dropna(subset=[rp_col]).groupby("CNTR_CODE")[rp_col].mean()
    nuts2_rp = (
        df.assign(_rp=df.apply(
            lambda r: r[rp_col] if pd.notna(r[rp_col]) else country_mean.get(r["CNTR_CODE"], 0.0),
            axis=1,
        ))
        .set_index("NUTS2 ID")["_rp"]
        .to_dict()
    )

    nuts2 = gpd.read_parquet(nuts2_path) if str(nuts2_path).endswith(".parquet") else gpd.read_file(nuts2_path)
    nuts2 = nuts2[nuts2["LEVL_CODE"] == 2].to_crs(3035).copy()
    nuts2["_rp"] = nuts2.apply(
        lambda r: nuts2_rp.get(r["NUTS_ID"], country_mean.get(r["CNTR_CODE"], 0.0)), axis=1
    )

    feats_3035 = features.to_crs(3035)
    centroids = gpd.GeoDataFrame(geometry=feats_3035.geometry.centroid, index=features.index, crs=3035)
    cols = nuts2[["geometry", "_rp"]]
    joined = gpd.sjoin(centroids, cols, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    unmatched = joined.index[joined["_rp"].isna()]
    if len(unmatched) > 0:
        nearest = gpd.sjoin_nearest(centroids.loc[unmatched], cols, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched, "_rp"] = nearest["_rp"].values

    prot = joined["_rp"].reindex(features.index).fillna(0.0).clip(lower=0).to_numpy(np.float64)
    print(
        f"  [coastal] protection RP: mean {prot.mean():.0f} yr, max {prot.max():.0f} yr, "
        f"{(prot == 0).mean() * 100:.1f}% unprotected"
    )
    return prot.astype(np.float32)
