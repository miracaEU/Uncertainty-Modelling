"""Stage 1: one-off geospatial preprocessing (expensive, no uncertainty factors).

Extracts everything the fast risk model (Stage 2) needs, so that no GIS
operation has to be repeated during the thousands of EMA Workbench runs, for
any asset type (roads, airports, education, power - anything registered in
src/curves.py's ASSET_CONFIGS) and any of its geometry types (line/polygon/
point features can coexist within one asset, e.g. power has all three):

  1. Per feature, per hazard, per return period: the exact per-raster-cell
     fragments (hazard intensity, exposed quantity). "Quantity" is in the
     physically correct unit for that feature's geometry - metres for lines,
     m^2 for polygons, a count of 1 for points - computed here so Stage 2 can
     multiply directly by a matching EUR/unit cost without knowing about
     geometry at all. This mirrors damagescanner's own per-geometry-type
     damage formula (vector.py::_get_damage_per_object) exactly: coverage is
     already in metres for lines (VectorExposure converts it), a 0-1 fraction
     of one raster cell for polygons (multiplied here by cell_area_m2), and a
     fixed 1 for points.
  2. Per feature: FLOPROS flood protection standard (design return period),
     sampled at the feature centroid from the 500 m protection raster.
  3. Per feature: HydroBASINS basin id + the basin-level "new return period"
     anchors for RP10/100/500 under 1.5/2.0/3.0/4.0 degC warming
     (river flood only - absolute shifted RPs, as in AssetRisk_PanEU).

Outputs (in intermediate_dir), shared across every modeling scenario for the
same (country, asset) pair - scenarios only vary Stage 2:
  {country}_{asset}_segments.parquet          - one row per feature
  {country}_{asset}_{hazard}_profiles.parquet - one row per (feature, RP, cell)
  {country}_{asset}_meta.json                 - provenance and summary stats

Run:  python -m src.preprocess --country DNK --asset power
      python -m src.preprocess --hazards earthquake   # subset
"""

import argparse
import json
import time
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from damagescanner.core import VectorExposure

from .curves import get_asset_config, maxdam_arrays, report_class_for
from .paths import base_stem, hazard_stem, load_config, set_asset_override, set_country_override

WARMING_CODES = ("15", "20", "30", "40")
ANCHOR_RPS = (10, 100, 500)

# shapely geom_type -> quantity kind: 0 = line (metres), 1 = polygon (m^2), 2 = point (count)
GEOM_KIND = {
    "LineString": 0, "MultiLineString": 0,
    "Polygon": 1, "MultiPolygon": 1,
    "Point": 2,
}


def load_exposure(cfg: dict) -> gpd.GeoDataFrame:
    path = cfg["exposure_dir"] / f"{cfg['country']}_{cfg['asset_type']}_exposure.parquet"
    print(f"Loading exposure: {path}")
    gdf = gpd.read_parquet(path, columns=["osm_id", "object_type", "geometry"])
    gdf = gdf.reset_index(drop=True)
    print(f"  {len(gdf)} features, CRS EPSG:{gdf.crs.to_epsg()}")
    return gdf


def classify_geometry(features: gpd.GeoDataFrame) -> np.ndarray:
    """Per-feature geometry kind: 0=line, 1=polygon, 2=point."""
    geom_types = features.geometry.geom_type
    kind = geom_types.map(GEOM_KIND)
    if kind.isna().any():
        bad = sorted(geom_types[kind.isna()].unique())
        raise ValueError(f"Unsupported geometry type(s) in exposure data: {bad}")
    return kind.to_numpy(np.int8)


def extract_hazard_profiles(
    features: gpd.GeoDataFrame, geom_kind: np.ndarray, hazard_cfg: dict
) -> tuple[pd.DataFrame, dict, float | None]:
    """Overlay every return-period raster of one hazard; return fragment table.

    Returns (profiles, exposed_quantity_per_rp, cell_area_m2). cell_area_m2 is
    None if the hazard/country combination has no polygon features (never
    needed in that case).
    """
    bounds = features.to_crs(4326).total_bounds
    buffer = 0.1
    clip_kw = dict(
        minx=bounds[0] - buffer,
        miny=bounds[1] - buffer,
        maxx=bounds[2] + buffer,
        maxy=bounds[3] + buffer,
    )

    seg_parts, rp_parts, val_parts, qty_parts = [], [], [], []
    exposed_qty_per_rp = {}
    cell_area_m2 = None

    for rp in hazard_cfg["return_periods"]:
        t0 = time.time()
        path = hazard_cfg["dir"] / hazard_cfg["filename_template"].format(rp=rp)
        hazard = xr.open_dataset(path, engine="rasterio").rio.clip_box(**clip_kw)

        exposed, _, _, rp_cell_area_m2 = VectorExposure(
            hazard_file=hazard,
            feature_file=features[["object_type", "geometry"]],
            hazard_value_col="band_data",
            disable_progress=True,
            return_full=False,
        )
        hazard.close()
        if rp_cell_area_m2 is not None:
            cell_area_m2 = rp_cell_area_m2

        n_frag = 0
        rp_exposed_qty = 0.0
        for seg_idx, values, coverage in zip(
            exposed.index, exposed["values"], exposed["coverage"]
        ):
            v = np.asarray(values, dtype=np.float64)
            c = np.asarray(coverage, dtype=np.float64)
            keep = np.isfinite(v) & (v > 0) & (c > 0)
            if not keep.any():
                continue
            v, c = v[keep], c[keep]
            if geom_kind[seg_idx] == 1:  # polygon: fraction-of-cell -> m^2
                q = c * cell_area_m2
            else:  # line: already metres: point: already 1
                q = c
            seg_parts.append(np.full(len(v), seg_idx, dtype=np.int32))
            val_parts.append(v.astype(np.float32))
            qty_parts.append(q.astype(np.float32))
            rp_parts.append(np.full(len(v), rp, dtype=np.int32))
            n_frag += len(v)
            rp_exposed_qty += float(q.sum())

        exposed_qty_per_rp[rp] = rp_exposed_qty
        print(
            f"  RP{rp:>5}: {n_frag:>9} exposed cell fragments, "
            f"exposed quantity {rp_exposed_qty:14.1f} "
            f"({time.time() - t0:.1f}s)"
        )

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
        print("  NOTE: zero exposed fragments at every RP for this hazard - "
              "this asset/country combination has no exposure to it.")
        profiles = pd.DataFrame(
            {
                "seg": pd.Series(dtype=np.int32),
                "rp": pd.Series(dtype=np.int32),
                "intensity": pd.Series(dtype=np.float32),
                "quantity": pd.Series(dtype=np.float32),
            }
        )
    return profiles, exposed_qty_per_rp, cell_area_m2


def sample_protection_standards(features: gpd.GeoDataFrame, cfg: dict) -> np.ndarray:
    """Sample the FLOPROS-based protection raster (EPSG:3035, 500 m) at centroids.

    Note: AssetRisk_PanEU coarsens this raster 10x (to 5 km, mean) before the
    overlay for memory reasons; for a single country we sample at native
    resolution instead, which is more precise.
    """
    print("Sampling protection standards at feature centroids...")
    ds = xr.open_dataset(cfg["protection_standard_path"], engine="rasterio")
    ds = ds.rio.write_crs("EPSG:3035")
    feats_3035 = features.to_crs(3035)
    b = feats_3035.total_bounds
    ds = ds.rio.clip_box(minx=b[0] - 5000, miny=b[1] - 5000, maxx=b[2] + 5000, maxy=b[3] + 5000)

    cent = feats_3035.geometry.centroid
    vals = (
        ds["band_data"]
        .squeeze("band", drop=True)
        .sel(
            x=xr.DataArray(cent.x.to_numpy(), dims="pts"),
            y=xr.DataArray(cent.y.to_numpy(), dims="pts"),
            method="nearest",
        )
        .to_numpy()
    )
    ds.close()
    vals = np.nan_to_num(vals, nan=0.0)
    vals = np.clip(vals, 0.0, None)
    print(
        f"  protection RP: mean {vals.mean():.0f} yr, max {vals.max():.0f} yr, "
        f"{(vals == 0).mean() * 100:.1f}% unprotected"
    )
    return vals.astype(np.float32)


def join_basin_anchors(features: gpd.GeoDataFrame, cfg: dict) -> pd.DataFrame:
    """Spatially join feature centroids to basins; return anchor new-RP columns."""
    print("Joining basin-level climate RP shifts...")
    basins = gpd.read_parquet(cfg["basin_data_path"])
    anchor_cols = [
        f"{a}_rp_change_{w}" for a in ANCHOR_RPS for w in WARMING_CODES
    ]
    basins = basins[["HYBAS_ID", "geometry"] + anchor_cols]

    cent = gpd.GeoDataFrame(
        geometry=features.geometry.to_crs(basins.crs).centroid, index=features.index
    )
    joined = gpd.sjoin(cent, basins, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(features.index)

    out = joined[["HYBAS_ID"] + anchor_cols].copy()
    out = out.rename(
        columns={
            f"{a}_rp_change_{w}": f"new_rp{a}_w{w}"
            for a in ANCHOR_RPS
            for w in WARMING_CODES
        }
    )
    n_matched = out["HYBAS_ID"].notna().sum()
    print(f"  {n_matched}/{len(out)} features matched to a basin")
    return out


def build_segments(features: gpd.GeoDataFrame, geom_kind: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Asset-generic feature attributes (curve groups, costs, protection, basins)."""
    asset_cfg = get_asset_config(cfg["asset_type"])
    object_types = features["object_type"]

    segments = pd.DataFrame(
        {
            "osm_id": features["osm_id"].to_numpy(),
            "object_type": object_types.to_numpy(),
            "geom_kind": geom_kind,
        }
    )
    segments["flood_group"] = object_types.map(asset_cfg.flood_object_group).to_numpy()
    segments["eq_group"] = object_types.map(asset_cfg.eq_object_group).to_numpy()
    segments["report_class"] = report_class_for(asset_cfg, object_types).to_numpy()

    md = maxdam_arrays(asset_cfg, object_types)
    segments["maxdam_min"] = md[:, 0].astype(np.float32)
    segments["maxdam_mean"] = md[:, 1].astype(np.float32)
    segments["maxdam_max"] = md[:, 2].astype(np.float32)

    n_unmapped_flood = segments["flood_group"].isna().sum()
    n_unmapped_eq = segments["eq_group"].isna().sum()
    if n_unmapped_flood or n_unmapped_eq:
        unknown = sorted(
            set(object_types[segments["flood_group"].isna()])
            | set(object_types[segments["eq_group"].isna()])
        )
        raise ValueError(
            f"{n_unmapped_flood} features have no flood curve group, "
            f"{n_unmapped_eq} have no EQ curve group. Unknown object_type(s): "
            f"{unknown}. Add them to FLOOD_CURVES_RAW/EQ_CURVES_RAW in src/curves.py."
        )

    segments["prot_rp"] = sample_protection_standards(features, cfg)

    basin_df = join_basin_anchors(features, cfg)
    return pd.concat([segments, basin_df.reset_index(drop=True)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hazards", nargs="+", default=None,
        help="subset of configured hazards to (re)process (default: all)",
    )
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    parser.add_argument("--asset", default=None, help="asset type override (roads/airports/education/power)")
    args = parser.parse_args()

    set_country_override(args.country)
    set_asset_override(args.asset)
    cfg = load_config()
    hazards = args.hazards or list(cfg["hazards"].keys())
    unknown = set(hazards) - set(cfg["hazards"])
    if unknown:
        raise SystemExit(f"Unknown hazards {unknown}; configured: {list(cfg['hazards'])}")

    print(f"Country: {cfg['country']}  Asset: {cfg['asset_type']}")
    t_start = time.time()

    features = load_exposure(cfg)
    geom_kind = classify_geometry(features)
    kind_counts = pd.Series(geom_kind).map({0: "line", 1: "polygon", 2: "point"}).value_counts()
    print(f"  geometry mix: {kind_counts.to_dict()}")

    segments = build_segments(features, geom_kind, cfg)
    seg_path = cfg["intermediate_dir"] / f"{base_stem(cfg)}_segments.parquet"
    segments.to_parquet(seg_path, index=False)
    print(f"Saved {seg_path}")

    meta_path = cfg["intermediate_dir"] / f"{base_stem(cfg)}_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta.update(
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "country": cfg["country"],
            "asset_type": cfg["asset_type"],
            "n_segments": int(len(segments)),
            "geometry_mix": kind_counts.to_dict(),
        }
    )

    for hazard in hazards:
        print(f"\nExtracting {hazard} profiles (the expensive step)...")
        profiles, exposed_per_rp, cell_area_m2 = extract_hazard_profiles(
            features, geom_kind, cfg["hazards"][hazard]
        )
        prof_path = cfg["intermediate_dir"] / f"{hazard_stem(cfg, hazard)}_profiles.parquet"
        profiles.to_parquet(prof_path, index=False)
        print(f"Saved {prof_path}")
        meta.setdefault("hazards", {})[hazard] = {
            "return_periods": cfg["hazards"][hazard]["return_periods"],
            "n_profile_rows": int(len(profiles)),
            "n_segments_exposed": int(profiles["seg"].nunique()),
            "cell_area_m2": cell_area_m2,
            "exposed_quantity_per_rp": exposed_per_rp,
        }

    meta["elapsed_s"] = round(time.time() - t_start, 1)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {meta_path}\nDone in {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
