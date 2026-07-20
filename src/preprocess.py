"""Stage 1: one-off geospatial preprocessing (expensive, no uncertainty factors).

Extracts everything the fast risk model (Stage 2) needs, so that no GIS
operation has to be repeated during the thousands of EMA Workbench runs:

  1. Per road segment, per hazard, per return period: the exact per-raster-cell
     fragments (hazard intensity, exposed length in metres). This is what
     damagescanner's VectorScanner internally consumes; caching it makes every
     downstream modeling choice (curves, costs, intensity offsets, aggregation)
     a cheap numpy operation. Intensity units: river = water depth (m),
     earthquake = PGA (g).
  2. Per segment: FLOPROS flood protection standard (design return period),
     sampled at the segment centroid from the 500 m protection raster.
  3. Per segment: HydroBASINS basin id + the basin-level "new return period"
     anchors for RP10/100/500 under 1.5/2.0/3.0/4.0 degC warming
     (river flood only — absolute shifted RPs, as in AssetRisk_PanEU).

Outputs (in intermediate_dir):
  {country}_{asset}_segments.parquet          — one row per road segment
  {country}_{asset}_{hazard}_profiles.parquet — one row per (segment, RP, cell)
  {country}_{asset}_meta.json                 — provenance and summary stats

Run:  python -m src.preprocess                      # all configured hazards
      python -m src.preprocess --hazards earthquake # subset
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

from .curves import MAIN_ROAD_TYPES, REPORT_CLASS, DEFAULT_REPORT_CLASS, maxdam_arrays
from .paths import load_config, base_stem, hazard_stem, set_country_override

WARMING_CODES = ("15", "20", "30", "40")
ANCHOR_RPS = (10, 100, 500)


def load_exposure(cfg: dict) -> gpd.GeoDataFrame:
    path = cfg["exposure_dir"] / f"{cfg['country']}_{cfg['asset_type']}_exposure.parquet"
    print(f"Loading exposure: {path}")
    gdf = gpd.read_parquet(path, columns=["osm_id", "object_type", "geometry"])
    gdf = gdf.reset_index(drop=True)
    print(f"  {len(gdf)} features, CRS EPSG:{gdf.crs.to_epsg()}")
    return gdf


def extract_hazard_profiles(
    features: gpd.GeoDataFrame, hazard_cfg: dict
) -> tuple[pd.DataFrame, dict]:
    """Overlay every return-period raster of one hazard; return fragment table."""
    bounds = features.to_crs(4326).total_bounds
    buffer = 0.1
    clip_kw = dict(
        minx=bounds[0] - buffer,
        miny=bounds[1] - buffer,
        maxx=bounds[2] + buffer,
        maxy=bounds[3] + buffer,
    )

    seg_parts, rp_parts, val_parts, len_parts = [], [], [], []
    exposed_length_per_rp = {}

    for rp in hazard_cfg["return_periods"]:
        t0 = time.time()
        path = hazard_cfg["dir"] / hazard_cfg["filename_template"].format(rp=rp)
        hazard = xr.open_dataset(path, engine="rasterio").rio.clip_box(**clip_kw)

        exposed, _, _, _ = VectorExposure(
            hazard_file=hazard,
            feature_file=features[["object_type", "geometry"]],
            hazard_value_col="band_data",
            disable_progress=True,
            return_full=False,
        )
        hazard.close()

        n_frag = 0
        rp_exposed_len = 0.0
        for seg_idx, values, coverage in zip(
            exposed.index, exposed["values"], exposed["coverage"]
        ):
            v = np.asarray(values, dtype=np.float64)
            c = np.asarray(coverage, dtype=np.float64)
            keep = np.isfinite(v) & (v > 0) & (c > 0)
            if not keep.any():
                continue
            v, c = v[keep], c[keep]
            seg_parts.append(np.full(len(v), seg_idx, dtype=np.int32))
            val_parts.append(v.astype(np.float32))
            len_parts.append(c.astype(np.float32))
            rp_parts.append(np.full(len(v), rp, dtype=np.int32))
            n_frag += len(v)
            rp_exposed_len += float(c.sum())

        exposed_length_per_rp[rp] = rp_exposed_len
        print(
            f"  RP{rp:>5}: {n_frag:>9} exposed cell fragments, "
            f"exposed length {rp_exposed_len / 1000:9.1f} km "
            f"({time.time() - t0:.1f}s)"
        )

    profiles = pd.DataFrame(
        {
            "seg": np.concatenate(seg_parts),
            "rp": np.concatenate(rp_parts),
            "intensity": np.concatenate(val_parts),
            "length_m": np.concatenate(len_parts),
        }
    ).sort_values(["seg", "rp"], ignore_index=True)
    return profiles, exposed_length_per_rp


def sample_protection_standards(features: gpd.GeoDataFrame, cfg: dict) -> np.ndarray:
    """Sample the FLOPROS-based protection raster (EPSG:3035, 500 m) at centroids.

    Note: AssetRisk_PanEU coarsens this raster 10x (to 5 km, mean) before the
    overlay for memory reasons; for a single small country we sample at native
    resolution instead, which is more precise.
    """
    print("Sampling protection standards at segment centroids...")
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
    """Spatially join segment centroids to basins; return anchor new-RP columns."""
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
    print(f"  {n_matched}/{len(out)} segments matched to a basin")
    return out


def build_segments(features: gpd.GeoDataFrame, cfg: dict) -> pd.DataFrame:
    """Hazard-independent segment attributes + river-specific extras."""
    segments = pd.DataFrame(
        {
            "osm_id": features["osm_id"].to_numpy(),
            "object_type": features["object_type"].to_numpy(),
        }
    )
    segments["length_m"] = features.to_crs(3035).geometry.length.to_numpy().astype(np.float32)
    segments["group"] = np.where(
        segments["object_type"].isin(MAIN_ROAD_TYPES), 0, 1
    ).astype(np.int8)
    segments["report_class"] = (
        segments["object_type"].map(REPORT_CLASS).fillna(DEFAULT_REPORT_CLASS)
    )
    md = maxdam_arrays(segments["object_type"])
    segments["maxdam_min"] = md[:, 0].astype(np.float32)
    segments["maxdam_mean"] = md[:, 1].astype(np.float32)
    segments["maxdam_max"] = md[:, 2].astype(np.float32)

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
    args = parser.parse_args()

    set_country_override(args.country)
    cfg = load_config()
    hazards = args.hazards or list(cfg["hazards"].keys())
    unknown = set(hazards) - set(cfg["hazards"])
    if unknown:
        raise SystemExit(f"Unknown hazards {unknown}; configured: {list(cfg['hazards'])}")

    t_start = time.time()
    features = load_exposure(cfg)

    segments = build_segments(features, cfg)
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
            "total_length_km": float(segments["length_m"].sum() / 1000),
        }
    )

    for hazard in hazards:
        print(f"\nExtracting {hazard} profiles (the expensive step)...")
        profiles, exposed_per_rp = extract_hazard_profiles(features, cfg["hazards"][hazard])
        prof_path = cfg["intermediate_dir"] / f"{hazard_stem(cfg, hazard)}_profiles.parquet"
        profiles.to_parquet(prof_path, index=False)
        print(f"Saved {prof_path}")
        meta.setdefault("hazards", {})[hazard] = {
            "return_periods": cfg["hazards"][hazard]["return_periods"],
            "n_profile_rows": int(len(profiles)),
            "n_segments_exposed": int(profiles["seg"].nunique()),
            "exposed_length_km_per_rp": {
                str(rp): v / 1000 for rp, v in exposed_per_rp.items()
            },
        }

    meta["elapsed_s"] = round(time.time() - t_start, 1)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {meta_path}\nDone in {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
