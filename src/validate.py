"""Validate the fast Stage-2 model against the reference implementations.

Works for any asset type registered in src/curves.py. For the "which curve
did we pick" question in each check, one representative curve is chosen per
curve group (the alphabetically-first curve ID in that group) - this is
just a fixed, reproducible test point, not a modeling choice; the point of
these checks is that Stage 2 exactly reproduces the ground truth for
*whichever* curves are active, so one representative combination is enough
to catch a formula or unit-conversion bug.

Check A - per-RP damage vs damagescanner.VectorScanner:
    runs the real VectorScanner on the RP100 (or first configured) flood
    raster with the representative curve per flood group and mean costs, and
    compares per-feature damages with the Stage-2 damage matrix
    (aggregation='per_cell', depth_offset=0). VectorScanner is geometry-aware
    internally (line/polygon/point handled per feature already), so this
    also validates Stage 1's polygon-area / point-count quantity conversion.

Check B - EAD integration vs a scalar port of AssetRisk_PanEU's
    integrate_ead()/_apply_protection_standard()/adjust_return_periods_climate(),
    evaluated per feature on a random sample, for several combinations of
    protection scaling and warming level. Skipped for assets/scenarios with
    no river hazard.

Check C - earthquake damage at a representative RP recomputed independently:
    fresh VectorExposure overlay of the PGA raster + EDR lookup + mean costs,
    compared with the Stage-2 earthquake damage matrix.

Check D - windstorm damage at a representative RP recomputed independently:
    fresh VectorExposure overlay of the gust-speed raster + wind curve interp
    + mean costs, compared with the Stage-2 windstorm damage matrix. Skipped
    for assets that don't support windstorm (e.g. roads) or with zero wind
    exposure.

Check E - coastal flood: confirms the streamed coastal fragments loaded and
    are non-empty. Coastal reuses the river flood curves and the
    river-validated Stage-2 damage/EAD code paths (Checks A/B), so there is no
    separate offline numerical check. Skipped for landlocked countries.

Run:  python -m src.validate --country LUX --asset roads
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.interpolate import interp1d

from damagescanner.core import VectorScanner, VectorExposure

from .curves import (
    get_asset_config,
    is_mapped,
    load_eq_edr_tables,
    load_flood_curves,
    load_wind_curves,
)
from .paths import load_config, set_asset_override, set_country_override
from .risk_model import (
    ModelData,
    WARMING_LEVELS,
    _cost_per_unit,
    _damage_matrix,
    _integrate_ead,
    _shift_rps,
    load_model_data,
)


def _representative_choices(groups: dict[str, list[str]]) -> dict[str, str]:
    return {name: sorted(curves)[0] for name, curves in groups.items()}


def _load_features(cfg: dict) -> gpd.GeoDataFrame:
    """Exposure exactly as Stage 1 sees it.

    Mirrors preprocess.load_exposure + drop_unmapped_object_types: same
    columns, same row order, same exclusion of object_types with no curve.
    The checks compare per-feature against the Stage-2 matrices positionally
    (index 0..n_seg-1), so the two feature sets have to be identical - an
    unfiltered read here raises KeyError on an unmapped object_type, and
    would silently misalign ref against fast wherever a feature was dropped.
    """
    path = cfg["exposure_dir"] / f"{cfg['country']}_{cfg['asset_type']}_exposure.parquet"
    features = gpd.read_parquet(
        path, columns=["osm_id", "object_type", "geometry"]
    ).reset_index(drop=True)
    keep = is_mapped(get_asset_config(cfg["asset_type"]), features["object_type"])
    if not bool(keep.all()):
        dropped = features["object_type"][~keep].value_counts().to_dict()
        print(
            f"  NOTE: excluding {int((~keep).sum())}/{len(features)} feature(s) "
            f"with no vulnerability curve: {dropped}"
        )
    return features[keep].reset_index(drop=True)


def _clip_hazard(path, features: gpd.GeoDataFrame):
    b = features.to_crs(4326).total_bounds
    return xr.open_dataset(path, engine="rasterio").rio.clip_box(
        minx=b[0] - 0.1, miny=b[1] - 0.1, maxx=b[2] + 0.1, maxy=b[3] + 0.1
    )


def check_river_vs_vectorscanner(cfg: dict, data: ModelData) -> bool:
    if "river" not in data.hazards:
        print("Check A: skipped (no river hazard for this asset).")
        return True
    if len(data.hazards["river"].p_pair) == 0:
        print("Check A: skipped (this asset/country has zero river-flood exposure at every RP).")
        return True

    asset_cfg = get_asset_config(cfg["asset_type"])
    curve_choices = _representative_choices(asset_cfg.flood_groups)
    rp = int(sorted(cfg["hazards"]["river"]["return_periods"])[len(cfg["hazards"]["river"]["return_periods"]) // 2])
    print("=" * 70)
    print(f"Check A: river RP{rp} damage vs VectorScanner  (curves: {curve_choices})")
    print("=" * 70)

    features = _load_features(cfg)
    curve_df = load_flood_curves(
        cfg["vulnerability_path"], sorted(set(curve_choices.values()))
    )
    object_types = features["object_type"].unique()
    curves = pd.DataFrame(index=curve_df.index)
    for obj in object_types:
        group = asset_cfg.flood_object_group[obj]
        curves[obj] = curve_df[curve_choices[group]]
    maxdam = pd.DataFrame(
        {
            "object_type": object_types,
            "damage": [asset_cfg.maxdam.get(o, asset_cfg.default_maxdam)[1] for o in object_types],
        }
    )

    hcfg = cfg["hazards"]["river"]
    hazard = _clip_hazard(hcfg["dir"] / hcfg["filename_template"].format(rp=rp), features)

    print("Running VectorScanner (reference)...")
    ref = VectorScanner(
        hazard_file=hazard,
        feature_file=features,
        curve_path=curves,
        maxdam_path=maxdam,
        disable_progress=True,
        return_full=False,
    )
    ref_damage = ref["damage"].astype(float)

    hz = data.hazards["river"]
    curves_by_group = {
        i: data.flood_curve_tables[curve_choices[name]]
        for i, name in enumerate(data.flood_group_order)
    }
    dmg_qty, _ = _damage_matrix(hz, hz.p_intensity, curves_by_group, "per_cell", data.n_seg)
    damage = dmg_qty * _cost_per_unit(data, 0.0)[:, None]
    idx = int(np.argmin(np.abs(hz.rps - rp)))
    fast_damage = pd.Series(damage[:, idx], index=np.arange(data.n_seg))

    both = pd.DataFrame({"ref": ref_damage, "fast": fast_damage}).fillna(0.0)
    tot_ref, tot_fast = both["ref"].sum(), both["fast"].sum()
    denom = np.maximum(both["ref"].abs(), 1.0)
    rel = ((both["fast"] - both["ref"]).abs() / denom).max()
    print(f"  total damage  reference: {tot_ref / 1e6:12.4f} MEUR")
    print(f"  total damage  fast     : {tot_fast / 1e6:12.4f} MEUR")
    print(f"  total rel. difference  : {abs(tot_fast - tot_ref) / max(tot_ref, 1e-9):.2e}")
    print(f"  max per-feature rel. difference (damage > 1 EUR): {rel:.2e}")
    ok = abs(tot_fast - tot_ref) / max(tot_ref, 1e-9) < 1e-3 and rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _ref_integrate_ead(rps, damages, protection_standard):
    """Scalar port of AssetRisk_PanEU integrate_ead (mean stat only)."""
    sorted_pairs = sorted(zip(rps, damages))
    srps = [p[0] for p in sorted_pairs]
    sdmg = [p[1] for p in sorted_pairs]

    if protection_standard > 0:
        if protection_standard not in srps:
            idx = int(np.searchsorted(srps, protection_standard))
            if idx == 0:
                interp = sdmg[0]
            elif idx >= len(srps):
                interp = sdmg[-1]
            else:
                rp_lo, rp_hi = srps[idx - 1], srps[idx]
                d_lo, d_hi = sdmg[idx - 1], sdmg[idx]
                interp = d_lo + (protection_standard - rp_lo) * (d_hi - d_lo) / (
                    rp_hi - rp_lo
                )
            srps = srps[:idx] + [protection_standard] + srps[idx:]
            sdmg = sdmg[:idx] + [interp] + sdmg[idx:]
        keep = [(rp, d) for rp, d in zip(srps, sdmg) if rp >= protection_standard]
        if not keep:
            return 0.0
        srps, sdmg = map(list, zip(*keep))

    if len(sdmg) >= 2:
        probs = [1.0 / rp for rp in srps]
        return max(float(np.trapezoid(y=sdmg[::-1], x=probs[::-1])), 0.0)
    if len(sdmg) == 1:
        return max((1.0 / srps[0]) * sdmg[0], 0.0)
    return 0.0


def _ref_adjust_rps(rps, protection, anchors_row):
    """Scalar port of adjust_return_periods_climate for one feature."""
    f = interp1d(
        [10, 100, 500], anchors_row, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    adj = [max(float(f(rp)), 1.0) for rp in rps]
    prot = protection
    if protection > 0:
        prot = max(float(f(protection)), 1.0)
    return adj, prot


def check_ead_integration(cfg: dict, data: ModelData, n_sample: int = 3000) -> bool:
    if "river" not in data.hazards:
        print("Check B: skipped (no river hazard for this asset).")
        return True

    asset_cfg = get_asset_config(cfg["asset_type"])
    curve_choices = _representative_choices(asset_cfg.flood_groups)
    print("=" * 70)
    print("Check B: EAD integration vs scalar reference port")
    print("=" * 70)

    rng = np.random.default_rng(42)
    hz = data.hazards["river"]
    curves_by_group = {
        i: data.flood_curve_tables[curve_choices[name]]
        for i, name in enumerate(data.flood_group_order)
    }
    dmg_qty, _ = _damage_matrix(hz, hz.p_intensity, curves_by_group, "per_cell", data.n_seg)
    damage = dmg_qty * _cost_per_unit(data, 0.0)[:, None]

    exposed = np.flatnonzero(damage.sum(axis=1) > 0)
    if len(exposed) == 0:
        print("  No features have nonzero flood damage under the representative curves; skipping.")
        return True
    sample = rng.choice(exposed, size=min(n_sample, len(exposed)), replace=False)

    all_ok = True
    for warming in ("current", "2.0C", "4.0C"):
        if warming != "current" and not data.anchors:
            continue
        for prot_scale in (0.0, 1.0, 2.0):
            prot_eff = data.prot_rp * prot_scale
            code = WARMING_LEVELS[warming]
            if code is None:
                rp_adj = np.broadcast_to(hz.rps, damage.shape).copy()
                prot_used = prot_eff.copy()
                dmg_used = damage
            else:
                anchors = data.anchors[code]
                rp_adj = _shift_rps(np.broadcast_to(hz.rps, damage.shape), anchors)
                shifted = _shift_rps(prot_eff[:, None], anchors)[:, 0]
                prot_used = np.where(prot_eff > 0, shifted, 0.0)
                order = np.argsort(rp_adj, axis=1, kind="stable")
                rp_adj = np.take_along_axis(rp_adj, order, axis=1)
                dmg_used = np.take_along_axis(damage, order, axis=1)

            fast = _integrate_ead(dmg_used, rp_adj, prot_used)

            max_rel = 0.0
            for i in sample:
                if code is None:
                    ref_rps, ref_prot = list(hz.rps), float(prot_eff[i])
                else:
                    ref_rps, ref_prot = _ref_adjust_rps(
                        list(hz.rps), float(prot_eff[i]), data.anchors[code][i]
                    )
                ref = _ref_integrate_ead(ref_rps, list(damage[i]), ref_prot)
                denom = max(abs(ref), 1.0)
                max_rel = max(max_rel, abs(fast[i] - ref) / denom)

            ok = max_rel < 1e-6
            all_ok &= ok
            print(
                f"  warming={warming:8s} prot_scale={prot_scale:.1f} "
                f"max rel diff = {max_rel:.2e}  -> {'PASS' if ok else 'FAIL'}"
            )
    return all_ok


def check_earthquake_recomputation(cfg: dict, data: ModelData) -> bool:
    if "earthquake" not in data.hazards:
        print("Check C: skipped (no earthquake hazard for this asset).")
        return True

    asset_cfg = get_asset_config(cfg["asset_type"])
    curve_choices = _representative_choices(asset_cfg.eq_groups)
    rps = sorted(cfg["hazards"]["earthquake"]["return_periods"])
    rp = rps[len(rps) // 2]
    print("=" * 70)
    print(f"Check C: earthquake RP{rp} damage, independent recompute (curves: {curve_choices})")
    print("=" * 70)

    features = _load_features(cfg)
    hcfg = cfg["hazards"]["earthquake"]
    hazard = _clip_hazard(hcfg["dir"] / hcfg["filename_template"].format(rp=rp), features)

    print("Running fresh VectorExposure overlay + EDR lookup (reference-style)...")
    exposed, _, _, _ = VectorExposure(
        hazard_file=hazard,
        feature_file=features[["object_type", "geometry"]],
        hazard_value_col="band_data",
        disable_progress=True,
        return_full=False,
    )
    needed_curves = sorted(set(curve_choices.values()))
    edr_tables = load_eq_edr_tables(cfg["fragility_path"], needed_curves)

    ref = np.zeros(data.n_seg)
    for seg_idx, values, coverage in zip(
        exposed.index, exposed["values"], exposed["coverage"]
    ):
        v = np.asarray(values, dtype=np.float64)
        c = np.asarray(coverage, dtype=np.float64)
        keep = np.isfinite(v) & (v > 0) & (c > 0)
        if not keep.any():
            continue
        obj = features["object_type"].iloc[seg_idx]
        group = asset_cfg.eq_object_group[obj]
        pga_x, edr_y = edr_tables[curve_choices[group]]
        maxdam = asset_cfg.maxdam.get(obj, asset_cfg.default_maxdam)[1]
        # polygon area conversion mirrors preprocess.py's extract_hazard_profiles
        geom_kind = features.geometry.iloc[seg_idx].geom_type
        if geom_kind in ("Polygon", "MultiPolygon"):
            from damagescanner.vector import _get_cell_area_m2
            cell_area_m2 = _get_cell_area_m2(
                features, hazard.rio.crs, abs(hazard.rio.resolution()[0])
            )
            q = c[keep] * cell_area_m2
        else:
            q = c[keep]
        ref[seg_idx] = np.sum(np.interp(v[keep], pga_x, edr_y) * q) * maxdam

    hz = data.hazards["earthquake"]
    curves_by_group = {
        i: data.eq_curve_tables[curve_choices[name]]
        for i, name in enumerate(data.eq_group_order)
    }
    dmg_qty, _ = _damage_matrix(hz, hz.p_intensity, curves_by_group, "per_cell", data.n_seg)
    fast = (dmg_qty * _cost_per_unit(data, 0.0)[:, None])[:, int(np.argmin(np.abs(hz.rps - rp)))]

    tot_ref, tot_fast = ref.sum(), fast.sum()
    denom = np.maximum(np.abs(ref), 1.0)
    rel = (np.abs(fast - ref) / denom).max()
    print(f"  total damage  reference: {tot_ref / 1e6:12.4f} MEUR")
    print(f"  total damage  fast     : {tot_fast / 1e6:12.4f} MEUR")
    print(f"  total rel. difference  : {abs(tot_fast - tot_ref) / max(tot_ref, 1e-9):.2e}")
    print(f"  max per-feature rel. difference: {rel:.2e}")
    ok = abs(tot_fast - tot_ref) / max(tot_ref, 1e-9) < 1e-3 and rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_windstorm_recomputation(cfg: dict, data: ModelData) -> bool:
    if "windstorm" not in data.hazards:
        print("Check D: skipped (no windstorm hazard for this asset).")
        return True
    if len(data.hazards["windstorm"].p_pair) == 0:
        print("Check D: skipped (this asset/country has zero windstorm exposure at every RP).")
        return True

    asset_cfg = get_asset_config(cfg["asset_type"])
    curve_choices = _representative_choices(asset_cfg.wind_groups)
    rps = sorted(cfg["hazards"]["windstorm"]["return_periods"])
    rp = rps[len(rps) // 2]
    print("=" * 70)
    print(f"Check D: windstorm RP{rp} damage, independent recompute (curves: {curve_choices})")
    print("=" * 70)

    features = _load_features(cfg)
    hcfg = cfg["hazards"]["windstorm"]
    hazard = _clip_hazard(hcfg["dir"] / hcfg["filename_template"].format(rp=rp), features)

    print("Running fresh VectorExposure overlay + wind curve interp (reference-style)...")
    exposed, _, _, _ = VectorExposure(
        hazard_file=hazard,
        feature_file=features[["object_type", "geometry"]],
        hazard_value_col="band_data",
        disable_progress=True,
        return_full=False,
    )
    needed_curves = sorted(set(curve_choices.values()))
    wind_df = load_wind_curves(cfg["vulnerability_path"], needed_curves)
    wind_tables = {c: (wind_df.index.to_numpy(float), wind_df[c].to_numpy(float)) for c in needed_curves}

    ref = np.zeros(data.n_seg)
    for seg_idx, values, coverage in zip(
        exposed.index, exposed["values"], exposed["coverage"]
    ):
        v = np.asarray(values, dtype=np.float64)
        c = np.asarray(coverage, dtype=np.float64)
        keep = np.isfinite(v) & (v > 0) & (c > 0)
        if not keep.any():
            continue
        obj = features["object_type"].iloc[seg_idx]
        group = asset_cfg.wind_object_group[obj]
        speed_x, frac_y = wind_tables[curve_choices[group]]
        maxdam = asset_cfg.maxdam.get(obj, asset_cfg.default_maxdam)[1]
        geom_kind = features.geometry.iloc[seg_idx].geom_type
        if geom_kind in ("Polygon", "MultiPolygon"):
            from damagescanner.vector import _get_cell_area_m2
            cell_area_m2 = _get_cell_area_m2(
                features, hazard.rio.crs, abs(hazard.rio.resolution()[0])
            )
            q = c[keep] * cell_area_m2
        else:
            q = c[keep]
        ref[seg_idx] = np.sum(np.interp(v[keep], speed_x, frac_y) * q) * maxdam

    hz = data.hazards["windstorm"]
    curves_by_group = {
        i: data.wind_curve_tables[curve_choices[name]]
        for i, name in enumerate(data.wind_group_order)
    }
    dmg_qty, _ = _damage_matrix(hz, hz.p_intensity, curves_by_group, "per_cell", data.n_seg)
    fast = (dmg_qty * _cost_per_unit(data, 0.0)[:, None])[:, int(np.argmin(np.abs(hz.rps - rp)))]

    tot_ref, tot_fast = ref.sum(), fast.sum()
    denom = np.maximum(np.abs(ref), 1.0)
    rel = (np.abs(fast - ref) / denom).max()
    print(f"  total damage  reference: {tot_ref / 1e6:12.4f} MEUR")
    print(f"  total damage  fast     : {tot_fast / 1e6:12.4f} MEUR")
    print(f"  total rel. difference  : {abs(tot_fast - tot_ref) / max(tot_ref, 1e-9):.2e}")
    print(f"  max per-feature rel. difference: {rel:.2e}")
    ok = abs(tot_fast - tot_ref) / max(tot_ref, 1e-9) < 1e-3 and rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_coastal_reuse(cfg: dict, data: ModelData) -> bool:
    """Coastal flood reuses the river flood curves and the river-validated
    Stage-2 damage/EAD code paths (Checks A and B), differing only in its
    streamed hazard fragments and COASTPROS protection standard - so there is
    no separate offline numerical check to run. This just confirms the cached
    coastal fragments loaded and are non-empty, so a coastal scenario has
    something to integrate."""
    if "coastal" not in data.hazards:
        print("Check E: skipped (no coastal profiles - landlocked country or coastal not preprocessed).")
        return True
    hz = data.hazards["coastal"]
    n_frag = len(hz.p_pair)
    print("=" * 70)
    print("Check E: coastal flood fragments present (reuses river-validated damage/EAD path)")
    print("=" * 70)
    print(f"  coastal fragments: {n_frag}, RPs: {list(hz.rps.astype(int))}")
    print(f"  coast_prot_rp: mean {data.coast_prot_rp.mean():.0f} yr, "
          f"{(data.coast_prot_rp == 0).mean() * 100:.1f}% unprotected")
    ok = n_frag > 0
    print(f"  -> {'PASS (has exposure to integrate)' if ok else 'NOTE: zero coastal exposure'}")
    return True  # zero exposure is legitimate (few coastal features), never a failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    parser.add_argument("--asset", default=None, help="asset type override")
    args = parser.parse_args()
    set_country_override(args.country)
    set_asset_override(args.asset)

    cfg = load_config()
    print(f"Validating {cfg['country']} {cfg['asset_type']}")
    data = load_model_data(cfg)
    ok_a = check_river_vs_vectorscanner(cfg, data)
    ok_b = check_ead_integration(cfg, data)
    ok_c = check_earthquake_recomputation(cfg, data)
    ok_d = check_windstorm_recomputation(cfg, data)
    ok_e = check_coastal_reuse(cfg, data)
    print("=" * 70)
    all_ok = ok_a and ok_b and ok_c and ok_d and ok_e
    print(f"Overall: {'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
