"""Validate the fast Stage-2 model against the reference implementations.

Check A — river per-RP damage vs damagescanner.VectorScanner:
    runs the real VectorScanner on the RP100 flood raster with one fixed curve
    per road class group and mean costs, and compares per-segment damages with
    the Stage-2 damage matrix (aggregation='per_cell', depth_offset=0).

Check B — EAD integration vs a scalar port of AssetRisk_PanEU's
    integrate_ead()/_apply_protection_standard()/adjust_return_periods_climate(),
    evaluated per segment on a random sample, for several combinations of
    protection scaling and warming level.

Check C — earthquake damage at RP476 recomputed independently:
    fresh VectorExposure overlay of the PGA raster + EDR lookup + mean costs,
    compared with the Stage-2 earthquake damage matrix. Validates the cached
    fragments and the EDR pipeline end-to-end.

Run:  python -m src.validate
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.interpolate import interp1d

from damagescanner.core import VectorScanner, VectorExposure

from .curves import (
    DEFAULT_MAXDAM,
    MAIN_ROAD_TYPES,
    ROAD_MAXDAM,
    load_eq_edr_tables,
    load_flood_curves,
)
from .paths import load_config, set_country_override
from .risk_model import (
    ModelData,
    WARMING_LEVELS,
    _cost_per_m,
    _damage_matrix,
    _integrate_ead,
    _shift_rps,
    load_model_data,
)

CURVE_MAIN = "F7.5"
CURVE_OTHER = "F7.9"
EQ_CURVE = "E7.2"
RIVER_VALIDATION_RP = 100
EQ_VALIDATION_RP = 476


def _load_features(cfg: dict) -> gpd.GeoDataFrame:
    path = cfg["exposure_dir"] / f"{cfg['country']}_{cfg['asset_type']}_exposure.parquet"
    return gpd.read_parquet(
        path, columns=["osm_id", "object_type", "geometry"]
    ).reset_index(drop=True)


def _clip_hazard(path, features: gpd.GeoDataFrame):
    b = features.to_crs(4326).total_bounds
    return xr.open_dataset(path, engine="rasterio").rio.clip_box(
        minx=b[0] - 0.1, miny=b[1] - 0.1, maxx=b[2] + 0.1, maxy=b[3] + 0.1
    )


def _river_damage_stage2(data: ModelData, rp: int) -> np.ndarray:
    hz = data.hazards["river"]
    dmg_m, _ = _damage_matrix(
        hz,
        hz.p_intensity,
        {0: data.flood_curves[CURVE_MAIN], 1: data.flood_curves[CURVE_OTHER]},
        "per_cell",
        data.n_seg,
    )
    damage = dmg_m * _cost_per_m(data, 0.0)[:, None]
    return damage[:, int(np.argmin(np.abs(hz.rps - rp)))]


# ---------------------------------------------------------------------------
# Check A: river damage per RP vs VectorScanner
# ---------------------------------------------------------------------------


def check_river_vs_vectorscanner(cfg: dict, data: ModelData) -> bool:
    print("=" * 70)
    print(f"Check A: river RP{RIVER_VALIDATION_RP} damage vs VectorScanner")
    print("=" * 70)

    features = _load_features(cfg)
    curve_df = load_flood_curves(cfg["vulnerability_path"])
    object_types = features["object_type"].unique()
    curves = pd.DataFrame(index=curve_df.index)
    for obj in object_types:
        cid = CURVE_MAIN if obj in MAIN_ROAD_TYPES else CURVE_OTHER
        curves[obj] = curve_df[cid]
    maxdam = pd.DataFrame(
        {
            "object_type": object_types,
            "damage": [ROAD_MAXDAM.get(o, DEFAULT_MAXDAM)[1] for o in object_types],
        }
    )

    hcfg = cfg["hazards"]["river"]
    hazard = _clip_hazard(
        hcfg["dir"] / hcfg["filename_template"].format(rp=RIVER_VALIDATION_RP), features
    )

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

    fast_damage = pd.Series(
        _river_damage_stage2(data, RIVER_VALIDATION_RP), index=np.arange(data.n_seg)
    )

    both = pd.DataFrame({"ref": ref_damage, "fast": fast_damage}).fillna(0.0)
    tot_ref, tot_fast = both["ref"].sum(), both["fast"].sum()
    denom = np.maximum(both["ref"].abs(), 1.0)
    rel = ((both["fast"] - both["ref"]).abs() / denom).max()
    print(f"  total damage  reference: {tot_ref / 1e6:12.3f} MEUR")
    print(f"  total damage  fast     : {tot_fast / 1e6:12.3f} MEUR")
    print(f"  total rel. difference  : {abs(tot_fast - tot_ref) / tot_ref:.2e}")
    print(f"  max per-segment rel. difference (damage > 1 EUR): {rel:.2e}")
    ok = abs(tot_fast - tot_ref) / tot_ref < 1e-3 and rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Check B: EAD integration vs scalar reference port
# ---------------------------------------------------------------------------


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
    """Scalar port of adjust_return_periods_climate for one segment."""
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
    print("=" * 70)
    print("Check B: EAD integration vs scalar reference port")
    print("=" * 70)

    rng = np.random.default_rng(42)
    hz = data.hazards["river"]
    dmg_m, _ = _damage_matrix(
        hz,
        hz.p_intensity,
        {0: data.flood_curves[CURVE_MAIN], 1: data.flood_curves[CURVE_OTHER]},
        "per_cell",
        data.n_seg,
    )
    damage = dmg_m * _cost_per_m(data, 0.0)[:, None]

    flooded = np.flatnonzero(damage.sum(axis=1) > 0)
    sample = rng.choice(flooded, size=min(n_sample, len(flooded)), replace=False)

    all_ok = True
    for warming in ("current", "2.0C", "4.0C"):
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


# ---------------------------------------------------------------------------
# Check C: earthquake damage recomputed independently
# ---------------------------------------------------------------------------


def check_earthquake_recomputation(cfg: dict, data: ModelData) -> bool:
    print("=" * 70)
    print(f"Check C: earthquake RP{EQ_VALIDATION_RP} damage, independent recompute")
    print("=" * 70)

    features = _load_features(cfg)
    hcfg = cfg["hazards"]["earthquake"]
    hazard = _clip_hazard(
        hcfg["dir"] / hcfg["filename_template"].format(rp=EQ_VALIDATION_RP), features
    )

    print("Running fresh VectorExposure overlay + EDR lookup (reference-style)...")
    exposed, _, _, _ = VectorExposure(
        hazard_file=hazard,
        feature_file=features[["object_type", "geometry"]],
        hazard_value_col="band_data",
        disable_progress=True,
        return_full=False,
    )
    pga_x, edr_y = load_eq_edr_tables(cfg["fragility_path"], [EQ_CURVE])[EQ_CURVE]

    ref = np.zeros(data.n_seg)
    for seg_idx, values, coverage in zip(
        exposed.index, exposed["values"], exposed["coverage"]
    ):
        v = np.asarray(values, dtype=np.float64)
        c = np.asarray(coverage, dtype=np.float64)
        keep = np.isfinite(v) & (v > 0) & (c > 0)
        if not keep.any():
            continue
        maxdam = ROAD_MAXDAM.get(
            features["object_type"].iloc[seg_idx], DEFAULT_MAXDAM
        )[1]
        ref[seg_idx] = np.sum(np.interp(v[keep], pga_x, edr_y) * c[keep]) * maxdam

    hz = data.hazards["earthquake"]
    dmg_m, _ = _damage_matrix(
        hz, hz.p_intensity, {0: (pga_x, edr_y)}, "per_cell", data.n_seg
    )
    fast = (dmg_m * _cost_per_m(data, 0.0)[:, None])[
        :, int(np.argmin(np.abs(hz.rps - EQ_VALIDATION_RP)))
    ]

    tot_ref, tot_fast = ref.sum(), fast.sum()
    denom = np.maximum(np.abs(ref), 1.0)
    rel = (np.abs(fast - ref) / denom).max()
    print(f"  total damage  reference: {tot_ref / 1e6:12.4f} MEUR")
    print(f"  total damage  fast     : {tot_fast / 1e6:12.4f} MEUR")
    print(f"  total rel. difference  : {abs(tot_fast - tot_ref) / max(tot_ref, 1e-9):.2e}")
    print(f"  max per-segment rel. difference: {rel:.2e}")
    ok = abs(tot_fast - tot_ref) / max(tot_ref, 1e-9) < 1e-3 and rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    args = parser.parse_args()
    set_country_override(args.country)

    cfg = load_config()
    print(f"Validating {cfg['country']} {cfg['asset_type']}")
    data = load_model_data(cfg)
    ok_a = check_river_vs_vectorscanner(cfg, data)
    ok_b = check_ead_integration(cfg, data)
    ok_c = check_earthquake_recomputation(cfg, data)
    print("=" * 70)
    print(f"Overall: {'ALL CHECKS PASSED' if ok_a and ok_b and ok_c else 'CHECKS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
