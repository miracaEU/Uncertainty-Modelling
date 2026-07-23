"""Vulnerability/fragility curves and maximum damage values, per asset type.

Curve IDs and max-damage values are transcribed from the MIRACA AssetRisk_PanEU
pipeline (src/constants.py there: DICT_CIS_VULNERABILITY_FLOOD /
_EARTHQUAKE / _WIND, INFRASTRUCTURE_DAMAGE_VALUES), restricted to the object
types actually observed in the MIRACA_EXPOSURE parquet files for each asset.
Ten asset types are covered: roads, airports, education, power, rail, telecom,
healthcare, gas, oil, ports. Curve databases: MIRACA vulnerability table
(Nirandjan et al.), sheets ``F_Vuln_Depth`` (flood, damage fraction 0-1 vs
depth in m) and ``W_Vuln_V10m_3sec`` (windstorm, vs 3-sec gust in m/s), and the
earthquake fragility file, sheet ``E_Frag_PGA`` (P(damage state exceedance)
vs PGA in g). Coastal flood reuses the flood curves (see risk_model.py).

Windstorm applies only to assets with a non-degenerate wind curve set
(everything except roads and ports - their sole wind curve, W7.2, is
identically zero); see supports_windstorm / applicable_hazards below.

Design standard: within one asset type, several object types often share the
exact same curve list (e.g. every road class below "primary" uses F7.8/F7.9).
_derive_groups() collapses those into named groups automatically, named after
the group's first curve ID — this is what becomes one categorical uncertainty
factor (e.g. "curve_F7_4" for roads' high-class group). Groups with only one
possible curve carry no uncertainty and are held fixed rather than sampled
(see ema_model.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Raw per-object-type tables, transcribed from AssetRisk_PanEU src/constants.py
# (DICT_CIS_VULNERABILITY_FLOOD / _EARTHQUAKE / INFRASTRUCTURE_DAMAGE_VALUES),
# restricted to the object types present in MIRACA_EXPOSURE for each asset.
# maxdam is [min, mean, max]; units are EUR/m for line features, EUR/m^2 for
# polygon features, EUR/unit for point features (matches each object type's
# actual OSM geometry, verified against the exposure parquet files).
# ---------------------------------------------------------------------------

_ROAD_TYPES_HIGH = [
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
]
_ROAD_TYPES_LOW = [
    "secondary", "secondary_link", "tertiary", "tertiary_link",
    "residential", "road", "unclassified", "track", "service",
]

FLOOD_CURVES_RAW = {
    "roads": {
        **{t: ["F7.4", "F7.5", "F7.6", "F7.7"] for t in _ROAD_TYPES_HIGH},
        **{t: ["F7.8", "F7.9"] for t in _ROAD_TYPES_LOW},
    },
    "airports": {
        "aerodrome": ["F9.1", "F9.2", "F9.3"],
        "apron": ["F9.1", "F9.2", "F9.3"],
        "terminal": ["F9.1"],
        "runway": ["F7.4", "F7.5", "F7.6", "F7.7"],
    },
    "education": {
        t: ["F21.6", "F21.7", "F21.8", "F21.10", "F21.11", "F21.13"]
        for t in ("school", "kindergarten", "college", "university", "library")
    },
    "power": {
        "line": ["F6.1", "F6.2"],
        "cable": ["F5.1"],
        "minor_line": ["F6.1", "F6.2"],
        "plant": ["F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7"],
        "generator": ["F2.1", "F2.2", "F2.3"],
        "substation": ["F2.1", "F2.2", "F2.3"],
        "transformer": ["F2.1", "F2.2", "F2.3"],
        "pole": ["F6.1", "F6.2"],
        "portal": ["F2.1", "F2.2", "F2.3"],
        "tower": ["F6.1", "F6.2"],
        "terminal": ["F9.1"],
        "switch": ["F2.1", "F2.2", "F2.3"],
        "catenary_mast": ["F10.1"],
    },
    "rail": {
        "rail": ["F8.1", "F8.2", "F8.3", "F8.4", "F8.5", "F8.6", "F8.7"],
        "narrow_gauge": ["F8.1", "F8.2", "F8.3", "F8.4", "F8.5", "F8.6", "F8.7"],
    },
    "telecom": {
        "mast": ["F10.1"],
        "tower": ["F6.1", "F6.2"],
        "communications_tower": ["F6.1", "F6.2"],
    },
    "healthcare": {
        t: ["F21.6", "F21.8", "F21.9", "F21.12"] for t in ("hospital", "clinic")
    },
    "gas": {
        "pipeline": ["F16.1", "F16.2", "F16.3"],
        "storage_tank": ["F2.1", "F2.2", "F2.3"],
        "gasometer": ["F13.1", "F13.2", "F13.3", "F13.5"],
    },
    "oil": {
        "pipeline": ["F16.1", "F16.2", "F16.3"],
        "storage_tank": ["F2.1", "F2.2", "F2.3"],
    },
    "ports": {
        "port": ["F9.1"],
        "harbour": ["F9.1"],
    },
}

_EQ_ROAD_CURVES = ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"]
_EQ_EDU_CURVES = [
    "E21.26-C", "E21.27-C", "E21.29-C", "E21.30-C", "E21.31-C", "E21.32-C",
    "E21.33-C", "E21.34-C", "E21.35-C", "E21.36-C", "E21.37-C", "E21.38-C",
    "E21.39-C", "E21.40-C", "E21.41-C", "E21.42-C", "E21.43-C", "E21.48-C",
    "E21.49-C", "E21.50-C", "E21.51-C", "E21.52-C", "E21.53-C", "E21.54-C",
    "E21.55-C", "E21.56-C", "E21.57-C", "E21.58-C", "E21.59-C", "E21.60-C",
    "E21.61-C",
]
_EQ_SUB_CURVES = ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"]
_EQ_GEN_CURVES = ["E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6", "E1.7", "E1.8"]
_EQ_LINE_CURVES = ["E6.1", "E6.2", "E6.3", "E6.4"]
_EQ_POLE_CURVES = ["E4.1", "E4.2", "E4.3", "E4.4"]
_EQ_RAIL_CURVES = [
    "E8.1", "E8.2", "E8.3", "E8.4", "E8.5", "E8.6", "E8.7", "E8.8", "E8.9", "E8.10",
    "E8.11", "E8.12", "E8.13", "E8.14", "E8.15", "E8.16", "E8.17", "E8.18", "E8.19", "E8.20",
]
_EQ_HEALTH_CURVES = [
    "E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C",
]

EQ_CURVES_RAW = {
    "roads": {t: _EQ_ROAD_CURVES for t in _ROAD_TYPES_HIGH + _ROAD_TYPES_LOW},
    "airports": {
        "aerodrome": ["E9.2", "E9.3", "E9.4"],
        "apron": ["E9.2", "E9.3", "E9.4"],
        "terminal": ["E9.2", "E9.3", "E9.4"],
        "runway": _EQ_ROAD_CURVES,
    },
    "education": {
        t: _EQ_EDU_CURVES
        for t in ("school", "kindergarten", "college", "university", "library")
    },
    "power": {
        "line": _EQ_LINE_CURVES,
        "cable": _EQ_LINE_CURVES,
        "minor_line": _EQ_LINE_CURVES,
        "plant": _EQ_GEN_CURVES,
        "generator": _EQ_GEN_CURVES,
        "substation": _EQ_SUB_CURVES,
        "transformer": _EQ_SUB_CURVES,
        "pole": _EQ_POLE_CURVES,
        "portal": _EQ_SUB_CURVES,
        "tower": ["E3.1", "E3.2"],
        "terminal": _EQ_SUB_CURVES,
        "switch": _EQ_SUB_CURVES,
        "catenary_mast": _EQ_POLE_CURVES,
    },
    "rail": {
        "rail": _EQ_RAIL_CURVES,
        "narrow_gauge": _EQ_RAIL_CURVES,
    },
    "telecom": {
        "mast": ["E11.1"],
        "tower": ["E3.1", "E3.2"],
        "communications_tower": ["E3.1", "E3.2"],
    },
    "healthcare": {
        t: _EQ_HEALTH_CURVES for t in ("hospital", "clinic")
    },
    "gas": {
        "pipeline": _EQ_LINE_CURVES,
        "storage_tank": _EQ_GEN_CURVES,
        "gasometer": _EQ_GEN_CURVES,
    },
    "oil": {
        "pipeline": _EQ_LINE_CURVES,
        "storage_tank": _EQ_GEN_CURVES,
    },
    "ports": {
        "port": ["E9.2", "E9.3", "E9.4"],
        "harbour": ["E9.2", "E9.3", "E9.4"],
    },
}

# ---------------------------------------------------------------------------
# Windstorm vulnerability curves (transcribed from AssetRisk_PanEU
# DICT_CIS_VULNERABILITY_WIND, sheet ``W_Vuln_V10m_3sec`` - damage fraction
# 0-1 vs 3-second gust speed at 10 m, 0-120 m/s). Roads are intentionally
# NOT included: the only roads wind curve (W7.2) is identically zero in the
# database, so roads carry no windstorm damage and windstorm is skipped for
# them entirely (see supports_windstorm / applicable_hazards below).
#
# W7.2 is likewise identically zero (verified against the table) and is used
# here only as the "wind does not damage this" placeholder for enclosed /
# below-grade object types (power plants, substations, transformers, ...,
# and airport aerodrome/apron/runway surfaces): a single-curve, all-zero
# group is held FIXED and contributes no damage, keeping the "every object
# type has a group" invariant the preprocessing relies on.
# ---------------------------------------------------------------------------

WIND_ZERO_CURVE = "W7.2"  # identically zero in the MIRACA table (verified)

_WIND_TOWER_CURVES = [
    "W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14",
]
_WIND_POLE_CURVES = ["W4.33", "W4.34", "W4.35", "W4.36", "W4.37"]
_WIND_LINE_CURVES = ["W6.1", "W6.2", "W6.3"]
_WIND_BUILDING_CURVES = ["W21.11", "W21.12", "W21.13", "W21.14"]
_WIND_COMMS_TOWER_CURVES = ["W10.3", "W10.4", "W10.5", "W10.6", "W10.7", "W10.8", "W10.9"]
# Rail has no dedicated wind curve; the reference uses power-tower damage curves
# (150/120/180 km/h design speeds) as a per-track catenary-mast proxy.
_WIND_RAIL_CURVES = ["W3.9", "W3.6", "W3.12"]

WIND_CURVES_RAW = {
    "airports": {
        "aerodrome": [WIND_ZERO_CURVE],
        "apron": [WIND_ZERO_CURVE],
        "runway": [WIND_ZERO_CURVE],
        "terminal": ["W21.13", "W21.14"],
    },
    "education": {
        t: _WIND_BUILDING_CURVES
        for t in ("school", "kindergarten", "college", "university", "library")
    },
    "power": {
        "line": _WIND_LINE_CURVES,
        "minor_line": _WIND_LINE_CURVES,
        "cable": [WIND_ZERO_CURVE],
        "pole": _WIND_POLE_CURVES,
        "catenary_mast": _WIND_POLE_CURVES,
        "tower": _WIND_TOWER_CURVES,
        # wind-immune (enclosed / below-grade) -> zero placeholder curve
        "plant": [WIND_ZERO_CURVE],
        "generator": [WIND_ZERO_CURVE],
        "substation": [WIND_ZERO_CURVE],
        "transformer": [WIND_ZERO_CURVE],
        "portal": [WIND_ZERO_CURVE],
        "terminal": [WIND_ZERO_CURVE],
        "switch": [WIND_ZERO_CURVE],
    },
    "rail": {
        "rail": _WIND_RAIL_CURVES,
        "narrow_gauge": _WIND_RAIL_CURVES,
    },
    "telecom": {
        "mast": _WIND_TOWER_CURVES,
        "tower": _WIND_TOWER_CURVES,
        "communications_tower": _WIND_COMMS_TOWER_CURVES,
    },
    "healthcare": {
        t: _WIND_BUILDING_CURVES for t in ("hospital", "clinic")
    },
    "gas": {
        "gasometer": _WIND_BUILDING_CURVES,
        "storage_tank": _WIND_BUILDING_CURVES,
        "pipeline": [WIND_ZERO_CURVE],  # buried -> no wind damage
    },
    "oil": {
        "storage_tank": _WIND_BUILDING_CURVES,
        "pipeline": [WIND_ZERO_CURVE],  # buried -> no wind damage
    },
    # ports intentionally omitted: its only wind curve is the zero W7.2
    # (port/harbour), so ports carries no windstorm damage (like roads).
}

MAXDAM_RAW = {
    "roads": {
        "motorway": [1106, 2895, 3931], "motorway_link": [1106, 2895, 3931],
        "trunk": [848, 1242, 1636], "trunk_link": [848, 1242, 1636],
        "primary": [917, 1137, 1357], "primary_link": [917, 1137, 1357],
        "secondary": [257, 452, 678], "secondary_link": [257, 452, 678],
        "tertiary": [203, 271, 339], "tertiary_link": [203, 271, 339],
        "residential": [66, 136, 305], "road": [66, 136, 305],
        "unclassified": [66, 136, 305], "track": [66, 136, 305], "service": [66, 136, 305],
    },
    "airports": {
        "aerodrome": [113, 135, 165],   # EUR/m2 (polygon)
        "apron": [113, 135, 165],       # EUR/m2 (polygon)
        "terminal": [113, 165, 4271],   # EUR/m2 (polygon)
        "runway": [4133, 5511, 9078],   # EUR/m (line, mapped as centreline in OSM)
    },
    "education": {  # EUR/m2 (polygon, building footprint)
        "school": [267, 713, 1294], "kindergarten": [267, 713, 1294],
        "college": [267, 713, 1294], "university": [267, 713, 1294], "library": [267, 713, 1294],
    },
    "power": {
        "line": [108, 183, 1151],           # EUR/m (line)
        "cable": [215, 1818, 5497],          # EUR/m (line)
        "minor_line": [71, 102, 103],        # EUR/m (line)
        "plant": [649, 1558, 11110],         # EUR/m2 (polygon)
        "generator": [1299, 1904, 6349],     # EUR/m2 (polygon)
        "substation": [1299, 1904, 6349],    # EUR/m2 (polygon)
        "transformer": [1299, 1904, 6349],   # EUR/unit (point)
        "pole": [73005, 97627, 369547],      # EUR/unit (point)
        "portal": [1299, 1904, 6349],        # EUR/unit (point)
        "tower": [6171, 103928, 275472],     # EUR/unit (point)
        "terminal": [1299, 1904, 6349],      # EUR/unit (point)
        "switch": [1299, 1904, 6349],        # EUR/unit (point)
        "catenary_mast": [67506, 76630, 111998],  # EUR/unit (point)
    },
    "rail": {  # EUR/m (line)
        "rail": [491, 2858, 14186], "narrow_gauge": [491, 2858, 14186],
    },
    "telecom": {  # EUR/unit (point)
        "mast": [67506, 76630, 111998],
        "tower": [139610, 152468, 229376],
        "communications_tower": [139610, 152468, 229376],
    },
    "healthcare": {  # EUR/m2 (polygon, building footprint)
        "hospital": [591, 1294, 2227], "clinic": [591, 1294, 2227],
    },
    "gas": {
        "pipeline": [71, 102, 103],          # EUR/m (line)
        "storage_tank": [157, 4181, 7840],   # EUR/m2 (polygon)
        "gasometer": [558, 14885, 27910],    # EUR/m2 (polygon)
    },
    "oil": {
        "pipeline": [71, 102, 103],          # EUR/m (line)
        "storage_tank": [157, 4181, 7840],   # EUR/m2 (polygon)
    },
    "ports": {  # EUR/m2 (polygon)
        "port": [113, 135, 165], "harbour": [113, 135, 165],
    },
}

# Roads-specific reporting rollup (kept for continuity with earlier analyses);
# other assets report per-class outcomes by raw object_type instead.
ROAD_REPORT_CLASS = {
    "motorway": "motorway_trunk", "motorway_link": "motorway_trunk",
    "trunk": "motorway_trunk", "trunk_link": "motorway_trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
}
ROAD_DEFAULT_REPORT_CLASS = "other"
ROAD_REPORT_CLASSES = ["motorway_trunk", "primary", "secondary", "tertiary", "other"]

# Backward-compatible aliases used elsewhere in the codebase (roads only).
MAIN_ROAD_TYPES = set(_ROAD_TYPES_HIGH)
REPORT_CLASS = ROAD_REPORT_CLASS
DEFAULT_REPORT_CLASS = ROAD_DEFAULT_REPORT_CLASS
REPORT_CLASSES = ROAD_REPORT_CLASSES

EQ_DAMAGE_RATIOS = {
    "minor": 0.05, "moderate": 0.20, "extensive": 0.70,
    "severe": 0.85, "complete": 1.00, "collapse": 1.00,
}
EQ_DAMAGE_STATE_MAP = {
    "Slight": "minor", "Minor": "minor", "DS1": "minor",
    "Moderate": "moderate", "DS2": "moderate",
    "Extensive": "extensive", "DS3": "extensive",
    "Severe": "severe", "DS4": "severe",
    "Complete": "complete", "DS5": "complete",
    "Collapse": "collapse",
}
EQ_PGA_RANGE = np.arange(0.0, 3.35, 0.05)  # for parametric (median/beta) curves


# ---------------------------------------------------------------------------
# Automatic curve-group derivation and the AssetConfig registry
# ---------------------------------------------------------------------------


def _derive_groups(curve_dict: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Collapse per-object-type curve lists into named groups.

    Object types that use the exact same curve list become one group, named
    after that list's (sorted) first curve ID with '.'/'-' replaced by '_'
    (e.g. curves ["F7.5","F7.4","F7.6","F7.7"] -> group name "F7_4"). This
    keeps names deterministic and traceable back to the source database
    without hand-maintaining a name per asset.

    Two *different* curve lists can share the same sorted-first ID (e.g.
    airports' aerodrome/apron = [F9.1,F9.2,F9.3] vs terminal = [F9.1] alone)
    — the naming loop below disambiguates with a "_v2", "_v3", ... suffix so
    neither group silently overwrites the other.

    Returns (group_name -> sorted curve_id list, object_type -> group_name).
    """
    seen: dict[tuple[str, ...], str] = {}
    groups: dict[str, list[str]] = {}
    obj_to_group: dict[str, str] = {}
    for obj, curves in curve_dict.items():
        key = tuple(sorted(curves))
        if key not in seen:
            base = key[0].replace(".", "_").replace("-", "_")
            name, suffix = base, 2
            while name in groups:
                name = f"{base}_v{suffix}"
                suffix += 1
            seen[key] = name
            groups[name] = list(key)
        obj_to_group[obj] = seen[key]
    return groups, obj_to_group


@dataclass
class AssetConfig:
    name: str
    flood_groups: dict[str, list[str]]       # group_name -> curve ids (>=1)
    flood_object_group: dict[str, str]       # object_type -> group_name
    eq_groups: dict[str, list[str]]
    eq_object_group: dict[str, str]
    maxdam: dict[str, list[float]]           # object_type -> [min, mean, max]
    wind_groups: dict[str, list[str]] = field(default_factory=dict)     # empty -> no windstorm
    wind_object_group: dict[str, str] = field(default_factory=dict)
    default_maxdam: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    report_class: dict[str, str] | None = None      # None -> use object_type directly
    default_report_class: str = "other"
    report_classes: list[str] = field(default_factory=list)  # only used if report_class is set

    @property
    def supports_windstorm(self) -> bool:
        """True iff at least one wind curve group is non-degenerate.

        A group made up solely of the identically-zero placeholder curve
        (W7.2) carries no damage; an asset whose wind groups are all like
        that (or that has no wind config at all, e.g. roads) has nothing to
        vary for windstorm and is skipped.
        """
        return any(
            any(c != WIND_ZERO_CURVE for c in curves)
            for curves in self.wind_groups.values()
        )


def _build_asset_config(asset: str) -> AssetConfig:
    flood_groups, flood_obj_group = _derive_groups(FLOOD_CURVES_RAW[asset])
    eq_groups, eq_obj_group = _derive_groups(EQ_CURVES_RAW[asset])
    wind_groups, wind_obj_group = (
        _derive_groups(WIND_CURVES_RAW[asset]) if asset in WIND_CURVES_RAW else ({}, {})
    )
    maxdam = MAXDAM_RAW[asset]
    default_maxdam = list(np.mean(list(maxdam.values()), axis=0))

    if asset == "roads":
        return AssetConfig(
            name=asset,
            flood_groups=flood_groups,
            flood_object_group=flood_obj_group,
            eq_groups=eq_groups,
            eq_object_group=eq_obj_group,
            wind_groups=wind_groups,
            wind_object_group=wind_obj_group,
            maxdam=maxdam,
            default_maxdam=default_maxdam,
            report_class=ROAD_REPORT_CLASS,
            default_report_class=ROAD_DEFAULT_REPORT_CLASS,
            report_classes=ROAD_REPORT_CLASSES,
        )

    report_classes = sorted(maxdam.keys())
    return AssetConfig(
        name=asset,
        flood_groups=flood_groups,
        flood_object_group=flood_obj_group,
        eq_groups=eq_groups,
        eq_object_group=eq_obj_group,
        wind_groups=wind_groups,
        wind_object_group=wind_obj_group,
        maxdam=maxdam,
        default_maxdam=default_maxdam,
        report_class=None,
        default_report_class=report_classes[-1] if report_classes else "other",
        report_classes=report_classes,
    )


ASSET_CONFIGS: dict[str, AssetConfig] = {
    asset: _build_asset_config(asset) for asset in FLOOD_CURVES_RAW
}

# Hazards available in this pipeline, and which apply to which asset/country.
# River flood and earthquake apply to every asset; windstorm only to assets
# with a non-degenerate wind curve set (airports/education/power - not roads);
# coastal flood only to coastal (non-landlocked) countries, for any asset. This
# is the single source of truth the preprocessing, the scenario applicability
# in ema_model.py, and the study orchestrator all consult.
ALL_HAZARDS = ("river", "earthquake", "windstorm", "coastal")

# European ISO3 countries with no coastline - coastal flood is skipped for
# them. (Only LUX is in the current study set; the rest are listed so the
# check is correct if more countries are added later.)
LANDLOCKED = frozenset({
    "LUX", "AUT", "CHE", "CZE", "HUN", "SVK", "LIE", "AND", "SMR",
    "MKD", "SRB", "BLR", "MDA", "VAT", "XKX", "RKS",
})


def country_has_coast(country: str | None) -> bool:
    return bool(country) and country.upper() not in LANDLOCKED


def applicable_hazards(asset: str, country: str | None = None) -> list[str]:
    """Hazards that apply to this asset (and country, for coastal).

    country is optional: when omitted, coastal is left out (it is the only
    country-dependent hazard, so callers that know the country - preprocess,
    the orchestrator - pass it to include coastal for coastal countries).
    """
    cfg = get_asset_config(asset)
    hazards = ["river", "earthquake"]
    if cfg.supports_windstorm:
        hazards.append("windstorm")
    if country_has_coast(country):
        hazards.append("coastal")
    return hazards


def get_asset_config(asset: str) -> AssetConfig:
    if asset not in ASSET_CONFIGS:
        raise KeyError(
            f"No curve/cost configuration for asset '{asset}'. "
            f"Available: {sorted(ASSET_CONFIGS)}"
        )
    return ASSET_CONFIGS[asset]


def maxdam_arrays(cfg: AssetConfig, object_types: pd.Series) -> np.ndarray:
    """Per-segment [min, mean, max] max damage as an (n, 3) array."""
    out = np.empty((len(object_types), 3), dtype=np.float64)
    for i, obj in enumerate(object_types):
        out[i, :] = cfg.maxdam.get(obj, cfg.default_maxdam)
    return out


def report_class_for(cfg: AssetConfig, object_types: pd.Series) -> pd.Series:
    if cfg.report_class is None:
        return object_types.astype(str)
    return object_types.map(cfg.report_class).fillna(cfg.default_report_class)


# ---------------------------------------------------------------------------
# Flood depth-damage curves
# ---------------------------------------------------------------------------


def load_vulnerability_curves(
    vulnerability_path: Path, curve_ids: list[str], sheet: str, axis_name: str = "intensity"
) -> pd.DataFrame:
    """Load intensity-damage curves for the given IDs from a MIRACA table sheet.

    Returns a DataFrame indexed by the hazard intensity axis with one column
    per curve ID, values = damage fraction in [0, 1]. Works for any sheet
    that follows the shared layout (an "ID number" intensity column plus one
    column per curve, data in rows iloc[4:125]) - i.e. both the flood
    depth-damage sheet (F_Vuln_Depth, depth 0-6 m) and the windstorm
    speed-damage sheet (W_Vuln_V10m_3sec, 3-sec gust 0-120 m/s). Row slice
    mirrors prepare_flood_curves() / prepare_wind_curves() in AssetRisk_PanEU.
    """
    vul_df = pd.read_excel(vulnerability_path, sheet_name=sheet).ffill()
    curves = (
        vul_df[["ID number"] + curve_ids]
        .iloc[4:125]
        .set_index("ID number")
        .rename_axis(axis_name)
        .astype(np.float64)
        .ffill()
    )
    curves.index = curves.index.astype(np.float64)
    if not curves.index.is_monotonic_increasing:
        raise ValueError(f"Curve intensity index ({sheet}) is not monotonically increasing")
    return curves


def load_flood_curves(vulnerability_path: Path, curve_ids: list[str]) -> pd.DataFrame:
    """Depth-damage curves (F_Vuln_Depth sheet), indexed by depth in metres."""
    return load_vulnerability_curves(
        vulnerability_path, curve_ids, sheet="F_Vuln_Depth", axis_name="depth"
    )


def load_wind_curves(vulnerability_path: Path, curve_ids: list[str]) -> pd.DataFrame:
    """Speed-damage curves (W_Vuln_V10m_3sec sheet), indexed by 3-sec gust m/s."""
    return load_vulnerability_curves(
        vulnerability_path, curve_ids, sheet="W_Vuln_V10m_3sec", axis_name="speed"
    )


# ---------------------------------------------------------------------------
# Earthquake fragility -> expected damage ratio (EDR) lookup tables
# ---------------------------------------------------------------------------


def _edr_from_exceedance(
    pga_index: np.ndarray, exceed_by_state: dict[str, np.ndarray]
) -> np.ndarray:
    """Collapse exceedance probabilities per damage state to an EDR curve.

    Port of AssetRisk_PanEU hazard_earthquake.py::_build_edr_lookup:
    states are ordered by loss ratio, converted to individual state
    probabilities (P(state_i) = P(exceed_i) - P(exceed_{i+1})), row-normalised
    with P(no damage) = 1 - P(exceed first), then dotted with the loss ratios.
    """
    states = sorted(exceed_by_state, key=lambda s: EQ_DAMAGE_RATIOS.get(s, 0.5))
    n_pga, n_states = len(pga_index), len(states)
    exceed = np.column_stack([exceed_by_state[s] for s in states])

    individual = np.zeros((n_pga, n_states + 1))
    individual[:, 0] = np.maximum(0.0, 1.0 - exceed[:, 0])
    for j in range(n_states - 1):
        individual[:, j + 1] = np.maximum(0.0, exceed[:, j] - exceed[:, j + 1])
    individual[:, n_states] = exceed[:, -1]

    row_sums = individual.sum(axis=1, keepdims=True)
    individual /= np.where(row_sums > 0, row_sums, 1.0)

    weights = np.array([0.0] + [EQ_DAMAGE_RATIOS.get(s, 0.0) for s in states])
    return np.clip(individual @ weights, 0.0, 1.0)


def load_eq_edr_tables(
    fragility_path: Path, curve_ids: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load earthquake fragility curves and collapse each to an EDR lookup.

    Handles both pre-computed curves (PGA-indexed exceedance probabilities)
    and parametric ones (median/beta -> lognormal CDF over EQ_PGA_RANGE),
    mirroring prepare_earthquake_fragility() in the reference.

    Returns {curve_id: (pga_array_g, edr_array)} for use with np.interp.
    """
    frag = pd.read_excel(fragility_path, sheet_name="E_Frag_PGA", header=[0, 1])
    pga_numeric = pd.to_numeric(frag.iloc[:, 0], errors="coerce")
    valid = ~pga_numeric.isna()

    tables = {}
    for cid in curve_ids:
        cols = [c for c in frag.columns if c[0] == cid]
        if not cols:
            raise KeyError(f"Fragility curve {cid} not found in {fragility_path}")

        cell_strings = [
            str(v).lower() for c in cols for v in frag[c].dropna().tolist()
        ]
        parametric = any("median" in s for s in cell_strings) and any(
            "beta" in s for s in cell_strings
        )

        if not parametric:
            pga_index = pga_numeric[valid].to_numpy(np.float64)
            exceed = {}
            for c in cols:
                state = EQ_DAMAGE_STATE_MAP.get(c[1], str(c[1]).lower())
                vals = pd.to_numeric(frag[c], errors="coerce")[valid]
                exceed[state] = vals.ffill().fillna(0).to_numpy(np.float64)
        else:
            pga_index = EQ_PGA_RANGE
            exceed = {}
            for c in cols:
                state = EQ_DAMAGE_STATE_MAP.get(c[1], str(c[1]).lower())
                median_val = beta_val = None
                column = frag[c]
                for i in range(1, len(column)):
                    prev = str(column.iloc[i - 1]).lower()
                    cell = column.iloc[i]
                    if pd.isna(cell):
                        continue
                    if "median" in prev and median_val is None:
                        median_val = float(str(cell).replace(",", "."))
                    elif "beta" in prev and beta_val is None:
                        beta_val = float(str(cell).replace(",", "."))
                if median_val is None or beta_val is None or median_val <= 0:
                    continue
                ln_pga = np.log(np.maximum(EQ_PGA_RANGE, 1e-6))
                exceed[state] = np.clip(
                    norm.cdf((ln_pga - np.log(median_val)) / beta_val), 0, 1
                )

        tables[cid] = (pga_index, _edr_from_exceedance(pga_index, exceed))

    return tables
