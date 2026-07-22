"""Vulnerability/fragility curves and maximum damage values, per asset type.

Curve IDs and max-damage values are transcribed from the MIRACA AssetRisk_PanEU
pipeline (src/constants.py there: DICT_CIS_VULNERABILITY_FLOOD,
DICT_CIS_VULNERABILITY_EARTHQUAKE, INFRASTRUCTURE_DAMAGE_VALUES), restricted
to object types actually observed in the MIRACA_EXPOSURE parquet files for
each asset. Curve databases: MIRACA vulnerability table (Nirandjan et al.),
sheet ``F_Vuln_Depth`` (damage fraction 0-1 vs depth in m) and the earthquake
fragility file, sheet ``E_Frag_PGA`` (P(damage state exceedance) vs PGA in g).

Windstorm is not modelled: the roads wind curve (W7.2) is identically zero in
the database, and the other three asset types here are not currently wired
into a windstorm hazard block either.

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
    default_maxdam: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    report_class: dict[str, str] | None = None      # None -> use object_type directly
    default_report_class: str = "other"
    report_classes: list[str] = field(default_factory=list)  # only used if report_class is set


def _build_asset_config(asset: str) -> AssetConfig:
    flood_groups, flood_obj_group = _derive_groups(FLOOD_CURVES_RAW[asset])
    eq_groups, eq_obj_group = _derive_groups(EQ_CURVES_RAW[asset])
    maxdam = MAXDAM_RAW[asset]
    default_maxdam = list(np.mean(list(maxdam.values()), axis=0))

    if asset == "roads":
        return AssetConfig(
            name=asset,
            flood_groups=flood_groups,
            flood_object_group=flood_obj_group,
            eq_groups=eq_groups,
            eq_object_group=eq_obj_group,
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
        maxdam=maxdam,
        default_maxdam=default_maxdam,
        report_class=None,
        default_report_class=report_classes[-1] if report_classes else "other",
        report_classes=report_classes,
    )


ASSET_CONFIGS: dict[str, AssetConfig] = {
    asset: _build_asset_config(asset) for asset in FLOOD_CURVES_RAW
}


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


def load_flood_curves(vulnerability_path: Path, curve_ids: list[str]) -> pd.DataFrame:
    """Load depth-damage curves for the given curve IDs from the MIRACA table.

    Returns a DataFrame indexed by depth (m, 0.00-6.00 in 0.05 steps) with one
    column per curve ID, values = damage fraction in [0, 1].

    Row slice iloc[4:125] mirrors prepare_flood_curves() in AssetRisk_PanEU:
    rows 0-3 are header/metadata, data rows are depths 0 to 6 m.
    """
    vul_df = pd.read_excel(vulnerability_path, sheet_name="F_Vuln_Depth").ffill()
    curves = (
        vul_df[["ID number"] + curve_ids]
        .iloc[4:125]
        .set_index("ID number")
        .rename_axis("depth")
        .astype(np.float64)
        .ffill()
    )
    curves.index = curves.index.astype(np.float64)
    if not curves.index.is_monotonic_increasing:
        raise ValueError("Curve depth index is not monotonically increasing")
    return curves


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
