"""Vulnerability/fragility curves and maximum damage values for roads.

Curve IDs and max-damage values mirror the MIRACA AssetRisk_PanEU pipeline
(src/constants.py there), so results stay comparable with the pan-EU run:

  - river flood, high-class roads (motorway/trunk/primary): curves F7.4-F7.7
  - river flood, low-class roads (secondary and below):     curves F7.8-F7.9
  - earthquake, all road classes: fragility curves E7.2-E7.10, collapsed to
    an expected-damage-ratio (EDR) vs PGA lookup exactly as in the reference
    (hazard_earthquake.py::_build_edr_lookup)
  - max damage per object type as [min, mean, max] EUR per metre
  - windstorm deliberately absent: the roads wind curve W7.2 is identically
    zero in the database (wind does not damage pavements)

Curve databases: MIRACA vulnerability table (Nirandjan et al.), sheet
``F_Vuln_Depth`` (damage fraction 0-1 vs depth in m) and the earthquake
fragility file, sheet ``E_Frag_PGA`` (P(damage state exceedance) vs PGA in g).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Flood curve sets per road class group (group 0 = main, group 1 = other)
CURVES_MAIN = ["F7.4", "F7.5", "F7.6", "F7.7"]
CURVES_OTHER = ["F7.8", "F7.9"]

# Earthquake fragility curves for roads (same set for every road class)
EQ_CURVES = ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"]

# Damage state -> mean loss ratio, and name standardisation (from reference)
EQ_DAMAGE_RATIOS = {
    "minor": 0.05,
    "moderate": 0.20,
    "extensive": 0.70,
    "severe": 0.85,
    "complete": 1.00,
    "collapse": 1.00,
}
EQ_DAMAGE_STATE_MAP = {
    "Slight": "minor",
    "Minor": "minor",
    "DS1": "minor",
    "Moderate": "moderate",
    "DS2": "moderate",
    "Extensive": "extensive",
    "DS3": "extensive",
    "Severe": "severe",
    "DS4": "severe",
    "Complete": "complete",
    "DS5": "complete",
    "Collapse": "collapse",
}
EQ_PGA_RANGE = np.arange(0.0, 3.35, 0.05)  # for parametric (median/beta) curves

MAIN_ROAD_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "Motorways and Trunks",
    "Primary Roads",
}

# {object_type: [min, mean, max]} in EUR per metre — from AssetRisk_PanEU
ROAD_MAXDAM = {
    "motorway": [1106, 2895, 3931],
    "motorway_link": [1106, 2895, 3931],
    "trunk": [848, 1242, 1636],
    "trunk_link": [848, 1242, 1636],
    "primary": [917, 1137, 1357],
    "primary_link": [917, 1137, 1357],
    "secondary": [257, 452, 678],
    "secondary_link": [257, 452, 678],
    "tertiary": [203, 271, 339],
    "tertiary_link": [203, 271, 339],
    "residential": [66, 136, 305],
    "road": [66, 136, 305],
    "unclassified": [66, 136, 305],
    "track": [66, 136, 305],
    "service": [66, 136, 305],
    "Motorways and Trunks": [1106, 2895, 3931],
    "Primary Roads": [917, 1137, 1357],
    "Secondary roads": [257, 452, 678],
    "Tertiary roads": [203, 271, 339],
    "Other roads": [66, 136, 305],
}
DEFAULT_MAXDAM = [66, 136, 305]  # fallback for unmapped object types

# Reporting classes for outcome breakdown
REPORT_CLASS = {
    "motorway": "motorway_trunk",
    "motorway_link": "motorway_trunk",
    "trunk": "motorway_trunk",
    "trunk_link": "motorway_trunk",
    "Motorways and Trunks": "motorway_trunk",
    "primary": "primary",
    "primary_link": "primary",
    "Primary Roads": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "Secondary roads": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "Tertiary roads": "tertiary",
}
DEFAULT_REPORT_CLASS = "other"
REPORT_CLASSES = ["motorway_trunk", "primary", "secondary", "tertiary", "other"]


def load_flood_curves(vulnerability_path: Path) -> pd.DataFrame:
    """Load road depth-damage curves from the MIRACA vulnerability Excel table.

    Returns a DataFrame indexed by depth (m, 0.00-6.00 in 0.05 steps) with one
    column per curve ID (F7.4 ... F7.9), values = damage fraction in [0, 1].

    Row slice iloc[4:125] mirrors prepare_flood_curves() in AssetRisk_PanEU:
    rows 0-3 are header/metadata, data rows are depths 0 to 6 m.
    """
    vul_df = pd.read_excel(vulnerability_path, sheet_name="F_Vuln_Depth").ffill()
    curve_ids = CURVES_MAIN + CURVES_OTHER
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


def maxdam_arrays(object_types: pd.Series) -> np.ndarray:
    """Per-segment [min, mean, max] max damage (EUR/m) as an (n, 3) array."""
    out = np.empty((len(object_types), 3), dtype=np.float64)
    for i, obj in enumerate(object_types):
        out[i, :] = ROAD_MAXDAM.get(obj, DEFAULT_MAXDAM)
    return out


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
    fragility_path: Path, curve_ids: list[str] = EQ_CURVES
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load earthquake fragility curves and collapse each to an EDR lookup.

    Handles both pre-computed curves (PGA-indexed exceedance probabilities,
    the format of E7.2-E7.10) and parametric ones (median/beta -> lognormal
    CDF over EQ_PGA_RANGE), mirroring prepare_earthquake_fragility() in the
    reference.

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
