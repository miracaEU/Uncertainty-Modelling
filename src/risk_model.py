"""Stage 2: fast, fully vectorized multi-hazard risk model, any asset type.

Consumes the intermediate files written by src.preprocess and evaluates one
parameterization (one EMA Workbench experiment) in well under a second,
without touching any raster or geometry. Hazards: river flood, earthquake,
windstorm and coastal flood - each computed independently (the study runs one
hazard at a time; see src/ema_model.py). Windstorm applies to
airports/education/power only (the roads wind curve is identically zero);
coastal flood applies to coastal countries only, reusing the river flood
curves with its own COASTPROS protection standard (see src/curves.py,
src/coastal.py).

Damage formula (identical to damagescanner.vector._get_damage_per_object):

    damage(feature, RP) = sum_over_cells( f(intensity_cell) * quantity_cell )
                          * maxdam_per_unit(object_type)

with f = depth-damage curve (river) or fragility-derived expected damage
ratio vs PGA (earthquake), and "quantity" already unit-matched to maxdam by
Stage 1 (metres/m^2/count for line/polygon/point features respectively).
EAD integration, protection-standard cutoff and climate RP shifting mirror
AssetRisk_PanEU/src/risk_integration.py:

    EAD = trapezoid of damage over exceedance probability p = 1/RP,
          integrated between p(RP_max) and p(min(RP_min, protection RP)),
          damages below the protection RP set to zero (with the damage at
          the protection RP itself linearly interpolated in RP space);
    warming (river only) shifts every RP via per-basin piecewise-linear
    anchor maps (RP10/100/500 -> new RP) before integration;
    earthquake has no protection standard (as in the reference).

Curve groups (see src/curves.py::_derive_groups) are asset-specific and
identified by name (e.g. roads: "F7_4"/"F7_8" for flood, "E7_10" for
earthquake). compute_risk() takes curve choices as a {group_name: curve_id}
dict rather than fixed keyword arguments, so this module needs no per-asset
branching - src/ema_model.py is what turns each asset's group set into
concrete EMA Workbench Parameters/Constants.

Uncertainty factors (declared per-scenario in src/ema_model.py):
    warming           categorical: current / 1.5C / 2.0C / 3.0C / 4.0C (river)
    curve_<group>     one categorical per curve group with >1 curve option
    cost_level        -1 (min) .. 0 (mean) .. +1 (max) cost per unit (shared)
    protection_scale  multiplier on the FLOPROS design return period (river)
    protection_abs_rp absolute return period (years), replaces protection_scale
                      when set - applied uniformly to every feature regardless
                      of its FLOPROS baseline, so features FLOPROS marks as
                      unprotected (baseline RP = 0) are actually swept too
                      (protection_scale's multiplicative 0 * x = 0 cannot do
                      that - see README "Method notes").
    depth_offset      additive water-depth error in metres (river)
    depth_scale       multiplicative water-depth factor, e.g. 0.9-1.1 (river);
                      an alternative to depth_offset - a scenario uses one or
                      the other, the unused one stays at its no-op default
    pga_scale         multiplier on PGA (earthquake hazard-map uncertainty)
    gust_scale        multiplier on 3-sec gust speed (windstorm hazard-map
                      uncertainty, analogous to pga_scale)
    aggregation       'per_cell'  = curve per raster cell, then sum (reference)
                      'mean_depth' = length-weighted mean intensity per feature
                                     first, then curve applied once (all hazards)
    include_river / include_earthquake / include_windstorm / include_coastal
                      bool - which hazard to compute at all (each scenario
                      computes exactly one, skipping the others' numpy work
                      entirely). Windstorm uses a fixed RP50 design-standard
                      protection cutoff (WIND_DESIGN_RP) and no climate shift.
                      Coastal reuses the river flood curves and the
                      depth_offset/depth_scale + protection_scale/
                      protection_abs_rp factors, but against its own coastal
                      protection standard (coast_prot_rp) and with no warming.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .curves import (
    AssetConfig,
    get_asset_config,
    load_eq_edr_tables,
    load_flood_curves,
    load_wind_curves,
)
from .paths import base_stem, hazard_stem, load_config

WARMING_LEVELS = {"current": None, "1.5C": "15", "2.0C": "20", "3.0C": "30", "4.0C": "40"}
# Clip bounds for shifted anchor RPs, mirroring _safe_rp() in AssetRisk_PanEU
ANCHOR_CLIPS = {10: (1.0, 99.0), 100: (1.0, 499.0), 500: (1.0, 1000.0)}
# Windstorm design standard: RP50 for every asset type (IEC 60826 lower
# bound), applied as a fixed protection cutoff - mirrors
# WIND_PROTECTION_STANDARD_RP in AssetRisk_PanEU/hazard_windstorm.py.
WIND_DESIGN_RP = 50.0


@dataclass
class HazardProfiles:
    """Cached exposure fragments of one hazard, ready for vectorized damage."""

    rps: np.ndarray            # (n_rp,) ascending return periods
    p_pair: np.ndarray         # (n_rows,) seg * n_rp + rp_idx
    p_intensity: np.ndarray    # (n_rows,) hazard intensity (m depth / g PGA)
    p_qty: np.ndarray          # (n_rows,) exposed quantity (m / m^2 / count)
    p_group: np.ndarray        # (n_rows,) curve-group index per row
    seg_group: np.ndarray      # (n_seg,) curve-group index per feature

    @property
    def n_rp(self) -> int:
        return len(self.rps)


@dataclass
class ModelData:
    """All arrays the risk model needs, loaded once and reused."""

    n_seg: int
    maxdam3: np.ndarray             # (n_seg, 3) min/mean/max EUR per unit
    prot_rp: np.ndarray             # (n_seg,) FLOPROS river design RP (0 = unprotected)
    coast_prot_rp: np.ndarray       # (n_seg,) COASTPROS coastal design RP (0 = unprotected)
    class_idx: np.ndarray           # (n_seg,) index into report_classes
    report_classes: list[str]
    hazards: dict[str, HazardProfiles] = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)         # warming code -> (n_seg, 3)
    flood_curve_tables: dict = field(default_factory=dict)  # curve_id -> (x, y)
    eq_curve_tables: dict = field(default_factory=dict)
    wind_curve_tables: dict = field(default_factory=dict)
    flood_group_order: list[str] = field(default_factory=list)
    eq_group_order: list[str] = field(default_factory=list)
    wind_group_order: list[str] = field(default_factory=list)


def _load_profiles(path, rps: np.ndarray, seg_group: np.ndarray) -> HazardProfiles:
    prof = pd.read_parquet(path)
    n_rp = len(rps)
    rp_to_idx = {int(rp): i for i, rp in enumerate(rps)}
    p_seg = prof["seg"].to_numpy(np.int64)
    p_rp_idx = prof["rp"].map(rp_to_idx).to_numpy(np.int64)
    return HazardProfiles(
        rps=rps,
        p_pair=p_seg * n_rp + p_rp_idx,
        p_intensity=prof["intensity"].to_numpy(np.float64),
        p_qty=prof["quantity"].to_numpy(np.float64),
        p_group=seg_group[p_seg],
        seg_group=seg_group,
    )


def load_model_data(cfg: dict | None = None, asset_cfg: AssetConfig | None = None) -> ModelData:
    if cfg is None:
        cfg = load_config()
    if asset_cfg is None:
        asset_cfg = get_asset_config(cfg["asset_type"])

    seg = pd.read_parquet(cfg["intermediate_dir"] / f"{base_stem(cfg)}_segments.parquet")
    n_seg = len(seg)

    flood_group_order = sorted(asset_cfg.flood_groups)
    eq_group_order = sorted(asset_cfg.eq_groups)
    wind_group_order = sorted(asset_cfg.wind_groups)
    seg_flood_group_idx = (
        seg["flood_group"].map({g: i for i, g in enumerate(flood_group_order)}).to_numpy(np.int32)
    )
    seg_eq_group_idx = (
        seg["eq_group"].map({g: i for i, g in enumerate(eq_group_order)}).to_numpy(np.int32)
    )
    # wind_group is only written by Stage 1 for assets that support windstorm;
    # absent for roads (and any pre-windstorm intermediate files).
    if wind_group_order and "wind_group" in seg.columns:
        seg_wind_group_idx = (
            seg["wind_group"].map({g: i for i, g in enumerate(wind_group_order)}).to_numpy(np.int32)
        )
    else:
        seg_wind_group_idx = np.zeros(n_seg, dtype=np.int32)

    anchors = {}
    if "river" in cfg["hazards"]:
        for code in ("15", "20", "30", "40"):
            arr = np.empty((n_seg, 3), dtype=np.float64)
            for j, a in enumerate((10, 100, 500)):
                col = seg[f"new_rp{a}_w{code}"].to_numpy(np.float64)
                lo, hi = ANCHOR_CLIPS[a]
                col = np.clip(col, lo, hi)
                arr[:, j] = np.where(np.isnan(col), float(a), col)
            anchors[code] = arr

    flood_curve_ids = sorted({c for curves in asset_cfg.flood_groups.values() for c in curves})
    flood_curve_tables = {}
    if flood_curve_ids:
        curve_df = load_flood_curves(cfg["vulnerability_path"], flood_curve_ids)
        flood_curve_tables = {
            cid: (curve_df.index.to_numpy(np.float64), curve_df[cid].to_numpy(np.float64))
            for cid in flood_curve_ids
        }

    eq_curve_ids = sorted({c for curves in asset_cfg.eq_groups.values() for c in curves})
    eq_curve_tables = (
        load_eq_edr_tables(cfg["fragility_path"], eq_curve_ids) if eq_curve_ids else {}
    )

    wind_curve_ids = sorted({c for curves in asset_cfg.wind_groups.values() for c in curves})
    wind_curve_tables = {}
    if wind_curve_ids:
        wind_df = load_wind_curves(cfg["vulnerability_path"], wind_curve_ids)
        wind_curve_tables = {
            cid: (wind_df.index.to_numpy(np.float64), wind_df[cid].to_numpy(np.float64))
            for cid in wind_curve_ids
        }

    # Coastal flood reuses the river flood curve groups (same F-curves), so it
    # shares the flood group order/index and curve tables - only its hazard
    # profiles and its protection standard differ.
    group_idx_by_hazard = {
        "river": seg_flood_group_idx,
        "earthquake": seg_eq_group_idx,
        "windstorm": seg_wind_group_idx,
        "coastal": seg_flood_group_idx,
    }
    hazards = {}
    for hazard, hcfg in cfg["hazards"].items():
        prof_path = cfg["intermediate_dir"] / f"{hazard_stem(cfg, hazard)}_profiles.parquet"
        if not prof_path.exists():
            continue
        rps = np.asarray(sorted(hcfg["return_periods"]), dtype=np.float64)
        group_idx = group_idx_by_hazard.get(hazard, seg_flood_group_idx)
        hazards[hazard] = _load_profiles(prof_path, rps, group_idx)

    class_to_idx = {c: i for i, c in enumerate(asset_cfg.report_classes)}
    class_idx = seg["report_class"].map(class_to_idx).to_numpy(np.int64)

    # coast_prot_rp only written by Stage 1 for coastal countries (absent for
    # landlocked ones and pre-coastal intermediate files).
    if "coast_prot_rp" in seg.columns:
        coast_prot_rp = seg["coast_prot_rp"].to_numpy(np.float64)
    else:
        coast_prot_rp = np.zeros(n_seg, dtype=np.float64)

    return ModelData(
        n_seg=n_seg,
        maxdam3=seg[["maxdam_min", "maxdam_mean", "maxdam_max"]].to_numpy(np.float64),
        prot_rp=seg["prot_rp"].to_numpy(np.float64),
        coast_prot_rp=coast_prot_rp,
        class_idx=class_idx,
        report_classes=asset_cfg.report_classes,
        hazards=hazards,
        anchors=anchors,
        flood_curve_tables=flood_curve_tables,
        eq_curve_tables=eq_curve_tables,
        wind_curve_tables=wind_curve_tables,
        flood_group_order=flood_group_order,
        eq_group_order=eq_group_order,
        wind_group_order=wind_group_order,
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _cost_per_unit(data: ModelData, cost_level: float) -> np.ndarray:
    """Piecewise-linear interpolation min <- mean -> max, cost_level in [-1, 1]."""
    lo, mid, hi = data.maxdam3[:, 0], data.maxdam3[:, 1], data.maxdam3[:, 2]
    if cost_level >= 0:
        return mid + cost_level * (hi - mid)
    return mid + cost_level * (mid - lo)


def _damage_fraction(
    intensity: np.ndarray, group: np.ndarray, curves_by_group: dict[int, tuple]
) -> np.ndarray:
    """Damage fraction per element, using the group-specific curve."""
    if len(curves_by_group) == 1:
        x, y = next(iter(curves_by_group.values()))
        return np.interp(intensity, x, y)
    frac = np.empty_like(intensity)
    for g, (x, y) in curves_by_group.items():
        mask = group == g
        frac[mask] = np.interp(intensity[mask], x, y)
    return frac


def _damage_matrix(
    hz: HazardProfiles,
    intensity: np.ndarray,
    curves_by_group: dict[int, tuple],
    aggregation: str,
    n_seg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Damage per (feature, RP) in unmultiplied-by-cost units.

    Returns (damage_qty, exposed_qty) both shaped (n_seg, n_rp); damage_qty
    must still be multiplied by per-unit costs. exposed_qty is the exposed
    quantity after the intensity transform (cells pushed to <= 0 drop out).
    """
    n_bins = n_seg * hz.n_rp
    active = intensity > 0
    w = np.where(active, hz.p_qty, 0.0)
    exposed_qty = np.bincount(hz.p_pair, weights=w, minlength=n_bins)

    if aggregation == "per_cell":
        frac = _damage_fraction(intensity, hz.p_group, curves_by_group)
        dmg = np.bincount(hz.p_pair, weights=frac * w, minlength=n_bins)
    elif aggregation == "mean_depth":
        isum = np.bincount(hz.p_pair, weights=w * intensity, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_int = np.where(exposed_qty > 0, isum / exposed_qty, 0.0)
        group_pair = np.repeat(hz.seg_group, hz.n_rp)
        frac = _damage_fraction(mean_int, group_pair, curves_by_group)
        dmg = frac * exposed_qty
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    shape = (n_seg, hz.n_rp)
    return dmg.reshape(shape), exposed_qty.reshape(shape)


def _shift_rps(rp_values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Piecewise-linear map of return periods through basin anchor points.

    Equivalent to scipy interp1d over (10, 100, 500) -> anchors with linear
    extrapolation, floored at RP 1 (as in adjust_return_periods_climate).

    Args:
        rp_values: (n_seg, k) or broadcastable original RPs
        anchors:   (n_seg, 3) new RPs for original RP 10 / 100 / 500
    """
    a10 = anchors[:, [0]]
    a100 = anchors[:, [1]]
    a500 = anchors[:, [2]]
    below = a10 + (rp_values - 10.0) * (a100 - a10) / 90.0
    above = a100 + (rp_values - 100.0) * (a500 - a100) / 400.0
    return np.maximum(np.where(rp_values <= 100.0, below, above), 1.0)


def _integrate_ead(
    damage: np.ndarray, rp_adj: np.ndarray, prot_rp: np.ndarray
) -> np.ndarray:
    """Vectorized EAD per feature: trapezoid over p = 1/RP with protection cutoff.

    Mirrors integrate_ead()/_apply_protection_standard() in AssetRisk_PanEU:
    damages for events more frequent than the protection RP are zero, the
    damage at the protection RP itself is interpolated linearly in RP space,
    and no tail beyond the largest RP or below the smallest RP is added.

    Args:
        damage:  (n_seg, n_rp) damage per return period, EUR
        rp_adj:  (n_seg, n_rp) (climate-shifted) return periods, ascending
        prot_rp: (n_seg,) protection RP (0 = unprotected)
    """
    p = 1.0 / rp_adj                            # descending along axis 1
    rp_lo, rp_hi = rp_adj[:, :-1], rp_adj[:, 1:]
    d_lo, d_hi = damage[:, :-1], damage[:, 1:]
    p_lo, p_hi = p[:, :-1], p[:, 1:]

    full = 0.5 * (d_lo + d_hi) * (p_lo - p_hi)  # unprotected trapezoid per pair

    rp_c = prot_rp[:, None]                     # (n_seg, 1) broadcast
    with np.errstate(invalid="ignore", divide="ignore"):
        d_c = d_lo + (rp_c - rp_lo) * (d_hi - d_lo) / (rp_hi - rp_lo)
        p_c = np.where(rp_c > 0, 1.0 / rp_c, np.inf)
        partial = 0.5 * (d_c + d_hi) * (p_c - p_hi)

    contrib = np.where(
        rp_c <= rp_lo,
        full,
        np.where(rp_c >= rp_hi, 0.0, partial),
    )
    return contrib.sum(axis=1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def compute_risk(
    data: ModelData,
    curve_choices: dict[str, str],
    warming: str = "current",
    cost_level: float = 0.0,
    protection_scale: float = 1.0,
    protection_abs_rp: float | None = None,
    depth_offset: float = 0.0,
    depth_scale: float = 1.0,
    pga_scale: float = 1.0,
    gust_scale: float = 1.0,
    aggregation: str = "per_cell",
    include_river: bool = True,
    include_earthquake: bool = True,
    include_windstorm: bool = False,
    include_coastal: bool = False,
) -> dict:
    """Evaluate one parameterization; returns scalar outcomes in a dict.

    curve_choices maps every group name that will actually be used (i.e.
    every entry of data.flood_group_order if include_river, data.eq_group_order
    if include_earthquake, and/or data.wind_group_order if include_windstorm)
    to a chosen curve ID from that group.

    River water depth is transformed as ``depth * depth_scale + depth_offset``:
    depth_offset is an additive bias in metres (default 0), depth_scale a
    multiplicative factor (default 1). A scenario varies one or the other
    (see src/ema_model.py); the unused one keeps its no-op default. Windstorm
    gust speed is scaled multiplicatively by gust_scale (analogous to
    pga_scale for earthquake).
    """
    cost = _cost_per_unit(data, cost_level)
    ead_river = np.zeros(data.n_seg)
    ead_eq = np.zeros(data.n_seg)
    ead_wind = np.zeros(data.n_seg)
    ead_coast = np.zeros(data.n_seg)
    out: dict[str, float] = {}

    if include_river:
        if "river" not in data.hazards:
            raise ValueError("include_river=True but no river profiles were loaded")
        hz = data.hazards["river"]
        curves_by_group = {
            i: data.flood_curve_tables[curve_choices[name]]
            for i, name in enumerate(data.flood_group_order)
        }
        depth = np.maximum(hz.p_intensity * depth_scale + depth_offset, 0.0)
        dmg_qty, exposed_qty = _damage_matrix(hz, depth, curves_by_group, aggregation, data.n_seg)
        damage_river = dmg_qty * cost[:, None]
        idx100 = int(np.argmin(np.abs(hz.rps - 100)))
        damage_rp100 = damage_river[:, idx100].copy()  # before any warming re-sort

        if protection_abs_rp is not None:
            prot_eff = np.full(data.n_seg, float(protection_abs_rp), dtype=np.float64)
        else:
            prot_eff = data.prot_rp * protection_scale

        code = WARMING_LEVELS[warming]
        if code is None:
            rp_adj = np.broadcast_to(hz.rps, damage_river.shape)
        else:
            anchors = data.anchors[code]
            rp_adj = _shift_rps(np.broadcast_to(hz.rps, damage_river.shape), anchors)
            shifted_prot = _shift_rps(prot_eff[:, None], anchors)[:, 0]
            prot_eff = np.where(prot_eff > 0, shifted_prot, 0.0)
            # A non-monotonic basin anchor map can unsort the shifted RPs; the
            # reference sorts by RP before integrating, so do the same.
            order = np.argsort(rp_adj, axis=1, kind="stable")
            rp_adj = np.take_along_axis(rp_adj, order, axis=1)
            damage_river = np.take_along_axis(damage_river, order, axis=1)

        ead_river = _integrate_ead(damage_river, rp_adj, prot_eff)
        out["EAD_river_MEUR"] = float(ead_river.sum() / 1e6)
        out["damage_RP100_river_MEUR"] = float(damage_rp100.sum() / 1e6)
        out["exposed_qty_RP100_river"] = float(exposed_qty[:, idx100].sum())

    if include_earthquake:
        if "earthquake" not in data.hazards:
            raise ValueError("include_earthquake=True but no earthquake profiles were loaded")
        hz_eq = data.hazards["earthquake"]
        curves_by_group_eq = {
            i: data.eq_curve_tables[curve_choices[name]]
            for i, name in enumerate(data.eq_group_order)
        }
        pga = hz_eq.p_intensity * pga_scale
        dmg_qty_eq, _ = _damage_matrix(hz_eq, pga, curves_by_group_eq, aggregation, data.n_seg)
        damage_eq = dmg_qty_eq * cost[:, None]
        ead_eq = _integrate_ead(
            damage_eq, np.broadcast_to(hz_eq.rps, damage_eq.shape), np.zeros(data.n_seg)
        )
        out["EAD_earthquake_MEUR"] = float(ead_eq.sum() / 1e6)

    if include_windstorm:
        if "windstorm" not in data.hazards:
            raise ValueError("include_windstorm=True but no windstorm profiles were loaded")
        hz_w = data.hazards["windstorm"]
        curves_by_group_w = {
            i: data.wind_curve_tables[curve_choices[name]]
            for i, name in enumerate(data.wind_group_order)
        }
        speed = hz_w.p_intensity * gust_scale
        dmg_qty_w, exposed_qty_w = _damage_matrix(
            hz_w, speed, curves_by_group_w, aggregation, data.n_seg
        )
        damage_wind = dmg_qty_w * cost[:, None]
        idx100_w = int(np.argmin(np.abs(hz_w.rps - 100)))
        # Fixed RP50 design standard for every feature (no per-feature
        # protection raster and no climate shift for wind).
        prot_wind = np.full(data.n_seg, WIND_DESIGN_RP, dtype=np.float64)
        ead_wind = _integrate_ead(
            damage_wind, np.broadcast_to(hz_w.rps, damage_wind.shape), prot_wind
        )
        out["EAD_windstorm_MEUR"] = float(ead_wind.sum() / 1e6)
        out["damage_RP100_windstorm_MEUR"] = float(damage_wind[:, idx100_w].sum() / 1e6)
        out["exposed_qty_RP100_windstorm"] = float(exposed_qty_w[:, idx100_w].sum())

    if include_coastal:
        if "coastal" not in data.hazards:
            raise ValueError("include_coastal=True but no coastal profiles were loaded")
        # Coastal reuses the river flood depth-damage curves and groups; the
        # depth transform (depth_scale/depth_offset) is identical. It differs
        # from river only in its own protection standard (coast_prot_rp, from
        # COASTPROS) and in having no climate-warming RP shift.
        hz_c = data.hazards["coastal"]
        curves_by_group_c = {
            i: data.flood_curve_tables[curve_choices[name]]
            for i, name in enumerate(data.flood_group_order)
        }
        depth_c = np.maximum(hz_c.p_intensity * depth_scale + depth_offset, 0.0)
        dmg_qty_c, exposed_qty_c = _damage_matrix(
            hz_c, depth_c, curves_by_group_c, aggregation, data.n_seg
        )
        damage_coast = dmg_qty_c * cost[:, None]
        idx100_c = int(np.argmin(np.abs(hz_c.rps - 100)))
        if protection_abs_rp is not None:
            prot_coast = np.full(data.n_seg, float(protection_abs_rp), dtype=np.float64)
        else:
            prot_coast = data.coast_prot_rp * protection_scale
        ead_coast = _integrate_ead(
            damage_coast, np.broadcast_to(hz_c.rps, damage_coast.shape), prot_coast
        )
        out["EAD_coastal_MEUR"] = float(ead_coast.sum() / 1e6)
        out["damage_RP100_coastal_MEUR"] = float(damage_coast[:, idx100_c].sum() / 1e6)
        out["exposed_qty_RP100_coastal"] = float(exposed_qty_c[:, idx100_c].sum())

    ead_total = ead_river + ead_eq + ead_wind + ead_coast
    out["total_EAD_MEUR"] = float(ead_total.sum() / 1e6)

    class_ead = np.bincount(data.class_idx, weights=ead_total, minlength=len(data.report_classes))
    for i, cls in enumerate(data.report_classes):
        out[f"EAD_{cls}_MEUR"] = float(class_ead[i] / 1e6)

    return out
