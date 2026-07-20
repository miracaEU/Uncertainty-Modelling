"""Stage 2: fast, fully vectorized multi-hazard risk model for roads.

Consumes the intermediate files written by src.preprocess and evaluates one
parameterization (one EMA Workbench experiment) in tens of milliseconds,
without touching any raster or geometry. Hazards: river flood + earthquake
(windstorm is excluded because the roads wind curve W7.2 is identically zero).

Damage formula (identical to damagescanner.vector._get_damage_per_object):

    damage(segment, RP) = sum_over_cells( f(intensity_cell) * length_m_cell )
                          * maxdam_eur_per_m(object_type)

with f = depth-damage curve (river) or fragility-derived expected damage
ratio vs PGA (earthquake). EAD integration, protection-standard cutoff and
climate RP shifting mirror AssetRisk_PanEU/src/risk_integration.py:

    EAD = trapezoid of damage over exceedance probability p = 1/RP,
          integrated between p(RP_max) and p(min(RP_min, protection RP)),
          damages below the protection RP set to zero (with the damage at
          the protection RP itself linearly interpolated in RP space);
    warming (river only) shifts every RP via per-basin piecewise-linear
    anchor maps (RP10/100/500 -> new RP) before integration;
    earthquake has no protection standard (as in the reference).

Uncertainty factors handled here:
    warming           categorical: current / 1.5C / 2.0C / 3.0C / 4.0C (river)
    curve_main        flood curve for motorway/trunk/primary (F7.4-F7.7)
    curve_other       flood curve for lower road classes (F7.8-F7.9)
    eq_curve          earthquake fragility curve (E7.2-E7.10, all classes)
    cost_level        -1 (min) .. 0 (mean) .. +1 (max) cost per metre (shared)
    protection_scale  multiplier on the FLOPROS design return period (river)
    depth_offset      additive water-depth error in metres (river)
    pga_scale         multiplier on PGA (earthquake hazard-map uncertainty)
    aggregation       'per_cell'  = curve per raster cell, then sum (reference)
                      'mean_depth' = length-weighted mean intensity per segment
                                     first, then curve applied once (all hazards)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .curves import (
    CURVES_MAIN,
    CURVES_OTHER,
    EQ_CURVES,
    REPORT_CLASSES,
    load_eq_edr_tables,
    load_flood_curves,
)
from .paths import load_config, base_stem, hazard_stem

WARMING_LEVELS = {"current": None, "1.5C": "15", "2.0C": "20", "3.0C": "30", "4.0C": "40"}
# Clip bounds for shifted anchor RPs, mirroring _safe_rp() in AssetRisk_PanEU
ANCHOR_CLIPS = {10: (1.0, 99.0), 100: (1.0, 499.0), 500: (1.0, 1000.0)}


@dataclass
class HazardProfiles:
    """Cached exposure fragments of one hazard, ready for vectorized damage."""

    rps: np.ndarray            # (n_rp,) ascending return periods
    p_pair: np.ndarray         # (n_rows,) seg * n_rp + rp_idx
    p_intensity: np.ndarray    # (n_rows,) hazard intensity (m depth / g PGA)
    p_len: np.ndarray          # (n_rows,) exposed length (m)
    p_group: np.ndarray        # (n_rows,) curve-group id per row
    seg_group: np.ndarray      # (n_seg,) curve-group id per segment

    @property
    def n_rp(self) -> int:
        return len(self.rps)


@dataclass
class ModelData:
    """All arrays the risk model needs, loaded once and reused."""

    n_seg: int
    maxdam3: np.ndarray             # (n_seg, 3) min/mean/max EUR per metre
    prot_rp: np.ndarray             # (n_seg,) FLOPROS design RP (0 = unprotected)
    class_idx: np.ndarray           # (n_seg,) index into REPORT_CLASSES
    hazards: dict = field(default_factory=dict)       # name -> HazardProfiles
    anchors: dict = field(default_factory=dict)       # warming code -> (n_seg, 3)
    flood_curves: dict = field(default_factory=dict)  # id -> (depths, fractions)
    eq_curves: dict = field(default_factory=dict)     # id -> (pga, EDR)


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
        p_len=prof["length_m"].to_numpy(np.float64),
        p_group=seg_group[p_seg],
        seg_group=seg_group,
    )


def load_model_data(cfg: dict | None = None) -> ModelData:
    if cfg is None:
        cfg = load_config()
    seg = pd.read_parquet(cfg["intermediate_dir"] / f"{base_stem(cfg)}_segments.parquet")
    n_seg = len(seg)
    road_group = seg["group"].to_numpy(np.int8)

    anchors = {}
    for code in ("15", "20", "30", "40"):
        arr = np.empty((n_seg, 3), dtype=np.float64)
        for j, a in enumerate((10, 100, 500)):
            col = seg[f"new_rp{a}_w{code}"].to_numpy(np.float64)
            lo, hi = ANCHOR_CLIPS[a]
            col = np.clip(col, lo, hi)
            arr[:, j] = np.where(np.isnan(col), float(a), col)
        anchors[code] = arr

    curve_df = load_flood_curves(cfg["vulnerability_path"])
    flood_curves = {
        cid: (curve_df.index.to_numpy(np.float64), curve_df[cid].to_numpy(np.float64))
        for cid in CURVES_MAIN + CURVES_OTHER
    }
    eq_curves = load_eq_edr_tables(cfg["fragility_path"])

    hazards = {}
    for hazard, hcfg in cfg["hazards"].items():
        rps = np.asarray(sorted(hcfg["return_periods"]), dtype=np.float64)
        # river uses the two road-class curve groups; earthquake one shared group
        group = road_group if hazard == "river" else np.zeros(n_seg, dtype=np.int8)
        hazards[hazard] = _load_profiles(
            cfg["intermediate_dir"] / f"{hazard_stem(cfg, hazard)}_profiles.parquet",
            rps,
            group,
        )

    class_idx = (
        seg["report_class"]
        .map({c: i for i, c in enumerate(REPORT_CLASSES)})
        .to_numpy(np.int64)
    )

    return ModelData(
        n_seg=n_seg,
        maxdam3=seg[["maxdam_min", "maxdam_mean", "maxdam_max"]].to_numpy(np.float64),
        prot_rp=seg["prot_rp"].to_numpy(np.float64),
        class_idx=class_idx,
        hazards=hazards,
        anchors=anchors,
        flood_curves=flood_curves,
        eq_curves=eq_curves,
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _cost_per_m(data: ModelData, cost_level: float) -> np.ndarray:
    """Piecewise-linear interpolation min <- mean -> max, cost_level in [-1, 1]."""
    lo, mid, hi = data.maxdam3[:, 0], data.maxdam3[:, 1], data.maxdam3[:, 2]
    if cost_level >= 0:
        return mid + cost_level * (hi - mid)
    return mid + cost_level * (mid - lo)


def _damage_fraction(
    intensity: np.ndarray, group: np.ndarray, curves_by_group: dict
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
    curves_by_group: dict,
    aggregation: str,
    n_seg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Damage per (segment, RP) in metre-equivalents (not yet costed).

    Returns (damage_m, exposed_len) both shaped (n_seg, n_rp); damage_m must
    still be multiplied by EUR/m costs. exposed_len is the exposed length
    after the intensity transform (cells pushed to <= 0 drop out).
    """
    n_bins = n_seg * hz.n_rp
    active = intensity > 0
    w = np.where(active, hz.p_len, 0.0)
    exposed_len = np.bincount(hz.p_pair, weights=w, minlength=n_bins)

    if aggregation == "per_cell":
        frac = _damage_fraction(intensity, hz.p_group, curves_by_group)
        dmg = np.bincount(hz.p_pair, weights=frac * w, minlength=n_bins)
    elif aggregation == "mean_depth":
        isum = np.bincount(hz.p_pair, weights=w * intensity, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_int = np.where(exposed_len > 0, isum / exposed_len, 0.0)
        group_pair = np.repeat(hz.seg_group, hz.n_rp)
        frac = _damage_fraction(mean_int, group_pair, curves_by_group)
        dmg = frac * exposed_len
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    shape = (n_seg, hz.n_rp)
    return dmg.reshape(shape), exposed_len.reshape(shape)


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
    """Vectorized EAD per segment: trapezoid over p = 1/RP with protection cutoff.

    Mirrors integrate_ead()/_apply_protection_standard() in AssetRisk_PanEU:
    damages for events more frequent than the protection RP are zero, the
    damage at the protection RP itself is interpolated linearly in RP space,
    and no tail beyond the largest RP or below the smallest RP is added.
    (Not reproduced: the reference's single-point fallback when the protection
    RP coincides exactly with the largest RP — a measure-zero edge case.)

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
        # damage interpolated at the protection RP (linear in RP space);
        # entries where the cutoff is outside (rp_lo, rp_hi) are masked below
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
    warming: str = "current",
    curve_main: str = "F7.5",
    curve_other: str = "F7.9",
    eq_curve: str = "E7.6",
    cost_level: float = 0.0,
    protection_scale: float = 1.0,
    depth_offset: float = 0.0,
    pga_scale: float = 1.0,
    aggregation: str = "per_cell",
) -> dict:
    """Evaluate one parameterization; returns scalar outcomes in a dict."""
    cost = _cost_per_m(data, cost_level)

    # --- river flood ---
    hz = data.hazards["river"]
    depth = np.maximum(hz.p_intensity + depth_offset, 0.0)
    dmg_m, exposed_len = _damage_matrix(
        hz,
        depth,
        {0: data.flood_curves[curve_main], 1: data.flood_curves[curve_other]},
        aggregation,
        data.n_seg,
    )
    damage_river = dmg_m * cost[:, None]
    idx100 = int(np.argmin(np.abs(hz.rps - 100)))
    damage_rp100 = damage_river[:, idx100].copy()  # before any warming re-sort

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
        # reference implementation sorts by RP before integrating, so do the same.
        order = np.argsort(rp_adj, axis=1, kind="stable")
        rp_adj = np.take_along_axis(rp_adj, order, axis=1)
        damage_river = np.take_along_axis(damage_river, order, axis=1)

    ead_river = _integrate_ead(damage_river, rp_adj, prot_eff)

    # --- earthquake (no protection standard, no climate dependence) ---
    hz_eq = data.hazards["earthquake"]
    pga = hz_eq.p_intensity * pga_scale
    dmg_m_eq, _ = _damage_matrix(
        hz_eq, pga, {0: data.eq_curves[eq_curve]}, aggregation, data.n_seg
    )
    damage_eq = dmg_m_eq * cost[:, None]
    ead_eq = _integrate_ead(
        damage_eq,
        np.broadcast_to(hz_eq.rps, damage_eq.shape),
        np.zeros(data.n_seg),
    )

    # --- outcomes ---
    ead_total = ead_river + ead_eq
    out = {
        "total_EAD_MEUR": float(ead_total.sum() / 1e6),
        "EAD_river_MEUR": float(ead_river.sum() / 1e6),
        "EAD_earthquake_MEUR": float(ead_eq.sum() / 1e6),
        "damage_RP100_river_MEUR": float(damage_rp100.sum() / 1e6),
        "exposed_km_RP100_river": float(exposed_len[:, idx100].sum() / 1000),
    }
    class_ead = np.bincount(
        data.class_idx, weights=ead_total, minlength=len(REPORT_CLASSES)
    )
    for i, cls in enumerate(REPORT_CLASSES):
        out[f"EAD_{cls}_MEUR"] = float(class_ead[i] / 1e6)
    return out
