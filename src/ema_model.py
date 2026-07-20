"""EMA Workbench model definition: the 9 uncertainty factors and outcomes.

Factor overview (all applied inside src.risk_model.compute_risk):

  warming           CategoricalParameter — climate change level (river flood
                    only). 'current' uses the baseline return-period maps; the
                    degree levels shift every return period via basin-scale
                    anchor maps derived from discharge projections.
  curve_main        CategoricalParameter — flood depth-damage curve for
                    motorway/trunk/primary roads (F7.4-F7.7).
  curve_other       CategoricalParameter — flood curve for secondary and
                    lower road classes (F7.8-F7.9).
  eq_curve          CategoricalParameter — earthquake fragility curve for all
                    road classes (E7.2-E7.10), collapsed to an expected
                    damage ratio vs PGA.
  cost_level        RealParameter [-1, 1] — reconstruction cost per metre,
                    piecewise-linear between min (-1), mean (0) and max (+1);
                    shared by both hazards.
  protection_scale  RealParameter [0, 2] — multiplier on the FLOPROS flood
                    design standard. 0 = no protection, 1 = FLOPROS estimate.
  depth_offset      RealParameter [-0.5, 0.5] m — additive bias on all
                    inundation depths (flood-map depth uncertainty).
  pga_scale         RealParameter [0.8, 1.2] — multiplier on PGA (seismic
                    hazard-map uncertainty).
  aggregation       CategoricalParameter — exposure aggregation order:
                    'per_cell' applies the curve to each raster-cell fragment
                    and sums (damagescanner behaviour); 'mean_depth' first
                    averages intensity over the segment's exposed part.

Windstorm is deliberately absent: the roads wind curve (W7.2) is identically
zero in the MIRACA vulnerability database.
"""

from ema_workbench import CategoricalParameter, Model, RealParameter, ScalarOutcome

from .curves import CURVES_MAIN, CURVES_OTHER, EQ_CURVES, REPORT_CLASSES
from .risk_model import WARMING_LEVELS, compute_risk, load_model_data

OUTCOME_NAMES = [
    "total_EAD_MEUR",
    "EAD_river_MEUR",
    "EAD_earthquake_MEUR",
    "damage_RP100_river_MEUR",
    "exposed_km_RP100_river",
] + [f"EAD_{cls}_MEUR" for cls in REPORT_CLASSES]

_DATA = None


def _get_data():
    """Lazy singleton so each (worker) process loads the arrays exactly once."""
    global _DATA
    if _DATA is None:
        _DATA = load_model_data()
    return _DATA


def flood_risk_model(
    warming="current",
    curve_main="F7.5",
    curve_other="F7.9",
    eq_curve="E7.6",
    cost_level=0.0,
    protection_scale=1.0,
    depth_offset=0.0,
    pga_scale=1.0,
    aggregation="per_cell",
) -> dict:
    return compute_risk(
        _get_data(),
        warming=str(warming),
        curve_main=str(curve_main),
        curve_other=str(curve_other),
        eq_curve=str(eq_curve),
        cost_level=float(cost_level),
        protection_scale=float(protection_scale),
        depth_offset=float(depth_offset),
        pga_scale=float(pga_scale),
        aggregation=str(aggregation),
    )


def build_model() -> Model:
    model = Model("multihazardrisk", function=flood_risk_model)
    model.uncertainties = [
        CategoricalParameter("warming", list(WARMING_LEVELS.keys())),
        CategoricalParameter("curve_main", CURVES_MAIN),
        CategoricalParameter("curve_other", CURVES_OTHER),
        CategoricalParameter("eq_curve", EQ_CURVES),
        RealParameter("cost_level", -1.0, 1.0),
        RealParameter("protection_scale", 0.0, 2.0),
        RealParameter("depth_offset", -0.5, 0.5),
        RealParameter("pga_scale", 0.8, 1.2),
        CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
    ]
    model.outcomes = [ScalarOutcome(name) for name in OUTCOME_NAMES]
    return model
