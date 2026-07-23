"""EMA Workbench model definition: scenario-aware, works for any asset type.

Every scenario computes exactly ONE hazard (the two hazards are never combined
- they are studied independently so their uncertainty structures don't mix).
The eight scenarios:

  River flood (six - three protection treatments x two depth treatments):
    flood_baseline        Protection = FLOPROS design standard x protection_scale
                          (multiplier in [0, 2]). Depth error additive:
                          depth_offset in [-0.5, +0.5] m.
    flood_baseline_ds     Same protection, but depth error MULTIPLICATIVE:
                          depth_scale in [0.9, 1.1] (i.e. +/-10%) instead of the
                          additive offset. ("ds" = depth-scale.)
    flood_absprot         Protection sampled as an ABSOLUTE return period
                          (protection_abs_rp in [5, 200] years) applied uniformly
                          to every feature, replacing protection_scale entirely.
                          Unlike the multiplier, this actually varies protection
                          for features FLOPROS marks as unprotected (baseline
                          RP=0). Additive depth error.
    flood_absprot_ds      Same absolute protection, multiplicative depth error.
    flood_noprot          Protection held fixed at FLOPROS (protection_scale=1.0,
                          not sampled), isolating the other flood factors.
                          Additive depth error.
    flood_noprot_ds       Same fixed protection, multiplicative depth error.

  Coastal flood (six - the river-flood set mirrored for the coastal hazard):
    coastal_baseline / _ds, coastal_absprot / _ds, coastal_noprot / _ds
                          Same three protection treatments x two depth
                          treatments as the river flood scenarios, reusing the
                          identical flood depth-damage curves, but for the
                          coastal hazard and against the COASTPROS coastal
                          protection standard. No 'warming' factor (coastal
                          sea-level rise is a separate mechanism, not modelled
                          here). Only defined for coastal (non-landlocked)
                          countries - skipped for e.g. LUX.

  Earthquake (one):
    earthquake            Earthquake only (eq curve group(s), cost_level,
                          pga_scale, aggregation). No protection standard.

  Windstorm (two - two protection treatments):
    windstorm             Windstorm only (wind curve group(s), cost_level,
                          gust_scale, aggregation). Fixed RP50 design standard
                          (WIND_DESIGN_RP, not sampled).
    windstorm_absprot     Same as windstorm, but the design standard is sampled
                          as an ABSOLUTE return period (protection_abs_rp in
                          [25, 200] years) applied uniformly to every feature,
                          replacing the fixed RP50 - isolating the sensitivity
                          to the assumed wind design standard.
                          Both only defined for assets that support windstorm
                          (not roads/ports).

Each scenario maps to exactly one hazard (SCENARIO_HAZARD); a scenario only
applies to an asset when that hazard applies to the asset
(src/curves.py::applicable_hazards) - so the study orchestrator skips e.g.
windstorm for roads automatically.

Every uncertainty factor is declared as either an ema_workbench Parameter
(sampled) or a Constant (held fixed but still passed to the model function -
this is what lets "flood_noprot" call compute_risk with protection_scale=1.0
without exposing it as a samplable uncertainty).

Curve-choice factors are named "curve_<group>" where <group> is one of the
asset's automatically-derived curve group names (src/curves.py). A group with
only one available curve becomes a Constant (there is nothing to vary);
groups with >1 curve become a CategoricalParameter.
"""

from ema_workbench import CategoricalParameter, Constant, Model, RealParameter, ScalarOutcome

from .curves import AssetConfig, applicable_hazards, get_asset_config
from .risk_model import WARMING_LEVELS, ModelData, compute_risk, load_model_data
from .paths import load_config

# Ordered so each hazard's protection treatments and their depth-scale twins
# sit together. Coastal mirrors the river flood scenarios (same treatments,
# same curves) but for the coastal hazard.
SCENARIOS = [
    "flood_baseline", "flood_baseline_ds",
    "flood_absprot", "flood_absprot_ds",
    "flood_noprot", "flood_noprot_ds",
    "coastal_baseline", "coastal_baseline_ds",
    "coastal_absprot", "coastal_absprot_ds",
    "coastal_noprot", "coastal_noprot_ds",
    "earthquake",
    "windstorm", "windstorm_absprot",
]

# The subset the study orchestrator runs by default. All 14 scenarios above
# remain available via `run_study.py --scenarios ...`; this just narrows the
# no-argument run. For both floods we keep only the absolute-protection +
# multiplicative-depth (_ds) variant, plus the two single-hazard scenarios.
DEFAULT_SCENARIOS = [
    "flood_absprot_ds",
    "coastal_absprot_ds",
    "earthquake",
    "windstorm",
    "windstorm_absprot",
]

# Which single hazard each scenario computes. Used to decide applicability
# (a scenario applies to an asset/country iff its hazard does) and to pick the
# include_* flag in build_model.
SCENARIO_HAZARD = {
    "flood_baseline": "river", "flood_baseline_ds": "river",
    "flood_absprot": "river", "flood_absprot_ds": "river",
    "flood_noprot": "river", "flood_noprot_ds": "river",
    "coastal_baseline": "coastal", "coastal_baseline_ds": "coastal",
    "coastal_absprot": "coastal", "coastal_absprot_ds": "coastal",
    "coastal_noprot": "coastal", "coastal_noprot_ds": "coastal",
    "earthquake": "earthquake",
    "windstorm": "windstorm", "windstorm_absprot": "windstorm",
}

# The protection treatment (scale/abs/fixed) and depth-error treatment
# (offset = additive, scale = multiplicative "_ds" twin) for each flood-type
# scenario. Coastal entries carry no 'warming' factor (see _build_flood_model).
_FLOOD_SCENARIOS = {
    "flood_baseline": ("scale", "offset"),
    "flood_baseline_ds": ("scale", "scale"),
    "flood_absprot": ("abs", "offset"),
    "flood_absprot_ds": ("abs", "scale"),
    "flood_noprot": ("fixed", "offset"),
    "flood_noprot_ds": ("fixed", "scale"),
}
_COASTAL_SCENARIOS = {
    "coastal_baseline": ("scale", "offset"),
    "coastal_baseline_ds": ("scale", "scale"),
    "coastal_absprot": ("abs", "offset"),
    "coastal_absprot_ds": ("abs", "scale"),
    "coastal_noprot": ("fixed", "offset"),
    "coastal_noprot_ds": ("fixed", "scale"),
}

CURVE_PREFIX = "curve_"

_DATA_CACHE: dict[tuple[str, str], ModelData] = {}


def scenario_applies(scenario: str, asset: str, country: str | None = None) -> bool:
    """True iff `scenario`'s hazard applies to this asset (and country).

    country matters only for coastal scenarios (skipped for landlocked
    countries); pass it so those are filtered correctly.
    """
    return SCENARIO_HAZARD[scenario] in applicable_hazards(asset, country)


def applicable_scenarios(asset: str, country: str | None = None) -> list[str]:
    return [s for s in SCENARIOS if scenario_applies(s, asset, country)]


def _get_data() -> ModelData:
    """Per-process cache: each MultiprocessingEvaluator worker loads once.

    Reads (country, asset) from config/env vars rather than a captured
    argument on purpose: the model function below must be a plain
    module-level function (not a closure) so MultiprocessingEvaluator can
    pickle it for spawned worker processes on Windows - closures over local
    variables are not picklable. Workers inherit the parent's environment at
    spawn time, so the MIRACA_COUNTRY/MIRACA_ASSET overrides set by the CLI
    scripts before evaluation starts are still visible here.
    """
    cfg = load_config()
    key = (cfg["country"], cfg["asset_type"])
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_model_data(cfg)
    return _DATA_CACHE[key]


def flood_risk_model(**kwargs) -> dict:
    """The EMA Workbench model function - identical across every scenario/asset.

    Accepts **kwargs so it can serve any (asset-specific, dynamically named)
    set of curve_<group> parameters and Constants without per-asset
    branching - they're split out here by the 'curve_' prefix and forwarded
    to compute_risk() as curve_choices.
    """
    data = _get_data()
    curve_choices = {
        k[len(CURVE_PREFIX):]: v for k, v in kwargs.items() if k.startswith(CURVE_PREFIX)
    }
    scalar_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(CURVE_PREFIX)}
    return compute_risk(data, curve_choices=curve_choices, **scalar_kwargs)


def _add_curve_params(
    uncertainties: list, constants: list, groups: dict[str, list[str]]
) -> None:
    for name in sorted(groups):
        curves = groups[name]
        pname = f"{CURVE_PREFIX}{name}"
        if len(curves) > 1:
            uncertainties.append(CategoricalParameter(pname, curves))
        else:
            constants.append(Constant(pname, curves[0]))


def _class_outcomes(asset_cfg: AssetConfig) -> list[str]:
    return [f"EAD_{c}_MEUR" for c in asset_cfg.report_classes]


def _build_water_model(cfg: dict, asset_cfg: AssetConfig, scenario: str, hazard: str) -> Model:
    """Build a river- or coastal-flood scenario model (they share everything
    but the hazard flag, the outcome names, and - for coastal - dropping the
    river-only 'warming' climate factor)."""
    treatments = _FLOOD_SCENARIOS if hazard == "river" else _COASTAL_SCENARIOS
    prot_kind, depth_kind = treatments[scenario]
    uncertainties: list = []
    constants: list = []

    _add_curve_params(uncertainties, constants, asset_cfg.flood_groups)
    uncertainties += [
        RealParameter("cost_level", -1.0, 1.0),
        CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
    ]
    if hazard == "river":
        # Climate-driven RP shift is river-basin-anchor based; coastal
        # sea-level rise is a separate (SSP-horizon) mechanism not modelled here.
        uncertainties.append(CategoricalParameter("warming", list(WARMING_LEVELS.keys())))

    # protection treatment (against FLOPROS for river, COASTPROS for coastal -
    # same parameter names, the baseline differs inside compute_risk)
    if prot_kind == "scale":
        uncertainties.append(RealParameter("protection_scale", 0.0, 2.0))
    elif prot_kind == "abs":
        uncertainties.append(RealParameter("protection_abs_rp", 5.0, 200.0))
    elif prot_kind == "fixed":
        constants.append(Constant("protection_scale", 1.0))

    # depth-error treatment: additive offset OR multiplicative scale
    if depth_kind == "offset":
        uncertainties.append(RealParameter("depth_offset", -0.5, 0.5))
    else:
        uncertainties.append(RealParameter("depth_scale", 0.9, 1.1))

    constants += [
        Constant("include_river", hazard == "river"),
        Constant("include_earthquake", False),
        Constant("include_windstorm", False),
        Constant("include_coastal", hazard == "coastal"),
    ]
    hz = "river" if hazard == "river" else "coastal"
    outcomes = (
        ["total_EAD_MEUR", f"EAD_{hz}_MEUR", f"damage_RP100_{hz}_MEUR", f"exposed_qty_RP100_{hz}"]
        + _class_outcomes(asset_cfg)
    )
    model = Model(f"{cfg['country']}_{cfg['asset_type']}_{scenario}", function=flood_risk_model)
    model.uncertainties = uncertainties
    model.constants = constants
    model.outcomes = [ScalarOutcome(name) for name in outcomes]
    return model


def _build_earthquake_model(cfg: dict, asset_cfg: AssetConfig) -> Model:
    uncertainties: list = []
    constants: list = []
    _add_curve_params(uncertainties, constants, asset_cfg.eq_groups)
    uncertainties += [
        RealParameter("cost_level", -1.0, 1.0),
        RealParameter("pga_scale", 0.8, 1.2),
        CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
    ]
    constants += [
        Constant("include_river", False),
        Constant("include_earthquake", True),
        Constant("include_windstorm", False),
        Constant("include_coastal", False),
    ]
    outcomes = ["total_EAD_MEUR", "EAD_earthquake_MEUR"] + _class_outcomes(asset_cfg)
    model = Model(f"{cfg['country']}_{cfg['asset_type']}_earthquake", function=flood_risk_model)
    model.uncertainties = uncertainties
    model.constants = constants
    model.outcomes = [ScalarOutcome(name) for name in outcomes]
    return model


def _build_windstorm_model(cfg: dict, asset_cfg: AssetConfig, scenario: str) -> Model:
    if not asset_cfg.supports_windstorm:
        raise ValueError(
            f"Asset '{cfg['asset_type']}' does not support windstorm "
            f"(no non-degenerate wind curve group); scenario '{scenario}' is not applicable."
        )
    uncertainties: list = []
    constants: list = []
    _add_curve_params(uncertainties, constants, asset_cfg.wind_groups)
    uncertainties += [
        RealParameter("cost_level", -1.0, 1.0),
        RealParameter("gust_scale", 0.9, 1.1),
        CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
    ]
    # 'windstorm' holds the design standard fixed at RP50 (WIND_DESIGN_RP inside
    # compute_risk); 'windstorm_absprot' samples it as an absolute return period
    # applied uniformly to every feature (analogous to flood/coastal absprot).
    if scenario == "windstorm_absprot":
        uncertainties.append(RealParameter("protection_abs_rp", 25.0, 200.0))
    constants += [
        Constant("include_river", False),
        Constant("include_earthquake", False),
        Constant("include_windstorm", True),
        Constant("include_coastal", False),
    ]
    outcomes = (
        ["total_EAD_MEUR", "EAD_windstorm_MEUR", "damage_RP100_windstorm_MEUR",
         "exposed_qty_RP100_windstorm"]
        + _class_outcomes(asset_cfg)
    )
    model = Model(f"{cfg['country']}_{cfg['asset_type']}_{scenario}", function=flood_risk_model)
    model.uncertainties = uncertainties
    model.constants = constants
    model.outcomes = [ScalarOutcome(name) for name in outcomes]
    return model


def build_model(cfg: dict | None = None) -> Model:
    if cfg is None:
        cfg = load_config()
    scenario = cfg["scenario"]
    asset_cfg = get_asset_config(cfg["asset_type"])

    if scenario in _FLOOD_SCENARIOS:
        return _build_water_model(cfg, asset_cfg, scenario, hazard="river")
    if scenario in _COASTAL_SCENARIOS:
        return _build_water_model(cfg, asset_cfg, scenario, hazard="coastal")
    if scenario == "earthquake":
        return _build_earthquake_model(cfg, asset_cfg)
    if scenario in ("windstorm", "windstorm_absprot"):
        return _build_windstorm_model(cfg, asset_cfg, scenario)
    raise ValueError(f"Unknown scenario '{scenario}'; choose from {SCENARIOS}")
