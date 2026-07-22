"""EMA Workbench model definition: scenario-aware, works for any asset type.

Four modeling scenarios, all built from the same src.risk_model.compute_risk:

  baseline              Full model: both hazards, protection as a FLOPROS
                        multiplier (protection_scale in [0, 2]).
  abs_protection        Same as baseline, but the flood protection standard
                        is sampled as an ABSOLUTE return period in [5, 200]
                        years (protection_abs_rp) applied uniformly to every
                        feature, replacing protection_scale entirely. Unlike
                        the multiplier, this actually varies protection for
                        features FLOPROS marks as unprotected (baseline RP=0)
                        - protection_scale's 0 * x = 0 cannot move those.
  flood_no_protection   River flood only. protection_scale is held fixed at
                        1.0 (i.e. exactly the recorded FLOPROS design
                        standards) rather than sampled, so this isolates how
                        much the OTHER flood factors matter once protection
                        uncertainty is set aside.
  earthquake_only       Earthquake only (eq_curve group(s), cost_level,
                        pga_scale, aggregation). Compares against the
                        earthquake slice of "baseline" to see whether the two
                        hazards' uncertainty structure is independent, or
                        whether joint sampling (shared cost_level/aggregation)
                        creates interaction between them.

Every uncertainty factor is declared as either an ema_workbench Parameter
(sampled) or a Constant (held fixed but still passed to the model function -
this is what lets "flood_no_protection" call compute_risk with
protection_scale=1.0 without exposing it as a samplable uncertainty).

Curve-choice factors are named "curve_<group>" where <group> is one of the
asset's automatically-derived curve group names (src/curves.py). A group with
only one available curve becomes a Constant (there is nothing to vary);
groups with >1 curve become a CategoricalParameter.
"""

from ema_workbench import CategoricalParameter, Constant, Model, RealParameter, ScalarOutcome

from .curves import AssetConfig, get_asset_config
from .risk_model import WARMING_LEVELS, ModelData, compute_risk, load_model_data
from .paths import load_config

SCENARIOS = ["baseline", "abs_protection", "flood_no_protection", "earthquake_only"]

CURVE_PREFIX = "curve_"

_DATA_CACHE: dict[tuple[str, str], ModelData] = {}


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


def build_model(cfg: dict | None = None) -> Model:
    if cfg is None:
        cfg = load_config()
    scenario = cfg["scenario"]
    asset_cfg = get_asset_config(cfg["asset_type"])

    uncertainties: list = []
    constants: list = []

    if scenario == "baseline":
        _add_curve_params(uncertainties, constants, asset_cfg.flood_groups)
        _add_curve_params(uncertainties, constants, asset_cfg.eq_groups)
        uncertainties += [
            CategoricalParameter("warming", list(WARMING_LEVELS.keys())),
            RealParameter("cost_level", -1.0, 1.0),
            RealParameter("protection_scale", 0.0, 2.0),
            RealParameter("depth_offset", -0.5, 0.5),
            RealParameter("pga_scale", 0.8, 1.2),
            CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
        ]
        constants += [Constant("include_river", True), Constant("include_earthquake", True)]
        outcomes = (
            ["total_EAD_MEUR", "EAD_river_MEUR", "EAD_earthquake_MEUR",
             "damage_RP100_river_MEUR", "exposed_qty_RP100_river"]
            + _class_outcomes(asset_cfg)
        )

    elif scenario == "abs_protection":
        _add_curve_params(uncertainties, constants, asset_cfg.flood_groups)
        _add_curve_params(uncertainties, constants, asset_cfg.eq_groups)
        uncertainties += [
            CategoricalParameter("warming", list(WARMING_LEVELS.keys())),
            RealParameter("cost_level", -1.0, 1.0),
            RealParameter("protection_abs_rp", 5.0, 200.0),
            RealParameter("depth_offset", -0.5, 0.5),
            RealParameter("pga_scale", 0.8, 1.2),
            CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
        ]
        constants += [Constant("include_river", True), Constant("include_earthquake", True)]
        outcomes = (
            ["total_EAD_MEUR", "EAD_river_MEUR", "EAD_earthquake_MEUR",
             "damage_RP100_river_MEUR", "exposed_qty_RP100_river"]
            + _class_outcomes(asset_cfg)
        )

    elif scenario == "flood_no_protection":
        _add_curve_params(uncertainties, constants, asset_cfg.flood_groups)
        uncertainties += [
            CategoricalParameter("warming", list(WARMING_LEVELS.keys())),
            RealParameter("cost_level", -1.0, 1.0),
            RealParameter("depth_offset", -0.5, 0.5),
            CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
        ]
        constants += [
            Constant("protection_scale", 1.0),
            Constant("include_river", True),
            Constant("include_earthquake", False),
        ]
        outcomes = (
            ["total_EAD_MEUR", "EAD_river_MEUR", "damage_RP100_river_MEUR", "exposed_qty_RP100_river"]
            + _class_outcomes(asset_cfg)
        )

    elif scenario == "earthquake_only":
        _add_curve_params(uncertainties, constants, asset_cfg.eq_groups)
        uncertainties += [
            RealParameter("cost_level", -1.0, 1.0),
            RealParameter("pga_scale", 0.8, 1.2),
            CategoricalParameter("aggregation", ["per_cell", "mean_depth"]),
        ]
        constants += [Constant("include_river", False), Constant("include_earthquake", True)]
        outcomes = ["total_EAD_MEUR", "EAD_earthquake_MEUR"] + _class_outcomes(asset_cfg)

    else:
        raise ValueError(f"Unknown scenario '{scenario}'; choose from {SCENARIOS}")

    model = Model(f"{cfg['country']}_{cfg['asset_type']}_{scenario}", function=flood_risk_model)
    model.uncertainties = uncertainties
    model.constants = constants
    model.outcomes = [ScalarOutcome(name) for name in outcomes]
    return model
