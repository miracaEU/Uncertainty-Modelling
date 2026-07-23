"""Configuration loading for the uncertainty-modeling pipeline.

Reads config.yml from the project root and exposes resolved paths. Country,
asset type, and modeling scenario can all be overridden via CLI flags on the
individual scripts (--country/--asset/--scenario); the overrides are passed
through environment variables so they survive into MultiprocessingEvaluator
worker processes, which re-import this module fresh in each subprocess.
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yml"
DEFAULT_SCENARIO = "baseline"


def set_country_override(country: str | None) -> None:
    if country:
        os.environ["MIRACA_COUNTRY"] = country.upper()


def set_asset_override(asset: str | None) -> None:
    if asset:
        os.environ["MIRACA_ASSET"] = asset.lower()


def set_scenario_override(scenario: str | None) -> None:
    if scenario:
        os.environ["MIRACA_SCENARIO"] = scenario.lower()


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_country = os.environ.get("MIRACA_COUNTRY")
    if env_country:
        cfg["country"] = env_country
    env_asset = os.environ.get("MIRACA_ASSET")
    if env_asset:
        cfg["asset_type"] = env_asset
    cfg["scenario"] = os.environ.get("MIRACA_SCENARIO", DEFAULT_SCENARIO)

    for key in ("intermediate_dir", "results_dir"):
        p = Path(cfg[key])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        cfg[key] = p

    for key in (
        "exposure_dir",
        "vulnerability_path",
        "fragility_path",
        "protection_standard_path",
        "basin_data_path",
    ):
        cfg[key] = Path(cfg[key])

    for hazard_cfg in cfg["hazards"].values():
        # Local-raster hazards have a dir; streamed ones (kind: stac, e.g.
        # coastal) don't - resolve whatever local paths each one declares.
        if "dir" in hazard_cfg:
            hazard_cfg["dir"] = Path(hazard_cfg["dir"])
        for path_key in ("coastpros_path", "nuts2_path"):
            if path_key in hazard_cfg:
                hazard_cfg[path_key] = Path(hazard_cfg[path_key])

    return cfg


def country_results_dir(cfg: dict, create: bool = True) -> Path:
    """Per-country results subfolder, e.g. results/LUX/.

    All per-combination Stage-2 outputs live under here - the experiment
    archives (*.tar.gz), the per-combo Sobol/feature-score CSVs, and that
    country's own figures/ - so results/ stays navigable when the study spans
    dozens of countries. Cross-country artefacts stay at the results_dir root:
    the aggregated workbook, the global run_study / sobol_convergence logs, the
    compare_countries outputs, and the shared vulnerability-curve figures.

    Pass create=False in read-only/skip checks so a dry run doesn't litter
    empty country folders.
    """
    d = cfg["results_dir"] / cfg["country"]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def base_stem(cfg: dict) -> str:
    """Hazard- and scenario-independent filename stem, e.g. 'LUX_roads'.

    Used for Stage-1 preprocessing outputs (segments.parquet), which are
    shared across every scenario for a given (country, asset) pair.
    """
    return f"{cfg['country']}_{cfg['asset_type']}"


def hazard_stem(cfg: dict, hazard: str) -> str:
    """Per-hazard Stage-1 filename stem, e.g. 'LUX_roads_river'."""
    return f"{base_stem(cfg)}_{hazard}"


def result_stem(cfg: dict) -> str:
    """Scenario-specific filename stem, e.g. 'LUX_roads_abs_protection'.

    Used for every Stage-2 output (experiment archives, summary CSVs,
    figures) so that different scenarios for the same (country, asset) never
    overwrite each other.
    """
    return f"{base_stem(cfg)}_{cfg['scenario']}"
