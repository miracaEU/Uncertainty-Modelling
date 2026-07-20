"""Configuration loading for the uncertainty-modeling pipeline.

Reads config.yml from the project root and exposes resolved paths.
Relative paths in the config are interpreted relative to the project root,
so the same config works no matter where scripts are started from.
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yml"


def set_country_override(country: str | None) -> None:
    """Set a country override that survives into multiprocessing workers.

    CLI scripts call this before building the EMA model; worker processes
    inherit the environment variable and pick it up in load_config().
    """
    if country:
        os.environ["MIRACA_COUNTRY"] = country.upper()


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_country = os.environ.get("MIRACA_COUNTRY")
    if env_country:
        cfg["country"] = env_country

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
        hazard_cfg["dir"] = Path(hazard_cfg["dir"])

    return cfg


def base_stem(cfg: dict) -> str:
    """Hazard-independent filename stem, e.g. 'LUX_roads'."""
    return f"{cfg['country']}_{cfg['asset_type']}"


def hazard_stem(cfg: dict, hazard: str) -> str:
    """Per-hazard filename stem, e.g. 'LUX_roads_river'."""
    return f"{base_stem(cfg)}_{hazard}"
