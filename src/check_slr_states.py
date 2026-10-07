"""Verify that every configured sea-level-rise state exists for every coastal combo.

Exists because a partially extracted combination fails SILENTLY at model time.
`load_model_data` skips a missing state file without comment, and `compute_risk`
resolves an unavailable state through ``data.coastal_slr.get(slr, hazards["coastal"])``
- so a combo holding only some of its states still runs, substituting present-day
coastal hazard for the missing horizons and understating the warming effect. The
guard inside `load_model_data` only fires below two states, which catches the
all-missing case and nothing else.

Run as a gate between SLR Stage 1 and Stage 2 (see submit_slr.sh), or directly:

    python -m src.check_slr_states              # report, exit 1 if incomplete
    python -m src.check_slr_states --write-list slr_retry.txt
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from .paths import load_config

# {country}_{asset}_coastal_profiles.parquet - the baseline extraction, whose
# presence is what makes a (country, asset) pair a coastal combo at all.
BASE_RE = re.compile(r"^([A-Z]{3})_(.+?)_coastal_profiles\.parquet$")


def expected_states(cfg: dict) -> list[str]:
    """Configured SLR states that need a file of their own.

    The state whose (time_horizon, climate_scenario) equals the baseline coastal
    setting is served by the baseline profiles, so no separate file is written
    for it and none should be demanded.
    """
    ccfg = cfg["hazards"].get("coastal") or {}
    base_pair = (str(ccfg.get("time_horizon")), str(ccfg.get("climate_scenario")))
    return [
        label for label, spec in (ccfg.get("slr_states") or {}).items()
        if (str(spec["time_horizon"]), str(spec["climate_scenario"])) != base_pair
    ]


def survey(cfg: dict) -> tuple[list[tuple[str, str]], list[str], dict]:
    """Return (coastal combos, expected states, {combo: [missing states]})."""
    inter = Path(cfg["intermediate_dir"])
    names = {p.name for p in inter.glob("*_coastal_*profiles.parquet")}
    combos = sorted({
        m.groups() for n in names if (m := BASE_RE.match(n))
    })
    states = expected_states(cfg)

    missing: dict[tuple[str, str], list[str]] = defaultdict(list)
    for country, asset in combos:
        for label in states:
            if f"{country}_{asset}_coastal_{label}_profiles.parquet" not in names:
                missing[(country, asset)].append(label)
    return combos, states, dict(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-list", default=None, metavar="PATH",
        help="write the incomplete combos as '<ISO3> <asset>' lines, for use as "
             "the Stage 1 retry list. Written even when nothing is missing, so a "
             "resubmission script always has a file to read.",
    )
    args = parser.parse_args()

    cfg = load_config()
    combos, states, missing = survey(cfg)
    n_missing = sum(len(v) for v in missing.values())

    print(f"coastal combos       : {len(combos)}")
    print(f"states needing files : {len(states)}")
    print(f"expected state files : {len(combos) * len(states)}")
    print(f"present              : {len(combos) * len(states) - n_missing}")
    print(f"missing              : {n_missing}  across {len(missing)} combos")

    if missing:
        print()
        for (country, asset), labels in sorted(
            missing.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            print(f"  {country} {asset:<12} missing {len(labels)}/{len(states)}")

    if args.write_list:
        out = Path(args.write_list)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            for country, asset in sorted(missing):
                fh.write(f"{country} {asset}\n")
        print(f"\nwrote {out} ({len(missing)} combos)")

    if missing:
        raise SystemExit(
            "\nIncomplete: Stage 2 is NOT safe to run. A combo with only some "
            "states still runs and silently substitutes present-day coastal "
            "hazard for the missing horizons."
        )
    print("\ncomplete - Stage 2 may proceed")


if __name__ == "__main__":
    main()
