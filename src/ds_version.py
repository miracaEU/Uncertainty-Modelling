"""Which DamageScanner is installed, and which one each asset type must run with.

DamageScanner develop commit f6599e00 (2026-11-11) fixes two bugs that 1.0.1 has
(issues #82 and #83):

  #82  polygon area on geographic rasters: 1.0.1 gives every cell of a combination
       one area, overstating polygon exposure by about 1/cos(latitude) (1.22 for
       Cyprus to 2.33 for Iceland). The fix returns polygon coverage in m2 per cell
       and a cell area of 1.
  #83  overlapping raster tiles: cells on a tile edge were counted twice, which
       lengthens lines by 0.1-0.5%. Points are unaffected.

The study keeps each asset type on ONE version:

  roads, rail, telecom     (lines and points only)  ->  1.0.1, as every result so far
  airports, education, gas, healthcare, oil, ports, power  ->  the fix, pinned to
                           POLYGON_FIX_COMMIT, for river, earthquake and windstorm

The fix still reports version 1.0.1, so the version string cannot tell the two
apart. `installed()` checks behaviour instead (the fix adds the helpers
_geographic_cell_area_m2 and _iter_raster_tiles) and reads the installed commit
from the package's direct_url.json. `check()` refuses a run on the wrong one;
Stage 1 and src.validate call it before touching DamageScanner. Set
MIRACA_ALLOW_DS_MISMATCH=1 to override, for a deliberate test only.

    python -m src.ds_version                      # what is installed
    python -m src.ds_version --asset power        # would a power job be allowed?
    python -m src.ds_version --expect-fix         # exit 1 unless the pinned fix is installed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import lru_cache

POLYGON_FIX_COMMIT = "f6599e00d732c0b41858f7c12c07f4039ed5de80"
LINE_ONLY_ASSETS = frozenset({"roads", "rail", "telecom"})
POLYGON_FIX_ASSETS = frozenset({"airports", "education", "gas", "healthcare", "oil", "ports", "power"})
_FIX_HELPERS = ("_geographic_cell_area_m2", "_iter_raster_tiles")
OVERRIDE_ENV = "MIRACA_ALLOW_DS_MISMATCH"


@lru_cache(maxsize=1)
def installed() -> dict:
    """Version, source, commit and whether the polygon fix is present."""
    from importlib import metadata

    import damagescanner.vector as vector

    dist = metadata.distribution("damagescanner")
    commit, source = None, "index"
    raw = dist.read_text("direct_url.json")
    if raw:
        direct = json.loads(raw)
        source = direct.get("url", source)
        commit = (direct.get("vcs_info") or {}).get("commit_id")
        if commit is None:                       # an archive URL carries the hash in its path
            m = re.search(r"\b[0-9a-f]{40}\b", source)
            commit = m.group(0) if m else None
    return {
        "version": dist.version,
        "polygon_fix": all(hasattr(vector, h) for h in _FIX_HELPERS),
        "commit": commit,
        "source": source,
    }


def required_fix(asset: str) -> bool | None:
    """True if this asset type must run with the fix, False if with 1.0.1, None if unknown."""
    if asset in POLYGON_FIX_ASSETS:
        return True
    if asset in LINE_ONLY_ASSETS:
        return False
    return None


def problem(asset: str, info: dict | None = None) -> str | None:
    """Why `asset` may not run with the installed DamageScanner, or None if it may."""
    info = info or installed()
    need = required_fix(asset)
    if need is None:
        return f"asset type '{asset}' is not assigned to a DamageScanner version in src/ds_version.py"
    if need and not info["polygon_fix"]:
        return (f"{asset} has polygons and must run with the DamageScanner polygon fix "
                f"({POLYGON_FIX_COMMIT[:8]}), but the installed one ({info['version']}, {info['source']}) "
                f"lacks it. Point VENV at the fixed environment (see setup_dsfix_venv.sh).")
    if need and info["commit"] != POLYGON_FIX_COMMIT:
        return (f"{asset} must run with DamageScanner commit {POLYGON_FIX_COMMIT[:8]}, "
                f"but the installed one is commit {info['commit'] or 'unknown'} ({info['source']}). "
                f"Every polygon result has to come from the same build.")
    if not need and info["polygon_fix"]:
        return (f"{asset} has only lines and points and stays on DamageScanner 1.0.1, but the installed "
                f"one has the polygon fix ({info['commit'] or info['source']}). Use the 1.0.1 environment.")
    return None


def check(asset: str, what: str) -> dict:
    """Stop unless `asset` may run with the installed DamageScanner; return its provenance."""
    info = dict(installed())
    why = problem(asset, info)
    if why:
        if os.environ.get(OVERRIDE_ENV):
            print(f"  WARNING ({OVERRIDE_ENV} set, continuing): {why}")
            info["override"] = why
        else:
            raise SystemExit(f"{what} refused: {why}")
    print(f"  DamageScanner {info['version']}, polygon fix: {'yes' if info['polygon_fix'] else 'no'}"
          + (f", commit {info['commit'][:8]}" if info["commit"] else ""))
    return info


def check_cell_area(cell_area_m2: float | None, geographic: bool, has_polygons: bool) -> None:
    """The behavioural check: on a geographic raster with polygons, the fix returns a
    cell area of 1 (coverage already in m2) and 1.0.1 the area of one cell in m2.
    A mismatch means the import check above was fooled, so stop."""
    if cell_area_m2 is None or not (geographic and has_polygons):
        return
    fix = installed()["polygon_fix"]
    if fix and cell_area_m2 != 1:
        raise SystemExit(f"DamageScanner reports the polygon fix but returned a cell area of {cell_area_m2:g} m2 "
                         f"on a geographic raster; the fix returns 1. Check the installed package.")
    if not fix and cell_area_m2 == 1:
        raise SystemExit("DamageScanner lacks the polygon fix helpers but returned a cell area of 1 on a "
                         "geographic raster, as only the fix does. Check the installed package.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", help="report whether a job for this asset type would be allowed")
    ap.add_argument("--expect-fix", action="store_true",
                    help="exit 1 unless the pinned polygon fix is installed")
    ap.add_argument("--expect-old", action="store_true",
                    help="exit 1 unless DamageScanner without the polygon fix is installed")
    args = ap.parse_args()
    info = installed()
    print(json.dumps(info, indent=2))
    bad = []
    if args.expect_fix and not (info["polygon_fix"] and info["commit"] == POLYGON_FIX_COMMIT):
        bad.append(f"expected the polygon fix at {POLYGON_FIX_COMMIT[:8]}")
    if args.expect_old and info["polygon_fix"]:
        bad.append("expected DamageScanner without the polygon fix")
    if args.asset:
        why = problem(args.asset, info)
        print(f"{args.asset}: " + ("allowed" if why is None else f"refused - {why}"))
        if why:
            bad.append(why)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
