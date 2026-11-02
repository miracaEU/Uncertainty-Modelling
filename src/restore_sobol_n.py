"""Put a combination's Sobol indices back to a chosen sample size.

A Sobol rerun at a higher N overwrites
{country}_{asset}_{scenario}_sobol_indices.csv in place - that is what makes
the new run the default, and it is what was asked for when the N=32768 reruns
were started. When the study later standardises on N=8192 instead, the handful
of combinations that finished at the higher N are the only ones left there, and
the workbook ends up mixing two sample sizes.

Nothing is lost in either direction: every rerun leaves its own
experiments_..._sobol_n<N>_<timestamp>.tar.gz behind, so the indices at any N
that was ever run can be recomputed with SALib rather than re-run on the
cluster. This script does that, and backs up whatever it replaces first.

    python -m src.restore_sobol_n                           # dry run (default)
    python -m src.restore_sobol_n --apply                   # rewrite to N=8192
    python -m src.restore_sobol_n --target-n 32768 --apply  # and back again

The backup directory mirrors results/: results_snapshot_n<N>/<COUNTRY>/<file>,
where <N> is the sample size of the values being displaced, so a restore in one
direction always leaves the other direction recoverable from disk.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_results import _analyze_archive, find_sobol_rerun_pairs
from .ema_model import build_model
from .paths import (PROJECT_ROOT, load_config, set_asset_override,
                    set_country_override, set_scenario_override)

CSV_COLUMNS = ["outcome", "factor", "S1", "S1_conf", "ST", "ST_conf", "interaction"]


def indices_path(results_dir: Path, country: str, asset: str, scenario: str) -> Path:
    return results_dir / country / f"{country}_{asset}_{scenario}_sobol_indices.csv"


def rebuild_from_archive(archive: Path, problem: dict, outcomes: list[str]) -> pd.DataFrame:
    """Recompute the per-combo indices table from one archive, all outcomes."""
    frames = []
    for outcome in outcomes:
        df = _analyze_archive(archive, problem, outcome)
        if df is None:
            # Zero-variance outcome: the original file records it as all zeros
            # rather than dropping the rows, so match that.
            df = pd.DataFrame({"factor_code": problem["names"], "S1": 0.0,
                               "S1_conf": 0.0, "ST": 0.0, "ST_conf": 0.0})
        out = pd.DataFrame({
            "outcome": outcome,
            "factor": df["factor_code"],
            "S1": df["S1"], "S1_conf": df["S1_conf"],
            "ST": df["ST"], "ST_conf": df["ST_conf"],
            "interaction": df["ST"] - df["S1"],
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)[CSV_COLUMNS]


def summarise(df: pd.DataFrame, outcome: str = "total_EAD_MEUR") -> tuple[str, float]:
    """Top factor and its ST for one outcome - the headline the workbook uses."""
    sub = df[df["outcome"] == outcome]
    if sub.empty or not np.isfinite(sub["ST"]).any():
        return ("-", float("nan"))
    row = sub.loc[sub["ST"].idxmax()]
    return (str(row["factor"]), float(row["ST"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-n", type=int, default=8192,
                    help="sample size the indices CSVs should end up at (default 8192)")
    ap.add_argument("--results-dir", default=None, help="override results/")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite. Without this the script only reports.")
    ap.add_argument("--backup-dir", default=None,
                    help="where displaced CSVs go (default results_snapshot_n<displaced N>/)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results"
    pairs = find_sobol_rerun_pairs(results_dir)
    if not pairs:
        raise SystemExit(f"No combination in {results_dir} has archives at more than one N.")

    usable = [p for p in pairs if args.target_n in (p["old_n"], p["new_n"])]
    skipped = [p for p in pairs if p not in usable]
    print(f"{len(pairs)} rerun combination(s); {len(usable)} have an N={args.target_n} archive")
    for p in skipped:
        print(f"  SKIP {p['country']} {p['asset']} {p['scenario']}: "
              f"archives at N={p['old_n']} and N={p['new_n']} only")
    print(f"mode: {'APPLY - files will be rewritten' if args.apply else 'DRY RUN - nothing written'}")
    print()

    header = f"{'country':5s} {'asset':10s} {'top factor now':34s} {'ST':>7s}   {'-> restored':34s} {'ST':>7s}"
    print(header)
    print("-" * len(header))

    changed = unchanged = failed = 0
    for p in pairs:
        if p not in usable:
            continue
        c, a, s = p["country"], p["asset"], p["scenario"]
        archive = p["old_path"] if p["old_n"] == args.target_n else p["new_path"]
        csv = indices_path(results_dir, c, a, s)
        if not csv.exists():
            print(f"{c:5s} {a:10s} !! no indices CSV at {csv.name}")
            failed += 1
            continue

        current = pd.read_csv(csv)
        outcomes = list(dict.fromkeys(current["outcome"]))

        set_country_override(c)
        set_asset_override(a)
        set_scenario_override(s)
        from ema_workbench.em_framework.salib_samplers import get_SALib_problem
        problem = get_SALib_problem(build_model(load_config()).uncertainties)

        try:
            restored = rebuild_from_archive(archive, problem, outcomes)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{c:5s} {a:10s} !! {type(exc).__name__}: {exc}")
            failed += 1
            continue

        cf, cst = summarise(current)
        rf, rst = summarise(restored)
        same = (len(current) == len(restored) and np.allclose(
            current[["S1", "ST"]].to_numpy(float),
            restored[["S1", "ST"]].to_numpy(float), rtol=1e-9, atol=1e-12))
        mark = "  (already at target)" if same else ""
        print(f"{c:5s} {a:10s} {cf[:34]:34s} {cst:7.4f}   {rf[:34]:34s} {rst:7.4f}{mark}")

        if same:
            unchanged += 1
            continue
        changed += 1

        if args.apply:
            displaced_n = p["new_n"] if archive is p["old_path"] else p["old_n"]
            bdir = (Path(args.backup_dir) if args.backup_dir
                    else PROJECT_ROOT / f"results_snapshot_n{displaced_n}") / c
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv, bdir / csv.name)
            restored.to_csv(csv, index=False)

    print()
    print(f"would change: {changed}   already at N={args.target_n}: {unchanged}   failed: {failed}"
          if not args.apply else
          f"changed: {changed}   already at N={args.target_n}: {unchanged}   failed: {failed}")
    if changed and not args.apply:
        print("\nRe-run with --apply to rewrite. The displaced CSVs are copied to")
        print("results_snapshot_n<displaced N>/ first, and every archive stays on disk,")
        print("so this is reversible with --target-n <the other N> --apply.")
    if changed and args.apply:
        print("\nNow re-aggregate so the workbook picks the restored values up:")
        print(f"    python -m src.aggregate_results --sample-size {args.target_n}")


if __name__ == "__main__":
    main()
