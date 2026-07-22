#!/usr/bin/env python
"""Orchestrates the full multi-country, multi-asset, multi-scenario study.

For every (asset, country) pair, in order:
  1. Stage 1 once:  preprocess (both hazards) + validate.
  2. For every scenario (baseline, abs_protection, flood_no_protection,
     earthquake_only): an LHS run + a Sobol run, each followed by its
     analysis script.
  3. Once every combination is done (or the run is aborted/fails), the
     aggregated summary workbook (MIRACA_uncertainty_study_summary.xlsx, in
     the project root - see src/aggregate_results.py) is regenerated from
     whatever results/*.csv files exist at that point - so it's always
     current after any run of this script, full or partial.

Default order matches the study design: roads for LUX then DNK, fully
through all four scenarios, before moving on to airports, then education,
then power, each again LUX then DNK. Override with --assets/--countries/
--scenarios to run a subset (e.g. one array task on a cluster).

Nothing is ever overwritten:
  - Stage-1 outputs and experiment archives are skipped (not regenerated) if
    they already exist for that exact combination - safe to re-run this
    script after an interruption; already-done work is not repeated.
  - Every rerun of an experiment archive gets a fresh timestamp, so even
    with --force old archives are never deleted, only added to.
  - analyze/analyze_sobol always (re)run - they are cheap, deterministic
    given the archive they read, and their prefixed CSV/figure outputs are
    just a re-derivable summary of already-protected raw data.

Multi-processor / cluster use:
  --workers N is passed straight through to every run_experiments.py call
  (ema_workbench.MultiprocessingEvaluator). Each stage is a fresh Python
  subprocess (via `--python -m src.<stage>`), so a crash in one combination
  cannot corrupt or hang a later one; by default the script logs the failure
  and continues (see --fail-fast to abort instead). A JSONL run log is kept
  at results/run_study_log.jsonl so progress survives interruption/preemption
  and can be inspected without re-parsing stdout.

Usage:
    python run_study.py --dry-run                          # preview the plan
    python run_study.py --workers 8                         # full study
    python run_study.py --assets roads --countries LUX --workers 8
    python run_study.py --assets roads --countries LUX --scenarios baseline \\
        --workers 16                                        # one SLURM array task
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.ema_model import SCENARIOS  # noqa: E402
from src.paths import base_stem, load_config, set_asset_override, set_country_override, set_scenario_override  # noqa: E402

DEFAULT_ASSETS = ["roads", "airports", "education", "power"]
DEFAULT_COUNTRIES = ["LUX", "DNK"]


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


class StudyLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **fields) -> None:
        fields["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields) + "\n")


def run_step(python: str, module: str, extra_args: list[str], dry_run: bool) -> tuple[bool, float]:
    cmd = [python, "-m", module] + extra_args
    log("  $ " + " ".join(cmd))
    if dry_run:
        return True, 0.0
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def stage1_done(cfg: dict) -> bool:
    stem = base_stem(cfg)
    seg_path = cfg["intermediate_dir"] / f"{stem}_segments.parquet"
    if not seg_path.exists():
        return False
    for hazard in cfg["hazards"]:
        prof_path = cfg["intermediate_dir"] / f"{stem}_{hazard}_profiles.parquet"
        if not prof_path.exists():
            return False
    return True


def validated_marker(cfg: dict) -> Path:
    return cfg["intermediate_dir"] / f"{base_stem(cfg)}_validated.ok"


def experiments_exist(cfg: dict, scenario: str, sampler: str, n: int) -> bool:
    from src.paths import result_stem

    set_scenario_override(scenario)
    cfg2 = load_config()
    pattern = f"experiments_{result_stem(cfg2)}_{sampler}_n{n}_*.tar.gz"
    return any(cfg["results_dir"].glob(pattern))


def analysis_exists(cfg: dict, scenario: str, kind: str) -> bool:
    from src.paths import result_stem

    set_scenario_override(scenario)
    cfg2 = load_config()
    prefix = result_stem(cfg2)
    name = f"{prefix}_feature_scores.csv" if kind == "lhs" else f"{prefix}_sobol_indices.csv"
    return (cfg["results_dir"] / name).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, choices=DEFAULT_ASSETS)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--lhs-n", type=int, default=3000, help="LHS experiment count")
    parser.add_argument("--sobol-n", type=int, default=512, help="Sobol base sample size N (use a power of 2)")
    parser.add_argument("--workers", type=int, default=4, help="parallel worker processes per experiment run")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to invoke for every stage")
    parser.add_argument("--force", action="store_true", help="re-run steps even if their output already exists")
    parser.add_argument("--fail-fast", action="store_true", help="abort the whole study on the first failed step")
    parser.add_argument("--dry-run", action="store_true", help="print the planned steps without running them")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    study_log = StudyLog(root / "results" / "run_study_log.jsonl")

    plan = [(a, c) for a in args.assets for c in args.countries]
    log(f"Plan: {len(plan)} (asset, country) pairs x {len(args.scenarios)} scenarios")
    for a, c in plan:
        log(f"  {a} / {c}")

    failures: list[str] = []

    for asset, country in plan:
        set_country_override(country)
        set_asset_override(asset)
        set_scenario_override(None)  # scenario-independent for Stage 1
        cfg = load_config()
        combo = f"{country}/{asset}"
        log("=" * 70)
        log(f"{combo}: Stage 1 (preprocess + validate)")
        log("=" * 70)

        if not args.force and stage1_done(cfg):
            log(f"  preprocess: SKIP (already done)")
        else:
            ok, elapsed = run_step(
                args.python, "src.preprocess",
                ["--country", country, "--asset", asset],
                args.dry_run,
            )
            study_log.record(combo=combo, step="preprocess", ok=ok, elapsed_s=round(elapsed, 1))
            if not ok:
                log(f"  preprocess: FAILED after {elapsed:.0f}s")
                failures.append(f"{combo} preprocess")
                if args.fail_fast:
                    break
                continue
            log(f"  preprocess: done in {elapsed:.0f}s")

        marker = validated_marker(cfg)
        if not args.force and marker.exists():
            log(f"  validate: SKIP (already validated)")
        else:
            ok, elapsed = run_step(
                args.python, "src.validate",
                ["--country", country, "--asset", asset],
                args.dry_run,
            )
            study_log.record(combo=combo, step="validate", ok=ok, elapsed_s=round(elapsed, 1))
            if not ok:
                log(f"  validate: FAILED after {elapsed:.0f}s - continuing to experiments anyway")
                failures.append(f"{combo} validate")
                if args.fail_fast:
                    break
            else:
                log(f"  validate: PASSED in {elapsed:.0f}s")
                if not args.dry_run:
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text(datetime.now(timezone.utc).isoformat())

        for scenario in args.scenarios:
            log("-" * 70)
            log(f"{combo} / {scenario}")
            log("-" * 70)

            for sampler, n in (("lhs", args.lhs_n), ("sobol", args.sobol_n)):
                if not args.force and experiments_exist(cfg, scenario, sampler, n):
                    log(f"  {sampler} n={n}: SKIP (archive already exists)")
                else:
                    extra = [
                        "--country", country, "--asset", asset, "--scenario", scenario,
                        "--n", str(n), "--workers", str(args.workers),
                    ]
                    if sampler == "sobol":
                        extra += ["--sampler", "sobol"]
                    ok, elapsed = run_step(args.python, "src.run_experiments", extra, args.dry_run)
                    study_log.record(
                        combo=combo, step=f"run_experiments_{sampler}", scenario=scenario,
                        n=n, ok=ok, elapsed_s=round(elapsed, 1),
                    )
                    if not ok:
                        log(f"  {sampler} n={n}: FAILED after {elapsed:.0f}s")
                        failures.append(f"{combo}/{scenario} {sampler}")
                        if args.fail_fast:
                            log("Aborting (--fail-fast).")
                            _finalize(args, failures)
                        continue
                    log(f"  {sampler} n={n}: done in {elapsed:.0f}s")

                analyze_module = "src.analyze" if sampler == "lhs" else "src.analyze_sobol"
                ok, elapsed = run_step(
                    args.python, analyze_module,
                    ["--country", country, "--asset", asset, "--scenario", scenario],
                    args.dry_run,
                )
                study_log.record(
                    combo=combo, step=f"analyze_{sampler}", scenario=scenario,
                    ok=ok, elapsed_s=round(elapsed, 1),
                )
                if not ok:
                    log(f"  analyze ({sampler}): FAILED")
                    failures.append(f"{combo}/{scenario} analyze_{sampler}")
                    if args.fail_fast:
                        log("Aborting (--fail-fast).")
                        _finalize(args, failures)

    _finalize(args, failures)


def _finalize(args: argparse.Namespace, failures: list[str]) -> None:
    """Regenerate the aggregated summary workbook, print the run summary, exit.

    Runs regardless of failures, so the workbook always reflects whatever
    results actually exist - and is the single exit point for the whole
    script, so a --fail-fast abort still leaves the summary up to date.
    """
    log("=" * 70)
    log("Regenerating aggregated summary workbook...")
    ok, elapsed = run_step(args.python, "src.aggregate_results", [], args.dry_run)
    if ok:
        log(f"  aggregate_results: done in {elapsed:.0f}s")
    else:
        log("  aggregate_results: FAILED (raw results are unaffected, just the summary workbook)")
        failures.append("aggregate_results")

    log("=" * 70)
    if failures:
        log(f"STUDY COMPLETE WITH {len(failures)} FAILURE(S):")
        for f in failures:
            log(f"  - {f}")
    else:
        log("STUDY COMPLETE - all steps succeeded.")
    log("=" * 70)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
