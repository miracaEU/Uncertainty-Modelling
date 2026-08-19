#!/usr/bin/env python
"""Orchestrates the full multi-country, multi-asset, multi-scenario study.

For every (asset, country) pair, in order:
  1. Stage 1 once:  preprocess (every hazard applicable to the asset) + validate.
  2. For every scenario applicable to that asset (see src/ema_model.py;
     windstorm is skipped for roads/ports and coastal for landlocked
     countries automatically): an LHS run (fixed N) followed by src.analyze,
     then a Sobol run followed by src.analyze_sobol. Sobol uses a fixed
     base sample size N (--sobol-n, default 8192) for every combination by
     default, for consistency; pass --adaptive to instead double N from
     --sobol-min-n up to --sobol-max-n, stopping early once the
     confidence-interval criterion (--sobol-threshold) is met.
  3. Once every combination is done (or the run is aborted/fails), the
     aggregated summary workbook (MIRACA_uncertainty_study_summary.xlsx, in
     the project root - see src/aggregate_results.py) is regenerated from
     whatever results/*.csv files exist at that point - so it's always
     current after any run of this script, full or partial.

Default order matches the study design: roads for LUX/DNK/GRC/PRT, through
every applicable scenario, before moving on to airports, then education,
then power. Override with --assets/--countries/--scenarios to run a subset
(e.g. one array task on a cluster). Non-applicable (asset, scenario) pairs
are skipped automatically.

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
  --workers N is passed straight through to every experiment run
  (ema_workbench.MultiprocessingEvaluator). Each stage is a fresh Python
  subprocess (via `--python -m src.<stage>`), so a crash in one combination
  cannot corrupt or hang a later one; by default the script logs the failure
  and continues (see --fail-fast to abort instead). A JSONL run log is kept
  at results/run_study_log.jsonl (per-step timing) and a Sobol convergence log
  at results/sobol_convergence_log.jsonl (the N sequence each combination
  stopped at) so progress survives interruption/preemption and can be
  inspected without re-parsing stdout.

Usage:
    python run_study.py --dry-run                          # preview the plan
    python run_study.py --workers 8                         # full study (LUX/DNK/GRC/PRT)
    python run_study.py --assets power --countries LUX --workers 8
    python run_study.py --assets power --countries LUX --scenarios windstorm \\
        --workers 16                                        # one SLURM array task
    python run_study.py --sobol-n 8192                       # fixed N=8192 (default)
    python run_study.py --adaptive --sobol-min-n 128 --sobol-max-n 8192 --sobol-threshold 0.2
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.curves import ASSET_CONFIGS, applicable_hazards  # noqa: E402
from src.ema_model import DEFAULT_SCENARIOS, SCENARIOS, scenario_applies  # noqa: E402
from src.paths import base_stem, load_config, set_asset_override, set_country_override, set_scenario_override  # noqa: E402

# All registered asset classes, so a no-argument run covers every asset the
# pipeline supports (src/curves.py::ASSET_CONFIGS) - stays complete if more
# are added later.
DEFAULT_ASSETS = list(ASSET_CONFIGS)
DEFAULT_COUNTRIES = ["LUX", "DNK", "GRC", "PRT"]


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
    # Only the hazards that actually apply to this asset/country are extracted
    # (windstorm skipped for roads, coastal for landlocked countries), so only
    # those profiles are required for Stage 1 to count as done.
    applicable = [h for h in applicable_hazards(cfg["asset_type"], cfg["country"]) if h in cfg["hazards"]]
    for hazard in applicable:
        prof_path = cfg["intermediate_dir"] / f"{stem}_{hazard}_profiles.parquet"
        if not prof_path.exists():
            return False
    return True


def validated_marker(cfg: dict) -> Path:
    return cfg["intermediate_dir"] / f"{base_stem(cfg)}_validated.ok"


def experiments_exist(cfg: dict, scenario: str, sampler: str, n: int) -> bool:
    from src.paths import country_results_dir, result_stem

    set_scenario_override(scenario)
    cfg2 = load_config()
    pattern = f"experiments_{result_stem(cfg2)}_{sampler}_n{n}_*.tar.gz"
    return any(country_results_dir(cfg2, create=False).glob(pattern))


def analysis_exists(cfg: dict, scenario: str, kind: str) -> bool:
    from src.paths import country_results_dir, result_stem

    set_scenario_override(scenario)
    cfg2 = load_config()
    prefix = result_stem(cfg2)
    name = f"{prefix}_feature_scores.csv" if kind == "lhs" else f"{prefix}_sobol_indices.csv"
    return (country_results_dir(cfg2, create=False) / name).exists()


def adaptive_done(cfg: dict, scenario: str) -> bool:
    """True if results/sobol_convergence_log.jsonl already has a record for
    this (country, asset, scenario) - i.e. the adaptive Sobol search ran to a
    stop before, so it can be skipped on a resumed study (unless --force)."""
    log_path = cfg["results_dir"] / "sobol_convergence_log.jsonl"
    if not log_path.exists():
        return False
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("country") == cfg["country"]
                and rec.get("asset") == cfg["asset_type"]
                and rec.get("scenario") == scenario
                and rec.get("stop_n") is not None
            ):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, choices=DEFAULT_ASSETS)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS, choices=SCENARIOS,
                        help="scenarios to run (default: the absolute-protection + multiplicative-depth "
                             "flood/coastal variants, earthquake, windstorm; pass names to run others, "
                             "or all 14 explicitly)")
    parser.add_argument("--lhs-n", type=int, default=3000, help="LHS experiment count")
    parser.add_argument("--sobol-n", type=int, default=8192,
                        help="fixed Sobol base sample size N used for EVERY combination (default; power of 2). "
                             "Ignored when --adaptive is set.")
    parser.add_argument("--adaptive", action="store_true",
                        help="instead of a fixed N, adaptively double N from --sobol-min-n up to --sobol-max-n, "
                             "stopping once --sobol-threshold is met (variable N per combination)")
    parser.add_argument("--sobol-min-n", type=int, default=128, help="adaptive mode: starting base N (power of 2)")
    parser.add_argument("--sobol-max-n", type=int, default=8192, help="adaptive mode: maximum base N (power of 2)")
    parser.add_argument("--sobol-threshold", type=float, default=0.2,
                        help="adaptive mode: stop once max ST_conf/ST among relevant factors < this")
    parser.add_argument("--workers", type=int, default=4, help="parallel worker processes per experiment run")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to invoke for every stage")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="skip regenerating the aggregated workbook at the end. Use this when many "
                             "per-combination jobs run in PARALLEL (e.g. one SLURM job per country/asset): "
                             "otherwise every job rewrites the same .xlsx concurrently. Run "
                             "`python -m src.aggregate_results` once after they all finish instead.")
    parser.add_argument("--force", action="store_true", help="re-run steps even if their output already exists")
    parser.add_argument("--force-scenarios", action="store_true",
                        help="re-run the Stage-2 scenario steps (LHS/Sobol/analysis) even if their archives "
                             "exist, but LEAVE Stage 1 alone if it is already done. Use to recompute selected "
                             "--scenarios (e.g. coastal after a return-period/config change) without repeating "
                             "the expensive GIS overlay. analyze_sobol reads the newest archive, so the fresh "
                             "run supersedes the old one.")
    parser.add_argument("--fail-fast", action="store_true", help="abort the whole study on the first failed step")
    parser.add_argument("--dry-run", action="store_true", help="print the planned steps without running them")
    args = parser.parse_args()
    # Stage-2 scenario steps re-run when either --force (everything) or
    # --force-scenarios (only the scenario steps, Stage 1 left intact) is set.
    force_scen = args.force or args.force_scenarios

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

        scenarios_here = [s for s in args.scenarios if scenario_applies(s, asset, country)]
        skipped_scen = [s for s in args.scenarios if s not in scenarios_here]
        if skipped_scen:
            log(f"  (scenarios not applicable to {country}/{asset}: {skipped_scen})")

        for scenario in scenarios_here:
            log("-" * 70)
            log(f"{combo} / {scenario}")
            log("-" * 70)

            # 1. LHS (fixed N) - quick extra-trees importance complement.
            if not force_scen and experiments_exist(cfg, scenario, "lhs", args.lhs_n):
                log(f"  lhs n={args.lhs_n}: SKIP (archive already exists)")
            else:
                extra = [
                    "--country", country, "--asset", asset, "--scenario", scenario,
                    "--n", str(args.lhs_n), "--workers", str(args.workers),
                ]
                ok, elapsed = run_step(args.python, "src.run_experiments", extra, args.dry_run)
                study_log.record(
                    combo=combo, step="run_experiments_lhs", scenario=scenario,
                    n=args.lhs_n, ok=ok, elapsed_s=round(elapsed, 1),
                )
                if not ok:
                    log(f"  lhs n={args.lhs_n}: FAILED after {elapsed:.0f}s")
                    failures.append(f"{combo}/{scenario} lhs")
                    if args.fail_fast:
                        log("Aborting (--fail-fast).")
                        _finalize(args, failures)
                else:
                    log(f"  lhs n={args.lhs_n}: done in {elapsed:.0f}s")

            ok, elapsed = run_step(
                args.python, "src.analyze",
                ["--country", country, "--asset", asset, "--scenario", scenario],
                args.dry_run,
            )
            study_log.record(
                combo=combo, step="analyze_lhs", scenario=scenario,
                ok=ok, elapsed_s=round(elapsed, 1),
            )
            if not ok:
                log("  analyze (lhs): FAILED")
                failures.append(f"{combo}/{scenario} analyze_lhs")
                if args.fail_fast:
                    log("Aborting (--fail-fast).")
                    _finalize(args, failures)

            # 2. Sobol. By default a fixed N (--sobol-n, 8192) for every
            #    combination, for consistency. With --adaptive, N is instead
            #    doubled from --sobol-min-n up to --sobol-max-n, stopping once
            #    the CI criterion is met. Both go through src.adaptive_sobol
            #    (fixed = a min==max single round); it writes each N's archive
            #    plus a convergence-log record, then we analyze the highest-N
            #    archive it produced.
            if args.adaptive:
                sobol_min, sobol_max = args.sobol_min_n, args.sobol_max_n
                mode_label = "adaptive"
            else:
                sobol_min = sobol_max = args.sobol_n
                mode_label = f"fixed N={args.sobol_n}"

            if not force_scen and adaptive_done(cfg, scenario):
                log(f"  sobol ({mode_label}): SKIP (convergence record already exists)")
            else:
                extra = [
                    "--country", country, "--asset", asset, "--scenario", scenario,
                    "--min-n", str(sobol_min), "--max-n", str(sobol_max),
                    "--threshold", str(args.sobol_threshold), "--workers", str(args.workers),
                ]
                if force_scen:
                    extra.append("--force")
                ok, elapsed = run_step(args.python, "src.adaptive_sobol", extra, args.dry_run)
                study_log.record(
                    combo=combo, step="adaptive_sobol", scenario=scenario,
                    ok=ok, elapsed_s=round(elapsed, 1),
                )
                if not ok:
                    log(f"  sobol ({mode_label}): FAILED after {elapsed:.0f}s")
                    failures.append(f"{combo}/{scenario} adaptive_sobol")
                    if args.fail_fast:
                        log("Aborting (--fail-fast).")
                        _finalize(args, failures)
                    continue
                log(f"  sobol ({mode_label}): done in {elapsed:.0f}s")

            ok, elapsed = run_step(
                args.python, "src.analyze_sobol",
                ["--country", country, "--asset", asset, "--scenario", scenario],
                args.dry_run,
            )
            study_log.record(
                combo=combo, step="analyze_sobol", scenario=scenario,
                ok=ok, elapsed_s=round(elapsed, 1),
            )
            if not ok:
                log("  analyze (sobol): FAILED")
                failures.append(f"{combo}/{scenario} analyze_sobol")
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
    if getattr(args, "no_aggregate", False):
        log("Skipping aggregated workbook (--no-aggregate); run "
            "`python -m src.aggregate_results` once all jobs have finished.")
    else:
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
