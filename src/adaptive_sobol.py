"""Adaptive Sobol sampling: keep doubling N until the estimate is precise enough.

Instead of a single fixed Sobol base sample size N, this runs a sequence
N = 128, 256, 512, ... up to a cap (default 8192), and stops as soon as the
result is "precise enough" on the headline outcome (total_EAD_MEUR) - or when
the cap is reached, whichever comes first. This spends cluster time only where
it's needed: a combination whose driver ranking is already crisp at N=128
stops immediately, while a genuinely noisy one runs up to the cap.

Convergence criterion (per Saltelli round, on total_EAD_MEUR):
  Among factors that actually matter (total-effect ST > RELEVANCE), take the
  worst confidence-to-estimate ratio max(ST_conf / ST). Stop when it drops
  below --threshold (default 0.2, i.e. every relevant factor's 95% CI half-
  width is within 20% of its own estimate). If NO factor is relevant (e.g. a
  hazard with no exposure -> zero-variance outcome), there is nothing to
  resolve and it stops at the first N.

Nothing is overwritten: every round's Saltelli sample is saved as its own
timestamped experiments_..._sobol_n{N}_*.tar.gz (same naming as
run_experiments.py, so src.analyze_sobol and the aggregation pick up the
highest-N archive automatically). A per-combination record of the whole
search - every N tried, its worst ratio, the top factor, and why it stopped -
is appended to results/sobol_convergence_log.jsonl, which the summary workbook
turns into a Sobol_Convergence sheet.

Resumable: if an archive for a given N already exists (and --force is not
set) it is loaded instead of re-run, so an interrupted search continues
without repeating completed rounds.

Usage:
    python -m src.adaptive_sobol --country LUX --asset power --scenario flood_baseline \\
        --workers 8
    python -m src.adaptive_sobol --country LUX --asset roads --scenario earthquake \\
        --min-n 128 --max-n 8192 --threshold 0.2 --workers 16
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ema_workbench import (
    MultiprocessingEvaluator,
    Samplers,
    SequentialEvaluator,
    ema_logging,
    load_results,
    save_results,
)
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol

from .ema_model import SCENARIOS, build_model
from .paths import (
    country_results_dir,
    load_config,
    result_stem,
    set_asset_override,
    set_country_override,
    set_scenario_override,
)

DEFAULT_MIN_N = 128
DEFAULT_MAX_N = 8192
DEFAULT_THRESHOLD = 0.2   # max ST_conf/ST among relevant factors to accept
RELEVANCE = 0.05          # ST above this = "matters", so its precision is tracked
CONVERGENCE_OUTCOME = "total_EAD_MEUR"
CONVERGENCE_LOG = "sobol_convergence_log.jsonl"


def _powers_of_two(min_n: int, max_n: int) -> list[int]:
    ns, n = [], min_n
    while n <= max_n:
        ns.append(n)
        n *= 2
    if not ns:
        ns = [max_n]
    return ns


def _existing_archive(cfg: dict, n: int) -> Path | None:
    pattern = f"experiments_{result_stem(cfg)}_sobol_n{n}_*.tar.gz"
    files = sorted(country_results_dir(cfg).glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _run_or_load(cfg: dict, model, n: int, workers: int, force: bool):
    """Return (experiments, outcomes) for base sample N, running if needed."""
    if not force:
        existing = _existing_archive(cfg, n)
        if existing is not None:
            print(f"  N={n}: loading existing archive {existing.name}")
            return load_results(str(existing))

    kwargs = {"scenarios": n, "uncertainty_sampling": Samplers.SOBOL}
    if workers > 1:
        with MultiprocessingEvaluator(model, n_processes=workers) as evaluator:
            results = evaluator.perform_experiments(**kwargs)
    else:
        with SequentialEvaluator(model) as evaluator:
            results = evaluator.perform_experiments(**kwargs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = country_results_dir(cfg) / f"experiments_{result_stem(cfg)}_sobol_n{n}_{stamp}.tar.gz"
    save_results(results, out)
    print(f"  N={n}: saved {out.name}")
    return results


def _worst_ratio(problem: dict, y: np.ndarray) -> tuple[float | None, str | None, int]:
    """Return (worst ST_conf/ST among relevant factors, top factor, n_relevant).

    worst ratio is None when the outcome has (near-)zero variance or no factor
    clears the relevance floor - i.e. nothing to resolve.
    """
    if np.var(y) < 1e-30:
        return None, None, 0
    si = sobol.analyze(problem, y, calc_second_order=True, print_to_console=False)
    st = np.nan_to_num(np.asarray(si["ST"]), nan=0.0)
    stc = np.nan_to_num(np.asarray(si["ST_conf"]), nan=0.0)
    names = np.asarray(problem["names"])
    top_factor = str(names[int(np.argmax(st))])
    relevant = st > RELEVANCE
    if not relevant.any():
        return None, top_factor, 0
    ratios = stc[relevant] / st[relevant]
    return float(np.max(ratios)), top_factor, int(relevant.sum())


def run_adaptive(
    cfg: dict, workers: int, min_n: int, max_n: int, threshold: float, force: bool
) -> dict:
    model = build_model(cfg)
    problem = get_SALib_problem(model.uncertainties)
    k = problem["num_vars"]
    combo = f"{cfg['country']}/{cfg['asset_type']}/{cfg['scenario']}"
    print(f"Adaptive Sobol: {combo}  (k={k} factors, N {min_n}..{max_n}, threshold {threshold})")

    rounds = []
    stop_n, stop_reason = None, None
    t0 = time.time()

    for n in _powers_of_two(min_n, max_n):
        experiments, outcomes = _run_or_load(cfg, model, n, workers, force)
        y = np.asarray(outcomes[CONVERGENCE_OUTCOME], dtype=float)
        worst, top_factor, n_relevant = _worst_ratio(problem, y)
        rounds.append(
            {
                "n": n,
                "n_runs": len(experiments),
                "worst_ratio": worst,
                "top_factor": top_factor,
                "n_relevant": n_relevant,
            }
        )
        ratio_str = "n/a" if worst is None else f"{worst:.3f}"
        print(f"  N={n}: worst ST_conf/ST = {ratio_str}  (relevant factors: {n_relevant}, top: {top_factor})")

        if worst is None:
            stop_n, stop_reason = n, "no_relevant_factors"
            break
        if worst < threshold:
            stop_n, stop_reason = n, "converged"
            break
        if n >= max_n:
            stop_n, stop_reason = n, "max_n_reached"
            break

    elapsed = round(time.time() - t0, 1)
    record = {
        "country": cfg["country"],
        "asset": cfg["asset_type"],
        "scenario": cfg["scenario"],
        "outcome": CONVERGENCE_OUTCOME,
        "threshold": threshold,
        "min_n": min_n,
        "max_n": max_n,
        "stop_n": stop_n,
        "stop_reason": stop_reason,
        "n_factors": k,
        "rounds": rounds,
        "elapsed_s": elapsed,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    log_path = cfg["results_dir"] / CONVERGENCE_LOG
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  -> stopped at N={stop_n} ({stop_reason}) in {elapsed}s; logged to {log_path.name}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    parser.add_argument("--asset", default=None, help="asset type override")
    parser.add_argument("--scenario", choices=SCENARIOS, default=None)
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="starting Sobol base N (power of 2)")
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N, help="maximum Sobol base N (power of 2)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="stop once max ST_conf/ST among relevant factors < this")
    parser.add_argument("--force", action="store_true", help="re-run rounds even if an archive for that N exists")
    args = parser.parse_args()

    set_country_override(args.country)
    set_asset_override(args.asset)
    set_scenario_override(args.scenario)
    ema_logging.log_to_stderr(ema_logging.INFO)
    cfg = load_config()
    run_adaptive(cfg, args.workers, args.min_n, args.max_n, args.threshold, args.force)


if __name__ == "__main__":
    main()
