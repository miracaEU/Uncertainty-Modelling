"""Run EMA Workbench experiments for one (country, asset, scenario) combination.

Usage:
    python -m src.run_experiments --country LUX --asset roads --scenario baseline --n 3000
    python -m src.run_experiments --country LUX --asset roads --scenario baseline \\
        --sampler sobol --n 512 --workers 8

Scenarios (src/ema_model.py::SCENARIOS): baseline, abs_protection,
flood_no_protection, earthquake_only.

With --sampler sobol, --n is the SALib base sample size N (use a power of 2);
the actual number of model runs is N * (2k + 2), k = number of uncertainty
factors (varies by scenario - see ema_model.py). Analyze Sobol results with
`python -m src.analyze_sobol`.

Results are saved as a tar.gz, named so that no (country, asset, scenario,
sampler) combination can ever overwrite another - re-running the same combo
adds a new timestamped file rather than replacing the old one.
"""

import argparse
from datetime import datetime

from ema_workbench import (
    MultiprocessingEvaluator,
    Samplers,
    SequentialEvaluator,
    ema_logging,
    save_results,
)

from .ema_model import SCENARIOS, build_model
from .paths import load_config, result_stem, set_asset_override, set_country_override, set_scenario_override


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000,
                        help="number of experiments (LHS) or base sample size N (Sobol)")
    parser.add_argument("--sampler", choices=["lhs", "sobol"], default="lhs")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--tag", type=str, default="", help="optional extra tag for the output filename")
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    parser.add_argument("--asset", default=None, help="asset type override (roads/airports/education/power)")
    parser.add_argument("--scenario", choices=SCENARIOS, default=None,
                        help=f"modeling scenario (default: baseline); one of {SCENARIOS}")
    args = parser.parse_args()

    set_country_override(args.country)
    set_asset_override(args.asset)
    set_scenario_override(args.scenario)
    ema_logging.log_to_stderr(ema_logging.INFO)
    cfg = load_config()
    print(f"Country: {cfg['country']}  Asset: {cfg['asset_type']}  Scenario: {cfg['scenario']}")
    model = build_model(cfg)

    kwargs = {"scenarios": args.n}
    if args.sampler == "sobol":
        if args.n & (args.n - 1) != 0:
            print(f"WARNING: Sobol base sample n={args.n} is not a power of 2; "
                  "SALib convergence properties are better with one.")
        k = len(model.uncertainties)
        print(f"Sobol/Saltelli: N={args.n}, k={k} -> {args.n * (2 * k + 2)} model runs")
        kwargs["uncertainty_sampling"] = Samplers.SOBOL

    if args.workers > 1:
        with MultiprocessingEvaluator(model, n_processes=args.workers) as evaluator:
            results = evaluator.perform_experiments(**kwargs)
    else:
        with SequentialEvaluator(model) as evaluator:
            results = evaluator.perform_experiments(**kwargs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = (
        cfg["results_dir"]
        / f"experiments_{result_stem(cfg)}_{args.sampler}_n{args.n}{tag}_{stamp}.tar.gz"
    )
    save_results(results, out)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
