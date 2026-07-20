"""Run EMA Workbench experiments for the multi-hazard risk model.

Usage:
    python -m src.run_experiments --n 1000                      # LHS
    python -m src.run_experiments --sampler sobol --n 512       # Saltelli/Sobol
    python -m src.run_experiments --n 5000 --workers 4

With --sampler sobol, --n is the SALib base sample size N (use a power of 2);
the actual number of model runs is N * (2k + 2) with k = 9 factors, i.e.
20 N runs. Analyze Sobol results with `python -m src.analyze_sobol`.

Results are saved as a tar.gz readable with ema_workbench.load_results().
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

from .ema_model import build_model
from .paths import load_config, set_country_override


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000,
                        help="number of experiments (LHS) or base sample size N (Sobol)")
    parser.add_argument("--sampler", choices=["lhs", "sobol"], default="lhs")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--tag", type=str, default="", help="optional tag for the output filename")
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    args = parser.parse_args()

    set_country_override(args.country)
    ema_logging.log_to_stderr(ema_logging.INFO)
    cfg = load_config()
    print(f"Country: {cfg['country']}")
    model = build_model()

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
        / f"experiments_{cfg['country']}_{cfg['asset_type']}_{args.sampler}_n{args.n}{tag}_{stamp}.tar.gz"
    )
    save_results(results, out)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
