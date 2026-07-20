"""Sobol variance decomposition of experiment results.

Expects results produced with `python -m src.run_experiments --sampler sobol`
(Saltelli sample, second order enabled). Computes first-order (S1) and total
(ST) Sobol indices per uncertainty factor for the main outcomes.

Produces (in results/):
  sobol_indices.csv          — S1/ST (+95% conf) per factor per outcome
  figures/sobol_indices.png  — grouped bar chart per outcome

Usage:
    python -m src.analyze_sobol                 # newest *_sobol_* results file
    python -m src.analyze_sobol --results results/experiments_..._sobol_...tar.gz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from ema_workbench import load_results
from ema_workbench.em_framework.salib_samplers import get_SALib_problem

from .ema_model import build_model
from .paths import load_config, set_country_override

# palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"    # ST (total effect)
GREEN = "#008300"   # S1 (first order)

SOBOL_OUTCOMES = [
    "total_EAD_MEUR",
    "EAD_river_MEUR",
    "EAD_earthquake_MEUR",
    "damage_RP100_river_MEUR",
]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "font.size": 10,
    }
)


def newest_sobol_results(cfg: dict) -> Path:
    pattern = f"experiments_{cfg['country']}_{cfg['asset_type']}_sobol_*.tar.gz"
    files = sorted(cfg["results_dir"].glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            f"No {pattern} in {cfg['results_dir']}. "
            "Run: python -m src.run_experiments --sampler sobol --n 512"
        )
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    args = parser.parse_args()

    set_country_override(args.country)
    cfg = load_config()
    prefix = cfg["country"]
    path = Path(args.results) if args.results else newest_sobol_results(cfg)
    print(f"Loading {path}")
    experiments, outcomes = load_results(path)

    model = build_model()
    problem = get_SALib_problem(model.uncertainties)
    k = problem["num_vars"]
    n_runs = len(experiments)
    if n_runs % (2 * k + 2) != 0:
        raise SystemExit(
            f"{n_runs} runs is not a multiple of 2k+2={2 * k + 2}: "
            "this results file was not produced with --sampler sobol."
        )
    print(f"{n_runs} runs, k={k} factors, base N={n_runs // (2 * k + 2)}")

    rows = []
    indices = {}
    for outcome in SOBOL_OUTCOMES:
        y = np.asarray(outcomes[outcome], dtype=float)
        if np.var(y) < 1e-30:
            print(f"  NOTE: {outcome} has (near-)zero variance; indices set to 0.")
            zeros = np.zeros(k)
            indices[outcome] = {"S1": zeros, "S1_conf": zeros, "ST": zeros, "ST_conf": zeros}
        else:
            si = sobol.analyze(problem, y, calc_second_order=True, print_to_console=False)
            # guard against NaN from degenerate outcomes
            si = {key: np.nan_to_num(np.asarray(si[key]), nan=0.0)
                  for key in ("S1", "S1_conf", "ST", "ST_conf")}
            indices[outcome] = si
        si = indices[outcome]
        for name, s1, s1c, st, stc in zip(
            problem["names"], si["S1"], si["S1_conf"], si["ST"], si["ST_conf"]
        ):
            rows.append(
                {
                    "outcome": outcome,
                    "factor": name,
                    "S1": s1,
                    "S1_conf": s1c,
                    "ST": st,
                    "ST_conf": stc,
                    "interaction": st - s1,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = cfg["results_dir"] / f"{prefix}_sobol_indices.csv"
    df.to_csv(csv_path, index=False)

    # --- figure: one panel per outcome, grouped horizontal bars (ST + S1) ---
    names = problem["names"]
    order = np.argsort(indices["total_EAD_MEUR"]["ST"])  # rank by total-EAD ST
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5), layout="constrained", sharey=True
    )
    ypos = np.arange(len(names))
    for ax, outcome in zip(axes.ravel(), SOBOL_OUTCOMES):
        si = indices[outcome]
        st = np.asarray(si["ST"])[order]
        s1 = np.asarray(si["S1"])[order]
        stc = np.asarray(si["ST_conf"])[order]
        s1c = np.asarray(si["S1_conf"])[order]
        ax.barh(ypos + 0.2, st, height=0.36, color=BLUE, label="ST (total effect)",
                xerr=stc, error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.barh(ypos - 0.2, s1, height=0.36, color=GREEN, label="S1 (first order)",
                xerr=s1c, error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.set_yticks(ypos, np.array(names)[order])
        ax.set_title(outcome)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=min(0.0, float(min(st.min(), s1.min())) - 0.02))
    leg = axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(f"Sobol sensitivity indices per outcome ({prefix})", color=INK, fontsize=13)
    fig_dir = cfg["results_dir"] / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / f"{prefix}_sobol_indices.png", dpi=200)
    plt.close(fig)

    # --- text summary ---
    print("\nRanked total-effect (ST) indices for total_EAD_MEUR:")
    si = indices["total_EAD_MEUR"]
    ranked = sorted(
        zip(problem["names"], si["ST"], si["ST_conf"], si["S1"]),
        key=lambda t: -t[1],
    )
    for name, st, stc, s1 in ranked:
        print(
            f"  {name:18s} ST={st:6.3f} (+/-{stc:.3f})  S1={s1:6.3f}  "
            f"interactions={st - s1:6.3f}"
        )
    print(f"\nSaved {csv_path}")
    print(f"Saved {fig_dir / f'{prefix}_sobol_indices.png'}")


if __name__ == "__main__":
    main()
