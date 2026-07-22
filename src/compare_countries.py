"""Cross-country comparison of uncertainty structure and risk levels, for one
(asset, scenario) combination.

Reads, for every country that has completed the workflow:
  - {country}_{asset}_{scenario}_sobol_indices.csv  (Sobol ST/S1 per factor per outcome)
  - newest matching LHS results archive              (EAD distributions)

Produces:
  results/{asset}_{scenario}_country_comparison.csv
  results/figures/{asset}_{scenario}_country_comparison_sobol.png  (ST heatmap, factors x countries)

Usage:
    python -m src.compare_countries --asset roads --scenario baseline
    python -m src.compare_countries --asset roads --scenario baseline --countries LUX DNK
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ema_workbench import load_results
from matplotlib.colors import LinearSegmentedColormap

from .ema_model import SCENARIOS
from .paths import load_config, set_asset_override, set_country_override, set_scenario_override

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)

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
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
        "font.size": 10,
    }
)


def find_countries(cfg: dict, asset: str, scenario: str) -> list[str]:
    suffix = f"_{asset}_{scenario}_sobol_indices.csv"
    return sorted(p.name[: -len(suffix)] for p in cfg["results_dir"].glob(f"*{suffix}"))


def newest_lhs(cfg: dict, country: str, asset: str, scenario: str):
    pattern = f"experiments_{country}_{asset}_{scenario}_lhs_*.tar.gz"
    files = sorted(cfg["results_dir"].glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--asset", default="roads")
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    args = parser.parse_args()

    set_asset_override(args.asset)
    set_scenario_override(args.scenario)
    cfg = load_config()
    asset, scenario = cfg["asset_type"], cfg["scenario"]
    countries = args.countries or find_countries(cfg, asset, scenario)
    print(f"Comparing {asset}/{scenario} across: {countries}")
    prefix = f"{asset}_{scenario}"

    st_rows = {}
    summary_rows = []
    for iso in countries:
        set_country_override(iso)
        cfg_i = load_config()
        sob = pd.read_csv(cfg["results_dir"] / f"{iso}_{asset}_{scenario}_sobol_indices.csv")
        total = sob[sob["outcome"] == "total_EAD_MEUR"].set_index("factor")
        st_rows[iso] = total["ST"]

        row = {"country": iso}
        lhs_path = newest_lhs(cfg, iso, asset, scenario)
        if lhs_path is not None:
            _, outcomes = load_results(lhs_path)
            for oc, label in [
                ("total_EAD_MEUR", "total"),
                ("EAD_river_MEUR", "river"),
                ("EAD_earthquake_MEUR", "earthquake"),
            ]:
                v = np.asarray(outcomes[oc])
                row[f"{label}_EAD_median"] = float(np.median(v))
                row[f"{label}_EAD_p5"] = float(np.percentile(v, 5))
                row[f"{label}_EAD_p95"] = float(np.percentile(v, 95))
        top = total["ST"].sort_values(ascending=False)
        row["top_factor"] = top.index[0]
        row["top_factor_ST"] = float(top.iloc[0])
        row["second_factor"] = top.index[1]
        row["second_factor_ST"] = float(top.iloc[1])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("country")
    csv_path = cfg["results_dir"] / f"{prefix}_country_comparison.csv"
    summary.to_csv(csv_path)

    # --- heatmap: ST for total EAD, factors x countries ---
    # Factor set is identical across countries for a fixed (asset, scenario) -
    # it depends only on curve-group structure and scenario config, not data -
    # so any one country's factor list defines the row order.
    factor_order = list(next(iter(st_rows.values())).index)
    st = pd.DataFrame(st_rows).reindex(factor_order)
    fig, ax = plt.subplots(
        figsize=(2.1 + 1.35 * len(countries), 4.8), layout="constrained"
    )
    mat = st.to_numpy()
    vmax = max(0.5, np.nanmax(mat))
    im = ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(st.columns)), st.columns)
    ax.set_yticks(range(len(st.index)), st.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.55 * vmax else INK,
                )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Sobol total effect (ST) on total EAD", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title(f"What drives total-EAD uncertainty, per country? ({asset}/{scenario})")
    fig_dir = cfg["results_dir"] / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / f"{prefix}_country_comparison_sobol.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n" + summary.round(3).to_string())
    print(f"\nSaved {csv_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
