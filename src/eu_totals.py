"""Pan-European per-scenario EAD totals for one asset type, from existing output.

Reads the per-(country, scenario) summary that src/ead_ranges.py already wrote
and sums it across countries to give one European figure per scenario. No model
runs, no archive reads - it is arithmetic over overview_figures/ead_ranges/.

WHAT IS AND IS NOT EXACT
------------------------
    sum of MEANS        exact. Expectation is linear, so it holds whatever the
                        dependence between countries happens to be. This is the
                        number to quote without qualification.
    sum of PERCENTILES  a comonotonic bracket. Summing p5 (or p95) assumes every
                        country sits at its 5th (or 95th) percentile at the same
                        time. That is the perfectly-correlated case, so the
                        resulting band is as wide as the dependence can make it.

The bracket is tighter than that warning usually implies for the scenarios where
protection is held at its recorded value. src/cascade.py defines

    COUNTRY_SPECIFIC_FACTORS = {"protection_abs_rp", "protection_scale"}

and neither is sampled in flood_noprot_ds / coastal_noprot_ds / earthquake /
windstorm - every remaining factor is a single shared epistemic unknown (one
curve database, one cost table, one aggregation choice, one climate future), so
national totals really do move together. In the *_absprot_* variants protection
is drawn per country, so those brackets are the more inflated ones.

For the correlation-correct answer, run src/cascade.py, which draws the shared
factors once and evaluates every country at that same draw before summing.

Outputs (default overview_figures/{asset}/):
    EU_totals_{asset}.xlsx   sheets Pan_European / By_Country / Notes
    EU_totals_{asset}.png    summed mean per scenario, p5-p95 whisker, median tick

Usage:
    python -m src.eu_totals
    python -m src.eu_totals --asset roads
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .ead_ranges import HAZARD_COLORS, hazard_of
from .plot_pyramid import (
    INK,
    INK_2,
    MUTED,
    PROJECT_ROOT,
    SURFACE,
    scenario_label,
    scenario_sort_key,
)

RANGES_CSV = PROJECT_ROOT / "overview_figures" / "ead_ranges" / "EAD_Ranges.csv"
REFERENCE_CSV = PROJECT_ROOT / "overview_figures" / "reference" / "Reference_EAD.csv"
EUR_BN = 1e9

# MIRACA_RISK names Andorra twice: a stray 2-letter "AD" alongside "AND", with
# identical values. reference_ead.FILE_RE accepts 2-3 letter codes, so both are
# read. Summing without dropping one double-counts Andorra.
DUPLICATE_COUNTRY_CODES = {"AD"}


def load_country_rows(asset: str, ranges_csv: Path = RANGES_CSV) -> pd.DataFrame:
    """Per-(country, scenario) statistics for one asset, in EUR/yr."""
    if not ranges_csv.exists():
        raise SystemExit(
            f"No {ranges_csv} - run `python -m src.ead_ranges` first."
        )
    df = pd.read_csv(ranges_csv)
    sub = df[df["asset"] == asset].copy()
    if sub.empty:
        raise SystemExit(f"No rows for asset '{asset}' in {ranges_csv.name}.")
    sub["hazard"] = sub["scenario"].map(hazard_of)
    sub["scenario_label"] = sub["scenario"].map(scenario_label)
    cols = ["country", "asset", "scenario", "scenario_label", "hazard",
            "mean", "median", "p5", "p95", "n", "source"]
    return sub[cols].sort_values(
        ["scenario", "country"], key=lambda s: s.map(scenario_sort_key) if s.name == "scenario" else s
    ).reset_index(drop=True)


def load_reference(asset: str, reference_csv: Path = REFERENCE_CSV,
                   climate: str = "current") -> pd.DataFrame:
    """Deterministic MIRACA_RISK totals per hazard, in EUR/yr."""
    if not reference_csv.exists():
        return pd.DataFrame(columns=["hazard", "ref_min", "ref_mid", "ref_max"])
    ref = pd.read_csv(reference_csv)
    ref = ref[(ref["asset"] == asset) & (ref["climate"] == climate)]
    ref = ref[~ref["country"].isin(DUPLICATE_COUNTRY_CODES)]
    out = ref.groupby("hazard", as_index=False)[["ead_min", "ead_mid", "ead_max"]].sum()
    return out.rename(columns={"ead_min": "ref_min", "ead_mid": "ref_mid", "ead_max": "ref_max"})


def summarise(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: the European sums plus the reference comparison."""
    g = rows.groupby("scenario", as_index=False).agg(
        n_countries=("country", "nunique"),
        mean_EUR=("mean", "sum"),
        p5_EUR=("p5", "sum"),
        median_EUR=("median", "sum"),
        p95_EUR=("p95", "sum"),
        n_draws=("n", "max"),
        source=("source", "first"),
    )
    g["hazard"] = g["scenario"].map(hazard_of)
    g["scenario_label"] = g["scenario"].map(scenario_label)
    g = g.merge(ref, on="hazard", how="left")

    for c in ("mean", "p5", "median", "p95"):
        g[f"{c}_bn"] = g[f"{c}_EUR"] / EUR_BN
    g["ref_mid_bn"] = g["ref_mid"] / EUR_BN
    g["factor_p95_p5"] = np.where(g["p5_EUR"] > 0, g["p95_EUR"] / g["p5_EUR"], np.nan)
    g["mean_over_ref"] = np.where(g["ref_mid"] > 0, g["mean_EUR"] / g["ref_mid"], np.nan)
    # The reference is a single deterministic run at cost_level=0, not the mean of a
    # distribution. These damage distributions are strongly right-skewed, so the mean
    # sits well above the median and mean_over_ref flatters the gap. median_over_ref
    # is the like-for-like central-case comparison.
    g["median_over_ref"] = np.where(g["ref_mid"] > 0, g["median_EUR"] / g["ref_mid"], np.nan)
    g["mean_over_median"] = np.where(g["median_EUR"] > 0, g["mean_EUR"] / g["median_EUR"], np.nan)

    g = g.sort_values("scenario", key=lambda s: s.map(scenario_sort_key)).reset_index(drop=True)
    return g[[
        "scenario", "scenario_label", "hazard", "n_countries", "n_draws", "source",
        "mean_bn", "p5_bn", "median_bn", "p95_bn", "factor_p95_p5", "mean_over_median",
        "ref_mid_bn", "median_over_ref", "mean_over_ref",
        "mean_EUR", "p5_EUR", "median_EUR", "p95_EUR", "ref_min", "ref_mid", "ref_max",
    ]]


NOTES = [
    ("Units", "All *_bn columns are EUR billion per year; *_EUR columns are EUR per year."),
    ("Sum of means", "EXACT. Expectation is linear, so the summed mean is the European "
                     "mean whatever the dependence between countries. Quote this without "
                     "qualification."),
    ("Sum of percentiles", "COMONOTONIC BRACKET, not a confidence interval. Summing p5 or "
                           "p95 assumes every country sits at that same percentile "
                           "simultaneously, i.e. perfect correlation, so the band is as wide "
                           "as dependence can make it."),
    ("Why the bracket is not absurd here",
     "In flood_noprot_ds / coastal_noprot_ds / earthquake / windstorm no factor is drawn "
     "per country (cascade.py COUNTRY_SPECIFIC_FACTORS = protection_abs_rp, "
     "protection_scale, neither of which is sampled there), so every factor is one shared "
     "epistemic unknown and national totals genuinely move together."),
    ("The *_absprot_ variants",
     "Protection return period is drawn independently per country in these, so their "
     "brackets are the inflated ones. They are also a sensitivity sweep of the design "
     "standard rather than an uncertainty representation."),
    ("Correlation-correct route",
     "src/cascade.py draws the shared factors once and evaluates every country at that "
     "same draw before summing. It has not been run yet."),
    ("Multi-hazard totals",
     "NOT provided. Each scenario computes exactly one hazard, and the per-scenario "
     "archives cannot be summed row-by-row: factor sets differ per scenario and no seed "
     "is fixed, so pairing them would impose independence on cost_level, which multiplies "
     "every hazard identically."),
    ("Mean vs median", "These distributions are strongly right-skewed: the summed mean "
                       "runs 2-3x the summed median. The mean is the EXACT aggregate, but "
                       "the reference is a single deterministic run at cost_level=0, not a "
                       "distribution mean - so median_over_ref is the like-for-like "
                       "central-case comparison and mean_over_ref overstates the gap."),
    ("Reference", "MIRACA_RISK deterministic totals, climate=current, summed over the same "
                  "countries. Its ref_min/ref_mid/ref_max is the pipeline's own COST range "
                  "(cost_level at -1/0/+1) - one factor, not the full uncertainty."),
    ("Andorra", "The reference file carries both 'AD' and 'AND' with identical values; "
                "'AD' is dropped here so Andorra is counted once."),
]


def write_excel(path: Path, summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = pd.DataFrame(NOTES, columns=["Topic", "Note"])
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Pan_European", index=False)
        rows.to_excel(xl, sheet_name="By_Country", index=False)
        notes.to_excel(xl, sheet_name="Notes", index=False)


def plot_totals(summary: pd.DataFrame, asset: str, out: Path, scale: str = "log") -> None:
    """Summed mean per scenario, with the comonotonic p5-p95 bracket.

    scale="linear" gives the bars a true zero baseline, so bar LENGTH is
    readable - but coastal (~0.13 bn) is then barely visible beside earthquake
    (~5.1 bn). scale="log" keeps every scenario legible at the cost of the
    baseline: on a log axis the bar bottom is only the axis floor, so just the
    top edge carries meaning. Both are written; pick per audience.
    """
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "text.color": INK,
    })
    n = len(summary)
    fig, ax = plt.subplots(figsize=(1.55 * n + 3.4, 6.4), layout="constrained")
    x = np.arange(n)
    colors = [HAZARD_COLORS.get(h, MUTED) for h in summary["hazard"]]

    ax.bar(x, summary["mean_bn"], width=0.62, color=colors, zorder=3,
           edgecolor=SURFACE, linewidth=1.0)
    for i, r in summary.iterrows():
        ax.vlines(i, r["p5_bn"], r["p95_bn"], color=INK, linewidth=1.3, zorder=5)
        for cap in ("p5_bn", "p95_bn"):
            ax.hlines(r[cap], i - 0.13, i + 0.13, color=INK, linewidth=1.3, zorder=5)
        ax.hlines(r["median_bn"], i - 0.31, i + 0.31, color=SURFACE, linewidth=2.4, zorder=6)
        if np.isfinite(r["ref_mid_bn"]) and r["ref_mid_bn"] > 0:
            ax.hlines(r["ref_mid_bn"], i - 0.36, i + 0.36, color=INK_2,
                      linewidth=1.6, linestyle=(0, (3, 2)), zorder=7)

    if scale == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)
    ax.set_ylabel("pan-European EAD (EUR bn / yr)")
    ax.set_xticks(x, [scenario_label(s, short=True).replace(" (", "\n(")
                      for s in summary["scenario"]], fontsize=9)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # pad must clear the TWO-line subtitle below it: 7pt offset + 2 lines at
    # ~11pt each. A pad tuned for a one-line caption overlaps the title.
    ax.set_title(f"Pan-European EAD by scenario - {asset} ({scale} scale)",
                 loc="left", pad=38,
                 fontsize=14, fontweight="bold")
    baseline_note = (
        "  Log axis: the bar bottom is the axis floor, not zero - read the top edge, "
        "not the bar length."
        if scale == "log" else
        "  Linear axis (zero baseline), so bar length is readable; coastal is small "
        "beside the other hazards.")
    ax.annotate(
        "Bar = sum of country MEANS (exact, dependence-free).  Whisker = sum of country "
        "p5-p95, a COMONOTONIC bracket, not a confidence interval.\n"
        "White tick = sum of medians.  Dashed = MIRACA_RISK deterministic total for that "
        "hazard (its own min-max spans the cost factor only)." + baseline_note,
        xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 7),
        textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color=MUTED)

    handles = [Line2D([], [], color=HAZARD_COLORS[h], linewidth=8, label=h)
               for h in HAZARD_COLORS if (summary["hazard"] == h).any()]
    handles.append(Line2D([], [], color=INK_2, linewidth=1.6, linestyle=(0, (3, 2)),
                          label="MIRACA_RISK deterministic"))
    leg = fig.legend(handles=handles, frameon=False, ncol=len(handles),
                     loc="outside lower center", fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", default="power", help="asset type (default: power)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="restrict to these scenarios (default: all present)")
    parser.add_argument("--out-dir", default=None,
                        help="default: overview_figures/{asset}")
    parser.add_argument("--scale", choices=["log", "linear", "both"], default="both",
                        help="y-axis scale for the chart (default: both)")
    parser.add_argument("--climate", default="current",
                        help="reference climate horizon (default: current)")
    args = parser.parse_args()

    rows = load_country_rows(args.asset)
    if args.scenarios:
        rows = rows[rows["scenario"].isin(args.scenarios)]
        if rows.empty:
            raise SystemExit(f"No rows left after --scenarios {args.scenarios}.")
    ref = load_reference(args.asset, climate=args.climate)
    summary = summarise(rows, ref)

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures" / args.asset
    xlsx = out_dir / f"EU_totals_{args.asset}.xlsx"
    scales = ["log", "linear"] if args.scale == "both" else [args.scale]
    write_excel(xlsx, summary, rows)
    pngs = []
    for sc in scales:
        png_path = out_dir / f"EU_totals_{args.asset}_{sc}.png"
        plot_totals(summary, args.asset, png_path, scale=sc)
        pngs.append(png_path)

    show = summary[["scenario", "n_countries", "mean_bn", "p5_bn", "median_bn",
                    "p95_bn", "mean_over_median", "ref_mid_bn", "median_over_ref",
                    "mean_over_ref"]]
    print(f"{args.asset}: {rows['country'].nunique()} countries, "
          f"{len(rows)} (country, scenario) rows")
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nwrote {xlsx}")
    for png_path in pngs:
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
