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
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .ead_ranges import HAZARD_COLORS, hazard_of
from .ema_model import protection_range
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


ALL_ASSETS = "all"

# The headline comparison. Coastal is represented by its SEA-LEVEL-RISE variants,
# not the present-day ones, so both flood families carry climate uncertainty:
# river flood samples `warming` over five levels, of which four are future, so a
# present-day-only coastal scenario beside it compares a warmed river world
# against an unwarmed coast.
#
# The two parametrisations are not interchangeable and should not be described as
# such - river flood is sampled over warming LEVELS, coastal over SSP x HORIZON
# combinations, and the coastal set puts 30% of its weight on 2100 where the
# river set is horizon-free.
#
# Present-day coastal is kept but demoted to a reference case: the deterministic
# benchmark is a current-climate estimate, so it is the only like-for-like
# coastal comparison against it.
HEADLINE_SCENARIOS = [
    "flood_absprot_ds",
    "flood_noprot_ds",
    "coastal_absprot_ds_slr",
    "coastal_noprot_ds_slr",
    "earthquake",
    "windstorm",
    "windstorm_absprot",
]
REFERENCE_SCENARIOS = ["coastal_absprot_ds", "coastal_noprot_ds"]



# Spelled out and matching src/plot_pyramid.py, so the two figure families use
# one unit wording.
VALUE_LABEL = "Expected annual damage (Billion Euros per year)"
# Both estimates are MIRACA; what separates them is that the reference is a
# single deterministic run rather than a sampled ensemble, so the label says so.
REFERENCE_LABEL = "Deterministic reference"


def load_country_rows(asset: str, ranges_csv: Path = RANGES_CSV) -> pd.DataFrame:
    """Per-(country, scenario) statistics for one asset, in EUR/yr.

    asset="all" keeps every asset, so the sums below run over (country, asset)
    rather than country alone. That widens the comonotonic bracket further -
    it now assumes the ten infrastructure types sit at their 5th (or 95th)
    percentile simultaneously as well as the countries - and the warning in the
    module docstring applies with correspondingly more force.
    """
    if not ranges_csv.exists():
        raise SystemExit(
            f"No {ranges_csv} - run `python -m src.ead_ranges` first."
        )
    df = pd.read_csv(ranges_csv)
    sub = (df.copy() if asset == ALL_ASSETS else df[df["asset"] == asset].copy())
    if sub.empty:
        raise SystemExit(f"No rows for asset '{asset}' in {ranges_csv.name}.")
    sub["hazard"] = sub["scenario"].map(hazard_of)
    sub["scenario_label"] = sub["scenario"].map(scenario_label)
    cols = ["country", "asset", "scenario", "scenario_label", "hazard",
            "mean", "median", "p5", "p25", "p75", "p95", "n", "source"]
    return sub[cols].sort_values(
        ["scenario", "country"], key=lambda s: s.map(scenario_sort_key) if s.name == "scenario" else s
    ).reset_index(drop=True)


def drop_partial_coverage(rows: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove scenarios that ran on only a SUBSET of their hazard's countries.

    Coastal covering 26 countries rather than 36 is correct - the rest are
    landlocked - so coverage cannot be judged against a single global country
    count. It is judged per HAZARD instead: within one hazard every scenario is
    applicable to the same countries, so any scenario covering fewer than that
    hazard's maximum was run on a deliberate test subset.

    Those exist to test how an assumed parameter RANGE changes a factor's
    importance, not to estimate European damage. Summed onto the same axis as a
    full-coverage scenario they read as a collapse in damage when they are only
    a smaller denominator, which is exactly how the coastal range tests looked
    at ~20x below the full coastal scenario.
    """
    best = rows.groupby("hazard")["country"].nunique()
    keep, dropped = [], []
    for scen, g in rows.groupby("scenario"):
        hz = g["hazard"].iloc[0]
        if g["country"].nunique() >= int(best[hz]):
            keep.append(scen)
        else:
            dropped.append((scen, g["country"].nunique(), int(best[hz])))
    if dropped and verbose:
        print("Excluded from the comparison - partial country coverage:")
        for scen, n, full in sorted(dropped):
            print(f"  {scen:34s} {n:3d} of {full} {rows.loc[rows['scenario'] == scen, 'hazard'].iloc[0]} countries")
        print("  (pass --allow-partial to include them anyway)")
    return rows[rows["scenario"].isin(keep)]


def load_reference(asset: str, reference_csv: Path = REFERENCE_CSV,
                   climate: str = "current") -> pd.DataFrame:
    """Deterministic MIRACA_RISK totals per hazard, in EUR/yr."""
    if not reference_csv.exists():
        return pd.DataFrame(columns=["hazard", "ref_min", "ref_mid", "ref_max"])
    ref = pd.read_csv(reference_csv)
    ref = ref[ref["climate"] == climate]
    if asset != ALL_ASSETS:
        ref = ref[ref["asset"] == asset]
    ref = ref[~ref["country"].isin(DUPLICATE_COUNTRY_CODES)]
    out = ref.groupby("hazard", as_index=False)[["ead_min", "ead_mid", "ead_max"]].sum()
    return out.rename(columns={"ead_min": "ref_min", "ead_mid": "ref_mid", "ead_max": "ref_max"})


def summarise(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: the European sums plus the reference comparison."""
    g = rows.groupby("scenario", as_index=False).agg(
        n_countries=("country", "nunique"),
        n_assets=("asset", "nunique"),
        n_combos=("country", "size"),
        mean_EUR=("mean", "sum"),
        p5_EUR=("p5", "sum"),
        p25_EUR=("p25", "sum"),
        median_EUR=("median", "sum"),
        p75_EUR=("p75", "sum"),
        p95_EUR=("p95", "sum"),
        n_draws=("n", "max"),
        source=("source", "first"),
    )
    g["hazard"] = g["scenario"].map(hazard_of)
    g["scenario_label"] = g["scenario"].map(scenario_label)
    g = g.merge(ref, on="hazard", how="left")

    for c in ("mean", "p5", "p25", "median", "p75", "p95"):
        g[f"{c}_bn"] = g[f"{c}_EUR"] / EUR_BN
    # The reference is a RANGE, not a point: min/mid/max per hazard. Carrying
    # only the midpoint hid that earthquake spans 0.45-19.1 bn in the source.
    for c in ("ref_min", "ref_mid", "ref_max"):
        g[f"{c}_bn"] = g[c] / EUR_BN
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
        "scenario", "scenario_label", "hazard", "n_countries", "n_assets", "n_combos",
        "n_draws", "source",
        "mean_bn", "p5_bn", "p25_bn", "median_bn", "p75_bn", "p95_bn",
        "factor_p95_p5", "mean_over_median",
        "ref_min_bn", "ref_mid_bn", "ref_max_bn", "median_over_ref", "mean_over_ref",
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


def _lighten(hex_colour: str, fraction: float) -> str:
    """Blend a colour toward white. Used instead of alpha.

    Transparency made the bars look drawn with a highlighter, and where the two
    brackets overlap it also compounded into a third shade that means nothing.
    Two opaque tints of the same hue keep the nesting readable and print cleanly.
    """
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    out = [round(c + (255 - c) * fraction) for c in rgb]
    return "#%02x%02x%02x" % tuple(out)


def headline_labels(scenarios) -> dict[str, str]:
    """Figure-local scenario labels.

    With the sea-level-rise coastal runs standing in as THE coastal case, their
    "+ SLR" suffix is noise - and it hid the sampled return period that the river
    rows state, so the two hazards read as if described differently. The suffix is
    dropped and the protection treatment spelled out to match river flood.

    Only when the present-day twin is absent, though: with both plotted the
    labels would collide and the distinction is exactly what matters.
    """
    plotted = set(scenarios)
    out = {}
    for scen in scenarios:
        if not scen.endswith("_slr"):
            continue
        base = scen[: -len("_slr")]
        if base in plotted:
            continue
        hazard = "Coastal flood"
        if "absprot" in scen:
            lo, hi = protection_range(base, "abs")
            out[scen] = f"{hazard}\n(protection sampled, RP {lo:.0f}-{hi:.0f})"
        else:
            out[scen] = f"{hazard}\n(protection as recorded)"
    return out

def wrap_scenario(scenario: str) -> str:
    """Scenario label broken at the bracket, so the two halves stack."""
    return scenario_label(scenario, short=True).replace(" (", "\n(")


def split_scenario(scenario: str, overrides: dict[str, str] | None = None) -> tuple[str, str]:
    """(hazard, qualifier) - "River flood", "(protection sampled, RP 5-200)".

    Split so the two halves can be set at different sizes: the hazard is what
    the reader navigates by, the bracket is a detail that need not grow with it.
    `overrides` lets one figure relabel a scenario without changing the
    study-wide label, which other figures still need.
    """
    label = (overrides or {}).get(scenario) or scenario_label(scenario, short=True)
    if "\n" in label:                 # pre-split by headline_labels
        head, _, tail = label.partition("\n")
        return head, tail
    head, sep, tail = label.partition(" (")
    return head, (sep.strip() + tail) if sep else ""


HAZARD_FS = 16.0      # the hazard name
QUALIFIER_FS = 11.5   # the bracketed scenario detail
AXIS_FS = 15.0
TICK_FS = 13.0
LEGEND_FS = 14.0
TITLE_FS = 17.0
PAGE = "#ffffff"

# The vertical range figure is the one that goes into the report, where it is
# reduced to the width of a text column - roughly a third of the 20 inches it
# is drawn at. At that reduction the sizes above land near 5 pt on the page, so
# this figure is set a size class larger. Everything else keeps the sizes above,
# because the horizontal variant reserves a fixed left margin for its labels and
# type this big would run out of it.
VR_HAZARD_FS = 26.0
VR_QUALIFIER_FS = 17.0
VR_AXIS_FS = 22.0
VR_TICK_FS = 20.0
VR_LEGEND_FS = 19.0
VR_TITLE_FS = 26.0

# The ensemble mean, as a white diamond with a black edge - the same mark the
# network-level figures use for the mean. The deterministic best estimate is a
# grey circle instead, so the figure carries only one kind of diamond.
MEAN_MARKER = dict(marker="D", linestyle="none", markerfacecolor=PAGE,
                   markeredgecolor=INK, markeredgewidth=1.6)
BEST_MARKER = dict(marker="o", linestyle="none", color=INK_2,
                   markeredgecolor=PAGE, markeredgewidth=1.2)


def _range_scope_note(summary: pd.DataFrame, asset: str) -> str:
    """The scope sentence. Returned rather than drawn: the report carries it as
    body text, so the figure no longer prints a subheadline."""
    lo, hi = int(summary["n_combos"].min()), int(summary["n_combos"].max())
    span = f"{lo:,}" if lo == hi else f"{lo:,}-{hi:,}"
    if asset == ALL_ASSETS:
        return (f"Summed over every (country, infrastructure) pair that ran: {span} "
                f"per scenario, up to {int(summary['n_countries'].max())} countries "
                f"x {int(summary['n_assets'].max())} infrastructure types.")
    return f"Summed over up to {int(summary['n_countries'].max())} countries."


def _wrap_qualifier(qual: str, width: int) -> str:
    """Break the bracketed detail onto a second line, at its comma if it has one.

    textwrap on its own breaks "(protection sampled, RP 5-200)" after the first
    word, which separates the treatment from its own adjective. The comma is
    the one place the phrase is meant to come apart.
    """
    if len(qual) <= width:
        return qual
    head, sep, tail = qual.partition(", ")
    if sep and len(head) + 1 <= width and len(tail) <= width:
        return f"{head},{chr(10)}{tail}"
    return chr(10).join(textwrap.wrap(qual, width) or [qual])


def _range_legend(fig, y: float, fontsize: float = LEGEND_FS,
                  markersize: float = 7.0, columnspacing: float = 1.9):
    """Legend below the axes. `y` is figure fraction - set low enough that it
    clears the two-line scenario labels rather than sitting on them."""
    handles = [
        Line2D([], [], color=INK, linewidth=3.0, label="median"),
        Line2D([], [], markersize=markersize, label="mean", **MEAN_MARKER),
        Line2D([], [], color=MUTED, linewidth=13, label="p25 to p75"),
        Line2D([], [], color=_lighten(MUTED, 0.55), linewidth=5, label="p5 to p95"),
        Line2D([], [], color=INK_2, linewidth=1.8, linestyle=(0, (3, 2)),
               label=REFERENCE_LABEL),
        Line2D([], [], markersize=markersize, label="Deterministic best estimate",
               **BEST_MARKER),
    ]
    # One row always: a wrapped legend reads as two groups and invites the
    # reader to look for a distinction between the rows that is not there.
    leg = fig.legend(handles=handles, frameon=False, ncol=len(handles),
                     loc="lower center", bbox_to_anchor=(0.5, y),
                     fontsize=fontsize, handlelength=2.2,
                     columnspacing=columnspacing)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _range_rcparams():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": PAGE, "savefig.facecolor": PAGE,
        "axes.facecolor": PAGE, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })


def plot_ranges(summary: pd.DataFrame, asset: str, out: Path, scale: str = "log") -> None:
    """One horizontal range per scenario: p5-p95 and p25-p75, median marked."""
    _range_rcparams()
    n = len(summary)
    labels = headline_labels(list(summary["scenario"]))
    fig, ax = plt.subplots(figsize=(14.0, 1.15 * n + 2.6))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.905, bottom=0.175)
    y = np.arange(n)[::-1]

    for yi, (_, r) in zip(y, summary.iterrows()):
        colour = HAZARD_COLORS.get(r["hazard"], MUTED)
        pale = _lighten(colour, 0.55)
        ax.hlines(yi, r["p5_bn"], r["p95_bn"], color=pale, linewidth=6, zorder=3)
        ax.hlines(yi, r["p25_bn"], r["p75_bn"], color=colour, linewidth=17, zorder=4)
        for cap in ("p5_bn", "p95_bn"):
            ax.vlines(r[cap], yi - 0.17, yi + 0.17, color=colour, linewidth=2.2, zorder=5)
        ax.vlines(r["median_bn"], yi - 0.30, yi + 0.30, color=INK, linewidth=3.0, zorder=7)
        # These distributions are strongly right-skewed, so the mean lands near
        # p75 rather than on the median - which is the reason to show both.
        ax.plot(r["mean_bn"], yi, markersize=8, zorder=8, **MEAN_MARKER)
        lo, hi = r.get("ref_min_bn", np.nan), r.get("ref_max_bn", np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
            ax.hlines(yi - 0.44, lo, hi, color=INK_2, linewidth=1.8,
                      linestyle=(0, (3, 2)), zorder=6)
            for v in (lo, hi):
                ax.vlines(v, yi - 0.52, yi - 0.36, color=INK_2, linewidth=1.6, zorder=6)
            # The reference's own central estimate, inside its min-max range.
            mid = r.get("ref_mid_bn", np.nan)
            if np.isfinite(mid) and mid > 0:
                ax.plot(mid, yi - 0.44, markersize=7.5, zorder=7, **BEST_MARKER)

        # Drawn rather than set as tick labels, so the two lines can differ in
        # size. clip_on=False because they sit outside the axes.
        hz, qual = split_scenario(r["scenario"], labels)
        tr = ax.get_yaxis_transform()
        if qual:
            ax.annotate(hz, (0, yi), xycoords=tr, xytext=(-14, 2),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=HAZARD_FS, color=INK, annotation_clip=False)
            ax.annotate(qual, (0, yi), xycoords=tr, xytext=(-14, -4),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=QUALIFIER_FS, color=INK, annotation_clip=False)
        else:
            ax.annotate(hz, (0, yi), xycoords=tr, xytext=(-14, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=HAZARD_FS, color=INK, annotation_clip=False)

    if scale == "log":
        ax.set_xscale("log")
    else:
        # Zero floor: a negative lower limit would imply damages below zero.
        ax.set_xlim(0, float(summary["p95_bn"].max()) * 1.04)
    ax.set_xlabel(VALUE_LABEL, fontsize=AXIS_FS)
    ax.set_yticks(y, ["" for _ in y])
    ax.set_ylim(-0.8, n - 0.2)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=TICK_FS)

    scope = ("all 10 infrastructure types" if asset == ALL_ASSETS else asset)
    ax.set_title(f"Pan-European EAD by scenario - {scope}", loc="left", pad=16,
                 fontsize=TITLE_FS, fontweight="bold", color=INK)
    _range_legend(fig, 0.045)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_ranges_vertical(summary: pd.DataFrame, asset: str, out: Path,
                         scale: str = "log") -> None:
    """The same ranges with EAD on the y-axis and scenarios across the x-axis."""
    _range_rcparams()
    n = len(summary)
    labels = headline_labels(list(summary["scenario"]))
    fig, ax = plt.subplots(figsize=(2.45 * n + 3.0, 9.8))
    left, right = 0.105, 0.985
    # Deeper than the horizontal variant's: at report type size the hazard name
    # and a wrapped two-line qualifier stack to well over an inch, and the
    # legend still has to clear them.
    fig.subplots_adjust(left=left, right=right, top=0.915, bottom=0.275)
    x = np.arange(n)
    # How long a qualifier line can be before it has to wrap: one column of the
    # axes, in points. At this size "(protection sampled, RP 5-200)" is wider
    # than the column it labels and would run into its neighbour.
    col_pt = fig.get_figwidth() * (right - left) / n * 72.0
    qual_wrap = max(10, int(col_pt / (VR_QUALIFIER_FS * 0.47)))
    qual_texts = []

    for xi, (_, r) in zip(x, summary.iterrows()):
        colour = HAZARD_COLORS.get(r["hazard"], MUTED)
        pale = _lighten(colour, 0.55)
        ax.vlines(xi, r["p5_bn"], r["p95_bn"], color=pale, linewidth=9, zorder=3)
        ax.vlines(xi, r["p25_bn"], r["p75_bn"], color=colour, linewidth=28, zorder=4)
        for cap in ("p5_bn", "p95_bn"):
            ax.hlines(r[cap], xi - 0.13, xi + 0.13, color=colour, linewidth=2.2, zorder=5)
        ax.hlines(r["median_bn"], xi - 0.22, xi + 0.22, color=INK, linewidth=3.0, zorder=7)
        # These distributions are strongly right-skewed, so the mean lands near
        # p75 rather than on the median - which is the reason to show both.
        ax.plot(xi, r["mean_bn"], markersize=11, zorder=8, **MEAN_MARKER)
        lo, hi = r.get("ref_min_bn", np.nan), r.get("ref_max_bn", np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
            ax.vlines(xi + 0.34, lo, hi, color=INK_2, linewidth=1.8,
                      linestyle=(0, (3, 2)), zorder=6)
            for v in (lo, hi):
                ax.hlines(v, xi + 0.27, xi + 0.41, color=INK_2, linewidth=1.6, zorder=6)
            mid = r.get("ref_mid_bn", np.nan)
            if np.isfinite(mid) and mid > 0:
                ax.plot(xi + 0.34, mid, markersize=10.5, zorder=7, **BEST_MARKER)

        hz, qual = split_scenario(r["scenario"], labels)
        tr = ax.get_xaxis_transform()
        ax.annotate(hz, (xi, 0), xycoords=tr, xytext=(0, -20),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=VR_HAZARD_FS, color=INK, annotation_clip=False)
        if qual:
            qual_texts.append(ax.annotate(
                _wrap_qualifier(qual, qual_wrap), (xi, 0), xycoords=tr,
                xytext=(0, -20 - VR_HAZARD_FS * 1.35),
                textcoords="offset points", ha="center", va="top",
                fontsize=VR_QUALIFIER_FS, color=INK,
                annotation_clip=False, linespacing=1.25))

    top = float(summary["p95_bn"].max())
    if scale == "log":
        ax.set_yscale("log")
        floor = float(summary["p5_bn"].replace(0, np.nan).min())
        ax.set_ylim(floor / 2.2, top * 1.7)
    else:
        # Starts exactly at zero and never below it.
        ax.set_ylim(0, top * 1.10)
    ax.set_ylabel(VALUE_LABEL, fontsize=VR_AXIS_FS)
    ax.set_xticks(x, ["" for _ in x])
    ax.set_xlim(-0.85, n - 0.15)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=VR_TICK_FS)

    scope = ("all 10 infrastructure types" if asset == ALL_ASSETS else asset)
    ax.set_title(f"Pan-European EAD by scenario - {scope}", loc="left", pad=16,
                 fontsize=VR_TITLE_FS, fontweight="bold", color=INK)
    # qual_wrap is a character count, which is an estimate. This is the check:
    # if a qualifier still overruns the column it labels once drawn, every one
    # of them shrinks by the same factor, so the row keeps one type size.
    fig.canvas.draw()
    room = 0.96 * (ax.transData.transform((1.0, 0))[0]
                   - ax.transData.transform((0.0, 0))[0])
    worst = min((room / t.get_window_extent().width for t in qual_texts
                 if t.get_window_extent().width > room), default=1.0)
    if worst < 1.0:
        for t in qual_texts:
            t.set_fontsize(t.get_fontsize() * worst)

    _range_legend(fig, 0.012, fontsize=VR_LEGEND_FS, markersize=10,
                  columnspacing=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)



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
    parser.add_argument("--asset", default="power",
                        help=f"asset type, or '{ALL_ASSETS}' to sum over every "
                             f"infrastructure type as well as every country "
                             f"(default: power)")
    parser.add_argument("--all-comparable", action="store_true",
                        help="plot every full-coverage scenario rather than just the "
                             "headline set, i.e. also the present-day coastal "
                             "reference cases.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="keep scenarios that ran on only a subset of their "
                             "hazard's countries. Off by default: such runs are "
                             "range tests, and summing them onto the same axis as a "
                             "full-coverage scenario compares different denominators.")
    parser.add_argument("--style", choices=["bar", "range", "vrange", "both"],
                        default="both",
                        help="bar = mean as a bar with a p5-p95 whisker; range = "
                             "one horizontal p5-p95 range per scenario with the "
                             "mean and median marked (default: both)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="restrict to these scenarios (default: the headline set, "
                             "which represents coastal by its sea-level-rise variants "
                             "so both flood families carry climate uncertainty)")
    parser.add_argument("--out-dir", default=None,
                        help="default: overview_figures/{asset}")
    parser.add_argument("--scale", choices=["log", "linear", "both"], default="both",
                        help="y-axis scale for the chart (default: both)")
    parser.add_argument("--climate", default="current",
                        help="reference climate horizon (default: current)")
    args = parser.parse_args()

    rows = load_country_rows(args.asset)
    if not args.allow_partial:
        rows = drop_partial_coverage(rows)
    if args.scenarios is None and not args.all_comparable:
        args.scenarios = HEADLINE_SCENARIOS
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
    styles = ["bar", "range", "vrange"] if args.style == "both" else [args.style]
    pngs = []
    for sc in scales:
        for st in styles:
            suffix = {"bar": f"_{sc}", "range": f"_range_{sc}",
                      "vrange": f"_range_vertical_{sc}"}[st]
            png_path = out_dir / f"EU_totals_{args.asset}{suffix}.png"
            {"bar": plot_totals, "range": plot_ranges,
             "vrange": plot_ranges_vertical}[st](
                summary, args.asset, png_path, scale=sc)
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
