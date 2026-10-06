"""Overview pies: which parameter drives total-EAD uncertainty, per scenario.

For every finished (country, asset, scenario) experiment the uncertainty factors
are ranked by Sobol total effect (ST) on total_EAD_MEUR. The highest-ranked
factor is that experiment's "driver"; this module counts drivers by category and
draws one pie per infrastructure type plus an all-infrastructure pie - one
composite figure per scenario, plus a cross-scenario overview.

TWO RANKS are drawn, into separate PNGs:

    drivers_{scenario}.png                    rank 1 - most influential
    drivers_{scenario}_second.png             rank 2 - the runner-up
    drivers_ALL_scenarios_overview.png        rank 1, all scenarios side by side
    drivers_ALL_scenarios_overview_second.png rank 2, same layout

The runner-up is worth its own figure because the top driver is often structural
and already known (protection standard nearly everywhere). Rank 2 shows what
would dominate once that one is pinned down, so it is the better guide to where
further data collection actually pays off.

NOT EVERY EXPERIMENT HAS A DRIVER
---------------------------------
273 of the 2,113 ranked combinations have EVERY Sobol index at exactly zero.
Ranking them puts an arbitrary tie-break winner on top, and until this was
fixed those 273 were counted as real "vulnerability curve" or "cost" wedges -
about 13% of every rank-1 pie, distributed unevenly across asset types
(airports 55, gas 52, oil 38), so it skewed the very comparison the pies exist
to support. They now get their own slices, and there are two of them, because
zero variance has two quite different causes:

    No exposure              nothing of that asset type lies inside the RP100
                             hazard footprint. 188 combinations. Nothing to
                             model, and nothing to fix.
    Exposed, never damaged   assets ARE in the footprint and no draw produced
                             damage anyway. 85 combinations - and this one is a
                             finding, not a non-event: FIN, GRC, CZE, ROU and
                             HUN each have ~2e8 m2 of airport inside the
                             windstorm footprint and the model reports zero
                             damage with certainty. That is the hazard never
                             crossing the vulnerability curve's onset, and it is
                             worth checking the curve rather than hiding.

The split needs exposed_qty_RP100_* from EAD_Ranges_all_outcomes.csv (written by
`python -m src.ead_ranges`). Without that file the two fold into one honest
"No damage in any draw" slice rather than silently reverting to a fake driver.

Grey "not finished yet" slices come from the expected matrix: every
(country, asset) with exposure data, crossed with the scenarios whose hazard
applies to it (src/curves.py::applicable_hazards), minus the pairs that carry no
curve-mapped feature at all (see drop_unrunnable). An experiment with no Sobol
row yet is counted as unfinished rather than silently dropped, so coverage stays
visible on the face of the figure. "Unfinished" and the three non-driver states
are different things: unfinished means no Sobol row exists, the others mean the
row exists and is all zeros.

Input:   MIRACA_uncertainty_study_summary.xlsx, sheet All_Sobol_Indices
         (regenerate it first with `python -m src.aggregate_results`)
Output:  overview_figures/            (override with --out-dir)

Usage:
    python -m src.plot_drivers
    python -m src.plot_drivers --scenarios flood_absprot_ds earthquake
    python -m src.plot_drivers --ranks 1        # skip the runner-up figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Patch

from .curves import (
    ASSET_CONFIGS,
    applicable_hazards,
    get_asset_config,
    mapped_object_types,
)
from .ema_model import SCENARIO_HAZARD
from .paths import load_config
from .plot_pyramid import (
    INK,
    INK_2,
    MUTED,
    PROJECT_ROOT,
    SURFACE,
    scenario_label,
    scenario_sort_key,
)

NL = chr(10)  # newline for multi-line figure text, without escaping noise
WORKBOOK_NAME = "MIRACA_uncertainty_study_summary.xlsx"
SHEET = "All_Sobol_Indices"
OUTCOME = "total_EAD_MEUR"

# Every default study scenario, ordered so each flood pair sits side by side:
# the sampled-protection variant next to its protection-fixed twin. Comparing
# the two is the point - it shows what takes over once protection is pinned
# down. Read with the caveat that protection is a CONSTANT in the *_noprot
# variants, so its absence there is by construction and not a finding; the
# figures state this in their subtitle. Pass --scenarios to narrow the set.
DEFAULT_OVERVIEW_SCENARIOS = [
    "flood_absprot_ds",
    "flood_noprot_ds",
    "coastal_absprot_ds",
    "coastal_noprot_ds",
    "earthquake",
    "windstorm",
    "windstorm_absprot",
]

# Scenario display names live in src/plot_pyramid.py (SCENARIO_LABEL).
RANK_WORD = {1: "Most", 2: "Second most", 3: "Third most"}
RANK_SUFFIX = {1: "", 2: "_second", 3: "_third"}
RANK_BLURB = {
    1: "Highest",
    2: "Second-highest",
    3: "Third-highest",
}

# The first four are the original colourblind-validated set (blue/orange/green/
# violet, all-pairs checked). Warming and aggregation were folded into a grey
# "Other" before; showing them separately needs two more hues, chosen well away
# from the first four - note this wider set has NOT been through the same
# all-pairs validation, so re-check it if these figures go to print.
# A combination with no variance to attribute is not a driver category, so the
# three states below are deliberately neutral - chroma is reserved for identity,
# and these say "there is nothing here to identify". They step clear of each
# other in lightness (adjacent dL >= 0.06, checked); "Exposed, never damaged"
# additionally carries a hatch, because it is the one state on this figure that
# someone should go and look at.
NO_EXPOSURE = "No exposure"
EXPOSED_UNDAMAGED = "Exposed, never damaged"
NO_DAMAGE = "No damage in any draw"          # fallback: exposure file absent
NON_DRIVER_STATES = (NO_EXPOSURE, EXPOSED_UNDAMAGED, NO_DAMAGE)

COLORS = {
    "Protection standard": "#2a78d6",
    "Vulnerability curve": "#eb6834",
    "Hazard intensity": "#1baf7a",
    "Cost / max-damage": "#4a3aa7",
    "Climate warming": "#c2185b",
    "Aggregation method": "#8c6d1f",
    "Other": "#8a8983",
    NO_EXPOSURE: "#bfbdb1",
    EXPOSED_UNDAMAGED: "#a09c8e",
    NO_DAMAGE: "#b0ada0",
    "Unfinished": "#d9d8d1",
}
HATCH = {EXPOSED_UNDAMAGED: "///"}

# Report mode (--report): the same pies, restyled for inclusion in a written
# report rather than for monitoring a running study. Set once in main().
#   - coverage captions collapse to a bare "(n)": in a report the study IS
#     finished, so a fraction and a percent are both 100% and say nothing.
#   - "Unfinished" leaves the legend entirely, for the same reason.
#   - the legend wraps onto two rows so it can carry a larger font.
#   - margins, inter-pie gaps and the header band all tighten, because the
#     monitoring layout leaves room for captions a report does not print.
REPORT = False
SLICE_ORDER = [
    "Protection standard",
    "Vulnerability curve",
    "Hazard intensity",
    "Cost / max-damage",
    "Climate warming",
    "Aggregation method",
    "Other",
    NO_EXPOSURE,
    EXPOSED_UNDAMAGED,
    NO_DAMAGE,
    "Unfinished",
]
# No folding: every factor category gets its own slice. "Other" survives only as
# a catch-all for a factor_code that category() does not recognise, which should
# not happen and is worth noticing if it does.
FOLD: dict[str, str] = {}


def category(code: str) -> str:
    """Normalise a factor_code to the category shown as a pie slice."""
    if not isinstance(code, str):
        return "Other"
    if code.startswith("curve_"):
        return "Vulnerability curve"
    if code in ("protection_abs_rp", "protection_scale"):
        return "Protection standard"
    if code in ("warming", "slr"):
        # River warming levels and coastal sea-level-rise states are both "which
        # climate future", and the two never occur in the same scenario - warming
        # is river-only, slr coastal-only - so one slice lets the question "does
        # the climate future dominate?" be read across both flood hazards.
        return "Climate warming"
    if code == "cost_level":
        return "Cost / max-damage"
    if code in ("depth_scale", "depth_offset", "pga_scale", "gust_scale"):
        return "Hazard intensity"
    if code == "aggregation":
        return "Aggregation method"
    return code


def display_category(cat):
    if cat is None or (isinstance(cat, float) and pd.isna(cat)):
        return None
    return FOLD.get(cat, cat)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def expected_matrix(cfg: dict, scenarios: list[str],
                    countries: list[str] | None = None) -> pd.DataFrame:
    """Every (country, asset, scenario) that SHOULD exist.

    Derived from the exposure files actually on disk crossed with the
    applicability rules, so a combination that was never runnable (windstorm for
    roads, coastal for a landlocked country) is never counted as missing.

    `countries` restricts the matrix to a subset. That matters when a scenario
    was only run for part of the study - the range tests cover seven countries -
    because comparing their pie against a full 36-country baseline would differ
    by country mix as much as by the thing under test.
    """
    expo = Path(cfg["exposure_dir"])
    combos = set()
    for name in os.listdir(expo):
        if not name.endswith("_exposure.parquet"):
            continue
        base = name[: -len("_exposure.parquet")]
        country, _, asset = base.partition("_")
        if country and asset in ASSET_CONFIGS:
            combos.add((country, asset))

    if countries is not None:
        keep = set(countries)
        combos = {(c, a) for c, a in combos if c in keep}

    rows = []
    for country, asset in sorted(combos):
        hazards = set(applicable_hazards(asset, country))
        for scen in scenarios:
            if SCENARIO_HAZARD[scen] in hazards:
                rows.append((country, asset, scen))
    return pd.DataFrame(rows, columns=["country", "asset", "scenario"])


def _has_mapped_feature(path: Path, allowed: set[str]) -> bool:
    """True as soon as one feature carries a curve-mapped object_type."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=8192, columns=["object_type"]):
        if not allowed.isdisjoint(batch.column("object_type").to_pylist()):
            return True
    return False


def drop_unrunnable(expected: pd.DataFrame, ranked: pd.DataFrame,
                    cfg: dict) -> pd.DataFrame:
    """Drop expected rows whose (country, asset) has no curve-mapped feature.

    src/preprocess.py::drop_unmapped_object_types skips a combination when every
    one of its features has an object_type with no vulnerability curve, so that
    combination can never produce a Sobol row. It is "not runnable", like
    windstorm for roads, rather than "not finished yet" - counting it grey reads
    a permanent data gap as a queue that will eventually drain.

    Deciding this needs the exposure file itself, which is slow to open over the
    network share, so only pairs with NO Sobol row at all are probed: a pair that
    produced any row is runnable by demonstration. The cost is therefore
    proportional to the number of gaps, not to the size of the study.
    """
    key = ["country", "asset"]
    done = set(map(tuple, ranked[key].drop_duplicates().to_numpy()))
    gaps = sorted({tuple(t) for t in expected[key].drop_duplicates().to_numpy()}
                  - done)
    if not gaps:
        return expected

    expo = Path(cfg["exposure_dir"])
    unrunnable = []
    for country, asset in gaps:
        path = expo / f"{country}_{asset}_exposure.parquet"
        if not path.exists():
            continue
        if not _has_mapped_feature(path, set(mapped_object_types(get_asset_config(asset)))):
            unrunnable.append((country, asset))
    if not unrunnable:
        return expected

    print(f"  {len(unrunnable)} pair(s) have no curve-mapped feature at all and are "
          "excluded from the expected matrix rather than counted unfinished: "
          + ", ".join(f"{c}/{a}" for c, a in unrunnable))
    drop = set(unrunnable)
    keep = [t not in drop for t in map(tuple, expected[key].to_numpy())]
    return expected[keep].reset_index(drop=True)


RANGES_ALL_OUTCOMES = (PROJECT_ROOT / "overview_figures" / "ead_ranges"
                       / "EAD_Ranges_all_outcomes.csv")


def exposure_flags(path: Path = RANGES_ALL_OUTCOMES) -> pd.DataFrame | None:
    """(country, asset, hazard) -> was anything inside the RP100 footprint?

    None if the file is not there, which is not an error - it only costs the
    split between "no exposure" and "exposed but never damaged".

    exposed_qty_RP100_* is a physical quantity (m / m2 / count) and carries no
    uncertainty in this study - w90 is 0 on all 1,769 rows - so `max > 0` is an
    exact test rather than a summary. Earthquake has no exposed_qty outcome at
    all, but no zero-variance combination is an earthquake one, so nothing that
    needs the flag is missing it.
    """
    if not path.exists():
        return None
    allo = pd.read_csv(path)
    exp = allo[allo["outcome"].str.startswith("exposed_qty_RP100_")]
    if exp.empty:
        return None
    # Keyed on HAZARD, not scenario. What is inside the RP100 footprint depends
    # on the hazard map and the asset geometry, never on how the protection
    # standard is treated - verified across all 349 river combinations, where
    # exposed_qty_RP100_river is identical in every scenario. Keying on scenario
    # meant a scenario absent from this file (any newly added one) had no flag
    # at all; keying on hazard makes one computation serve every scenario of
    # that hazard, including ones added later.
    exp = exp.copy()
    exp["hazard"] = exp["outcome"].str.replace("exposed_qty_RP100_", "", regex=False)
    g = exp.groupby(["country", "asset", "hazard"], as_index=False)["max"].max()
    return g.rename(columns={"max": "exposed_qty_max"})


def ranked_drivers(workbook: Path, max_rank: int,
                   exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """Factors ranked by ST on total EAD, per (country, asset, scenario).

    A combination whose factors are ALL at ST = 0 has no driver at any rank -
    the sort is choosing between ties at zero - so every one of its rows is
    relabelled to a non-driver state instead. See the module docstring.
    """
    asi = pd.read_excel(workbook, sheet_name=SHEET)
    main = asi[asi["outcome"] == OUTCOME].copy()
    if main.empty:
        raise SystemExit(
            f"No {OUTCOME} rows in {workbook.name}::{SHEET} - "
            "run `python -m src.aggregate_results` first."
        )
    key = ["country", "asset", "scenario"]
    main = main.sort_values(key + ["ST"], ascending=[True, True, True, False])
    main["rank"] = main.groupby(key).cumcount() + 1

    # An experiment with fewer factors than the requested rank simply has no row
    # there; say so rather than letting it appear as a grey "unfinished" slice.
    n_factors = main.groupby(key)["rank"].max()
    for r in range(2, max_rank + 1):
        short = int((n_factors < r).sum())
        if short:
            print(f"  NOTE: {short} experiment(s) have fewer than {r} factors; "
                  f"absent from the rank-{r} figure.")

    # No variance to attribute: relabel the whole combination, every rank.
    no_var = main.groupby(key)["ST"].transform("max") <= 0
    main["no_variance"] = no_var
    main["category"] = main["factor_code"].map(category)
    if exposure is None:
        main.loc[no_var, "category"] = NO_DAMAGE
    else:
        main["hazard"] = main["scenario"].map(SCENARIO_HAZARD)
        main = main.merge(exposure, on=["country", "asset", "hazard"], how="left")
        no_var = main["no_variance"]
        # The flag is keyed on SCENARIO, so a scenario absent from
        # EAD_Ranges_all_outcomes.csv has no flag at all. Treating a missing
        # flag as "not exposed" silently relabels exposed-but-undamaged
        # combinations as unexposed, and exposure cannot differ by scenario for
        # one hazard anyway - so fall back to the pooled state and say so.
        known = main["exposed_qty_max"].notna()
        exposed = main["exposed_qty_max"].fillna(0) > 0
        main.loc[no_var & known & exposed, "category"] = EXPOSED_UNDAMAGED
        main.loc[no_var & known & ~exposed, "category"] = NO_EXPOSURE
        main.loc[no_var & ~known, "category"] = NO_DAMAGE
        blind = sorted(main.loc[no_var & ~known, "hazard"].dropna().unique())
        if blind:
            print(f"  NOTE: no exposure data for hazard(s) {', '.join(blind)} - their "
                  "zero-variance combinations cannot be split into 'no exposure' "
                  "vs 'exposed, never damaged'. Run `python -m src.ead_ranges` "
                  "to get the split.")
    n = int(main.loc[main["rank"] == 1, "no_variance"].sum())
    if n:
        by = main.loc[(main["rank"] == 1) & main["no_variance"], "category"].value_counts()
        print(f"  {n} combination(s) have every ST = 0 and are NOT given a driver: "
              + ", ".join(f"{v} {k.lower()}" for k, v in by.items()))

    out = main[main["rank"] <= max_rank].copy()
    return out[key + ["rank", "factor_code", "ST", "category", "no_variance"]]


def coverage_for_rank(expected: pd.DataFrame, ranked: pd.DataFrame, rank: int) -> pd.DataFrame:
    """Expected matrix left-joined with the rank-N driver; unmatched = unfinished."""
    at_rank = ranked[ranked["rank"] == rank]
    merged = expected.merge(at_rank, on=["country", "asset", "scenario"],
                            how="left", indicator=True)
    merged["covered"] = merged["_merge"] == "both"
    merged["dcat"] = merged["category"].map(display_category)
    return merged


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def slice_counts(sub: pd.DataFrame) -> tuple[dict, int, int]:
    """{slice label: count} for one scope, including the Unfinished remainder."""
    n_exp = len(sub)
    fin = sub[sub["covered"]]
    counts = {}
    for cat in SLICE_ORDER:
        if cat == "Unfinished":
            continue
        v = int((fin["dcat"] == cat).sum())
        if v:
            counts[cat] = v
    unfinished = n_exp - len(fin)
    if unfinished:
        counts["Unfinished"] = unfinished
    return counts, len(fin), n_exp


def draw_pie(ax, counts: dict, title: str, cover_txt: str, big: bool = False) -> None:
    labels = [s for s in SLICE_ORDER if s in counts]
    sizes = [counts[s] for s in labels]
    cols = [COLORS[s] for s in labels]
    if not sizes:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        ax.axis("off")
        return
    total = sum(sizes)
    wedges, _ = ax.pie(
        sizes, colors=cols, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor=SURFACE, linewidth=1.6), radius=1.0,
    )
    # "Exposed, never damaged" is the one non-driver state worth chasing, so it
    # is textured as well as coloured - three neutrals in one pie are otherwise
    # separated by lightness alone.
    for w, lab in zip(wedges, labels):
        if lab in HATCH:
            w.set_hatch(HATCH[lab])
            w.set_edgecolor(INK_2)
    for w, s, lab in zip(wedges, sizes, labels):
        if s / total < 0.05:
            continue
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        x, y = 0.6 * np.cos(ang), 0.6 * np.sin(ang)
        light = lab in ("Protection standard", "Vulnerability curve",
                        "Hazard intensity", "Cost / max-damage")
        ax.text(x, y, str(s), ha="center", va="center",
                fontsize=(13 if big else 10), fontweight="bold",
                color="#ffffff" if light else INK)
    ax.set_title(title, fontsize=(15 if big else 12), fontweight="bold",
                 color=INK, pad=(10 if big else 6))
    cap_y = (-0.04 if big else -0.02) if REPORT else (-0.14 if big else -0.10)
    ax.text(0.5, cap_y, cover_txt, transform=ax.transAxes,
            ha="center", va="top",
            fontsize=(12 if big else 10) if REPORT else (11 if big else 9),
            color=INK_2)


def coverage_text(fin: int, nexp: int, noun: str = "") -> str:
    """Per-pie caption under the wedges.

    Report mode prints just the completed count in brackets; the monitoring
    default also carries the expected total and the percentage, which is what
    makes an unfinished study legible while it runs.
    """
    if REPORT:
        return f"({fin})"
    pct = f"{100 * fin / nexp:.0f}%" if nexp else "-"
    return f"{fin}/{nexp}{noun} ({pct})"


def wrap_scen_name(scen: str) -> str:
    """Scenario label with its parenthetical qualifier moved to a second line.

    Keeps the seven side-by-side pies legible without truncating the qualifier,
    which is what distinguishes a flood pair from each other.
    """
    name = scenario_label(scen, short=True)
    head, sep, tail = name.partition(" (")
    return head + NL + "(" + tail if sep else head


def legend_handles(present: set[str] | None = None) -> list:
    """Legend entries, restricted to the slices actually drawn on this figure.

    With every factor shown separately the full list runs to eight. Dropping the
    ones with no wedge keeps the legend to a single row and stops it implying
    categories that never occur.
    """
    labels = [s for s in SLICE_ORDER if present is None or s in present]
    if REPORT and "Unfinished" in labels:
        # Report mode assumes a finished study. If unfinished wedges exist the
        # legend must still explain them, or the figure shows a grey slice with
        # no key - so keep the entry and say so rather than quietly dropping it.
        print("  WARNING: --report but unfinished experiments are present; keeping "
              "'Unfinished' in the legend. Rebuild the workbook, or the figure "
              "understates coverage.")
    return [Patch(facecolor=COLORS[s], edgecolor=INK_2 if s in HATCH else SURFACE,
                  hatch=HATCH.get(s),
                  label=REPORT_LEGEND_LABEL.get(s, s) if REPORT else s)
            for s in labels]


# Display-only legend text. The pie category stays "Vulnerability curve" as the
# data key; the report legend spells out both kinds because the study uses both:
# flood and windstorm read damage-fraction VULNERABILITY curves (F_Vuln_Depth,
# W_Vuln_V10m_3sec), while earthquake reads FRAGILITY curves (E_Frag_PGA,
# exceedance probability per damage state) collapsed to an EDR.
REPORT_LEGEND_LABEL = {"Vulnerability curve": "Vulnerability / fragility curve"}


def legend_layout(handles: list) -> dict:
    """ncol/fontsize for fig.legend.

    Report mode splits the entries over two rows, which buys a larger font
    without widening the figure; monitoring keeps them on one row.
    """
    if REPORT:
        return {"ncol": int(np.ceil(len(handles) / 2)), "fontsize": 13}
    return {"ncol": min(len(handles), 8), "fontsize": 11}


def scenario_figure(merged: pd.DataFrame, scen: str, rank: int, out_dir: Path) -> Path:
    """All-infrastructure pie plus one pie per asset, for one scenario and rank."""
    sc = merged[merged["scenario"] == scen]
    assets = sorted(sc["asset"].unique())
    ncols = 4
    nrows = int(np.ceil((1 + len(assets)) / ncols))
    cell = 3.5 if REPORT else 4.0
    colw = 3.5 if REPORT else 4.1
    fig = plt.figure(figsize=(colw * ncols, cell * nrows + (1.7 if REPORT else 1.4)))
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=0.30 if REPORT else 0.55, wspace=0.06 if REPORT else 0.25,
        top=0.915 if REPORT else 0.88, bottom=0.105 if REPORT else 0.14,
        left=0.02 if REPORT else 0.04, right=0.98 if REPORT else 0.96,
    )

    present: set[str] = set()
    ax0 = fig.add_subplot(gs[0, 0])
    counts, fin, nexp = slice_counts(sc)
    present |= set(counts)
    draw_pie(ax0, counts, "ALL INFRASTRUCTURE",
             coverage_text(fin, nexp, " experiments done"), big=True)

    for i, a in enumerate(assets, start=1):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        counts, fin, nexp = slice_counts(sc[sc["asset"] == a])
        present |= set(counts)
        draw_pie(ax, counts, a, coverage_text(fin, nexp, " countries"))

    fin_all, n_all = int(sc["covered"].sum()), len(sc)
    fig.suptitle(f"{RANK_WORD[rank]} influential parameter - {scenario_label(scen)}",
                 fontsize=19, fontweight="bold", color=INK,
                 y=0.972 if REPORT else 0.965)
    # Report mode carries no explainer: the caption belongs in the report text,
    # where it can be written once instead of repeated on every figure.
    if not REPORT:
        fig.text(0.5, 0.915,
                 f"{RANK_BLURB[rank]} Sobol total-effect driver of total EAD, per country. "
                 f"Coverage so far: {fin_all}/{n_all} experiments "
                 f"({100 * fin_all / n_all:.0f}%)."
                 f"{NL}Neutral slices are not drivers: pale grey = not finished, and the "
                 f"two darker greys are combinations whose Sobol indices are ALL zero.",
                 ha="center", fontsize=12, color=INK_2)
    handles = legend_handles(present)
    fig.legend(handles=handles, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.015 if REPORT else 0.02),
               **legend_layout(handles))
    out = out_dir / f"drivers_{scen}{RANK_SUFFIX[rank]}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def overview_figure(merged: pd.DataFrame, scenarios: list[str], rank: int,
                    out_dir: Path, ncols: int = 4, asset: str | None = None) -> Path:
    """One pie per scenario, wrapped into a grid.

    With `asset` set the figure covers that infrastructure type alone, which is
    the cross-scenario view for a single network - the same question the
    all-infrastructure overview answers, narrowed to one asset.

    Seven scenarios on a single row made a figure nearly 30 inches wide. The
    header and legend bands are sized in INCHES and converted to figure
    fractions here, so the layout holds for whatever number of rows the scenario
    count produces rather than needing hand-tuned fractions per case.
    """
    if asset is not None:
        merged = merged[merged["asset"] == asset]
        if merged.empty:
            raise SystemExit(f"No rows for asset '{asset}'.")
    scope = asset if asset is not None else "all infrastructure"
    n = len(scenarios)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    # The subtitle is built before the geometry because the header has to be
    # tall enough for it: it runs to two or three lines depending on the
    # scenario set, and a fixed header put the last line on top of the pie
    # titles in the first row.
    subtitle = ""
    if not REPORT:
        subtitle = (f"Share of experiments whose {RANK_BLURB[rank].lower()} Sobol driver of "
                    "total EAD is each parameter." + NL +
                    "Neutral slices are not drivers: pale grey = not finished, the darker "
                    "greys = every Sobol index is zero, so there is nothing to attribute.")
        if any("noprot" in s for s in scenarios):
            subtitle += (NL + "In the 'fixed' scenarios the protection standard is held "
                         "constant, so it cannot appear as a driver there.")
    # footer must clear BOTH the legend and the per-pie coverage caption that
    # hangs below the bottom row of axes, or the two collide.
    if REPORT:
        # Two-row legend needs more than one row did, so the footer shrinks only
        # a little; the real savings are the gaps, margins and shorter header.
        # No subtitle, but the band must still clear the suptitle AND the
        # two-line scenario titles that sit above the top row of axes.
        header = 1.15
        footer, cell = 1.35, 3.5
    else:
        header = 1.35 + 0.24 * (subtitle.count(NL) + 1)
        footer, cell = 1.7, 4.0
    h = cell * nrows + header + footer
    fig = plt.figure(figsize=((3.5 if REPORT else 4.1) * ncols, h))
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=0.30 if REPORT else 0.5, wspace=0.06 if REPORT else 0.25,
        top=1 - header / h, bottom=footer / h,
        left=0.02 if REPORT else 0.04, right=0.98 if REPORT else 0.96,
    )

    present: set[str] = set()
    for j, scen in enumerate(scenarios):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        counts, fin, nexp = slice_counts(merged[merged["scenario"] == scen])
        present |= set(counts)
        draw_pie(ax, counts, wrap_scen_name(scen), coverage_text(fin, nexp))

    fig.suptitle(
        f"{RANK_WORD[rank]} influential parameter across {scope} - by scenario",
        fontsize=17, fontweight="bold", color=INK, y=1 - 0.40 / h)
    if subtitle:
        fig.text(0.5, 1 - 0.82 / h, subtitle, ha="center", va="top",
                 fontsize=11.5, color=INK_2)
    handles = legend_handles(present)
    fig.legend(handles=handles, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.14 / h), **legend_layout(handles))
    # Per-asset overviews live in their own folder, so everything for one asset
    # (these plus src/eu_totals.py's tables) sits together.
    tag = "" if asset is None else f"_{asset}"
    target_dir = out_dir if asset is None else out_dir / asset
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"drivers_ALL_scenarios_overview{tag}{RANK_SUFFIX[rank]}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def report_unexpected(expected: pd.DataFrame, ranked: pd.DataFrame,
                      scenarios: list[str]) -> None:
    """Warn if a finished experiment is not in the expected matrix.

    A non-zero count means the applicability rules applied here disagree with
    what the study actually ran - rule or scenario-name drift worth fixing, not
    a data problem. Cheap to check and it fails loudly rather than quietly
    under-reporting coverage.
    """
    key = ["country", "asset", "scenario"]
    done = ranked[(ranked["rank"] == 1) & ranked["scenario"].isin(scenarios)][key]
    extra = done.drop_duplicates().merge(expected, on=key, how="left", indicator=True)
    rogue = extra[extra["_merge"] == "left_only"]
    if len(rogue):
        print(f"  WARNING: {len(rogue)} finished experiment(s) are NOT in the expected "
              "matrix (applicability-rule mismatch):")
        for _, r in rogue.head(10).iterrows():
            print(f"     {r['country']}/{r['asset']}/{r['scenario']}")
    else:
        print("  applicability check: every finished experiment is in the expected matrix")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", default=None,
                        help=f"summary workbook (default: {PROJECT_ROOT / WORKBOOK_NAME})")
    parser.add_argument("--out-dir", default=None,
                        help=f"output folder (default: {PROJECT_ROOT / 'overview_figures'})")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help=f"scenarios to draw (default: {' '.join(DEFAULT_OVERVIEW_SCENARIOS)})")
    parser.add_argument("--asset-overviews", nargs="*", default=["power"],
                        help="also draw a cross-scenario overview for each named asset "
                             "(default: power); pass with no names to skip")
    parser.add_argument("--overview-cols", type=int, default=4,
                        help="columns in the cross-scenario overview grid (default: 4)")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2],
                        help="which driver ranks to draw, 1 = most influential (default: 1 2)")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="restrict every figure to these ISO3 countries. Use it when "
                             "comparing a scenario that was only run for a subset against "
                             "one that covers the whole study.")
    parser.add_argument("--report", action="store_true",
                        help="restyle for a written report: bare (n) coverage counts, no "
                             "Unfinished legend entry, two-row legend, tighter margins. "
                             "Writes to <out-dir>/report/ unless --out-dir is given.")
    args = parser.parse_args()

    global REPORT, SURFACE
    REPORT = args.report
    if REPORT:
        # Pure white so the PNG blends into a report page instead of sitting on
        # a faintly off-white tile. SURFACE is read at draw time by the wedge
        # separators and the legend patch edges as well as the rcParams below,
        # so swapping it here covers every use.
        SURFACE = "#ffffff"

    scenarios = sorted(args.scenarios or DEFAULT_OVERVIEW_SCENARIOS, key=scenario_sort_key)
    unknown = [s for s in scenarios if s not in SCENARIO_HAZARD]
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {unknown}; known: {sorted(SCENARIO_HAZARD)}")
    ranks = sorted(set(args.ranks))
    bad = [r for r in ranks if r not in RANK_WORD]
    if bad:
        raise SystemExit(f"--ranks must be within {sorted(RANK_WORD)}; got {bad}")

    workbook = Path(args.workbook) if args.workbook else PROJECT_ROOT / WORKBOOK_NAME
    if not workbook.exists():
        raise SystemExit(f"No workbook at {workbook} - run `python -m src.aggregate_results`.")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "overview_figures"
        if REPORT:
            # Report copies sit beside the monitoring set, never on top of it.
            out_dir = out_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "text.color": INK,
    })

    cfg = load_config()
    expected = expected_matrix(cfg, scenarios, args.countries)
    if args.countries:
        print(f"restricted to {len(args.countries)} countries: {' '.join(sorted(args.countries))}")
    exposure = exposure_flags()
    if exposure is None:
        print(f"  NOTE: {RANGES_ALL_OUTCOMES.name} not found - combinations with no "
              "variance are pooled into one slice instead of being split into "
              "'no exposure' and 'exposed, never damaged'. Run "
              "`python -m src.ead_ranges` to get the split.")
    ranked = ranked_drivers(workbook, max_rank=max(ranks), exposure=exposure)
    expected = drop_unrunnable(expected, ranked, cfg)
    print(f"expected experiments: {len(expected)}")
    report_unexpected(expected, ranked, scenarios)

    for rank in ranks:
        merged = coverage_for_rank(expected, ranked, rank)
        fin, tot = int(merged["covered"].sum()), len(merged)
        print(f"\n=== rank {rank} ({RANK_WORD[rank].lower()} influential) - "
              f"coverage {fin}/{tot} ({100 * fin / tot:.1f}%) ===")
        for scen in scenarios:
            sub = merged[(merged["scenario"] == scen) & merged["covered"]]
            vc = sub["dcat"].value_counts()
            print(f"  {scen}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
            print(f"    wrote {scenario_figure(merged, scen, rank, out_dir).name}")
        print(f"    wrote {overview_figure(merged, scenarios, rank, out_dir, args.overview_cols).name}")
        for a in args.asset_overviews:
            if a not in set(merged["asset"]):
                print(f"    SKIP asset overview for '{a}': no rows")
                continue
            print(f"    wrote {overview_figure(merged, scenarios, rank, out_dir, args.overview_cols, asset=a).name}")

    print(f"\nAll figures in {out_dir}")


if __name__ == "__main__":
    main()
