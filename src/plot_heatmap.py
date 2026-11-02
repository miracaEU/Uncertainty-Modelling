"""The whole study on one page: countries x (asset, scenario), coloured.

Two grids, both 36 countries down the side and every (asset, scenario) column
the study actually ran across the top:

    Spread_heatmap.png    colour = p95 / p5 of total EAD - how wide the
                          full-uncertainty band is. The headline grid.
    Zeros_heatmap.png     colour = the fraction of draws that produced no
                          damage at all. The companion, because a quarter of
                          the study's cells are zero-inflated and the spread
                          grid deliberately refuses to put a number on those.

Why a grid at all
-----------------
src/ead_ranges.py draws 2,116 range bars across 132 figures and
src/country_ranges.py already pivots the same numbers into Spread_matrix /
Width_matrix - but only into Excel, one workbook per asset, which is where
cross-asset structure goes to hide. The whole point of a study this size is the
block structure: airports-under-windstorm being empty everywhere, gas-under-
coastal being empty in the small countries, earthquake being the one hazard
whose spread barely varies. None of that is visible one figure at a time.

Column order is asset, then src/plot_pyramid.py's scenario order, with a gap
between assets. Row order is total median EAD across everything, descending, so
the countries that carry the European total are at the top - the same ordering
rule as src/country_ranges.py::country_order.

The colour vocabulary (ramp, bin edges, the p5 = 0 hatch, the "no damage in any
draw" grey) is imported from src/plot_maps.py rather than restated, so the maps
and these grids cannot drift apart.

Usage:
    python -m src.plot_heatmap
    python -m src.plot_heatmap --assets power roads
"""

from __future__ import annotations

import argparse
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

# Imported as a MODULE, not by name, for everything use_palette/use_measure can
# rebind: `from .plot_maps import SPREAD_RAMP` copies the value at import, which
# would pin these grids to the blue ramp and the p95/p5 edges no matter what the
# flags say. The names below really are constant, so they are safe to bind.
from . import plot_maps as pm
from .plot_maps import (
    BORDER,
    CLS_DEGENERATE,
    CLS_EMPTY,
    CLS_MISSING,
    EMPTY_FILL,
    MISSING_FILL,
    P5_ZERO_HATCH,
    RANGES_CSV,
    spread_class,
    spread_legend_handles,
)
from .plot_pyramid import (
    INK,
    INK_2,
    MUTED,
    PROJECT_ROOT,
    SURFACE,
    scenario_label,
    scenario_sort_key,
)

OUT_DIR = PROJECT_ROOT / "overview_figures" / "heatmaps"

# Cell geometry. 0.20 in per column holds 66 columns in a 13-inch figure, which
# is a readable width in print and on a screen without scrolling.
CELL_W = 0.20
CELL_H = 0.20

# Set from main(). PORTRAIT transposes the grid - countries across the top,
# the (asset, scenario) columns down the side - which turns a 16x11 landscape
# sheet into a tall page that fits a report column. TEXT_SCALE grows every
# font, and the cells with them, so the tick labels keep their spacing.
PORTRAIT = False
TEXT_SCALE = 1.0
# Cell size is separate from font size: the cell sets how tall the sheet is,
# the font sets how readable it is, and they trade off against each other.
CELL_SCALE = 1.0
# Black text throughout and no explanatory line above the legend.
PLAIN = False
BAND_PAD = 0.35


def _band_geometry():
    """Width of the group band, and the font and wrap that fit inside it.

    Rotated onto the side, a band label runs ALONG its band, so the band's
    extent caps the label length, not the band's width. Wrapping onto two
    lines halves the length each line has to fit in, which buys roughly a
    70% larger font - the band then has to be wide enough to hold the lines.
    """
    target = 7.5 * TEXT_SCALE
    lines = 2
    width_in = lines * target * 1.35 / 72.0
    return max(0.75, width_in / (CELL_W * CELL_SCALE)), target, lines


# Scenario wording for these grids.
#
# SCENARIO_LABEL_SHORT spells out the sampling bracket and flags sea-level
# rise, which a pyramid needs when it puts two brackets side by side. Here the
# label has to fit inside a narrow band on a tall sheet, so it carries the
# hazard and how the protection standard was treated and nothing else, worded
# the same way for every scenario so the bands read as one list. Sea-level rise
# is the default coastal run, so it needs no qualifier of its own.
#
# Dropping those qualifiers can make two scenarios read alike - the headline
# seven never do, the full set does. The second entry is what sets a scenario
# apart from the others sharing its wording, and band_labels() adds it only to
# the ones that actually clash. The label then grows where it has to without
# the grid changing how it words anything else.
SCENARIO_BAND_LABEL = {
    #  scenario:                     (label, what sets this one apart)
    "flood_absprot": ("River flood (protection sampled)", "no depth error"),
    "flood_absprot_ds": ("River flood (protection sampled)", "RP 5-200"),
    "flood_absprot_ds_rp10to100": ("River flood (protection sampled)", "RP 10-100"),
    "flood_baseline": ("River flood (protection scaled)", "no depth error"),
    "flood_baseline_ds": ("River flood (protection scaled)", "x0-2"),
    "flood_baseline_ds_sc0p5to1p5": ("River flood (protection scaled)", "x0.5-1.5"),
    "flood_noprot": ("River flood (protection fixed)", "no depth error"),
    "flood_noprot_ds": ("River flood (protection fixed)", "depth error sampled"),
    "coastal_absprot": ("Coastal flood (protection sampled)", "no depth error"),
    "coastal_absprot_ds": ("Coastal flood (protection sampled)", "no sea-level rise"),
    "coastal_absprot_ds_rp10to100": ("Coastal flood (protection sampled)",
                                     "RP 10-100, no sea-level rise"),
    "coastal_absprot_ds_slr": ("Coastal flood (protection sampled)", "with sea-level rise"),
    "coastal_baseline": ("Coastal flood (protection scaled)", "no depth error"),
    "coastal_baseline_ds": ("Coastal flood (protection scaled)", "x0-2"),
    "coastal_baseline_ds_sc0p5to1p5": ("Coastal flood (protection scaled)", "x0.5-1.5"),
    "coastal_noprot": ("Coastal flood (protection fixed)", "no depth error"),
    "coastal_noprot_ds": ("Coastal flood (protection fixed)", "no sea-level rise"),
    "coastal_noprot_ds_slr": ("Coastal flood (protection fixed)", "with sea-level rise"),
    "earthquake": ("Earthquake", ""),
    # Windstorm's factor is a design standard rather than a flood protection
    # standard, so the noun differs; the sentence shape does not.
    "windstorm": ("Windstorm (design fixed)", "RP50"),
    "windstorm_absprot": ("Windstorm (design sampled)", "RP 25-200"),
}


def band_labels(scenarios) -> dict[str, str]:
    """Label per scenario, worded alike and unique within the set being drawn."""
    pairs = {s: SCENARIO_BAND_LABEL.get(s, (scenario_label(s, short=True), ""))
             for s in scenarios}
    seen = Counter(base for base, _ in pairs.values())
    out = {}
    for s, (base, qual) in pairs.items():
        if seen[base] > 1 and qual:
            base = f"{base[:-1]}, {qual})" if base.endswith(")") else f"{base} ({qual})"
        out[s] = base
    return out


ASSET_GAP = 0.9  # columns of blank space between one asset block and the next

# zero_fraction bins. 0 and 1 are their own classes, not ends of a ramp: "damage
# in every single draw" and "damage in not one draw" are qualitative states, and
# 1.0 is exactly the "no damage in any draw" case the other figures flag.
ZERO_EDGES = (0.05, 0.20, 0.40, 0.60, 0.90)
ZERO_LABELS = ("under 5%", "5 - 20%", "20 - 40%", "40 - 60%", "60 - 90%", "90 - under 100%")
NEVER_ZERO_FILL = "#f7f6f1"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def column_layout(ranges: pd.DataFrame, group_by: str = "asset") -> tuple[list, list, dict]:
    """(columns, blocks, x_of) - one x position per (asset, scenario).

    Blocks are separated by ASSET_GAP columns of blank, so the eye reads a
    handful of blocks rather than one 66-wide run. `blocks` carries
    (label, x0, x1) for the label band and the separators.

    group_by picks which of the two keys forms the blocks:
      asset    ten infrastructure blocks, scenarios within each. Shows how one
               asset behaves across hazards.
      hazard   one block per scenario, assets within each. Shows how one hazard
               behaves across infrastructure.
    The cells are identical either way; only the column order differs, and
    which structure is adjacent enough to see.
    """
    columns, blocks, x_of = [], [], {}
    x = 0.0
    if group_by == "hazard":
        outer = sorted(ranges["scenario"].unique(), key=scenario_sort_key)
        labels = band_labels(outer)
        inner_of = lambda scen: sorted(  # noqa: E731
            ranges.loc[ranges["scenario"] == scen, "asset"].unique())
        key_of = lambda scen, asset: (asset, scen)  # noqa: E731
        label_of = lambda scen: labels[scen]  # noqa: E731
    else:
        outer = sorted(ranges["asset"].unique())
        inner_of = lambda asset: sorted(  # noqa: E731
            ranges.loc[ranges["asset"] == asset, "scenario"].unique(),
            key=scenario_sort_key)
        key_of = lambda asset, scen: (asset, scen)  # noqa: E731
        label_of = lambda asset: asset  # noqa: E731

    for head in outer:
        x0 = x
        for item in inner_of(head):
            key = key_of(head, item)
            columns.append(key)
            x_of[key] = x
            x += 1.0
        blocks.append((label_of(head), x0, x - 1.0))
        x += ASSET_GAP
    return columns, blocks, x_of


def drop_partial_scenarios(ranges: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop scenarios that ran on only a SUBSET of their hazard's countries.

    Same rule as eu_totals.drop_partial_coverage, and the same reason: the
    range-test scenarios exist to show how an assumed parameter range moves a
    factor's importance, not to describe Europe. On this grid they are not
    wrong so much as enormous - four of them add 40 columns while covering 7
    to 10 countries of 36, which is most of the near-empty space in the sheet.

    Judged per hazard, because coastal covering 26 of 36 is correct.
    """
    from .ema_model import SCENARIO_HAZARD
    hz = ranges["scenario"].map(SCENARIO_HAZARD)
    best = ranges.assign(_hz=hz).groupby("_hz")["country"].nunique()
    keep, dropped = [], []
    for scen, g in ranges.groupby("scenario"):
        h = SCENARIO_HAZARD.get(scen)
        if h is None or g["country"].nunique() >= int(best[h]):
            keep.append(scen)
        else:
            dropped.append((scen, g["country"].nunique(), int(best[h])))
    if dropped and verbose:
        print("  excluding subset test scenarios (--allow-partial keeps them):")
        for scen, n, full in sorted(dropped):
            print(f"      {scen:34s} {n:2d} of {full} countries")
    return ranges[ranges["scenario"].isin(keep)]


def row_order(ranges: pd.DataFrame) -> list[str]:
    """Countries by total median EAD across every combination, largest first."""
    tot = ranges.groupby("country")["median"].sum().sort_values(ascending=False)
    return list(tot.index)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _fit_band_labels(ax, band_texts, n_rows, bw, margin=0.94):
    """Wrap and shrink every band label until it fits inside its own band.

    Neither the wrap nor the size can be settled up front. A cell is CELL_H
    inches in the figure's own arithmetic only: constrained layout then takes
    a few inches off the axes for the title and the legend, and the axes carry
    limits this function has to be called AFTER, so a label wrapped from the
    cell size alone overruns the grey box it sits on. The one reliable
    comparison is drawn text against drawn band - so draw, re-wrap to the room
    that actually came out, and shrink if the lines still do not fit. Rotated,
    a label's length is its display HEIGHT and its stack of lines is its
    WIDTH, so both directions have to be checked.
    """
    fig = ax.figure

    def span(a, b, axis):
        tr = ax.transData.transform
        return abs(tr(b)[axis] - tr(a)[axis])

    for _ in range(4):               # a re-wrap moves the layout a little
        fig.canvas.draw()
        across = span((n_rows + BAND_PAD, 0.0), (n_rows + BAND_PAD + bw, 0.0), 0) * margin
        worst = 1.0
        for t, raw, x0, x1 in band_texts:
            along = span((0.0, x0), (0.0, x1 + 1.0), 1) * margin
            per_line = max(6, int(along / (t.get_fontsize() * 0.58)))
            t.set_text(chr(10).join(textwrap.wrap(raw, per_line) or [raw]))
            ext = t.get_window_extent()
            if ext.height > along:
                worst = min(worst, along / ext.height)
            if ext.width > across:
                worst = min(worst, across / ext.width)
        if worst >= 0.995:
            break
        # One factor for every label, so the bands keep a common type size.
        for t, *_ in band_texts:
            t.set_fontsize(max(4.5, t.get_fontsize() * worst))


def _draw_grid(ax, codes, countries, columns, blocks, x_of, cmap, hatch_code=None,
               group_by="asset"):
    """Cells as one Rectangle each, plus the block band and the separators.

    Rectangles rather than imshow because the columns are not evenly spaced -
    the gap between asset blocks is part of the layout - and because the p5 = 0
    class needs a per-cell hatch, which an image cannot carry.
    """
    # One place decides orientation: a cell is (column position, country index)
    # in landscape and the transpose of that in portrait.
    def at(x, i):
        return (i, x) if PORTRAIT else (x, i)

    for j, key in enumerate(columns):
        x = x_of[key]
        for i, _ in enumerate(countries):
            code = codes[i, j]
            if code == CLS_MISSING:
                continue
            ax.add_patch(Rectangle(at(x, i), 1.0, 1.0, facecolor=cmap(code),
                                   edgecolor=SURFACE, linewidth=0.5, zorder=2))
            if hatch_code is not None and code == hatch_code:
                ax.add_patch(Rectangle(at(x, i), 1.0, 1.0, facecolor="none",
                                       edgecolor=pm.P5_ZERO_HATCH_COLOR,
                                       hatch=P5_ZERO_HATCH, linewidth=0.0, zorder=3))

    n_rows = len(countries)
    band_texts = []
    for label, x0, x1 in blocks:
        if PORTRAIT:
            # The band runs down the right-hand side, so its label reads along
            # it rather than across the cells.
            bw, target, _ = _band_geometry()
            ax.add_patch(Rectangle((n_rows + BAND_PAD, x0), bw, x1 - x0 + 1.0,
                                   facecolor="#e8e7e0", edgecolor="none", zorder=2))
            # Wrapped by _fit_band_labels once the axes limits are final.
            t = ax.text(n_rows + BAND_PAD + bw / 2, (x0 + x1 + 1.0) / 2,
                        label, ha="center", va="center",
                        rotation=90, fontsize=target, color=INK, zorder=3,
                        linespacing=1.15)
            band_texts.append((t, label, x0, x1))
        else:
            ax.add_patch(Rectangle((x0, n_rows + 0.35), x1 - x0 + 1.0, 0.75,
                                   facecolor="#e8e7e0", edgecolor="none", zorder=2))
            ax.text((x0 + x1 + 1.0) / 2, n_rows + 0.72, label, ha="center",
                    va="center", fontsize=7.5 * TEXT_SCALE, color=INK, zorder=3)

    # The band carries the grouping key, so the per-column tick carries the
    # OTHER one - otherwise every tick inside a block repeats the band's text.
    if group_by == "hazard":
        ticks = [a for a, _ in columns]
    else:
        labels = band_labels({s for _, s in columns})
        ticks = [labels[s] for _, s in columns]
    span = max(x_of.values()) + 1.4
    col_at = [x_of[k] + 0.5 for k in columns]
    if PORTRAIT:
        ax.set_xticks(np.arange(n_rows) + 0.5, countries,
                      rotation=90, fontsize=7 * TEXT_SCALE,
                      color=INK if PLAIN else MUTED)
        ax.set_yticks(col_at, ticks, fontsize=6 * TEXT_SCALE,
                      color=INK if PLAIN else INK_2)
        bw, _, _ = _band_geometry()
        ax.set_xlim(-0.4, n_rows + BAND_PAD + bw + 0.4)
        # First column at the top, and the band sits off the right-hand edge.
        ax.set_ylim(span, -0.4)
    else:
        ax.set_yticks(np.arange(n_rows) + 0.5, countries, fontsize=7 * TEXT_SCALE,
                      color=INK if PLAIN else MUTED)
        ax.set_xticks(col_at, ticks, rotation=90, fontsize=6 * TEXT_SCALE,
                      color=INK if PLAIN else INK_2)
        ax.set_xlim(-0.4, span)
        # Row 0 at the top; the extra 1.4 at the top is the asset band.
        ax.set_ylim(n_rows + 1.4, 0)
    # Only now are the limits final, so only now can a band be measured.
    if band_texts:
        _fit_band_labels(ax, band_texts, n_rows, _band_geometry()[0])
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _figure(n_rows: int, n_cols_span: float):
    # Cells deliberately do NOT grow with the text. A cell is 0.20 in = 14 pt
    # of space for a tick label that is 10 pt even at TEXT_SCALE 1.7, so the
    # labels have room already; scaling the cells too turned a 126-column
    # portrait sheet into fifty inches of paper.
    cw, ch = CELL_W * CELL_SCALE, CELL_H * CELL_SCALE
    if PORTRAIT:
        bw, _, _ = _band_geometry()
        size = (cw * (n_rows + bw) + 2.6, ch * n_cols_span + 3.9)
    else:
        size = (cw * n_cols_span + 2.6, ch * n_rows + 3.9)
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    # NOTE: a tall portrait sheet makes constrained layout emit "axes sizes
    # collapsed to zero" and fall back to the default subplot position.
    # Tightening its pads does not help. The figure is still correct, because
    # the figsize arithmetic above happens to land close to those default
    # margins, and because _fit_band_labels measures the SAME drawn state that
    # savefig then writes - so what is measured is what is saved. It is a
    # coincidence worth replacing with explicit margins if this is ever
    # reworked; the blocker is that the legend uses an "outside" location,
    # which only constrained layout supports.
    return fig, ax


def _legend(fig, handles, ncol, subtitle):
    if PORTRAIT:
        ncol = min(ncol, 3)
        subtitle = textwrap.fill(subtitle, 78)
    leg = fig.legend(handles=handles, frameon=False, fontsize=8 * TEXT_SCALE,
                     ncol=ncol, loc="outside lower center",
                     title=None if PLAIN else subtitle, alignment="center")
    if not PLAIN:
        leg.get_title().set(fontsize=8.2 * TEXT_SCALE, color=MUTED)
    for t in leg.get_texts():
        t.set_color(INK if PLAIN else INK_2)


def plot_spread_heatmap(ranges: pd.DataFrame, out: Path, group_by: str = "asset") -> None:
    countries = row_order(ranges)
    columns, blocks, x_of = column_layout(ranges, group_by)
    # The whole row, because which statistics spread_class needs depends on
    # --measure: p5 and p95 for the ratio, the median too for relwidth, p25 and
    # p75 for iqr.
    idx = {(r["country"], r["asset"], r["scenario"]): r
           for r in ranges.to_dict("records")}

    codes = np.full((len(countries), len(columns)), CLS_MISSING, dtype=int)
    for i, country in enumerate(countries):
        for j, (asset, scen) in enumerate(columns):
            hit = idx.get((country, asset, scen))
            if hit is not None:
                codes[i, j] = spread_class(hit)

    palette = {CLS_EMPTY: EMPTY_FILL, CLS_DEGENERATE: pm.P5_ZERO_FILL,
               CLS_MISSING: MISSING_FILL}
    palette.update({k: c for k, c in enumerate(pm.SPREAD_RAMP)})

    fig, ax = _figure(len(countries), max(x_of.values()) + 1)
    _draw_grid(ax, codes, countries, columns, blocks, x_of, palette.get,
               hatch_code=CLS_DEGENERATE, group_by=group_by)

    n = int((codes != CLS_MISSING).sum())
    fig.suptitle("How wide is the plausible range? - every combination in the study",
                 fontsize=13 * TEXT_SCALE)
    grouped = "hazard, infrastructure within" if group_by == "hazard" else \
              "infrastructure, hazard within"
    _legend(fig, spread_legend_handles(include_missing=False), 8,
            f"{pm.SPREAD_SUBTITLE}  |  {n:,} combinations, "
            f"{len(countries)} countries x {len(columns)} columns grouped by {grouped}"
            f"  |  blank = not run  |  countries ordered by total median EAD")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_zero_heatmap(ranges: pd.DataFrame, out: Path, group_by: str = "asset") -> None:
    countries = row_order(ranges)
    columns, blocks, x_of = column_layout(ranges, group_by)
    idx = {(r.country, r.asset, r.scenario): r.zero_fraction for r in ranges.itertuples()}

    # 0 -> code -2, 1.0 -> code -1, everything between -> 0..5 on the ramp.
    NEVER, ALWAYS = -2, -1
    codes = np.full((len(countries), len(columns)), CLS_MISSING, dtype=int)
    for i, country in enumerate(countries):
        for j, (asset, scen) in enumerate(columns):
            zf = idx.get((country, asset, scen))
            if zf is None or not np.isfinite(zf):
                continue
            if zf <= 0:
                codes[i, j] = NEVER
            elif zf >= 1.0:
                codes[i, j] = ALWAYS
            else:
                codes[i, j] = int(np.searchsorted(ZERO_EDGES, zf, side="right"))

    palette = {NEVER: NEVER_ZERO_FILL, ALWAYS: EMPTY_FILL, CLS_MISSING: MISSING_FILL}
    palette.update({k: c for k, c in enumerate(pm.SPREAD_RAMP)})

    fig, ax = _figure(len(countries), max(x_of.values()) + 1)
    _draw_grid(ax, codes, countries, columns, blocks, x_of, palette.get,
               group_by=group_by)

    handles = [Patch(facecolor=NEVER_ZERO_FILL, edgecolor=BORDER, linewidth=0.4,
                     label="0% - damage in every draw")]
    handles += [Patch(facecolor=c, edgecolor=BORDER, linewidth=0.4, label=lab)
                for c, lab in zip(pm.SPREAD_RAMP, ZERO_LABELS)]
    handles.append(Patch(facecolor=EMPTY_FILL, edgecolor=BORDER, linewidth=0.4,
                         label="100% - no damage in any draw"))

    n_always = int((codes == ALWAYS).sum())
    n_any = int(((codes != CLS_MISSING) & (codes != NEVER)).sum())
    fig.suptitle("How often is there no damage at all?", fontsize=13 * TEXT_SCALE)
    _legend(fig, handles, 8,
            f"colour = share of the {int(ranges['n'].max()):,} draws that produced zero "
            f"damage  |  {n_any:,} combinations have at least one zero draw, "
            f"{n_always:,} have nothing but  |  blank = not run")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranges", default=None, help=f"default: {RANGES_CSV}")
    ap.add_argument("--out-dir", default=None, help=f"default: {OUT_DIR}")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--group-by", nargs="+", choices=["asset", "hazard"],
                    default=["asset", "hazard"],
                    help="which key forms the column blocks. Both by default, "
                         "written as *_by_infra.png and *_by_hazard.png - the "
                         "same cells, reordered so a different structure is "
                         "adjacent enough to read.")
    ap.add_argument("--palette", choices=["blue", "spectral", "traffic"], default="blue",
                    help="see src/plot_maps.py --palette.")
    ap.add_argument("--measure", choices=["ratio", "relwidth", "iqr"], default="ratio",
                    help="see src/plot_maps.py --measure. Affects the spread "
                         "grid only, not the zero grid.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="keep scenarios that only ran on a subset of their "
                         "hazard's countries (the range tests). Excluded by "
                         "default - they are mostly blank columns.")
    ap.add_argument("--headline", action="store_true",
                    help="restrict to eu_totals.HEADLINE_SCENARIOS, the seven "
                         "comparable scenarios the report leads with")
    ap.add_argument("--cell-scale", type=float, default=1.0,
                    help="multiplier on cell size, which sets the sheet's "
                         "overall height (default 1.0)")
    ap.add_argument("--plain", action="store_true",
                    help="black text throughout and no explanatory line "
                         "between the grid and the legend")
    ap.add_argument("--portrait", action="store_true",
                    help="transpose the grid: countries across the top, the "
                         "(asset, scenario) columns down the side")
    ap.add_argument("--text-scale", type=float, default=1.0,
                    help="multiplier on every font size, and on the cell size "
                         "with it so tick labels keep their spacing")
    ap.add_argument("--compare", action="store_true",
                    help="draw BOTH relative-width measures on one shared scale, "
                         "into heatmaps_compare/. Each measure's own edges are "
                         "tuned to its own distribution, which makes two grids "
                         "incomparable - the same colour means a different "
                         "number in each - so this forces SHARED_EDGES on both. "
                         "Implies --measure relwidth iqr; the zero grid is "
                         "measure-independent and is written once.")
    args = ap.parse_args()

    global PORTRAIT, TEXT_SCALE, CELL_SCALE, PLAIN
    PORTRAIT, TEXT_SCALE = args.portrait, args.text_scale
    CELL_SCALE, PLAIN = args.cell_scale, args.plain

    csv = Path(args.ranges) if args.ranges else RANGES_CSV
    if not csv.exists():
        raise SystemExit(f"{csv} not found - run `python -m src.ead_ranges` first.")

    ranges = pd.read_csv(csv)
    if args.assets:
        ranges = ranges[ranges["asset"].isin(args.assets)]
    if args.headline:
        from .eu_totals import HEADLINE_SCENARIOS
        ranges = ranges[ranges["scenario"].isin(HEADLINE_SCENARIOS)]
    if args.scenarios:
        ranges = ranges[ranges["scenario"].isin(args.scenarios)]
    if not (args.allow_partial or args.scenarios):
        ranges = drop_partial_scenarios(ranges)
    if ranges.empty:
        raise SystemExit("No rows left after filtering.")

    pm.use_palette(args.palette)

    if args.compare:
        odir = Path(args.out_dir) if args.out_dir else OUT_DIR.with_name("heatmaps_compare")
        n = 0
        for measure, tag in (("relwidth", "w90"), ("iqr", "iqr")):
            pm.use_measure(measure)
            pm.use_shared_edges()   # after use_measure, which resets the edges
            for group_by in args.group_by:
                gtag = "by_hazard" if group_by == "hazard" else "by_infra"
                plot_spread_heatmap(
                    ranges, odir / f"Spread_heatmap_{tag}_{gtag}.png", group_by)
                n += 1
        for group_by in args.group_by:
            gtag = "by_hazard" if group_by == "hazard" else "by_infra"
            plot_zero_heatmap(ranges, odir / f"Zeros_heatmap_{gtag}.png", group_by)
            n += 1
        print(f"{n} heatmaps written to {odir}")
        return

    pm.use_measure(args.measure)
    suffix = "".join(
        f"_{v}" for v, default in ((args.palette, "blue"), (args.measure, "ratio"))
        if v != default
    )
    odir = Path(args.out_dir) if args.out_dir else OUT_DIR.with_name(OUT_DIR.name + suffix)

    n = 0
    for group_by in args.group_by:
        tag = "by_hazard" if group_by == "hazard" else "by_infra"
        plot_spread_heatmap(ranges, odir / f"Spread_heatmap_{tag}.png", group_by)
        plot_zero_heatmap(ranges, odir / f"Zeros_heatmap_{tag}.png", group_by)
        n += 2
    print(f"{n} heatmaps written to {odir}")


if __name__ == "__main__":
    main()
