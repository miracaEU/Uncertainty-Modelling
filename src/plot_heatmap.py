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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from .plot_maps import (
    BORDER,
    CLS_EMPTY,
    CLS_MISSING,
    CLS_P5_ZERO,
    EMPTY_FILL,
    MISSING_FILL,
    P5_ZERO_FILL,
    P5_ZERO_HATCH,
    P5_ZERO_HATCH_COLOR,
    RANGES_CSV,
    SPREAD_RAMP,
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


def column_layout(ranges: pd.DataFrame) -> tuple[list, list, dict]:
    """(columns, blocks, x_of) - one x position per (asset, scenario).

    Assets are separated by ASSET_GAP columns of blank, so the eye reads ten
    blocks rather than one 66-wide run. `blocks` carries (asset, x0, x1) for the
    label band and the separators.
    """
    columns, blocks, x_of = [], [], {}
    x = 0.0
    for asset in sorted(ranges["asset"].unique()):
        scens = sorted(ranges.loc[ranges["asset"] == asset, "scenario"].unique(),
                       key=scenario_sort_key)
        x0 = x
        for scen in scens:
            columns.append((asset, scen))
            x_of[(asset, scen)] = x
            x += 1.0
        blocks.append((asset, x0, x - 1.0))
        x += ASSET_GAP
    return columns, blocks, x_of


def row_order(ranges: pd.DataFrame) -> list[str]:
    """Countries by total median EAD across every combination, largest first."""
    tot = ranges.groupby("country")["median"].sum().sort_values(ascending=False)
    return list(tot.index)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_grid(ax, codes, countries, columns, blocks, x_of, cmap, hatch_code=None):
    """Cells as one Rectangle each, plus the asset band and the separators.

    Rectangles rather than imshow because the columns are not evenly spaced -
    the gap between asset blocks is part of the layout - and because the p5 = 0
    class needs a per-cell hatch, which an image cannot carry.
    """
    for j, key in enumerate(columns):
        x = x_of[key]
        for i, _ in enumerate(countries):
            code = codes[i, j]
            if code == CLS_MISSING:
                continue
            ax.add_patch(Rectangle((x, i), 1.0, 1.0, facecolor=cmap(code),
                                   edgecolor=SURFACE, linewidth=0.5, zorder=2))
            if hatch_code is not None and code == hatch_code:
                ax.add_patch(Rectangle((x, i), 1.0, 1.0, facecolor="none",
                                       edgecolor=P5_ZERO_HATCH_COLOR,
                                       hatch=P5_ZERO_HATCH, linewidth=0.0, zorder=3))

    n_rows = len(countries)
    for asset, x0, x1 in blocks:
        ax.add_patch(Rectangle((x0, n_rows + 0.35), x1 - x0 + 1.0, 0.75,
                               facecolor="#e8e7e0", edgecolor="none", zorder=2))
        ax.text((x0 + x1 + 1.0) / 2, n_rows + 0.72, asset, ha="center", va="center",
                fontsize=7.5, color=INK, zorder=3)

    ax.set_yticks(np.arange(n_rows) + 0.5, countries, fontsize=7)
    ax.set_xticks([x_of[k] + 0.5 for k in columns],
                  [scenario_label(s, short=True) for _, s in columns],
                  rotation=90, fontsize=6, color=INK_2)
    ax.set_xlim(-0.4, max(x_of.values()) + 1.4)
    # Row 0 at the top; the extra 1.4 at the top is the asset band.
    ax.set_ylim(n_rows + 1.4, 0)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _figure(n_rows: int, n_cols_span: float):
    fig, ax = plt.subplots(figsize=(CELL_W * n_cols_span + 2.6,
                                    CELL_H * n_rows + 3.9),
                           layout="constrained")
    return fig, ax


def _legend(fig, handles, ncol, subtitle):
    leg = fig.legend(handles=handles, frameon=False, fontsize=8, ncol=ncol,
                     loc="outside lower center", title=subtitle, alignment="center")
    leg.get_title().set(fontsize=8.2, color=MUTED)
    for t in leg.get_texts():
        t.set_color(INK_2)


def plot_spread_heatmap(ranges: pd.DataFrame, out: Path) -> None:
    countries = row_order(ranges)
    columns, blocks, x_of = column_layout(ranges)
    idx = {(r.country, r.asset, r.scenario): (r.max, r.p5, r.p95)
           for r in ranges.itertuples()}

    codes = np.full((len(countries), len(columns)), CLS_MISSING, dtype=int)
    for i, country in enumerate(countries):
        for j, (asset, scen) in enumerate(columns):
            hit = idx.get((country, asset, scen))
            if hit is not None:
                codes[i, j] = spread_class(*hit)

    palette = {CLS_EMPTY: EMPTY_FILL, CLS_P5_ZERO: P5_ZERO_FILL, CLS_MISSING: MISSING_FILL}
    palette.update({k: c for k, c in enumerate(SPREAD_RAMP)})

    fig, ax = _figure(len(countries), max(x_of.values()) + 1)
    _draw_grid(ax, codes, countries, columns, blocks, x_of, palette.get,
               hatch_code=CLS_P5_ZERO)

    n = int((codes != CLS_MISSING).sum())
    fig.suptitle("How wide is the plausible range? - every combination in the study",
                 fontsize=13)
    _legend(fig, spread_legend_handles(include_missing=False), 8,
            f"colour = p95 / p5 of total EAD  |  {n:,} combinations, "
            f"{len(countries)} countries x {len(columns)} (asset, scenario) columns  |  "
            f"blank = not run  |  countries ordered by total median EAD")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_zero_heatmap(ranges: pd.DataFrame, out: Path) -> None:
    countries = row_order(ranges)
    columns, blocks, x_of = column_layout(ranges)
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
    palette.update({k: c for k, c in enumerate(SPREAD_RAMP)})

    fig, ax = _figure(len(countries), max(x_of.values()) + 1)
    _draw_grid(ax, codes, countries, columns, blocks, x_of, palette.get)

    handles = [Patch(facecolor=NEVER_ZERO_FILL, edgecolor=BORDER, linewidth=0.4,
                     label="0% - damage in every draw")]
    handles += [Patch(facecolor=c, edgecolor=BORDER, linewidth=0.4, label=lab)
                for c, lab in zip(SPREAD_RAMP, ZERO_LABELS)]
    handles.append(Patch(facecolor=EMPTY_FILL, edgecolor=BORDER, linewidth=0.4,
                         label="100% - no damage in any draw"))

    n_always = int((codes == ALWAYS).sum())
    n_any = int(((codes != CLS_MISSING) & (codes != NEVER)).sum())
    fig.suptitle("How often is there no damage at all?", fontsize=13)
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
    args = ap.parse_args()

    csv = Path(args.ranges) if args.ranges else RANGES_CSV
    if not csv.exists():
        raise SystemExit(f"{csv} not found - run `python -m src.ead_ranges` first.")
    odir = Path(args.out_dir) if args.out_dir else OUT_DIR

    ranges = pd.read_csv(csv)
    if args.assets:
        ranges = ranges[ranges["asset"].isin(args.assets)]
    if args.scenarios:
        ranges = ranges[ranges["scenario"].isin(args.scenarios)]
    if ranges.empty:
        raise SystemExit("No rows left after filtering.")

    plot_spread_heatmap(ranges, odir / "Spread_heatmap.png")
    plot_zero_heatmap(ranges, odir / "Zeros_heatmap.png")
    print(f"2 heatmaps written to {odir}")


if __name__ == "__main__":
    main()
