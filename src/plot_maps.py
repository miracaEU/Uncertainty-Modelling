"""Choropleth maps of how well we know the number - not of how big it is.

Two map families, both drawn from output that already exists on disk:

    spread    the width of the full-uncertainty band, as the factor p95 / p5.
              This is the study's actual product. A map of EAD LEVEL mostly
              redraws population and GDP, which nobody needs a Monte Carlo to
              learn; a map of the SPREAD says where the number we publish is
              least worth trusting, and that is not predictable in advance.

    drivers   which uncertainty factor ranks first by Sobol total effect. The
              same content as the pies in src/plot_drivers.py, put on geography,
              which is where "protection dominates in the north-west, curve
              choice in the south" becomes visible.

Both are drawn in both orientations, because the two readings answer different
questions and the data supports each equally:

    by_asset/{asset}.png        one panel per scenario  - "for power lines, how
                                does the picture change between hazards?"
    by_scenario/{scenario}.png  one panel per asset     - "for river flood, which
                                asset types are we worst at?"

There is deliberately NO all-assets or all-scenarios aggregate map. Every cell
here is one real (country, asset, scenario) statistic; a median of p95/p5 across
asset types would be a number with no referent, and the countries where half the
assets have p5 = 0 have no defensible aggregate at all.

Geometry
--------
Eurostat GISCO CNTR_RG_20M, cached under geo/ on first run (data/ is read-only
here). GISCO rather than Natural Earth because NE 110m has no Andorra,
Liechtenstein or Malta at all and carries ISO_A3 = "-99" for France and Norway;
GISCO carries a clean ISO3_CODE for all 36 study countries. Drawn in EPSG:3035
(ETRS89-LAEA), the standard European equal-area projection - an equal-area
projection matters here because the eye reads area as weight.

Overseas territories are clipped away before projection, or the extent would be
set by French Guiana and the map of Europe would be a postage stamp.

Colour
------
Spread is MAGNITUDE, so it takes a single-hue sequential ramp, light to dark,
validated as an ordinal ramp (monotone lightness, adjacent dL >= 0.06, light end
>= 2:1 on the surface). "p5 = 0" is not a number on that scale - the band reaches
zero and the factor is unbounded - so it takes the dark end plus a hatch, which
is what keeps it from being read as "about 300x".

Drivers is IDENTITY, so it takes the categorical set already fixed in
src/plot_drivers.py. That set has seven slots and does NOT pass an all-pairs
check as a whole - but a choropleth only needs the colours that can appear in
ONE panel to be pairwise separable, and at most three driver categories are ever
top-ranked within a single scenario. All four triples that actually occur

    {protection, curve, cost}   {curve, cost}
    {curve, cost, hazard}       {curve, cost, warming}

pass all-pairs under protanopia and deuteranopia (worst dE 9.2, normal-vision
worst 16.3). Hazard intensity is the one slot below 3:1 contrast on the surface,
so every country carries a visible ISO3 label - identity is never colour alone.

Usage:
    python -m src.plot_maps
    python -m src.plot_maps --kinds spread
    python -m src.plot_maps --assets power roads --scenarios earthquake
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import box

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
WORKBOOK = PROJECT_ROOT / "MIRACA_uncertainty_study_summary.xlsx"
OUT_DIR = PROJECT_ROOT / "overview_figures" / "maps"

GISCO_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/"
             "CNTR_RG_20M_2024_4326.geojson")
# data/ is read-only on this machine, so the cache lives at the project root.
GEO_CACHE = PROJECT_ROOT / "geo" / "CNTR_RG_20M_2024_4326.geojson"

# Mainland Europe plus Iceland and Cyprus; everything else a study country owns
# (Canaries, Azores, Madeira, French Guiana, Svalbard) is clipped away.
CLIP_LONLAT = (-26.0, 33.0, 45.0, 72.0)
EPSG_EUROPE = 3035

# Below this a country is a couple of pixels at European extent and the fill
# cannot be read, so it is drawn as a fixed-size square at an interior point.
MICRO_KM2 = 5_000.0
MICRO_MARKER_PT = 7.0


# ---------------------------------------------------------------------------
# The spread vocabulary - shared with src/plot_heatmap.py
# ---------------------------------------------------------------------------

# Single-hue blue, validated as an ordinal ramp against surface #fcfcfb:
# monotone lightness, every adjacent gap >= 0.06, light end 2.09:1, hue spread 8.
SPREAD_RAMP = ("#8fb4e0", "#6f99cc", "#5480b6", "#3b649c", "#264a80", "#132a55")
# Edges chosen so the six classes are roughly equally occupied across the 1,606
# combinations with a finite factor (deciles: 3.3 / 12.3 / 24.9 / 69.8 / 2791).
# Even bins on a round 2-5-10-30-100 scale put two thirds of Europe in one class
# and the map stopped discriminating.
SPREAD_EDGES = (5.0, 10.0, 25.0, 60.0, 200.0)
SPREAD_LABELS = ("under 5x", "5 - 10x", "10 - 25x", "25 - 60x", "60 - 200x", "over 200x")

CLS_P5_ZERO = -1   # p5 == 0: the band reaches zero, the factor is unbounded
CLS_EMPTY = -2     # max == 0: not one draw produced damage
CLS_MISSING = -3   # no experiment for this combination

# p5 = 0 shares the dark end because it IS the extreme of the same scale, and
# takes a hatch because it is not a number on it. The hatch is drawn as its own
# overlay so its colour can be light against that navy.
P5_ZERO_FILL = "#132a55"
P5_ZERO_HATCH = "////"
P5_ZERO_HATCH_COLOR = "#9dc2ee"
# Clearly darker than CONTEXT_FILL and than the heatmap's "damage in every
# draw" tint: those two states mean opposite things and sat 4 values apart.
EMPTY_FILL = "#c9c7bc"
MISSING_FILL = "#ffffff"
CONTEXT_FILL = "#f1f0ea"     # non-study countries, for a map that reads as Europe
BORDER = "#b9b8b0"


def spread_class(max_v: float, p5: float, p95: float) -> int:
    """Ordinal bin index, or one of the CLS_* states.

    max == 0 is the exact test for "not one draw produced damage" and is checked
    first: such a row has p5 == 0 too, and calling it "unbounded spread" would
    be exactly backwards - there is no spread, there is nothing.
    """
    if not (max_v > 0):
        return CLS_EMPTY
    if not (p5 > 0):
        return CLS_P5_ZERO
    return int(np.searchsorted(SPREAD_EDGES, p95 / p5, side="right"))


def spread_color(cls: int) -> str:
    if cls == CLS_MISSING:
        return MISSING_FILL
    if cls == CLS_EMPTY:
        return EMPTY_FILL
    if cls == CLS_P5_ZERO:
        return P5_ZERO_FILL
    return SPREAD_RAMP[cls]


def spread_legend_handles(include_missing: bool = True) -> list:
    handles = [Patch(facecolor=c, edgecolor=BORDER, linewidth=0.4, label=lab)
               for c, lab in zip(SPREAD_RAMP, SPREAD_LABELS)]
    handles.append(Patch(facecolor=P5_ZERO_FILL, edgecolor=P5_ZERO_HATCH_COLOR,
                         linewidth=0.4, hatch=P5_ZERO_HATCH,
                         label="p5 = 0 (band reaches zero)"))
    handles.append(Patch(facecolor=EMPTY_FILL, edgecolor=BORDER, linewidth=0.4,
                         label="no damage in any draw"))
    if include_missing:
        handles.append(Patch(facecolor=MISSING_FILL, edgecolor=BORDER, linewidth=0.4,
                             label="not run"))
    return handles


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def load_geometry(cache: Path = GEO_CACHE) -> gpd.GeoDataFrame:
    """GISCO country polygons, clipped to Europe and projected to EPSG:3035.

    Indexed by ISO3 so it joins straight onto the study's country column, with
    an `is_micro` flag for the countries that need a minimum-size symbol.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Geometry: downloading GISCO countries -> {cache}")
        urllib.request.urlretrieve(GISCO_URL, cache)
    g = gpd.read_file(cache)
    g = g[g["ISO3_CODE"].notna()].copy()
    g = gpd.clip(g, box(*CLIP_LONLAT))
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    g = g.to_crs(EPSG_EUROPE)
    g = g.dissolve(by="ISO3_CODE").reset_index()
    g["is_micro"] = (g.geometry.area / 1e6) < MICRO_KM2
    g["point"] = g.geometry.representative_point()
    return g.set_index("ISO3_CODE")


def _extent(geo: gpd.GeoDataFrame, countries) -> tuple[float, float, float, float]:
    """Shared x/y limits, from the study countries only plus a small pad."""
    present = [c for c in countries if c in geo.index]
    minx, miny, maxx, maxy = geo.loc[present].total_bounds
    padx, pady = 0.03 * (maxx - minx), 0.03 * (maxy - miny)
    return minx - padx, maxx + padx, miny - pady, maxy + pady


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _panel(ax, geo, fills: dict[str, str], extent, hatched=(),
           hatch=P5_ZERO_HATCH, hatch_color=P5_ZERO_HATCH_COLOR, labels=True):
    """One map. `fills` is ISO3 -> face colour; `hatched` gets the p5 = 0 texture."""
    study = [c for c in fills if c in geo.index]
    context = geo.index.difference(study)

    if len(context):
        # Clipped to the drawn extent, not to the lon/lat box: a lon/lat box
        # projects to a curved shape in LAEA and left an empty white wedge in
        # the top corner of every panel.
        ctx = gpd.clip(geo.loc[context],
                       box(extent[0], extent[2], extent[1], extent[3]))
        ctx.plot(ax=ax, facecolor=CONTEXT_FILL, edgecolor=BORDER,
                 linewidth=0.3, zorder=1)
    big = [c for c in study if not geo.loc[c, "is_micro"]]
    if big:
        geo.loc[big].plot(ax=ax, color=[fills[c] for c in big],
                          edgecolor=BORDER, linewidth=0.45, zorder=2)
    hatch_here = [c for c in hatched if c in big]
    if hatch_here:
        # facecolor none + linewidth 0 keeps the fill and the border from the
        # layer below; the hatch takes its colour from edgecolor.
        geo.loc[hatch_here].plot(ax=ax, facecolor="none", edgecolor=hatch_color,
                                 hatch=hatch, linewidth=0.0, zorder=3)

    # Micro-states: a fixed-size square at an interior point, so a 147 km2
    # country is still readable at a 4,500 km extent.
    for iso in study:
        if not geo.loc[iso, "is_micro"]:
            continue
        pt = geo.loc[iso, "point"]
        ax.plot([pt.x], [pt.y], marker="s", markersize=MICRO_MARKER_PT,
                markerfacecolor=fills[iso], markeredgecolor=INK, markeredgewidth=0.6,
                linestyle="none", zorder=4)

    if labels:
        halo = [pe.withStroke(linewidth=1.7, foreground=SURFACE)]
        for iso in study:
            pt = geo.loc[iso, "point"]
            # A micro-state's square is drawn over its neighbours, so its label
            # is pushed clear of the square instead of being centred on it -
            # centred, LIE sat on top of CHE and LUX on top of BEL.
            offset = (0, -MICRO_MARKER_PT - 1.5) if geo.loc[iso, "is_micro"] else (0, 0)
            va = "top" if geo.loc[iso, "is_micro"] else "center"
            ax.annotate(iso, (pt.x, pt.y), xytext=offset, textcoords="offset points",
                        ha="center", va=va, fontsize=5.4, color=INK, zorder=6,
                        path_effects=halo)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


def _grid(n: int, max_cols: int = 4):
    cols = min(max_cols, n)
    return int(np.ceil(n / cols)), cols


def _figure(n_panels: int, aspect: float, max_cols: int = 4):
    """Grid sized to the map's own height/width ratio.

    A fixed panel height left about a fifth of every row blank: the axes cell
    was taller than the equal-aspect map that had to sit inside it.
    """
    rows, cols = _grid(n_panels, max_cols)
    panel_w = 3.35
    # The floor is for the suptitle, not the maps: a one-panel figure is under
    # 4 inches wide and "What drives the uncertainty? - River flood - protection
    # as recorded (FLOPROS)" was clipped at both ends.
    width = max(panel_w * cols + 0.6, 8.2)
    fig, axes = plt.subplots(
        rows, cols, squeeze=False, layout="constrained",
        figsize=(width, (panel_w * aspect + 0.45) * rows + 1.3))
    return fig, axes.ravel()


def _legend(fig, handles, ncol, subtitle: str):
    """Legend below the grid, with the subtitle as its title.

    The subtitle rides on the legend rather than being its own artist because a
    bare fig.text is not laid out at all and supxlabel is placed in the same
    slot as an outside legend - both landed on top of the keys. A legend title
    is negotiated with everything else, at any panel count.
    """
    leg = fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=ncol,
                     loc="outside lower center", title=subtitle,
                     alignment="center")
    leg.get_title().set(fontsize=8.2, color=MUTED)
    for t in leg.get_texts():
        t.set_color(INK_2)


# ---------------------------------------------------------------------------
# Spread maps
# ---------------------------------------------------------------------------


def _spread_fills(rows: pd.DataFrame) -> tuple[dict, list]:
    fills, hatched = {}, []
    cols = zip(rows["country"], rows["max"], rows["p5"], rows["p95"])
    for country, mx, p5, p95 in cols:
        cls = spread_class(mx, p5, p95)
        fills[country] = spread_color(cls)
        if cls == CLS_P5_ZERO:
            hatched.append(country)
    return fills, hatched


def plot_spread_map(ranges: pd.DataFrame, geo, facet: str, fixed: str, value: str,
                    out: Path, title: str) -> None:
    """One panel per level of `facet`, for a single `fixed` == `value`."""
    sub = ranges[ranges[fixed] == value]
    if sub.empty:
        return
    levels = sorted(sub[facet].unique(),
                    key=scenario_sort_key if facet == "scenario" else str)
    extent = _extent(geo, ranges["country"].unique())

    fig, axes = _figure(len(levels), (extent[3] - extent[2]) / (extent[1] - extent[0]))
    for ax, lev in zip(axes, levels):
        rows = sub[sub[facet] == lev]
        fills, hatched = _spread_fills(rows)
        _panel(ax, geo, fills, extent, hatched=hatched)
        name = scenario_label(lev, short=True) if facet == "scenario" else lev
        ax.set_title(f"{name}  (n={len(rows)})", fontsize=9.5, color=INK_2)
    for ax in axes[len(levels):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12.5)
    _legend(fig, spread_legend_handles(include_missing=False), 4,
            "colour = p95 / p5 of total EAD, the width of the full-uncertainty band  |  "
            "equal-area (EPSG:3035)  |  AND, LIE, LUX, MLT drawn at minimum size")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver maps
# ---------------------------------------------------------------------------


def top_drivers(workbook: Path = WORKBOOK) -> pd.DataFrame:
    """(country, asset, scenario) -> the factor category ranked first by ST.

    Delegates the whole ranking to src/plot_drivers.py rather than repeating the
    sort, so the maps and the pies cannot disagree about what a driver is -
    including the guard for the 273 combinations whose Sobol indices are all
    zero, which get "No exposure" or "Exposed, never damaged" instead of an
    arbitrary tie-break winner.

    plot_drivers is imported lazily because it reaches ema_model and therefore
    ema_workbench; the spread maps must stay runnable without it.
    """
    from .plot_drivers import exposure_flags, ranked_drivers

    top = ranked_drivers(workbook, max_rank=1, exposure=exposure_flags())
    return top.rename(columns={"category": "driver"})


def plot_driver_map(top: pd.DataFrame, geo, facet: str, fixed: str, value: str,
                    out: Path, title: str) -> None:
    from .plot_drivers import COLORS, EXPOSED_UNDAMAGED, HATCH, SLICE_ORDER

    sub = top[top[fixed] == value]
    if sub.empty:
        return
    levels = sorted(sub[facet].unique(),
                    key=scenario_sort_key if facet == "scenario" else str)
    extent = _extent(geo, top["country"].unique())

    fig, axes = _figure(len(levels), (extent[3] - extent[2]) / (extent[1] - extent[0]))
    for ax, lev in zip(axes, levels):
        rows = sub[sub[facet] == lev]
        fills = {c: COLORS.get(d, COLORS["Other"]) for c, d in
                 zip(rows["country"], rows["driver"])}
        # Same relief as the pies: the one non-driver state worth chasing is
        # textured as well as coloured.
        hatched = list(rows.loc[rows["driver"] == EXPOSED_UNDAMAGED, "country"])
        _panel(ax, geo, fills, extent, hatched=hatched, hatch=HATCH[EXPOSED_UNDAMAGED],
               hatch_color=INK_2)
        name = scenario_label(lev, short=True) if facet == "scenario" else lev
        ax.set_title(f"{name}  (n={len(rows)})", fontsize=9.5, color=INK_2)
    for ax in axes[len(levels):]:
        ax.set_visible(False)

    present = [c for c in SLICE_ORDER if (sub["driver"] == c).any()]
    handles = [Patch(facecolor=COLORS[c],
                     edgecolor=INK_2 if c in HATCH else BORDER,
                     hatch=HATCH.get(c), linewidth=0.4, label=c)
               for c in present]
    handles.append(Patch(facecolor=CONTEXT_FILL, edgecolor=BORDER, linewidth=0.4,
                         label="not in the study"))
    fig.suptitle(title, fontsize=12.5)
    _legend(fig, handles, min(len(handles), 4),
            "colour = the factor with the highest Sobol total effect on total EAD  |  "
            "neutral = no variance to attribute, every ST is zero  |  "
            "every country carries its ISO3 code, so identity is never colour alone")
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
    ap.add_argument("--workbook", default=None, help=f"default: {WORKBOOK}")
    ap.add_argument("--out-dir", default=None, help=f"default: {OUT_DIR}")
    ap.add_argument("--kinds", nargs="+", choices=["spread", "drivers"],
                    default=["spread", "drivers"])
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--scenarios", nargs="+", default=None)
    args = ap.parse_args()

    ranges_csv = Path(args.ranges) if args.ranges else RANGES_CSV
    if not ranges_csv.exists():
        raise SystemExit(f"{ranges_csv} not found - run `python -m src.ead_ranges` first.")
    odir = Path(args.out_dir) if args.out_dir else OUT_DIR

    ranges = pd.read_csv(ranges_csv)
    if args.assets:
        ranges = ranges[ranges["asset"].isin(args.assets)]
    if args.scenarios:
        ranges = ranges[ranges["scenario"].isin(args.scenarios)]
    if ranges.empty:
        raise SystemExit("No rows left after filtering.")

    geo = load_geometry()
    missing = sorted(set(ranges["country"]) - set(geo.index))
    if missing:
        print(f"  NOTE: no geometry for {missing} - those countries are not drawn")

    n = 0
    if "spread" in args.kinds:
        for asset in sorted(ranges["asset"].unique()):
            plot_spread_map(ranges, geo, "scenario", "asset", asset,
                            odir / "spread" / "by_asset" / f"{asset}.png",
                            f"How wide is the plausible range? - {asset}")
            n += 1
        for scen in sorted(ranges["scenario"].unique(), key=scenario_sort_key):
            plot_spread_map(ranges, geo, "asset", "scenario", scen,
                            odir / "spread" / "by_scenario" / f"{scen}.png",
                            f"How wide is the plausible range? - {scenario_label(scen)}")
            n += 1
        print(f"  spread: {n} maps")

    if "drivers" in args.kinds:
        top = top_drivers(Path(args.workbook) if args.workbook else WORKBOOK)
        if args.assets:
            top = top[top["asset"].isin(args.assets)]
        if args.scenarios:
            top = top[top["scenario"].isin(args.scenarios)]
        m = 0
        for asset in sorted(top["asset"].unique()):
            plot_driver_map(top, geo, "scenario", "asset", asset,
                            odir / "drivers" / "by_asset" / f"{asset}.png",
                            f"What drives the uncertainty? - {asset}")
            m += 1
        for scen in sorted(top["scenario"].unique(), key=scenario_sort_key):
            plot_driver_map(top, geo, "asset", "scenario", scen,
                            odir / "drivers" / "by_scenario" / f"{scen}.png",
                            f"What drives the uncertainty? - {scenario_label(scen)}")
            m += 1
        print(f"  drivers: {m} maps")
        n += m

    print(f"{n} maps written to {odir}")


if __name__ == "__main__":
    main()
