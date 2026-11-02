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

# The names below come in two kinds. SPREAD_RAMP, SPREAD_EDGES, SPREAD_LABELS,
# SPREAD_SUBTITLE, SPREAD_DEGENERATE_LABEL, P5_ZERO_FILL and
# P5_ZERO_HATCH_COLOR are the CURRENT selection, rebound by use_palette() and
# use_measure(). Everything suffixed _BLUE / _SPECTRAL / RATIO_ / RELWIDTH_ is a
# fixed definition those two read from, so a swap is reversible.

# Single-hue blue, validated as an ordinal ramp against surface #fcfcfb:
# monotone lightness, every adjacent gap >= 0.06, light end 2.09:1, hue spread 8.
SPREAD_RAMP_BLUE = ("#8fb4e0", "#6f99cc", "#5480b6", "#3b649c", "#264a80", "#132a55")
SPREAD_RAMP = SPREAD_RAMP_BLUE

# --palette spectral. Six bins of one hue separate only by lightness, and with
# the light end pinned at 2.09:1 against the surface there is only so much
# lightness to divide: adjacent classes of the blue ramp sit 9.9 CIE76 dE apart,
# which is why the middle of it reads as one colour.
#
# This ramp keeps the blue ramp's lightness structure exactly - same L* targets
# 72/61/51/41/29/17, same monotone ordering, so it is still an ORDINAL scale
# and still survives greyscale printing - and spends hue on top of it, which
# raises adjacent separation to 27.1 dE. Built in LCh at fixed L* rather than
# picked by eye, because the light end has to clear 2.09:1 against #fcfcfb and
# every off-the-shelf multi-hue ramp (viridis, YlGnBu, YlOrRd) fails that: their
# pale first class disappears into the surface.
#
# Measured, normal vision then simulated dichromacy (Vienot/Brettel), as minimum
# adjacent dE - blue ramp in brackets for reference:
#   normal 27.1 (9.9) | deuteranopia 8.3 (7.9) | protanopia 7.8 (8.1) | trit 12.2 (10.5)
# So it is far better for most readers and no worse than the status quo for
# dichromats. A warm gold-to-purple ramp scored better on looks and collapsed to
# 2.2 dE under deuteranopia, so it was rejected.
SPREAD_RAMP_SPECTRAL = ("#bdb41f", "#53a552", "#00878c", "#00649f", "#54357a", "#491739")

# --palette traffic. Green = narrowest band = the number we know best, running
# through to dark red. The most immediately interpretable of the three, and the
# right choice when two grids are being read against each other rather than
# each on its own.
#
# Green-to-red is normally the worst choice for red-green colour blindness, so
# this is built on the SAME monotone L* targets as the other two ramps and the
# lightness does the discriminating. Measured that way it is not a compromise:
#   normal 22.6 | deuteranopia 7.9 | protanopia 12.1 | tritanopia 15.6
# against the blue ramp's 9.9 / 7.9 / 8.1 / 10.5 - better for normal vision and
# equal or better on all three dichromacies. The cost is that the middle of the
# ramp goes olive and brown: yellow only exists at high lightness, so a yellow
# hue pinned to L* 50 is brown. Accepted, because breaking the lightness ramp
# to get a bright yellow is what would make it unreadable for dichromats.
SPREAD_RAMP_TRAFFIC = ("#71c36d", "#829e23", "#987100", "#9e4200", "#8b0d18", "#60001c")
# Edges chosen so the six classes are roughly equally occupied across the 1,606
# combinations with a finite factor (deciles: 3.3 / 12.3 / 24.9 / 69.8 / 2791).
# Even bins on a round 2-5-10-30-100 scale put two thirds of Europe in one class
# and the map stopped discriminating.
RATIO_EDGES = (5.0, 10.0, 25.0, 60.0, 200.0)
RATIO_LABELS = ("under 5x", "5 - 10x", "10 - 25x", "25 - 60x", "60 - 200x", "over 200x")
RATIO_SUBTITLE = "colour = p95 / p5 of total EAD, the width of the full-uncertainty band"
RATIO_DEGENERATE_LABEL = "p5 = 0 (band reaches zero)"

SPREAD_EDGES = RATIO_EDGES
SPREAD_LABELS = RATIO_LABELS
SPREAD_SUBTITLE = RATIO_SUBTITLE
SPREAD_DEGENERATE_LABEL = RATIO_DEGENERATE_LABEL

# --measure relwidth. p95/p5 is a RATIO, so it diverges as p5 approaches zero:
# the 237 combinations with p5 == 0 have no finite value at all and have to be
# shown as a separate hatched class, and those just above zero are pushed into
# the top bin by a small denominator rather than by a wide band.
#
# (p95 - p5) / median measures the same band against the central estimate
# instead, which stays finite wherever a central estimate exists. It is already
# the quantity the pyramid figures report as w90_rel, so the two figure families
# now state the same number, and it is a column of EAD_Ranges.csv rather than
# anything recomputed here.
#
# The degenerate case MOVES rather than disappearing: p5 == 0 becomes an
# ordinary value (237 combinations gained), while median == 0 - a nonzero band
# around a zero central estimate - becomes the undefined one (126 combinations).
# Edges are round numbers with occupancy 11.6-21.3% across the 1,717 finite
# values, chosen the same way as the ratio edges above.
RELWIDTH_EDGES = (2.0, 4.0, 6.0, 8.0, 20.0)
RELWIDTH_LABELS = ("under 2x", "2 - 4x", "4 - 6x", "6 - 8x", "8 - 20x", "over 20x")
RELWIDTH_SUBTITLE = ("colour = (p95 - p5) / median of total EAD, the band as a "
                     "multiple of the central estimate")
RELWIDTH_DEGENERATE_LABEL = "median = 0 (no central estimate)"

# --measure iqr. The same relative-width idea on the INTERQUARTILE band rather
# than the 5-95 one: it answers "how uncertain is the bulk of the distribution"
# instead of "how far do the tails reach". Much less tail-sensitive, which
# matters here because several combinations are zero-inflated and their p5 and
# p95 are set by how often the model produced no damage at all rather than by
# how uncertain the damage is.
#
# Edges are quarters and halves rather than integers because the distribution
# is tight - sextiles fall at 0.82 / 1.29 / 1.58 / 2.03 / 4.06 - and integer
# edges put a third of Europe in one class. Occupancy 14.3-19.6%, the most even
# of the three measures.
IQR_EDGES = (0.75, 1.25, 1.5, 2.0, 4.0)
IQR_LABELS = ("under 0.75x", "0.75 - 1.25x", "1.25 - 1.5x",
              "1.5 - 2x", "2 - 4x", "over 4x")
IQR_SUBTITLE = ("colour = (p75 - p25) / median of total EAD, the interquartile "
                "band as a multiple of the central estimate")
IQR_DEGENERATE_LABEL = "median = 0 (no central estimate)"

# One scale for both relative-width measures, so two grids can be read against
# each other. Each measure's own edges are tuned to its own distribution, which
# makes each grid maximally informative ALONE and the two of them incomparable:
# the same colour means a different number in each. These edges double, 0.75 to
# 12, and are the only candidate both measures use essentially all of - the
# interquartile width falls in the lower three classes (15/31/34%), the 5-95
# width in the upper three (28/22/22%). That separation IS the finding, so a
# shared scale has to be wide enough to show it rather than balanced for one.
SHARED_EDGES = (0.75, 1.5, 3.0, 6.0, 12.0)
SHARED_LABELS = ("under 0.75x", "0.75 - 1.5x", "1.5 - 3x",
                 "3 - 6x", "6 - 12x", "over 12x")


def use_shared_edges() -> None:
    """Put the current measure on SHARED_EDGES. Call AFTER use_measure, which
    resets the edges to whatever the measure prefers."""
    global SPREAD_EDGES, SPREAD_LABELS
    SPREAD_EDGES = SHARED_EDGES
    SPREAD_LABELS = SHARED_LABELS

MEASURE = "ratio"

CLS_DEGENERATE = -1  # the measure has no finite value: see *_DEGENERATE_LABEL
CLS_P5_ZERO = CLS_DEGENERATE  # old name, kept so nothing importing it breaks
CLS_EMPTY = -2     # max == 0: not one draw produced damage
CLS_MISSING = -3   # no experiment for this combination

# p5 = 0 shares the dark end because it IS the extreme of the same scale, and
# takes a hatch because it is not a number on it. The hatch is drawn as its own
# overlay so its colour can be light against that navy.
P5_ZERO_FILL_BLUE = "#132a55"
P5_ZERO_HATCH_COLOR_BLUE = "#9dc2ee"
# The same two, matched to the spectral ramp's dark end instead of the navy.
P5_ZERO_FILL_SPECTRAL = "#491739"
P5_ZERO_HATCH_COLOR_SPECTRAL = "#edb9d2"
P5_ZERO_FILL_TRAFFIC = "#60001c"
P5_ZERO_HATCH_COLOR_TRAFFIC = "#f2b9b0"

P5_ZERO_HATCH = "////"
P5_ZERO_FILL = P5_ZERO_FILL_BLUE
P5_ZERO_HATCH_COLOR = P5_ZERO_HATCH_COLOR_BLUE


def use_palette(name: str) -> None:
    """Swap the spread ramp before anything draws.

    Rebinds module globals rather than threading a palette argument through
    every drawing function, which is how src/cascade.py handles the same
    problem. Note that src/plot_heatmap.py imports SPREAD_RAMP by name at import
    time, so it keeps the blue ramp regardless - deliberately: the heatmap is
    not part of the map set and its colours are referenced in the text.
    """
    global SPREAD_RAMP, P5_ZERO_FILL, P5_ZERO_HATCH_COLOR
    # Both branches assign, so calling this twice restores rather than latches.
    # A set-only-on-non-default version silently left the previous palette in
    # place on the way back, which is the kind of thing that corrupts a
    # comparison run inside one process rather than a figure.
    if name == "spectral":
        SPREAD_RAMP = SPREAD_RAMP_SPECTRAL
        P5_ZERO_FILL = P5_ZERO_FILL_SPECTRAL
        P5_ZERO_HATCH_COLOR = P5_ZERO_HATCH_COLOR_SPECTRAL
    elif name == "traffic":
        SPREAD_RAMP = SPREAD_RAMP_TRAFFIC
        P5_ZERO_FILL = P5_ZERO_FILL_TRAFFIC
        P5_ZERO_HATCH_COLOR = P5_ZERO_HATCH_COLOR_TRAFFIC
    else:
        SPREAD_RAMP = SPREAD_RAMP_BLUE
        P5_ZERO_FILL = P5_ZERO_FILL_BLUE
        P5_ZERO_HATCH_COLOR = P5_ZERO_HATCH_COLOR_BLUE


def use_measure(name: str) -> None:
    """Swap which spread statistic the maps bin, before anything draws.

    Reversible for the same reason as use_palette: a caller that compares the
    two measures in one process must get the ratio edges back when it asks for
    them.
    """
    global MEASURE, SPREAD_EDGES, SPREAD_LABELS, SPREAD_SUBTITLE
    global SPREAD_DEGENERATE_LABEL
    MEASURE = name
    if name == "relwidth":
        SPREAD_EDGES = RELWIDTH_EDGES
        SPREAD_LABELS = RELWIDTH_LABELS
        SPREAD_SUBTITLE = RELWIDTH_SUBTITLE
        SPREAD_DEGENERATE_LABEL = RELWIDTH_DEGENERATE_LABEL
    elif name == "iqr":
        SPREAD_EDGES = IQR_EDGES
        SPREAD_LABELS = IQR_LABELS
        SPREAD_SUBTITLE = IQR_SUBTITLE
        SPREAD_DEGENERATE_LABEL = IQR_DEGENERATE_LABEL
    else:
        SPREAD_EDGES = RATIO_EDGES
        SPREAD_LABELS = RATIO_LABELS
        SPREAD_SUBTITLE = RATIO_SUBTITLE
        SPREAD_DEGENERATE_LABEL = RATIO_DEGENERATE_LABEL


# Clearly darker than CONTEXT_FILL and than the heatmap's "damage in every
# draw" tint: those two states mean opposite things and sat 4 values apart.
EMPTY_FILL = "#c9c7bc"
MISSING_FILL = "#ffffff"
CONTEXT_FILL = "#f1f0ea"     # non-study countries, for a map that reads as Europe
BORDER = "#b9b8b0"


def spread_value(st) -> float:
    """What MEASURE bins, or nan where its denominator vanishes.

    `st` is any mapping with max / p5 / p25 / p75 / p95 / median - a DataFrame
    row works. A mapping rather than positionals because the three measures
    need different subsets of six statistics.
    """
    median = st["median"]
    if MEASURE == "relwidth":
        return (st["p95"] - st["p5"]) / median if median > 0 else float("nan")
    if MEASURE == "iqr":
        return (st["p75"] - st["p25"]) / median if median > 0 else float("nan")
    p5 = st["p5"]
    return st["p95"] / p5 if p5 > 0 else float("nan")


def spread_class(st) -> int:
    """Ordinal bin index, or one of the CLS_* states.

    max == 0 is the exact test for "not one draw produced damage" and is checked
    first: such a row has p5 == 0 too, and calling it "unbounded spread" would
    be exactly backwards - there is no spread, there is nothing.

    Which denominator can vanish depends on MEASURE - p5 for the ratio, the
    median for the other two - so the undefined case is whatever spread_value
    could not compute rather than a condition restated here.
    """
    if not (st["max"] > 0):
        return CLS_EMPTY
    v = spread_value(st)
    if not np.isfinite(v):
        return CLS_DEGENERATE
    return int(np.searchsorted(SPREAD_EDGES, v, side="right"))


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
                         label=SPREAD_DEGENERATE_LABEL))
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
    g = gpd.clip(g, box(*(CONTEXT_LONLAT if MAINLAND else CLIP_LONLAT)))
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    g = g.to_crs(EPSG_EUROPE)
    g = g.dissolve(by="ISO3_CODE").reset_index()
    g["is_micro"] = (g.geometry.area / 1e6) < MICRO_KM2
    g["point"] = g.geometry.representative_point()
    return g.set_index("ISO3_CODE")


# --- presentation switches, set from main() ---------------------------------
# Module globals rather than threaded arguments, matching use_palette/
# use_measure: the figure builders are called from several places and only the
# CLI ever varies these.
#
# PLAIN strips everything that is scaffolding rather than result - the "(n=36)"
# counts, the explanatory line above the legend, the per-country ISO codes -
# and puts the remaining text in INK instead of the greys, for a figure that
# sits in a report next to body text rather than on a monitoring wall.
PLAIN = False
MAX_COLS = 4
MAINLAND = False

# The study reaches from the Azores (x 1.27 M) to Cyprus (x 6.53 M), but the
# Atlantic islands are specks in a great deal of empty ocean: they stretch the
# frame by a third and shrink the continent in every panel. MAINLAND crops to
# the continental block, which drops the Azores, Madeira, the Canaries and
# Iceland. Chosen in lon/lat because that is where the islands separate
# cleanly - mainland Portugal reaches -9.5, Madeira is at -17.
MAINLAND_LONLAT = (-11.0, 34.0, 35.0, 72.0)

# With the frame cropped to the continent, the context layer's own eastern
# edge (CLIP_LONLAT, lon 45) falls INSIDE the panel and leaves a diagonal gap
# across the north-east corner, so the map does not read as a rectangle.
# Context is cheap and the cached GISCO file is worldwide, so reach further
# out for it; every panel is clipped to its own extent regardless.
CONTEXT_LONLAT = (-26.0, 28.0, 62.0, 75.0)

# Multiplier on every font size, for a figure that has to stay legible at
# report width rather than on a monitoring wall.
TEXT_SCALE = 1.0
# Put the legend in the grid cells left empty by the last row, instead of
# below the figure.
LEGEND_IN_GRID = False


def _clip_to_mainland(geo: gpd.GeoDataFrame, study=()) -> gpd.GeoDataFrame:
    """Cut the geometry itself down to the continental block.

    Cropping the extent alone is not enough: the extent is a rectangle, and
    Iceland sits inside its north-west corner even though it is well outside
    the lon/lat frame. Clipping the geometry drops it, and trims the Atlantic
    island groups off Portugal and Spain, so the frame that is then measured
    from it holds continental Europe only.
    """
    # Segmentized before reprojecting: a lon/lat box has only four corners, and
    # reprojecting those alone gives straight edges in LAEA where the true
    # boundary curves. The straight southern edge cut below Malta and dropped
    # it from the study. Half a degree is well under the curvature.
    frame = (gpd.GeoSeries([box(*MAINLAND_LONLAT)], crs=4326)
             .segmentize(0.5).to_crs(geo.crs).iloc[0])
    out = geo.copy()
    # Only the study countries: they set the frame, so their islands are what
    # stretch it. The context countries are clipped to each panel's extent when
    # they are drawn, and cutting them here instead left the north-east corner
    # empty on a diagonal - the panel stopped being a rectangle of map.
    hit = out.index.isin(set(study)) if len(study) else slice(None)
    out.loc[hit, "geometry"] = out.loc[hit].geometry.intersection(frame)
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    # representative_point() was taken before the cut, so a country that lost
    # its islands would otherwise keep a label anchor out in the ocean.
    out["point"] = out.geometry.representative_point()
    return out


def _frame(geo: gpd.GeoDataFrame, countries) -> tuple[float, float, float, float]:
    # geo is already clipped when MAINLAND is set, so one measure serves both.
    return _extent(geo, countries)


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
           hatch=P5_ZERO_HATCH, hatch_color=None, labels=True):
    """One map. `fills` is ISO3 -> face colour; `hatched` gets the p5 = 0 texture.

    hatch_color resolves from the module global at CALL time, not as a default
    argument: a default is bound at import, which would pin it to the blue
    ramp's light blue and leave the hatch unreadable under --palette spectral.
    """
    if hatch_color is None:
        hatch_color = P5_ZERO_HATCH_COLOR
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


def _grid(n: int, max_cols: int | None = None):
    cols = min(max_cols if max_cols is not None else MAX_COLS, n)
    return int(np.ceil(n / cols)), cols


def _figure(n_panels: int, aspect: float, max_cols: int | None = None):
    """Grid sized to the map's own height/width ratio.

    A fixed panel height left about a fifth of every row blank: the axes cell
    was taller than the equal-aspect map that had to sit inside it.
    """
    rows, cols = _grid(n_panels, max_cols)
    panel_w = 3.35
    # The floor is for the suptitle, not the maps: a one-panel figure is under
    # 4 inches wide and "What drives the uncertainty? - River flood - protection
    # as recorded (FLOPROS)" was clipped at both ends.
    width = max(panel_w * cols + 0.6, 8.2 * TEXT_SCALE)
    fig, axes = plt.subplots(
        rows, cols, squeeze=False, layout="constrained",
        figsize=(width,
                 (panel_w * aspect + 0.45 * TEXT_SCALE) * rows + 1.3 * TEXT_SCALE))
    return fig, axes.ravel()


def _legend(fig, handles, ncol, subtitle: str):
    """Legend below the grid, with the subtitle as its title.

    The subtitle rides on the legend rather than being its own artist because a
    bare fig.text is not laid out at all and supxlabel is placed in the same
    slot as an outside legend - both landed on top of the keys. A legend title
    is negotiated with everything else, at any panel count.
    """
    leg = fig.legend(handles=handles, frameon=False, fontsize=8.5 * TEXT_SCALE,
                     ncol=ncol, loc="outside lower center",
                     title=None if PLAIN else subtitle, alignment="center")
    if not PLAIN:
        leg.get_title().set(fontsize=8.2 * TEXT_SCALE, color=MUTED)
    for t in leg.get_texts():
        t.set_color(INK if PLAIN else INK_2)


def _place_legend(fig, spare, handles, ncol, subtitle: str):
    """Legend below the grid, or inside the cells the last row leaves empty.

    A grid of 10 panels three wide ends with two unused cells; dropping the
    legend into them uses space that is otherwise blank and keeps the figure
    shorter. Anchored to the LAST spare cell so a two-column legend spills
    leftwards into the other empty one rather than under a map.
    """
    if not (LEGEND_IN_GRID and spare):
        _legend(fig, handles, ncol, subtitle)
        return
    for ax in spare:
        ax.set_visible(False)
    # Centred on the block the empty cells form together, which means knowing
    # where they ended up: constrained layout only settles the positions at
    # draw time. Draw once, then freeze the engine so that adding the legend
    # (and the save that follows) cannot shift the cells out from under the
    # coordinates just measured.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    boxes = [ax.get_position() for ax in spare]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    leg = fig.legend(handles=handles, loc="center",
                     bbox_to_anchor=(x0, y0, x1 - x0, y1 - y0),
                     bbox_transform=fig.transFigure, frameon=False,
                     ncol=2 if len(handles) > 3 else 1,
                     fontsize=8.5 * TEXT_SCALE, handlelength=1.8,
                     labelspacing=0.9, columnspacing=1.4)
    leg.set_in_layout(False)
    for t in leg.get_texts():
        t.set_color(INK if PLAIN else INK_2)


# ---------------------------------------------------------------------------
# Spread maps
# ---------------------------------------------------------------------------


def _spread_fills(rows: pd.DataFrame) -> tuple[dict, list]:
    fills, hatched = {}, []
    for row in rows.to_dict("records"):
        cls = spread_class(row)
        fills[row["country"]] = spread_color(cls)
        if cls == CLS_DEGENERATE:
            hatched.append(row["country"])
    return fills, hatched


def plot_spread_map(ranges: pd.DataFrame, geo, facet: str, fixed: str, value: str,
                    out: Path, title: str) -> None:
    """One panel per level of `facet`, for a single `fixed` == `value`."""
    sub = ranges[ranges[fixed] == value]
    if sub.empty:
        return
    levels = sorted(sub[facet].unique(),
                    key=scenario_sort_key if facet == "scenario" else str)
    extent = _frame(geo, ranges["country"].unique())

    fig, axes = _figure(len(levels), (extent[3] - extent[2]) / (extent[1] - extent[0]))
    for ax, lev in zip(axes, levels):
        rows = sub[sub[facet] == lev]
        fills, hatched = _spread_fills(rows)
        _panel(ax, geo, fills, extent, hatched=hatched, labels=not PLAIN)
        name = scenario_label(lev, short=True) if facet == "scenario" else lev
        ax.set_title(name if PLAIN else f"{name}  (n={len(rows)})",
                     fontsize=9.5 * TEXT_SCALE, color=INK if PLAIN else INK_2)
    spare = list(axes[len(levels):])
    for ax in spare:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12.5 * TEXT_SCALE)
    _place_legend(fig, spare, spread_legend_handles(include_missing=False), 4,
            f"{SPREAD_SUBTITLE}  |  "
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
    extent = _frame(geo, top["country"].unique())

    fig, axes = _figure(len(levels), (extent[3] - extent[2]) / (extent[1] - extent[0]))
    for ax, lev in zip(axes, levels):
        rows = sub[sub[facet] == lev]
        fills = {c: COLORS.get(d, COLORS["Other"]) for c, d in
                 zip(rows["country"], rows["driver"])}
        # Same relief as the pies: the one non-driver state worth chasing is
        # textured as well as coloured.
        hatched = list(rows.loc[rows["driver"] == EXPOSED_UNDAMAGED, "country"])
        _panel(ax, geo, fills, extent, hatched=hatched, hatch=HATCH[EXPOSED_UNDAMAGED],
               hatch_color=INK_2, labels=not PLAIN)
        name = scenario_label(lev, short=True) if facet == "scenario" else lev
        ax.set_title(name if PLAIN else f"{name}  (n={len(rows)})",
                     fontsize=9.5 * TEXT_SCALE, color=INK if PLAIN else INK_2)
    spare = list(axes[len(levels):])
    for ax in spare:
        ax.set_visible(False)

    present = [c for c in SLICE_ORDER if (sub["driver"] == c).any()]
    handles = [Patch(facecolor=COLORS[c],
                     edgecolor=INK_2 if c in HATCH else BORDER,
                     hatch=HATCH.get(c), linewidth=0.4, label=c)
               for c in present]
    handles.append(Patch(facecolor=CONTEXT_FILL, edgecolor=BORDER, linewidth=0.4,
                         label="not in the study"))
    fig.suptitle(title, fontsize=12.5 * TEXT_SCALE)
    _place_legend(fig, spare, handles, min(len(handles), 4),
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
    ap.add_argument("--palette", choices=["blue", "spectral", "traffic"], default="blue",
                    help="spread-map colour ramp. 'blue' is the published "
                         "single-hue ramp; 'spectral' keeps the same lightness "
                         "steps but spends hue on them too, raising adjacent-class "
                         "separation from 9.9 to 27.1 dE. With --palette spectral "
                         "and no --out-dir, output goes to maps_spectral/ so the "
                         "two sets sit side by side.")
    ap.add_argument("--max-cols", type=int, default=4,
                    help="most panels per row (default 4)")
    ap.add_argument("--plain", action="store_true",
                    help="report styling: black text, no per-panel (n=) counts, "
                         "no explanatory line above the legend, no country codes")
    ap.add_argument("--mainland", action="store_true",
                    help="crop to continental Europe, dropping the Azores, "
                         "Madeira, the Canaries and Iceland")
    ap.add_argument("--text-scale", type=float, default=1.0,
                    help="multiplier on every font size (default 1.0)")
    ap.add_argument("--legend-in-grid", action="store_true",
                    help="put the legend in the cells the last row leaves "
                         "empty instead of below the figure")
    ap.add_argument("--measure", choices=["ratio", "relwidth", "iqr"], default="ratio",
                    help="what the spread maps bin. 'ratio' is p95/p5, which "
                         "diverges as p5 approaches zero. 'relwidth' is "
                         "(p95-p5)/median, the same band against the central "
                         "estimate; it stays finite where p5 == 0 and matches "
                         "the pyramid figures' w90_rel. Affects only the spread "
                         "maps, not the driver maps.")
    args = ap.parse_args()

    use_palette(args.palette)
    use_measure(args.measure)

    global PLAIN, MAX_COLS, MAINLAND, TEXT_SCALE, LEGEND_IN_GRID
    PLAIN, MAX_COLS, MAINLAND = args.plain, args.max_cols, args.mainland
    TEXT_SCALE, LEGEND_IN_GRID = args.text_scale, args.legend_in_grid

    ranges_csv = Path(args.ranges) if args.ranges else RANGES_CSV
    if not ranges_csv.exists():
        raise SystemExit(f"{ranges_csv} not found - run `python -m src.ead_ranges` first.")
    # A non-default palette gets its own folder unless one was named, so a
    # second set never silently overwrites the published maps.
    suffix = "".join(
        f"_{v}" for v, default in ((args.palette, "blue"), (args.measure, "ratio"))
        if v != default
    )
    if args.out_dir:
        odir = Path(args.out_dir)
    else:
        odir = OUT_DIR.with_name(OUT_DIR.name + suffix)

    ranges = pd.read_csv(ranges_csv)
    if args.assets:
        ranges = ranges[ranges["asset"].isin(args.assets)]
    if args.scenarios:
        ranges = ranges[ranges["scenario"].isin(args.scenarios)]
    if ranges.empty:
        raise SystemExit("No rows left after filtering.")

    geo = load_geometry()
    if MAINLAND:
        geo = _clip_to_mainland(geo, set(ranges["country"]))
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
