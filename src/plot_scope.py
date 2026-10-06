"""What the study covers: asset types, their sub-types, and hazard applicability.

Two reference figures for the methods section:

    scope_assets.png           the ten asset types, how many object sub-types
                               each carries, and which hazards apply to it
    scope_country_matrix.png   every country x asset pair, with the number of
                               hazards analysed in that cell

Applicability is read from src/curves.py::applicable_hazards, the single source
of truth the preprocessing and the orchestrator also consult, rather than being
restated here. Two rules produce every blank and every low number:

  * windstorm is skipped for roads and ports, whose only wind curve (W7.2) is
    identically zero, so there is nothing to vary;
  * coastal flood is skipped for the ten landlocked study countries.

A blank cell in the country matrix means the country has no exposure file for
that asset at all, which is a data gap rather than a modelling choice.

Usage:
    python -m src.plot_scope
    python -m src.plot_scope --out-dir overview_figures/report
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .curves import ASSET_CONFIGS, LANDLOCKED, applicable_hazards, get_asset_config, mapped_object_types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERGENCE_LOG = PROJECT_ROOT / "results" / "sobol_convergence_log.jsonl"
# Written on first run of src/plot_maps.py; read here only for its country
# names, so this module keeps working without geopandas.
GEO_CACHE = PROJECT_ROOT / "geo" / "CNTR_RG_20M_2024_4326.geojson"

HAZARDS = ["river", "coastal", "earthquake", "windstorm"]
HAZARD_LABEL = {"river": "River\nflood", "coastal": "Coastal\nflood",
                "earthquake": "Earth-\nquake", "windstorm": "Wind-\nstorm"}

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#d8d7d0"
YES = "#1c5cab"
NO = "#ebeae4"
# Number of hazards in a country x asset cell.
COUNT_FILL = {2: "#cde2fb", 3: "#86b6ef", 4: "#1c5cab"}
ABSENT = "#f6f5f1"


def asset_rows() -> list[dict]:
    """Per asset: sub-types analysed, and hazard applicability.

    Applicability is evaluated for a coastal country, so the coastal column
    shows whether the asset can ever carry coastal flood at all; whether a
    given country does is the second figure's job.
    """
    rows = []
    for a in sorted(ASSET_CONFIGS):
        cfg = get_asset_config(a)
        subs = sorted(mapped_object_types(cfg))
        hz = set(applicable_hazards(a, "ITA"))
        rows.append({"asset": a, "subtypes": subs, "hazards": hz})
    return rows


def country_assets() -> dict[str, set[str]]:
    """country -> assets with a completed run, from the convergence log."""
    present: dict[str, set[str]] = collections.defaultdict(set)
    with open(CONVERGENCE_LOG, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                present[r["country"]].add(r["asset"])
    return present


def country_names() -> dict[str, str]:
    """ISO3 -> English country name, from the GISCO cache.

    Falls back to an empty mapping if the cache has not been downloaded yet,
    in which case the matrix keeps its ISO3 labels rather than failing.
    """
    if not GEO_CACHE.exists():
        print(f"  NOTE: {GEO_CACHE.name} not found - labelling with ISO3 codes. "
              "Run `python -m src.plot_maps` once to fetch it.")
        return {}
    with open(GEO_CACHE, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for feat in data["features"]:
        props = feat.get("properties", {})
        iso, name = props.get("ISO3_CODE"), props.get("NAME_ENGL")
        if iso and name:
            out[iso] = name
    return out


def _wrap(items: list[str], width: int = 52) -> str:
    out, line = [], ""
    for it in items:
        add = it if not line else f", {it}"
        if len(line) + len(add) > width:
            out.append(line)
            line = it
        else:
            line += add
    if line:
        out.append(line)
    return "\n".join(out)


def assets_figure(out_dir: Path) -> Path:
    rows = asset_rows()
    n = len(rows)
    # Row band plus header plus a clear strip at the bottom for the legend;
    # a tighter ylim put the keys on top of the last row.
    fig, ax = plt.subplots(figsize=(11.6, 0.62 * n + 2.6), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, n)
    ax.axis("off")

    x_hz = {h: 3.0 + i * 0.78 for i, h in enumerate(HAZARDS)}
    x_sub = 6.5

    ax.text(0.05, n - 0.35, "Asset type", fontsize=10, fontweight="bold", color=INK)
    ax.text(2.05, n - 0.35, "Sub-\ntypes", fontsize=9, fontweight="bold",
            color=INK, ha="center", va="center")
    for h in HAZARDS:
        ax.text(x_hz[h], n - 0.35, HAZARD_LABEL[h], fontsize=8.5, fontweight="bold",
                color=INK, ha="center", va="center")
    ax.text(x_sub, n - 0.35, "Object sub-types analysed", fontsize=10,
            fontweight="bold", color=INK)

    for i, r in enumerate(rows):
        y = n - 1.4 - i
        ax.plot([0.05, 9.95], [y + 0.46, y + 0.46], color=GRID, linewidth=0.7, zorder=0)
        ax.text(0.05, y, r["asset"], fontsize=10, color=INK, va="center", fontweight="bold")
        ax.text(2.05, y, str(len(r["subtypes"])), fontsize=10, color=INK_2,
                ha="center", va="center")
        for h in HAZARDS:
            on = h in r["hazards"]
            ax.add_patch(plt.Rectangle((x_hz[h] - 0.26, y - 0.22), 0.52, 0.44,
                                       facecolor=YES if on else NO,
                                       edgecolor=GRID, linewidth=0.6, zorder=2))
            if not on:
                ax.text(x_hz[h], y, "-", fontsize=10, color=MUTED,
                        ha="center", va="center", zorder=3)
        ax.text(x_sub, y, _wrap(r["subtypes"]), fontsize=7.6, color=INK_2,
                va="center", linespacing=1.25)

    handles = [Patch(facecolor=YES, edgecolor=GRID, label="hazard analysed"),
               Patch(facecolor=NO, edgecolor=GRID, label="not applicable")]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 0.0))
    ax.set_title("Asset types, object sub-types and hazard coverage",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    out = out_dir / "scope_assets.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def country_matrix_figure(out_dir: Path) -> Path:
    present = country_assets()
    names = country_names()
    # Sorted by the label actually shown, so the reader can scan it: an ISO3
    # sort puts Germany under D and the United Kingdom under G.
    countries = sorted(present, key=lambda c: names.get(c, c))
    assets = sorted(ASSET_CONFIGS)

    fig, ax = plt.subplots(figsize=(9.9, 0.30 * len(countries) + 2.3),
                           layout="constrained")
    totals = []
    for ci, c in enumerate(countries):
        y = len(countries) - 1 - ci
        tot = 0
        for ai, a in enumerate(assets):
            if a in present[c]:
                k = len(applicable_hazards(a, c))
                tot += k
                ax.add_patch(plt.Rectangle((ai, y), 1, 1, facecolor=COUNT_FILL[k],
                                           edgecolor=SURFACE, linewidth=1.2))
                ax.text(ai + 0.5, y + 0.5, str(k), ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="#ffffff" if k == 4 else INK)
            else:
                ax.add_patch(plt.Rectangle((ai, y), 1, 1, facecolor=ABSENT,
                                           edgecolor=SURFACE, linewidth=1.2))
        totals.append(tot)
        ax.text(len(assets) + 0.35, y + 0.5, str(tot), ha="center", va="center",
                fontsize=8, color=INK_2)

    ax.set_xlim(0, len(assets) + 0.8)
    ax.set_ylim(0, len(countries))
    ax.set_xticks([i + 0.5 for i in range(len(assets))] + [len(assets) + 0.35])
    ax.set_xticklabels(assets + ["total"], rotation=45, ha="left", fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([len(countries) - 0.5 - i for i in range(len(countries))])
    ax.set_yticklabels(
        [f"{names.get(c, c)}*" if c in LANDLOCKED else names.get(c, c)
         for c in countries], fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    handles = [Patch(facecolor=COUNT_FILL[k], edgecolor=SURFACE, label=f"{k} hazards")
               for k in (2, 3, 4)]
    handles.append(Patch(facecolor=ABSENT, edgecolor=SURFACE, label="no exposure data"))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.015))
    ax.set_title("Hazards analysed per country and asset type\n"
                 "* landlocked: no coastal flood",
                 fontsize=12, fontweight="bold", color=INK, pad=28)
    out = out_dir / "scope_country_matrix.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out, countries, assets, present, totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="default: overview_figures/report")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
    })

    print(f"wrote {assets_figure(out_dir).name}")
    out, countries, assets, present, totals = country_matrix_figure(out_dir)
    print(f"wrote {out.name}")

    pairs = sum(len(present[c] & set(assets)) for c in countries)
    triples = sum(totals)
    dist = collections.Counter(
        len(applicable_hazards(a, c)) for c in countries for a in assets if a in present[c])
    print(f"\n{len(countries)} countries x {len(assets)} assets")
    print(f"  country-asset pairs analysed      : {pairs}")
    print(f"  country-asset-hazard combinations : {triples}")
    print(f"  cells by hazard count             : {dict(sorted(dist.items()))}")
    print(f"\nFigures in {out_dir}")


if __name__ == "__main__":
    main()
