"""Absolute uncertainty (p95 - p5 of EAD, EUR/yr) against the curve and cost inputs
-> paper/width_vs_input/.

x: the input spans of src.plot_paper_inputs - vulnerability curves to choose from,
   and the cost span (max - min) / mean (value-weighted from paper/data/input_spans.csv,
   or the unweighted stand-in until src.paper_input_spans has run).
y: p95 - p5 of EAD from paper/data/ead_quantiles.parquet, for the headline scenario of
   each hazard (plot_paper_infra.HAZARDS).

  a_countries   one panel per hazard: every country faint, the median over countries bold
  b_europe      one panel per hazard: the sum over countries of p95 - p5, which is the
                pan-European width if all countries move together (an upper bound)
  c_combined    one panel, medians over countries, hazard colours, infra symbols

Each infrastructure type has one symbol in its sector's colour, as in design d of
the driver-vs-input figures. Types sharing an x value are set side by side.

    python -m src.plot_paper_width_vs_input
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_paper_infra import (ASSETS, DATA, FULL_WIDTH, GRID, HAZARDS, INK, INK_2, OUT, SECTOR_COLOR,
                               SECTOR_OF, save, set_style)
from .plot_paper_inputs import (ASSET_MARKER, ASSET_NAME, HAZARD_COLOR, HAZARD_KEY, MARKER_SIZE, METRICS,
                                SPANS, quick_spans, x_axis)
from .plot_paper_ranges import QUANTILES, euro_ticks, legend_above

FOLDER = OUT / "width_vs_input"
FIRST_FIG = {"curves": 16, "cost": 17}


def load_widths() -> pd.DataFrame:
    q = pd.read_parquet(QUANTILES, columns=["country", "asset", "scenario", "mean", "q0050", "q0950"])
    q = q[q.scenario.isin(HAZARDS.values())].copy()
    q["hazard"] = q.scenario.map({s: h for h, s in HAZARDS.items()})
    q["w90"] = q.q0950 - q.q0050
    return q


def load_spans() -> pd.DataFrame:
    if SPANS.exists():
        spans = pd.read_csv(SPANS)
    else:
        print("NOTE: no input_spans.csv - cost span is the unweighted median over object types")
        spans = quick_spans()
    spans["hazard"] = spans.hazard.map({v: k for k, v in HAZARD_KEY.items()})
    return spans[["asset", "hazard", "n_curves", "cost_range"]]


def side_by_side(xs: pd.Series, width: float) -> pd.Series:
    """Offset the assets that share an x value, in ASSETS order, centred on the true x."""
    out = xs.astype(float).copy()
    for _, idx in xs.groupby(xs.round(6)).groups.items():
        idx = sorted(idx, key=ASSETS.index)
        if len(idx) > 1:
            out[idx] += width * (np.arange(len(idx)) - (len(idx) - 1) / 2)
    return out


def marker_kw(asset, scale=1.0, **kw):
    m = ASSET_MARKER[asset]
    return {"marker": m, "s": MARKER_SIZE.get(m, 34) * scale, "edgecolors": INK, "linewidths": 0.4} | kw


def asset_legend(fig):
    """One column per sector, as in design d."""
    by_sector = {}
    for a in ASSET_MARKER:
        by_sector.setdefault(SECTOR_OF[a], []).append(a)
    rows = max(len(v) for v in by_sector.values())
    handles = []
    for sector, assets in by_sector.items():
        col = [Line2D([], [], ls="", marker=ASSET_MARKER[a], ms=9 if ASSET_MARKER[a] == "*" else 6.5,
                      mfc=SECTOR_COLOR[sector], mec=INK, mew=0.4, label=ASSET_NAME[a]) for a in assets]
        handles += col + [Line2D([], [], ls="", marker="", label="")] * (rows - len(col))
    legend_above(fig, handles, ncol=len(by_sector), columnspacing=1.6, handletextpad=0.3)


def y_log(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    euro_ticks(ax.yaxis, lo, hi, max_ticks=6)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def limits(values):
    v = values[values > 0]
    return 10 ** np.floor(np.log10(v.min())), 10 ** np.ceil(np.log10(v.max()))


def per_hazard(widths, spans, metric, kind):
    _, xcol, ticks, xlabel = METRICS[metric]
    med = widths.groupby(["hazard", "asset"]).w90.median().rename("median")
    tot = widths.groupby(["hazard", "asset"]).w90.sum().rename("europe")
    table = pd.concat([med, tot], axis=1).reset_index().merge(spans, on=["asset", "hazard"])
    table = table[table[xcol].notna() & ((table.n_curves > 0) if metric == "curves" else True)]
    ycol = "median" if kind == "a_countries" else "europe"
    pts = widths.merge(table[["hazard", "asset", xcol]], on=["hazard", "asset"])
    lo, hi = limits(pd.concat([pts.w90, table[ycol]]) if kind == "a_countries" else table[ycol])
    lo = max(lo, hi / 1e10)
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(1)
    for ax, hazard in zip(axes, HAZARDS):
        x_axis(ax, ticks, table[xcol].max())
        x0, x1 = ax.get_xlim()
        width = 0.035 * (x1 - x0)
        sub = table[table.hazard == hazard].set_index("asset")
        xs = side_by_side(sub[xcol], width)
        if kind == "a_countries":
            for asset, x in xs.items():
                c = pts[(pts.hazard == hazard) & (pts.asset == asset) & (pts.w90 > 0)]
                jitter = rng.uniform(-0.3, 0.3, len(c)) * width
                ax.scatter(x + jitter, c.w90, **marker_kw(asset, 0.35, linewidths=0),
                           color=SECTOR_COLOR[SECTOR_OF[asset]], alpha=0.35, zorder=2)
        for asset, x in xs.items():
            ax.scatter(x, sub.loc[asset, ycol], **marker_kw(asset, 1.15),
                       color=SECTOR_COLOR[SECTOR_OF[asset]], zorder=3)
        y_log(ax, lo, hi)
        ax.set_title(hazard)
    fig.supxlabel(xlabel, fontsize=8, color=INK_2, y=-0.06)
    ylabel = ("p95 − p5 of EAD (€/yr)\nfaint: each country; bold: median over countries"
              if kind == "a_countries" else "Σ over countries of p95 − p5 (€/yr)\n(all countries move together)")
    fig.supylabel(ylabel, fontsize=8, color=INK_2, x=0.02)
    asset_legend(fig)
    save(fig, f"fig{FIRST_FIG[metric]}_width_abs_vs_{metric}_{kind}", FOLDER)
    return table.assign(metric=metric)


def combined(table, metric):
    _, xcol, ticks, xlabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 3.6))
    x_axis(ax, ticks, table[xcol].max())
    x0, x1 = ax.get_xlim()
    lo, hi = limits(table["median"])
    for i, hazard in enumerate(HAZARDS):
        sub = table[table.hazard == hazard].set_index("asset")
        xs = side_by_side(sub[xcol], 0.02 * (x1 - x0)) + (i - 1.5) * 0.008 * (x1 - x0)
        for asset, x in xs.items():
            ax.scatter(x, sub.loc[asset, "median"], **marker_kw(asset), color=HAZARD_COLOR[hazard], zorder=3)
    y_log(ax, lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("p95 − p5 of EAD, median over countries (€/yr)")
    shapes = [Line2D([], [], ls="", marker=ASSET_MARKER[a], ms=9 if ASSET_MARKER[a] == "*" else 6,
                     mfc="white", mec=INK, mew=0.6, label=ASSET_NAME[a]) for a in ASSET_MARKER]
    colours = [Line2D([], [], ls="", marker="o", ms=6, mfc=HAZARD_COLOR[h], mec=INK, mew=0.4, label=h)
               for h in HAZARDS]
    ax.legend(handles=colours + shapes, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              handletextpad=0.3, labelspacing=0.45)
    save(fig, f"fig{FIRST_FIG[metric]}_width_abs_vs_{metric}_c_combined", FOLDER)


def main() -> None:
    set_style()
    FOLDER.mkdir(parents=True, exist_ok=True)
    widths, spans = load_widths(), load_spans()
    tables = []
    for metric in ("curves", "cost"):
        t = per_hazard(widths, spans, metric, "a_countries")
        per_hazard(widths, spans, metric, "b_europe")
        combined(t, metric)
        tables.append(t)
    pd.concat(tables).to_csv(DATA / "fig16_17_width_abs_vs_input.csv", index=False)


if __name__ == "__main__":
    main()
