"""Uncertainty ranges as boxes and histograms -> paper/uncertainty_ranges/.

Every box is p5 / p25 / median / p75 / p95 of a combination's EAD distribution,
with the mean as a separate marker. Two scales:

  relative  EAD / mean EAD of the same combination. The mean sits at 1, the
            whisker span is the (p95 - p5) / mean width of fig01 and fig07, and
            the box shows the skew. Across countries the TYPICAL box is the
            median of each statistic over countries (order is preserved, since
            p5 <= p25 <= ... holds in every country).
  absolute  EUR/yr, one box per country. Every absolute figure comes twice: on a log
            axis, and as <name>_linear on a linear axis from 0. Linear axes are
            scaled per panel (shared axes would flatten all but the largest
            panel), and linear histograms end at the p99 of what they show.

Histograms come from paper/data/ead_quantiles.parquet (src/paper_ead_samples.py),
1,001 quantiles per combination read as an equal-weight sample. Pooled
histograms give every country the same weight.

  box_a_relative_by_asset        rows = infra types, hazards side by side in each row
  box_b_relative_by_hazard       one panel per hazard, infra types in sector colours
  box_c_countries_by_asset/      one figure per infra type: countries x hazards, absolute
  box_d_countries_by_hazard/     one figure per hazard: countries x infra types, absolute
  box_e_treatments               protection treatments, relative, river/coastal/wind
  hist_a_grid                    infra types x hazards, pooled EAD / mean
  hist_b_hazards_per_asset       one panel per infra type, hazards overlaid
  hist_c_ridgeline_per_hazard    one panel per hazard, infra types stacked
  hist_d_example_<country>_<asset>  one combination per hazard, absolute, box on top
  hist_e_treatments_<...>        one combination under every protection treatment
  treatments_by_hazard/          one combination, four panels (one per hazard), every
                                 protection treatment: box_f_* boxes, hist_f_* histograms
                                 as lines (each scaled to its peak); log and _linear

    python -m src.plot_paper_ranges
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from .plot_paper_infra import (ASSET_LABEL, ASSETS, BASE, DATA, FULL_WIDTH, GRID, HAZARDS, INK, INK_2,
                               OUT, RANGES_CSV, SECTOR_COLOR, SECTOR_OF, SECTORS, asset_axis, load_ranges,
                               save, set_style, y_positions)
from .plot_paper_inputs import HAZARD_COLOR, TREATMENT_COLOR, TREATMENTS

FOLDER = OUT / "uncertainty_ranges"
QUANTILES = DATA / "ead_quantiles.parquet"
STATS = ["p5", "p25", "p50", "p75", "p95"]
QCOL = {"p5": "q0050", "p25": "q0250", "p50": "q0500", "p75": "q0750", "p95": "q0950"}
REL_MAX = 8.0                      # relative x axes end here; anything beyond is counted
TREATMENT_SCENARIOS = {
    "River flood": ["flood_absprot_ds", "flood_absprot_ds_rp25to200", "flood_baseline_ds_sc0p5to2p0",
                    "flood_baseline_ds_sc0p75to1p33", "flood_noprot_ds"],
    "Coastal flood": ["coastal_absprot_ds_slr", "coastal_absprot_ds_slr_rp25to200",
                      "coastal_baseline_ds_slr_sc0p5to2p0", "coastal_baseline_ds_slr_sc0p75to1p33",
                      "coastal_noprot_ds_slr"],
    "Windstorm": ["windstorm_absprot", "windstorm", "windstorm_unprotected"],
}
PANEL_SCENARIOS = {"River flood": TREATMENT_SCENARIOS["River flood"],
                   "Coastal flood": TREATMENT_SCENARIOS["Coastal flood"],
                   "Earthquake": ["earthquake_pga0p9to1p1"],
                   "Windstorm": TREATMENT_SCENARIOS["Windstorm"]}
# rail and telecom are the only types run under every treatment; education is the social example
PANEL_EXAMPLES = [("DEU", "rail"), ("ITA", "telecom"), ("GBR", "education")]
TREATMENT_OF = {s: t for t, (_, ss) in TREATMENTS.items() for s in ss}
TREATMENT_NAME = {"Recorded / fixed RP50": "As recorded (flood) / fixed RP 50 (wind)"}
EXAMPLES = [("DEU", "roads"), ("ITA", "power")]
TREATMENT_EXAMPLES = [("DEU", "roads", "River flood"), ("NLD", "roads", "Coastal flood"),
                      ("DEU", "rail", "Windstorm")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_quantiles() -> pd.DataFrame | None:
    return pd.read_parquet(QUANTILES) if QUANTILES.exists() else None


def load_stats(q: pd.DataFrame | None) -> pd.DataFrame:
    """country, asset, scenario, p5..p95, mean (EUR/yr)."""
    if q is not None:
        stats = q[["country", "asset", "scenario", "mean"]].copy()
        for s, col in QCOL.items():
            stats[s] = q[col]
        return stats
    print(f"  NOTE: no {QUANTILES.name} yet; boxes from {RANGES_CSV.name}, histograms skipped")
    r = load_ranges()
    return r[["country", "asset", "scenario", "mean"] + STATS]


def relative(stats: pd.DataFrame) -> pd.DataFrame:
    rel = stats[stats["mean"] > 0].copy()
    for s in STATS:
        rel[s] = rel[s] / rel["mean"]
    rel["mean"] = 1.0
    return rel


def typical(rel: pd.DataFrame, keys) -> pd.DataFrame:
    """Median over countries of every relative statistic."""
    out = rel.groupby(keys)[STATS].median()
    out["mean"] = 1.0
    out["n"] = rel.groupby(keys).size()
    return out


def pooled_relative(q: pd.DataFrame, scenario: str, asset: str | None = None) -> tuple[np.ndarray, int]:
    sub = q[(q["scenario"] == scenario) & (q["mean"] > 0)]
    if asset is not None:
        sub = sub[sub["asset"] == asset]
    cols = [c for c in q.columns if c.startswith("q")]
    values = sub[cols].to_numpy() / sub["mean"].to_numpy()[:, None]
    return values.ravel(), len(sub)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def tint(color, amount=0.55):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def draw_box(ax, y, s, color, height=0.5, lo_clip=None):
    """Whisker p5-p95, box p25-p75, median bar, mean diamond. Values at or under lo_clip
    (log axes) are drawn at the clip with an open end."""
    v = {k: s[k] for k in STATS + ["mean"]}
    if lo_clip is not None:
        v = {k: max(x, lo_clip) for k, x in v.items()}
    ax.plot([v["p5"], v["p95"]], [y, y], color=color, lw=0.9, zorder=2, solid_capstyle="butt")
    for k in ("p5", "p95"):
        ax.plot([v[k], v[k]], [y - height * 0.25, y + height * 0.25], color=color, lw=0.9, zorder=2)
    ax.add_patch(Rectangle((v["p25"], y - height / 2), v["p75"] - v["p25"], height,
                           facecolor=tint(color), edgecolor=color, lw=0.7, zorder=3))
    ax.plot([v["p50"], v["p50"]], [y - height / 2, y + height / 2], color=INK, lw=1.1, zorder=4)
    ax.scatter(v["mean"], y, marker="D", s=12, facecolor="white", edgecolor=INK, linewidths=0.6, zorder=5)
    if lo_clip is not None and s["p5"] <= lo_clip:
        ax.scatter(lo_clip, y, marker="<", s=10, color=color, zorder=5, clip_on=False)


def legend_above(fig, handles, ncol, gap=0.006, **kw):
    """A figure legend just above everything drawn so far (titles and earlier legends)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tops = [ax.get_tightbbox(r).y1 for ax in fig.axes if ax.get_visible()]
    tops += [lg.get_window_extent(r).y1 for lg in fig.legends]
    y = max(tops) / fig.bbox.height + gap
    kw = {"handletextpad": 0.4, "columnspacing": 1.2} | kw
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, **kw)


def box_key(fig, extra=(), ncol=None):
    """Legend explaining the box anatomy, plus any colour entries."""
    handles = list(extra) + [
        Line2D([], [], color=INK_2, lw=0.9, label="p5–p95"),
        Patch(facecolor=tint(INK_2), edgecolor=INK_2, lw=0.7, label="p25–p75"),
        Line2D([], [], color=INK, lw=1.1, marker="|", ms=7, ls="", label="Median"),
        Line2D([], [], marker="D", ms=4, ls="", markerfacecolor="white", markeredgecolor=INK, label="Mean"),
    ]
    legend_above(fig, handles, ncol or min(len(handles), 8))


def euro_label(v, _pos=None):
    for unit, size in (("bn", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= size:
            return f"{v / size:g}{unit}"
    return f"{v:g}"


def euro_ticks(axis, lo, hi, max_ticks=6):
    """Decade ticks on a log € axis, thinned to at most max_ticks, labelled 1k / 1M / 1bn."""
    a, b = int(np.ceil(np.log10(lo) - 1e-9)), int(np.floor(np.log10(hi) + 1e-9))
    step = max(1, int(np.ceil((b - a + 1) / max_ticks)))
    axis.set_major_locator(FixedLocator(10.0 ** np.arange(a, b + 1, step)))
    axis.set_major_formatter(FuncFormatter(euro_label))
    axis.set_minor_locator(NullLocator())


def euro_linear(axis, nbins=3):
    """Round ticks on a linear € axis, labelled 0 / 50M / 100M."""
    axis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2))
    axis.set_major_formatter(FuncFormatter(euro_label))
    axis.set_minor_locator(NullLocator())


SUFFIX = {"log": "", "linear": "_linear"}


def relative_axis(ax, label=True):
    ax.set_xlim(0, REL_MAX)
    ax.set_xticks(range(0, int(REL_MAX) + 1, 2))
    ax.axvline(1, color=BASE, lw=0.6, zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if label:
        ax.set_xlabel("EAD / mean EAD of the same combination")


def mark_overflow(ax, s, y):
    if s["p95"] > REL_MAX:
        ax.annotate(f"p95 {s['p95']:.0f}", (REL_MAX, y), xytext=(-2, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6, color=INK_2)


def hazard_handles(hazards=HAZARDS):
    return [Patch(facecolor=tint(HAZARD_COLOR[h]), edgecolor=HAZARD_COLOR[h], label=h) for h in hazards]


def sector_handles():
    return [Patch(facecolor=tint(c), edgecolor=c, label=s) for s, c in SECTOR_COLOR.items()]


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def box_relative_by_asset(rel):
    """Rows = infra types; within a row one box per hazard."""
    positions = y_positions()
    t = typical(rel, ["scenario", "asset"])
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 6.4))
    offsets = dict(zip(HAZARDS, np.linspace(-0.3, 0.3, len(HAZARDS))))
    for hazard, scenario in HAZARDS.items():
        for asset, y in positions.items():
            if (scenario, asset) not in t.index:
                continue
            s = t.loc[(scenario, asset)]
            draw_box(ax, y + offsets[hazard], s, HAZARD_COLOR[hazard], height=0.17)
            mark_overflow(ax, s, y + offsets[hazard])
    relative_axis(ax, label=False)
    ax.set_xlabel("EAD / mean EAD of the same combination\n(typical country: median over countries of each statistic)")
    asset_axis(ax, positions, labels=True)
    box_key(fig, hazard_handles(), ncol=4)
    save(fig, "box_a_relative_by_asset", FOLDER)
    t.reset_index().to_csv(FOLDER / "box_a_b_relative_typical.csv", index=False)


def box_relative_by_hazard(rel):
    """One panel per hazard; infra types in sector colours."""
    positions = y_positions()
    t = typical(rel, ["scenario", "asset"])
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (hazard, scenario) in zip(axes, HAZARDS.items()):
        for asset, y in positions.items():
            if (scenario, asset) in t.index:
                s = t.loc[(scenario, asset)]
                draw_box(ax, y, s, SECTOR_COLOR[SECTOR_OF[asset]], height=0.55)
                mark_overflow(ax, s, y)
        relative_axis(ax, label=False)
        asset_axis(ax, positions, labels=ax is axes[0])
        ax.set_title(hazard)
    fig.supxlabel("EAD / mean EAD of the same combination (typical country)", fontsize=8, color=INK_2, y=-0.01)
    box_key(fig, sector_handles())
    save(fig, "box_b_relative_by_hazard", FOLDER)


def _country_panel(ax, rows, countries, color_of, lo, hi, scale="log"):
    """lo/hi are the shared log limits; a linear panel runs from 0 to its own largest p95 or mean."""
    log = scale == "log"
    if rows.empty:
        ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", color=INK_2)
    ypos = {c: i for i, c in enumerate(countries)}
    for _, s in rows.iterrows():
        if s["p95"] <= 0:
            ax.scatter(lo if log else 0, ypos[s["country"]], marker="|", s=10, color=BASE, zorder=2,
                       clip_on=False)
            continue
        draw_box(ax, ypos[s["country"]], s, color_of(s), height=0.6, lo_clip=lo if log else None)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(lo / 3, hi * 3)               # outer tick labels clear of the next panel
        euro_ticks(ax.xaxis, lo, hi, max_ticks=5)
    else:
        top = rows[["p95", "mean"]].to_numpy().max() if not rows.empty else 0
        top = top if top > 0 else 1
        ax.set_xlim(-0.03 * top, top * 1.05)      # marks at 0 are not cut in half by the edge
        euro_linear(ax.xaxis, nbins=4)
        if top <= 0:
            ax.set_xticks([0])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=6.5)
    ax.set_ylim(len(countries) - 0.4, -0.6)


def _log_limits(values):
    v = values[values > 0]
    return 10 ** np.floor(np.log10(v.min())), 10 ** np.ceil(np.log10(v.max()))


def _axis_note(scale):
    if scale == "log":
        return "Grey tick at the left: no damage in any run; arrow: p5 below the axis"
    return "Linear axis from 0, scaled per panel. Grey tick at 0: no damage in any run"


def box_countries_by_asset(stats, scale="log"):
    folder = FOLDER / "box_c_countries_by_asset"
    head = stats[stats["scenario"].isin(HAZARDS.values())]
    countries = sorted(head["country"].unique())
    log = scale == "log"
    for asset in ASSETS:
        sub = head[head["asset"] == asset]
        if sub.empty:
            continue
        lo, hi = _log_limits(pd.concat([sub["p5"], sub["p95"], sub["mean"]]))
        lo = max(lo, hi / 1e8)
        fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 6.6), sharey=True, sharex=log,
                                 gridspec_kw={"wspace": 0.08 if log else 0.2})
        for ax, (hazard, scenario) in zip(axes, HAZARDS.items()):
            rows = sub[sub["scenario"] == scenario]
            _country_panel(ax, rows, countries, lambda s, h=hazard: HAZARD_COLOR[h], lo, hi, scale)
            ax.set_title(hazard)
            ax.tick_params(axis="y", labelleft=ax is axes[0])
        fig.supxlabel(f"{ASSET_LABEL[asset]}: expected annual damage (€/yr). {_axis_note(scale)}",
                      fontsize=8, color=INK_2, y=0.0)
        box_key(fig, hazard_handles(), ncol=4)
        save(fig, f"box_c_{asset}{SUFFIX[scale]}", folder)


def box_countries_by_hazard(stats, scale="log"):
    folder = FOLDER / "box_d_countries_by_hazard"
    log = scale == "log"
    for hazard, scenario in HAZARDS.items():
        sub = stats[stats["scenario"] == scenario]
        countries = sorted(sub["country"].unique())
        lo, hi = _log_limits(pd.concat([sub["p5"], sub["p95"], sub["mean"]]))
        lo = max(lo, hi / 1e8)
        fig, axes = plt.subplots(2, 5, figsize=(FULL_WIDTH * 1.35, 10.5), sharey=True, sharex=log,
                                 gridspec_kw={"wspace": 0.08 if log else 0.2, "hspace": 0.14})
        for ax, asset in zip(axes.flat, ASSETS):
            rows = sub[sub["asset"] == asset]
            _country_panel(ax, rows, countries, lambda s: SECTOR_COLOR[SECTOR_OF[s["asset"]]], lo, hi, scale)
            ax.set_title(ASSET_LABEL[asset])
            ax.tick_params(axis="x", labelbottom=True, labelsize=6.5)
        fig.supxlabel(f"{hazard}: expected annual damage (€/yr). {_axis_note(scale)}",
                      fontsize=8, color=INK_2, y=0.03)
        box_key(fig, sector_handles())
        save(fig, f"box_d_{hazard.lower().replace(' ', '_')}{SUFFIX[scale]}", folder)


def box_treatments(rel):
    positions = y_positions()
    order = list(TREATMENTS)
    offsets = dict(zip(order, np.linspace(-0.34, 0.34, len(order))))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 6.2), sharey=True, gridspec_kw={"wspace": 0.08})
    used, rows = set(), []
    for ax, (hazard, scens) in zip(axes, TREATMENT_SCENARIOS.items()):
        sub = rel[rel["scenario"].isin(scens)]
        for asset, y in positions.items():
            mine = sub[sub["asset"] == asset]
            if mine.empty:
                continue
            common = set.intersection(*(set(g["country"]) for _, g in mine.groupby("scenario")))
            t = typical(mine[mine["country"].isin(common)], ["scenario"])
            for scenario, s in t.iterrows():
                tr = TREATMENT_OF[scenario]
                used.add(tr)
                draw_box(ax, y + offsets[tr], s, TREATMENT_COLOR[tr], height=0.11)
                mark_overflow(ax, s, y + offsets[tr])
                rows.append({"hazard": hazard, "asset": asset, "treatment": tr, "n_countries": len(common),
                             **s[STATS].to_dict()})
        relative_axis(ax, label=False)
        asset_axis(ax, positions, labels=ax is axes[0])
        ax.set_title(hazard)
    fig.supxlabel("EAD / mean EAD (typical country, countries common to all treatments of the asset)",
                  fontsize=8, color=INK_2, y=0.0)
    handles = [Patch(facecolor=tint(TREATMENT_COLOR[t]), edgecolor=TREATMENT_COLOR[t],
                     label=TREATMENT_NAME.get(t, t)) for t in order if t in used]
    box_key(fig)
    legend_above(fig, handles, ncol=3)
    save(fig, "box_e_treatments", FOLDER)
    pd.DataFrame(rows).to_csv(FOLDER / "box_e_treatments.csv", index=False)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

BINS = np.linspace(0, REL_MAX, 41)


def rel_hist(values):
    """Share of values per bin (bins of 0.2), and the share beyond REL_MAX (not drawn)."""
    counts, _ = np.histogram(values, bins=BINS)
    share = counts / len(values)
    return share, float((values > REL_MAX).mean())


def hist_grid(q):
    hazards = list(HAZARDS)
    fig, axes = plt.subplots(len(ASSETS), 4, figsize=(FULL_WIDTH, 9.5), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.06, "hspace": 0.18})
    centers = (BINS[:-1] + BINS[1:]) / 2
    for j, hazard in enumerate(hazards):
        axes[0, j].set_title(hazard)
        for i, asset in enumerate(ASSETS):
            ax = axes[i, j]
            values, n = pooled_relative(q, HAZARDS[hazard], asset)
            ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.axvline(1, color=BASE, lw=0.6, zorder=0)
            if n == 0:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.5, color=INK_2)
            else:
                share, over = rel_hist(values)
                ax.bar(centers, share, width=BINS[1] - BINS[0], color=HAZARD_COLOR[hazard], linewidth=0)
                note = f"n={n}" + (f", {over:.0%} > {REL_MAX:g}" if over >= 0.005 else "")
                ax.text(0.98, 0.9, note, transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                        color=INK_2)
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(ASSET_LABEL[asset], rotation=0, ha="right", va="center", color=INK, fontsize=7)
            for side in ("left",):
                ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlim(0, REL_MAX)
        ax.set_xticks([0, 1, 2, 4, 6, 8])
    fig.supxlabel("EAD / mean EAD of the same combination (all countries pooled, equal weight)",
                  fontsize=8, color=INK_2, y=0.06)
    save(fig, "hist_a_grid", FOLDER)


def hist_hazards_per_asset(q):
    fig, axes = plt.subplots(2, 5, figsize=(FULL_WIDTH, 3.6), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.35})
    for ax, asset in zip(axes.flat, ASSETS):
        for hazard, scenario in HAZARDS.items():
            values, n = pooled_relative(q, scenario, asset)
            if n == 0:
                continue
            share, _ = rel_hist(values)
            ax.stairs(share, BINS, color=HAZARD_COLOR[hazard], lw=1.1)
        ax.axvline(1, color=BASE, lw=0.6, zorder=0)
        ax.set_title(ASSET_LABEL[asset], fontsize=8)
        ax.set_xlim(0, REL_MAX)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.supxlabel("EAD / mean EAD (all countries pooled)", fontsize=8, color=INK_2, y=-0.04)
    handles = [Line2D([], [], color=HAZARD_COLOR[h], lw=1.4, label=h) for h in HAZARDS]
    legend_above(fig, handles, ncol=4)
    save(fig, "hist_b_hazards_per_asset", FOLDER)


def hist_ridgeline(q):
    positions = y_positions()
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (hazard, scenario) in zip(axes, HAZARDS.items()):
        peak = max((rel_hist(pooled_relative(q, scenario, a)[0])[0].max()
                    for a in ASSETS if pooled_relative(q, scenario, a)[1]), default=1)
        for asset, y in positions.items():
            values, n = pooled_relative(q, scenario, asset)
            if n == 0:
                continue
            share, _ = rel_hist(values)
            h = share / peak * 1.4                      # ridges may overlap the row above
            color = SECTOR_COLOR[SECTOR_OF[asset]]
            ax.stairs(y + 0.4 - h, BINS, baseline=y + 0.4, fill=True, color=tint(color, 0.35), lw=0, zorder=2)
            ax.stairs(y + 0.4 - h, BINS, baseline=None, color=color, lw=0.8, zorder=3)
        ax.axvline(1, color=BASE, lw=0.6, zorder=0)
        ax.set_xlim(0, REL_MAX)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        asset_axis(ax, positions, labels=ax is axes[0])
        ax.set_ylim(max(positions.values()) + 0.6, min(positions.values()) - 1.2)
        ax.set_title(hazard)
    fig.supxlabel("EAD / mean EAD (all countries pooled); heights scaled within each hazard",
                  fontsize=8, color=INK_2, y=-0.01)
    legend_above(fig, sector_handles(), ncol=3)
    save(fig, "hist_c_ridgeline_per_hazard", FOLDER)


def _abs_hist(ax, row, color, bins, label=None, filled=True):
    cols = [c for c in row.index if c.startswith("q")]
    values = row[cols].to_numpy(float)
    zero = row["zero_share"]
    pos = values[values > 0]
    counts, _ = np.histogram(pos, bins=bins)
    share = counts / len(values)
    if filled:
        ax.stairs(share, bins, fill=True, color=tint(color, 0.3), lw=0)
    ax.stairs(share, bins, color=color, lw=1.0, label=label)
    return share.max(), zero


def _abs_bins(values, top, scale):
    """Log: 40 log bins over the positive values (at most 6 decades). Linear: 40 bins from 0
    to `top` (a p99, so a long tail does not squeeze everything into the first bin)."""
    if scale == "linear":
        return np.linspace(0, top, 41)
    pos = values[values > 0]
    hi = np.log10(pos.max())
    lo = max(np.log10(pos.min()), hi - 6)
    return np.logspace(lo, hi, 41)


def _abs_xaxis(ax, bins, scale, max_ticks):
    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        euro_ticks(ax.xaxis, bins[0], bins[-1], max_ticks=max_ticks)
    else:
        ax.set_xlim(-0.02 * bins[-1], bins[-1])    # marks at 0 are not cut in half by the edge
        euro_linear(ax.xaxis, nbins=max_ticks)


def hist_examples(q, scale="log"):
    log = scale == "log"
    for country, asset in EXAMPLES:
        sub = q[(q["country"] == country) & (q["asset"] == asset)]
        rows = {h: sub[sub["scenario"] == s] for h, s in HAZARDS.items()}
        rows = {h: r.iloc[0] for h, r in rows.items() if len(r)}
        if not rows:
            continue
        fig, axes = plt.subplots(2, len(rows), figsize=(FULL_WIDTH, 3.4), sharex="col",
                                 gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05, "wspace": 0.2})
        axes = np.atleast_2d(axes)
        for j, (hazard, row) in enumerate(rows.items()):
            cols = [c for c in row.index if c.startswith("q")]
            values = row[cols].to_numpy(float)
            if not (values > 0).any():
                continue
            bins = _abs_bins(values, row["q0990"], scale)
            top, bot = axes[0, j], axes[1, j]
            _, zero = _abs_hist(bot, row, HAZARD_COLOR[hazard], bins)
            s = {k: row[QCOL[k]] for k in STATS} | {"mean": row["mean"]}
            draw_box(top, 0, s, HAZARD_COLOR[hazard], height=0.7, lo_clip=bins[0] if log else None)
            for ax in (top, bot):
                _abs_xaxis(ax, bins, scale, max_ticks=4)
            top.set_ylim(-0.6, 0.6)
            top.axis("off")
            top.set_title(hazard)
            bot.set_yticks([])
            bot.spines["left"].set_visible(False)
            if zero >= 0.005:
                bot.text(0.98 if not log else 0.02, 0.95, f"{zero:.0%} of runs zero", transform=bot.transAxes,
                         fontsize=6.5, color=INK_2, va="top", ha="left" if log else "right")
        note = "" if log else "; linear axis from 0 to the p99 of each panel, scaled per panel"
        fig.supxlabel(f"{country} {ASSET_LABEL[asset].lower()}: expected annual damage (€/yr){note}",
                      fontsize=8, color=INK_2, y=-0.04)
        box_key(fig)
        save(fig, f"hist_d_example_{country}_{asset}{SUFFIX[scale]}", FOLDER)


def hist_treatments(q, scale="log"):
    log = scale == "log"
    for country, asset, hazard in TREATMENT_EXAMPLES:
        sub = q[(q["country"] == country) & (q["asset"] == asset)
                & q["scenario"].isin(TREATMENT_SCENARIOS[hazard])]
        if sub.empty:
            continue
        cols = [c for c in q.columns if c.startswith("q")]
        bins = _abs_bins(sub[cols].to_numpy(float).ravel(), sub["q0990"].max(), scale)
        order = [s for s in TREATMENT_SCENARIOS[hazard] if s in set(sub["scenario"])]
        fig, axes = plt.subplots(len(order), 1, figsize=(FULL_WIDTH * 0.6, 0.95 * len(order) + 0.8),
                                 sharex=True, gridspec_kw={"hspace": 0.4})
        axes = np.atleast_1d(axes)
        for ax, scenario in zip(axes, order):
            row = sub[sub["scenario"] == scenario].iloc[0]
            t = TREATMENT_OF[scenario]
            color = TREATMENT_COLOR[t]
            _, zero = _abs_hist(ax, row, color, bins)
            s = {k: row[QCOL[k]] for k in STATS} | {"mean": row["mean"]}
            y = ax.get_ylim()[1] * 1.08
            draw_box(ax, y, s, color, height=ax.get_ylim()[1] * 0.14, lo_clip=bins[0] if log else None)
            ax.set_ylim(0, y * 1.12)
            _abs_xaxis(ax, bins, scale, max_ticks=5)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            name = TREATMENT_NAME.get(t, t)
            if t == "Recorded / fixed RP50":
                name = "Fixed RP 50" if hazard == "Windstorm" else "As recorded"
            ax.text(0.0, 1.0, name + (f"  ({zero:.0%} of runs zero)" if zero >= 0.005 else ""),
                    transform=ax.transAxes, fontsize=7, color=INK, va="bottom")
        for ax in axes[:-1]:
            ax.tick_params(axis="x", bottom=False)
        note = "" if log else "\nlinear axis from 0 to the largest p99 over the treatments"
        axes[-1].set_xlabel(f"{country} {ASSET_LABEL[asset].lower()}, {hazard.lower()}: EAD (€/yr){note}")
        box_key(fig)
        save(fig, f"hist_e_treatments_{country}_{asset}_{hazard.lower().replace(' ', '_')}{SUFFIX[scale]}",
             FOLDER)


def short_treatment(t, hazard):
    if t == "Recorded / fixed RP50":
        return "Fixed RP 50" if hazard == "Windstorm" else "As recorded"
    return t


def _panel_rows(q, country, asset):
    """{hazard: {scenario: row or None}}; None = part of the panel but not run for this combination."""
    sub = q[(q["country"] == country) & (q["asset"] == asset)].set_index("scenario")
    return {h: {s: (sub.loc[s] if s in sub.index else None) for s in scens}
            for h, scens in PANEL_SCENARIOS.items()}


def _panel_log_limits(panels):
    rows = [r for p in panels.values() for r in p.values() if r is not None]
    lo, hi = _log_limits(pd.Series([r[k] for r in rows for k in ("q0010", "q0950", "mean", "q1000")]))
    return max(lo, hi / 1e7), hi


def _not_run(ax, y, text="not run"):
    ax.text(0.02, y, text, transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha="left", va="center", fontsize=6, color=INK_2)


def box_treatments_by_hazard(q, scale="log"):
    folder = FOLDER / "treatments_by_hazard"
    log = scale == "log"
    order = list(TREATMENTS)
    for country, asset in PANEL_EXAMPLES:
        panels = _panel_rows(q, country, asset)
        if not any(r is not None for p in panels.values() for r in p.values()):
            continue
        lo, hi = _panel_log_limits(panels)
        fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.6), sharey=True, sharex=log,
                                 gridspec_kw={"wspace": 0.08 if log else 0.2})
        for ax, (hazard, rows) in zip(axes, panels.items()):
            top = 0.0
            for scenario, row in rows.items():
                t = TREATMENT_OF[scenario]
                y = order.index(t)
                if row is None:
                    _not_run(ax, y)
                    continue
                s = {k: row[QCOL[k]] for k in STATS} | {"mean": row["mean"]}
                top = max(top, s["p95"], s["mean"])
                if s["p95"] <= 0:
                    ax.scatter(lo if log else 0, y, marker="|", s=14, color=BASE, zorder=2, clip_on=False)
                    continue
                draw_box(ax, y, s, TREATMENT_COLOR[t], height=0.55, lo_clip=lo if log else None)
            if log:
                ax.set_xscale("log")
                ax.set_xlim(lo / 3, hi * 3)
                euro_ticks(ax.xaxis, lo, hi, max_ticks=4)
            else:
                top = top if top > 0 else 1
                ax.set_xlim(-0.03 * top, top * 1.05)
                euro_linear(ax.xaxis, nbins=3)
            ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(hazard)
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels([TREATMENT_NAME.get(t, t) for t in order], fontsize=7)
        axes[0].set_ylim(len(order) - 0.5, -0.5)
        note = ("Grey tick: no damage in any run; arrow: p5 below the axis" if log
                else "Linear axis from 0, scaled per panel")
        fig.supxlabel(f"{country} {ASSET_LABEL[asset].lower()}: expected annual damage (€/yr). {note}",
                      fontsize=8, color=INK_2, y=-0.06)
        box_key(fig)
        save(fig, f"box_f_treatments_{country}_{asset}{SUFFIX[scale]}", folder)


def _line_hist(values, bins):
    """Share of the positive runs per bin, smoothed over three bins and scaled to its peak."""
    counts, _ = np.histogram(values[values > 0], bins=bins)
    smooth = np.convolve(np.pad(counts.astype(float), 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return smooth / smooth.max() if smooth.max() > 0 else smooth


def hist_treatments_by_hazard(q, scale="log"):
    folder = FOLDER / "treatments_by_hazard"
    log = scale == "log"
    cols = [c for c in q.columns if c.startswith("q")]
    for country, asset in PANEL_EXAMPLES:
        panels = _panel_rows(q, country, asset)
        if not any(r is not None for p in panels.values() for r in p.values()):
            continue
        lo, hi = _panel_log_limits(panels)
        fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.6), sharey=True, sharex=log,
                                 gridspec_kw={"wspace": 0.08 if log else 0.2})
        used = []
        for ax, (hazard, rows) in zip(axes, panels.items()):
            run = {s: r for s, r in rows.items() if r is not None}
            if log:
                bins = np.logspace(np.log10(lo), np.log10(hi), 61)
                centres = np.sqrt(bins[:-1] * bins[1:])
            else:
                top = max((r["q0990"] for r in run.values()), default=0)
                bins = np.linspace(0, top if top > 0 else 1, 61)
                centres = (bins[:-1] + bins[1:]) / 2
            notes = []
            for scenario, row in rows.items():
                t = TREATMENT_OF[scenario]
                name = short_treatment(t, hazard)
                if row is None:
                    notes.append(f"{name}: not run")
                    continue
                values = row[cols].to_numpy(float)
                if row["zero_share"] >= 0.005:
                    notes.append(f"{name}: {row['zero_share']:.0%} of runs zero")
                if not (values > 0).any():
                    continue
                ax.plot(centres, _line_hist(values, bins), color=TREATMENT_COLOR[t], lw=1.3, zorder=3)
                if t not in used:
                    used.append(t)
            if log:
                ax.set_xscale("log")
                ax.set_xlim(lo / 3, hi * 3)           # outer tick labels clear of the next panel
                euro_ticks(ax.xaxis, lo, hi, max_ticks=4)
            else:
                ax.set_xlim(-0.02 * bins[-1], bins[-1])
                euro_linear(ax.xaxis, nbins=3)
            ax.set_ylim(0, 1.08 + 0.13 * len(notes))
            for i, text in enumerate(notes):
                ax.text(0.98, 0.98 - 0.075 * i, text, transform=ax.transAxes, ha="right", va="top",
                        fontsize=5.5, color=INK_2)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(hazard)
        note = ("log axis from the smallest p1, at most 7 decades below the largest run" if log
                else "linear axis from 0 to the largest p99 in each panel")
        fig.supxlabel(f"{country} {ASSET_LABEL[asset].lower()}: expected annual damage (€/yr); {note}.\n"
                      "Each line is the share of runs per bin, smoothed over three bins and scaled to its own "
                      "peak; zero-damage runs are not drawn", fontsize=7.5, color=INK_2, y=-0.12)
        order = [t for t in TREATMENTS if t in used]
        handles = [Line2D([], [], color=TREATMENT_COLOR[t], lw=1.6, label=TREATMENT_NAME.get(t, t))
                   for t in order]
        legend_above(fig, handles, ncol=3)
        save(fig, f"hist_f_treatments_{country}_{asset}{SUFFIX[scale]}", folder)


def main() -> None:
    set_style()
    FOLDER.mkdir(parents=True, exist_ok=True)
    q = load_quantiles()
    stats = load_stats(q)
    rel = relative(stats)
    box_relative_by_asset(rel)
    box_relative_by_hazard(rel)
    for scale in SUFFIX:
        box_countries_by_asset(stats, scale)
        box_countries_by_hazard(stats, scale)
    if q is None:
        return
    box_treatments(rel)
    hist_grid(q)
    hist_hazards_per_asset(q)
    hist_ridgeline(q)
    for scale in SUFFIX:
        hist_examples(q, scale)
        hist_treatments(q, scale)
        box_treatments_by_hazard(q, scale)
        hist_treatments_by_hazard(q, scale)


if __name__ == "__main__":
    main()
