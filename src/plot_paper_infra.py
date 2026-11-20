"""Paper figures comparing uncertainty and its drivers across infrastructure types.

Every figure reads results already on disk; nothing is re-run.

    fig01  width of the uncertainty range, per country, by infrastructure
    fig02  what drives it: normalised Sobol ST by driver (damage-weighted)
    fig03  the same with protection not sampled
    fig04  top driver for every country x infrastructure pair
    fig05  how much of the driver pattern infrastructure vs country explains
    fig06  how much of the variance comes from factor interactions
    fig07  damage size against relative uncertainty
    fig08  width with and without sampled protection
    fig09  where the pan-European range opens up along the modelling chain

fig01, fig07 and fig08 are each drawn twice: with the p5-p95 range divided by the
median, and in a *_mean version divided by the mean. The mean version also covers
pairs whose median is zero, which have no median-based width.

Coastal is the sea-level-rise scenario throughout, except fig09: no SLR cascade
has been run yet, so that figure leaves coastal out instead of showing a
present-day run.

Sources: All_Sobol_Indices in MIRACA_uncertainty_study_summary.xlsx,
overview_figures/ead_ranges/EAD_Ranges.csv and results/cascade/workflow-35e52770.
Each figure's numbers are written to paper/data/.

Usage:
    python -m src.plot_paper_infra
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from .paths import PROJECT_ROOT
from .plot_drivers import COLORS, OUTCOME, SHEET, WORKBOOK_NAME, category

OUT = PROJECT_ROOT / "paper"
DATA = OUT / "data"
SUPP = OUT / "supplement"
RANGES_CSV = PROJECT_ROOT / "overview_figures" / "ead_ranges" / "EAD_Ranges.csv"
# Flood, coastal (SLR) and wind scenarios only, including the November 2026 treatments
# and the river reruns with the 5 km protection. Only fig08 ..._all_treatments reads it.
RANGES_TREATMENTS_CSV = PROJECT_ROOT / "overview_figures" / "ead_ranges_20261117" / "EAD_Ranges.csv"
CASCADE_DIR = PROJECT_ROOT / "results" / "cascade" / "workflow-35e52770"

SECTORS = {
    "Transport": ["roads", "rail", "airports", "ports"],
    "Utilities": ["power", "gas", "oil", "telecom"],
    "Social": ["education", "healthcare"],
}
SECTOR_COLOR = {"Transport": "#2a78d6", "Utilities": "#eb6834", "Social": "#1baf7a"}
ASSETS = [a for group in SECTORS.values() for a in group]
SECTOR_OF = {a: s for s, group in SECTORS.items() for a in group}
ASSET_LABEL = {a: a.capitalize() for a in ASSETS}

HAZARDS = {
    "River flood": "flood_absprot_ds",
    "Coastal flood": "coastal_absprot_ds_slr",
    "Earthquake": "earthquake",
    "Windstorm": "windstorm_absprot",
}
NOPROT = {
    "River flood": "flood_noprot_ds",
    "Coastal flood": "coastal_noprot_ds_slr",
    "Windstorm": "windstorm",
}

DRIVERS = [
    "Protection standard",
    "Vulnerability curve",
    "Hazard intensity",
    "Cost / max-damage",
    "Climate warming",
    "Aggregation method",
]
DRIVER_LABEL = {
    "Protection standard": "Protection",
    "Vulnerability curve": "Vulnerability",
    "Hazard intensity": "Hazard intensity",
    "Cost / max-damage": "Cost",
    "Climate warming": "Climate",
    "Aggregation method": "Aggregation",
}

INK, INK_2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WIDTH_MAX, WINDSTORM_MAX = 70, 5e4  # x-axis ends in fig01; values past them are marked
FULL_WIDTH = 7.2  # inches, two-column journal width
MEAN_WIDTH_MAX = 10  # shared linear x-axis end of the mean-based fig01


@dataclass(frozen=True)
class WidthMetric:
    column: str
    base: str  # a pair has a width only if this statistic is above zero
    label: str
    suffix: str  # appended to figure and data file names
    log: bool  # the median version spans four decades, the mean version 0 to 10


BY_MEDIAN = WidthMetric("w90_rel", "median", "(p95 − p5) / median of EAD", "", True)
BY_MEAN = WidthMetric("w90_mean", "mean", "(p95 − p5) / mean of EAD", "_mean", False)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ranges(path=RANGES_CSV) -> pd.DataFrame:
    ranges = pd.read_csv(path)
    if ranges.duplicated(["country", "asset", "scenario"]).any():
        raise SystemExit(f"Duplicate combinations in {path}")
    ranges["w90_mean"] = ranges["w90"] / ranges["mean"].where(ranges["mean"] > 0)
    return ranges


def load_sobol() -> pd.DataFrame:
    """One row per factor on total EAD, dropping combinations with no variance."""
    sobol = pd.read_excel(PROJECT_ROOT / WORKBOOK_NAME, sheet_name=SHEET)
    sobol = sobol[sobol["outcome"] == OUTCOME].copy()
    key = ["country", "asset", "scenario"]
    sobol = sobol[sobol.groupby(key)["ST"].transform("sum") > 0]
    sobol["driver"] = sobol["factor_code"].map(category)
    unknown = set(sobol["driver"]) - set(DRIVERS)
    if unknown:
        raise SystemExit(f"Unrecognised driver categories: {unknown}")
    return sobol


def driver_shares(sobol: pd.DataFrame, ranges: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Each driver's share of summed ST, per country x asset, with the EAD weight."""
    sub = sobol[sobol["scenario"] == scenario]
    shares = sub.groupby(["country", "asset", "driver"], as_index=False)["ST"].sum()
    shares["share"] = shares["ST"] / shares.groupby(["country", "asset"])["ST"].transform("sum")
    weight = ranges.loc[ranges["scenario"] == scenario, ["country", "asset", "mean"]]
    return shares.merge(weight, on=["country", "asset"], how="left")


def mean_shares(shares: pd.DataFrame, weighted: bool) -> pd.DataFrame:
    """asset x driver table of mean shares, reindexed so every driver is a column."""
    if weighted:
        grouped = shares[shares["mean"] > 0].copy()
        # Weights come from the pairs, not the rows of one driver: a driver that
        # a pair lacks simply contributes nothing to it.
        pair_weight = grouped.drop_duplicates(["country", "asset"]).groupby("asset")["mean"].sum()
        grouped["value"] = grouped["share"] * grouped["mean"]
        table = grouped.pivot_table(index="asset", columns="driver", values="value", aggfunc="sum")
        table = table.div(pair_weight, axis=0)
    else:
        n_pairs = shares.drop_duplicates(["country", "asset"]).groupby("asset").size()
        table = shares.pivot_table(index="asset", columns="driver", values="share", aggfunc="sum")
        table = table.div(n_pairs, axis=0)
    return table.reindex(index=ASSETS, columns=DRIVERS).fillna(0.0)


def y_positions() -> dict[str, float]:
    """Assets top to bottom, with a gap between sectors."""
    positions, y = {}, 0.0
    for group in SECTORS.values():
        for asset in group:
            positions[asset] = y
            y += 1.0
        y += 0.55
    return positions


def asset_axis(ax, positions: dict[str, float], labels: bool) -> None:
    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels([ASSET_LABEL[a] for a in positions])
    ax.tick_params(axis="y", labelleft=labels)
    ax.set_ylim(max(positions.values()) + 0.7, min(positions.values()) - 0.7)


def log_ticks(axis, subs=(1.0,)) -> None:
    """Plain numbers on a log axis, with no minor ticks or labels."""
    def plain(value, _pos=None):
        return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"
    axis.set_major_locator(LogLocator(base=10, subs=subs, numticks=12))
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def save(fig, name: str, folder=OUT) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png")


def driver_legend(fig, y: float = 1.0, drivers=DRIVERS) -> None:
    handles = [Patch(facecolor=COLORS[d], label=DRIVER_LABEL[d]) for d in drivers]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), handlelength=1.0, handleheight=1.0, columnspacing=1.4)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_width(ranges: pd.DataFrame, metric: WidthMetric) -> None:
    """fig01: one dot per country, median tick per infrastructure."""
    rows = []
    positions = y_positions()
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 3.5), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    rng = np.random.default_rng(20261109)
    n_off_scale = 0
    for ax, (hazard, scenario) in zip(axes, HAZARDS.items()):
        sub = ranges[(ranges["scenario"] == scenario) & (ranges[metric.base] > 0)]
        windstorm = metric.log and hazard == "Windstorm"
        x_max = WINDSTORM_MAX if windstorm else WIDTH_MAX if metric.log else MEAN_WIDTH_MAX
        for asset, y in positions.items():
            values = sub.loc[sub["asset"] == asset, metric.column].dropna().to_numpy()
            if not values.size:
                continue
            color = SECTOR_COLOR[SECTOR_OF[asset]]
            on_scale = values <= x_max
            jitter = y + rng.uniform(-0.2, 0.2, values.size)
            ax.scatter(values[on_scale], jitter[on_scale], s=7, color=color,
                       alpha=0.45, linewidths=0, zorder=2)
            ax.scatter(np.full((~on_scale).sum(), x_max * 0.88), jitter[~on_scale], s=11,
                       marker=">", color=color, linewidths=0, zorder=2, clip_on=False)
            n_off_scale += int((~on_scale).sum())
            ax.plot([np.median(values)] * 2, [y - 0.36, y + 0.36], color=INK, lw=1.5,
                    solid_capstyle="round", zorder=3)
            rows.append({"hazard": hazard, "asset": asset, "n_countries": values.size,
                         "median": np.median(values), "min": values.min(), "max": values.max()})
        if metric.log:
            ax.set_xscale("log")
            log_ticks(ax.xaxis, subs=(1.0,) if windstorm else (1.0, 3.0))
            ax.set_xlim(2 if windstorm else 0.6, x_max)
        else:
            ax.set_xlim(-0.3, x_max)  # a few coastal pairs sit at exactly zero
            ax.set_xticks([0, 5, 10])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(hazard)
        asset_axis(ax, positions, labels=ax is axes[0])
    fig.supxlabel(metric.label, fontsize=8, color=INK_2, y=-0.03)
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=c, label=s)
               for s, c in SECTOR_COLOR.items()]
    handles.append(Line2D([], [], color=INK, lw=1.5, label="Median across countries"))
    if n_off_scale:
        handles.append(Line2D([], [], ls="", marker=">", ms=4, color=MUTED, label="Off scale"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=len(handles),
               handletextpad=0.3, columnspacing=1.6)
    print(f"  fig01{metric.suffix}: {n_off_scale} country values beyond the axis")
    save(fig, f"fig01_uncertainty_width{metric.suffix}")
    pd.DataFrame(rows).to_csv(DATA / f"fig01_uncertainty_width{metric.suffix}.csv", index=False)


def stacked_panels(tables: dict[str, pd.DataFrame], name: str, folder=OUT, height=3.3,
                   xlabel: str = "Share of total-order sensitivity (%)") -> None:
    positions = y_positions()
    fig, axes = plt.subplots(1, len(tables), figsize=(FULL_WIDTH, height), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    axes = np.atleast_1d(axes)
    for ax, (hazard, table) in zip(axes, tables.items()):
        present = table.sum(axis=1) > 0
        ys = np.array([positions[a] for a in table.index[present]])
        left = np.zeros(len(ys))
        for driver in DRIVERS:
            values = table.loc[present, driver].to_numpy()
            ax.barh(ys, values, left=left, height=0.74, color=COLORS[driver],
                    edgecolor="white", linewidth=0.8, zorder=2)
            left += values
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50", "100" if ax is axes[-1] else ""])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(hazard)
        asset_axis(ax, positions, labels=ax is axes[0])
    used = [d for d in DRIVERS if any(t[d].sum() > 0.02 for t in tables.values())]
    driver_legend(fig, y=0.93, drivers=used)
    fig.supxlabel(xlabel, fontsize=8, color=INK_2, y=-0.03)
    save(fig, name, folder)


def fig_drivers(sobol, ranges) -> None:
    """fig02 and fig03, plus the equal-country-weight variant in supplement/."""
    all_rows = []
    weighted, unweighted = {}, {}
    for hazard, scenario in HAZARDS.items():
        shares = driver_shares(sobol, ranges, scenario)
        weighted[hazard] = mean_shares(shares, weighted=True)
        unweighted[hazard] = mean_shares(shares, weighted=False)
        for kind, table in (("damage_weighted", weighted[hazard]),
                            ("equal_country", unweighted[hazard])):
            all_rows.append(table.assign(hazard=hazard, weighting=kind).reset_index())
    stacked_panels(weighted, "fig02_driver_shares")
    stacked_panels(unweighted, "fig02_driver_shares_equal_country_weight", folder=SUPP)
    pd.concat(all_rows).to_csv(DATA / "fig02_driver_shares.csv", index=False)

    tables, rows = {}, []
    for hazard, scenario in NOPROT.items():
        shares = driver_shares(sobol, ranges, scenario)
        tables[hazard] = mean_shares(shares, weighted=True)
        rows.append(tables[hazard].assign(hazard=hazard).reset_index())
    stacked_panels(tables, "fig03_driver_shares_no_protection")
    pd.concat(rows).to_csv(DATA / "fig03_driver_shares_no_protection.csv", index=False)


def fig_top_driver(sobol, ranges) -> None:
    """fig04: columns that are one colour mean the infrastructure sets the driver."""
    codes = {d: i for i, d in enumerate(DRIVERS)}
    cmap = ListedColormap([COLORS[d] for d in DRIVERS])
    tops = []
    for hazard, scenario in HAZARDS.items():
        shares = driver_shares(sobol, ranges, scenario)
        top = shares.sort_values("share", ascending=False).drop_duplicates(["country", "asset"])
        tops.append(top.assign(hazard=hazard))
    tops = pd.concat(tops)
    countries = sorted(tops["country"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 6.2), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, hazard in zip(axes, HAZARDS):
        sub = tops[tops["hazard"] == hazard]
        grid = (sub.assign(code=sub["driver"].map(codes))
                .pivot(index="country", columns="asset", values="code")
                .reindex(index=countries, columns=ASSETS))
        ax.set_facecolor("#f0efec")
        ax.pcolormesh(np.arange(len(ASSETS) + 1), np.arange(len(countries) + 1),
                      np.ma.masked_invalid(grid.to_numpy(dtype=float)), cmap=cmap, vmin=-0.5,
                      vmax=len(DRIVERS) - 0.5, edgecolors="white", linewidth=0.5)
        ax.set_xticks(np.arange(len(ASSETS)) + 0.5)
        ax.set_xticklabels([ASSET_LABEL[a] for a in ASSETS], rotation=90, fontsize=7)
        ax.tick_params(axis="x", length=0, colors=INK)
        ax.set_yticks(np.arange(len(countries)) + 0.5)
        ax.set_yticklabels(countries, fontsize=6.5)
        ax.set_ylim(len(countries), 0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(hazard)
    used = [d for d in DRIVERS if d in set(tops["driver"])]
    handles = [Patch(facecolor=COLORS[d], label=DRIVER_LABEL[d]) for d in used]
    handles.append(Patch(facecolor="#f0efec", label="No damage or no asset"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.90), ncol=len(handles),
               handlelength=1.0, handleheight=1.0, columnspacing=1.2)
    save(fig, "fig04_top_driver_by_country")
    tops[["hazard", "country", "asset", "driver", "share"]].to_csv(
        DATA / "fig04_top_driver_by_country.csv", index=False)


def _r_squared(y: np.ndarray, *factors: pd.Series) -> float:
    columns = [np.ones(len(y))]
    for factor in factors:
        columns.append(pd.get_dummies(factor, drop_first=True).to_numpy(dtype=float))
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    total = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - design @ beta) ** 2).sum() / total if total > 0 else np.nan


def fig_infra_vs_country(sobol, ranges) -> None:
    """fig05: Shapley split of the R2 of each driver's share on asset and country."""
    rows = []
    for hazard, scenario in HAZARDS.items():
        shares = driver_shares(sobol, ranges, scenario)
        wide = (shares.pivot_table(index=["country", "asset"], columns="driver", values="share")
                .fillna(0.0).reset_index())
        for driver in DRIVERS:
            if driver not in wide or wide[driver].mean() < 0.03:
                continue
            y = wide[driver].to_numpy()
            r_asset = _r_squared(y, wide["asset"])
            r_country = _r_squared(y, wide["country"])
            r_both = _r_squared(y, wide["asset"], wide["country"])
            rows.append({
                "hazard": hazard, "driver": driver,
                "infrastructure": (r_asset + r_both - r_country) / 2,
                "country": (r_country + r_both - r_asset) / 2,
                "unexplained": 1 - r_both,
            })
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "fig05_infrastructure_vs_country.csv", index=False)

    colors = {"infrastructure": "#2a78d6", "country": "#a09c8e", "unexplained": "#e8e6df"}
    counts = [int((table["hazard"] == h).sum()) for h in HAZARDS]
    fig, axes = plt.subplots(len(HAZARDS), 1, figsize=(4.6, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": counts, "hspace": 0.32})
    for ax, hazard in zip(axes, HAZARDS):
        sub = table[table["hazard"] == hazard].reset_index(drop=True)
        left = np.zeros(len(sub))
        for part, color in colors.items():
            ax.barh(sub.index, sub[part], left=left, height=0.72, color=color,
                    edgecolor="white", linewidth=0.8)
            left += sub[part].to_numpy()
        ax.set_yticks(sub.index)
        ax.set_yticklabels([DRIVER_LABEL[d] for d in sub["driver"]])
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50", "100"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(hazard, pad=2)
    handles = [Patch(facecolor=colors["infrastructure"], label="Infrastructure type"),
               Patch(facecolor=colors["country"], label="Country"),
               Patch(facecolor=colors["unexplained"], label="Neither alone")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=3,
               handlelength=1.0, handleheight=1.0)
    fig.supxlabel("Variation in driver share explained (%)", fontsize=8, color=INK_2, y=0.0)
    save(fig, "fig05_infrastructure_vs_country")


def fig_interactions(sobol) -> None:
    """fig06: 1 - sum(S1), median across countries."""
    rows = []
    for hazard, scenario in HAZARDS.items():
        sub = sobol[sobol["scenario"] == scenario]
        additive = sub.assign(S1=sub["S1"].clip(lower=0)).groupby(["country", "asset"])["S1"].sum()
        inter = (1 - additive).clip(0, 1).groupby("asset").median()
        rows.append(inter.rename(hazard))
    table = pd.concat(rows, axis=1).reindex(ASSETS)
    table.to_csv(DATA / "fig06_interactions.csv")

    ramp = LinearSegmentedColormap.from_list("blue", ["#eaf2fd", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.set_facecolor("#f0efec")
    mesh = ax.pcolormesh(np.arange(5), np.arange(11), np.ma.masked_invalid(table.to_numpy()),
                         cmap=ramp, vmin=0, vmax=0.7, edgecolors="white", linewidth=1.2)
    ax.set_ylim(10, 0)
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_xticklabels([h.replace(" flood", "") for h in table.columns])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, colors=INK)
    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_yticklabels([ASSET_LABEL[a] for a in ASSETS])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, ticks=[0, 0.35, 0.7])
    bar.ax.set_yticklabels(["0", "35", "70%"])
    bar.ax.tick_params(length=0, colors=INK_2)
    bar.outline.set_visible(False)
    bar.set_label("Variance from interactions", color=INK_2)
    save(fig, "fig06_interactions")


def _place_labels(ax, points: list[tuple[float, float, str]]) -> None:
    """Label each point at the first nearby spot clear of points and other labels."""
    candidates = [
        (5, 0, "left", "center"), (-5, 0, "right", "center"),
        (0, 5, "center", "bottom"), (0, -5, "center", "top"),
        (5, 5, "left", "bottom"), (5, -5, "left", "top"),
        (-5, 5, "right", "bottom"), (-5, -5, "right", "top"),
        (14, 0, "left", "center"), (-14, 0, "right", "center"),
        (0, 13, "center", "bottom"), (0, -13, "center", "top"),
    ]
    # Further out, joined to the point by a leader line: only when nothing close is free.
    candidates += [(dx * r, dy * r, ha, va)
                   for r in (20, 30, 40)
                   for dx, dy, ha, va in ((0, 1, "center", "bottom"), (0, -1, "center", "top"),
                                          (0.8, 0.8, "left", "bottom"), (-0.8, 0.8, "right", "bottom"),
                                          (0.8, -0.8, "left", "top"), (-0.8, -0.8, "right", "top"),
                                          (1, 0, "left", "center"), (-1, 0, "right", "center"))]
    leader = {"arrowstyle": "-", "lw": 0.4, "color": INK_2, "shrinkA": 1, "shrinkB": 3}
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    reach = 3 * fig.dpi / 72
    pixels = [tuple(ax.transData.transform((x, y))) for x, y, _ in points]
    taken = [Bbox.from_extents(px - reach, py - reach, px + reach, py + reach) for px, py in pixels]

    def gap(box, px, py):
        return np.hypot(max(box.x0 - px, 0, px - box.x1), max(box.y0 - py, 0, py - box.y1))

    for x, y, text in sorted(points, key=lambda p: p[1], reverse=True):
        chosen = None
        own = tuple(ax.transData.transform((x, y)))
        for dx, dy, ha, va in candidates:
            far = abs(dx) + abs(dy) > 15
            label = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=7, color=INK,
                                arrowprops=leader if far else None)
            label.update_positions(renderer)
            box = Text.get_window_extent(label, renderer)      # the text alone, not the leader
            inside = (box.x0 >= frame.x0 and box.x1 <= frame.x1
                      and box.y0 >= frame.y0 and box.y1 <= frame.y1)
            # Without a leader line the label must sit nearer its own dot than any other.
            mine = far or gap(box, *own) < min((gap(box, *p) for p in pixels if p != own),
                                               default=np.inf)
            if inside and mine and not any(box.overlaps(other) for other in taken):
                chosen = box
                break
            label.remove()
        if chosen is None:
            print(f"  WARNING: no clear spot for {text} in {ax.get_title() or ax.get_title(loc='left')}")
            label = ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points",
                                va="center", fontsize=7, color=INK)
            chosen = label.get_window_extent(renderer)
        taken.append(chosen)


def _padded_limits(values, log: bool, pad: float = 0.22) -> tuple[float, float]:
    """Data limits widened by pad of their span on each side (in log space if log)."""
    v = np.log10(values) if log else np.asarray(values, float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = hi - lo if hi > lo else 1.0
    lo, hi = lo - pad * span, hi + pad * span
    return (10 ** lo, 10 ** hi) if log else (lo, hi)


def fig_size_vs_width(ranges, metric: WidthMetric, shared: bool = False) -> None:
    """fig07: pan-European expected damage against typical relative width.

    shared=True gives every hazard panel the same x and y limits (fig07 ..._shared).
    """
    tables = {}
    for hazard, scenario in HAZARDS.items():
        sub = ranges[(ranges["scenario"] == scenario) & (ranges[metric.base] > 0)]
        table = sub.groupby("asset").agg(ead=("mean", "sum"), width=(metric.column, "median"))
        table = table.reindex([a for a in ASSETS if a in table.index])
        table["ead"] /= 1e6
        tables[hazard] = table
    if shared:
        every = pd.concat(tables.values())
        xlim = _padded_limits(every["ead"], log=True, pad=0.06)
        ylim = _padded_limits(every["width"], log=metric.log, pad=0.08)
        if not metric.log:
            ylim = (max(ylim[0], 0.0), ylim[1])
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH * 0.85, 5.0), sharex=shared, sharey=shared,
                             gridspec_kw={"wspace": 0.12 if shared else 0.2, "hspace": 0.32})
    for ax, (hazard, table) in zip(axes.flat, tables.items()):
        ax.scatter(table["ead"], table["width"], s=24, linewidths=0, zorder=3,
                   c=[SECTOR_COLOR[SECTOR_OF[a]] for a in table.index])
        ax.set_xscale("log")
        ax.grid(color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        if metric.log:
            ax.set_yscale("log")
        if shared:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.tick_params(labelbottom=True)
        else:
            ax.margins(x=0.22, y=0.22)
        log_ticks(ax.xaxis)
        if metric.log:
            log_ticks(ax.yaxis, subs=(1.0, 2.0, 5.0))
        ax.tick_params(axis="y", colors=INK_2, length=0)
        ax.set_title(hazard)
        rows.append(table.assign(hazard=hazard).reset_index())
    for ax, table in zip(axes.flat, tables.values()):      # limits are final only now
        _place_labels(ax, [(r.ead, r.width, ASSET_LABEL[a]) for a, r in table.iterrows()])
    fig.supylabel(metric.label, fontsize=8, color=INK_2, x=0.04)
    fig.supxlabel("Pan-European expected annual damage (M€)", fontsize=8, color=INK_2, y=0.02)
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=c, label=s)
               for s, c in SECTOR_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=3,
               handletextpad=0.3)
    name = f"fig07_damage_vs_width{metric.suffix}{'_shared' if shared else ''}"
    save(fig, name)
    pd.concat(rows).to_csv(DATA / f"{name}.csv", index=False)


def fig_protection_effect(ranges, metric: WidthMetric) -> None:
    """fig08: the same country x asset pairs with and without sampled protection."""
    rows = []
    positions = y_positions()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH * 0.82, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    on_color, off_color = COLORS["Protection standard"], INK_2
    column, on_col, off_col = metric.column, f"{metric.column}_on", f"{metric.column}_off"
    for ax, (hazard, off_scenario) in zip(axes, NOPROT.items()):
        key = ["country", "asset"]
        on = ranges[(ranges["scenario"] == HAZARDS[hazard]) & (ranges[metric.base] > 0)]
        off = ranges[(ranges["scenario"] == off_scenario) & (ranges[metric.base] > 0)]
        both = on[key + [column]].merge(off[key + [column]], on=key, suffixes=("_on", "_off"))
        table = both.groupby("asset")[[on_col, off_col]].median().reindex(ASSETS).dropna()
        for asset, row in table.iterrows():
            y = positions[asset]
            ax.plot([row[off_col], row[on_col]], [y, y], color=BASE, lw=1.2, zorder=1)
            ax.scatter(row[off_col], y, s=26, color=off_color, linewidths=0, zorder=2)
            ax.scatter(row[on_col], y, s=26, color=on_color, linewidths=0, zorder=3)
        if metric.log:
            ax.set_xscale("log")
            log_ticks(ax.xaxis, subs=(1.0, 3.0))
            ax.margins(x=0.14)
        else:
            ax.set_xlim(0, 8)
            ax.set_xticks([0, 2, 4, 6, 8])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(hazard)
        asset_axis(ax, positions, labels=ax is axes[0])
        rows.append(table.assign(hazard=hazard).reset_index())
    fig.supxlabel(metric.label, fontsize=8, color=INK_2, y=-0.03)
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=on_color, label="Protection sampled"),
               Line2D([], [], ls="", marker="o", ms=5, color=off_color, label="Protection not sampled")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=2,
               handletextpad=0.3)
    save(fig, f"fig08_protection_effect{metric.suffix}")
    pd.concat(rows).to_csv(DATA / f"fig08_protection_effect{metric.suffix}.csv", index=False)


def fig_protection_treatments(ranges, metric: WidthMetric) -> None:
    """fig08 ..._all_treatments: every protection treatment run, per hazard and asset.

    Each asset is compared on the countries common to all of its treatments, so a
    treatment run for fewer countries narrows that asset's set for all of them.
    """
    from .plot_paper_inputs import TREATMENT_COLOR, TREATMENTS     # it imports this module

    scenarios = {
        "River flood": ["flood_absprot_ds", "flood_absprot_ds_rp25to200", "flood_baseline_ds_sc0p5to2p0",
                        "flood_baseline_ds_sc0p75to1p33", "flood_noprot_ds"],
        "Coastal flood": ["coastal_absprot_ds_slr", "coastal_absprot_ds_slr_rp25to200",
                          "coastal_baseline_ds_slr_sc0p5to2p0", "coastal_baseline_ds_slr_sc0p75to1p33",
                          "coastal_noprot_ds_slr"],
        "Windstorm": ["windstorm_absprot", "windstorm", "windstorm_unprotected"],
    }
    treatment_of = {s: t for t, (_, ss) in TREATMENTS.items() for s in ss}
    order = list(TREATMENTS)
    offsets = dict(zip(order, np.linspace(-0.32, 0.32, len(order))))  # same slot in every panel
    rows = []
    positions = y_positions()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH * 0.9, 4.2), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    for ax, (hazard, scens) in zip(axes, scenarios.items()):
        sub = ranges[ranges["scenario"].isin(scens) & (ranges[metric.base] > 0)]
        for asset in ASSETS:
            mine = sub[sub["asset"] == asset]
            if mine.empty:
                continue
            common = set.intersection(*(set(g["country"]) for _, g in mine.groupby("scenario")))
            mine = mine[mine["country"].isin(common)]
            med = mine.groupby("scenario")[metric.column].median()
            y = positions[asset]
            ax.plot([med.min(), med.max()], [y, y], color=BASE, lw=1.0, zorder=1)
            for scenario, value in med.items():
                t = treatment_of[scenario]
                marker = TREATMENTS[t][0]
                ax.scatter(value, y + offsets[t], s=22, marker=marker, color=TREATMENT_COLOR[t],
                           edgecolors=INK, linewidths=0.35, zorder=3)
                rows.append({"hazard": hazard, "asset": asset, "scenario": scenario, "treatment": t,
                             "n_countries": len(common), metric.column: value})
        if metric.log:
            ax.set_xscale("log")
            log_ticks(ax.xaxis, subs=(1.0, 3.0))
            ax.margins(x=0.14)
        else:
            ax.set_xlim(0, 8)
            ax.set_xticks([0, 2, 4, 6, 8])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(hazard)
        asset_axis(ax, positions, labels=ax is axes[0])
    fig.supxlabel(metric.label, fontsize=8, color=INK_2, y=-0.02)
    names = {"Recorded / fixed RP50": "As recorded (flood) / fixed RP 50 (wind)"}
    handles = [Line2D([], [], ls="", marker=TREATMENTS[t][0], ms=5, color=TREATMENT_COLOR[t],
                      markeredgecolor=INK, markeredgewidth=0.35, label=names.get(t, t))
               for t in order if t in {r["treatment"] for r in rows}]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=3,
               handletextpad=0.3, columnspacing=1.2)
    name = f"fig08_protection_effect{metric.suffix}_all_treatments"
    save(fig, name)
    pd.DataFrame(rows).to_csv(DATA / f"{name}.csv", index=False)


def fig_cascade() -> None:
    """fig09: each modelling stage's share of the final pan-European p5-p95 width."""
    tables, rows, negative = {}, [], []
    for hazard in ("River flood", "Earthquake", "Windstorm"):
        scenario = HAZARDS[hazard]
        table = {}
        for asset in ASSETS:
            path = CASCADE_DIR / f"{asset}_{scenario}_cascade.parquet"
            if not path.exists():
                continue
            df = pq.read_table(path, columns=["scope", "step", "factor_added", "total_EAD_MEUR"],
                               filters=[("scope", "==", "eu_sum")]).to_pandas()
            grouped = df.groupby(["step", "factor_added"])["total_EAD_MEUR"]
            stats = grouped.quantile([0.05, 0.95]).unstack().reset_index().sort_values("step")
            stats["width"] = stats[0.95] - stats[0.05]
            stats["gain"] = stats["width"].diff()
            stats = stats[stats["factor_added"] != ""]
            stats["driver"] = stats["factor_added"].map(category)
            final = stats["width"].iloc[-1]
            gains = stats.groupby("driver")["gain"].sum()
            negative.append({"hazard": hazard, "asset": asset,
                             "negative_share": float(-gains.clip(upper=0).sum() / final)})
            gains = gains.clip(lower=0)
            table[asset] = gains / gains.sum()
        tables[hazard] = pd.DataFrame(table).T.reindex(index=ASSETS, columns=DRIVERS).fillna(0.0)
        rows.append(tables[hazard].assign(hazard=hazard).reset_index())
    stacked_panels(tables, "fig09_pan_european_cascade",
                   xlabel="Share of the pan-European p5–p95 range (%)")
    pd.concat(rows).to_csv(DATA / "fig09_pan_european_cascade.csv", index=False)
    worst = max(negative, key=lambda item: item["negative_share"])
    print(f"  cascade: largest narrowing step {worst}")


def main() -> None:
    set_style()
    for folder in (OUT, DATA, SUPP):
        folder.mkdir(parents=True, exist_ok=True)
    ranges = load_ranges()
    sobol = load_sobol()
    for metric in (BY_MEDIAN, BY_MEAN):
        fig_width(ranges, metric)
        fig_size_vs_width(ranges, metric)
        fig_size_vs_width(ranges, metric, shared=True)
        fig_protection_effect(ranges, metric)
    if RANGES_TREATMENTS_CSV.exists():
        treatments = load_ranges(RANGES_TREATMENTS_CSV)
        for metric in (BY_MEDIAN, BY_MEAN):
            fig_protection_treatments(treatments, metric)
    else:
        print(f"  skipped fig08 ..._all_treatments: no {RANGES_TREATMENTS_CSV}")
    fig_drivers(sobol, ranges)
    fig_top_driver(sobol, ranges)
    fig_infra_vs_country(sobol, ranges)
    fig_interactions(sobol)
    fig_cascade()


if __name__ == "__main__":
    main()
