"""Top driver, uncertainty width and driver pies for two scenario sets -> paper/scenario_sets/.

    default          one scenario per hazard: river and coastal (SLR) with the recorded
                     protection standard x 0.5-2, earthquake, windstorm at the fixed
                     RP 50 design standard
    all_treatments   every protection treatment that has been run or is planned: river
                     and coastal x {sampled RP 5-200, sampled RP 25-200, recorded x 0.5-2,
                     recorded x 0.75-1.33, as recorded}, earthquake, windstorm x {sampled
                     RP 25-200, fixed RP 50, no design standard}

Figures in each folder:

    fig04_top_driver_by_country   country x infrastructure, colour = the driver with the
                                  largest summed total-order index (ST), as paper fig04
    heatmap_w90_median            (p95 - p5) / median of EAD, the portrait heatmap of
                                  src/plot_heatmap.py on the spectral ramp
    heatmap_w90_mean              the same band divided by the mean, on the same classes
    pies_top_driver               per scenario, how many country x infrastructure pairs
                                  each parameter leads, as src/plot_drivers.py's overview
    share_of_effect_by_asset[_<hazard>]
                                  one stacked bar per infrastructure type: the mean over
                                  countries of each driver's share of summed ST, after the
                                  power-network study's by-country figure. default: one
                                  figure, all hazards; all_treatments: one per hazard

"Planned, not run yet" is a combination that runs under another scenario of the same
hazard but not yet under this one: the river x 0.5-2 and x 0.75-1.33 runs and the
unprotected windstorm runs exist for line and point assets only, and the polygon
types follow after the DamageScanner polygon fix.

Sources: convergence/bootstrap_factors.csv (point ST per factor, each combination at
the largest N it was run at, newest archive) and paper/data/ead_quantiles.parquet
(src/paper_ead_samples.py). The summary workbook has no November scenarios, so it is
not read here.

    python -m src.plot_paper_scenarios
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from . import plot_drivers as pdv
from . import plot_heatmap as ph
from . import plot_maps as pm
from .ema_model import SCENARIO_HAZARD
from .paths import PROJECT_ROOT
from .plot_drivers import COLORS, category
from .plot_paper_infra import (ASSET_LABEL, ASSETS, DRIVER_LABEL, DRIVERS, FULL_WIDTH, GRID, INK_2, OUT, save,
                               set_style)
from .plot_paper_ranges import QUANTILES, legend_above

FOLDER = OUT / "scenario_sets"
FACTORS = PROJECT_ROOT / "convergence" / "bootstrap_factors.csv"

DEFAULT = ["flood_baseline_ds_sc0p5to2p0", "coastal_baseline_ds_slr_sc0p5to2p0", "earthquake_pga0p9to1p1", "windstorm"]
ALL = ["flood_absprot_ds", "flood_absprot_ds_rp25to200", "flood_baseline_ds_sc0p5to2p0",
       "flood_baseline_ds_sc0p75to1p33", "flood_noprot_ds",
       "coastal_absprot_ds_slr", "coastal_absprot_ds_slr_rp25to200", "coastal_baseline_ds_slr_sc0p5to2p0",
       "coastal_baseline_ds_slr_sc0p75to1p33", "coastal_noprot_ds_slr",
       "earthquake_pga0p9to1p1",
       "windstorm_absprot", "windstorm", "windstorm_unprotected"]
SETS = {"default": DEFAULT, "all_treatments": ALL}

HAZARD_NAME = {"flood": "River flood", "coastal": "Coastal flood", "earthquake": "Earthquake",
               "windstorm": "Windstorm"}
TREATMENT = {
    "flood_absprot_ds": "sampled, RP 5–200", "flood_absprot_ds_rp25to200": "sampled, RP 25–200",
    "flood_baseline_ds_sc0p5to2p0": "recorded × 0.5–2", "flood_baseline_ds_sc0p75to1p33": "recorded × 0.75–1.33",
    "flood_noprot_ds": "as recorded",
    "coastal_absprot_ds_slr": "sampled, RP 5–200", "coastal_absprot_ds_slr_rp25to200": "sampled, RP 25–200",
    "coastal_baseline_ds_slr_sc0p5to2p0": "recorded × 0.5–2",
    "coastal_baseline_ds_slr_sc0p75to1p33": "recorded × 0.75–1.33", "coastal_noprot_ds_slr": "as recorded",
    "earthquake": "", "earthquake_pga0p9to1p1": "",
    "windstorm_absprot": "sampled, RP 25–200", "windstorm": "fixed, RP 50",
    "windstorm_unprotected": "no design standard",
}
NO_DAMAGE_FILL = "#f0efec"
NOT_RUN_EDGE = "#b9b8b0"
NOT_RUN_LABEL = "Planned, not run yet"


def hazard_of(scen: str) -> str:
    return HAZARD_NAME[scen.split("_")[0]]


def title(scen: str, sep: str = "\n") -> str:
    t = TREATMENT[scen]
    return hazard_of(scen) + (f"{sep}{t}" if t else "")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_quantiles() -> pd.DataFrame:
    q = pd.read_parquet(QUANTILES, columns=["country", "asset", "scenario", "mean", "zero_share", "n_draws",
                                            "q0050", "q0250", "q0500", "q0750", "q0950", "q1000"])
    return q[q.scenario.isin(ALL)]


def planned(q: pd.DataFrame) -> pd.DataFrame:
    """(country, asset, scenario) for every scenario, wherever the pair runs under any
    scenario of the same hazard; `run` says whether this scenario has run."""
    q = q.assign(hazard=q.scenario.map(hazard_of))
    pairs = q[["country", "asset", "hazard"]].drop_duplicates()
    rows = [pairs[pairs.hazard == hazard_of(s)].assign(scenario=s) for s in ALL]
    out = pd.concat(rows, ignore_index=True)
    run = set(map(tuple, q[["country", "asset", "scenario"]].to_numpy()))
    out["run"] = [t in run for t in map(tuple, out[["country", "asset", "scenario"]].to_numpy())]
    return out


def load_factors() -> pd.DataFrame:
    """Point ST per factor: newest archive at each N, then the largest N."""
    f = pd.read_csv(FACTORS, usecols=["country", "asset", "scenario", "n", "archive", "index",
                                      "factor_code", "point"])
    f = f[(f["index"] == "ST") & f.scenario.isin(ALL)]
    key = ["country", "asset", "scenario"]
    newest = f.groupby(key + ["n"]).archive.transform("max")
    f = f[f.archive == newest]
    f = f[f.n == f.groupby(key).n.transform("max")]
    return f.assign(driver=f.factor_code.map(category), st=f.point.clip(lower=0))


def top_drivers(f: pd.DataFrame) -> pd.DataFrame:
    """fig04's rule: the driver with the largest summed ST; pairs with no ST left out."""
    key = ["country", "asset", "scenario"]
    d = f.groupby(key + ["driver"], as_index=False).st.sum()
    d["share"] = d.st / d.groupby(key).st.transform("sum")
    d = d[d.share.notna()]
    return d.sort_values("share", ascending=False).drop_duplicates(key)


# ---------------------------------------------------------------------------
# fig04: top driver per country x infrastructure
# ---------------------------------------------------------------------------

def top_driver_panel(ax, scen, tops, plan, countries, show_y):
    codes = {d: i for i, d in enumerate(DRIVERS)}
    cmap = ListedColormap([COLORS[d] for d in DRIVERS])
    sub = tops[tops.scenario == scen]
    grid = (sub.assign(code=sub.driver.map(codes)).pivot(index="country", columns="asset", values="code")
            .reindex(index=countries, columns=ASSETS))
    ax.set_facecolor(NO_DAMAGE_FILL)
    ax.pcolormesh(np.arange(len(ASSETS) + 1), np.arange(len(countries) + 1),
                  np.ma.masked_invalid(grid.to_numpy(dtype=float)), cmap=cmap, vmin=-0.5,
                  vmax=len(DRIVERS) - 0.5, edgecolors="white", linewidth=0.5)
    waiting = plan[(plan.scenario == scen) & ~plan.run]
    for _, r in waiting.iterrows():
        x, y = ASSETS.index(r.asset), countries.index(r.country)
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="white", edgecolor=NOT_RUN_EDGE, hatch="////",
                               linewidth=0, zorder=3))
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="none", edgecolor="white", linewidth=0.5, zorder=4))
    ax.set_xticks(np.arange(len(ASSETS)) + 0.5)
    ax.set_xticklabels([ASSET_LABEL[a] for a in ASSETS], rotation=90, fontsize=7)
    ax.tick_params(axis="x", length=0, colors="#0b0b0b")
    ax.set_yticks(np.arange(len(countries)) + 0.5)
    ax.set_yticklabels(countries, fontsize=6.5)
    ax.tick_params(axis="y", labelleft=show_y)
    ax.set_ylim(len(countries), 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title(scen), fontsize=8.5)
    return len(waiting)


def fig_top_driver(tops, plan, scenarios, folder):
    countries = sorted(plan.country.unique())
    if len(scenarios) <= 4:
        rows = [scenarios]
    else:
        rows = [[s for s in scenarios if hazard_of(s) == h] for h in ("River flood", "Coastal flood")]
        rows.append([s for s in scenarios if hazard_of(s) in ("Earthquake", "Windstorm")])
    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(FULL_WIDTH * ncols / 4, 6.2 * len(rows)),
                             squeeze=False, gridspec_kw={"wspace": 0.06, "hspace": 0.32})
    waiting = 0
    for i, row in enumerate(rows):
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(row):
                ax.set_visible(False)
                continue
            waiting += top_driver_panel(ax, row[j], tops, plan, countries, show_y=j == 0)
    used = [d for d in DRIVERS if d in set(tops[tops.scenario.isin(scenarios)].driver)]
    handles = [Patch(facecolor=COLORS[d], label=DRIVER_LABEL[d]) for d in used]
    handles.append(Patch(facecolor=NO_DAMAGE_FILL, label="No damage or no asset"))
    if waiting:
        handles.append(Patch(facecolor="white", edgecolor=NOT_RUN_EDGE, hatch="////", label=NOT_RUN_LABEL))
    legend_above(fig, handles, ncol=4 if len(scenarios) <= 4 else len(handles), gap=0.004,
                 handlelength=1.0, handleheight=1.0, columnspacing=1.2)
    save(fig, "fig04_top_driver_by_country", folder)
    tops[tops.scenario.isin(scenarios)][["scenario", "country", "asset", "driver", "share"]].to_csv(
        folder / "fig04_top_driver_by_country.csv", index=False)


# ---------------------------------------------------------------------------
# Heatmaps of the p5-p95 width
# ---------------------------------------------------------------------------

def heatmap_ranges(q: pd.DataFrame) -> pd.DataFrame:
    """The EAD_Ranges.csv columns plot_heatmap reads, from the quantile cache."""
    return pd.DataFrame({
        "country": q.country, "asset": q.asset, "scenario": q.scenario,
        "p5": q.q0050, "p25": q.q0250, "median": q.q0500, "p75": q.q0750, "p95": q.q0950,
        "max": q.q1000, "mean": q["mean"], "zero_fraction": q.zero_share, "n": q.n_draws,
    })


def fig_heatmaps(q, plan, scenarios, folder):
    ranges = heatmap_ranges(q[q.scenario.isin(scenarios)])
    ph.PORTRAIT, ph.PLAIN, ph.TEXT_SCALE, ph.CELL_SCALE = True, True, 1.5, 1.0
    ph.BAND_ASSETS = {s: sorted(plan[plan.scenario == s].asset.unique()) for s in scenarios}
    ph.scenario_sort_key = lambda s: (ALL.index(s), "")
    ph.MISSING_LABEL = "not run (planned, or no coast)" if any(
        hazard_of(s) == "Coastal flood" for s in scenarios) else "not run (planned)"
    pm.use_palette("spectral")
    for measure, tag, what in (("relwidth", "median", "median"), ("relwidth_mean", "mean", "mean")):
        pm.use_measure(measure)
        ph.plot_spread_heatmap(ranges, folder / f"heatmap_w90_{tag}.png", group_by="hazard",
                               title=f"(p95 − p5) / {what} of EAD")
        print(f"  wrote heatmap_w90_{tag}.png")
    pm.use_measure("ratio")
    ranges.assign(w90_median=(ranges.p95 - ranges.p5) / ranges["median"].where(ranges["median"] > 0),
                  w90_mean=(ranges.p95 - ranges.p5) / ranges["mean"].where(ranges["mean"] > 0)).to_csv(
        folder / "heatmap_w90.csv", index=False)


# ---------------------------------------------------------------------------
# Pies
# ---------------------------------------------------------------------------

def pie_table(f, plan, scenarios) -> pd.DataFrame:
    """plot_drivers.coverage_for_rank's output: one row per planned combination, with the
    category of its top FACTOR (plot_drivers' rule), a no-variance state, or not run."""
    key = ["country", "asset", "scenario"]
    top = f.sort_values("point", ascending=False).drop_duplicates(key)
    top = top[top.st > 0][key + ["driver"]]
    m = plan[plan.scenario.isin(scenarios)].merge(top, on=key, how="left")
    exposure = pdv.exposure_flags()
    m["hazard"] = m.scenario.map(SCENARIO_HAZARD)
    if exposure is not None:
        m = m.merge(exposure, on=["country", "asset", "hazard"], how="left")
    else:
        m["exposed_qty_max"] = np.nan
    no_var = m.run & m.driver.isna()
    known = m.exposed_qty_max.notna()
    m.loc[no_var & known & (m.exposed_qty_max > 0), "driver"] = pdv.EXPOSED_UNDAMAGED
    m.loc[no_var & known & (m.exposed_qty_max.fillna(0) <= 0), "driver"] = pdv.NO_EXPOSURE
    m.loc[no_var & ~known, "driver"] = pdv.NO_DAMAGE
    m["covered"] = m.run
    m["dcat"] = m.driver
    return m


def fig_pies(f, plan, scenarios, folder):
    pdv.REPORT, pdv.SURFACE = True, "#ffffff"
    pdv.REPORT_LEGEND_LABEL.setdefault("Unfinished", NOT_RUN_LABEL)
    pdv.REPORT_TITLE.update({s: title(s, sep=" (") + (")" if TREATMENT[s] else "") for s in scenarios})
    pdv.is_bracket_variant = lambda s: False          # REPORT_TITLE carries the bracket itself
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": "#ffffff",
                         "savefig.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
                         "text.color": pdv.INK, "savefig.bbox": None}):
        m = pie_table(f, plan, scenarios)
        out = pdv.overview_figure(m, scenarios, 1, folder, ncols=4 if len(scenarios) == 4 else 5)
    target = folder / "pies_top_driver.png"
    target.unlink(missing_ok=True)
    out.rename(target)
    print("  wrote pies_top_driver.png")
    counts = m.assign(state=np.where(m.run, m.dcat, "Planned, not run yet")).groupby(
        ["scenario", "state"]).size().rename("pairs").reset_index()
    counts.to_csv(folder / "pies_top_driver.csv", index=False)


# ---------------------------------------------------------------------------
# Share of total effect per infrastructure type
# ---------------------------------------------------------------------------

SHARE_DRIVERS = DRIVERS + ["Other"]
SHARE_LABEL = DRIVER_LABEL | {"Other": "Other"}
HAZARD_FILE = {"River flood": "river_flood", "Coastal flood": "coastal_flood", "Earthquake": "earthquake",
               "Windstorm": "windstorm"}


def effect_shares(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (scenario, asset): the mean over countries of each country's ST share per
    driver (%), each country weighted equally, and the mean raw ST sum. Countries whose
    ST is zero for every factor (no damage) are left out of the mean."""
    key = ["country", "asset", "scenario"]
    d = f.groupby(key + ["driver"], as_index=False).st.sum()
    d["total"] = d.groupby(key).st.transform("sum")
    d = d[d.total > 0].assign(share=lambda x: 100 * x.st / x.total)
    per_country = d.pivot_table(index=key, columns="driver", values="share", fill_value=0.0)
    shares = per_country.groupby(level=["scenario", "asset"]).mean()
    shares = shares.reindex(columns=[c for c in SHARE_DRIVERS if c in shares.columns])
    stats = d.drop_duplicates(key).groupby(["scenario", "asset"]).agg(
        st_sum=("total", "mean"), countries=("country", "size"))
    return shares, stats


def share_panel(ax, scen, shares, stats, plan, show_y, heading):
    xs = np.arange(len(ASSETS))
    states = {}
    bottom = np.zeros(len(ASSETS))
    rows = shares.reindex([(scen, a) for a in ASSETS])
    for drv in shares.columns:
        vals = rows[drv].fillna(0).to_numpy()
        ax.bar(xs, vals, 0.78, bottom=bottom, color=COLORS[drv], edgecolor="white", linewidth=0.5, zorder=2)
        bottom += vals
    sub = plan[plan.scenario == scen]
    for x, a in zip(xs, ASSETS):
        pa = sub[sub.asset == a]
        if (scen, a) in stats.index:
            ax.text(x, 101.5, f"{stats.loc[(scen, a), 'st_sum']:.1f}", ha="center", va="bottom",
                    fontsize=5.5, color=INK_2)
        elif len(pa) and not pa.run.any():
            states["planned"] = True
            ax.add_patch(Rectangle((x - 0.39, 0), 0.78, 100, facecolor="white", edgecolor=NOT_RUN_EDGE,
                                   hatch="////", linewidth=0, zorder=2))
        elif len(pa):
            states["no damage"] = True
            ax.add_patch(Rectangle((x - 0.39, 0), 0.78, 100, facecolor=NO_DAMAGE_FILL, linewidth=0, zorder=2))
        else:
            ax.text(x, 50, "not modelled", rotation=90, ha="center", va="center", fontsize=6, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([ASSET_LABEL[a] for a in ASSETS], rotation=90, fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(ASSETS) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=7, labelleft=show_y)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    t = title(scen, sep=", ") if heading is None else TREATMENT[scen]
    ax.set_title(t[:1].upper() + t[1:], fontsize=8)
    return states


def fig_share(shares, stats, plan, scenarios, folder, name, heading=None):
    """`heading`: one hazard for the whole figure, set above the legend; the panel
    titles then carry the treatment alone."""
    ncols = 1 if len(scenarios) == 1 else (2 if len(scenarios) == 4 else 3)
    nrows = -(-len(scenarios) // ncols)
    width = FULL_WIDTH * (0.42 if ncols == 1 else 1.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.55 * nrows), squeeze=False, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.75})
    states = {}
    for i, ax in enumerate(axes.flat):
        if i >= len(scenarios):
            ax.set_visible(False)
            continue
        states |= share_panel(ax, scenarios[i], shares, stats, plan, show_y=i % ncols == 0, heading=heading)
    fig.supylabel("Share of total effect (%)", fontsize=8, x=0.06 if ncols > 1 else -0.02)
    used = [d for d in shares.columns if (shares.loc[shares.index.get_level_values(0).isin(scenarios), d] > 0).any()]
    handles = [Patch(facecolor=COLORS[d], label=SHARE_LABEL[d]) for d in used]
    if "no damage" in states:
        handles.append(Patch(facecolor=NO_DAMAGE_FILL, label="No damage in any country"))
    if "planned" in states:
        handles.append(Patch(facecolor="white", edgecolor=NOT_RUN_EDGE, hatch="////", label=NOT_RUN_LABEL))
    lg = legend_above(fig, handles, ncol=min(len(handles), 4 if ncols > 1 else 2), gap=0.01,
                      handlelength=1.0, handleheight=1.0, fontsize=7)
    if heading:
        fig.canvas.draw()
        y = lg.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height
        fig.text(0.5, y + 0.004, heading, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(r).y0 for ax in fig.axes if ax.get_visible()) / fig.bbox.height
    fig.text(0.5, bottom - 0.012,
             "Bars: mean over countries of each country's ST share (countries weighted equally). "
             "Number above bar: mean ΣST over countries\n(above 1 = variance from interactions).",
             ha="center", va="top", fontsize=6, color=INK_2, wrap=True)
    save(fig, name, folder)
    keep = shares.index.get_level_values(0).isin(scenarios)
    shares[keep].join(stats).reset_index().to_csv(folder / f"{name}.csv", index=False)


def main() -> None:
    q = load_quantiles()
    plan = planned(q)
    f = load_factors()
    tops = top_drivers(f)
    shares, stats = effect_shares(f)
    for name, scenarios in SETS.items():
        folder = FOLDER / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {len(scenarios)} scenarios")
        set_style()
        fig_top_driver(tops, plan, scenarios, folder)
        fig_heatmaps(q, plan, scenarios, folder)
        fig_pies(f, plan, scenarios, folder)
        if name == "default":
            fig_share(shares, stats, plan, scenarios, folder, "share_of_effect_by_asset")
        else:
            for hazard, tag in HAZARD_FILE.items():
                fig_share(shares, stats, plan, [s for s in scenarios if hazard_of(s) == hazard], folder,
                          f"share_of_effect_by_asset_{tag}", heading=hazard)
        waiting = plan[plan.scenario.isin(scenarios) & ~plan.run].groupby("scenario").size()
        for s, n in waiting.items():
            print(f"  {s}: {n} planned, not run yet")


if __name__ == "__main__":
    main()
