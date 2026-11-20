"""Paper figures: does the span of an input decide whether it comes out as a main driver?

y is the share of countries (with any damage) where the driver is the top driver, or
among the top three, by total-order Sobol index (ST) summed over its factors (driver
categories as in plot_drivers.category). x is the span of that input:

    curves      number of vulnerability curves to choose from          (drivers: curves)
    curves_per_subtype  mean curves per subtype: object types that share one candidate
                curve set form a subtype, and the model draws one curve per subtype,
                so 10 curves over 5 subtypes is 2 per draw, over 2 subtypes 5 (fig10b/11b)
    cost        relative cost span, (max - min) / mean                  (drivers: cost)
    protection  span of the sampled protection range, max / min         (drivers: protection)
each as top1 (top driver) and top3 (among the top three).

Four designs, one subfolder each under paper/driver_vs_input/, so they can be compared:
    a_hazard_colour_treatment_symbol  colour = hazard, symbol = protection treatment,
                                      infrastructure named next to its highest dot
    b_infra_letter_codes              as a, with a two-letter infrastructure code
                                      inside every marker
    c_infra_symbols                   symbol = infrastructure type; colour = protection
                                      treatment (by-hazard panels) or hazard (combined)
    c_alt_zeros_collapsed             as c, with the treatment colours of TREATMENT_COLOR_ALT,
                                      and an infrastructure type that is 0 under every
                                      treatment of a hazard drawn once, with a thick outline
    d_per_hazard_by_treatment         one figure per hazard, one panel per protection
                                      treatment, dots coloured by sector and named
a-c come as *_by_hazard.png and *_combined.png; d has no protection figure, because
within one treatment panel the protection span is a single value.

Sources: convergence/bootstrap_factors.csv (point ST per factor, each combination at
the largest N it was run at) and paper/data/input_spans.csv (src.paper_input_spans).
Numbers in paper/data/fig10_15_driver_vs_input.csv.

    python -m src.plot_paper_inputs
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from .paths import PROJECT_ROOT
from .plot_drivers import category
from .plot_paper_infra import (DATA, FULL_WIDTH, GRID, INK, INK_2, OUT, SECTOR_COLOR, SECTOR_OF, _place_labels,
                               save, set_style)

FACTORS = PROJECT_ROOT / "convergence" / "bootstrap_factors.csv"
SPANS = DATA / "input_spans.csv"
ROOT_OUT = OUT / "driver_vs_input"
DESIGNS = {
    "a": ROOT_OUT / "a_hazard_colour_treatment_symbol",
    "b": ROOT_OUT / "b_infra_letter_codes",
    "c": ROOT_OUT / "c_infra_symbols",
    "c_alt": ROOT_OUT / "c_alt_zeros_collapsed",
    "d": ROOT_OUT / "d_per_hazard_by_treatment",
}

# Validated for all pairs (dataviz validator, light surface): worst CVD dE 9.1, normal
# vision >= 15. Aqua and yellow sit below 3:1 on white, so markers carry a dark rim.
HAZARD_COLOR = {"River flood": "#1baf7a", "Coastal flood": "#2a78d6",
                "Earthquake": "#d95926", "Windstorm": "#eda100"}
HAZARD_KEY = {"River flood": "river", "Coastal flood": "coastal",
              "Earthquake": "earthquake", "Windstorm": "windstorm"}

# Protection treatment -> (marker, scenarios). Coastal is the sea-level-rise set throughout.
TREATMENTS = {
    "Sampled, RP 5–200": ("o", ["flood_absprot_ds", "coastal_absprot_ds_slr"]),
    "Sampled, RP 25–200": ("s", ["flood_absprot_ds_rp25to200", "coastal_absprot_ds_slr_rp25to200",
                                 "windstorm_absprot"]),
    "Recorded × 0.5–2": ("^", ["flood_baseline_ds_sc0p5to2p0", "coastal_baseline_ds_slr_sc0p5to2p0"]),
    "Recorded × 0.75–1.33": ("v", ["flood_baseline_ds_sc0p75to1p33", "coastal_baseline_ds_slr_sc0p75to1p33"]),
    "Recorded / fixed RP50": ("D", ["flood_noprot_ds", "coastal_noprot_ds_slr", "windstorm"]),
    "No protection": ("P", ["earthquake", "windstorm_unprotected"]),
}
# Design c: treatments on one blue ramp (reference palette steps 250-700), darker = wider
# protection range; "no protection" is off that scale, so neutral grey.
TREATMENT_COLOR = {"Sampled, RP 5–200": "#0d366b", "Sampled, RP 25–200": "#1c5cab",
                   "Recorded × 0.5–2": "#2a78d6", "Recorded × 0.75–1.33": "#5598e7",
                   "Recorded / fixed RP50": "#86b6ef", "No protection": "#a09c8e"}
# Design c_alt: one distinct hue per treatment (user's choice), grey for no protection.
TREATMENT_COLOR_ALT = dict(zip(list(TREATMENTS)[:5], ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]))
TREATMENT_COLOR_ALT["No protection"] = "#a09c8e"
ZERO_EDGE = 1.7          # outline of a collapsed all-zero symbol
# max / min of each sampled protection range
PROTECTION_SPAN = {"Sampled, RP 5–200": 40.0, "Sampled, RP 25–200": 8.0,
                   "Recorded × 0.5–2": 4.0, "Recorded × 0.75–1.33": 16 / 9}
SCEN_TREATMENT = {s: t for t, (_, ss) in TREATMENTS.items() for s in ss}

ASSET_NAME = {"roads": "Roads", "rail": "Rail", "airports": "Airports", "ports": "Ports",
              "power": "Power", "gas": "Gas", "oil": "Oil", "telecom": "Telecom",
              "education": "Education", "healthcare": "Healthcare"}
ASSET_CODE = {"roads": "Rd", "rail": "Rl", "airports": "Ai", "ports": "Po", "power": "Pw",
              "gas": "Gs", "oil": "Ol", "telecom": "Tc", "education": "Ed", "healthcare": "Hc"}
# Design c: one symbol per infrastructure type, round-ish for transport, angular for
# utilities, many-sided for social.
ASSET_MARKER = {"roads": "o", "rail": "s", "airports": "^", "ports": "v",
                "power": "D", "gas": "P", "oil": "X", "telecom": "*",
                "education": "h", "healthcare": "p"}
MARKER_SIZE = {"*": 70, "P": 40, "X": 40}

METRICS = {   # stem: (driver, x column, ticks, x label)
    "curves": ("Vulnerability curve", "n_curves", [0, 5, 10, 15, 20, 25, 30],
               "Vulnerability curves to choose from"),
    "curves_per_subtype": ("Vulnerability curve", "curves_per_subtype", [0, 5, 10, 15, 20, 25, 30],
                           "Average number of curves per subtype (subtype = object types sharing one curve set)"),
    "cost": ("Cost / max-damage", "cost_range", [0, 1, 2, 3, 4, 5], "Cost span, (max − min) / mean"),
    "protection": ("Protection standard", "protection_span", [0, 10, 20, 30, 40],
                   "Span of the protection range (max / min)"),
}
Y_LABEL = {"top1": "Countries where it is the top driver (%)",
           "top3": "Countries where it is among the top three drivers (%)"}
FIG_NO = {("curves", "top1"): 10, ("curves", "top3"): 11, ("cost", "top1"): 12,
          ("curves_per_subtype", "top1"): "10b", ("curves_per_subtype", "top3"): "11b",
          ("cost", "top3"): 13, ("protection", "top1"): 14, ("protection", "top3"): 15}


def hazard_of(scenario: str) -> str:
    if scenario.startswith("flood_"):
        return "River flood"
    if scenario.startswith("coastal_"):
        return "Coastal flood"
    if scenario.startswith("earthquake"):
        return "Earthquake"
    return "Windstorm"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def driver_table() -> pd.DataFrame:
    """Per asset x scenario: countries, and the share where each driver is #1 / in the top 3."""
    f = pd.read_csv(FACTORS, usecols=["country", "asset", "scenario", "n", "index", "factor_code", "point"])
    f = f[(f["index"] == "ST") & f.scenario.isin(SCEN_TREATMENT)]
    top_n = f.groupby(["country", "asset", "scenario"]).n.transform("max")
    f = f[f.n == top_n]
    f["driver"] = f.factor_code.map(category)
    f["st"] = f.point.clip(lower=0)
    d = f.groupby(["country", "asset", "scenario", "driver"], as_index=False).st.sum()
    d = d[d.groupby(["country", "asset", "scenario"]).st.transform("sum") > 0]
    d["rank"] = d.groupby(["country", "asset", "scenario"]).st.rank(ascending=False, method="first")
    n = d.groupby(["asset", "scenario"]).country.nunique().rename("countries")
    rows = []
    for drv in ("Vulnerability curve", "Cost / max-damage", "Protection standard"):
        x = d[d.driver == drv]
        t = pd.DataFrame({"top1": x[x["rank"] == 1].groupby(["asset", "scenario"]).size(),
                          "top3": x[x["rank"] <= 3].groupby(["asset", "scenario"]).size(),
                          "present": x.groupby(["asset", "scenario"]).size()})
        t = t.reindex(n.index).fillna(0).join(n)
        t["driver"] = drv
        rows.append(t.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["share_top1"] = out.top1 / out.countries
    out["share_top3"] = out.top3 / out.countries
    out["hazard"] = out.scenario.map(hazard_of)
    out["treatment"] = out.scenario.map(SCEN_TREATMENT)
    return out


CURVE_GROUP_ATTR = {"river": "flood_groups", "coastal": "flood_groups", "earthquake": "eq_groups",
                    "windstorm": "wind_groups"}


def curve_counts() -> pd.DataFrame:
    """Per asset x hazard: distinct curves, subtypes (curve groups: object types sharing one
    candidate curve set, one curve draw each) and the mean number of curves per subtype.
    W7.2, the all-zero wind curve, is not counted, and a group of only W7.2 is no subtype."""
    from .curves import ASSET_CONFIGS, get_asset_config
    rows = []
    for a in ASSET_CONFIGS:
        acfg = get_asset_config(a)
        for h, attr in CURVE_GROUP_ATTR.items():
            groups = [[c for c in g if c != "W7.2"] for g in getattr(acfg, attr).values()]
            groups = [g for g in groups if g]
            rows.append({"asset": a, "hazard": h, "n_curves": len({c for g in groups for c in g}),
                         "n_subtypes": len(groups),
                         "curves_per_subtype": float(np.mean([len(g) for g in groups])) if groups else np.nan})
    return pd.DataFrame(rows)


def quick_spans() -> pd.DataFrame:
    """Stand-in while src.paper_input_spans has not run: exact curve counts, but the
    cost span as the plain median over object types instead of value-weighted."""
    from .curves import ASSET_CONFIGS, MAXDAM_RAW, get_asset_config
    groups = {"river": "flood_groups", "coastal": "flood_groups", "earthquake": "eq_groups",
              "windstorm": "wind_groups"}
    rows = []
    for a in ASSET_CONFIGS:
        acfg = get_asset_config(a)
        rel = [(v[2] - v[0]) / v[1] for v in MAXDAM_RAW.get(a, {}).values() if v[1] > 0]
        for h, attr in groups.items():
            ids = {c for g in getattr(acfg, attr).values() for c in g} - {"W7.2"}
            rows.append({"asset": a, "hazard": h, "n_curves": len(ids),
                         "cost_range": float(np.median(rel)) if rel else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Marks: one style function per design, returning how a row is drawn
# ---------------------------------------------------------------------------

def style_a(r, panel_is_hazard):
    return dict(marker=TREATMENTS[r.treatment][0], face=HAZARD_COLOR[r.hazard], edge=INK, lw=0.4,
                s=30, code=None)


def style_b(r, panel_is_hazard):
    c = HAZARD_COLOR[r.hazard]
    return dict(marker=TREATMENTS[r.treatment][0], face=to_rgba(c, 0.35), edge=c, lw=0.9,
                s=170, code=ASSET_CODE[r.asset])


def style_c(r, panel_is_hazard):
    m = ASSET_MARKER[r.asset]
    face = TREATMENT_COLOR[r.treatment] if panel_is_hazard else HAZARD_COLOR[r.hazard]
    return dict(marker=m, face=face, edge=INK, lw=0.4, s=MARKER_SIZE.get(m, 30), code=None)


def style_c_alt(r, panel_is_hazard):
    st = style_c(r, panel_is_hazard)
    if panel_is_hazard:
        st["face"] = TREATMENT_COLOR_ALT[r.treatment]
    return st


STYLES = {"a": style_a, "b": style_b, "c": style_c, "c_alt": style_c_alt}


def x_axis(ax, ticks, xmax, fmt="{:g}"):
    """Linear x from 0, ending a little past the last tick or the largest value."""
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, max(ticks[-1], xmax) * 1.06)


def y_axis(ax, show_labels=True):
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelleft=show_labels, length=0, colors=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def scatter(ax, df, xcol, ycol, style, panel_is_hazard, spread=0.009, collapse=False):
    """Same-x dots of one asset are spread sideways by treatment so none hides another.
    collapse: an asset whose dots of one hazard all sit at 0 at the same x is drawn once
    there (also when the hazard has a single treatment), white in hazard panels (it stands for every treatment) or in
    the hazard colour otherwise, with a thick outline."""
    order = list(TREATMENTS)
    lo, hi = ax.get_xlim()
    if collapse:
        keys = [df.asset, df.hazard, df[xcol].round(6)]
        zero = df.groupby(keys)[ycol].transform(lambda v: (v == 0).all()).astype(bool)
        for (asset, hazard, x), _ in df[zero].groupby(keys):
            m = ASSET_MARKER[asset]
            face = "white" if panel_is_hazard else HAZARD_COLOR[hazard]
            ax.scatter(x, 0, s=MARKER_SIZE.get(m, 30) * 1.25, marker=m, facecolors=[face], edgecolors=[INK],
                       linewidths=ZERO_EDGE, zorder=3)
        df = df[~zero]
    for _, r in df.iterrows():
        st = style(r, panel_is_hazard)
        dx = spread * (hi - lo) * (order.index(r.treatment) - 2.5)
        x, y = r[xcol] + dx, 100 * r[ycol]
        ax.scatter(x, y, s=st["s"], marker=st["marker"], facecolors=[st["face"]], edgecolors=[st["edge"]],
                   linewidths=st["lw"], zorder=3)
        if st["code"]:
            ax.text(x, y, st["code"], ha="center", va="center", fontsize=5.2, color=INK, zorder=4,
                    fontweight="bold")


def legend(fig, design, df, panel_is_hazard, y, collapsed=False):
    hazards = [h for h in HAZARD_COLOR if h in set(df.hazard)]
    treatments = [t for t in TREATMENTS if t in set(df.treatment)]
    assets = [a for a in ASSET_MARKER if a in set(df.asset)]
    grey = "#d9d8d3"
    rows = 4

    def block(handles):
        """Pad a group to whole legend columns, so groups never share a column."""
        pad = (-len(handles)) % rows
        return handles + [Line2D([], [], ls="", marker="", label="")] * pad
    if design in ("c", "c_alt"):
        colours = TREATMENT_COLOR_ALT if design == "c_alt" else TREATMENT_COLOR
        h_inf = [Line2D([], [], ls="", marker=ASSET_MARKER[a], ms=6.5 if ASSET_MARKER[a] != "*" else 9,
                        mfc=grey, mec=INK, mew=0.4, label=ASSET_NAME[a]) for a in assets]
        if panel_is_hazard:
            h_col = [Line2D([], [], ls="", marker="o", ms=6, mfc=colours[t], mec=INK, mew=0.4, label=t)
                     for t in treatments]
        else:
            h_col = [Line2D([], [], ls="", marker="o", ms=6, mfc=HAZARD_COLOR[h], mec=INK, mew=0.4, label=h)
                     for h in hazards]
        if collapsed:
            h_col.append(Line2D([], [], ls="", marker="o", ms=6.5, mfc="white" if panel_is_hazard else grey,
                                mec=INK, mew=ZERO_EDGE, label="0 under every treatment"))
        handles = block(h_col) + block(h_inf)
    else:
        h_haz = [Line2D([], [], ls="", marker="o", ms=6, mfc=HAZARD_COLOR[h], mec=INK, mew=0.4, label=h)
                 for h in hazards]
        h_tr = [Line2D([], [], ls="", marker=TREATMENTS[t][0], ms=6, mfc=grey, mec=INK, mew=0.4, label=t)
                for t in treatments]
        handles = block(h_haz) + block(h_tr)
        if design == "b":
            handles += block([Line2D([], [], ls="", marker=f"${ASSET_CODE[a]}$", ms=11, mfc=INK, mec=INK,
                                     mew=0, label=ASSET_NAME[a]) for a in assets])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles) // rows,
               handletextpad=0.3, columnspacing=1.1, fontsize=7.2)


def label_assets(ax, df, xcol, ycol):
    """Name each infrastructure type once, at its highest dot in the panel."""
    top = df.sort_values(ycol).groupby("asset").tail(1)
    _place_labels(ax, [(r[xcol], 100 * r[ycol], ASSET_NAME[r.asset]) for _, r in top.iterrows()])


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

def by_hazard(df, metric, tag, design, hazards):
    _, xcol, ticks, xlabel = METRICS[metric]
    ycol = f"share_{tag}"
    ncols, nrows = 2, int(np.ceil(len(hazards) / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 2.75 * nrows), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.3})
    axes = np.atleast_1d(axes).ravel()
    for ax, h in zip(axes, hazards):
        sub = df[df.hazard == h]
        x_axis(ax, ticks, df[xcol].max())
        y_axis(ax, show_labels=ax in axes[::ncols])
        scatter(ax, sub, xcol, ycol, STYLES[design], panel_is_hazard=True,
                collapse=design == "c_alt" and metric != "protection")
        ax.set_title(h)
        if design == "a":
            label_assets(ax, sub, xcol, ycol)
    for ax in axes[len(hazards):]:
        ax.set_visible(False)
    fig.supxlabel(xlabel, fontsize=8, color=INK_2, y=0.01 if nrows > 1 else -0.04)
    fig.supylabel(Y_LABEL[tag], fontsize=8, color=INK_2, x=0.04)
    legend(fig, design, df[df.hazard.isin(hazards)], True, y=0.97 if nrows > 1 else 1.0,
           collapsed=design == "c_alt" and metric != "protection")
    save(fig, f"fig{FIG_NO[(metric, tag)]}_{metric}_{tag}_by_hazard", DESIGNS[design])


def combined(df, metric, tag, design):
    _, xcol, ticks, xlabel = METRICS[metric]
    ycol = f"share_{tag}"
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.66, 3.3))
    x_axis(ax, ticks, df[xcol].max())
    y_axis(ax)
    scatter(ax, df, xcol, ycol, STYLES[design], panel_is_hazard=False,
            collapse=design == "c_alt" and metric != "protection")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(Y_LABEL[tag])
    legend(fig, design, df, False, y=0.92, collapsed=design == "c_alt" and metric != "protection")
    save(fig, f"fig{FIG_NO[(metric, tag)]}_{metric}_{tag}_combined", DESIGNS[design])


def treatment_title(treatment: str, hazard: str) -> str:
    """Panel heading: the fixed treatment is the recorded standard for floods, RP50 for wind."""
    if treatment == "Recorded / fixed RP50":
        return "Fixed RP 50" if hazard == "Windstorm" else "As recorded"
    return treatment


def spread_ties(x: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    """Nudge dots that land on the same spot sideways, centred on the true x."""
    x = x.astype(float).copy()
    key = pd.Series(list(zip(np.round(x, 6), np.round(y / 4.0))))     # same x, within ~4 points
    for _, idx in key.groupby(key).groups.items():
        if len(idx) > 1:
            x[list(idx)] += width * (np.arange(len(idx)) - (len(idx) - 1) / 2)
    return x


def per_hazard_by_treatment(df, metric, tag, hazard):
    """Design d: one figure for one hazard, one panel per protection treatment. Each
    infrastructure type keeps one symbol, filled in its sector's colour, in every figure."""
    _, xcol, ticks, xlabel = METRICS[metric]
    ycol = f"share_{tag}"
    sub = df[df.hazard == hazard]
    treatments = [t for t in TREATMENTS if t in set(sub.treatment)]
    ncols = min(3, len(treatments))
    nrows = int(np.ceil(len(treatments) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH if ncols > 1 else FULL_WIDTH * 0.5,
                                                    2.6 * nrows), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.35}, squeeze=False)
    axes = axes.ravel()
    for ax, t in zip(axes, treatments):
        s = sub[sub.treatment == t]
        x_axis(ax, ticks, df[xcol].max())
        y_axis(ax, show_labels=ax in axes[::ncols])
        lo, hi = ax.get_xlim()
        xs = spread_ties(s[xcol].to_numpy(), (100 * s[ycol]).to_numpy(), 0.025 * (hi - lo))
        for xv, (_, r) in zip(xs, s.iterrows()):
            m = ASSET_MARKER[r.asset]
            ax.scatter(xv, 100 * r[ycol], s=MARKER_SIZE.get(m, 34) * 1.15, marker=m,
                       color=SECTOR_COLOR[SECTOR_OF[r.asset]], edgecolors=INK, linewidths=0.4, zorder=3)
        ax.set_title(treatment_title(t, hazard), fontsize=8.5)
    for ax in axes[len(treatments):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:len(treatments)]):
        if i + ncols >= len(treatments):        # nothing below it: show its x labels
            ax.tick_params(axis="x", labelbottom=True)
    fig.supxlabel(xlabel, fontsize=8, color=INK_2, y=0.01 if nrows > 1 else -0.05)
    fig.supylabel(Y_LABEL[tag], fontsize=8, color=INK_2, x=0.04 if ncols > 1 else -0.02)
    top = 1.0 if nrows == 1 else 0.95
    # One column per sector, so the colour groups read as sectors.
    by_sector = {}
    for a in ASSET_MARKER:
        by_sector.setdefault(SECTOR_OF[a], []).append(a)
    rows = max(len(v) for v in by_sector.values())
    handles = []
    for sector, assets in by_sector.items():
        col = [Line2D([], [], ls="", marker=ASSET_MARKER[a], ms=9 if ASSET_MARKER[a] == "*" else 6.5,
                      mfc=SECTOR_COLOR[sector], mec=INK, mew=0.4, label=ASSET_NAME[a]) for a in assets]
        handles += col + [Line2D([], [], ls="", marker="", label="")] * (rows - len(col))
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, top), ncol=len(by_sector),
                     handletextpad=0.3, columnspacing=1.6)
    fig.canvas.draw()
    leg_top = leg.get_window_extent().y1 / fig.bbox.height     # the title sits just above the legend
    fig.text(0.5, leg_top + 0.01, hazard, ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    key = HAZARD_KEY[hazard]
    save(fig, f"fig{FIG_NO[(metric, tag)]}_{metric}_{tag}_{key}", DESIGNS["d"])


def main() -> None:
    set_style()
    tab = driver_table()
    if SPANS.exists():
        spans = pd.read_csv(SPANS)
    else:
        print("NOTE: paper/data/input_spans.csv not found - cost span is the unweighted median over "
              "object types (run python -m src.paper_input_spans for the value-weighted one)")
        spans = quick_spans()
    spans["hazard"] = spans.hazard.map({v: k for k, v in HAZARD_KEY.items()})
    tab = tab.merge(spans[["asset", "hazard", "n_curves", "cost_range"]], on=["asset", "hazard"], how="left")
    counts = curve_counts()
    counts["hazard"] = counts.hazard.map({v: k for k, v in HAZARD_KEY.items()})
    tab = tab.merge(counts[["asset", "hazard", "n_subtypes", "curves_per_subtype"]], on=["asset", "hazard"],
                    how="left")
    tab["protection_span"] = tab.treatment.map(PROTECTION_SPAN)
    tab.to_csv(DATA / "fig10_15_driver_vs_input.csv", index=False)

    for metric, (driver, xcol, _, _) in METRICS.items():
        data = tab[(tab.driver == driver) & tab[xcol].notna()]
        if metric.startswith("curves"):
            data = data[data.n_curves > 0]
        hazards = list(HAZARD_COLOR)
        if metric == "protection":
            hazards = ["River flood", "Coastal flood"]
            data = data[data.hazard.isin(hazards)]
        for tag in ("top1", "top3"):
            for design in ("a", "b", "c", "c_alt"):
                by_hazard(data, metric, tag, design, hazards)
                combined(data, metric, tag, design)
            if metric != "protection":
                for h in hazards:
                    per_hazard_by_treatment(data, metric, tag, h)


if __name__ == "__main__":
    main()
