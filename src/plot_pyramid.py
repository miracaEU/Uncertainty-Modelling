"""Trickle-down / pyramid figures + summary tables from the cascade ensembles.

Reads whatever src/cascade.py has written to results/cascade/ and draws, for
every (scope, asset, scenario), how the plausible range of total EAD widens as
each uncertainty factor is switched on.

Each cascade step shows, on one row:
    - the p5-p95 band          (light outer funnel)
    - the p25-p75 band         (darker inner funnel)
    - the MEDIAN               (solid vertical tick)
    - the MEAN                 (open diamond)
Median and mean are drawn for every step, intermediate ones included, and are
distinguished by SHAPE rather than colour so the figure survives greyscale
printing and colour-vision deficiency. A dashed reference line marks the
step-0 nominal value, so drift of the centre is as visible as the widening.

Output layout under overview_figures/pyramids/{order}/:
    pan_european/{asset}_{scenario}_pyramid.png          scope "eu_sum"
    pan_european_native/{asset}_{scenario}_pyramid.png   scope "eu_native"
    countries/{ISO3}/{asset}_{scenario}_pyramid.png      scope "country"
    pyramid_summary.xlsx / .csv                          the numbers

{order} mirrors the cascade folder ("workflow-a1b2c3d4"), so figures built from
different factor orderings cannot overwrite each other - the cascade is
order-dependent, and two orderings give different intermediate steps for the
same asset/scenario/country. It is dropped when --cascade-dir points at a plain
folder of parquets (no cascade_order.txt), and when --out-dir is given.

Scope folders are derived from the scope column, so a cascade run tagged with a
new --scope-label (e.g. a genuinely pan-European run built on a single European
exposure set rather than a sum of national totals) lands in its own folder with
no code change here.

Runs on numpy/pandas/matplotlib/openpyxl alone - no scipy, no ema_workbench -
so the figures can be rebuilt anywhere the parquet files are readable.

Usage:
    python -m src.plot_pyramid
    python -m src.plot_pyramid --assets power --scenarios flood_noprot_ds
    python -m src.plot_pyramid --scopes eu_sum          # skip the ~2000 country figures
    python -m src.plot_pyramid --countries DEU FRA --log-x
    python -m src.plot_pyramid --cascade-dir results/cascade/workflow-a1b2c3d4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- palette (matches src/analyze.py so the whole study looks like one set) ---
# Pure white, so a figure dropped into the report has no visible panel edge
# against the page. SURFACE doubles as the colour of anything meant to read
# as a hole in the page - marker faces, cell edges, text halos - so it has
# to stay in step with figure.facecolor wherever it is used.
SURFACE = "#ffffff"

# --report switches these three. The cascade stores EAD in millions of euros;
# the report wants billions, spelled out, and a pure white page rather than the
# off-white the screen figures use.
REPORT = False
# The twin axis carrying p5-p95 width as a % of the median. On by default; the
# report figures drop it with --no-width-axis, where the same numbers are in the
# summary CSV and a second tick column only crowds the page.
WIDTH_AXIS = True
VALUE_SCALE = 1.0
VALUE_LABEL = "total EAD (M EUR / yr)"
REPORT_SURFACE = "#ffffff"
REPORT_VALUE_SCALE = 1.0 / 1000.0
REPORT_VALUE_LABEL = "Expected annual damage (Billion Euros per year)"


def use_report_style() -> None:
    """Pure white page, EAD in billions, black furniture, larger type."""
    global REPORT, SURFACE, VALUE_SCALE, VALUE_LABEL
    REPORT = True
    SURFACE = REPORT_SURFACE
    VALUE_SCALE = REPORT_VALUE_SCALE
    VALUE_LABEL = REPORT_VALUE_LABEL
    # rcParams were set at import from the old SURFACE, so rebinding the name is
    # not enough - the page would stay off-white.
    #
    # The greys below go black. On screen a grey tick label reads as secondary
    # to the data, which is what it is; printed, or scaled down into a report
    # column, it reads as faint. The GRID stays pale deliberately - it is the
    # one piece of furniture that sits BEHIND the bands, and a black grid would
    # compete with them rather than support them.
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
    })


def furniture(colour: str) -> str:
    """A non-data line's colour: black on a report page, its grey on screen."""
    return INK if REPORT else colour


def legend_size() -> int:
    """Legend type, matched to the rest of the report set."""
    return 12 if REPORT else 9
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND_90 = "#cde2fb"
BAND_50 = "#86b6ef"
ACCENT = "#1c5cab"

# --- scenario vocabulary, shared by every figure module ----------------------
# One definition so a scenario reads the same everywhere. The study's internal
# name "_noprot" is misleading in a figure: it does NOT mean protection is zero,
# it means the protection standard is held at the value FLOPROS/COASTPROS
# actually records, instead of being sampled. The labels below say that plainly.
# "_ds" twins use a multiplicative depth error rather than an additive offset;
# they are the study defaults, so they carry the plain label and the additive
# variants are the ones marked.
SCENARIO_LABEL = {
    "flood_absprot_ds": "River flood - protection sampled, RP 5-200",
    "flood_absprot": "River flood - protection sampled (add. depth err.)",
    "flood_noprot_ds": "River flood - protection as recorded (FLOPROS)",
    "flood_noprot": "River flood - protection as recorded (add. depth err.)",
    "flood_baseline_ds": "River flood - protection scaled, x0-2",
    "flood_baseline": "River flood - protection scaled (add. depth err.)",
    "coastal_absprot_ds": "Coastal flood - protection sampled, RP 5-200",
    "coastal_absprot": "Coastal flood - protection sampled (add. depth err.)",
    "coastal_noprot_ds": "Coastal flood - protection as recorded (COASTPROS)",
    "coastal_noprot": "Coastal flood - protection as recorded (add. depth err.)",
    "coastal_baseline_ds": "Coastal flood - protection scaled, x0-2",
    "coastal_baseline": "Coastal flood - protection scaled (add. depth err.)",
    "earthquake": "Earthquake",
    "windstorm": "Windstorm - design standard RP50",
    "windstorm_absprot": "Windstorm - design standard sampled, RP 25-200",
    # Protection-range tests: same model, different sampling bracket for the
    # protection factor. The bracket is in the label because it is the variable
    # under test - without it two pies in one figure look like duplicates.
    "flood_absprot_ds_rp10to100": "River flood - protection sampled, RP 10-100",
    "coastal_absprot_ds_rp10to100": "Coastal flood - protection sampled, RP 10-100",
    "flood_baseline_ds_sc0p5to1p5": "River flood - protection scaled, x0.5-1.5",
    "coastal_baseline_ds_sc0p5to1p5": "Coastal flood - protection scaled, x0.5-1.5",
    "coastal_absprot_ds_slr": "Coastal flood - protection sampled + sea-level rise",
    "coastal_noprot_ds_slr": "Coastal flood - protection as recorded + sea-level rise",
    # Intensity-range test: the PGA bias narrowed from x0.8-1.2.
    "earthquake_pga0p9to1p1": "Earthquake - PGA scaled, x0.9-1.1",
}

# Short forms for tight spots (pie titles, panel headings), where the
# qualifier is wrapped onto a second line.
SCENARIO_LABEL_SHORT = {
    "flood_absprot_ds": "River flood (protection sampled, RP 5-200)",
    "flood_absprot": "River flood (protection sampled)",
    "flood_noprot_ds": "River flood (protection as recorded)",
    "flood_noprot": "River flood (protection as recorded)",
    "flood_baseline_ds": "River flood (protection scaled, x0-2)",
    "flood_baseline": "River flood (protection scaled)",
    "coastal_absprot_ds": "Coastal flood (protection sampled, RP 5-200)",
    "coastal_absprot": "Coastal flood (protection sampled)",
    "coastal_noprot_ds": "Coastal flood (protection as recorded)",
    "coastal_noprot": "Coastal flood (protection as recorded)",
    "coastal_baseline_ds": "Coastal flood (protection scaled, x0-2)",
    "coastal_baseline": "Coastal flood (protection scaled)",
    "earthquake": "Earthquake",
    "windstorm": "Windstorm (design RP50)",
    "windstorm_absprot": "Windstorm (design sampled, RP 25-200)",
    "flood_absprot_ds_rp10to100": "River flood (protection sampled, RP 10-100)",
    "coastal_absprot_ds_rp10to100": "Coastal flood (protection sampled, RP 10-100)",
    "flood_baseline_ds_sc0p5to1p5": "River flood (protection scaled, x0.5-1.5)",
    "coastal_baseline_ds_sc0p5to1p5": "Coastal flood (protection scaled, x0.5-1.5)",
    "coastal_absprot_ds_slr": "Coastal flood (protection sampled + SLR)",
    "coastal_noprot_ds_slr": "Coastal flood (protection as recorded + SLR)",
    "earthquake_pga0p9to1p1": "Earthquake (PGA x0.9-1.1)",
}

# Reading order for every figure that shows more than one scenario: both river
# flood variants, then both coastal, then the single-hazard ones. Alphabetical
# order put coastal before river and split the pairs apart.
# Protection treatments run from most to least freedom: the standard replaced
# by a wide sampled bracket, then a narrower one, then the recorded standard
# merely perturbed, then held fixed. Reading left to right the protection factor
# is progressively constrained, which is the story these figures tell.
SCENARIO_ORDER = [
    "flood_absprot", "flood_absprot_ds", "flood_absprot_ds_rp10to100",
    "flood_baseline", "flood_baseline_ds", "flood_baseline_ds_sc0p5to1p5",
    "flood_noprot", "flood_noprot_ds",
    "coastal_absprot", "coastal_absprot_ds", "coastal_absprot_ds_rp10to100",
    "coastal_baseline", "coastal_baseline_ds", "coastal_baseline_ds_sc0p5to1p5",
    "coastal_noprot", "coastal_noprot_ds",
    "coastal_absprot_ds_slr", "coastal_noprot_ds_slr",
    "earthquake", "earthquake_pga0p9to1p1",
    "windstorm", "windstorm_absprot",
]


def scenario_label(scen: str, short: bool = False) -> str:
    """Human-readable scenario name; falls back to the raw code if unmapped."""
    table = SCENARIO_LABEL_SHORT if short else SCENARIO_LABEL
    return table.get(scen, scen)


def scenario_sort_key(scen: str) -> tuple[int, str]:
    """Sort key placing scenarios in SCENARIO_ORDER, unknowns last by name."""
    try:
        return (SCENARIO_ORDER.index(scen), "")
    except ValueError:
        return (len(SCENARIO_ORDER), scen)


COUNTRY_SCOPE = "country"
SCOPE_DIRS = {"eu_sum": "pan_european", "eu_native": "pan_european_native"}

# Marker file src/cascade.py drops in each order-keyed folder. Duplicated here
# rather than imported: importing cascade would pull in ema_workbench and scipy,
# which this module deliberately does without (see the header).
ORDER_FILE = "cascade_order.txt"

PERCENTILES = (5, 25, 50, 75, 95)

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
        "font.size": 10,
    }
)

_PRETTY = {
    "cost_level": "cost level (min-mean-max)",
    "aggregation": "aggregation method",
    "warming": "warming level",
    "depth_scale": "water depth scale",
    "depth_offset": "water depth offset",
    "pga_scale": "PGA scale",
    "gust_scale": "gust speed scale",
    "protection_scale": "protection multiplier",
    "protection_abs_rp": "protection RP (absolute)",
}


def readable(factor: str) -> str:  # noqa: D401
    """Human label for a factor code, without importing the heavy stack.

    The fully-resolved curve descriptions (which object types sit behind
    curve_F1_1 for THIS asset) live in the workbook's Curve_Groups sheet; here
    the curve id alone keeps the axis narrow enough to read.
    """
    if factor == MERGED_LABEL:
        return factor
    if factor in _PRETTY:
        return _PRETTY[factor]
    if factor.startswith("curve_"):
        return "curve " + factor[len("curve_"):].replace("_", ".")
    return factor.replace("_", " ")


def scope_dir(scope: str, country: str) -> Path:
    if scope == COUNTRY_SCOPE:
        return Path("countries") / country
    return Path(SCOPE_DIRS.get(scope, scope))


def resolve_cascade_dir(cdir: Path) -> Path:
    """Accept either a folder of parquets or the parent holding order folders.

    src/cascade.py writes into results/cascade/{order_mode}-{hash}/ so that runs
    under different factor orderings can never be pooled. Point --cascade-dir at
    either level: a parent with exactly one order folder resolves to it, and an
    ambiguous parent lists the choices rather than silently picking one - which
    would mix incomparable cascades into a single summary table.
    """
    if sorted(cdir.glob("*_cascade.parquet")):
        return cdir
    subs = sorted(
        d for d in cdir.glob("*") if d.is_dir() and any(d.glob("*_cascade.parquet"))
    )
    if len(subs) == 1:
        print(f"Reading cascade folder {subs[0].name}")
        return subs[0]
    if len(subs) > 1:
        listing = "\n".join(f"  --cascade-dir {d}" for d in subs)
        raise SystemExit(
            f"{cdir} holds cascades for {len(subs)} different factor orderings."
            f"\nEach folder's {ORDER_FILE} says which; pick one explicitly:"
            f"\n{listing}"
        )
    return cdir


def default_results_dir() -> Path:
    """results/ from config.yml when the full stack is installed, else the
    conventional location.

    Kept optional so this module really does run on
    numpy/pandas/matplotlib/openpyxl alone: src/paths.py needs pyyaml, which a
    plotting-only environment (or a laptop pointed at a copy of the parquet
    files) has no reason to carry.
    """
    try:
        from .paths import load_config

        return load_config()["results_dir"]
    except Exception:
        return PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


# Set from --merge-curves. Curve groups partition an asset by object sub-type -
# for roads, F7_4 is motorway/trunk/primary and F7_8 is secondary/tertiary/
# residential - so freeing them one at a time reads as a sequence of asset
# classes rather than as a single "which curve is right" unknown. Merging keeps
# the step where every curve group is free and drops the partial ones, so the
# cascade shows one vulnerability-curve level.
MERGE_CURVES = False
MERGED_LABEL = "vulnerability curves (all)"


def merge_curve_steps(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive curve_ steps into the last of them.

    The curve groups are adjacent in WORKFLOW_ORDER, so the last one carries the
    state with every group free - exactly the aggregate the merged level should
    show. The derived quantities are recomputed afterwards, because d_w90 must
    now measure the jump from the step before the first curve group.
    """
    is_curve = rows["factor_added"].astype(str).str.startswith(CURVE_STEP_PREFIX)
    if is_curve.sum() < 2:
        return rows
    # Keep a curve row only if the next row is not also a curve row.
    keep = ~(is_curve & is_curve.shift(-1, fill_value=False))
    out = rows[keep].copy()
    out.loc[out["factor_added"].astype(str).str.startswith(CURVE_STEP_PREFIX),
            "factor_added"] = MERGED_LABEL
    out["d_w90"] = out["w90"].diff().fillna(0.0)
    final = out["w90"].iloc[-1]
    out["share_w90"] = out["d_w90"] / final if final > 0 else np.nan
    out["drift_median"] = out["median"] - out["median"].iloc[0]
    return out.reset_index(drop=True)


CURVE_STEP_PREFIX = "curve_"


def step_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-step distribution summary for one (scope, country, asset, scenario).

    Adds the derived quantities the pyramid is actually about: how much each
    newly-freed factor widened the band (d_w90) and what share of the final
    width it accounts for.
    """
    g = df.groupby(["step", "factor_added"], as_index=False)["total_EAD_MEUR"]
    rows = g.agg(
        n="size", mean="mean", median="median", std="std", min="min", max="max"
    )
    q = (
        df.groupby("step")["total_EAD_MEUR"]
        .quantile([p / 100 for p in PERCENTILES])
        .unstack()
    )
    q.columns = [f"p{p}" for p in PERCENTILES]
    rows = rows.merge(q, left_on="step", right_index=True).sort_values("step")

    rows["w90"] = rows["p95"] - rows["p5"]
    rows["w50"] = rows["p75"] - rows["p25"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["w90_rel"] = np.where(rows["median"] > 0, rows["w90"] / rows["median"], np.nan)
        rows["mean_median_ratio"] = np.where(
            rows["median"] > 0, rows["mean"] / rows["median"], np.nan
        )
    rows["d_w90"] = rows["w90"].diff().fillna(0.0)
    final_w90 = rows["w90"].iloc[-1]
    rows["share_w90"] = rows["d_w90"] / final_w90 if final_w90 > 0 else np.nan
    rows["drift_median"] = rows["median"] - rows["median"].iloc[0]
    if MERGE_CURVES:
        rows = merge_curve_steps(rows)
    return rows


def dependence_comparison(df_all: pd.DataFrame, eu_scope: str, seed: int = 7) -> dict | None:
    """The three ways to build a European total, from the SAME final-step draws.

    crn         - sum at equal draw index. The shared epistemic factors carried
                  the same draw in every country, so this is the real joint
                  distribution rather than an assumed one.
    comonotonic - sum of per-country p95 (and p5). This is EXACTLY the answer
                  under maximal dependence: quantiles are additive if and only
                  if the terms are comonotonic, so percentile-summing silently
                  assumes every country hits its 95th percentile simultaneously.
    independent - each country's draws permuted independently before summing,
                  i.e. assuming the national errors diversify away. The relative
                  band then collapses like 1/sqrt(m_eff).

    The spread between them measures how much of the European uncertainty is
    irreducible-because-shared versus how much genuinely diversifies. Returns
    None when the country-level rows needed for the comparison are absent.
    """
    eu = df_all[df_all["scope"] == eu_scope]
    cty = df_all[df_all["scope"] == COUNTRY_SCOPE]
    if eu.empty or cty.empty:
        return None

    last = int(eu["step"].max())
    eu_v = eu.loc[eu["step"] == last, "total_EAD_MEUR"].to_numpy()
    wide = (
        cty[cty["step"] == last]
        .pivot_table(index="draw", columns="country", values="total_EAD_MEUR")
        .to_numpy()
    )
    if wide.size == 0:
        return None

    crn_p5, crn_p95 = np.percentile(eu_v, [5, 95])
    como_p5 = float(np.percentile(wide, 5, axis=0).sum())
    como_p95 = float(np.percentile(wide, 95, axis=0).sum())

    rng = np.random.default_rng(seed)
    shuffled = np.column_stack([rng.permutation(wide[:, j]) for j in range(wide.shape[1])])
    ind_p5, ind_p95 = np.percentile(shuffled.sum(axis=1), [5, 95])

    mu = wide.mean(axis=0)
    total_mu = float(mu.sum())
    # Effective number of contributing countries (inverse Herfindahl). A total
    # dominated by a handful of large countries has far fewer effective terms
    # than the country count, which limits how much anything can diversify.
    m_eff = float(total_mu ** 2 / np.square(mu).sum()) if np.square(mu).sum() > 0 else np.nan
    top5 = float(np.sort(mu)[::-1][:5].sum() / total_mu) if total_mu > 0 else np.nan

    crn_w = float(crn_p95 - crn_p5)
    return {
        "n_countries": int(wide.shape[1]),
        "crn_p5": float(crn_p5), "crn_p95": float(crn_p95), "crn_w90": crn_w,
        "comonotonic_p5": como_p5, "comonotonic_p95": como_p95,
        "comonotonic_w90": como_p95 - como_p5,
        "independent_p5": float(ind_p5), "independent_p95": float(ind_p95),
        "independent_w90": float(ind_p95 - ind_p5),
        "comonotonic_over_crn": (como_p95 - como_p5) / crn_w if crn_w > 0 else np.nan,
        "independent_over_crn": float(ind_p95 - ind_p5) / crn_w if crn_w > 0 else np.nan,
        "m_eff_countries": m_eff,
        "top5_country_share": top5,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def draw_axes(ax, stats: pd.DataFrame, title: str, log_x: bool,
              xlim: tuple[float, float] | None = None, width_axis: bool = True) -> None:
    """One pyramid on a given axes. Shared by the single and paired figures.

    xlim forces a common value axis, which is what makes two panels comparable;
    without it matplotlib scales each to its own data and the wider scenario
    merely looks like the narrower one. It is given in DISPLAY units, i.e.
    already multiplied by VALUE_SCALE.
    """
    n_steps = len(stats)
    y = np.arange(n_steps)
    # The cascade stores EAD in millions; --report divides to billions. Scaled
    # here rather than in step_stats so the summary tables keep their own unit
    # and only the drawing changes.
    p5, p25, p50, p75, p95 = (stats[c].to_numpy() * VALUE_SCALE
                              for c in ("p5", "p25", "p50", "p75", "p95"))
    mean = stats["mean"].to_numpy() * VALUE_SCALE

    # Funnel silhouette: this is what makes it read as a pyramid rather than as
    # a stack of unrelated bars.
    ax.fill_betweenx(y, p5, p95, color=BAND_90, alpha=0.75, linewidth=0, zorder=1)
    ax.fill_betweenx(y, p25, p75, color=BAND_50, alpha=0.85, linewidth=0, zorder=2)

    for i in y:
        ax.hlines(i, p5[i], p95[i], color=ACCENT, linewidth=1.0, alpha=0.75, zorder=3)
        ax.hlines(i, p25[i], p75[i], color=ACCENT, linewidth=3.4, alpha=0.95, zorder=4)
        ax.plot([p50[i]], [i], marker="|", markersize=15, markeredgewidth=2.4,
                color=INK, zorder=6)
        ax.plot([mean[i]], [i], marker="D", markersize=5.6, markerfacecolor=SURFACE,
                markeredgecolor=INK, markeredgewidth=1.3, linestyle="none", zorder=6)

    # Nominal reference, so a drifting centre is as legible as a widening band.
    ax.axvline(p50[0], color=furniture(BASELINE), linestyle="--", linewidth=1.0,
               zorder=0)

    labels = [
        "nominal (all factors fixed)" if s == 0 else f"+ {readable(f)}"
        for s, f in zip(stats["step"], stats["factor_added"])
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(VALUE_LABEL)
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    # In report mode the subtitle is dropped but its space is kept, so the title
    # does not sit hard against the top of the plot.
    ax.set_title(title, pad=30 if REPORT else 16)
    if xlim is not None:
        ax.set_xlim(*xlim)

    # Relative 90% width per step, on a twin axis so it cannot collide with the
    # data no matter how far the band extends.
    if width_axis and WIDTH_AXIS:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y, [
            "-" if not np.isfinite(v) else f"{v:.0%}" for v in stats["w90_rel"]
        ])
        ax2.set_ylabel("p5-p95 width (% of median)", color=MUTED, fontsize=9)
        ax2.tick_params(labelsize=9)
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)


# The forked final step. A second hue for the alternative branch, chosen to sit
# beside the blue trunk rather than compete with it.
FORK_BAND_90 = "#fadfc6"
FORK_BAND_50 = "#f0b073"
FORK_ACCENT = "#b35c0a"


def draw_fork_split(ax, trunk: pd.DataFrame, term_abs: pd.Series, term_mul: pd.Series,
                    label_abs: str, label_mul: str, title: str, log_x: bool) -> None:
    """Chain ending in the multiplicative step, then a detached alternative.

    Scaling the recorded standard by 0.5-1.5x PERTURBS the nominal, so it
    belongs in the chain and is drawn as its continuation. Replacing the
    standard with a draw over RP 5-200 discards the recorded value instead, so
    it is a different scenario rather than a further step, and is drawn below a
    visible break with no funnel connecting it.
    """
    chain = pd.concat([trunk, term_mul.to_frame().T], ignore_index=True)
    n = len(chain)
    y = np.arange(n)
    # Just enough clearance for the break rule: the separation is carried by the
    # rule and the colour, so extra whitespace only pushes the alternative out of
    # the reader's eye.
    y_alt = float(n - 1) + 0.85

    p5, p25, p50, p75, p95 = (chain[c].to_numpy(dtype=float) * VALUE_SCALE
                              for c in ("p5", "p25", "p50", "p75", "p95"))
    mean = chain["mean"].to_numpy(dtype=float) * VALUE_SCALE

    ax.fill_betweenx(y, p5, p95, color=BAND_90, alpha=0.75, linewidth=0, zorder=1)
    ax.fill_betweenx(y, p25, p75, color=BAND_50, alpha=0.85, linewidth=0, zorder=2)
    for i in y:
        ax.hlines(i, p5[i], p95[i], color=ACCENT, linewidth=1.0, alpha=0.75, zorder=3)
        ax.hlines(i, p25[i], p75[i], color=ACCENT, linewidth=3.4, alpha=0.95, zorder=4)
        ax.plot([p50[i]], [i], marker="|", markersize=15, markeredgewidth=2.4,
                color=INK, zorder=6)
        ax.plot([mean[i]], [i], marker="D", markersize=5.6, markerfacecolor=SURFACE,
                markeredgecolor=INK, markeredgewidth=1.3, linestyle="none", zorder=6)

    # The break. Drawn across the full width in axes coordinates so it reads as
    # "the chain stops here", not as anything belonging to the data. Unlabelled:
    # the row label and the change of colour already say it.
    y_break = (n - 1) + 0.42
    ax.axhline(y_break, color=furniture(MUTED), linewidth=1.0,
               linestyle=(0, (6, 4)), zorder=7)

    a5, a25, a50, a75, a95 = (float(term_abs[c]) * VALUE_SCALE
                              for c in ("p5", "p25", "p50", "p75", "p95"))
    amean = float(term_abs["mean"]) * VALUE_SCALE
    ax.barh(y_alt, a95 - a5, left=a5, height=0.60, color=FORK_BAND_90, alpha=0.80,
            linewidth=0, zorder=1)
    ax.barh(y_alt, a75 - a25, left=a25, height=0.60, color=FORK_BAND_50, alpha=0.90,
            linewidth=0, zorder=2)
    ax.hlines(y_alt, a5, a95, color=FORK_ACCENT, linewidth=1.0, alpha=0.85, zorder=3)
    ax.hlines(y_alt, a25, a75, color=FORK_ACCENT, linewidth=3.4, alpha=0.95, zorder=4)
    ax.plot([a50], [y_alt], marker="|", markersize=15, markeredgewidth=2.4,
            color=INK, zorder=6)
    ax.plot([amean], [y_alt], marker="D", markersize=5.6, markerfacecolor=SURFACE,
            markeredgecolor=INK, markeredgewidth=1.3, linestyle="none", zorder=6)

    ax.axvline(p50[0], color=furniture(BASELINE), linestyle="--", linewidth=1.0,
               zorder=0)

    labels = ["nominal (all factors fixed)" if s == 0 else f"+ {readable(f)}"
              for s, f in zip(trunk["step"], trunk["factor_added"])] + [label_mul]
    ax.set_yticks(list(y) + [y_alt], labels + [label_abs])
    ax.set_ylim(y_alt + 0.9, -0.8)
    ax.set_xlabel(VALUE_LABEL)
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=30 if REPORT else 16)

    if not WIDTH_AXIS:
        return
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = ["-" if not np.isfinite(v) else f"{v:.0%}" for v in chain["w90_rel"]]
    ax2.set_yticks(list(y) + [y_alt], ticks + [f"{float(term_abs['w90_rel']):.0%}"])
    ax2.set_ylabel("p5-p95 width (% of median)", color=MUTED, fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)


def draw_fork(ax, trunk: pd.DataFrame, term_a: pd.Series, term_b: pd.Series,
              label_a: str, label_b: str, title: str, log_x: bool) -> None:
    """One cascade whose LAST step forks into two alternatives.

    Valid only because the two scenarios share their trunk bit-for-bit: with
    protection ordered last and the LHS matrix filled column by column from one
    seeded generator, every factor before it occupies the same column in both,
    so steps 0..n-1 are the same draws, not merely the same distribution. The
    caller checks that; here it is assumed.

    The two terminal rows are drawn below a gap and in different hues, because
    they are ALTERNATIVES to one another rather than successive steps - reading
    them as a continuation of the funnel would say the band widens twice.
    """
    n = len(trunk)
    y_trunk = np.arange(n)
    y_final = float(n)

    def cols(s):
        return [s[c] * VALUE_SCALE for c in ("p5", "p25", "p50", "p75", "p95", "mean")]

    p5, p25, p50, p75, p95 = (trunk[c].to_numpy() * VALUE_SCALE
                              for c in ("p5", "p25", "p50", "p75", "p95"))
    mean = trunk["mean"].to_numpy() * VALUE_SCALE

    ax.fill_betweenx(y_trunk, p5, p95, color=BAND_90, alpha=0.75, linewidth=0, zorder=1)
    ax.fill_betweenx(y_trunk, p25, p75, color=BAND_50, alpha=0.85, linewidth=0, zorder=2)
    for i in y_trunk:
        ax.hlines(i, p5[i], p95[i], color=ACCENT, linewidth=1.0, alpha=0.75, zorder=3)
        ax.hlines(i, p25[i], p75[i], color=ACCENT, linewidth=3.4, alpha=0.95, zorder=4)
        ax.plot([p50[i]], [i], marker="|", markersize=15, markeredgewidth=2.4,
                color=INK, zorder=6)
        ax.plot([mean[i]], [i], marker="D", markersize=5.6, markerfacecolor=SURFACE,
                markeredgecolor=INK, markeredgewidth=1.3, linestyle="none", zorder=6)

    # Both treatments on ONE row, nested: the two are alternative versions of the
    # same step, so giving each its own row said the cascade had two more levels
    # than it has. Drawn outer-first with the taller band behind, so whichever
    # reaches further is visible and the other sits inside it.
    outer, inner = term_a, term_b
    o5, o25, o50, o75, o95, omean = cols(outer)
    i5, i25, i50, i75, i95, imean = cols(inner)
    H_OUT, H_IN = 0.62, 0.30
    OFF = 0.0

    ax.barh(y_final, o95 - o5, left=o5, height=H_OUT, color=BAND_90, alpha=0.75,
            linewidth=0, zorder=1)
    ax.barh(y_final, o75 - o25, left=o25, height=H_OUT, color=BAND_50, alpha=0.85,
            linewidth=0, zorder=2)
    ax.barh(y_final, i95 - i5, left=i5, height=H_IN, color=FORK_BAND_90, alpha=0.95,
            linewidth=0, zorder=3)
    ax.barh(y_final, i75 - i25, left=i25, height=H_IN, color=FORK_BAND_50, alpha=0.95,
            linewidth=0, zorder=4)

    # Markers split above and below the centre line so the two sets never collide.
    for s_, acc, dy, half in ((outer, ACCENT, -0.19, H_OUT / 2),
                              (inner, FORK_ACCENT, 0.19, H_IN / 2)):
        v5, v25, v50, v75, v95, vmean = cols(s_)
        yy = y_final + dy + OFF
        ax.hlines(yy, v5, v95, color=acc, linewidth=1.0, alpha=0.85, zorder=5)
        ax.hlines(yy, v25, v75, color=acc, linewidth=3.0, alpha=0.95, zorder=5)
        ax.plot([v50], [yy], marker="|", markersize=11, markeredgewidth=2.4,
                color=INK, zorder=6)
        ax.plot([vmean], [yy], marker="D", markersize=5.6, markerfacecolor=SURFACE,
                markeredgecolor=INK, markeredgewidth=1.3, linestyle="none", zorder=6)

    ax.axvline(p50[0], color=furniture(BASELINE), linestyle="--", linewidth=1.0,
               zorder=0)

    labels = ["nominal (all factors fixed)" if s == 0 else f"+ {readable(f)}"
              for s, f in zip(trunk["step"], trunk["factor_added"])]
    ax.set_yticks(list(y_trunk) + [y_final], labels + ["+ protection standard"])
    ax.set_ylim(y_final + 0.9, -0.8)
    ax.set_xlabel(VALUE_LABEL)
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=30 if REPORT else 16)

    if not WIDTH_AXIS:
        return
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = ["-" if not np.isfinite(v) else f"{v:.0%}" for v in trunk["w90_rel"]]
    # One row, two numbers: outer first, matching the legend's order.
    ticks.append(f"{term_a['w90_rel']:.0%} / {term_b['w90_rel']:.0%}")
    ax2.set_yticks(list(y_trunk) + [y_final], ticks)
    ax2.set_ylabel("p5-p95 width (% of median)", color=MUTED, fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)


def legend_handles_pyramid() -> list:
    return [
        Patch(facecolor=BAND_90, label="p5-p95"),
        Patch(facecolor=BAND_50, label="p25-p75"),
        Line2D([], [], marker="|", color=INK, markersize=13, markeredgewidth=2.4,
               linestyle="none", label="median"),
        Line2D([], [], marker="D", markerfacecolor=SURFACE, markeredgecolor=INK,
               markersize=6, linestyle="none", label="mean"),
    ]


def draw_pyramid(stats: pd.DataFrame, title: str, subtitle: str, out: Path, log_x: bool) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 0.55 * len(stats) + 2.6), layout="constrained")
    draw_axes(ax, stats, title, log_x)
    if not REPORT:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    handles = legend_handles_pyramid()
    # Below the axes, not inside: the bottom-right corner is exactly where the
    # widest band lands, so an in-axes legend would sit on top of the result.
    leg = ax.legend(
        handles=handles, frameon=False, fontsize=legend_size(), ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
    )
    for t in leg.get_texts():
        t.set_color(furniture(INK_2))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------



# Scenario pairs that differ only in how the protection standard is treated:
# sampled over a range, against held at the value the source database records.
# Drawn side by side on a shared value axis, because the whole point is how much
# of the final spread is produced by sampling protection rather than by the
# physical factors - and two panels auto-scaled to their own data would hide
# exactly that.
PROTECTION_PAIRS = [
    ("flood_absprot_ds", "flood_noprot_ds"),
    ("coastal_absprot_ds", "coastal_noprot_ds"),
    ("windstorm_absprot", "windstorm"),
]


def pair_figure(stats_a: pd.DataFrame, stats_b: pd.DataFrame, scen_a: str, scen_b: str,
                asset: str, scope_label: str, n_draws: int, out: Path, log_x: bool) -> None:
    """Two cascades on one shared value axis."""
    # Display units: draw_axes scales the data, so a shared xlim must be scaled
    # too or the panels would be framed in millions while plotted in billions.
    lo = min(stats_a["p5"].min(), stats_b["p5"].min()) * VALUE_SCALE
    hi = max(stats_a["p95"].max(), stats_b["p95"].max()) * VALUE_SCALE
    pad = 0.04 * (hi - lo) if hi > lo else 1.0
    xlim = (max(lo - pad, 0.0) if not log_x else lo / 1.1, hi + pad)

    rows = max(len(stats_a), len(stats_b))
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 0.58 * rows + 3.0), layout="constrained")
    for ax, st, scen in ((axes[0], stats_a, scen_a), (axes[1], stats_b, scen_b)):
        draw_axes(ax, st, scenario_label(scen, short=True), log_x,
                  xlim=xlim, width_axis=True)

    fig.suptitle(f"Protection standard sampled vs held at the recorded value - "
                 f"{asset} - {scope_label}", fontsize=13, fontweight="bold", color=INK)
    # The note rides on the legend rather than being its own artist: a bare
    # fig.text is not laid out at all and landed on the panel titles.
    leg = fig.legend(handles=legend_handles_pyramid(), frameon=False,
                     fontsize=legend_size(),
                     ncol=4, loc="outside lower center",
                     title=f"{n_draws:,} draws per step  |  shared value axis  |  "
                           f"right-hand numbers are p5-p95 width as a share of the median")
    leg.get_title().set(fontsize=legend_size(), color=furniture(MUTED))
    for t in leg.get_texts():
        t.set_color(furniture(INK_2))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_pairs(cdir: Path, odir: Path, assets, scopes, countries, log_x) -> int:
    """One paired figure per (asset, protection pair, scope, country)."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    meta: dict[tuple[str, str], dict] = {}
    for path in sorted(cdir.glob("*_cascade.parquet")):
        df = pd.read_parquet(path)
        a, sc = df["asset"].iloc[0], df["scenario"].iloc[0]
        if assets and a not in assets:
            continue
        frames[(a, sc)] = df
        mp = path.with_name(path.name.replace(".parquet", "_meta.json"))
        meta[(a, sc)] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}

    n = 0
    for a in sorted({k[0] for k in frames}):
        for scen_a, scen_b in PROTECTION_PAIRS:
            da, db = frames.get((a, scen_a)), frames.get((a, scen_b))
            if da is None or db is None:
                continue
            groups_a = {(sc, c) for sc, c in da[["scope", "country"]].drop_duplicates().to_numpy()}
            groups_b = {(sc, c) for sc, c in db[["scope", "country"]].drop_duplicates().to_numpy()}
            for scope, country in sorted(groups_a & groups_b):
                if scopes and scope not in scopes:
                    continue
                if scope == COUNTRY_SCOPE and countries and country not in countries:
                    continue
                sa = step_stats(da[(da["scope"] == scope) & (da["country"] == country)])
                sb = step_stats(db[(db["scope"] == scope) & (db["country"] == country)])
                label = country if scope == COUNTRY_SCOPE else "pan-European"
                nd = int(meta.get((a, scen_a), {}).get("n_draws", da["draw"].nunique()))
                out = (odir / scope_dir(scope, country)
                       / f"{a}_{scen_a}_vs_{scen_b}_pyramid.png")
                pair_figure(sa, sb, scen_a, scen_b, a, label, nd, out, log_x)
                n += 1
    return n


FORK_PAIR = ("flood_absprot_ds", "flood_baseline_ds_sc0p5to1p5")
FORK_LABELS = ("replaced by a draw, RP 5-200 yr",
               "scaled 0.5-1.5x of the recorded standard")
# For the split style the two rows are labelled in place, so they read without
# the legend. Scaling PERTURBS the recorded standard, so it continues the chain;
# replacing DISCARDS it, so it is a scenario of its own.
SPLIT_LABEL_MUL = "+ protection scaled 0.5-1.5x"
SPLIT_LABEL_ABS = "protection sampled, RP 5-200 years"


def build_fork(cdir: Path, odir: Path, scopes, log_x: bool, style: str = "split") -> int:
    """One figure per (asset, scope): shared trunk, then the two protection
    treatments - nested on one row ("nested") or chain-plus-detached ("split")."""
    scen_a, scen_b = FORK_PAIR
    frames = {}
    for path in sorted(cdir.glob("*_cascade.parquet")):
        df = pd.read_parquet(path)
        frames[(df["asset"].iloc[0], df["scenario"].iloc[0])] = df

    n = 0
    for asset in sorted({k[0] for k in frames}):
        da, db = frames.get((asset, scen_a)), frames.get((asset, scen_b))
        if da is None or db is None:
            continue
        for scope, country in sorted({(s, c) for s, c in
                                      da[["scope", "country"]].drop_duplicates().to_numpy()}):
            if scopes and scope not in scopes:
                continue
            sa = da[(da["scope"] == scope) & (da["country"] == country)]
            sb = db[(db["scope"] == scope) & (db["country"] == country)]
            if sb.empty:
                continue

            # The whole figure rests on the two sharing a trunk. Verified on the
            # raw draws, not on the summary statistics: equal percentiles could
            # still come from different samples, and the claim being made is the
            # stronger one.
            last = int(sa["step"].max())
            for step in range(last):
                x = sa[sa["step"] == step].sort_values("draw")["total_EAD_MEUR"].to_numpy()
                y = sb[sb["step"] == step].sort_values("draw")["total_EAD_MEUR"].to_numpy()
                if len(x) != len(y) or not np.array_equal(x, y):
                    raise SystemExit(
                        f"{asset} / {scope}: step {step} differs between {scen_a} "
                        f"and {scen_b}, so they do not share a trunk and cannot be "
                        "drawn as one forked cascade. Re-run both with protection "
                        "ordered last and the same seed."
                    )

            stats_a, stats_b = step_stats(sa), step_stats(sb)
            trunk = stats_a.iloc[:-1].reset_index(drop=True)
            label = country if scope == COUNTRY_SCOPE else "pan-European"
            title = f"Uncertainty cascade - {asset} - {label} - protection treated two ways"
            out = (odir / scope_dir(scope, country)
                   / f"{asset}_protection_{style}_pyramid.png")

            fig, ax = plt.subplots(figsize=(10.2, 0.55 * (len(trunk) + 2) + 3.0),
                                   layout="constrained")
            if style == "split":
                draw_fork_split(ax, trunk, stats_a.iloc[-1], stats_b.iloc[-1],
                                SPLIT_LABEL_ABS, SPLIT_LABEL_MUL, title, log_x)
                handles = legend_handles_pyramid() + [
                    Patch(facecolor=FORK_BAND_50, label="alternative scenario"),
                ]
            else:
                draw_fork(ax, trunk, stats_a.iloc[-1], stats_b.iloc[-1],
                          *FORK_LABELS, title, log_x)
                # The final row carries both treatments nested, so the legend is
                # what tells them apart - the row label cannot.
                handles = legend_handles_pyramid() + [
                    Patch(facecolor=BAND_50, label=f"protection {FORK_LABELS[0]}"),
                    Patch(facecolor=FORK_BAND_50, label=f"protection {FORK_LABELS[1]}"),
                ]
            leg = ax.legend(handles=handles, frameon=False,
                            fontsize=legend_size(), ncol=5,
                            loc="upper center", bbox_to_anchor=(0.5, -0.09))
            for t in leg.get_texts():
                t.set_color(furniture(INK_2))
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=200)
            plt.close(fig)
            n += 1
    return n


def _write_summary(steps: list[pd.DataFrame], meta: list[dict], dep: list[dict], out_dir: Path) -> None:
    steps_df = pd.concat(steps, ignore_index=True) if steps else pd.DataFrame()
    meta_df = pd.DataFrame(meta)
    dep_df = pd.DataFrame(dep)

    out_dir.mkdir(parents=True, exist_ok=True)
    steps_df.to_csv(out_dir / "pyramid_summary_steps.csv", index=False)
    meta_df.to_csv(out_dir / "pyramid_summary_meta.csv", index=False)
    if not dep_df.empty:
        dep_df.to_csv(out_dir / "pyramid_summary_dependence.csv", index=False)

    xlsx = out_dir / "pyramid_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        meta_df.round(6).to_excel(writer, sheet_name="Meta", index=False)
        if not dep_df.empty:
            dep_df.round(6).to_excel(writer, sheet_name="Dependence", index=False)
        steps_df.round(6).to_excel(writer, sheet_name="Steps", index=False)
    print(f"Summary written to {xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cascade-dir", default=None, help="default: results/cascade")
    parser.add_argument("--out-dir", default=None, help="default: overview_figures/pyramids")
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--scopes", nargs="+", default=None,
                        help="e.g. eu_sum (skips the per-country figures)")
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--log-x", action="store_true", help="log-scale the value axis")
    parser.add_argument("--no-figures", action="store_true", help="summary tables only")
    parser.add_argument("--pair-protection", action="store_true",
                        help="draw sampled-vs-fixed protection pairs side by side on a "
                             "shared value axis, instead of the single-scenario pyramids.")
    parser.add_argument("--fork-protection", action="store_true",
                        help="one cascade with a SHARED trunk whose final step forks "
                             "into the absolute (RP 5-200) and multiplicative "
                             "(x0.5-1.5) protection treatments. Valid only because "
                             "the two scenarios share every earlier step draw for "
                             "draw, which is checked before drawing.")
    parser.add_argument("--fork-style", choices=["split", "nested"], default="split",
                        help="split (default): the multiplicative treatment continues "
                             "the chain, the absolute one sits below a visible break "
                             "as a separate scenario. nested: both on one row, the "
                             "narrower inside the wider.")
    parser.add_argument("--no-width-axis", action="store_true",
                        help="drop the right-hand axis showing p5-p95 width as a "
                             "percent of the median. The same numbers are in the "
                             "summary CSV; on a report page the second tick column "
                             "only crowds the row labels.")
    parser.add_argument("--report", action="store_true",
                        help="report styling: pure white page, EAD in billions of "
                             "euros with the unit spelled out, and no subtitle "
                             "(its space is kept so the title still clears the plot).")
    parser.add_argument("--merge-curves", action="store_true",
                        help="show one vulnerability-curve level instead of one per "
                             "object sub-type group. Keeps the step where every curve "
                             "group is free and drops the partial ones.")
    args = parser.parse_args()

    global MERGE_CURVES, WIDTH_AXIS
    MERGE_CURVES = args.merge_curves
    WIDTH_AXIS = not args.no_width_axis
    if args.report:
        use_report_style()

    cdir = Path(args.cascade_dir) if args.cascade_dir else default_results_dir() / "cascade"
    cdir = resolve_cascade_dir(cdir)
    odir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures" / "pyramids"

    # Keep figures from different orderings apart, exactly as the cascades are.
    if args.out_dir is None and (cdir / ORDER_FILE).exists():
        odir = odir / cdir.name

    files = sorted(cdir.glob("*_cascade.parquet"))
    if not files:
        raise SystemExit(f"No *_cascade.parquet in {cdir} - run `python -m src.cascade` first.")

    if args.pair_protection:
        n = build_pairs(cdir, odir, args.assets, args.scopes, args.countries, args.log_x)
        print(f"{n} paired figure(s) written under {odir}")
        return

    if args.fork_protection:
        n = build_fork(cdir, odir, args.scopes, args.log_x, args.fork_style)
        print(f"{n} forked figure(s) written under {odir}")
        return

    all_steps, all_meta, all_dep = [], [], []
    n_figs = 0

    for path in files:
        df = pd.read_parquet(path)
        asset = df["asset"].iloc[0]
        scenario = df["scenario"].iloc[0]
        if args.assets and asset not in args.assets:
            continue
        if args.scenarios and scenario not in args.scenarios:
            continue

        meta_path = path.with_name(path.name.replace(".parquet", "_meta.json"))
        run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        order_mode = run_meta.get("order_mode", df["order_mode"].iloc[0])
        n_draws = int(run_meta.get("n_draws", df["draw"].nunique()))

        print(f"{asset} / {scenario}: {df['scope'].nunique()} scope(s)")

        for eu_scope in sorted(set(df["scope"].unique()) - {COUNTRY_SCOPE}):
            comp = dependence_comparison(df, eu_scope)
            if comp:
                all_dep.append({"scope": eu_scope, "asset": asset, "scenario": scenario, **comp})

        groups = df.groupby(["scope", "country"], sort=True)
        for (scope, country), sub in groups:
            if args.scopes and scope not in args.scopes:
                continue
            if scope == COUNTRY_SCOPE and args.countries and country not in args.countries:
                continue

            stats = step_stats(sub)
            stats.insert(0, "scenario", scenario)
            stats.insert(0, "asset", asset)
            stats.insert(0, "country", country)
            stats.insert(0, "scope", scope)
            all_steps.append(stats)

            final = stats.iloc[-1]
            widen = stats[stats["step"] > 0].sort_values("d_w90", ascending=False)
            top = widen[["factor_added", "share_w90"]].head(3).to_numpy()
            all_meta.append({
                "scope": scope, "country": country, "asset": asset, "scenario": scenario,
                "order_mode": order_mode, "n_draws": n_draws,
                "n_steps": int(len(stats)),
                "n_countries_summed": len(run_meta.get("countries", [])) if scope != COUNTRY_SCOPE else 1,
                "nominal_EAD": float(stats["median"].iloc[0]),
                "final_median": float(final["median"]),
                "final_mean": float(final["mean"]),
                "final_p5": float(final["p5"]), "final_p25": float(final["p25"]),
                "final_p75": float(final["p75"]), "final_p95": float(final["p95"]),
                "final_w90": float(final["w90"]), "final_w50": float(final["w50"]),
                "final_w90_rel": float(final["w90_rel"]),
                "median_drift_from_nominal": float(final["drift_median"]),
                "mean_median_ratio": float(final["mean_median_ratio"]),
                "top1_widener": top[0][0] if len(top) > 0 else "",
                "top1_share": float(top[0][1]) if len(top) > 0 else np.nan,
                "top2_widener": top[1][0] if len(top) > 1 else "",
                "top2_share": float(top[1][1]) if len(top) > 1 else np.nan,
                "top3_widener": top[2][0] if len(top) > 2 else "",
                "top3_share": float(top[2][1]) if len(top) > 2 else np.nan,
                "factor_order": " -> ".join(run_meta.get("factor_order", [])),
            })

            if args.no_figures:
                continue
            label = "pan-European (sum of countries)" if scope == "eu_sum" else (
                "pan-European (native run)" if scope == "eu_native" else country
            )
            subtitle = (
                f"cascade order: {order_mode}  |  {n_draws:,} draws per step  |  "
                f"common random numbers across countries"
                if scope != COUNTRY_SCOPE
                else f"cascade order: {order_mode}  |  {n_draws:,} draws per step"
            )
            out = odir / scope_dir(scope, country) / f"{asset}_{scenario}_pyramid.png"
            draw_pyramid(
                stats,
                title=f"Uncertainty cascade - {asset} / {scenario} - {label}",
                subtitle=subtitle, out=out, log_x=args.log_x,
            )
            n_figs += 1

    _write_summary(all_steps, all_meta, all_dep, odir)
    print(f"{n_figs} figure(s) written under {odir}")


if __name__ == "__main__":
    main()
