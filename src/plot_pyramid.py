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

Output layout under overview_figures/pyramids/:
    pan_european/{asset}_{scenario}_pyramid.png          scope "eu_sum"
    pan_european_native/{asset}_{scenario}_pyramid.png   scope "eu_native"
    countries/{ISO3}/{asset}_{scenario}_pyramid.png      scope "country"
    pyramid_summary.xlsx / .csv                          the numbers

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
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND_90 = "#cde2fb"
BAND_50 = "#86b6ef"
ACCENT = "#1c5cab"

COUNTRY_SCOPE = "country"
SCOPE_DIRS = {"eu_sum": "pan_european", "eu_native": "pan_european_native"}

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


def readable(factor: str) -> str:
    """Human label for a factor code, without importing the heavy stack.

    The fully-resolved curve descriptions (which object types sit behind
    curve_F1_1 for THIS asset) live in the workbook's Curve_Groups sheet; here
    the curve id alone keeps the axis narrow enough to read.
    """
    if factor in _PRETTY:
        return _PRETTY[factor]
    if factor.startswith("curve_"):
        return "curve " + factor[len("curve_"):].replace("_", ".")
    return factor.replace("_", " ")


def scope_dir(scope: str, country: str) -> Path:
    if scope == COUNTRY_SCOPE:
        return Path("countries") / country
    return Path(SCOPE_DIRS.get(scope, scope))


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


def draw_pyramid(stats: pd.DataFrame, title: str, subtitle: str, out: Path, log_x: bool) -> None:
    n_steps = len(stats)
    y = np.arange(n_steps)
    p5, p25, p50, p75, p95 = (stats[c].to_numpy() for c in ("p5", "p25", "p50", "p75", "p95"))
    mean = stats["mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.2, 0.55 * n_steps + 2.6), layout="constrained")

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
    ax.axvline(p50[0], color=BASELINE, linestyle="--", linewidth=1.0, zorder=0)

    labels = [
        "nominal (all factors fixed)" if s == 0 else f"+ {readable(f)}"
        for s, f in zip(stats["step"], stats["factor_added"])
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("total EAD (M EUR / yr)")
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=16)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)

    # Relative 90% width per step, on a twin axis so it cannot collide with the
    # data no matter how far the band extends.
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y, [
        "-" if not np.isfinite(v) else f"{v:.0%}" for v in stats["w90_rel"]
    ])
    ax2.set_ylabel("p5-p95 width\n(% of median)", color=MUTED, fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    handles = [
        Patch(facecolor=BAND_90, label="p5-p95"),
        Patch(facecolor=BAND_50, label="p25-p75"),
        Line2D([], [], marker="|", color=INK, markersize=13, markeredgewidth=2.4,
               linestyle="none", label="median"),
        Line2D([], [], marker="D", markerfacecolor=SURFACE, markeredgecolor=INK,
               markersize=6, linestyle="none", label="mean"),
    ]
    # Below the axes, not inside: the bottom-right corner is exactly where the
    # widest band lands, so an in-axes legend would sit on top of the result.
    leg = ax.legend(
        handles=handles, frameon=False, fontsize=9, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
    )
    for t in leg.get_texts():
        t.set_color(INK_2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
    args = parser.parse_args()

    cdir = Path(args.cascade_dir) if args.cascade_dir else default_results_dir() / "cascade"
    odir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures" / "pyramids"

    files = sorted(cdir.glob("*_cascade.parquet"))
    if not files:
        raise SystemExit(f"No *_cascade.parquet in {cdir} - run `python -m src.cascade` first.")

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
