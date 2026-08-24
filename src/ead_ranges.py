"""Full-uncertainty EAD ranges for every finished combination - from the
archives already on disk, with NO new model runs.

Where the samples come from
---------------------------
Each combination has two archives, and BOTH hold valid draws:

  lhs      n=3000 plain Latin-hypercube draws over the scenario's full factor set.
  sobol    a Saltelli design of N*(2k+2) rows. Its layout is N repeating blocks
           of (2k+2): row 0 is the base matrix A, rows 1..k are AB_i, rows
           k+1..2k are BA_i, and the last row is the base matrix B.

A and B are two INDEPENDENT samples of the factor distribution, so pooling them
gives 2N = 16,384 draws at N=8192 - about 5.5x the LHS sample, already computed,
free to use. That is the default here (--sampler sobol-ab).

The AB_i/BA_i rows are individually valid too (the factors are sampled
independently, so a row mixing A and B coordinates still follows the joint
distribution - verified empirically: pooling all 114,688 rows reproduces the
A+B percentiles to within 1%). But every AB_i row shares k-1 coordinates with an
A row, so the full set is anchored on A's particular realisation and its
EFFECTIVE sample size is far closer to 2N than to N*(2k+2). Pooling everything
would buy almost nothing over A+B while making a naive standard error look ~2.5x
better than it really is, so it is deliberately not offered.

Relation to the cascade
-----------------------
In src/cascade.py terms this is the BOTTOM row of every per-country pyramid: all
factors varying at once. The intermediate steps cannot be recovered from these
samples - conditioning them on a frozen complement leaves single-digit counts -
and the pan-European rows cannot either, since each combination was sampled
independently and summing independent draws assumes the national errors
diversify away.

Reading the archives
--------------------
ema_workbench's save_results writes a tarball of experiments.csv, one plain-text
.cls file per scalar outcome (one value per line), and metadata.json. That is
read here with tarfile + numpy directly rather than through
ema_workbench.load_results, so this module runs in any environment that can read
the files - no ema_workbench, no scipy, no SALib.

Units
-----
Money outcomes are stored as M EUR but reported and plotted in EUR: the smallest
combinations land around 1e-9 M EUR, which is unreadable. exposed_qty_* outcomes
are physical quantities (m / m^2 / count) and are left untouched; every row
carries a `units` column saying which it is.

Outputs (under overview_figures/ead_ranges/)
--------------------------------------------
    EAD_Ranges.xlsx                 Ranges / All_Outcomes / Meta
    {scale}/by_combo/               all countries, one figure per (asset, scenario)
    {scale}/by_asset_panels/        one panel per scenario, countries down the y axis
    {scale}/flood_by_country/       the flood + coastal scenarios stacked per country
    {scale}/all_scenarios_by_country/  every scenario stacked per country
with {scale} in {log, linear}, so the same data can be read both ways.

Usage:
    python -m src.ead_ranges
    python -m src.ead_ranges --sampler lhs --scales log
    python -m src.ead_ranges --assets power --countries DEU FRA ITA
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_pyramid import (
    ACCENT,
    BAND_90,
    INK,
    INK_2,
    MUTED,
    PROJECT_ROOT,
    SURFACE,
    default_results_dir,
)

PERCENTILES = (5, 25, 50, 75, 95)
HEADLINE = "total_EAD_MEUR"
MEUR_TO_EUR = 1e6

_KEEP_PATTERNS = (
    re.compile(r"^EAD_.*_MEUR$"),
    re.compile(r"^damage_RP100_.*_MEUR$"),
    re.compile(r"^exposed_qty_RP100_.*$"),
)
ARCHIVE_RE = re.compile(r"^experiments_(.+)\.tar\.gz$")

# Categorical slots 1-4 of the dataviz reference palette, validated for this
# 4-series set (worst adjacent CVD dE 9.1, normal-vision 22.9). Aqua and yellow
# fall below 3:1 on the light surface, so the relief rule applies - every row
# carries a visible scenario label, never colour alone.
HAZARD_COLORS = {
    "coastal": "#2a78d6",     # blue
    "earthquake": "#eb6834",  # orange
    "flood": "#1baf7a",       # aqua
    "windstorm": "#eda100",   # yellow
}


def hazard_of(scenario: str) -> str:
    for hz in ("coastal", "flood", "windstorm"):
        if scenario.startswith(hz):
            return hz
    return "earthquake"


def is_water(scenario: str) -> bool:
    """River flood and coastal flood - the 'flood scenarios' as a family."""
    return hazard_of(scenario) in ("flood", "coastal")


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------


def parse_archive_name(path: Path) -> tuple[str, str, str, str, int] | None:
    """(country, asset, scenario, sampler, n) from an archive filename.

    Layout is experiments_{C}_{asset}_{scenario}_{sampler}_n{N}_{date}_{time},
    and the scenario itself contains underscores (flood_absprot_ds), so the
    fixed-width tail is peeled off the right and whatever remains is the scenario.
    """
    m = ARCHIVE_RE.match(path.name)
    if not m:
        return None
    parts = m.group(1).split("_")
    if len(parts) < 7:
        return None
    sampler, n_tok = parts[-4], parts[-3]
    if sampler not in ("lhs", "sobol") or not n_tok.startswith("n"):
        return None
    try:
        n = int(n_tok[1:])
    except ValueError:
        return None
    scenario = "_".join(parts[2:-4])
    return (parts[0], parts[1], scenario, sampler, n) if scenario else None


def read_archive_outcomes(path: Path) -> dict[str, np.ndarray]:
    """Scalar outcomes from one archive. experiments.csv is skipped - only the
    outcome distributions are needed, and it is by far the largest member."""
    out: dict[str, np.ndarray] = {}
    with tarfile.open(path) as t:
        meta_f = t.extractfile("metadata.json")
        if meta_f is None:
            return out
        for entry in json.loads(meta_f.read().decode("utf-8")).get("outcomes", []):
            if len(entry) < 3 or entry[0] != "ScalarOutcome":
                continue
            f = t.extractfile(entry[2])
            if f is not None:
                out[entry[1]] = np.fromstring(f.read().decode("utf-8"), sep="\n")
    return out


def extract_ab(v: np.ndarray, n_base: int) -> np.ndarray | None:
    """The two independent base matrices A and B out of a Saltelli sample.

    block = rows / N is (2k+2) when second-order indices were computed and
    (k+2) otherwise; A sits at offset 0 and B at offset block-1 in BOTH
    layouts, so this needs no knowledge of k. Returns None if the row count is
    not a clean multiple of N, rather than silently mis-slicing.
    """
    if n_base <= 0 or len(v) % n_base != 0:
        return None
    block = len(v) // n_base
    if block < 3:
        return None
    return np.concatenate([v[0::block], v[block - 1::block]])


def find_archives(results_dir: Path, countries, assets, scenarios) -> dict:
    """Newest lhs and sobol archive per (country, asset, scenario).

    Iterates the per-country folders explicitly instead of a recursive glob:
    results/ sits on network storage where one listing per country is far
    cheaper than walking the whole tree.
    """
    found: dict[tuple[str, str, str], dict[str, tuple[Path, int]]] = {}
    for d in sorted(p for p in results_dir.iterdir()
                    if p.is_dir() and re.fullmatch(r"[A-Z]{3}", p.name)):
        if countries and d.name not in countries:
            continue
        for path in d.glob("experiments_*.tar.gz"):
            parsed = parse_archive_name(path)
            if not parsed:
                continue
            country, asset, scenario, sampler, n = parsed
            if (assets and asset not in assets) or (scenarios and scenario not in scenarios):
                continue
            slot = found.setdefault((country, asset, scenario), {})
            prev = slot.get(sampler)
            if prev is None or path.stat().st_mtime > prev[0].stat().st_mtime:
                slot[sampler] = (path, n)
    return found


def load_combo(slot: dict, mode: str) -> tuple[dict[str, np.ndarray], str]:
    """Outcome arrays for one combination under the chosen sampling mode.

    Falls back to whatever archive exists, so a partially-finished study still
    reports; the returned label records what was actually used.
    """
    lhs = slot.get("lhs")
    sob = slot.get("sobol")

    def _lhs():
        return (read_archive_outcomes(lhs[0]), "lhs") if lhs else ({}, "")

    if mode == "lhs":
        return _lhs()

    ab: dict[str, np.ndarray] = {}
    if sob:
        raw = read_archive_outcomes(sob[0])
        for name, v in raw.items():
            got = extract_ab(v, sob[1])
            if got is not None:
                ab[name] = got
    if mode == "sobol-ab":
        return (ab, "sobol A+B") if ab else _lhs()

    # both: pool the LHS draws with A and B - all three are independent samples
    # of the same distribution, so concatenating them is legitimate.
    lv, _ = _lhs()
    if not ab:
        return lv, "lhs"
    if not lv:
        return ab, "sobol A+B"
    merged = {n: np.concatenate([lv[n], ab[n]]) for n in ab if n in lv}
    return (merged, "lhs + sobol A+B") if merged else (ab, "sobol A+B")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wanted(name: str) -> bool:
    return name == HEADLINE or any(p.match(name) for p in _KEEP_PATTERNS)


def stats_for(name: str, v: np.ndarray) -> dict:
    """Distribution summary, converted to EUR for the money outcomes."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    if name.endswith("_MEUR"):
        v = v * MEUR_TO_EUR
        units = "EUR"
    else:
        units = "quantity (m / m2 / count)"

    q = np.percentile(v, PERCENTILES)
    med, mean = float(q[2]), float(v.mean())
    w90 = float(q[4] - q[0])
    return {
        "units": units,
        "n": int(v.size),
        "mean": mean,
        "median": med,
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "p5": float(q[0]), "p25": float(q[1]), "p50": med,
        "p75": float(q[3]), "p95": float(q[4]),
        "w90": w90,
        "w50": float(q[3] - q[1]),
        "w90_rel": w90 / med if med > 0 else np.nan,
        "mean_median_ratio": mean / med if med > 0 else np.nan,
        # A large zero fraction means the band is driven by how OFTEN damage
        # occurs at all, not by how large it is - and it makes w90_rel
        # meaningless, since the median it divides by is ~0.
        "zero_fraction": float((v == 0).mean()),
        "p5_is_zero": bool(q[0] <= 0),
    }


# ---------------------------------------------------------------------------
# Shared drawing
# ---------------------------------------------------------------------------


def _floor_for(values: pd.Series, scale: str) -> float | None:
    """Left edge for a log axis: three decades below the smallest positive median."""
    if scale != "log":
        return None
    pos = values[values > 0]
    return float(pos.min()) / 1000.0 if not pos.empty else None


def _draw_rows(ax, rows: pd.DataFrame, y, floor, color=ACCENT, thick=3.4, scale="log"):
    """One range bar per row at the given y positions.

    When the reference columns are present, the deterministic MIRACA_RISK value
    is drawn as a triangle just below the bar, pointing up at its position on
    the axis, with a thin bracket for the reference run's own min-max. That
    idiom keeps it clearly a REFERENCE annotation rather than another statistic
    of the sampled distribution, so it cannot be confused with the median tick
    or the mean diamond.
    """
    p5, p25, p50, p75, p95 = (rows[c].to_numpy() for c in ("p5", "p25", "p50", "p75", "p95"))
    mean_v = rows["mean"].to_numpy()
    zero = rows["p5_is_zero"].to_numpy()
    lo = np.maximum(p5, floor) if floor is not None else p5
    if floor is not None:
        p25, p75, p95 = (np.maximum(a, floor) for a in (p25, p75, p95))

    has_ref = "ref_mid" in rows.columns
    if has_ref:
        r_mid = rows["ref_mid"].to_numpy(float)
        r_lo = rows.get("ref_min", pd.Series(np.nan, index=rows.index)).to_numpy(float)
        r_hi = rows.get("ref_max", pd.Series(np.nan, index=rows.index)).to_numpy(float)

    for j, i in enumerate(y):
        ax.hlines(i, lo[j], p95[j], color=color, linewidth=1.1, alpha=0.85, zorder=3)
        ax.hlines(i, p25[j], p75[j], color=color, linewidth=thick, alpha=1.0, zorder=4)
        ax.plot([max(p50[j], floor) if floor else p50[j]], [i], marker="|",
                markersize=11, markeredgewidth=2.0, color=INK, zorder=6)
        ax.plot([max(mean_v[j], floor) if floor else mean_v[j]], [i], marker="D",
                markersize=4.4, markerfacecolor=SURFACE, markeredgecolor=INK,
                markeredgewidth=1.1, linestyle="none", zorder=6)
        # Distinguish "the lower bound is genuinely zero" from "it is small and
        # the log axis cannot reach it" - on a log scale both otherwise just
        # stop at the left edge.
        if zero[j]:
            ax.plot([lo[j]], [i], marker="o", markersize=5.0, markerfacecolor="none",
                    markeredgecolor=MUTED, markeredgewidth=1.3, linestyle="none", zorder=7)
        elif floor is not None and p5[j] < floor:
            ax.plot([floor], [i], marker="<", markersize=5.0, color=MUTED,
                    linestyle="none", zorder=7)

        if has_ref and np.isfinite(r_mid[j]) and r_mid[j] > 0:
            yr = i + 0.30
            if np.isfinite(r_lo[j]) and np.isfinite(r_hi[j]) and r_hi[j] > r_lo[j]:
                b_lo = max(r_lo[j], floor) if floor else r_lo[j]
                ax.hlines(yr, b_lo, max(r_hi[j], floor) if floor else r_hi[j],
                          color=MUTED, linewidth=1.0, alpha=0.9, zorder=5)
            ax.plot([max(r_mid[j], floor) if floor else r_mid[j]], [yr], marker="^",
                    markersize=5.6, color=INK, linestyle="none", zorder=7)


def _legend(fig, extra=(), ncol=None, interval_color=ACCENT, reference=False):
    """Legend below the whole figure.

    fig.legend(loc="outside lower center") rather than an axes legend with a
    negative bbox offset: that offset is in AXES coordinates, so the gap it
    leaves shrinks with the axes height and collides with the x label on short
    figures. The outside placement is negotiated by constrained layout, so it
    clears the label at any figure size.

    interval_color is neutral on the grouped charts: there the bar colour
    carries the hazard, so drawing the p5-p95 key in the accent blue would read
    as the coastal series rather than as "an interval".
    """
    handles = [
        Line2D([], [], color=interval_color, linewidth=1.1, alpha=0.85, label="p5-p95"),
        Line2D([], [], color=interval_color, linewidth=3.4, label="p25-p75"),
        Line2D([], [], marker="|", color=INK, markersize=12, markeredgewidth=2.0,
               linestyle="none", label="median"),
        Line2D([], [], marker="D", markerfacecolor=SURFACE, markeredgecolor=INK,
               markersize=6, linestyle="none", label="mean"),
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=MUTED,
               markersize=6, linestyle="none", label="p5 = 0"),
        Line2D([], [], marker="<", color=MUTED, markersize=6, linestyle="none",
               label="p5 below axis"),
        *([Line2D([], [], marker="^", color=INK, markersize=6, linestyle="none",
                  label="MIRACA_RISK (mid, bracket = its min-max)")] if reference else []),
        *extra,
    ]
    leg = fig.legend(handles=handles, frameon=False, fontsize=9,
                     ncol=ncol or min(len(handles), 6), loc="outside lower center")
    for t in leg.get_texts():
        t.set_color(INK_2)


def _finish(ax, scale, floor, xlabel="total EAD (EUR / yr)"):
    if scale == "log" and floor is not None:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_country_ranges(sub: pd.DataFrame, asset, scenario, out: Path, scale: str) -> None:
    """All countries for one (asset, scenario), each as its own range bar."""
    sub = sub.sort_values("median", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.8, 0.30 * len(sub) + 3.1), layout="constrained")
    for i in range(0, len(sub), 2):
        ax.axhspan(i - 0.5, i + 0.5, color=BAND_90, alpha=0.25, linewidth=0, zorder=0)

    floor = _floor_for(sub["median"], scale)
    _draw_rows(ax, sub, np.arange(len(sub)), floor, thick=4.0, scale=scale)
    ax.set_yticks(np.arange(len(sub)), sub["country"])
    ax.invert_yaxis()
    _finish(ax, scale, floor)
    ax.set_title(f"Full-uncertainty range by country - {asset} / {scenario}", pad=20)
    ax.text(0.0, 1.012,
            f"{int(sub['n'].max()):,} draws per country ({sub['source'].iloc[0]}), every factor varying"
            f"  |  sampled independently per country - not summable to a European total",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _legend(fig, reference="ref_mid" in sub.columns)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_asset_panels(sub: pd.DataFrame, asset, out: Path, scale: str, max_cols=4) -> None:
    """Small multiples: one panel per scenario, x axis shared so the hazards
    are actually comparable. Countries are ordered by their median ACROSS
    scenarios - ordering on one coastal scenario would push the big inland
    countries to the bottom of every panel."""
    scenarios = sorted(sub["scenario"].unique())
    order = sub.groupby("country")["median"].median().sort_values(ascending=False).index.tolist()
    pos = {c: i for i, c in enumerate(order)}

    n_cols = min(max_cols, len(scenarios))
    n_rows = int(np.ceil(len(scenarios) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, sharey=True, sharex=True,
        figsize=(3.6 * n_cols + 1.2, (0.26 * len(order) + 1.5) * n_rows + 1.2),
        layout="constrained", squeeze=False,
    )
    flat = axes.ravel()
    floor = _floor_for(sub["median"], scale)

    for ax, scen in zip(flat, scenarios):
        rows = sub[sub["scenario"] == scen].copy()
        rows["_y"] = rows["country"].map(pos)
        rows = rows.sort_values("_y")
        for i in range(0, len(order), 2):
            ax.axhspan(i - 0.5, i + 0.5, color=BAND_90, alpha=0.25, linewidth=0, zorder=0)
        _draw_rows(ax, rows, rows["_y"].to_numpy(), floor,
                   color=HAZARD_COLORS[hazard_of(scen)], thick=3.2, scale=scale)
        ax.set_title(scen, fontsize=10)
        _finish(ax, scale, floor)
        ax.set_xlabel("total EAD (EUR / yr)", fontsize=9)
    for ax in flat[len(scenarios):]:
        ax.set_visible(False)
    for r in range(n_rows):
        axes[r][0].set_yticks(np.arange(len(order)), order)
    flat[0].invert_yaxis()
    _legend(fig, reference="ref_mid" in sub.columns)
    fig.suptitle(f"Full-uncertainty range by country and scenario - {asset}", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_scenarios_by_country(
    sub: pd.DataFrame, asset: str, out: Path, scale: str, title: str
) -> None:
    """Scenarios stacked directly above each other within each country block.

    This is the layout that answers "does the spread change materially between
    the scenarios we test?" - the bars for one country sit adjacent, so a
    difference in width is read directly instead of across two figures. Hazard
    colour is a redundant encoding; every row also carries its scenario name.
    """
    scenarios = sorted(sub["scenario"].unique())
    order = sub.groupby("country")["median"].median().sort_values(ascending=False).index.tolist()

    ypos, labels, rows_idx = [], [], []
    y = 0.0
    blocks = []
    for country in order:
        start = y
        for scen in scenarios:
            hit = sub[(sub["country"] == country) & (sub["scenario"] == scen)]
            if hit.empty:
                continue
            rows_idx.append(hit.index[0])
            ypos.append(y)
            labels.append(f"{country} · {scen}" if y == start else f"      {scen}")
            y += 1.0
        if y > start:
            blocks.append((start - 0.5, y - 0.5))
            y += 0.6  # visual gap between countries
    if not rows_idx:
        return

    rows = sub.loc[rows_idx].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.6, 0.19 * len(rows) + 0.35 * len(blocks) + 3.2),
                           layout="constrained")
    for bi, (lo, hi) in enumerate(blocks):
        if bi % 2 == 0:
            ax.axhspan(lo, hi, color=BAND_90, alpha=0.25, linewidth=0, zorder=0)

    floor = _floor_for(sub["median"], scale)
    yarr = np.asarray(ypos)
    for hz, color in HAZARD_COLORS.items():
        m = rows["scenario"].map(hazard_of).to_numpy() == hz
        if m.any():
            _draw_rows(ax, rows[m], yarr[m], floor, color=color, thick=3.2, scale=scale)

    ax.set_yticks(yarr, labels, fontsize=8)
    ax.invert_yaxis()
    _finish(ax, scale, floor)
    ax.set_title(title, pad=20)
    ax.text(0.0, 1.006,
            f"{int(rows['n'].max()):,} draws per bar ({rows['source'].iloc[0]})  |  "
            f"colour = hazard family (redundant with the row label)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    hz_present = [h for h in HAZARD_COLORS if (rows["scenario"].map(hazard_of) == h).any()]
    _legend(fig, interval_color=INK_2, reference="ref_mid" in rows.columns,
            extra=[Line2D([], [], color=HAZARD_COLORS[h], linewidth=3.2, label=h)
                   for h in hz_present])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default=None, help="default: results/")
    parser.add_argument("--out-dir", default=None, help="default: overview_figures/ead_ranges")
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--sampler", choices=["sobol-ab", "lhs", "both"], default="sobol-ab",
                        help="sobol-ab (default): the two independent Saltelli base matrices "
                             "A and B, 2N draws (16,384 at N=8192) - ~5x the LHS sample, already "
                             "computed. lhs: the 3,000-draw LHS archive only (much faster to "
                             "read). both: pool them.")
    parser.add_argument("--scales", nargs="+", choices=["log", "linear"],
                        default=["log", "linear"],
                        help="one figure folder per scale (default: both)")
    parser.add_argument("--reference", default=None,
                        help="Reference_EAD.csv from `python -m src.reference_ead` "
                             "(default: overview_figures/reference/Reference_EAD.csv if present). "
                             "Overlays the deterministic MIRACA_RISK EAD on every range.")
    parser.add_argument("--reference-climate", default="current",
                        help="which MIRACA_RISK climate horizon to overlay (default: current, "
                             "the one the study's own scenarios are built against)")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else default_results_dir()
    odir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures" / "ead_ranges"

    # Reference lookup: (country, asset, hazard) -> the deterministic totals.
    # Keyed on hazard rather than scenario, because MIRACA_RISK computes one
    # number per hazard while this study runs several scenario TREATMENTS of
    # that same hazard - all of which the one reference value belongs on.
    ref: dict[tuple[str, str, str], dict] = {}
    ref_path = Path(args.reference) if args.reference else (
        PROJECT_ROOT / "overview_figures" / "reference" / "Reference_EAD.csv")
    if not args.no_reference and ref_path.exists():
        rdf = pd.read_csv(ref_path)
        rdf = rdf[rdf["climate"] == args.reference_climate]
        for r in rdf.itertuples():
            ref[(r.country, r.asset, r.hazard)] = {
                "ref_min": float(r.ead_min), "ref_mid": float(r.ead_mid),
                "ref_max": float(r.ead_max),
            }
        print(f"Reference: {len(ref)} (country, asset, hazard) rows from {ref_path.name} "
              f"[climate={args.reference_climate}]")
    elif not args.no_reference:
        print(f"Reference: none at {ref_path} - run `python -m src.reference_ead` to add it")

    slots = find_archives(results_dir, set(args.countries or []), set(args.assets or []),
                          set(args.scenarios or []))
    if not slots:
        raise SystemExit(f"No experiment archives found under {results_dir}")
    print(f"Found {len(slots)} combinations; reading (sampler={args.sampler})...")

    rows, all_rows, meta_rows = [], [], []
    for i, (key, slot) in enumerate(sorted(slots.items()), 1):
        country, asset, scenario = key
        try:
            outcomes, source = load_combo(slot, args.sampler)
        except Exception as exc:
            print(f"  [{i}/{len(slots)}] {country}/{asset}/{scenario}: "
                  f"FAILED ({type(exc).__name__}: {exc})")
            continue
        if not outcomes:
            continue

        base = {"country": country, "asset": asset, "scenario": scenario}
        for name, v in outcomes.items():
            if not wanted(name):
                continue
            st = stats_for(name, v)
            if not st:
                continue
            all_rows.append({**base, "outcome": name, "source": source, **st})
            if name == HEADLINE:
                row = {**base, "source": source, **st}
                hit = ref.get((country, asset, hazard_of(scenario)))
                if hit:
                    # Percentile computed from the draws themselves, not
                    # interpolated between stored quantiles - this is the
                    # headline diagnostic ("where does the published number sit
                    # inside our range?"), so it should be exact.
                    v_eur = np.asarray(v, dtype=float) * MEUR_TO_EUR
                    row.update(hit)
                    row["ref_percentile"] = float((v_eur < hit["ref_mid"]).mean())
                    row["ref_in_p5_p95"] = bool(st["p5"] <= hit["ref_mid"] <= st["p95"])
                    row["ref_over_median"] = (
                        hit["ref_mid"] / st["median"] if st["median"] > 0 else np.nan)
                rows.append(row)

        meta_rows.append({
            **base, "source": source,
            "lhs_archive": slot["lhs"][0].name if "lhs" in slot else "",
            "sobol_archive": slot["sobol"][0].name if "sobol" in slot else "",
            "sobol_N": slot["sobol"][1] if "sobol" in slot else np.nan,
        })
        if i % 50 == 0 or i == len(slots):
            print(f"  [{i}/{len(slots)}] read")

    ranges = pd.DataFrame(rows).sort_values(["asset", "scenario", "median"], ascending=[1, 1, 0])
    odir.mkdir(parents=True, exist_ok=True)
    ranges.to_csv(odir / "EAD_Ranges.csv", index=False)
    pd.DataFrame(all_rows).to_csv(odir / "EAD_Ranges_all_outcomes.csv", index=False)
    with pd.ExcelWriter(odir / "EAD_Ranges.xlsx", engine="openpyxl") as writer:
        ranges.round(4).to_excel(writer, sheet_name="Ranges", index=False)
        pd.DataFrame(all_rows).round(4).to_excel(writer, sheet_name="All_Outcomes", index=False)
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Meta", index=False)
    print(f"Workbook: {odir / 'EAD_Ranges.xlsx'}  ({len(ranges)} combinations, "
          f"{len(all_rows)} outcome rows)")

    if args.no_figures or ranges.empty:
        return

    n_figs = 0
    for scale in args.scales:
        root = odir / scale
        for (asset, scenario), sub in ranges.groupby(["asset", "scenario"]):
            if sub["p95"].max() <= 0:
                continue
            plot_country_ranges(sub, asset, scenario,
                                root / "by_combo" / f"{asset}_{scenario}_country_ranges.png", scale)
            n_figs += 1
        for asset, sub in ranges.groupby("asset"):
            if sub["p95"].max() <= 0:
                continue
            if sub["scenario"].nunique() > 1:
                plot_asset_panels(sub, asset,
                                  root / "by_asset_panels" / f"{asset}_scenarios.png", scale)
                n_figs += 1
                plot_scenarios_by_country(
                    sub, asset, root / "all_scenarios_by_country" / f"{asset}_all_scenarios.png",
                    scale, f"All scenarios by country - {asset}")
                n_figs += 1
            water = sub[sub["scenario"].map(is_water)]
            if water["scenario"].nunique() > 1 and water["p95"].max() > 0:
                plot_scenarios_by_country(
                    water, asset, root / "flood_by_country" / f"{asset}_flood_scenarios.png",
                    scale, f"Flood + coastal scenarios by country - {asset}")
                n_figs += 1
        print(f"  {scale}: figures written")
    print(f"{n_figs} figure(s) written under {odir}")


if __name__ == "__main__":
    main()
