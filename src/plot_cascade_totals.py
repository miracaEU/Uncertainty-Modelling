"""Compare pan-European cascade totals: by scenario, and by summation method.

Two things are drawn from the same cascade output.

CORRELATION-CORRECT ("cascade")
    The cascade draws the shared factors once and evaluates every country at
    that same draw before summing, so the eu_sum scope IS the European
    distribution. Mean, median and percentiles all describe one distribution.

SUM OF NATIONAL PERCENTILES ("summed")
    What src/eu_totals.py and the national range tables do: take each country's
    own percentile and add them up. That is the comonotonic case - it assumes
    every country sits at its own 5th (or 95th) percentile simultaneously.
    Drawn beside the correct answer, the gap between them IS the error that
    assumption costs.

The mean is identical under both by construction, since expectation is linear.
Any visible difference in the mean would be a bug, and is worth checking.

Usage:
    python -m src.plot_cascade_totals                     # scenarios, horizontal
    python -m src.plot_cascade_totals --orient v          # scenarios on the x-axis
    python -m src.plot_cascade_totals --orient v --compare-methods
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_pyramid import INK, INK_2, PROJECT_ROOT, scenario_label

PAGE = "#ffffff"
EUR_M_PER_BN = 1000.0
DEFAULT_DIR = PROJECT_ROOT / "results" / "cascade" / "custom-0ef4a71e"
DEFAULT_SCENARIOS = ["flood_absprot_ds", "flood_baseline_ds_sc0p5to1p5"]

METHOD_COLOURS = {"cascade": ("#2e9e74", "#a9ddc6"),
                  "summed": ("#b06a2c", "#e8c49e")}
METHOD_LABELS = {"cascade": "pan-European analysis",
                 "summed": "sum of national percentiles"}

HAZARD_FS, QUALIFIER_FS = 16.0, 11.5
AXIS_FS, TICK_FS, LEGEND_FS, TITLE_FS = 15.0, 13.0, 14.0, 17.0
VALUE_LABEL = "Expected annual damage (Billion Euros per year)"
STATS = ("p5", "p25", "median", "mean", "p75", "p95")


def _final_step(cdir: Path, asset: str, scenario: str) -> pd.DataFrame:
    path = cdir / f"{asset}_{scenario}_cascade.parquet"
    if not path.exists():
        raise SystemExit(f"No {path} - run `python -m src.cascade` for that combination.")
    df = pd.read_parquet(path)
    return df[df["step"] == int(df["step"].max())]


def _summary_from(values: np.ndarray) -> dict:
    return {"p5": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def summarise(cdir: Path, asset: str, scenarios: list[str]) -> pd.DataFrame:
    """One row per (scenario, method), in EUR bn/yr."""
    rows = []
    for scen in scenarios:
        last = _final_step(cdir, asset, scen)

        eu = last[last["scope"] == "eu_sum"]["total_EAD_MEUR"].to_numpy() / EUR_M_PER_BN
        rows.append({"scenario": scen, "method": "cascade",
                     "n": len(eu), **_summary_from(eu)})

        # Per country, take that country's own percentile, then add up. The
        # comonotonic construction, built from the same draws so the comparison
        # isolates the summation rule rather than any sampling difference.
        nat = last[last["scope"] == "country"]
        per = nat.groupby("country")["total_EAD_MEUR"]
        summed = {"p5": per.quantile(0.05).sum(), "p25": per.quantile(0.25).sum(),
                  "median": per.median().sum(), "mean": per.mean().sum(),
                  "p75": per.quantile(0.75).sum(), "p95": per.quantile(0.95).sum(),
                  "std": np.nan, "min": per.min().sum(), "max": per.max().sum()}
        rows.append({"scenario": scen, "method": "summed",
                     "n": int(nat["country"].nunique()),
                     **{k: (v / EUR_M_PER_BN if np.isfinite(v) else v)
                        for k, v in summed.items()}})
    return pd.DataFrame(rows)


def split_scenario(scenario: str) -> tuple[str, str]:
    label = scenario_label(scenario, short=True)
    head, sep, tail = label.partition(" (")
    return head, (sep.strip() + tail) if sep else ""


def _rc():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": PAGE, "savefig.facecolor": PAGE, "axes.facecolor": PAGE,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })


def _draw_one(ax, pos: float, r: pd.Series, vertical: bool, width_scale: float = 1.0):
    solid, pale = METHOD_COLOURS[r["method"]]
    lw_thin, lw_thick = 6 * width_scale, 17 * width_scale
    if vertical:
        ax.vlines(pos, r["p5"], r["p95"], color=pale, linewidth=lw_thin, zorder=3)
        ax.vlines(pos, r["p25"], r["p75"], color=solid, linewidth=lw_thick, zorder=4)
        for cap in ("p5", "p95"):
            ax.hlines(r[cap], pos - 0.07, pos + 0.07, color=solid, linewidth=2.2, zorder=5)
        ax.hlines(r["median"], pos - 0.11, pos + 0.11, color=INK, linewidth=3.0, zorder=7)
        ax.plot(pos, r["mean"], "o", markersize=10, color=INK,
                markeredgecolor=PAGE, markeredgewidth=1.6, zorder=8)
    else:
        ax.hlines(pos, r["p5"], r["p95"], color=pale, linewidth=lw_thin, zorder=3)
        ax.hlines(pos, r["p25"], r["p75"], color=solid, linewidth=lw_thick, zorder=4)
        for cap in ("p5", "p95"):
            ax.vlines(r[cap], pos - 0.17, pos + 0.17, color=solid, linewidth=2.2, zorder=5)
        ax.vlines(r["median"], pos - 0.30, pos + 0.30, color=INK, linewidth=3.0, zorder=7)
        ax.plot(r["mean"], pos, "o", markersize=10, color=INK,
                markeredgecolor=PAGE, markeredgewidth=1.6, zorder=8)


def _legend(fig, methods: list[str], y: float):
    handles = [Line2D([], [], color=INK, marker="o", linestyle="none",
                      markersize=10, label="mean"),
               Line2D([], [], color=INK, linewidth=3.0, label="median"),
               Line2D([], [], color=INK_2, linewidth=13, label="p25 to p75"),
               Line2D([], [], color="#c8c7c2", linewidth=5, label="p5 to p95")]
    if len(methods) > 1:
        handles += [Line2D([], [], color=METHOD_COLOURS[m][0], linewidth=13,
                           label=METHOD_LABELS[m]) for m in methods]
    # One row always. The figure width below scales with the number of entries
    # so that nothing is clipped.
    leg = fig.legend(handles=handles, frameon=False, ncol=len(handles),
                     loc="lower center", bbox_to_anchor=(0.5, y),
                     fontsize=LEGEND_FS, handlelength=2.2, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(INK)


def plot(summary: pd.DataFrame, asset: str, out: Path, vertical: bool) -> None:
    _rc()
    scenarios = list(dict.fromkeys(summary["scenario"]))
    methods = list(dict.fromkeys(summary["method"]))
    k = len(methods)
    # Paired bars share a scenario slot, so the two methods sit next to each
    # other and the gap between them is read directly.
    offs = ([0.0] if k == 1 else list(np.linspace(-0.19, 0.19, k)))

    # The single-row legend sets a width floor: six entries do not fit in the
    # width the bars alone would ask for, and the outer labels get clipped.
    n_leg = 4 + (2 if k > 1 else 0)
    min_w = 2.85 * n_leg
    if vertical:
        # Width scales with the TOTAL number of bars, not the scenario count,
        # or the paired version crowds.
        fig, ax = plt.subplots(figsize=(max(1.95 * len(scenarios) * k + 3.4, min_w), 9.4))
        # Bottom trimmed and the legend raised to meet it: the gap between the
        # scenario labels and the legend was most of the lower third.
        fig.subplots_adjust(left=0.095, right=0.985, top=0.90,
                            bottom=0.235 if k == 1 else 0.265)
    else:
        fig, ax = plt.subplots(figsize=(max(14.0, min_w),
                                        1.5 * len(scenarios) * k + 3.0))
        fig.subplots_adjust(left=0.235, right=0.985, top=0.88, bottom=0.215)

    base = np.arange(len(scenarios))
    if not vertical:
        base = base[::-1]

    for bi, scen in zip(base, scenarios):
        for off, m in zip(offs, methods):
            r = summary[(summary["scenario"] == scen) & (summary["method"] == m)]
            if r.empty:
                continue
            _draw_one(ax, bi + off, r.iloc[0], vertical,
                      width_scale=1.0 if k == 1 else 0.75)

        hz, qual = split_scenario(scen)
        if vertical:
            tr = ax.get_xaxis_transform()
            ax.annotate(hz, (bi, 0), xycoords=tr, xytext=(0, -18),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=HAZARD_FS, color=INK, annotation_clip=False)
            if qual:
                ax.annotate(qual, (bi, 0), xycoords=tr,
                            xytext=(0, -18 - HAZARD_FS * 1.45),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=QUALIFIER_FS, color=INK, annotation_clip=False)
        else:
            tr = ax.get_yaxis_transform()
            ax.annotate(hz, (0, bi), xycoords=tr, xytext=(-14, 2),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=HAZARD_FS, color=INK, annotation_clip=False)
            if qual:
                ax.annotate(qual, (0, bi), xycoords=tr, xytext=(-14, -4),
                            textcoords="offset points", ha="right", va="top",
                            fontsize=QUALIFIER_FS, color=INK, annotation_clip=False)

    top = float(summary["p95"].max()) * 1.06
    if vertical:
        ax.set_ylim(0, top)
        ax.set_ylabel(VALUE_LABEL, fontsize=AXIS_FS)
        ax.set_xticks(base, ["" for _ in base])
        ax.set_xlim(-0.72, len(scenarios) - 0.28)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=TICK_FS)
    else:
        ax.set_xlim(0, top)
        ax.set_xlabel(VALUE_LABEL, fontsize=AXIS_FS)
        ax.set_yticks(base, ["" for _ in base])
        ax.set_ylim(-0.72, len(scenarios) - 0.28)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.set_axisbelow(True)

    ax.set_title(f"Pan-European expected annual damage - {asset}, river flood",
                 loc="left", pad=16, fontsize=TITLE_FS, fontweight="bold", color=INK)
    _legend(fig, methods, 0.085)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_large(summary: pd.DataFrame, asset: str, out: Path, k: float = 1.7) -> None:
    """Vertical method comparison with the text `k` times larger, same width.

    The marks follow the network-study range figures so the two can sit side
    by side: mean = white diamond with a black edge, median = black dash,
    p25-p75 a solid bar, p5-p95 a pale band with caps. At this size the
    one-row legend no longer fits the width, so it takes two rows (the
    statistics, then the methods), and the value label is split over two lines.
    """
    from matplotlib.patches import Patch
    _rc()
    scenarios = list(dict.fromkeys(summary["scenario"]))
    methods = list(dict.fromkeys(summary["method"]))
    n = len(methods)
    offs = [0.0] if n == 1 else list(np.linspace(-0.19, 0.19, n))
    W = max(1.95 * len(scenarios) * n + 3.4, 2.85 * (4 + (2 if n > 1 else 0)))
    H = 10.6
    left_in, right_in, top_in, bot_in = 1.95, 0.25, 0.95, 2.75
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=left_in / W, right=1 - right_in / W,
                        top=1 - top_in / H, bottom=bot_in / H)
    ws = (1.0 if n == 1 else 0.75) * 1.3
    base = np.arange(len(scenarios))
    for bi, scen in zip(base, scenarios):
        for off, m in zip(offs, methods):
            r = summary[(summary["scenario"] == scen) & (summary["method"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            solid, pale = METHOD_COLOURS[m]
            pos = bi + off
            ax.vlines(pos, r["p5"], r["p95"], color=pale, linewidth=6 * ws, zorder=3)
            ax.vlines(pos, r["p25"], r["p75"], color=solid, linewidth=17 * ws, zorder=4)
            # Dash and caps sized in points, as in the network figures: just
            # wider than the bar, not a fixed share of the slot.
            for cap in ("p5", "p95"):
                ax.plot(pos, r[cap], "_", ms=22, mew=2.2 * 1.3, color=solid,
                        zorder=5)
            ax.plot(pos, r["median"], "_", ms=30, mew=3.0 * 1.3, color=INK,
                    zorder=7)
            ax.plot(pos, r["mean"], "D", ms=7 * 1.6, color=PAGE,
                    markeredgecolor=INK, markeredgewidth=1.4 * 1.6, zorder=8)
        hz, qual = split_scenario(scen)
        tr = ax.get_xaxis_transform()
        ax.annotate(hz, (bi, 0), xycoords=tr, xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=HAZARD_FS * k, color=INK, annotation_clip=False)
        if qual:
            ax.annotate(qual, (bi, 0), xycoords=tr,
                        xytext=(0, -14 - HAZARD_FS * k * 1.4),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=QUALIFIER_FS * k, color=INK, annotation_clip=False)

    ax.set_ylim(0, float(summary["p95"].max()) * 1.06)
    ax.set_ylabel("Expected annual damage\n(Billion Euros per year)",
                  fontsize=AXIS_FS * k * 0.95, linespacing=1.3)
    ax.set_xticks(base, ["" for _ in base])
    ax.set_xlim(-0.72, len(scenarios) - 0.28)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=TICK_FS * k)
    ax.set_axisbelow(True)
    fig.text(0.15 / W, 1 - 0.2 / H,
             f"Pan-European expected annual damage - {asset}, river flood",
             ha="left", va="top", fontsize=TITLE_FS * 1.4, fontweight="bold",
             color=INK)

    fs = LEGEND_FS * k
    stats = [Line2D([], [], color=INK, lw=3.0 * 1.3, label="median"),
             Line2D([], [], marker="D", ls="", ms=7 * 1.6, color=PAGE,
                    markeredgecolor=INK, markeredgewidth=1.4 * 1.6, label="mean"),
             Patch(facecolor=INK_2, label="p25 to p75"),
             Patch(facecolor="#c8c7c2", label="p5 to p95")]
    xc = 0.5 * (left_in / W + 1 - right_in / W)
    fig.legend(handles=stats, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(xc, 0.62 / H), fontsize=fs, handlelength=1.6,
               columnspacing=1.8)
    if n > 1:
        meth = [Patch(facecolor=METHOD_COLOURS[m][0], label=METHOD_LABELS[m])
                for m in methods]
        fig.legend(handles=meth, frameon=False, ncol=n, loc="lower center",
                   bbox_to_anchor=(xc, 0.08 / H), fontsize=fs,
                   handlelength=1.6, columnspacing=1.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cascade-dir", default=None, help=f"default: {DEFAULT_DIR}")
    ap.add_argument("--asset", default="power")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--orient", choices=["h", "v", "both"], default="both")
    ap.add_argument("--compare-methods", action="store_true",
                    help="also draw the sum-of-national-percentiles estimate beside "
                         "the correlation-correct one, so the error the summation "
                         "costs is visible directly.")
    ap.add_argument("--out-dir", default=None,
                    help="default: overview_figures/cascade_totals")
    args = ap.parse_args()

    cdir = Path(args.cascade_dir) if args.cascade_dir else DEFAULT_DIR
    odir = (Path(args.out_dir) if args.out_dir
            else PROJECT_ROOT / "overview_figures" / "cascade_totals")

    full = summarise(cdir, args.asset, args.scenarios)
    use = full if args.compare_methods else full[full["method"] == "cascade"]

    odir.mkdir(parents=True, exist_ok=True)
    tag = "_methods" if args.compare_methods else ""
    full.to_csv(odir / f"{args.asset}_scenario_totals.csv", index=False)
    for orient in (["h", "v"] if args.orient == "both" else [args.orient]):
        png = odir / f"{args.asset}_scenario_totals{tag}_{orient}.png"
        plot(use, args.asset, png, vertical=(orient == "v"))
        print(f"wrote {png}")
        if orient == "v":
            big = odir / f"{args.asset}_scenario_totals{tag}_v_largefont.png"
            plot_large(use, args.asset, big)
            print(f"wrote {big}")

    show = use.set_index(["scenario", "method"])[list(STATS)]
    print(f"\n{args.asset}, EUR bn per year")
    print(show.to_string(float_format=lambda v: f"{v:9.3f}"))


if __name__ == "__main__":
    main()
