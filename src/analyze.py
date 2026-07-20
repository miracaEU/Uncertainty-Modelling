"""Analyze EMA Workbench experiment results: which factors drive flood risk?

Produces (in results/figures/):
  feature_scores.png     — extra-trees feature importance, factors x outcomes
  ead_by_warming.png     — total EAD distribution per warming level
  ead_vs_protection.png  — total EAD vs protection scale, colored by warming
  ead_vs_depth_offset.png— total EAD vs depth offset, split by aggregation choice
plus feature_scores.csv and a text summary on stdout.

Usage:
    python -m src.analyze                # newest results file
    python -m src.analyze --results results/experiments_... .tar.gz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ema_workbench import load_results
from ema_workbench.analysis import feature_scoring
from matplotlib.colors import LinearSegmentedColormap

from .paths import load_config, set_country_override

# --- palette (dataviz reference instance, light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
GREEN = "#008300"
# ordinal blue ramp (steps 250..650) for the 5 warming levels, light mode
WARMING_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
WARMING_ORDER = ["current", "1.5C", "2.0C", "3.0C", "4.0C"]
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)

FACTORS = [
    "warming",
    "curve_main",
    "curve_other",
    "eq_curve",
    "cost_level",
    "protection_scale",
    "depth_offset",
    "pga_scale",
    "aggregation",
]

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


def newest_results(cfg: dict) -> Path:
    pattern = f"experiments_{cfg['country']}_{cfg['asset_type']}_lhs_*.tar.gz"
    files = sorted(cfg["results_dir"].glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No {pattern} in {cfg['results_dir']}")
    return files[-1]


def plot_feature_scores(experiments, outcomes, fig_dir: Path, prefix: str) -> pd.DataFrame:
    x = experiments[FACTORS]
    scores = feature_scoring.get_feature_scores_all(x, outcomes)
    scores = scores.loc[FACTORS]

    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout="constrained")
    mat = scores.to_numpy()
    im = ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=max(0.5, mat.max()), aspect="auto")
    ax.set_xticks(range(scores.shape[1]), scores.columns, rotation=30, ha="right")
    ax.set_yticks(range(scores.shape[0]), scores.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center", fontsize=9,
                color="#ffffff" if v > 0.55 * max(0.5, mat.max()) else INK,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("feature importance (extra-trees)", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title(f"Which uncertainty factors drive each outcome? ({prefix})")
    fig.savefig(fig_dir / f"{prefix}_feature_scores.png", dpi=200)
    plt.close(fig)
    return scores


def plot_ead_by_warming(experiments, outcomes, fig_dir: Path, prefix: str) -> None:
    ead = np.asarray(outcomes["total_EAD_MEUR"])
    warming = experiments["warming"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    rng = np.random.default_rng(1)
    for i, level in enumerate(WARMING_ORDER):
        vals = ead[warming == level]
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            s=9, color=WARMING_RAMP[i], alpha=0.45, linewidths=0, zorder=2,
        )
        med = np.median(vals)
        ax.hlines(med, i - 0.28, i + 0.28, color=INK, linewidth=2, zorder=3)
        ax.annotate(
            f"{med:.2f}", (i + 0.32, med), fontsize=9, color=INK_2, va="center"
        )
    ax.set_xticks(range(len(WARMING_ORDER)), WARMING_ORDER)
    ax.set_ylabel("total EAD (M EUR / yr)")
    ax.set_xlabel("global warming level")
    ax.set_title(f"Expected annual damage by warming level, {prefix} (median marked)")
    ax.grid(axis="x", visible=False)
    fig.savefig(fig_dir / f"{prefix}_ead_by_warming.png", dpi=200)
    plt.close(fig)


def plot_ead_vs_protection(experiments, outcomes, fig_dir: Path, prefix: str) -> None:
    ead = np.asarray(outcomes["total_EAD_MEUR"])
    prot = experiments["protection_scale"].to_numpy(float)
    warming = experiments["warming"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    for i, level in enumerate(WARMING_ORDER):
        m = warming == level
        ax.scatter(
            prot[m], ead[m], s=10, color=WARMING_RAMP[i], alpha=0.55,
            linewidths=0, label=level,
        )
    leg = ax.legend(
        title="warming", frameon=False, loc="upper right", fontsize=9, title_fontsize=9
    )
    for t in leg.get_texts():
        t.set_color(INK_2)
    leg.get_title().set_color(INK_2)
    ax.set_xlabel("protection standard scale (x FLOPROS design RP)")
    ax.set_ylabel("total EAD (M EUR / yr)")
    ax.set_title(f"EAD vs protection scaling ({prefix})")
    fig.savefig(fig_dir / f"{prefix}_ead_vs_protection.png", dpi=200)
    plt.close(fig)


def plot_ead_vs_depth_offset(experiments, outcomes, fig_dir: Path, prefix: str) -> None:
    ead = np.asarray(outcomes["total_EAD_MEUR"])
    off = experiments["depth_offset"].to_numpy(float)
    agg = experiments["aggregation"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    for label, color in (("per_cell", BLUE), ("mean_depth", GREEN)):
        m = agg == label
        ax.scatter(off[m], ead[m], s=10, color=color, alpha=0.5, linewidths=0, label=label)
        # binned median line to make the trend readable
        bins = np.linspace(-0.5, 0.5, 11)
        mids, meds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = m & (off >= lo) & (off < hi)
            if sel.sum() > 5:
                mids.append(0.5 * (lo + hi))
                meds.append(np.median(ead[sel]))
        ax.plot(mids, meds, color=color, linewidth=2)
    leg = ax.legend(title="aggregation", frameon=False, loc="upper left", fontsize=9,
                    title_fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    leg.get_title().set_color(INK_2)
    ax.set_xlabel("water depth offset (m)")
    ax.set_ylabel("total EAD (M EUR / yr)")
    ax.set_title(f"Depth uncertainty and aggregation choice, {prefix} (binned medians)")
    fig.savefig(fig_dir / f"{prefix}_ead_vs_depth_offset.png", dpi=200)
    plt.close(fig)


def print_summary(experiments, outcomes, scores: pd.DataFrame) -> None:
    ead = np.asarray(outcomes["total_EAD_MEUR"])
    print("\n" + "=" * 70)
    print(f"Total EAD across {len(ead)} experiments (M EUR / yr)")
    print("=" * 70)
    q = np.percentile(ead, [5, 25, 50, 75, 95])
    print(
        f"  p5 {q[0]:8.3f} | p25 {q[1]:8.3f} | median {q[2]:8.3f} "
        f"| p75 {q[3]:8.3f} | p95 {q[4]:8.3f}"
    )
    for hz_outcome in ("EAD_river_MEUR", "EAD_earthquake_MEUR"):
        if hz_outcome in outcomes:
            v = np.asarray(outcomes[hz_outcome])
            print(
                f"  {hz_outcome:22s} median {np.median(v):8.3f} "
                f"| p5 {np.percentile(v, 5):8.3f} | p95 {np.percentile(v, 95):8.3f}"
            )
    print("\nRanked drivers of total EAD (extra-trees importance):")
    ranked = scores["total_EAD_MEUR"].sort_values(ascending=False)
    for name, v in ranked.items():
        print(f"  {name:18s} {v:.3f}")
    print("\nMedian total EAD per categorical level:")
    for factor in ("warming", "aggregation", "curve_main", "curve_other", "eq_curve"):
        vals = experiments[factor].astype(str)
        med = pd.Series(ead).groupby(vals.reset_index(drop=True)).median()
        parts = ", ".join(f"{k}={v:.2f}" for k, v in med.items())
        print(f"  {factor:12s} {parts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--country", default=None, help="ISO3 override of config country")
    args = parser.parse_args()

    set_country_override(args.country)
    cfg = load_config()
    prefix = cfg["country"]
    path = Path(args.results) if args.results else newest_results(cfg)
    print(f"Loading {path}")
    experiments, outcomes = load_results(path)

    fig_dir = cfg["results_dir"] / "figures"
    fig_dir.mkdir(exist_ok=True)

    scores = plot_feature_scores(experiments, outcomes, fig_dir, prefix)
    scores.to_csv(cfg["results_dir"] / f"{prefix}_feature_scores.csv")
    plot_ead_by_warming(experiments, outcomes, fig_dir, prefix)
    plot_ead_vs_protection(experiments, outcomes, fig_dir, prefix)
    plot_ead_vs_depth_offset(experiments, outcomes, fig_dir, prefix)
    print_summary(experiments, outcomes, scores)
    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
