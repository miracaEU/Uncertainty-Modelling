"""Plot every vulnerability/fragility curve the pipeline samples among.

One figure per hazard (flood, earthquake, windstorm), laid out as a grid of
subplots - one subplot per unique curve GROUP, since a curve group is exactly
one sampled uncertainty factor (curve_<group> in src/ema_model.py). Each
subplot draws every curve option in that group, so you can see the spread of
choices the Sobol/LHS runs pick among, and which assets/object types use it.

Curves are the same objects the model actually uses:
  flood       depth-damage fraction vs depth (m), F_Vuln_Depth sheet - ALSO
              used by coastal flood (coastal reuses the river flood curves).
  earthquake  expected damage ratio vs PGA (g), i.e. the EDR curve after the
              fragility -> loss collapse (src/curves.py::load_eq_edr_tables),
              which is what compute_risk interpolates - not the raw per-state
              fragility.
  windstorm   damage fraction vs 3-sec gust speed (m/s), W_Vuln_V10m_3sec sheet
              (airports/education/power only).

Output: results/figures/vulnerability_curves/{flood,earthquake,windstorm}_curves.png

Run:  python -m src.plot_curves
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .curves import ASSET_CONFIGS, load_eq_edr_tables, load_flood_curves, load_wind_curves
from .paths import load_config

# palette (dataviz reference instance, light mode - matches analyze.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)

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
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "font.size": 9,
    }
)

LEGEND_MAX = 8  # groups with more curves than this get a range annotation, not a legend


def unique_groups(hazard: str) -> list[dict]:
    """Deduplicate curve groups across all assets by their curve-id tuple.

    Returns one entry per distinct group: its ordered curve ids and, per asset,
    the object types that use it - so a group shared by several assets is drawn
    once but still credited to all of them.
    """
    attr = {"flood": "flood", "earthquake": "eq", "windstorm": "wind"}[hazard]
    groups: dict[tuple, dict] = {}
    for asset, cfg in ASSET_CONFIGS.items():
        gdict = getattr(cfg, f"{attr}_groups")
        objmap = getattr(cfg, f"{attr}_object_group")
        for gname, curves in gdict.items():
            key = tuple(curves)
            members = sorted(o for o, g in objmap.items() if g == gname)
            entry = groups.setdefault(key, {"curves": list(curves), "assets": {}})
            entry["assets"].setdefault(asset, []).extend(members)
    # sort groups by their first curve id for a stable, readable layout
    return [groups[k] for k in sorted(groups, key=lambda t: t[0])]


def _colors(n: int) -> list:
    if n == 1:
        return [SEQ_CMAP(0.55)]
    return [SEQ_CMAP(v) for v in np.linspace(0.15, 0.95, n)]


def _assets_caption(assets: dict) -> str:
    return "used by: " + "; ".join(
        f"{a} ({'/'.join(objs) if len(objs) <= 4 else '/'.join(objs[:4]) + f' +{len(objs) - 4}'})"
        for a, objs in assets.items()
    )


def plot_hazard(hazard: str, curve_xy, xlabel: str, ylabel: str, fig_dir: Path) -> Path:
    """curve_xy: callable curve_id -> (x_array, y_array)."""
    groups = unique_groups(hazard)
    n = len(groups)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows), layout="constrained", squeeze=False
    )
    for idx, entry in enumerate(groups):
        ax = axes.ravel()[idx]
        curves = entry["curves"]
        colors = _colors(len(curves))
        for cid, color in zip(curves, colors):
            x, y = curve_xy(cid)
            ax.plot(x, y, color=color, linewidth=1.8, label=cid)
        ax.set_title(f"{curves[0]} group  ({len(curves)} curve{'s' if len(curves) != 1 else ''})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(x=0.02)
        ax.set_ylim(-0.03, 1.03)
        if len(curves) <= LEGEND_MAX:
            # "best" lets matplotlib place the legend where it overlaps the
            # curves least (these are steep S-curves, so a fixed corner often
            # sits right on top of them); semi-opaque so any unavoidable
            # overlap stays readable.
            leg = ax.legend(frameon=True, framealpha=0.75, edgecolor="none",
                            facecolor=SURFACE, fontsize=7.5, loc="best", ncol=1)
            for t in leg.get_texts():
                t.set_color(INK_2)
        else:
            ax.text(0.97, 0.05, f"{curves[0]} … {curves[-1]}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(0.5, -0.32, _assets_caption(entry["assets"]), transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=MUTED, wrap=True)
    for j in range(n, nrows * ncols):
        axes.ravel()[j].set_visible(False)

    subtitle = {
        "flood": "River flood depth-damage curves (also used by coastal flood)",
        "earthquake": "Earthquake expected-damage-ratio curves (fragility -> EDR)",
        "windstorm": "Windstorm speed-damage curves",
    }[hazard]
    fig.suptitle(
        f"{subtitle}\nevery curve group sampled across all assets (one subplot = one curve_* factor)",
        color=INK, fontsize=13,
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{hazard}_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazards", nargs="+", default=["flood", "earthquake", "windstorm"],
                        choices=["flood", "earthquake", "windstorm"])
    args = parser.parse_args()

    cfg = load_config()
    fig_dir = cfg["results_dir"] / "figures" / "vulnerability_curves"

    # Gather every curve id used by each hazard, then load once.
    def all_ids(hazard):
        ids = set()
        for entry in unique_groups(hazard):
            ids.update(entry["curves"])
        return sorted(ids)

    saved = []
    if "flood" in args.hazards:
        df = load_flood_curves(cfg["vulnerability_path"], all_ids("flood"))
        xy = lambda cid: (df.index.to_numpy(float), df[cid].to_numpy(float))  # noqa: E731
        saved.append(plot_hazard("flood", xy, "water depth (m)", "damage fraction", fig_dir))

    if "earthquake" in args.hazards:
        tables = load_eq_edr_tables(cfg["fragility_path"], all_ids("earthquake"))
        xy = lambda cid: tables[cid]  # noqa: E731
        saved.append(plot_hazard("earthquake", xy, "PGA (g)", "expected damage ratio", fig_dir))

    if "windstorm" in args.hazards:
        df = load_wind_curves(cfg["vulnerability_path"], all_ids("windstorm"))
        xy = lambda cid: (df.index.to_numpy(float), df[cid].to_numpy(float))  # noqa: E731
        saved.append(plot_hazard("windstorm", xy, "3-sec gust speed (m/s)", "damage fraction", fig_dir))

    print(f"Saved {len(saved)} figure(s) to {fig_dir}:")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
