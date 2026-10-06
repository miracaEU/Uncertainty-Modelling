"""Maps for the protection-range tests: who was tested, and where it mattered.

Two figures, both written next to the range-test pies:

    range_test_coverage.png    which countries the range tests cover, and which
                               of them carry the coastal twin as well
    range_test_stability.png   per country, the share of asset types whose
                               dominant parameter is UNCHANGED across the four
                               protection treatments

The second is the one worth reading. The four treatments differ only in how the
protection standard is sampled - replaced by a draw from RP 5-200, from RP
10-100, multiplied by a factor in 0.5-1.5, or held at the recorded FLOPROS /
COASTPROS value - so a country where the dominant parameter is the same in all
four is one whose answer does not depend on that choice, and a country where it
flips is one whose headline result is an artefact of the bracket.

Only asset types that yield a real driver in all four treatments are counted. A
combination with no variance to attribute (no exposure, or exposed but never
damaged) is not a driver and would otherwise register as spurious agreement;
the per-country label reports how many assets survived that filter.

Usage:
    python -m src.plot_range_maps
    python -m src.plot_range_maps --out-dir overview_figures/report/range_tests
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

from .curves import LANDLOCKED
from .plot_drivers import (
    EXPOSED_UNDAMAGED,
    NO_DAMAGE,
    NO_EXPOSURE,
    exposure_flags,
    ranked_drivers,
)
from .plot_maps import (
    BORDER,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    _extent,
    _figure,
    _legend,
    _panel,
    load_geometry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT_ROOT / "MIRACA_uncertainty_study_summary.xlsx"

# Default subset; override with --countries as the tests are extended.
# Coastal panels drop whichever of these are landlocked automatically.
TEST_COUNTRIES = ["ITA", "NLD", "ROU", "AUT", "HUN", "EST", "ALB"]

RIVER_SET = ["flood_absprot_ds", "flood_absprot_ds_rp10to100",
             "flood_baseline_ds_sc0p5to1p5", "flood_noprot_ds"]
COASTAL_SET = ["coastal_absprot_ds", "coastal_absprot_ds_rp10to100",
               "coastal_baseline_ds_sc0p5to1p5", "coastal_noprot_ds"]

# The same sets without the "as recorded" treatment. Dropping it isolates
# bracket sensitivity: where protection is not sampled at all it cannot be a
# driver, so any pair it dominated elsewhere is forced to change for a
# structural reason rather than because the bracket mattered.
RIVER_SAMPLED = [s for s in RIVER_SET if "noprot" not in s]
COASTAL_SAMPLED = [s for s in COASTAL_SET if "noprot" not in s]

NON_DRIVER = {NO_EXPOSURE, EXPOSED_UNDAMAGED, NO_DAMAGE}

# Coverage map
BOTH_FILL = "#1c5cab"        # river and coastal tests
RIVER_ONLY_FILL = "#86b6ef"  # river only, i.e. landlocked
OUT_FILL = "#e4e3dc"         # study country, not in the test subset

# Stability map: share of assets whose top driver is unchanged across all four
# treatments. Diverging on purpose - the interesting end is the low one.
STABILITY_BINS = [(0.0, "#b2182b"), (0.25, "#ef8a62"), (0.5, "#fddbc7"),
                  (0.75, "#a6d3e8"), (1.0, "#1c5cab")]


def stability_color(share: float) -> str:
    """Colour for a share in [0, 1]; the last bin whose edge it reaches."""
    out = STABILITY_BINS[0][1]
    for edge, col in STABILITY_BINS:
        if share >= edge:
            out = col
    return out


def driver_table(workbook: Path) -> pd.DataFrame:
    """Rank-1 driver category per (country, asset, scenario)."""
    ranked = ranked_drivers(workbook, max_rank=1, exposure=exposure_flags())
    top = ranked[ranked["rank"] == 1]
    return top[["country", "asset", "scenario", "category"]]


def stability(top: pd.DataFrame, scenarios: list[str],
              countries: list[str]) -> pd.DataFrame:
    """Per country: assets compared, and how many keep the same rank-1 driver.

    An asset counts only if all four treatments produced a real driver for it.
    """
    sub = top[top["scenario"].isin(scenarios) & top["country"].isin(countries)]
    wide = sub.pivot_table(index=["country", "asset"], columns="scenario",
                           values="category", aggfunc="first")
    have_all = wide.notna().all(axis=1)
    is_driver = ~wide.isin(NON_DRIVER).any(axis=1)
    keep = wide[have_all & is_driver]
    same = keep.nunique(axis=1) == 1

    rows = []
    for c in countries:
        if c not in keep.index.get_level_values(0):
            rows.append({"country": c, "n": 0, "same": 0, "share": np.nan})
            continue
        m = same.loc[c]
        rows.append({"country": c, "n": int(len(m)), "same": int(m.sum()),
                     "share": float(m.mean())})
    return pd.DataFrame(rows)


def coverage_figure(geo, study: list[str], out_dir: Path,
                    tested: list[str], assets: dict[str, int] | None = None) -> Path:
    coastal = [c for c in tested if c not in LANDLOCKED]
    # Only STUDY countries get a fill. Anything else renders as context: passing
    # a fill for every GISCO country made _panel draw its micro-state square
    # marker for each one, scattering grey squares across the Atlantic.
    fills = {}
    for c in study:
        if c in coastal:
            fills[c] = BOTH_FILL
        elif c in tested:
            fills[c] = RIVER_ONLY_FILL
        else:
            fills[c] = OUT_FILL

    extent = _extent(geo, study)
    fig, axes = _figure(1, aspect=1.0, max_cols=1)
    _panel(axes[0], geo, fills, extent, labels=False)
    halo = [pe.withStroke(linewidth=2.0, foreground="#00000055")]
    for iso in tested:
        pt = geo.loc[iso, "point"]
        axes[0].annotate(iso, (pt.x, pt.y), ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color="#ffffff",
                         zorder=6, path_effects=halo)
    axes[0].set_title("Countries covered by the protection-range tests",
                      fontsize=11, fontweight="bold", color=INK, pad=6)

    handles = [
        Patch(facecolor=BOTH_FILL, edgecolor=BORDER,
              label=f"River and coastal tests ({len(coastal)})"),
        Patch(facecolor=RIVER_ONLY_FILL, edgecolor=BORDER,
              label=f"River only - landlocked ({len(tested) - len(coastal)})"),
        Patch(facecolor=OUT_FILL, edgecolor=BORDER,
              label=f"Study country, not tested ({len(study) - len(tested)})"),
    ]
    # Not every country carries every asset - Iceland has no gas exposure -
    # so the caption is built from what is actually there rather than asserting ten.
    note = "All ten asset types in every tested country."
    if assets:
        full = max(assets.values())
        short = sorted(c for c in tested if assets.get(c, full) < full)
        if short:
            bits = ", ".join(f"{c}: {assets[c]}" for c in short)
            note = f"{full} asset types per tested country, except {bits}."
    _legend(fig, handles, ncol=3, subtitle=note)
    out = out_dir / "range_test_coverage.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def stability_figure(top: pd.DataFrame, geo, out_dir: Path, river: list[str],
                     coastal: list[str], suptitle: str, treatments: str,
                     name: str, tested: list[str]) -> Path:
    panels = [("River flood", river, tested),
              ("Coastal flood", coastal,
               [c for c in tested if c not in LANDLOCKED])]
    extent = _extent(geo, tested)
    fig, axes = _figure(2, aspect=1.0, max_cols=2)

    for ax, (title, scens, countries) in zip(axes, panels):
        st = stability(top, scens, countries)
        # Tested countries only, so no micro-state square is drawn for the rest.
        fills = {}
        for c in countries:
            row = st[st["country"] == c]
            share = float(row["share"].iloc[0]) if len(row) else np.nan
            fills[c] = stability_color(share) if np.isfinite(share) else "#ffffff"
        _panel(ax, geo, fills, extent, labels=False)
        for _, r in st.iterrows():
            pt = geo.loc[r["country"], "point"]
            txt = f"{r['country']}\n{int(r['same'])}/{int(r['n'])}" if r["n"] else f"{r['country']}\n-"
            ax.annotate(txt, (pt.x, pt.y), ha="center", va="center",
                        fontsize=7, fontweight="bold", color=INK, zorder=6,
                        path_effects=None)
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=6)

    handles = [Patch(facecolor=c, edgecolor=BORDER,
                     label={0.0: "0-24%", 0.25: "25-49%", 0.5: "50-74%",
                            0.75: "75-99%", 1.0: "100%"}[e])
               for e, c in STABILITY_BINS]
    handles.append(Patch(facecolor="#ffffff", edgecolor=BORDER, label="no asset comparable"))
    _legend(fig, handles, ncol=6,
            subtitle=f"Labels: asset types unchanged / compared. Treatments: {treatments}.")
    fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", color=INK)
    out = out_dir / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: overview_figures/report/range_tests")
    ap.add_argument("--countries", nargs="+", default=None,
                    help=f"tested subset (default: {' '.join(TEST_COUNTRIES)})")
    args = ap.parse_args()
    tested = args.countries or TEST_COUNTRIES

    workbook = Path(args.workbook) if args.workbook else WORKBOOK
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "overview_figures" / "report" / "range_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    })

    geo = load_geometry()
    top = driver_table(workbook)
    study = sorted(set(top["country"]) & set(geo.index))

    assets = top.groupby("country")["asset"].nunique().to_dict()
    print(f"wrote {coverage_figure(geo, study, out_dir, tested, assets).name}  "
          f"({len(tested)} tested of {len(study)} study countries)")

    coastal_countries = [c for c in tested if c not in LANDLOCKED]
    variants = [
        ("all four treatments", RIVER_SET, COASTAL_SET,
         "Does the dominant parameter survive a change of protection bracket?",
         "RP 5-200, RP 10-100, x0.5-1.5, as recorded",
         "range_test_stability.png"),
        ("the three sampled treatments", RIVER_SAMPLED, COASTAL_SAMPLED,
         "Bracket sensitivity alone: protection sampled, only the range differs",
         "RP 5-200, RP 10-100, x0.5-1.5",
         "range_test_stability_sampled.png"),
    ]
    for head, river, coastal, suptitle, treatments, name in variants:
        print(f"\n{head}:")
        for label, scens, countries in (("river", river, tested),
                                        ("coastal", coastal, coastal_countries)):
            st = stability(top, scens, countries)
            tot_n, tot_same = int(st["n"].sum()), int(st["same"].sum())
            pct = f"{100 * tot_same / tot_n:.0f}%" if tot_n else "-"
            detail = "  ".join(f"{r['country']} {int(r['same'])}/{int(r['n'])}"
                               for _, r in st.iterrows())
            print(f"  {label:8s} {tot_same}/{tot_n} pairs keep the same driver ({pct})")
            print(f"           {detail}")
        out = stability_figure(top, geo, out_dir, river, coastal,
                               suptitle, treatments, name, tested)
        print(f"  wrote {out.name}")
    print(f"\nFigures in {out_dir}")


if __name__ == "__main__":
    main()
