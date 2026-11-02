"""What each modelled scenario was for: an overview of the study's design.

The result figures show what the scenarios FOUND. This one shows why each was
run, which is otherwise only recoverable by reading src/ema_model.py. Three
groups, and the distinction between them matters when reading any other figure:

  headline      full country coverage, the European damage estimates
  future        full coverage, adds sampled sea-level rise
  range test    a DELIBERATE country subset, run to test how much a factor's
                assumed range drives its apparent importance. These must never
                be summed onto the same axis as a full-coverage scenario - the
                denominator differs - which src/eu_totals.py now enforces.

Coverage is read from overview_figures/ead_ranges/EAD_Ranges.csv, so the table
cannot drift from what was actually run.

Usage:
    python -m src.plot_scenario_scope
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from .plot_maps import RANGES_CSV
from .plot_pyramid import INK, INK_2, MUTED, PROJECT_ROOT, scenario_label

PAGE = "#ffffff"
HAZARD_COLOUR = {"flood": "#2e9e74", "coastal": "#4a8fd4",
                 "earthquake": "#e2663f", "windstorm": "#e8a62e"}
BAND = "#f2f1ec"

# (scenario, protection treatment, sea-level rise, purpose)
DESIGN = [
    ("HEADLINE - full coverage, both flood families carrying climate uncertainty", [
        ("flood_absprot_ds", "sampled, RP 5-200 yr", "-",
         "Headline river flood estimate. Protection drawn as an absolute return "
         "period - the only treatment that can vary protection where the recorded "
         "standard is zero, since a multiplier leaves zero at zero."),
        ("flood_noprot_ds", "held at recorded", "-",
         "Isolates the non-protection factors. With protection fixed at the "
         "recorded standard, the remaining spread is curves, cost, warming and "
         "hazard-map bias alone."),
        ("coastal_noprot_ds_slr", "held at recorded", "sampled",
         "Primary coastal estimate. Protection taken as given, sea-level rise sampled "
         "over present day plus three SSPs across three horizons. Resolves on 98% of "
         "combinations, and is the case in which warming emerges as the leading "
         "coastal uncertainty."),
        ("coastal_absprot_ds_slr", "sampled, RP 5-200 yr", "sampled",
         "Supporting sensitivity, with protection and sea level both unknown. Read "
         "with care: 30% of its combinations did not resolve, and the 5-200 yr range "
         "sits below the recorded coastal standard in the most exposed country."),
        ("earthquake", "none (as in reference)", "-",
         "Headline seismic estimate. No design standard is applied, matching the "
         "reference assessment, so damage is driven by curves, cost and "
         "ground-motion bias."),
        ("windstorm", "fixed RP50 design", "-",
         "Headline wind estimate at the IEC design standard. Skipped for roads and "
         "ports, whose only wind curve is identically zero."),
        ("windstorm_absprot", "sampled, RP 25-200 yr", "-",
         "Tests how much the assumed wind design standard matters, by sampling it "
         "instead of holding it at RP50."),
    ]),
    ("REFERENCE COMPARISON - present-day coastal, for the deterministic benchmark", [
        ("coastal_noprot_ds", "held at recorded", "-",
         "Present-day sea level. Retained because the deterministic benchmark is a "
         "current-climate estimate, so this is the only like-for-like coastal "
         "comparison against it - not a headline result."),
        ("coastal_absprot_ds", "sampled, RP 5-200 yr", "-",
         "Present-day counterpart of the sampled-protection case. Five coastal "
         "countries - BGR, CYP, ISL, MLT, ROU - record no standard at all, so the "
         "absolute treatment is the only one that can protect them."),
    ]),
    ("RANGE TESTS - deliberate country subset, NOT damage estimates", [
        ("flood_absprot_ds_rp10to100", "sampled, RP 10-100 yr", "-",
         "Narrows the sampled protection range from 5-200 to 10-100 years. Tests "
         "how far a factor's apparent importance is an artefact of the range it "
         "was given rather than a property of the system."),
        ("coastal_absprot_ds_rp10to100", "sampled, RP 10-100 yr", "-",
         "The same narrowing applied to the coastal protection standard."),
        ("flood_baseline_ds_sc0p5to1p5", "scaled 0.5-1.5x recorded", "-",
         "Perturbs the recorded standard rather than replacing it, testing the "
         "same question with a treatment anchored to values countries have built."),
        ("coastal_baseline_ds_sc0p5to1p5", "scaled 0.5-1.5x recorded", "-",
         "The anchored treatment applied to the coastal protection standard."),
    ]),
]

COLS = [("Scenario", 0.030), ("Protection standard", 0.255),
        ("Sea-level rise", 0.410), ("Coverage", 0.510), ("What it was run for", 0.620)]
WRAP = 60


def coverage(ranges_csv: Path) -> dict[str, str]:
    df = pd.read_csv(ranges_csv)
    g = df.groupby("scenario").agg(c=("country", "nunique"), n=("country", "size"))
    return {s: f"{int(r.c)} countries\n{int(r.n)} combinations" for s, r in g.iterrows()}


def hazard_of_scenario(s: str) -> str:
    for h in ("coastal", "flood", "earthquake", "windstorm"):
        if s.startswith(h):
            return h
    return "flood"


def plot(cov: dict[str, str], out: Path) -> None:
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
                         "figure.facecolor": PAGE, "savefig.facecolor": PAGE,
                         "axes.facecolor": PAGE, "text.color": INK})
    lines = []
    for title, items in DESIGN:
        lines.append(("group", title))
        for scen, prot, slr, why in items:
            lines.append(("row", (scen, prot, slr, cov.get(scen, "not run"),
                                  textwrap.wrap(why, WRAP))))

    # 11pt at 1.45 linespacing needs ~0.222 in per line; 0.26 leaves margin so a
    # four-line description cannot bleed into the row beneath it.
    row_h = [0.44 if k == "group" else 0.26 * max(len(v[4]), 2) + 0.24
             for k, v in lines]
    height = sum(row_h) + 2.2
    fig, ax = plt.subplots(figsize=(17.5, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, height)
    ax.axis("off")

    y = height - 1.15
    for name, x in COLS:
        ax.text(x, y + 0.20, name, fontsize=12.5, fontweight="bold", color=INK, va="bottom")
    ax.plot([0.020, 0.985], [y + 0.10, y + 0.10], color=INK_2, linewidth=1.2)

    for (kind, payload), h in zip(lines, row_h):
        if kind == "group":
            y -= h
            ax.add_patch(Rectangle((0.020, y + 0.04), 0.965, h - 0.10,
                                   facecolor=BAND, edgecolor="none"))
            ax.text(0.030, y + h / 2 - 0.02, payload, fontsize=12,
                    fontweight="bold", color=INK, va="center")
            continue
        scen, prot, slr, cvg, body = payload
        y -= h
        col = HAZARD_COLOUR[hazard_of_scenario(scen)]
        ax.add_patch(Rectangle((0.0205, y + 0.05), 0.0045, h - 0.12,
                               facecolor=col, edgecolor="none"))
        top = y + h - 0.14
        ax.text(COLS[0][1], top, scenario_label(scen, short=True).replace(" (", "\n("),
                fontsize=11, color=INK, va="top", linespacing=1.4)
        ax.text(COLS[1][1], top, prot, fontsize=11, color=INK_2, va="top")
        ax.text(COLS[2][1], top, slr, fontsize=11, color=INK_2, va="top")
        ax.text(COLS[3][1], top, cvg, fontsize=11, color=INK_2, va="top", linespacing=1.4)
        ax.text(COLS[4][1], top, "\n".join(body), fontsize=11, color=INK, va="top",
                linespacing=1.45)

    ax.text(0.030, height - 0.60, "Modelled scenarios and what each was run for",
            fontsize=17, fontweight="bold", color=INK, va="bottom")
    ax.text(0.030, 0.25,
            "Range tests cover a deliberate subset of countries and are not European damage "
            "estimates; they are excluded from the totals figure, which compares only "
            "full-coverage scenarios.\n"
            "Coastal scenarios cover 26 of 36 countries because the remainder are landlocked. "
            "Windstorm is undefined for roads and ports, whose only wind curve is identically zero.",
            fontsize=10.5, color=MUTED, va="bottom", linespacing=1.55)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranges", default=None, help=f"default: {RANGES_CSV}")
    ap.add_argument("--out", default=None,
                    help="default: overview_figures/scope/Scenario_overview.png")
    args = ap.parse_args()

    csv = Path(args.ranges) if args.ranges else RANGES_CSV
    out = (Path(args.out) if args.out
           else PROJECT_ROOT / "overview_figures" / "scope" / "Scenario_overview.png")
    cov = coverage(csv)
    missing = [s for _, items in DESIGN for s, *_ in items if s not in cov]
    if missing:
        print(f"NOTE: no results yet for {missing} - shown as 'not run'")
    plot(cov, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
