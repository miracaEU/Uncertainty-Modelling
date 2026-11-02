"""Recorded flood protection standards, river and coastal, side by side.

Two sources, two native resolutions:

  river   FLOPROS (Scussolini et al., 2016), a 500 m raster that the model
          samples at asset centroids.
  coast   COASTPROS (van Maanen et al., 2025), tabulated per NUTS2 region.

The NUTS vintage matters. COASTPROS carries NUTS 2016 codes - including the
United Kingdom, and including Portuguese codes that 2021 renumbered - while the
model joins asset centroids to NUTS 2024. Rather than re-deriving that lookup
here and risking a figure that disagrees with the runs, the coastal panel calls
the model's own sample_coastal_protection() on one representative point per
region. What is drawn is therefore what the pipeline assigns.

    python -m src.plot_protection                  # native resolution (default)
    python -m src.plot_protection --level country  # the values the model applies
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .plot_maps import BORDER, CONTEXT_FILL, SPREAD_RAMP_SPECTRAL, _extent, _panel, load_geometry
from .plot_pyramid import INK, INK_2, MUTED, PROJECT_ROOT

PAGE = "#ffffff"
LINE_BREAK = chr(10)
CSV = PROJECT_ROOT / "overview_figures" / "scope" / "protection_standards.csv"

DATA = Path("S:/eks510/projects/AssetRisk_PanEU/data")
FLOPROS_RASTER = DATA / "floodProtection_v2019_paper3.tif"
RP_COL = "MODELLED RETURN PERIOD"       # best covered of the four

RIVER_REF = "River flood  -  based on FLOPROS (Scussolini et al., 2016)"
COAST_REF = "Coastal flood  -  based on COASTPROS (van Maanen et al., 2025)"

# The 26 countries the coastal scenarios actually run, as NUTS country codes.
# COASTPROS also lists ME and MK, which we never run, so they stay off the map.
COASTAL_RUN = {"AL", "BE", "BG", "CY", "DE", "DK", "EE", "EL", "ES", "FI", "FR",
               "HR", "IE", "IS", "IT", "LT", "LV", "MT", "NL", "NO", "PL", "PT",
               "RO", "SE", "SI", "UK"}
NUTS_TO_ISO3 = {"BG": "BGR", "CY": "CYP", "IS": "ISL", "RO": "ROU"}
# Finland's four regions carry a recorded policy standard (20 yr) where
# COASTPROS has no modelled one, so they are the one place this map visibly
# departs from the published modelled map. Marked apart so that difference
# reads as deliberate rather than as an error.
POLICY_HATCH = "...."
# Hatch colours are split by what they sit on. The country-mean regions take
# the mid and dark classes (Portugal's 100-200 blue, Germany, Norway), where a
# dark hatch disappears; Finland's policy regions are the pale "under 25 yr"
# yellow, where a white one would.
MEAN_HATCH_COLOR = "#ffffff"
POLICY_HATCH_COLOR = INK_2
# The country-mean key needs a mid-tone behind it or its white stripes vanish
# against the page.
MEAN_LEGEND_FACE = "#8a9099"

EDGES = (25.0, 50.0, 100.0, 200.0, 500.0)
LABELS = ("under 25 yr", "25 - 50 yr", "50 - 100 yr",
          "100 - 200 yr", "200 - 500 yr", "over 500 yr")
RAMP = SPREAD_RAMP_SPECTRAL

NONE_FILL = "#b9342e"
NA_FILL = "#e6e5df"
PARTIAL_HATCH = "////"
PARTIAL_MIN = 0.02
NONE_FRAC = 0.50

TITLE_FS, LABEL_FS, LEGEND_FS = 16.0, 14.0, 12.0
# The frame, in EPSG:3035 metres. The east edge reaches 6.58 M to take in
# Cyprus (6.34-6.53 M), which is one of the four countries we model as wholly
# unprotected and so needs to be visible. The west edge stops short of the
# Azores and Madeira, which would otherwise drag the frame 1.2 M west for two
# small islands.
VIEW = (2.45e6, 6.58e6, 1.35e6, 5.45e6)
# Sized so the panels nearly fill their boxes: with aspect="equal" a map only
# as wide as VIEW's 1.007 ratio leaves slack inside a wider box, and that slack
# shows up doubled as a gap down the middle of the figure.
FIGSIZE = (14.0, 9.9)


def classify(rp: float) -> int:
    return int(np.searchsorted(EDGES, rp, side="right"))


# ---------------------------------------------------------------------------
# Native-resolution figure
# ---------------------------------------------------------------------------


def coastal_applied():
    """Per-region applied RP, produced by the model's own function.

    The figure calls sample_coastal_protection() on one representative point
    per region rather than re-deriving the lookup, so it cannot drift from
    what the pipeline actually assigns. 'source' records where each number
    came from, which is what makes this comparable with the published map.
    """
    import geopandas as gpd

    from .coastal import (NUTS_SUCCESSORS, POLICY_OVERRIDES,
                          _read_nuts, sample_coastal_protection)
    from .paths import load_config

    ccfg = dict(load_config()["hazards"]["coastal"])
    regions = _read_nuts(ccfg["nuts2_path"])
    uk = _read_nuts(ccfg["nuts2_uk_path"])
    regions = pd.concat([regions, uk[uk["CNTR_CODE"] == "UK"]], ignore_index=True)
    regions = gpd.GeoDataFrame(regions, geometry="geometry", crs=3035)

    pts = regions.copy()
    pts["geometry"] = pts.geometry.representative_point()
    regions["rp"] = sample_coastal_protection(pts, ccfg)

    # Provenance, in the same precedence the model applies.
    cp = pd.read_excel(ccfg["coastpros_path"], sheet_name="COASTPROS-EU")
    cp.columns = [str(c).strip() for c in cp.columns]
    cp["NUTS2 ID"] = cp["NUTS2 ID"].astype(str).str.strip()
    cp[RP_COL] = pd.to_numeric(cp[RP_COL], errors="coerce")
    modelled = set(cp.loc[cp[RP_COL].notna(), "NUTS2 ID"])
    modelled |= {new for new, old in NUTS_SUCCESSORS.items() if old in modelled}

    def source(nid, rp):
        if nid in POLICY_OVERRIDES:
            return "policy"
        if nid in modelled:
            return "published"
        return "mean" if rp > 0 else "zero"

    regions["source"] = [source(n, r) for n, r in zip(regions["NUTS_ID"], regions["rp"])]

    # COASTPROS tabulates coastal regions only, so a region it does not list is
    # one it considers inland. The model still hands those a country mean, but
    # they carry no coastal hazard for it to bind to - drawing them would paint
    # Bavaria and the Massif Central as though they had sea defences.
    listed = set(cp["NUTS2 ID"])
    listed |= {new for new, old in NUTS_SUCCESSORS.items() if old in listed}
    on_coast = regions["NUTS_ID"].isin(listed)
    return regions[regions["CNTR_CODE"].isin(COASTAL_RUN) & on_coast].copy()


def read_flopros(max_px: int = 1600):
    """The raster, decimated on read so a 7781 x 7978 grid stays plottable."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(FLOPROS_RASTER) as src:
        step = max(1, int(max(src.width, src.height) / max_px))
        out_h, out_w = src.height // step, src.width // step
        # Nearest, not average: averaging return periods across a boundary
        # invents standards that no region has.
        arr = src.read(1, out_shape=(1, out_h, out_w),
                       resampling=Resampling.nearest).astype(float)
        b = src.bounds
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan
    return arr, (b.left, b.right, b.bottom, b.top)


def plot_native(out: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
                         "figure.facecolor": PAGE, "savefig.facecolor": PAGE,
                         "axes.facecolor": PAGE, "text.color": INK})

    arr, extent = read_flopros()
    g = coastal_applied()

    geo = load_geometry()
    frame = box(VIEW[0], VIEW[2], VIEW[1], VIEW[3])
    ctx = gpd.clip(geo[["geometry"]], frame)
    # Coastal-run countries COASTPROS does not cover at all. They have no
    # region to colour, and the model resolves every asset in them to 0, so
    # they are drawn whole rather than region by region.
    zero_cntr = sorted((COASTAL_RUN - set(g["CNTR_CODE"])) & set(NUTS_TO_ISO3))
    zero_iso3 = [NUTS_TO_ISO3[c] for c in zero_cntr]
    zero_geo = gpd.clip(geo.loc[[c for c in zero_iso3 if c in geo.index], ["geometry"]], frame)

    finite = np.isfinite(arr)
    riv_cls = np.full(arr.shape, np.nan)
    riv_cls[finite] = np.searchsorted(EDGES, arr[finite], side="right")
    river_zero = bool(np.any(arr[finite] == 0))
    riv_cls[finite & (arr == 0)] = -1        # 0 is "unprotected", not "under 25 yr"

    used = {int(c) for c in np.unique(riv_cls[np.isfinite(riv_cls)]) if c >= 0}
    used |= {classify(v) for v in g.loc[g["rp"] > 0, "rp"]}

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.115, wspace=0.01)

    # --- river: the raster itself
    ax = axes[0]
    cmap = ListedColormap([NONE_FILL] + list(RAMP))
    norm = BoundaryNorm([i + 0.5 for i in range(-2, len(RAMP))], cmap.N)
    ctx.plot(ax=ax, facecolor=NA_FILL, edgecolor="none", zorder=1)
    ax.imshow(riv_cls, extent=extent, origin="upper", cmap=cmap, norm=norm,
              interpolation="nearest", zorder=2)
    ctx.boundary.plot(ax=ax, color=BORDER, linewidth=0.3, zorder=3)

    # --- coast: NUTS2 polygons, coloured by the value the model applies
    ax = axes[1]
    ctx.plot(ax=ax, facecolor=NA_FILL, edgecolor="none", zorder=1)
    if len(zero_geo):
        zero_geo.plot(ax=ax, facecolor=NONE_FILL, edgecolor=BORDER,
                      linewidth=0.3, zorder=2)
    colours = [NONE_FILL if rp <= 0 else RAMP[classify(rp)] for rp in g["rp"]]
    g.plot(ax=ax, color=colours, edgecolor=BORDER, linewidth=0.2, zorder=3)
    fallback = g[g["source"] == "mean"]
    if len(fallback):
        fallback.plot(ax=ax, facecolor="none", edgecolor=MEAN_HATCH_COLOR,
                      linewidth=0.2, hatch=PARTIAL_HATCH, zorder=4)
    policy = g[g["source"] == "policy"]
    if len(policy):
        policy.plot(ax=ax, facecolor="none", edgecolor=POLICY_HATCH_COLOR,
                    linewidth=0.2, hatch=POLICY_HATCH, zorder=4)
    ctx.boundary.plot(ax=ax, color=BORDER, linewidth=0.3, zorder=5)

    for ax in axes:
        ax.set_xlim(VIEW[0], VIEW[1])
        ax.set_ylim(VIEW[2], VIEW[3])
        ax.set_aspect("equal")
        ax.set_axis_off()

    # Titles by hand rather than ax.set_title(), which anchors to each axes'
    # own box and put the two headings at visibly different heights on the
    # page. One y for both, taken after a draw so the aspect-adjusted boxes
    # are final.
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    title_y = max(b.y1 for b in boxes) + 0.012
    for box, label in zip(boxes, (RIVER_REF, COAST_REF)):
        fig.text(box.x0 + box.width / 2, title_y, label, ha="center", va="bottom",
                 fontsize=LABEL_FS, color=INK)

    ramp_h = [Patch(facecolor=RAMP[i], edgecolor=BORDER, linewidth=0.4, label=LABELS[i])
              for i in sorted(used)]
    note_h = []
    if river_zero or (g["rp"] <= 0).any() or len(zero_geo):
        note_h.append(Patch(facecolor=NONE_FILL, edgecolor=BORDER, linewidth=0.4,
                            label="no standard on record: assumed unprotected"))
    if len(fallback):
        note_h.append(Patch(facecolor=MEAN_LEGEND_FACE, edgecolor=MEAN_HATCH_COLOR,
                            linewidth=0.8, hatch=PARTIAL_HATCH,
                            label="no value: country mean assumed"))
    if len(policy):
        note_h.append(Patch(facecolor="#ffffff", edgecolor=POLICY_HATCH_COLOR,
                            linewidth=0.8, hatch=POLICY_HATCH,
                            label="from the COASTPROS policy layer"))
    note_h.append(Patch(facecolor=NA_FILL, edgecolor=BORDER, linewidth=0.4,
                        label="not covered"))

    leg1 = fig.legend(handles=ramp_h, frameon=False, ncol=len(ramp_h),
                      loc="lower center", bbox_to_anchor=(0.5, 0.060),
                      fontsize=LEGEND_FS, handlelength=1.8, columnspacing=1.5)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=note_h, frameon=False, ncol=len(note_h),
                      loc="lower center", bbox_to_anchor=(0.5, 0.012),
                      fontsize=LEGEND_FS, handlelength=1.8, columnspacing=1.5)
    for leg in (leg1, leg2):
        for t in leg.get_texts():
            t.set_color(INK)

    fig.suptitle("Flood protection standards as the analysis applies them",
                 fontsize=TITLE_FS, fontweight="bold", color=INK,
                 x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.945, LINE_BREAK.join([
        "Design return period as this study applies it, each source at its own resolution: "
        "FLOPROS as the 500 m raster sampled at asset centroids, COASTPROS as",
        "the NUTS2 table. The coastal panel is drawn by the pipeline's own assignment function, "
        "so it is what the runs use rather than a redrawing of the published map.",
        "COASTPROS reports no modelled return period for 45 of its 165 regions: four Finnish "
        "ones take the policy standard it records instead - the one place this",
        "map departs from the published one - and the rest their country mean. Bulgaria, Cyprus, "
        "Iceland and Romania are absent entirely and are assumed unprotected.",
    ]), fontsize=10.5, color=MUTED, va="top", linespacing=1.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  river classes present : {sorted(used)}")
    print(f"  coastal regions drawn : {len(g)}")
    for src, n in g["source"].value_counts().items():
        print(f"      {src:10s} {n}")
    print(f"  whole countries unprotected : {zero_iso3}")


# ---------------------------------------------------------------------------
# Country-level figure, from the values the model actually applies
# ---------------------------------------------------------------------------


def fills_for(df: pd.DataFrame, kind: str) -> tuple[dict, list, list]:
    fills, hatched, none_countries = {}, [], []
    med_col, none_col = f"{kind}_median", f"{kind}_none_frac"
    for r in df.to_dict("records"):
        c, med, nf = r["country"], r[med_col], r[none_col]
        if not np.isfinite(nf):
            continue
        if not np.isfinite(med) or nf >= NONE_FRAC:
            fills[c] = NONE_FILL
            none_countries.append(c)
            continue
        fills[c] = RAMP[classify(med)]
        if nf >= PARTIAL_MIN:
            hatched.append(c)
    return fills, hatched, none_countries


def plot_country(df: pd.DataFrame, geo, out: Path) -> None:
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
                         "figure.facecolor": PAGE, "savefig.facecolor": PAGE,
                         "axes.facecolor": PAGE, "text.color": INK})

    panels = [("river", RIVER_REF), ("coastal", COAST_REF)]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 9.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.115, wspace=0.01)

    used: set[int] = set()
    any_none = any_hatch = False
    for ax, (kind, title) in zip(axes, panels):
        fills, hatched, none_c = fills_for(df, kind)
        extent = _extent(geo, list(fills))
        _panel(ax, geo, fills, extent, hatched=hatched,
               hatch=PARTIAL_HATCH, hatch_color=INK_2, labels=True)
        ax.set_title(title, fontsize=LABEL_FS, color=INK, pad=10)
        med = df[f"{kind}_median"]
        used |= {classify(v) for v in med[np.isfinite(med)]}
        any_none |= bool(none_c)
        any_hatch |= bool(hatched)

    handles = [Patch(facecolor=RAMP[i], edgecolor=BORDER, linewidth=0.4, label=LABELS[i])
               for i in sorted(used)]
    if any_none:
        handles.append(Patch(facecolor=NONE_FILL, edgecolor=BORDER, linewidth=0.4,
                             label="no standard recorded"))
    if any_hatch:
        handles.append(Patch(facecolor="#ffffff", edgecolor=INK_2, linewidth=0.8,
                             hatch=PARTIAL_HATCH,
                             label=f"some assets unprotected (>{PARTIAL_MIN:.0%})"))
    handles.append(Patch(facecolor=CONTEXT_FILL, edgecolor=BORDER, linewidth=0.4,
                         label="not in the analysis"))
    leg = fig.legend(handles=handles, frameon=False, ncol=len(handles),
                     loc="lower center", bbox_to_anchor=(0.5, 0.018),
                     fontsize=LEGEND_FS, handlelength=1.8, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle("Flood protection standards applied, by country",
                 fontsize=TITLE_FS, fontweight="bold", color=INK,
                 x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.945, LINE_BREAK.join([
        "Median design return period across every asset in the country, as the model applies it "
        "- the per-feature values sampled from FLOPROS and COASTPROS,",
        "after each source's gaps have been filled. Hatching marks a country where some assets "
        "carry no standard at all; red marks one where most do not.",
    ]), fontsize=10.5, color=MUTED, va="top", linespacing=1.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  countries drawn : {len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", choices=("native", "country"), default="native",
                    help="native: each source at its own resolution (the "
                         "FLOPROS raster and the COASTPROS NUTS2 table). "
                         "country: the per-asset values the model applies, "
                         "aggregated to a country median.")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    if args.level == "native":
        out = Path(args.out) if args.out else (
            PROJECT_ROOT / "overview_figures" / "scope" / "Protection_standards.png")
        plot_native(out)
    else:
        if not CSV.exists():
            raise SystemExit(
                f"{CSV} not found - it holds the per-country medians this mode draws."
            )
        df = pd.read_csv(CSV)
        out = Path(args.out) if args.out else (
            PROJECT_ROOT / "overview_figures" / "scope" / "Protection_standards_country.png")
        plot_country(df, load_geometry(), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
