"""Cache the deterministic MIRACA_RISK results as a small Excel/CSV table.

This uncertainty study exists to characterise the figures produced by the main
MIRACA risk pipeline (S:/eks510/MIRACA_RISK). Those results are one parquet per
(country, asset) holding PER-ASSET expected annual damage - 400k+ rows and a
geometry column for a single large country - so they are far too heavy to read
every time a figure is redrawn.

This module reads them ONCE and writes country-level totals to a compact table
that src/ead_ranges.py can overlay on the uncertainty ranges in a fraction of a
second. Re-run it when MIRACA_RISK is regenerated; it is incremental, so files
whose modification time has not changed since the last run are skipped.

What is read
------------
Columns matching EAD_{min|mid|max}_{hazard}_{climate} and
exposure_abs_{hazard}_{climate}, plus object_type and asset_size. The geometry
column is never touched - it dominates the file size and nothing here needs it.

Hazards are river / coastal / windstorm / earthquake; climates are `current`
plus the 2050 and 2100 SSP245/SSP585 horizons where the pipeline produced them.
The min/mid/max triplet is the pipeline's own cost range, which corresponds to
this study's `cost_level` factor at -1 / 0 / +1 - so the reference min-max is
the spread from ONE factor, against which the study's p5-p95 spans all of them.

Note the naming difference: MIRACA_RISK calls river flooding `river`, while the
study's scenarios are named `flood_*`. HAZARD_ALIAS below is the single place
that translates, and src/ead_ranges.py imports it rather than re-deriving it.

Outputs
-------
overview_figures/reference/Reference_EAD.csv    country totals (what ead_ranges reads)
overview_figures/reference/Reference_EAD.xlsx   the same, plus object-type split and meta

Usage:
    python -m src.reference_ead
    python -m src.reference_ead --countries DEU FRA --force
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .curves import get_asset_config, is_mapped
from .plot_pyramid import PROJECT_ROOT

DEFAULT_RISK_DIR = Path("S:/eks510/MIRACA_RISK")
FILE_RE = re.compile(r"^([A-Z]{2,3})_([a-z]+)_hazards\.parquet$")
EAD_RE = re.compile(r"^EAD_(min|mid|max)_([a-z]+)_(.+)$")
EXP_RE = re.compile(r"^exposure_abs_([a-z]+)_(.+)$")

# MIRACA_RISK's hazard name -> this study's hazard name (src/ead_ranges.py
# hazard_of()). Only river differs; kept here so there is one source of truth.
HAZARD_ALIAS = {"river": "flood", "coastal": "coastal",
                "windstorm": "windstorm", "earthquake": "earthquake"}

BASE_COLS = ["object_type", "asset_size"]


def parse_file(path: Path) -> tuple[str, str] | None:
    m = FILE_RE.match(path.name)
    return (m.group(1), m.group(2)) if m else None


def classify_columns(names) -> tuple[dict, dict]:
    """Split the schema into EAD and exposure columns keyed by what they mean."""
    ead, exp = {}, {}
    for n in names:
        m = EAD_RE.match(n)
        if m:
            ead[n] = (m.group(1), m.group(2), m.group(3))  # stat, hazard, climate
            continue
        m = EXP_RE.match(n)
        if m:
            exp[n] = (m.group(1), m.group(2))              # hazard, climate
    return ead, exp


def summarise(path: Path, country: str, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country totals and per-object-type totals for one reference file."""
    schema = pq.ParquetFile(path).schema_arrow
    ead_cols, exp_cols = classify_columns(schema.names)
    if not ead_cols:
        return pd.DataFrame(), pd.DataFrame()

    have_base = [c for c in BASE_COLS if c in schema.names]
    df = pq.read_table(
        path, columns=list(ead_cols) + list(exp_cols) + have_base
    ).to_pandas()

    # Same exclusion as Stage 1 and the validation checks: MIRACA_RISK carries
    # every object_type it found, including ones this study has no vulnerability
    # curve for. Those features are absent from our own results, so dropping
    # them here keeps the reference totals comparable rather than inflated.
    n_raw = len(df)
    if "object_type" in df.columns:
        try:
            keep = is_mapped(get_asset_config(asset), df["object_type"])
        except KeyError:
            print(f"  NOTE: asset '{asset}' has no curve config; reference kept unfiltered.")
            keep = None
        if keep is not None and not bool(keep.all()):
            dropped = df["object_type"][~keep].value_counts().to_dict()
            print(
                f"  {country}/{asset}: excluding {int((~keep).sum())}/{n_raw} "
                f"reference feature(s) with no vulnerability curve: {dropped}"
            )
            df = df[keep].reset_index(drop=True)
    n_excluded = n_raw - len(df)

    # (hazard, climate) -> {stat: column}, so each combination becomes one row
    # with its min/mid/max side by side rather than three separate rows.
    combos: dict[tuple[str, str], dict[str, str]] = {}
    for col, (stat, hz, clim) in ead_cols.items():
        combos.setdefault((hz, clim), {})[stat] = col
    exp_lookup = {(hz, clim): col for col, (hz, clim) in exp_cols.items()}

    rows, by_type = [], []
    for (hz, clim), stats in sorted(combos.items()):
        mid_col = stats.get("mid")
        base = {
            "country": country, "asset": asset,
            "hazard": HAZARD_ALIAS.get(hz, hz), "hazard_raw": hz, "climate": clim,
            "n_assets": int(len(df)),
            "n_assets_excluded_unmapped": int(n_excluded),
        }
        for stat in ("min", "mid", "max"):
            col = stats.get(stat)
            base[f"ead_{stat}"] = float(df[col].sum()) if col else np.nan
        exp_col = exp_lookup.get((hz, clim))
        base["exposure_abs"] = float(df[exp_col].sum()) if exp_col else np.nan
        base["n_assets_nonzero"] = int((df[mid_col] > 0).sum()) if mid_col else 0
        base["asset_size_total"] = float(df["asset_size"].sum()) if "asset_size" in df else np.nan
        rows.append(base)

        if mid_col and "object_type" in df:
            g = df.groupby("object_type", dropna=False)
            part = pd.DataFrame({
                "ead_mid": g[mid_col].sum(),
                "n_assets": g[mid_col].size(),
                "n_assets_nonzero": g[mid_col].apply(lambda s: int((s > 0).sum())),
            }).reset_index()
            for stat in ("min", "max"):
                col = stats.get(stat)
                part[f"ead_{stat}"] = g[col].sum().to_numpy() if col else np.nan
            part.insert(0, "climate", clim)
            part.insert(0, "hazard", HAZARD_ALIAS.get(hz, hz))
            part.insert(0, "asset", asset)
            part.insert(0, "country", country)
            by_type.append(part)

    return pd.DataFrame(rows), (pd.concat(by_type, ignore_index=True) if by_type else pd.DataFrame())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--risk-dir", default=str(DEFAULT_RISK_DIR))
    parser.add_argument("--out-dir", default=None,
                        help="default: overview_figures/reference")
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--force", action="store_true",
                        help="re-read every file, ignoring the existing cache")
    args = parser.parse_args()

    risk_dir = Path(args.risk_dir)
    odir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures" / "reference"
    odir.mkdir(parents=True, exist_ok=True)
    totals_csv = odir / "Reference_EAD.csv"
    bytype_csv = odir / "Reference_EAD_by_object_type.csv"
    meta_csv = odir / "Reference_EAD_meta.csv"

    files = sorted(p for p in risk_dir.glob("*_hazards.parquet") if parse_file(p))
    plan = []
    for p in files:
        country, asset = parse_file(p)
        if args.countries and country not in args.countries:
            continue
        if args.assets and asset not in args.assets:
            continue
        plan.append((p, country, asset))
    if not plan:
        raise SystemExit(f"No matching *_hazards.parquet under {risk_dir}")

    # Incremental: keep rows whose source file has not been rewritten since the
    # cache was built, so a re-run after MIRACA_RISK regenerates only a few
    # countries costs only those countries.
    old_tot = pd.read_csv(totals_csv) if totals_csv.exists() and not args.force else pd.DataFrame()
    old_typ = pd.read_csv(bytype_csv) if bytype_csv.exists() and not args.force else pd.DataFrame()
    old_meta = pd.read_csv(meta_csv) if meta_csv.exists() and not args.force else pd.DataFrame()
    seen = {}
    if not old_meta.empty:
        seen = {(r.country, r.asset): r.source_mtime for r in old_meta.itertuples()}

    print(f"{len(plan)} reference file(s); {len(seen)} already cached")
    tot_new, typ_new, meta_new = [], [], []
    skipped = 0
    for i, (path, country, asset) in enumerate(plan, 1):
        mtime = path.stat().st_mtime
        if seen.get((country, asset)) == mtime:
            skipped += 1
            continue
        try:
            tot, typ = summarise(path, country, asset)
        except Exception as exc:
            print(f"  [{i}/{len(plan)}] {path.name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        if tot.empty:
            continue
        tot_new.append(tot)
        if not typ.empty:
            typ_new.append(typ)
        meta_new.append({
            "country": country, "asset": asset, "file": path.name,
            "source_mtime": mtime, "size_MB": round(path.stat().st_size / 1e6, 2),
            "n_assets": int(tot["n_assets"].iloc[0]),
        })
        if i % 25 == 0 or i == len(plan):
            print(f"  [{i}/{len(plan)}] read ({skipped} skipped as unchanged)")

    def _merge(old, new_frames):
        new = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
        if old.empty:
            return new
        if new.empty:
            return old
        keys = [(c, a) for c, a in zip(new["country"], new["asset"])]
        mask = ~pd.Series(list(zip(old["country"], old["asset"])), index=old.index).isin(set(keys))
        return pd.concat([old[mask], new], ignore_index=True)

    totals = _merge(old_tot, tot_new).sort_values(["country", "asset", "hazard", "climate"])
    bytype = _merge(old_typ, typ_new)
    meta = _merge(old_meta, [pd.DataFrame(meta_new)] if meta_new else [])

    totals.to_csv(totals_csv, index=False)
    bytype.to_csv(bytype_csv, index=False)
    meta.to_csv(meta_csv, index=False)
    with pd.ExcelWriter(odir / "Reference_EAD.xlsx", engine="openpyxl") as writer:
        totals.round(2).to_excel(writer, sheet_name="Reference_EAD", index=False)
        # Excel caps a sheet at ~1.05M rows; the object-type split is far
        # smaller than that in practice, but guard rather than fail at write.
        bytype.head(1_000_000).round(2).to_excel(
            writer, sheet_name="By_Object_Type", index=False)
        meta.to_excel(writer, sheet_name="Meta", index=False)

    print(f"\nWrote {len(totals):,} country-hazard-climate rows to {totals_csv}")
    print(f"      {len(bytype):,} object-type rows, {len(meta):,} files")
    print(f"      workbook: {odir / 'Reference_EAD.xlsx'}")


if __name__ == "__main__":
    main()
