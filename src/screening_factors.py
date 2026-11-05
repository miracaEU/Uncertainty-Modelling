"""Which factors the screening test finds non-influential, run by run.

Companion to src.convergence_tests. sobol_convergence.xlsx says whether the
screening test (Sarrazin et al. 2016, Eqs. 4-5) converged for each run, but
not which factors fell into X0 = {S_i < T}. This lists them, for T = 0.05 and
T = 0.01 and for both ST and S1, from the per-factor bootstrap table that
convergence_tests already wrote. Nothing is recomputed.

A factor is "screened out" when S_i < T and its own 95% CI is narrower than
0.05. A run's screening test converges exactly when every factor in its X0 is
screened out, so the two workbooks agree; that is checked on every run. A
factor below T whose CI is wider is "below T, CI wide" (* on the scenario
sheets): its index is small, but not pinned down.

Screening is meant for ST: a small S1 alone does not make a factor
non-influential, since it can still act through interactions. S1 is shown for
completeness.

Writes convergence/screening_factors.xlsx:
  Summary ST, Summary S1   factor x scenario: runs in which the factor is screened out
  one sheet per scenario   country x infrastructure matrices listing the factors in X0
  Below T                  every factor with S_i < 0.05, one row each, filterable
  Notes

    python -m src.screening_factors
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .convergence_tests import (BASE_N, CENTER, FILL, FONT, INDEX_TITLE, INDICES, OUT_DIR,
                                OUTCOME, RESAMPLES, SCEN_SHORT, SCREEN_T, VALUE_TOL, _put)
from .curves import ASSET_CONFIGS
from .plot_pyramid import scenario_label, scenario_sort_key

# Factors shared by every asset, in the order the summaries list them; the
# damage-curve factors follow, grouped by hazard.
SHARED_ORDER = ["warming", "slr", "depth_scale", "protection_scale", "protection_abs_rp",
                "pga_scale", "gust_scale", "cost_level", "aggregation"]
HAZARD_ORDER = {"Flood": 0, "Earthquake": 1, "Windstorm": 2}
# readable_factor's curve names, e.g. "Flood: runway (F7.4, 4 curves)"
CURVE_RE = re.compile(r"^(\w+): .*\(([A-Z][\w.-]*), \d+ curves?\)$")
WRAP = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEGEND = [("conv", "All screened out"), ("fail", "Some CI too wide (*)"),
          ("empty", "No factor < T"), ("zero", "Zero variance")]
KEYS = ["country", "asset", "scenario", "n", "index"]


def short_label(code: str, name: str) -> str:
    """The curve id ('F7.4') for a damage-curve factor, the code otherwise."""
    m = CURVE_RE.match(name)
    return m.group(2) if m else code


def long_label(code: str, name: str) -> str:
    m = CURVE_RE.match(name)
    return f"{m.group(1)} curve {m.group(2)}" if m else code


def factor_sort_key(code: str, name: str):
    m = CURVE_RE.match(name)
    if m is None:
        return (0, SHARED_ORDER.index(code) if code in SHARED_ORDER else len(SHARED_ORDER), code)
    hazard, cid = m.groups()
    return (1 + HAZARD_ORDER.get(hazard, 9), tuple(int(x) for x in re.findall(r"\d+", cid)), cid)


def share_fill(frac: float) -> PatternFill:
    """White -> blue by share screened out; not a good/bad scale, so no red or green."""
    lo, hi = (0xFF, 0xFF, 0xFF), (0x5B, 0x9B, 0xD5)
    rgb = [round(p + (q - p) * frac) for p, q in zip(lo, hi)]
    return PatternFill("solid", fgColor="%02X%02X%02X" % tuple(rgb))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fac = pd.read_csv(out_dir / "bootstrap_factors.csv")
    stats = pd.read_csv(out_dir / "convergence_stats.csv")
    narrow = fac["ci_width"] < VALUE_TOL
    for t in SCREEN_T:
        below = fac["boot_mean"] < t          # X0 membership, as in convergence_tests
        fac[f"below_{t}"] = below
        fac[f"out_{t}"] = below & narrow
        fac[f"wide_{t}"] = below & ~narrow
    fac["short"] = [short_label(c, n) for c, n in zip(fac["factor_code"], fac["factor"])]
    keys = {c: factor_sort_key(c, n) for c, n in zip(fac["factor_code"], fac["factor"])}
    fac["sort"] = fac["factor_code"].map({c: i for i, c in enumerate(sorted(keys, key=keys.get))})
    return fac, stats


def check_against_stats(fac: pd.DataFrame, stats: pd.DataFrame) -> None:
    """|X0| and the screening verdict must match convergence_stats.csv run by run."""
    ok = stats[stats["status"] == "ok"].set_index(KEYS)
    for t, key in zip(SCREEN_T, ("005", "001")):
        got = fac.groupby(KEYS)[[f"below_{t}", f"wide_{t}"]].sum()
        j = got.join(ok[[f"screen{key}_nX0", f"screen{key}_conv"]], how="outer")
        n_bad = int((j[f"below_{t}"] != j[f"screen{key}_nX0"]).sum()
                    + ((j[f"wide_{t}"] == 0) != j[f"screen{key}_conv"].astype(bool)).sum())
        if n_bad:
            raise SystemExit(f"T = {t}: {n_bad} runs disagree with convergence_stats.csv")
    print(f"Consistent with convergence_stats.csv for all {len(ok)} tested runs x index")


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def write_summary(ws, fac: pd.DataFrame, ix: str, scenarios) -> None:
    sub = fac[(fac["index"] == ix) & (fac["role"] == "base")]
    factors = (sub.drop_duplicates("factor_code").sort_values("sort")
               [["factor_code", "factor"]].itertuples(index=False))
    factors = list(factors)
    every = set(sub["asset"])
    used_by = sub.groupby("factor_code")["asset"].agg(
        lambda a: "all" if set(a) == every else ", ".join(x for x in ASSET_CONFIGS if x in set(a)))
    counts = sub.groupby(["factor_code", "scenario"]).agg(
        n=("factor", "size"), **{f"k_{t}": (f"out_{t}", "sum") for t in SCREEN_T},
        **{f"u_{t}": (f"wide_{t}", "sum") for t in SCREEN_T})
    totals = sub.groupby("factor_code").agg(
        n=("factor", "size"), **{f"k_{t}": (f"out_{t}", "sum") for t in SCREEN_T},
        **{f"u_{t}": (f"wide_{t}", "sum") for t in SCREEN_T})

    ws.cell(row=1, column=1, value=f"Factors screened out, {INDEX_TITLE[ix]}").font = \
        Font(bold=True, size=14)
    ws.cell(row=2, column=1,
            value=f"Each cell: runs where the factor has S < T and a 95% CI narrower than "
                  f"{VALUE_TOL} / runs where the factor is varied (accepted runs, N = {BASE_N}, "
                  "zero-variance runs excluded). (+u): a further u runs with S < T but a wider CI.")
    if ix == "S1":
        ws.cell(row=3, column=1,
                value="Screening should rest on ST: a factor with a small S1 can still matter "
                      "through interactions.").font = Font(italic=True, color="9C0006")
    r = 5
    for t in SCREEN_T:
        ws.cell(row=r, column=1, value=f"T = {t}").font = Font(bold=True, size=12)
        r += 1
        heads = ["Factor", "Used by"] + [SCEN_SHORT.get(s, s) for s in scenarios] + ["All scenarios"]
        for c, h in enumerate(heads, 1):
            _put(ws, r, c, h, bold=True, fill=FILL["head"], align=CENTER)
        for code, name in factors:
            r += 1
            _put(ws, r, 1, long_label(code, name), bold=True, fill=FILL["head"])
            _put(ws, r, 2, used_by[code], font=Font(color="595959", size=9))
            for c, scen in enumerate(list(scenarios) + [None], 3):
                row = totals.loc[code] if scen is None else counts.loc[(code, scen)] \
                    if (code, scen) in counts.index else None
                if row is None:
                    _put(ws, r, c, None)
                    continue
                n, k, u = int(row["n"]), int(row[f"k_{t}"]), int(row[f"u_{t}"])
                _put(ws, r, c, f"{k} / {n}" + (f" (+{u})" if u else ""),
                     fill=share_fill(k / n), align=CENTER, bold=scen is None)
        r += 3
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 34
    for c in range(3, 4 + len(scenarios)):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "C5"


def x0_cell(stat_row, runs: pd.DataFrame | None, t: float):
    """(text, style key, line count) for one country x asset cell."""
    if stat_row["status"] == "zero_variance":
        return "0 var", "zero", 1
    if stat_row["status"] == "error":
        return "err", "error", 1
    x0 = runs[runs[f"below_{t}"]].sort_values("sort") if runs is not None else runs
    if x0 is None or x0.empty:
        return "–", "empty", 1
    lines = [s + (" *" if w else "") for s, w in zip(x0["short"], x0[f"wide_{t}"])]
    return "\n".join(lines), ("fail" if x0[f"wide_{t}"].any() else "conv"), len(lines)


def write_x0_matrix(ws, r0: int, stats: pd.DataFrame, fac: pd.DataFrame, t: float,
                    assets, title: str) -> int:
    """ST and S1 matrices side by side, cells listing X0; returns the next free row."""
    countries = sorted(stats["country"].unique())
    ncol = len(assets)
    ws.cell(row=r0, column=1, value=title).font = Font(bold=True, size=12)
    r0 += 1
    groups = {k: g for k, g in fac.groupby(["country", "asset", "index"])}
    height = {}
    for b, ix in enumerate(INDICES):
        c0 = 2 + b * (ncol + 1)
        sub = stats[stats["index"] == ix]
        ws.cell(row=r0, column=c0, value=INDEX_TITLE[ix]).font = Font(bold=True)
        _put(ws, r0 + 1, 1, "Country", bold=True, fill=FILL["head"])
        for j, a in enumerate(assets):
            _put(ws, r0 + 1, c0 + j, a, bold=True, fill=FILL["head"], align=CENTER)
        lookup = {(r["country"], r["asset"]): r for _, r in sub.iterrows()}
        for i, c in enumerate(countries):
            rr = r0 + 2 + i
            _put(ws, rr, 1, c, bold=True, fill=FILL["head"])
            for j, a in enumerate(assets):
                row = lookup.get((c, a))
                if row is None:
                    _put(ws, rr, c0 + j, None)
                    continue
                text, key, lines = x0_cell(row, groups.get((c, a, ix)), t)
                _put(ws, rr, c0 + j, text, fill=FILL[key], font=FONT[key], align=WRAP)
                height[rr] = max(height.get(rr, 1), lines)
    for rr, lines in height.items():
        ws.row_dimensions[rr].height = 15 * lines
    return r0 + 2 + len(countries) + 2


def write_scenario_sheet(wb, scen: str, fac: pd.DataFrame, stats: pd.DataFrame, assets) -> None:
    ws = wb.create_sheet(SCEN_SHORT.get(scen, scen[:31]))
    ws.cell(row=1, column=1, value=f"{scenario_label(scen)}: factors in X0").font = \
        Font(bold=True, size=14)
    ws.cell(row=2, column=1,
            value=f"Cell = factors with S < T, by bootstrap mean. * = that factor's 95% CI is "
                  f"{VALUE_TOL} or wider, so it is below T but not pinned down. Curve factors "
                  "by curve id; full names on the Below T sheet.")
    for c, (key, label) in enumerate(LEGEND, 1):
        _put(ws, 3, c, label, fill=FILL[key], font=FONT[key], align=CENTER)
    r = 5
    for role in ("base", "rerun"):
        st = stats[(stats["scenario"] == scen) & (stats["role"] == role)]
        if st.empty:
            if role == "rerun":
                ws.cell(row=r, column=1, value="No reruns at a larger N for this scenario.").font = \
                    Font(italic=True, color="595959")
            continue
        f = fac[(fac["scenario"] == scen) & (fac["role"] == role)]
        label = (f"Accepted runs, N = {BASE_N}" if role == "base" else
                 "Reruns at N = " + ", ".join(str(n) for n in sorted(st["n"].unique())))
        for t in SCREEN_T:
            r = write_x0_matrix(ws, r, st, f, t, assets, f"{label}: T = {t}")
    ws.column_dimensions["A"].width = 10
    for col in range(2, 2 + 2 * (len(assets) + 1)):
        ws.column_dimensions[get_column_letter(col)].width = 17
    ws.column_dimensions[get_column_letter(2 + len(assets))].width = 3   # gap between ST and S1
    ws.freeze_panes = "B4"


LIST_COLS = [("Scenario", 12, None), ("Country", 9, None), ("Asset", 11, None), ("N", 7, None),
             ("Index", 7, None), ("Factor code", 18, None), ("Factor", 52, None),
             ("S_i (bootstrap mean)", 12, "0.0000"), ("Point estimate", 12, "0.0000"),
             ("CI lower", 10, "0.0000"), ("CI upper", 10, "0.0000"), ("CI width", 10, "0.0000"),
             ("Rank", 6, None), ("Factors in run", 9, None)] + [
             (f"Screened out, T = {t}", 13, None) for t in SCREEN_T]


def write_list(ws, fac: pd.DataFrame, scenarios) -> None:
    ws.title = "Below T"
    rows = fac[fac["boot_mean"] < max(SCREEN_T)].copy()
    rows["scen_order"] = rows["scenario"].map({s: i for i, s in enumerate(scenarios)})
    rows = rows.sort_values(["scen_order", "country", "asset", "n", "index", "sort"],
                            ascending=[True, True, True, True, False, True])
    for c, (h, w, _) in enumerate(LIST_COLS, 1):
        _put(ws, 1, c, h, bold=True, fill=FILL["head"],
             align=Alignment(horizontal="center", vertical="center", wrap_text=True))
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 32

    for f in rows.to_dict("records"):
        ws.append([SCEN_SHORT.get(f["scenario"], f["scenario"])]
                  + [f[c] for c in ("country", "asset", "n", "index", "factor_code", "factor",
                                    "boot_mean", "point", "ci_lb", "ci_ub", "ci_width", "rank",
                                    "n_factors")]
                  + ["yes" if f[f"out_{t}"] else "CI wide" if f[f"below_{t}"] else ""
                     for t in SCREEN_T])
    flag0 = len(LIST_COLS) - len(SCREEN_T) + 1
    for row in ws.iter_rows(min_row=2):
        for (_, _, fmt), cell in zip(LIST_COLS, row):
            if fmt:
                cell.number_format = fmt
        for cell in row[flag0 - 1:]:
            key = {"yes": "conv", "CI wide": "fail"}.get(cell.value)
            if key:
                cell.fill, cell.font, cell.alignment = FILL[key], FONT[key], CENTER
    ws.auto_filter.ref = f"A1:{get_column_letter(len(LIST_COLS))}{ws.max_row}"
    ws.freeze_panes = "A2"


def write_notes(wb, scenarios) -> None:
    ws = wb.create_sheet("Notes")
    ws.cell(row=1, column=1, value="Definitions").font = Font(bold=True, size=14)
    notes = [
        ("Source", "convergence/bootstrap_factors.csv and convergence_stats.csv, as written by "
                   "python -m src.convergence_tests; nothing is recomputed here. Outcome "
                   f"{OUTCOME}, {RESAMPLES} bootstrap resamples, 95% percentile CIs."),
        ("S_i", "Bootstrap mean of the index (Sarrazin et al. 2016, Eq. 4)."),
        ("X0", "Factors with S_i < T (Eq. 4), for T = " + " and ".join(map(str, SCREEN_T)) + "."),
        ("Screened out", f"In X0 and its own 95% CI narrower than {VALUE_TOL}. A run's screening "
                         "test (sobol_convergence.xlsx) converges exactly when every factor in "
                         "its X0 is screened out; this is checked for every run."),
        ("CI wide (*)", f"In X0, but its CI is {VALUE_TOL} or wider: small, but not pinned down."),
        ("ST vs S1", "Screening should be read on ST. S1 measures the effect of a factor alone, so "
                     "a factor with a small S1 can still be influential through interactions."),
        ("Summary", f"Accepted runs (N = {BASE_N}) only, zero-variance runs excluded. The N = 32768 "
                    "reruns are on the scenario sheets and in Below T."),
        ("Below T", f"Every factor with S_i < {max(SCREEN_T)} in any run, with its CI. Use the "
                    "column filters, e.g. Index = ST and Screened out, T = 0.01 = yes."),
        ("Built by", "python -m src.screening_factors"),
    ]
    r = 3
    for k, v in notes:
        _put(ws, r, 1, k, bold=True, fill=FILL["head"], align=Alignment(vertical="top"))
        _put(ws, r, 2, v, align=Alignment(wrap_text=True, vertical="top"))
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="Scenario codes").font = Font(bold=True)
    for scen in scenarios:
        r += 1
        _put(ws, r, 1, SCEN_SHORT.get(scen, scen), bold=True, fill=FILL["head"])
        _put(ws, r, 2, scenario_label(scen))
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 120


def write_workbook(out_dir: Path) -> Path:
    fac, stats = load(out_dir)
    check_against_stats(fac, stats)
    scenarios = sorted(stats["scenario"].unique(), key=scenario_sort_key)
    assets = [a for a in ASSET_CONFIGS if a in set(stats["asset"])]

    wb = Workbook()
    write_summary(wb.active, fac, "ST", scenarios)
    wb.active.title = "Summary ST"
    write_summary(wb.create_sheet("Summary S1"), fac, "S1", scenarios)
    for scen in scenarios:
        write_scenario_sheet(wb, scen, fac, stats, assets)
    write_list(wb.create_sheet(), fac, scenarios)
    write_notes(wb, scenarios)
    path = out_dir / "screening_factors.xlsx"
    wb.save(path)
    print(f"Saved {path} ({len(wb.sheetnames)} sheets)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help=f"folder with the convergence_tests CSVs (default {OUT_DIR})")
    args = ap.parse_args()
    write_workbook(Path(args.out_dir) if args.out_dir else OUT_DIR)


if __name__ == "__main__":
    main()
