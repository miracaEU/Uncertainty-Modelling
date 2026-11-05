"""Bootstrap convergence tests for every Sobol run, after Sarrazin et al. (2016).

The adaptive search accepted each run on SALib's ST_conf/ST ratio. This checks
the accepted runs, and the N=32768 reruns, against the three convergence
types of Sarrazin, Pianosi & Wagener (2016, Environ. Model. Softw. 79, 135-152),
for both the first-order (S1) and the total-order (ST) index:

  value         index values: the widest 95% bootstrap CI across all factors
                is below 0.05 (their Eq. 1)
  rank-rho      ranking: the 95th percentile, over every pair of bootstrap
                resamples, of the sensitivity-weighted rank distance is
                below 1 (Eqs. 2-3)
  rank-CI-all   ranking, alternative: no two factors have overlapping CIs
  rank-CI-top3  ranking, alternative: no overlap between #1-#2, #2-#3 and
                #3-#4, so both the order of the top three and which factors
                make it are resolved
  screen-0.05   screening: the widest CI among factors with S < T is below
  screen-0.01   0.05 (Eqs. 4-5), for T = 0.05 and T = 0.01

The CIs are bootstrap PERCENTILE intervals from 1000 resamples of the N base
rows, as in Sarrazin; SALib's own *_conf is a normal approximation (Z x std)
over 100 resamples, and is not used. The estimators are SALib's own
first_order / total_order, on SALib's own normalisation of the output, so the
point estimates reproduce the per-combination _sobol_indices.csv exactly -
which is checked for every archive at the base N.

Outcome: total_EAD_MEUR, the headline used throughout the study.

Writes to convergence/ (repo root):
  convergence_stats.csv   one row per (combination, N, index): every statistic
  bootstrap_factors.csv   one row per factor: point, bootstrap mean, CI, rank
  sobol_convergence.xlsx  overview + one sheet per scenario x criterion

    python -m src.convergence_tests                          # compute + workbook
    python -m src.convergence_tests --countries NLD LUX --out-dir <tmp>   # pilot
    python -m src.convergence_tests --resume                 # only errored or new archives
    python -m src.convergence_tests --workbook-only          # restyle from CSVs
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy.stats import rankdata

from .aggregate_results import index_sobol_archives, readable_factor
from .curves import ASSET_CONFIGS
from .paths import PROJECT_ROOT
from .plot_pyramid import scenario_label, scenario_sort_key
from .sobol_design import sobol_layout

OUTCOME = "total_EAD_MEUR"
BASE_N = 8192          # the study's accepted sample size
RESAMPLES = 1000
SEED = 20160101
BLOCK = 100            # resamples per bootstrap block; bounds memory at N=32768
CONF_LO, CONF_HI = 2.5, 97.5

VALUE_TOL = 0.05       # Sarrazin Eq. 1 and Eq. 5 threshold on CI width
RHO_TOL = 1.0          # Sarrazin Eq. 3: under one rank position
SCREEN_T = (0.05, 0.01)  # the screening threshold T; only T moves, the width cut stays VALUE_TOL

OUT_DIR = PROJECT_ROOT / "convergence"
RESULTS_DIR = PROJECT_ROOT / "results"
INDICES = ("ST", "S1")
INDEX_TITLE = {"ST": "Total-order index (ST)", "S1": "First-order index (S1)"}

# Sheet names must fit Excel's 31 characters, so each scenario has a short code.
SCEN_SHORT = {
    "flood_absprot_ds": "FL-abs",
    "flood_noprot_ds": "FL-rec",
    "flood_absprot_ds_rp10to100": "FL-RP10-100",
    "flood_baseline_ds_sc0p5to1p5": "FL-x0.5-1.5",
    "coastal_absprot_ds": "CO-abs",
    "coastal_noprot_ds": "CO-rec",
    "coastal_absprot_ds_slr": "CO-abs-SLR",
    "coastal_noprot_ds_slr": "CO-rec-SLR",
    "coastal_absprot_ds_rp10to100": "CO-RP10-100",
    "coastal_baseline_ds_sc0p5to1p5": "CO-x0.5-1.5",
    "earthquake": "EQ",
    "earthquake_pga0p9to1p1": "EQ-x0.9-1.1",
    "windstorm": "WS",
    "windstorm_absprot": "WS-abs",
}

# code, title, what the cell shows, converged-if, stat column, verdict column, number format
CRITERIA = [
    ("value", "Index values",
     "widest 95% CI across all factors", f"< {VALUE_TOL}",
     "value_stat", "value_conv", "0.000"),
    ("rank-rho", "Ranking (weighted rank distance)",
     "95th percentile of rho over all pairs of bootstrap resamples", f"< {RHO_TOL:g}",
     "rho_q95", "rho_conv", "0.00"),
    ("rank-CI-all", "Ranking, CI overlap, all factors",
     "number of factor pairs whose 95% CIs overlap", "= 0",
     "ci_all_overlaps", "ci_all_conv", "0"),
    ("rank-CI-top3", "Ranking, CI overlap, top 3",
     "overlapping pairs among #1-#2, #2-#3, #3-#4", "= 0",
     "ci_top3_overlaps", "ci_top3_conv", "0"),
] + [
    (f"screen-{t}", f"Screening, T = {t}",
     f"widest 95% CI among factors with S < {t}", f"< {VALUE_TOL}",
     f"screen{_k}_stat", f"screen{_k}_conv", "0.000")
    for t, _k in ((0.05, "005"), (0.01, "001"))
]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_indices(y: np.ndarray, d: int, resamples: int, seed: int, n: int | None = None):
    """Point S1/ST and their (resamples, d) bootstrap draws, SALib's way.

    Mirrors SALib.analyze.sobol.analyze for the first- and total-order part:
    the same output normalisation, the same row layout and the same
    estimators, with resampling of the N base rows. The second-order indices,
    which that function also bootstraps, are skipped - nothing here uses them.
    Works for samples with and without the second-order (BA) rows.
    """
    from SALib.analyze.sobol import first_order, separate_output_values, total_order

    n, second_order = sobol_layout(y.size, d, n)
    y = (y - y.mean()) / y.std()
    a, b, ab, _ = separate_output_values(y, d, n, second_order)
    s1 = np.array([first_order(a, ab[:, j], b) for j in range(d)], dtype=float).ravel()
    st = np.array([total_order(a, ab[:, j], b) for j in range(d)], dtype=float).ravel()

    rng = np.random.default_rng(seed)
    s1_boot = np.empty((resamples, d))
    st_boot = np.empty((resamples, d))
    for start in range(0, resamples, BLOCK):
        size = min(BLOCK, resamples - start)
        r = rng.integers(n, size=(n, size))
        ar, br = a[r], b[r]
        for j in range(d):
            abr = ab[r, j]
            # A constant resample makes SALib return a single 0.0; broadcast it.
            s1_boot[start:start + size, j] = np.broadcast_to(first_order(ar, abr, br), size)
            st_boot[start:start + size, j] = np.broadcast_to(total_order(ar, abr, br), size)
    return n, {"S1": s1, "ST": st}, {"S1": s1_boot, "ST": st_boot}


def rank_rho_q95(boot: np.ndarray, chunk: int = 50) -> float:
    """Sarrazin Eqs. 2-3: Q0.95 over all resample pairs of the weighted rank distance.

    Rank 1 is the most influential factor; ties take their average rank. The
    weight of factor i in pair (j, k) is max(S_i^j, S_i^k)^2, normalised over
    factors. Negative bootstrap estimates (S1 can dip below zero) are clipped to
    zero in the weight only - squared, two negative estimates would otherwise
    weigh as much as two positive ones. A pair whose weights are all zero
    scores 0.
    """
    ranks = rankdata(-boot, method="average", axis=1)
    w_src = np.clip(boot, 0.0, None)
    b = boot.shape[0]
    out = []
    for i0 in range(0, b - 1, chunk):
        i1 = min(i0 + chunk, b - 1)
        diff = np.abs(ranks[i0:i1, None, :] - ranks[None, :, :])
        w = np.maximum(w_src[i0:i1, None, :], w_src[None, :, :]) ** 2
        den = w.sum(-1)
        rho = np.divide((diff * w).sum(-1), den, out=np.zeros_like(den), where=den > 0)
        # keep pairs (j, k) with k > j only
        jj = np.arange(i0, i1)[:, None]
        kk = np.arange(b)[None, :]
        out.append(rho[kk > jj])
    return float(np.quantile(np.concatenate(out), 0.95))


def _overlap(lb: np.ndarray, ub: np.ndarray, i: int, k: int) -> bool:
    return bool(lb[i] <= ub[k] and lb[k] <= ub[i])


def index_stats(point: np.ndarray, boot: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """All six criteria for one index, plus the per-factor table."""
    mean = boot.mean(axis=0)
    lb, ub = np.percentile(boot, [CONF_LO, CONF_HI], axis=0)
    width = ub - lb
    d = len(mean)
    order = np.argsort(-mean, kind="stable")       # rank by bootstrap mean (Sarrazin Eq. 4)
    rank = np.empty(d, dtype=int)
    rank[order] = np.arange(1, d + 1)

    stats = {"value_stat": float(width.max())}
    stats["value_conv"] = stats["value_stat"] < VALUE_TOL

    stats["rho_q95"] = rank_rho_q95(boot)
    stats["rho_conv"] = stats["rho_q95"] < RHO_TOL

    all_pairs = [(i, k) for i in range(d) for k in range(i + 1, d)]
    stats["ci_all_pairs"] = len(all_pairs)
    stats["ci_all_overlaps"] = sum(_overlap(lb, ub, i, k) for i, k in all_pairs)
    stats["ci_all_conv"] = stats["ci_all_overlaps"] == 0

    top = [(order[p], order[p + 1]) for p in range(min(3, d - 1))]
    stats["ci_top3_pairs"] = len(top)
    stats["ci_top3_overlaps"] = sum(_overlap(lb, ub, i, k) for i, k in top)
    stats["ci_top3_conv"] = stats["ci_top3_overlaps"] == 0

    for t, key in zip(SCREEN_T, ("005", "001")):
        x0 = mean < t
        stats[f"screen{key}_nX0"] = int(x0.sum())
        # An empty X0 has nothing left to pin down, so it counts as converged;
        # the stat stays NaN so the workbook can show it apart.
        stats[f"screen{key}_stat"] = float(width[x0].max()) if x0.any() else np.nan
        stats[f"screen{key}_conv"] = (not x0.any()) or stats[f"screen{key}_stat"] < VALUE_TOL

    factors = pd.DataFrame({"point": point, "boot_mean": mean, "ci_lb": lb, "ci_ub": ub,
                            "ci_width": width, "rank": rank})
    return stats, factors


def read_outcome(path: str, name: str) -> np.ndarray:
    """One scalar outcome from an ema_workbench archive, without the rest of it.

    load_results decompresses and parses the whole archive - experiments.csv and
    every outcome - and looks up metadata.json, which sits at the end. Read off
    the network share that is most of the time spent here. The outcome is a
    headerless one-column CSV (ScalarOutcome.to_disk), and it follows
    experiments.csv directly, so a streaming read can stop as soon as it has
    it. Parsing matches ScalarOutcome.from_disk, so the values are identical.
    """
    import io
    import tarfile

    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name == f"{name}.cls":
                # A streamed member cannot seek, which read_csv needs; buffer it.
                f = io.BytesIO(archive.extractfile(member).read())
                return pd.read_csv(f, index_col=False, header=None).to_numpy().ravel()
    raise KeyError(f"{name}.cls not in {Path(path).name}")


def with_retries(fn, *args, tries: int = 6, wait: float = 15.0):
    """Call fn, retrying file errors with backoff.

    The network share drops out now and then, and while it is gone every file
    on it reports as missing: the first full run lost 1408 archives in whole
    country blocks that way, although every file was there. A real missing
    file still fails, just a few minutes later.
    """
    for attempt in range(tries):
        try:
            return fn(*args)
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(wait * 2 ** attempt)


def analyse_archive(task: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    """One archive -> one stats row per index, and its per-factor tables."""
    base = {k: task[k] for k in ("country", "asset", "scenario", "n", "role")}
    base["archive"] = Path(task["path"]).name
    try:
        csv = with_retries(pd.read_csv, task["csv"])
        csv = csv[csv["outcome"] == OUTCOME]
        if csv.empty:
            raise ValueError(f"no {OUTCOME} rows in {Path(task['csv']).name}")
        names = csv["factor"].tolist()
        d = len(names)

        y = np.asarray(with_retries(read_outcome, task["path"], OUTCOME), dtype=float)
        sobol_layout(y.size, d, task["n"])     # ValueError if the rows do not fit N
        base["n_factors"] = d
        if np.var(y) < 1e-30:     # same cut as analyze_sobol: indices undefined
            return [{**base, "index": ix, "status": "zero_variance"} for ix in INDICES], []

        _, point, boot = bootstrap_indices(y, d, task["resamples"], task["seed"], task["n"])
    except Exception as exc:                                       # noqa: BLE001
        return [{**base, "index": ix, "status": "error",
                 "error": f"{type(exc).__name__}: {exc}"} for ix in INDICES], []

    rows, frames = [], []
    for ix in INDICES:
        stats, fac = index_stats(point[ix], boot[ix])
        # Layout check: the point estimate must equal what the study stored.
        csv_diff = float(np.max(np.abs(point[ix] - csv[ix].to_numpy(float))))
        top = names[int(np.argmax(fac["boot_mean"]))]
        rows.append({**base, "index": ix, "status": "ok", **stats,
                     "top_factor": readable_factor(task["asset"], top),
                     "csv_max_abs_diff": csv_diff})
        fac.insert(0, "factor", [readable_factor(task["asset"], c) for c in names])
        fac.insert(0, "factor_code", names)
        for k, v in reversed(list({**base, "index": ix}.items())):
            fac.insert(0, k, v)
        frames.append(fac)
    return rows, frames


def collect_tasks(results_dir: Path, countries, resamples: int, seed: int):
    """Base-N archive for every combination, plus any larger-N rerun."""
    tasks, other = [], []
    for (c, a, s), by_n in sorted(index_sobol_archives(results_dir).items()):
        if countries and c not in countries:
            continue
        csv = results_dir / c / f"{c}_{a}_{s}_sobol_indices.csv"
        for n, path in sorted(by_n.items()):
            if n == BASE_N:
                role = "base"
            elif n > BASE_N and BASE_N in by_n and n == max(by_n):
                role = "rerun"
            else:
                other.append({"country": c, "asset": a, "scenario": s, "n": n,
                              "archive": path.name})
                continue
            tasks.append({"country": c, "asset": a, "scenario": s, "n": n, "role": role,
                          "path": str(path), "csv": str(csv),
                          "resamples": resamples, "seed": seed})
    return tasks, pd.DataFrame(other, columns=["country", "asset", "scenario", "n", "archive"])


def compute(args, out_dir: Path) -> None:
    tasks, other = collect_tasks(RESULTS_DIR, set(args.countries or []),
                                 args.resamples, args.seed)
    kept_stats, kept_factors = None, None
    if args.resume:
        # Test only the archives without a usable result yet - errored last time
        # (a share outage) or new since (a rerun) - and merge them in. Rows for
        # archives no longer tested, e.g. a rerun superseded by a larger one,
        # are dropped.
        prev = pd.read_csv(out_dir / "convergence_stats.csv")
        current = {Path(t["path"]).name for t in tasks}
        done = set(prev.loc[prev["status"] != "error", "archive"]) & current
        tasks = [t for t in tasks if Path(t["path"]).name not in done]
        kept_stats = prev[prev["archive"].isin(done)]
        kept_factors = pd.read_csv(out_dir / "bootstrap_factors.csv")
        kept_factors = kept_factors[kept_factors["archive"].isin(done)]
        print(f"--resume: {len(tasks)} archive(s) to test, {len(done)} kept")
    n_rerun = sum(t["role"] == "rerun" for t in tasks)
    print(f"{len(tasks)} archives ({len(tasks) - n_rerun} at N={BASE_N}, {n_rerun} reruns), "
          f"{len(other)} at another N (listed, not tested); "
          f"{args.resamples} resamples, {args.workers} workers")

    rows, frames = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyse_archive, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r, f = fut.result()
            rows += r
            frames += f
            if i % 100 == 0 or i == len(futures):
                el = time.time() - t0
                print(f"  {i:5d}/{len(futures)}  {el / 60:5.1f} min elapsed, "
                      f"~{el / i * (len(futures) - i) / 60:5.1f} min to go", flush=True)

    stats = pd.DataFrame(rows)
    factors = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if kept_stats is not None:
        stats = pd.concat([kept_stats, stats], ignore_index=True)
        factors = pd.concat([kept_factors, factors], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "convergence_stats.csv", index=False)
    factors.to_csv(out_dir / "bootstrap_factors.csv", index=False)
    other.to_csv(out_dir / "archives_not_tested.csv", index=False)

    err = stats[stats["status"] == "error"]
    if len(err):
        print(f"\n!! {len(err) // len(INDICES)} archive(s) failed:")
        for _, r in err.drop_duplicates("archive").iterrows():
            print(f"   {r['archive']}: {r['error']}")
    # The stored CSV holds whichever N the combination was last analysed (or
    # restored, src.restore_sobol_n) at, so one of its runs must match it.
    ok = stats[stats["status"] == "ok"]
    best = ok.groupby(["country", "asset", "scenario", "index"])["csv_max_abs_diff"].min()
    bad = best[best > 1e-9].reset_index()
    if len(bad):
        print(f"\n!! {len(bad)} combination(s) x index match _sobol_indices.csv at no tested N - "
              "the factor order or row layout is wrong for these:")
        print(bad.to_string(index=False))
    else:
        print(f"\nLayout check passed: all {len(best)} combinations x index match "
              "_sobol_indices.csv to 1e-9 at one of their tested N.")


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

FILL = {
    "conv": PatternFill("solid", fgColor="C6EFCE"),
    "fail": PatternFill("solid", fgColor="FFC7CE"),
    "zero": PatternFill("solid", fgColor="D9D9D9"),
    "empty": PatternFill("solid", fgColor="E2F0D9"),
    "error": PatternFill("solid", fgColor="FFEB9C"),
    "head": PatternFill("solid", fgColor="F2F2F2"),
}
FONT = {
    "conv": Font(color="006100"),
    "fail": Font(color="9C0006"),
    "zero": Font(color="595959", italic=True),
    "empty": Font(color="375623"),
    "error": Font(color="9C5700"),
}
# The same colours in bold: marks the cells of the highest-N matrix that come from a rerun.
FONT_RERUN = {k: Font(bold=True, color=f.color, italic=f.italic) for k, f in FONT.items()}
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
# Short enough for one matrix column; the Notes sheet has the full definitions.
LEGEND = [("conv", "Converged"), ("fail", "Not converged"), ("zero", "Zero variance"),
          ("empty", "No factor < T"), ("error", "Error")]


def sheet_name(scen: str, crit: str) -> str:
    return f"{SCEN_SHORT.get(scen, scen[:18])} {crit}"[:31]


def cell_state(row, stat_col: str, conv_col: str):
    """(display value, style key) for one matrix cell."""
    if row["status"] == "zero_variance":
        return "0 var", "zero"
    if row["status"] == "error":
        return "err", "error"
    v = row[stat_col]
    if stat_col.startswith("screen") and pd.isna(v):
        return "–", "empty"
    return v, ("conv" if bool(row[conv_col]) else "fail")


def highest_n(stats: pd.DataFrame) -> pd.DataFrame:
    """Each combination's run at the largest N tested: the rerun where there is one."""
    return (stats[stats["role"].isin(["base", "rerun"])].sort_values("n", kind="stable")
            .groupby(["country", "asset", "scenario", "index"]).tail(1))


def tally(df: pd.DataFrame, conv_col: str) -> tuple[int, int]:
    ok = df[df["status"] == "ok"]
    return int(ok[conv_col].astype(bool).sum()), len(ok)


def pct_fill(frac: float) -> PatternFill:
    """Red -> amber -> green by share converged."""
    stops = [(0.0, (0xF8, 0x69, 0x6B)), (0.5, (0xFF, 0xEB, 0x84)), (1.0, (0x63, 0xBE, 0x7B))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if frac <= x1:
            t = (frac - x0) / (x1 - x0)
            rgb = [round(p + (q - p) * t) for p, q in zip(c0, c1)]
            return PatternFill("solid", fgColor="%02X%02X%02X" % tuple(rgb))
    return PatternFill("solid", fgColor="63BE7B")


def _put(ws, r, c, v, bold=False, fill=None, font=None, fmt=None, align=None, border=True):
    cell = ws.cell(row=r, column=c, value=v)
    if bold:
        cell.font = Font(bold=True)
    if font is not None:
        cell.font = font
    if fill is not None:
        cell.fill = fill
    if fmt:
        cell.number_format = fmt
    if align is not None:
        cell.alignment = align
    if border:
        cell.border = BOX
    return cell


def write_matrix(ws, r0: int, df: pd.DataFrame, assets, crit, title: str,
                 mark_reruns: bool = False) -> int:
    """ST and S1 country x infrastructure matrices side by side; returns next free row.

    With mark_reruns, cells whose run is above BASE_N are set in bold.
    """
    _, _, _, _, stat_col, conv_col, fmt = crit
    countries = sorted(df["country"].unique())
    ncol = len(assets)
    ws.cell(row=r0, column=1, value=title).font = Font(bold=True, size=12)
    r0 += 1
    for b, ix in enumerate(INDICES):
        c0 = 2 + b * (ncol + 1)
        sub = df[df["index"] == ix]
        k, n = tally(sub, conv_col)
        head = ws.cell(row=r0, column=c0,
                       value=f"{INDEX_TITLE[ix]}: {k} / {n} converged" if n else INDEX_TITLE[ix])
        head.font = Font(bold=True)
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
                v, key = cell_state(row, stat_col, conv_col)
                font = FONT_RERUN if mark_reruns and row["n"] > BASE_N else FONT
                _put(ws, rr, c0 + j, v, fill=FILL[key], font=font[key],
                     fmt=fmt if key in ("conv", "fail") else None, align=CENTER)
    return r0 + 2 + len(countries) + 2


def write_scenario_sheet(wb, scen: str, crit, final: pd.DataFrame, base: pd.DataFrame,
                         rerun: pd.DataFrame, assets):
    code, title, shows, conv_if, *_ = crit
    ws = wb.create_sheet(sheet_name(scen, code))
    ws.cell(row=1, column=1, value=f"{scenario_label(scen)}: {title}").font = Font(bold=True, size=14)
    ws.cell(row=2, column=1,
            value=f"Cell = {shows}. Converged if {conv_if}. Outcome {OUTCOME}, "
                  f"{RESAMPLES} bootstrap resamples, 95% percentile CIs.")
    c = 1
    for key, label in LEGEND:
        if key == "empty" and not code.startswith("screen"):
            continue
        _put(ws, 3, c, label, fill=FILL[key], font=FONT[key], align=CENTER)
        c += 1
    r = 5
    if len(rerun):
        # Only worth a matrix of its own where some combination was rerun;
        # otherwise it would repeat the N = BASE_N one cell for cell.
        _put(ws, 3, c, "Bold = rerun", font=Font(bold=True), align=CENTER)
        r = write_matrix(ws, r, final, assets, crit,
                         f"Highest N run per combination (bold: rerun above N = {BASE_N})",
                         mark_reruns=True)
    r = write_matrix(ws, r, base, assets, crit, f"Accepted runs, N = {BASE_N}")
    if len(rerun):
        ns = ", ".join(str(n) for n in sorted(rerun["n"].unique()))
        write_matrix(ws, r, rerun, assets, crit,
                     f"Reruns at N = {ns} (same combinations as above, larger sample)")
    else:
        ws.cell(row=r, column=1, value="No reruns at a larger N for this scenario.").font = \
            Font(italic=True, color="595959")
    ws.column_dimensions["A"].width = 14
    for col in range(2, 2 + 2 * (len(assets) + 1)):
        ws.column_dimensions[get_column_letter(col)].width = 14
    ws.column_dimensions[get_column_letter(2 + len(assets))].width = 3   # gap between ST and S1
    ws.freeze_panes = "B8"      # title, legend and the asset header row stay in view


def write_overview(ws, stats: pd.DataFrame, scenarios) -> None:
    ws.title = "Overview"
    ws.cell(row=1, column=1, value="Sobol convergence after Sarrazin et al. (2016)").font = \
        Font(bold=True, size=14)
    ws.cell(row=2, column=1,
            value=f"Each cell: converged / tested (share). Tested excludes zero-variance "
                  f"combinations. Outcome {OUTCOME}; {RESAMPLES} bootstrap resamples. "
                  "Definitions on the Notes sheet.")

    def header(r: int, first: list[str]) -> int:
        for i, h in enumerate(first, 1):
            _put(ws, r, i, h, bold=True, fill=FILL["head"], align=CENTER)
            _put(ws, r + 1, i, None, fill=FILL["head"])
        c = len(first) + 1
        for _, ctitle, _, conv_if, *_ in CRITERIA:
            _put(ws, r, c, f"{ctitle} ({conv_if})", bold=True, fill=FILL["head"],
                 align=Alignment(horizontal="center", vertical="center", wrap_text=True))
            _put(ws, r, c + 1, None, fill=FILL["head"])
            # Style both halves first: once merged, the right one is read-only.
            ws.merge_cells(start_row=r, start_column=c, end_row=r, end_column=c + 1)
            for b, ix in enumerate(INDICES):
                _put(ws, r + 1, c + b, ix, bold=True, fill=FILL["head"], align=CENTER)
            c += 2
        ws.row_dimensions[r].height = 32
        return r + 2

    def counts_table(r: int, runs: pd.DataFrame, title: str, count_reruns: bool) -> int:
        ws.cell(row=r, column=1, value=title).font = Font(bold=True, size=12)
        first = ["Scenario", "Sheets", "Combinations", "Zero variance"]
        r = header(r + 1, first + ([f"Of which N > {BASE_N}"] if count_reruns else []))
        for scen in list(scenarios) + [None]:
            sub = runs if scen is None else runs[runs["scenario"] == scen]
            st = sub[sub["index"] == "ST"]
            _put(ws, r, 1, "All scenarios" if scen is None else scenario_label(scen),
                 bold=scen is None)
            _put(ws, r, 2, None if scen is None else SCEN_SHORT.get(scen, scen))
            _put(ws, r, 3, len(st), align=CENTER, bold=scen is None)
            _put(ws, r, 4, int((st["status"] == "zero_variance").sum()), align=CENTER)
            c = 5
            if count_reruns:
                _put(ws, r, c, int((st["n"] > BASE_N).sum()), align=CENTER)
                c += 1
            for *_, conv_col, _ in CRITERIA:
                for ix in INDICES:
                    k, n = tally(sub[sub["index"] == ix], conv_col)
                    frac = k / n if n else 0.0
                    _put(ws, r, c, f"{k} / {n} ({frac:.0%})" if n else "–",
                         fill=pct_fill(frac) if n else None, align=CENTER, bold=scen is None)
                    c += 1
            r += 1
        return r

    base = stats[stats["role"] == "base"]
    rerun = stats[stats["role"] == "rerun"]
    r = 4
    # --- table 0: each combination at its highest N, if anything was rerun ----
    if len(rerun):
        r = counts_table(r, highest_n(stats),
                         "Highest N run per combination (the rerun where there is one)",
                         count_reruns=True) + 2

    # --- table 1: accepted runs --------------------------------------------
    r = counts_table(r, base, f"Accepted runs (N = {BASE_N})",
                     count_reruns=False)

    # --- table 2: reruns, paired with the same combinations at BASE_N --------
    r += 2
    ws.cell(row=r, column=1,
            value="Reruns at a larger N, against the same combinations at "
                  f"N = {BASE_N}").font = Font(bold=True, size=12)
    ws.cell(row=r + 1, column=1,
            value=f"Each cell: converged at N = {BASE_N} -> converged at the larger N / tested.")
    if rerun.empty:
        ws.cell(row=r + 2, column=1, value="No reruns found.")
        r += 4
    else:
        r = header(r + 2, ["Scenario", "Sheets", "Reruns", "Zero variance"])
        keys = ["country", "asset", "scenario", "index"]
        paired = rerun.merge(base, on=keys, suffixes=("", "_base"))
        for scen in [s for s in scenarios if s in set(rerun["scenario"])] + [None]:
            sub = paired if scen is None else paired[paired["scenario"] == scen]
            st = sub[sub["index"] == "ST"]
            _put(ws, r, 1, "All reruns" if scen is None else scenario_label(scen), bold=scen is None)
            _put(ws, r, 2, None if scen is None else SCEN_SHORT.get(scen, scen))
            _put(ws, r, 3, len(st), align=CENTER, bold=scen is None)
            _put(ws, r, 4, int((st["status"] == "zero_variance").sum()), align=CENTER)
            c = 5
            for *_, conv_col, _ in CRITERIA:
                for ix in INDICES:
                    s = sub[(sub["index"] == ix) & (sub["status"] == "ok")
                            & (sub["status_base"] == "ok")]
                    n = len(s)
                    k_new = int(s[conv_col].astype(bool).sum())
                    k_old = int(s[f"{conv_col}_base"].astype(bool).sum())
                    _put(ws, r, c, f"{k_old} -> {k_new} / {n}" if n else "–",
                         fill=pct_fill(k_new / n) if n else None, align=CENTER, bold=scen is None)
                    c += 1
            r += 1

    # --- table 3: sheet index -------------------------------------------------
    r += 2
    ws.cell(row=r, column=1, value="Sheet index").font = Font(bold=True, size=12)
    r += 1
    _put(ws, r, 1, "Scenario", bold=True, fill=FILL["head"])
    for j, crit in enumerate(CRITERIA):
        _put(ws, r, 2 + j, crit[0], bold=True, fill=FILL["head"], align=CENTER)
    for scen in scenarios:
        r += 1
        _put(ws, r, 1, scenario_label(scen))
        for j, crit in enumerate(CRITERIA):
            name = sheet_name(scen, crit[0])
            cell = _put(ws, r, 2 + j, name, align=CENTER)
            cell.hyperlink = f"#'{name}'!A1"
            cell.font = Font(color="0563C1", underline="single")

    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 13
    for col in range(3, 6 + 2 * len(CRITERIA)):
        ws.column_dimensions[get_column_letter(col)].width = 16
    ws.freeze_panes = "B4"


NOTES = [
    ("Reference", "Sarrazin, F., Pianosi, F. & Wagener, T. (2016). Global Sensitivity Analysis of "
                  "environmental models: Convergence and validation. Environmental Modelling & "
                  "Software 79, 135-152."),
    ("Outcome", f"{OUTCOME} - total expected annual damage, the headline result of the study."),
    ("Bootstrap", f"{RESAMPLES} resamples (with replacement) of the N base rows of each Saltelli "
                  f"sample, fixed seed {SEED}. 95% CI = {CONF_LO}th to {CONF_HI}th percentile of the "
                  "bootstrap distribution. Estimators, output normalisation and row layout are "
                  "SALib's own (first_order / total_order, Saltelli 2010), so the point estimates "
                  "reproduce results/<ISO3>/*_sobol_indices.csv exactly. SALib's *_conf columns "
                  "(Z x std over 100 resamples) are not used."),
    ("S_i", "The bootstrap mean of the index (Sarrazin Eq. 4); used for ranking and for "
            "membership of X0."),
    ("value", f"Index values (Eq. 1): Stat = max over all factors of the CI width (ub - lb). "
              f"Converged if Stat < {VALUE_TOL}."),
    ("rank-rho", "Ranking (Eqs. 2-3): rho(j,k) = sum_i |R_i^j - R_i^k| * max(S_i^j, S_i^k)^2 / "
                 "sum_i max(S_i^j, S_i^k)^2 for every pair of resamples j < k (499,500 pairs); "
                 f"Stat = 95th percentile of rho. Converged if Stat < {RHO_TOL:g}, i.e. the ranking "
                 "of the influential factors moves by less than one position. Rank 1 = most "
                 "influential; ties take their average rank. Negative bootstrap S1 values are "
                 "clipped to 0 in the weight only."),
    ("rank-CI-all", "Alternative ranking test: the number of factor pairs (all pairs, not only "
                    "neighbours) whose 95% CIs overlap. Converged if 0, i.e. every factor's "
                    "position is statistically resolved. Two factors that are both exactly zero "
                    "(CI [0, 0]) count as overlapping: their order is genuinely undetermined."),
    ("rank-CI-top3", "Alternative ranking test, top three by S_i: overlaps between #1-#2, #2-#3 "
                     "and #3-#4. Converged if 0, so both the order within the top three and which "
                     "factors make it are resolved. With three factors only #1-#2 and #2-#3 apply."),
    ("screen-T", f"Screening (Eqs. 4-5): X0 = factors with S_i < T; Stat = max CI width within X0. "
                 f"Converged if Stat < {VALUE_TOL}. Run for T = 0.05 (Sarrazin's choice) and "
                 f"T = 0.01; only T changes, the width cut stays {VALUE_TOL}. An empty X0 is shown "
                 "as '-' and counted as converged. No validation test (Sarrazin Sect. 2.1.4) is run."),
    ("Zero variance", "Combinations whose output never varies (no damage in any run). Sobol "
                      "indices are undefined; shown grey and excluded from every denominator."),
    ("Reruns", f"Combinations re-run at a larger N. Where a scenario has any, its sheet opens "
               "with a 'highest N' matrix: every combination at the largest N it was run at, "
               f"rerun cells in bold. Below it the N = {BASE_N} results, then the reruns alone; "
               "the Overview has the same three views."),
    ("Files", "convergence_stats.csv: one row per combination x N x index with every statistic. "
              "bootstrap_factors.csv: per factor point estimate, bootstrap mean, CI and rank. "
              "archives_not_tested.csv: archives at a sample size other than the accepted N or "
              "the largest rerun."),
    ("Built by", "python -m src.convergence_tests"),
]


def write_notes(wb, other: pd.DataFrame) -> None:
    ws = wb.create_sheet("Notes")
    ws.cell(row=1, column=1, value="Definitions").font = Font(bold=True, size=14)
    r = 3
    for k, v in NOTES:
        _put(ws, r, 1, k, bold=True, fill=FILL["head"], align=Alignment(vertical="top"))
        _put(ws, r, 2, v, align=Alignment(wrap_text=True, vertical="top"))
        r += 1
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 120
    r += 1
    ws.cell(row=r, column=1, value=f"Archives not tested ({len(other)})").font = Font(bold=True)
    for _, o in other.iterrows():
        r += 1
        ws.cell(row=r, column=2, value=f"{o['country']} {o['asset']} {o['scenario']} N={o['n']}")


def write_workbook(out_dir: Path) -> Path:
    stats = pd.read_csv(out_dir / "convergence_stats.csv")
    other_path = out_dir / "archives_not_tested.csv"
    other = pd.read_csv(other_path) if other_path.exists() else pd.DataFrame()
    for _, _, _, _, _, conv_col, _ in CRITERIA:
        stats[conv_col] = stats[conv_col].fillna(False).astype(bool)

    scenarios = sorted(stats["scenario"].unique(), key=scenario_sort_key)
    assets = [a for a in ASSET_CONFIGS if a in set(stats["asset"])]

    wb = Workbook()
    write_overview(wb.active, stats, scenarios)
    for scen in scenarios:
        sub = stats[stats["scenario"] == scen]
        for crit in CRITERIA:
            write_scenario_sheet(wb, scen, crit, highest_n(sub), sub[sub["role"] == "base"],
                                 sub[sub["role"] == "rerun"], assets)
    write_notes(wb, other)
    path = out_dir / "sobol_convergence.xlsx"
    wb.save(path)
    print(f"Saved {path} ({len(wb.sheetnames)} sheets)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", nargs="*", default=None, help="ISO3 codes (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=None, help=f"default {OUT_DIR}")
    ap.add_argument("--resume", "--only-failed", dest="resume", action="store_true",
                    help="test only archives that errored or are new since the CSVs already "
                         "in --out-dir (e.g. reruns), and merge them in")
    ap.add_argument("--workbook-only", action="store_true",
                    help="rebuild the workbook from the CSVs already in --out-dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if not args.workbook_only:
        compute(args, out_dir)
    write_workbook(out_dir)


if __name__ == "__main__":
    main()
