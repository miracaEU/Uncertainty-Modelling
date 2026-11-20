"""Plan the reruns for the DamageScanner polygon fix -> reruns/<date>_polygon_fix/.

The fix (src/ds_version.py) changes polygon exposure on the geographic rasters, so
for every polygon-bearing asset type (airports, education, gas, healthcare, oil,
ports, power) Stage 1 is redone for river, earthquake and windstorm, and then
every scenario of those hazards. Coastal is a projected grid and stays as it is.
Line and point types (roads, rail, telecom) are not touched: they stay on 1.0.1.

Written into the output folder:

  stage1_combos.txt        every (country, asset) to redo, longest Stage 1 first
  wave1_<scenario>.txt     Stage 2 jobs estimated over --wave1-hours, longest first
  wave2_<scenario>.txt     the rest, longest first
  extra_<scenario>.txt     scenarios named with --extras, longest first, submitted
                           after both waves of the default scenarios
  n32768_<scenario>.txt    combinations that were rerun at N = 32768 for the
                           convergence tests; they must be redone at that N too, or
                           "largest N wins" keeps picking the 1.0.1 result
  snapshot.csv             (country, asset, scenario) whose derived files the forced
                           reruns overwrite, for src.snapshot_results
  archives_before.txt      every existing LHS and Sobol archive of those, for the
                           completeness check afterwards
  plan.csv                 one row per job with its runtime estimate

Runtime estimates come from each job's own last run (results/sobol_convergence_log.jsonl
and run_study_log.jsonl). Runs that sampled second-order rows (N(2k+2)) are scaled
to the S1/ST-only design used now (N(k+2)). New scenarios take their twin's time:
the scaled river scenarios flood_noprot_ds's with one factor more, the unprotected
windstorm windstorm's. On the November river batch this method overestimated by
about a quarter, so treat the totals as an upper bound.

    python -m src.plan_polygon_fix --out reruns/20261120_polygon_fix
    python -m src.plan_polygon_fix --out ... --extras earthquake   # plus the wider PGA range
    python -m src.plan_polygon_fix --out ... --extras              # plus every extra
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ds_version import POLYGON_FIX_ASSETS
from .paths import PROJECT_ROOT, load_config

HAZARDS = ("river", "earthquake", "windstorm")
# Every river, earthquake and windstorm scenario the paper reports. Earthquake is the
# narrower PGA range (x0.9-1.1), the default for comparability.
SCENARIOS = [
    "flood_absprot_ds", "flood_absprot_ds_rp25to200", "flood_baseline_ds_sc0p5to2p0",
    "flood_baseline_ds_sc0p75to1p33", "flood_noprot_ds",
    "earthquake_pga0p9to1p1",
    "windstorm_absprot", "windstorm", "windstorm_unprotected",
]
# Run before, outside the default set: the RP 10-100 bracket, the wider PGA range and
# the superseded uniform x0.5-1.5 river scaling. Left on 1.0.1 unless named with
# --extras; they get their own lists (extra_<scenario>.txt), queued after the defaults.
EXTRAS = ["flood_absprot_ds_rp10to100", "earthquake", "flood_baseline_ds_sc0p5to1p5"]
# Never run for the polygon types yet: (scenario, twin it is timed from, extra factors)
NEW = {
    "flood_baseline_ds_sc0p5to2p0": ("flood_noprot_ds", 1),
    "flood_baseline_ds_sc0p75to1p33": ("flood_noprot_ds", 1),
    "windstorm_unprotected": ("windstorm", 0),
}
ARCH = re.compile(r"experiments_([A-Z]{3})_([a-z]+)_(.+)_(lhs|sobol)_n(\d+)_(\d{8}_\d{6})\.tar\.gz$")


def hazard_of(scenario: str) -> str:
    return {"flood": "river", "coastal": "coastal", "earthquake": "earthquake",
            "windstorm": "windstorm"}[scenario.split("_")[0]]


def archives(results: Path) -> pd.DataFrame:
    rows = []
    for c in sorted(os.listdir(results)):
        d = results / c
        if len(c) == 3 and d.is_dir():
            for f in os.listdir(d):
                m = ARCH.match(f)
                if m and m[2] in POLYGON_FIX_ASSETS:
                    rows.append(dict(country=m[1], asset=m[2], scenario=m[3], kind=m[4], n=int(m[5]),
                                     ts=m[6], path=f"{results.name}/{c}/{f}"))
    a = pd.DataFrame(rows)
    return a[a.scenario.map(hazard_of).isin(HAZARDS)]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def timings(results: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Last adaptive Sobol run per (combo, scenario, max N) and last LHS run time."""
    conv = []
    for r in read_jsonl(results / "sobol_convergence_log.jsonl"):
        rounds = r.get("rounds") or [{}]
        conv.append(dict(country=r.get("country"), asset=r.get("asset"), scenario=r.get("scenario"),
                         max_n=r.get("max_n"), k=r.get("n_factors"), s=r.get("elapsed_s"),
                         so=bool(rounds[-1].get("second_order", True)), ts=r.get("ts")))
    conv = pd.DataFrame(conv)
    conv = conv[conv.asset.isin(POLYGON_FIX_ASSETS)].sort_values("ts")
    lhs = []
    for r in read_jsonl(results / "run_study_log.jsonl"):
        if r.get("step") == "run_experiments_lhs" and r.get("ok"):
            c, _, a = r.get("combo", "").partition("/")
            lhs.append(dict(country=c, asset=a, scenario=r.get("scenario"), lhs_s=r.get("elapsed_s"), ts=r.get("ts")))
    lhs = pd.DataFrame(lhs).sort_values("ts").groupby(["country", "asset", "scenario"]).lhs_s.last()
    return conv, lhs


def first_order_seconds(s, k, so, extra=0):
    """A logged Sobol runtime rescaled to N(k+extra+2) rows."""
    base = np.where(so, 2 * k + 2, k + 2)
    return s * (k + extra + 2) / base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extras", nargs="*", default=None, metavar="SCENARIO",
                    help=f"also rerun these of {', '.join(EXTRAS)}; all of them if none is named")
    ap.add_argument("--wave1-hours", type=float, default=3.0)
    args = ap.parse_args()

    extras = [] if args.extras is None else (args.extras or EXTRAS)
    unknown = sorted(set(extras) - set(EXTRAS))
    if unknown:
        raise SystemExit(f"--extras: {unknown} not in {EXTRAS}")
    cfg = load_config()
    results = Path(cfg["results_dir"])
    idir = Path(cfg["intermediate_dir"])
    out = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS + [s for s in EXTRAS if s in extras]
    key = ["country", "asset", "scenario"]

    arch = archives(results)
    sob = arch[arch.kind == "sobol"]
    have = sob.groupby(key).n.max().rename("max_n").reset_index()
    combos = sob[["country", "asset"]].drop_duplicates().sort_values(["asset", "country"])

    # --- Stage 1 list, longest first (elapsed of the last full Stage 1, an upper bound)
    s1 = []
    for c, a in combos.itertuples(index=False):
        meta = idir / f"{c}_{a}_meta.json"
        m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        s1.append(dict(country=c, asset=a, s=m.get("elapsed_s", np.nan), has_meta=meta.exists()))
    s1 = pd.DataFrame(s1).sort_values("s", ascending=False)
    if not s1.has_meta.all():
        raise SystemExit(f"no Stage 1 meta for {s1[~s1.has_meta][['country', 'asset']].values.tolist()}")
    (out / "stage1_combos.txt").write_text(
        f"# Stage 1 for the DamageScanner polygon fix: {len(s1)} combinations, hazards {' '.join(HAZARDS)}, "
        f"longest first (about {s1.s.sum() / 3600:.0f} h in all, at most {s1.s.max() / 3600:.1f} h each)\n"
        + "".join(f"{c} {a}\n" for c, a in s1[["country", "asset"]].itertuples(index=False)))

    # --- Stage 2 jobs at N = 8192
    conv, lhs = timings(results)
    c8 = conv[conv.max_n <= 8192].groupby(key).tail(1).set_index(key)
    jobs = []
    for scen in scenarios:
        if scen in NEW:
            twin, extra = NEW[scen]
            todo = have[have.scenario == twin][["country", "asset"]].assign(scenario=scen)
        else:
            twin, extra = scen, 0
            todo = have[have.scenario == scen][key]
        for c, a, s in todo.itertuples(index=False):
            t = c8.loc[(c, a, twin)] if (c, a, twin) in c8.index else None
            sec = np.nan if t is None else float(first_order_seconds(t.s, t.k, t.so, extra))
            sec += float(lhs.get((c, a, twin), 0.0) or 0.0)
            jobs.append(dict(country=c, asset=a, scenario=s, n=8192, est_h=sec / 3600,
                             rerun=s not in NEW, timed_from=twin))
    jobs = pd.DataFrame(jobs)
    if jobs.est_h.isna().any():
        print(f"NOTE: {int(jobs.est_h.isna().sum())} jobs have no logged runtime; counted as 1 h")
        jobs["est_h"] = jobs.est_h.fillna(1.0)
    # waves 1 and 2 hold the default scenarios; the extras (wave 3) queue after them
    jobs["wave"] = np.where(jobs.scenario.isin(extras), 3, np.where(jobs.est_h > args.wave1_hours, 1, 2))
    for pattern in ("wave*_*.txt", "extra_*.txt"):
        for f in out.glob(pattern):
            f.unlink()
    for (wave, scen), g in jobs.groupby(["wave", "scenario"]):
        g = g.sort_values("est_h", ascending=False)
        name, what = (f"extra_{scen}", "extra") if wave == 3 else (f"wave{wave}_{scen}", f"wave {wave}")
        (out / f"{name}.txt").write_text(
            f"# {scen}, {what}: {len(g)} combinations, longest first ({g.est_h.sum():.0f} job-h estimated)\n"
            + "".join(f"{c} {a}\n" for c, a in g[["country", "asset"]].itertuples(index=False)))

    # --- the N = 32768 convergence reruns of the same combinations
    c32 = conv[conv.max_n > 8192].groupby(key).tail(1).set_index(key)
    big = have[(have.max_n > 8192) & have.scenario.isin(scenarios)]
    rows32 = []
    for c, a, s, n in big.itertuples(index=False):
        t = c32.loc[(c, a, s)] if (c, a, s) in c32.index else None
        sec = np.nan if t is None else float(first_order_seconds(t.s, t.k, t.so))
        rows32.append(dict(country=c, asset=a, scenario=s, n=int(n), est_h=sec / 3600, rerun=True,
                           timed_from=s, wave=4))
    big32 = pd.DataFrame(rows32, columns=list(jobs.columns))
    for f in out.glob("n32768_*.txt"):
        f.unlink()
    for scen, g in big32.groupby("scenario"):
        g = g.sort_values("est_h", ascending=False)
        (out / f"n32768_{scen}.txt").write_text(
            f"# {scen} at N = 32768, after its N = 8192 rerun: {len(g)} combinations ({g.est_h.sum():.0f} job-h)\n"
            + "".join(f"{c} {a}\n" for c, a in g[["country", "asset"]].itertuples(index=False)))

    plan = pd.concat([jobs, big32], ignore_index=True)
    plan.sort_values(["wave", "est_h"], ascending=[True, False]).to_csv(out / "plan.csv", index=False)

    # --- what the forced reruns overwrite, and what exists now
    rerun = plan[plan.rerun][key].drop_duplicates()
    rerun.to_csv(out / "snapshot.csv", index=False)
    before = arch.merge(rerun, on=key)
    (out / "archives_before.txt").write_text("".join(f"{p}\n" for p in sorted(before.path)))

    print(f"Stage 1: {len(s1)} combinations ({s1.s.sum() / 3600:.0f} h, longest {s1.s.max() / 3600:.1f} h)")
    summary = plan.groupby(["wave", "scenario"]).agg(jobs=("country", "size"), job_h=("est_h", "sum"),
                                                     longest_h=("est_h", "max"))
    print(summary.round(1).to_string())
    print(f"Total: {len(plan)} Stage 2 jobs, about {plan.est_h.sum():.0f} job-h (upper bound); "
          f"{len(before)} existing archives listed; {len(rerun)} results to snapshot")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
