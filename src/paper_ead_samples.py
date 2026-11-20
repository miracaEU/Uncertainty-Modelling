"""Quantile functions of total EAD per combination, for the paper's range figures.

For every (country, asset, scenario) the newest Sobol archive's A and B base
matrices (2N independent draws, as in src/ead_ranges.py) are reduced to 1,001
quantiles at probabilities 0, 0.001, ..., 1. Read as an equal-weight sample, the
quantiles reproduce the distribution closely enough for histograms and boxes,
at 1/16 to 1/65 of the size of the draws. Also kept: mean, zero share and N.

Writes paper/data/ead_quantiles.parquet (EUR/yr). Only total_EAD_MEUR is read
from each archive.

    python -m src.paper_ead_samples --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .ead_ranges import HEADLINE, MEUR_TO_EUR, extract_ab, find_archives
from .paths import PROJECT_ROOT

OUT = PROJECT_ROOT / "paper" / "data" / "ead_quantiles.parquet"
PROBS = np.linspace(0.0, 1.0, 1001)
SCENARIOS = [
    "flood_absprot_ds", "flood_absprot_ds_rp25to200", "flood_baseline_ds_sc0p5to2p0",
    "flood_baseline_ds_sc0p75to1p33", "flood_noprot_ds",
    "coastal_absprot_ds_slr", "coastal_absprot_ds_slr_rp25to200", "coastal_baseline_ds_slr_sc0p5to2p0",
    "coastal_baseline_ds_slr_sc0p75to1p33", "coastal_noprot_ds_slr",
    "earthquake_pga0p9to1p1", "earthquake", "windstorm_absprot", "windstorm", "windstorm_unprotected",
]


def headline_draws(path: Path) -> np.ndarray | None:
    with tarfile.open(path) as t:
        meta = json.loads(t.extractfile("metadata.json").read().decode("utf-8"))
        for entry in meta.get("outcomes", []):
            if len(entry) >= 3 and entry[0] == "ScalarOutcome" and entry[1] == HEADLINE:
                return np.fromstring(t.extractfile(entry[2]).read().decode("utf-8"), sep="\n")
    return None


def one(task):
    (country, asset, scenario), (path, n_base) = task
    try:
        v = headline_draws(path)
        ab = extract_ab(v, n_base) if v is not None else None
        if ab is None or not len(ab):
            return {"country": country, "asset": asset, "scenario": scenario, "error": "no A+B draws"}
        ab = ab * MEUR_TO_EUR
        return {"country": country, "asset": asset, "scenario": scenario, "archive": path.name,
                "n_base": n_base, "n_draws": len(ab), "mean": float(ab.mean()),
                "zero_share": float((ab <= 0).mean()), "q": np.quantile(ab, PROBS).astype(np.float64)}
    except Exception as exc:                                   # noqa: BLE001
        return {"country": country, "asset": asset, "scenario": scenario,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    args = ap.parse_args()
    slots = find_archives(PROJECT_ROOT / "results", set(), set(), set(args.scenarios))
    tasks = [(key, slot["sobol"]) for key, slot in sorted(slots.items()) if "sobol" in slot]
    print(f"{len(tasks)} combinations with a Sobol archive", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, row in enumerate(ex.map(one, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    if len(bad):
        print(bad[["country", "asset", "scenario", "error"]].to_string(index=False))
    good = df.drop(index=bad.index).drop(columns=["error"], errors="ignore")
    qcols = pd.DataFrame(np.vstack(good["q"].to_numpy()), index=good.index,
                         columns=[f"q{i:04d}" for i in range(len(PROBS))])
    out = pd.concat([good.drop(columns="q"), qcols], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, OUT)                       # readers never see a half-written file
    print(f"wrote {OUT} ({len(out)} combinations, {len(bad)} failed)")


if __name__ == "__main__":
    main()
