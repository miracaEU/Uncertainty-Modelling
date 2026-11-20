"""Input spans per infrastructure type and hazard, for the paper's driver-vs-input figures.

  n_curves     distinct vulnerability/fragility curves the model can choose from for
               the hazard (src/curves.py; coastal uses the flood curves)
  cost_range   relative span of the cost factor, (max - min) / mean, weighted over
               the asset's object types by the value they expose at one return period
               (quantity at that RP x mean cost), summed over every country. The cost
               factor moves all features from min to max together, so this is the
               relative swing it gives the damage at that RP.

Reads data/intermediate (segments and profiles); writes paper/data/input_spans.csv
and paper/data/input_spans_by_object_type.csv.

    python -m src.paper_input_spans --workers 8
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .curves import ASSET_CONFIGS, get_asset_config
from .paths import PROJECT_ROOT, load_config

# The return period whose exposure weights the object types, per hazard.
WEIGHT_RP = {"river": 100, "coastal": 100, "earthquake": 476, "windstorm": 100}
CURVE_GROUPS = {"river": "flood_groups", "coastal": "flood_groups",
                "earthquake": "eq_groups", "windstorm": "wind_groups"}
OUT = PROJECT_ROOT / "paper" / "data"


def one(task):
    country, asset = task
    cfg = load_config()
    idir = Path(cfg["intermediate_dir"])
    seg = pd.read_parquet(idir / f"{country}_{asset}_segments.parquet",
                          columns=["object_type", "maxdam_min", "maxdam_mean", "maxdam_max"])
    rows = []
    for hazard, rp in WEIGHT_RP.items():
        p = idir / f"{country}_{asset}_{hazard}_profiles.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p, columns=["seg", "rp", "intensity", "quantity"],
                          filters=[("rp", "==", rp)]).to_pandas()
        t = t[t.intensity > 0]
        if t.empty:
            continue
        q = t.groupby("seg").quantity.sum()
        s = seg.iloc[q.index.to_numpy()].assign(q=q.to_numpy())
        s["value"] = s.q * s.maxdam_mean
        s["swing"] = s.q * (s.maxdam_max - s.maxdam_min)
        g = s.groupby("object_type")[["value", "swing"]].sum().reset_index()
        rows.append(g.assign(country=country, asset=asset, hazard=hazard))
    return pd.concat(rows) if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    tasks = [tuple(re.match(r"([A-Z]{3})_([a-z]+)_segments", f.name).groups())
             for f in sorted(Path(cfg["intermediate_dir"]).glob("*_segments.parquet"))]
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=4) as ex:
        parts = [r for r in ex.map(one, tasks) if r is not None]
    by_type = (pd.concat(parts).groupby(["asset", "hazard", "object_type"])[["value", "swing"]]
               .sum().reset_index())
    by_type["cost_range"] = by_type.swing / by_type.value
    by_type["value_share"] = by_type.value / by_type.groupby(["asset", "hazard"]).value.transform("sum")
    rows = []
    for asset in ASSET_CONFIGS:
        acfg = get_asset_config(asset)
        for hazard, attr in CURVE_GROUPS.items():
            groups = getattr(acfg, attr)
            curves = {c for g in groups.values() for c in g} - {"W7.2"}     # W7.2 is the zero wind curve
            sub = by_type[(by_type.asset == asset) & (by_type.hazard == hazard)]
            rows.append({"asset": asset, "hazard": hazard, "n_curves": len(curves),
                         "n_curve_groups": len(groups),
                         "cost_range": sub.swing.sum() / sub.value.sum() if sub.value.sum() > 0 else np.nan,
                         "exposed_value_eur": sub.value.sum()})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "input_spans.csv", index=False)
    by_type.to_csv(OUT / "input_spans_by_object_type.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
