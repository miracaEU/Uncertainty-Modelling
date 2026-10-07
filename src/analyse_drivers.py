"""Why does one factor come out on top? Curve count, curve spread, or cost range.

Tests three candidate explanations for which uncertainty factor dominates a
(country, asset, hazard) combination:

  n_curves        how many candidate curves the group offers
  spread_flat     how far apart those curves lie, averaged over the whole
                  intensity axis - a property of the curve set alone
  spread_weighted the same dispersion, but weighted by where the hazard
                  actually puts intensity for THAT country and asset
  cost_ratio      max/min of the asset's maximum-damage triple

spread_flat and n_curves vary only by (asset, hazard); spread_weighted also
varies by country, which is what lets it explain within-cell variation.

Restricted to the scenarios where the protection standard is NOT sampled. Where
it is, protection dominates nearly everything and would mask the contest this is
about.

Dispersion is the relative range across candidates, (max - min) / mean, at a
given intensity. Points where the mean damage fraction is below 1% are skipped:
every curve is near zero there, so the ratio measures numerical noise rather
than real disagreement.

Weighting combines two things, because both decide whether an intensity matters:
the exposed quantity at that intensity, and the probability of the return period
it belongs to. The RP weight is the trapezoidal exceedance-probability increment
used by the EAD integral itself, so an intensity reached only at RP 1000 counts
far less than one reached at RP 10.

Usage:
    python -m src.analyse_drivers
    python -m src.analyse_drivers --assets education healthcare --out-dir .
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .curves import (
    ASSET_CONFIGS,
    get_asset_config,
    load_eq_edr_tables,
    load_flood_curves,
    load_wind_curves,
)
from .ema_model import SCENARIO_HAZARD
from .paths import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT_ROOT / "MIRACA_uncertainty_study_summary.xlsx"
SHEET = "All_Sobol_Indices"
OUTCOME = "total_EAD_MEUR"

# Protection is held fixed in these, so curve and cost compete on equal terms.
SCENARIOS = ["flood_noprot_ds", "coastal_noprot_ds", "earthquake", "windstorm"]

# Above this many profile rows, sample instead of reading all of them. The
# weighted dispersion is a mean over the intensity distribution, which a large
# random sample estimates to far better precision than the metric needs.
MAX_ROWS = 2_000_000
MIN_MEAN_DAMAGE = 0.01


def curve_matrix(cfg: dict, asset: str, hazard: str):
    """(intensities, matrix) of every candidate curve for this asset and hazard.

    Rows are curves, columns intensities. Coastal reuses the flood curves.
    Returns (None, None) where the hazard has no curve set.
    """
    ac = get_asset_config(asset)
    if hazard in ("river", "coastal"):
        groups, kind = ac.flood_groups, "flood"
    elif hazard == "earthquake":
        groups, kind = ac.eq_groups, "eq"
    else:
        groups, kind = ac.wind_groups, "wind"
    ids = sorted({c for v in groups.values() for c in v})
    if len(ids) < 2:
        return None, None
    if kind == "flood":
        df = load_flood_curves(cfg["vulnerability_path"], ids)
    elif kind == "wind":
        df = load_wind_curves(cfg["vulnerability_path"], ids)
    else:
        tabs = load_eq_edr_tables(cfg["fragility_path"], ids)
        xs = tabs[ids[0]][0]
        return xs, np.vstack([tabs[c][1] for c in ids])
    return df.index.to_numpy(float), np.vstack([df[c].to_numpy(float) for c in ids])


def relative_range(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-intensity (max - min)/mean across candidates, and where it is usable."""
    mean = M.mean(axis=0)
    ok = mean > MIN_MEAN_DAMAGE
    rng = np.zeros_like(mean)
    rng[ok] = (M.max(axis=0) - M.min(axis=0))[ok] / mean[ok]
    return rng, ok


def spread_flat(M: np.ndarray) -> float:
    """Dispersion averaged over the intensity axis, ignoring where the hazard is."""
    rng, ok = relative_range(M)
    return float(rng[ok].mean()) if ok.sum() >= 5 else np.nan


def rp_weights(rps: np.ndarray) -> dict[float, float]:
    """Trapezoidal exceedance-probability increment per return period.

    The same weighting the EAD integral applies, so an intensity reached only at
    a rare return period contributes proportionally little.
    """
    rps = np.asarray(sorted(set(float(r) for r in rps if r > 0)))
    p = 1.0 / rps
    w = np.zeros_like(p)
    for i in range(len(p)):
        lo = p[i + 1] if i + 1 < len(p) else 0.0
        hi = p[i - 1] if i - 1 >= 0 else p[i]
        w[i] = abs(hi - lo) / 2.0 if len(p) > 1 else p[i]
    return dict(zip(rps, w))


def spread_weighted(cfg: dict, country: str, asset: str, hazard: str,
                    xs: np.ndarray, M: np.ndarray, rng_state: int = 0) -> float:
    """Dispersion weighted by where this country's hazard actually puts intensity."""
    path = cfg["intermediate_dir"] / f"{country}_{asset}_{hazard}_profiles.parquet"
    if not path.exists():
        return np.nan
    df = pd.read_parquet(path, columns=["rp", "intensity", "quantity"])
    if df.empty:
        return np.nan
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=rng_state)
    w_rp = rp_weights(df["rp"].to_numpy())
    w = df["quantity"].to_numpy(float) * df["rp"].map(w_rp).to_numpy(float)
    inten = df["intensity"].to_numpy(float)

    rng, ok = relative_range(M)
    # Zero outside the usable band so low-intensity rows contribute no weight.
    disp = np.where(ok, rng, 0.0)
    usable = np.interp(inten, xs, ok.astype(float)) > 0.5
    vals = np.interp(inten, xs, disp)
    w = np.where(usable, w, 0.0)
    tot = w.sum()
    return float((vals * w).sum() / tot) if tot > 0 else np.nan


def build_features(cfg: dict, assets: list[str], countries: list[str] | None) -> pd.DataFrame:
    asi = pd.read_excel(WORKBOOK, sheet_name=SHEET)
    d = asi[(asi["outcome"] == OUTCOME) & (asi["scenario"].isin(SCENARIOS))].copy()
    d["hazard"] = d["scenario"].map(SCENARIO_HAZARD)
    d = d[d["asset"].isin(assets)]
    if countries:
        d = d[d["country"].isin(countries)]
    d["fam"] = d["factor_code"].astype(str).map(
        lambda c: "curve" if c.startswith("curve_") else c)

    key = ["country", "asset", "hazard"]
    per = d.groupby(key + ["fam"], as_index=False)["ST"].sum()
    wide = per.pivot_table(index=key, columns="fam", values="ST", fill_value=0.0)
    tot = d.groupby(key)["ST"].sum()
    top = d.sort_values("ST", ascending=False).groupby(key).head(1).set_index(key)["fam"]
    combo = pd.DataFrame({
        "curve_ST": wide.get("curve", 0.0),
        "cost_ST": wide.get("cost_level", 0.0),
        "total_ST": tot,
        "top": top,
    })
    combo = combo[combo["total_ST"] > 0].reset_index()
    combo["curve_share"] = combo["curve_ST"] / combo["total_ST"]
    combo["cost_share"] = combo["cost_ST"] / combo["total_ST"]

    cache: dict[tuple[str, str], tuple] = {}
    rows = []
    for i, r in enumerate(combo.itertuples(), 1):
        ck = (r.asset, r.hazard)
        if ck not in cache:
            xs, M = curve_matrix(cfg, r.asset, r.hazard)
            ac = get_asset_config(r.asset)
            md = np.array(list(ac.maxdam.values()), dtype=float)
            cache[ck] = (xs, M, float(np.median(md[:, 2] / np.maximum(md[:, 0], 1e-9))))
        xs, M, cost_ratio = cache[ck]
        if M is None:
            continue
        rows.append({
            "country": r.country, "asset": r.asset, "hazard": r.hazard,
            "curve_share": r.curve_share, "cost_share": r.cost_share, "top": r.top,
            "n_curves": M.shape[0],
            "spread_flat": spread_flat(M),
            "spread_weighted": spread_weighted(cfg, r.country, r.asset, r.hazard, xs, M),
            "cost_ratio": cost_ratio,
        })
        if i % 100 == 0:
            print(f"  {i}/{len(combo)} combinations")
    return pd.DataFrame(rows)


def report(f: pd.DataFrame) -> None:
    def corr(a, b):
        m = f[[a, b]].dropna()
        return m[a].corr(m[b]), m[a].corr(m[b], method="spearman"), len(m)

    print(f"\nn = {len(f)} (country, asset, hazard) combinations, "
          f"{f.groupby(['asset', 'hazard']).ngroups} asset-hazard cells\n")
    print("correlation with the curve's share of total effect:")
    for p, label in (("n_curves", "number of candidate curves"),
                     ("spread_flat", "spread, flat over the intensity axis"),
                     ("spread_weighted", "spread, weighted by where the hazard is"),
                     ("cost_ratio", "cost range (max/min)")):
        r, rho, n = corr(p, "curve_share")
        print(f"  {label:42s} r = {r:+.2f}  rho = {rho:+.2f}  (n={n})")
    print("\ncorrelation with the cost factor's share:")
    for p, label in (("n_curves", "number of candidate curves"),
                     ("spread_weighted", "spread, weighted by where the hazard is"),
                     ("cost_ratio", "cost range (max/min)")):
        r, rho, n = corr(p, "cost_share")
        print(f"  {label:42s} r = {r:+.2f}  rho = {rho:+.2f}  (n={n})")

    y = f["curve_share"].to_numpy()
    tss = ((y - y.mean()) ** 2).sum()
    print("\nvariance in the curve's share explained by group means:")
    for cols, label in ((["asset"], "asset alone"), (["hazard"], "hazard alone"),
                        (["country"], "country alone"),
                        (["asset", "hazard"], "asset x hazard")):
        g = f.groupby(cols)["curve_share"].transform("mean").to_numpy()
        print(f"  {label:42s} R2 = {1 - ((y - g) ** 2).sum() / tss:.2f}")

    m = f[["spread_weighted", "n_curves", "cost_ratio", "curve_share"]].dropna()
    if len(m) > 10:
        X = np.column_stack([np.ones(len(m))] + [
            ((m[c] - m[c].mean()) / m[c].std()).to_numpy()
            for c in ("spread_weighted", "n_curves", "cost_ratio")])
        yy = m["curve_share"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r2 = 1 - ((yy - X @ beta) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        print("\nall three together (standardised predictors), on curve_share:")
        for name, b in zip(["intercept", "spread_weighted", "n_curves", "cost_ratio"], beta):
            print(f"  {name:18s} {b:+.3f}")
        print(f"  R2 = {r2:.2f}  n = {len(m)}")


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=sorted(ASSET_CONFIGS))
    ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: overview_figures (writes driver_features.csv)")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "overview_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    f = build_features(cfg, args.assets, args.countries)
    out = out_dir / "driver_features.csv"
    f.to_csv(out, index=False)
    report(f)
    print(f"\nper-combination table written to {out}")


if __name__ == "__main__":
    main()
