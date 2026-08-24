"""Nested "unfreeze" uncertainty cascade with common random numbers (CRN).

Produces the data behind the trickle-down / pyramid figures: how wide the
plausible range of risk becomes as each uncertainty factor is switched on, one
at a time, on top of the ones already varying.

Design
------
For a factor order f1..fK, step k samples factors f1..fk and holds fk+1..fK at
their nominal value. Step 0 is fully deterministic (a single point); step K is
the full uncertainty of that scenario. The SAME uniform draw matrix backs every
step, so the columns for f1..fk are bit-identical between step k and step k+1 -
the widening between two steps is attributable to the newly-freed factor rather
than to resampling noise.

Common random numbers across countries
--------------------------------------
The dominant factors here are EPISTEMIC, not aleatory: there is one curve
database, one cost table, one aggregation choice, one climate future, one
pan-European flood map. If the true depth-damage curve is F1.3 it is F1.3 in
Germany and in Portugal simultaneously. So a pan-European total must be built
by evaluating every country at the SAME draw of those factors and summing -
never by summing independently-drawn per-country samples (which assumes the
errors diversify away, collapsing the relative band like 1/sqrt(m)) and never
by summing per-country percentiles (which is exactly the comonotonic upper
bound - see src/plot_pyramid.py, which reports all three side by side).

Factors in COUNTRY_SPECIFIC_FACTORS instead get an independent draw per
country, because they are genuinely national quantities rather than one shared
unknown: a single global protection return period answers "what if all of
Europe had standard X", which is a sensitivity sweep, not an uncertainty
representation. For a headline pan-European figure prefer the `_noprot_`
scenarios, which retain the real per-feature FLOPROS/COASTPROS standards and
leave every remaining factor genuinely shared.

Why not ema_workbench
---------------------
The evaluator there samples one Model at a time and cannot express "hold this
subset fixed" or "use this exact draw for a different country", which are the
two things this design is built on. The factor DEFINITIONS are still taken from
src/ema_model.py::build_model, so there is a single source of truth for what
varies in each scenario; only the sampling and the evaluation loop are local.

Cross-asset curve sharing (not needed here, deferred on purpose)
----------------------------------------------------------------
Some curve groups are literally the same unknown in different assets (F2.1-2.3
is power's substation/transformer group AND gas/oil's storage_tank group;
F6.1-6.2 is power's line/pole/tower group AND telecom's tower group). That only
matters when SUMMING ACROSS ASSETS, which these figures never do - each figure
is one asset x one scenario. If an all-asset country total is added later,
share those draws by identical curve LIST, not by group name: _derive_groups
names a group after its first curve id, and education's "F21_6" and
healthcare's "F21_6" have different member sets.

Outputs
-------
results/cascade/{asset}_{scenario}_cascade.parquet   per-draw long table
results/cascade/{asset}_{scenario}_cascade_meta.json reproducibility record

The parquet keeps every draw (not just summary statistics) so the plotting
layer can re-derive any statistic and run the three-way dependence comparison
without re-running the model.

Usage:
    python -m src.cascade --assets power --scenarios flood_noprot_ds --workers 16
    python -m src.cascade --n 2000 --workers 16              # everything
    python -m src.cascade --countries DEU FRA ITA --assets power
    python -m src.cascade --order workflow                   # modelling-chain order
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .curves import ASSET_CONFIGS
from .ema_model import DEFAULT_SCENARIOS, SCENARIOS, build_model, scenario_applies
from .paths import (
    base_stem,
    load_config,
    set_asset_override,
    set_country_override,
    set_scenario_override,
)
from .risk_model import compute_risk, load_model_data

CURVE_PREFIX = "curve_"

# Factors that are a genuinely NATIONAL quantity rather than one shared
# epistemic unknown, so each country gets its own independent draw at the same
# draw index. Everything else is drawn once and applied to every country.
COUNTRY_SPECIFIC_FACTORS = {"protection_abs_rp", "protection_scale"}

# Nominal ("frozen") value for a factor that is not yet switched on. Where a
# factor has a natural no-op that is used, so early cascade steps sit at the
# unbiased central estimate rather than at an arbitrary point.
CONTINUOUS_NOMINAL = {
    "cost_level": 0.0,      # 0 = the mean cost column, between min and max
    "depth_offset": 0.0,    # additive bias, no-op at 0
    "depth_scale": 1.0,     # multiplicative, no-op at 1
    "pga_scale": 1.0,
    "gust_scale": 1.0,
    "protection_scale": 1.0,  # 1 = exactly the FLOPROS/COASTPROS standard
}
CATEGORICAL_NOMINAL = {
    "warming": "current",     # present-day climate
    "aggregation": "per_cell",  # the reference pipeline's method
}
# protection_abs_rp has NO natural no-op - it replaces the design standard
# outright - so it falls back to the midpoint of its sampled range. That choice
# shifts where the early steps SIT, not how wide they are; the summary table
# reports median drift per step so the effect stays visible.

# Conceptual modelling-chain order, used by --order workflow. Matched by exact
# name first, then by the curve_ prefix, then anything unrecognised is appended.
WORKFLOW_ORDER = [
    "warming",                                        # climate future
    "depth_scale", "depth_offset", "pga_scale", "gust_scale",  # hazard map bias
    CURVE_PREFIX,                                     # vulnerability curves
    "cost_level",                                     # cost table
    "aggregation",                                    # method choice
    "protection_scale", "protection_abs_rp",          # protection standard
]

EU_SCOPE = "eu_sum"
COUNTRY_SCOPE = "country"


# ---------------------------------------------------------------------------
# Factor specification
# ---------------------------------------------------------------------------


class FactorSpec:
    """One sampled uncertainty factor, independent of ema_workbench types."""

    __slots__ = ("name", "kind", "lo", "hi", "categories")

    def __init__(self, name, kind, lo=None, hi=None, categories=None):
        self.name = name
        self.kind = kind          # "real" | "categorical"
        self.lo = lo
        self.hi = hi
        self.categories = categories

    def nominal(self):
        if self.kind == "real":
            if self.name in CONTINUOUS_NOMINAL:
                return CONTINUOUS_NOMINAL[self.name]
            return 0.5 * (self.lo + self.hi)
        want = CATEGORICAL_NOMINAL.get(self.name)
        if want is not None and want in self.categories:
            return want
        return self.categories[0]

    def value(self, u: float):
        """Map a uniform draw in [0, 1) to this factor's domain."""
        if self.kind == "real":
            return self.lo + u * (self.hi - self.lo)
        idx = min(int(u * len(self.categories)), len(self.categories) - 1)
        return self.categories[idx]

    def as_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "lo": self.lo, "hi": self.hi,
            "categories": list(self.categories) if self.categories else None,
            "nominal": self.nominal(),
        }


def _categories_of(param) -> list:
    """Extract plain category values from an ema_workbench CategoricalParameter.

    Its .categories holds Category objects carrying .value; older/newer
    versions have handed back bare values, so accept both.
    """
    return [getattr(c, "value", c) for c in param.categories]


def scenario_factors(asset: str, scenario: str) -> tuple[list[FactorSpec], dict]:
    """Sampled factors + fixed constants for one (asset, scenario).

    Read straight off src/ema_model.py::build_model so the factor set here can
    never drift from the one the Sobol/LHS runs used. The factor set depends
    only on asset + scenario, never on country - which is exactly why a
    pan-European pyramid for a fixed (asset, scenario) can label its steps with
    individual named factors instead of bundled factor classes.
    """
    set_asset_override(asset)
    set_scenario_override(scenario)
    cfg = load_config()
    model = build_model(cfg)

    specs: list[FactorSpec] = []
    for p in model.uncertainties:
        cats = getattr(p, "categories", None)
        if cats:
            specs.append(FactorSpec(p.name, "categorical", categories=_categories_of(p)))
        else:
            specs.append(
                FactorSpec(p.name, "real", lo=float(p.lower_bound), hi=float(p.upper_bound))
            )
    constants = {c.name: c.value for c in model.constants}
    return specs, constants


# ---------------------------------------------------------------------------
# Cascade order
# ---------------------------------------------------------------------------


def _mean_st(results_dir: Path, countries, asset: str, scenario: str) -> dict[str, float]:
    """Mean Sobol ST on total EAD across `countries`, from the per-combo CSVs.

    Missing combos are simply skipped, so this works on a partially-finished
    study; an empty result makes the caller fall back to the workflow order.
    """
    acc: dict[str, list[float]] = defaultdict(list)
    for iso in countries:
        path = results_dir / iso / f"{iso}_{asset}_{scenario}_sobol_indices.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = df[df["outcome"] == "total_EAD_MEUR"]
        for name, st in zip(df["factor"], df["ST"]):
            acc[str(name)].append(float(st))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def cascade_order(specs: list[FactorSpec], mode: str, st: dict[str, float]) -> list[str]:
    """Order factors for the cascade.

    "sobol"    - descending mean Sobol ST, so the band widens fastest early and
                 the figure reads as a classic pyramid. Factors with no ST
                 record sort last, alphabetically, for determinism.
    "workflow" - the conceptual modelling chain (climate -> hazard map ->
                 vulnerability -> cost -> method -> protection), which reads as
                 a story rather than a ranking.

    NOTE the cascade is genuinely order-dependent whenever interactions are
    strong - and they are here (sum of S1 ~ 0.44 vs sum of ST ~ 1.7 for
    power/flood, i.e. over half the variance is interaction). The order used is
    recorded in the meta JSON and shown in the figure caption; running both and
    comparing is worthwhile.
    """
    names = [s.name for s in specs]
    if mode == "sobol" and st:
        return sorted(names, key=lambda n: (-st.get(n, -1.0), n))

    rank: dict[str, int] = {}
    for n in names:
        pos = len(WORKFLOW_ORDER)
        for i, key in enumerate(WORKFLOW_ORDER):
            if key == CURVE_PREFIX:
                if n.startswith(CURVE_PREFIX):
                    pos = i
                    break
            elif n == key:
                pos = i
                break
        rank[n] = pos
    return sorted(names, key=lambda n: (rank[n], n))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def latin_hypercube(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Latin hypercube in [0, 1)^k - one stratified permutation per column.

    Implemented locally rather than via scipy.stats.qmc so this module has no
    dependency beyond numpy for the sampling itself, and so the stratification
    is transparent: column j is a random permutation of the n equal-probability
    strata, jittered inside each stratum.
    """
    u = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def step_values(
    specs: list[FactorSpec], order: list[str], u_eff: np.ndarray, step: int
) -> list[dict]:
    """Parameter dicts for every draw at one cascade step.

    Factors at position < step take their sampled value; the rest are pinned to
    nominal. Column j of u_eff belongs to order[j], so the draws for the
    already-freed factors are identical from one step to the next.
    """
    by_name = {s.name: s for s in specs}
    n = u_eff.shape[0]
    frozen = {name: by_name[name].nominal() for name in order[step:]}

    rows = []
    for i in range(n):
        vals = dict(frozen)
        for j in range(step):
            name = order[j]
            vals[name] = by_name[name].value(u_eff[i, j])
        rows.append(vals)
    return rows


def to_compute_kwargs(values: dict, constants: dict) -> dict:
    """Split a flat parameter dict into compute_risk's call signature.

    Mirrors src/ema_model.py::flood_risk_model exactly: everything prefixed
    curve_ becomes an entry of curve_choices (including the single-curve groups
    that build_model declared as Constants), everything else is passed through
    as a scalar keyword.
    """
    merged = {**constants, **values}
    curve_choices = {
        k[len(CURVE_PREFIX):]: v for k, v in merged.items() if k.startswith(CURVE_PREFIX)
    }
    scalars = {k: v for k, v in merged.items() if not k.startswith(CURVE_PREFIX)}
    return {"curve_choices": curve_choices, **scalars}


# ---------------------------------------------------------------------------
# Worker pool - ModelData is loaded ONCE per worker per (country, asset)
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(country: str, asset: str, env: dict) -> None:
    os.environ.update(env)
    set_country_override(country)
    set_asset_override(asset)
    cfg = load_config()
    _WORKER["data"] = load_model_data(cfg)


def _eval_chunk(task):
    """Evaluate a contiguous block of parameter dicts in one worker.

    Chunked rather than one-call-per-draw because compute_risk is fast enough
    that per-task IPC would dominate the runtime.
    """
    tag, kwargs_list = task
    data = _WORKER["data"]
    return tag, [float(compute_risk(data, **kw)["total_EAD_MEUR"]) for kw in kwargs_list]


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def available_countries(cfg: dict, asset: str) -> list[str]:
    """Countries whose Stage-1 segments file exists for this asset.

    Discovered from data/intermediate rather than hardcoded, so a country added
    to the study later is picked up with no code change.
    """
    idir = cfg["intermediate_dir"]
    suffix = f"_{asset}_segments.parquet"
    return sorted(p.name[: -len(suffix)] for p in idir.glob(f"*{suffix}"))


def run_combination(
    asset: str,
    scenario: str,
    countries: list[str],
    n: int,
    workers: int,
    order_mode: str,
    seed: int,
    out_dir: Path,
    scope_label: str = EU_SCOPE,
    chunk: int = 250,
) -> Path | None:
    """Run the full cascade for one (asset, scenario) across `countries`."""
    cfg = load_config()
    specs, constants = scenario_factors(asset, scenario)
    if not specs:
        print(f"  {asset}/{scenario}: no sampled factors - skipping")
        return None

    st = _mean_st(cfg["results_dir"], countries, asset, scenario)
    order = cascade_order(specs, order_mode, st)
    k = len(order)
    n_steps = k + 1
    print(f"  factors ({order_mode}): {' -> '.join(order)}")

    # One shared draw matrix for every country (CRN), plus a per-country matrix
    # used only for the columns of country-specific factors.
    rng = np.random.default_rng(seed)
    u_shared = latin_hypercube(n, k, rng)
    cs_cols = [j for j, name in enumerate(order) if name in COUNTRY_SPECIFIC_FACTORS]

    env = {key: os.environ[key] for key in ("MIRACA_CONFIG",) if key in os.environ}
    frames = []

    for iso in countries:
        set_country_override(iso)
        set_asset_override(asset)
        set_scenario_override(scenario)
        c_cfg = load_config()
        seg = c_cfg["intermediate_dir"] / f"{base_stem(c_cfg)}_segments.parquet"
        if not seg.exists():
            print(f"    {iso}: no Stage-1 output - skipped")
            continue

        u_eff = u_shared
        if cs_cols:
            u_eff = u_shared.copy()
            # Deterministic per-country stream, so a re-run reproduces exactly
            # and adding a country does not disturb the others' draws. crc32
            # rather than hash(): Python randomises string hashing per process
            # unless PYTHONHASHSEED is set, which would silently make these
            # draws irreproducible between runs.
            c_rng = np.random.default_rng(seed + zlib.crc32(iso.encode()))
            u_country = latin_hypercube(n, k, c_rng)
            for j in cs_cols:
                u_eff[:, j] = u_country[:, j]

        tasks = []
        for step in range(n_steps):
            kwargs_list = [
                to_compute_kwargs(v, constants)
                for v in step_values(specs, order, u_eff, step)
            ]
            for ci, block in enumerate(_chunks(kwargs_list, chunk)):
                tasks.append(((step, ci * chunk), block))

        results: dict[int, np.ndarray] = {s: np.empty(n) for s in range(n_steps)}
        try:
            if workers > 1:
                with Pool(workers, initializer=_init_worker, initargs=(iso, asset, env)) as pool:
                    for (step, off), vals in pool.imap_unordered(_eval_chunk, tasks):
                        results[step][off:off + len(vals)] = vals
            else:
                _init_worker(iso, asset, env)
                for task in tasks:
                    (step, off), vals = _eval_chunk(task)
                    results[step][off:off + len(vals)] = vals
        except Exception as exc:  # one bad country must not kill the sweep
            print(f"    {iso}: FAILED ({type(exc).__name__}: {exc})")
            continue

        for step in range(n_steps):
            frames.append(
                pd.DataFrame({
                    "scope": COUNTRY_SCOPE,
                    "country": iso,
                    "asset": asset,
                    "scenario": scenario,
                    "order_mode": order_mode,
                    "step": step,
                    "factor_added": "" if step == 0 else order[step - 1],
                    "draw": np.arange(n, dtype=np.int32),
                    "total_EAD_MEUR": results[step],
                })
            )
        print(f"    {iso}: done ({n_steps} steps x {n} draws)")

    if not frames:
        print(f"  {asset}/{scenario}: nothing ran")
        return None

    df = pd.concat(frames, ignore_index=True)

    # Pan-European total: sum ACROSS COUNTRIES AT EQUAL DRAW INDEX. Because the
    # shared factors carried the same draw in every country, this reproduces
    # the real dependence between national totals instead of assuming one.
    eu = (
        df.groupby(["asset", "scenario", "order_mode", "step", "factor_added", "draw"],
                   as_index=False)["total_EAD_MEUR"].sum()
    )
    eu.insert(0, "country", "EU")
    eu.insert(0, "scope", scope_label)
    df = pd.concat([df, eu[df.columns]], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{asset}_{scenario}_cascade.parquet"
    df.to_parquet(out, index=False)

    meta = {
        "asset": asset,
        "scenario": scenario,
        "order_mode": order_mode,
        "factor_order": order,
        "n_draws": n,
        "n_steps": n_steps,
        "seed": seed,
        "scope_label": scope_label,
        "countries": sorted(df.loc[df["scope"] == COUNTRY_SCOPE, "country"].unique().tolist()),
        "country_specific_factors": sorted(COUNTRY_SPECIFIC_FACTORS & set(order)),
        "factors": [s.as_dict() for s in specs],
        "mean_ST_used_for_order": st,
        "written": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"{asset}_{scenario}_cascade_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"  -> {out}  ({len(df):,} rows)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assets", nargs="+", default=list(ASSET_CONFIGS), choices=list(ASSET_CONFIGS))
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--countries", nargs="+", default=None,
                        help="default: every country with Stage-1 output for that asset")
    parser.add_argument("--n", type=int, default=2000, help="draws per cascade step")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--order", choices=["sobol", "workflow"], default="sobol")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--chunk", type=int, default=250, help="draws per worker task")
    parser.add_argument("--out-dir", default=None, help="default: results/cascade")
    parser.add_argument("--scope-label", default=EU_SCOPE,
                        help="scope name for the aggregate rows. Use a different label "
                             "(e.g. eu_native) for a run built from a genuinely "
                             "pan-European exposure set rather than a sum of countries; "
                             "the plotting layer gives every scope its own figure folder.")
    parser.add_argument("--force", action="store_true", help="recompute combos already written")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else cfg["results_dir"] / "cascade"

    plan = []
    for asset in args.assets:
        # Resolved once per asset: globbing data/intermediate is a network
        # round-trip, and it would otherwise repeat for every scenario.
        pool_countries = args.countries or available_countries(cfg, asset)
        for scenario in args.scenarios:
            # Applicability is per (asset, country): windstorm is skipped for
            # roads/ports, coastal for landlocked countries.
            countries = [c for c in pool_countries if scenario_applies(scenario, asset, c)]
            if countries:
                plan.append((asset, scenario, countries))

    print(f"Cascade plan: {len(plan)} (asset, scenario) combinations, n={args.n}, order={args.order}")
    for asset, scenario, countries in plan:
        out = out_dir / f"{asset}_{scenario}_cascade.parquet"
        print("=" * 70)
        print(f"{asset} / {scenario}  ({len(countries)} countries)")
        if out.exists() and not args.force:
            print("  SKIP (already written; --force to redo)")
            continue
        run_combination(
            asset=asset, scenario=scenario, countries=countries, n=args.n,
            workers=args.workers, order_mode=args.order, seed=args.seed,
            out_dir=out_dir, scope_label=args.scope_label, chunk=args.chunk,
        )

    print("=" * 70)
    print(f"Done. Now build the figures:  python -m src.plot_pyramid")


if __name__ == "__main__":
    sys.exit(main())
