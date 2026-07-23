# Uncertainty modeling for critical-infrastructure risk (MIRACA)

Which modeling choices and uncertainties matter most when estimating natural-
hazard risk for critical infrastructure? This project wraps the MIRACA
[AssetRisk_PanEU](https://github.com/miracaEU/AssetRisk_PanEU) risk methodology
(damagescanner-based) in the
[EMA Workbench](https://emaworkbench.readthedocs.io) for exploratory
uncertainty analysis, including Sobol variance decomposition.

**Asset types:** roads, airports, education, power (registered in
`src/curves.py::ASSET_CONFIGS` - extendable). Line, polygon, and point
geometries are all supported and can coexist within one asset (e.g. power
mixes cables/lines with point towers/poles and polygon substations/plants).

**Hazards:** river flooding + earthquakes - the two with non-trivial
vulnerability/fragility curves for roads in the MIRACA database. Windstorm is
excluded: the roads wind curve (W7.2) is identically zero (wind does not
damage pavements); it becomes relevant once an asset type with non-zero W
curves (e.g. rail, power) is added. Heat/wildfire/landslide have no damage
curves in the database (exposure-only in the reference pipeline).

## Two-stage architecture

The expensive geospatial work and the uncertainty analysis are strictly
separated, so no GIS operation is ever repeated across experiments or
scenarios:

```
Stage 1  (src/preprocess.py, run once per country x asset - shared by every
          scenario below, since scenarios only change Stage-2 sampling)
  exposure parquet x hazard rasters (9 flood RPs + 6 PGA RPs)  ->  per-
  feature, per-RP list of (intensity, exposed quantity) raster-cell
  fragments. "Quantity" is already unit-matched to each feature's geometry
  (metres / m^2 / count for line / polygon / point) so Stage 2 never has to
  know about geometry at all.
  + FLOPROS protection standard per feature (design return period)
  + HydroBASINS id + basin RP-shift anchors for 4 warming levels (river)
  ==> data/intermediate/{country}_{asset}_segments.parquet
      data/intermediate/{country}_{asset}_{river,earthquake}_profiles.parquet

Stage 2  (src/risk_model.py, pure numpy, well under a second per evaluation)
  cached fragments + one sampled parameterization -> damage per (feature, RP)
  per hazard -> protection cutoff (river) -> warming RP shift (river)
  -> trapezoidal EAD integration
  ==> scalar outcomes (per-hazard & total EAD, per-class EAD, RP100 metrics)
```

Stage 2 reproduces damagescanner's `VectorScanner` damage formula exactly
for every geometry type (validated per-asset in `src/validate.py` against a
fresh `VectorScanner`/`VectorExposure` run - see Check A/C), the EAD
integration / protection / climate-shift logic of AssetRisk_PanEU's
`risk_integration.py` (validated to machine precision against a scalar port
- Check B), and the earthquake fragility -> expected-damage-ratio pipeline
of `hazard_earthquake.py`.

## Modeling scenarios

Four scenarios, all built from the same `compute_risk()` - see
`src/ema_model.py::build_model()`:

| Scenario | Hazards | What it isolates |
|---|---|---|
| `baseline` | both | Full model: protection as a FLOPROS multiplier `protection_scale in [0, 2]`. |
| `abs_protection` | both | Flood protection sampled as an **absolute** return period `protection_abs_rp in [5, 200]` years, applied uniformly to every feature - replaces `protection_scale` entirely. |
| `flood_no_protection` | river only | `protection_scale` held fixed at exactly 1.0 (the recorded FLOPROS design standards), not sampled - isolates how much the *other* flood factors matter once protection uncertainty is set aside. Earthquake is not computed at all (not just excluded from reporting - the numpy work is skipped for speed). |
| `earthquake_only` | earthquake only | Isolates the earthquake-only factors. River is not computed. |

**Why `abs_protection` exists:** `protection_scale` is a *multiplier*
(`prot_eff = FLOPROS_rp * protection_scale`), so for any feature with a
FLOPROS baseline of exactly 0 (no recorded protection), every sampled
`protection_scale` still gives `0 * x = 0` - the multiplier can never
express uncertainty for features FLOPROS marks as unprotected. `LUX`/`SVK`
roads happen to have 0% unprotected features so this never mattered there,
but `DNK` roads have 7.2% unprotected - for those, `abs_protection` is the
only one of the four scenarios that actually varies their protection level.

**Comparing `baseline` against `flood_no_protection` + `earthquake_only`
run separately** is how to check whether the two hazards interact: if
`baseline`'s per-hazard Sobol indices (`EAD_river_MEUR`, `EAD_earthquake_MEUR`
rows) closely match the corresponding single-hazard scenario's indices for
the shared factors (`cost_level`, `aggregation`), the hazards behave
independently; if they diverge, joint sampling of the shared factors is
creating cross-hazard interaction.

## Uncertainty factors

| Factor | Type | Range | Hazard | Scenario(s) |
|---|---|---|---|---|
| `warming` | categorical | current, 1.5C, 2.0C, 3.0C, 4.0C | river | baseline, abs_protection, flood_no_protection |
| `curve_<group>` | categorical | asset- and hazard-specific curve IDs (see below) | both | all (only for groups with >1 curve option - see "Curve groups") |
| `cost_level` | real | [-1, 1] | both | all |
| `protection_scale` | real | [0, 2] | river | baseline |
| `protection_abs_rp` | real | [5, 200] years | river | abs_protection |
| `depth_offset` | real | [-0.5, 0.5] m | river | baseline, abs_protection, flood_no_protection |
| `pga_scale` | real | [0.8, 1.2] | earthquake | baseline, abs_protection, earthquake_only |
| `aggregation` | categorical | per_cell, mean_depth | both | all |

### Curve groups

Within one asset, object types that share the *exact same* curve list are
automatically collapsed into one named group (`src/curves.py::_derive_groups`),
named after the group's first curve ID. A group with only one available
curve carries no uncertainty and is held fixed as an `ema_workbench.Constant`
rather than sampled - only groups with >1 curve become a factor. Example for
roads: flood groups `curve_F7_4` (motorway/trunk/primary, 4 curves) and
`curve_F7_8` (secondary and below, 2 curves); one earthquake group
`curve_E7_10` (9 curves, shared by every road class). Power has 6 flood
groups (3 samplable, 3 single-curve/fixed) and 5 earthquake groups, since its
13 object types split into several distinct curve sets. Run
`python -c "from src.curves import ASSET_CONFIGS as A; print(A['power'].flood_groups)"`
to see any asset's exact grouping.

## How to run

Environment (venv lives outside OneDrive on purpose):

```powershell
uv venv $env:USERPROFILE\.venvs\miraca_uq --python 3.12
uv pip install --python $env:USERPROFILE\.venvs\miraca_uq\Scripts\python.exe -r requirements.txt
```

### One combination at a time

```powershell
$py = "$env:USERPROFILE\.venvs\miraca_uq\Scripts\python.exe"
& $py -m src.preprocess  --country LUX --asset roads             # Stage 1, once per (country, asset)
& $py -m src.validate    --country LUX --asset roads              # checks vs damagescanner + reference logic

& $py -m src.run_experiments --country LUX --asset roads --scenario baseline --n 3000 --workers 8
& $py -m src.analyze         --country LUX --asset roads --scenario baseline

& $py -m src.run_experiments --country LUX --asset roads --scenario baseline --sampler sobol --n 512 --workers 8
& $py -m src.analyze_sobol   --country LUX --asset roads --scenario baseline
```

### The full study (orchestrator)

`run_study.py` drives every (asset, country, scenario) combination in one
call - default order matches the study design (roads fully through all
scenarios for LUX then DNK, then airports, then education, then power):

```powershell
& $py run_study.py --dry-run                          # preview the plan, run nothing
& $py run_study.py --workers 16                        # the full study
& $py run_study.py --assets roads --countries LUX --scenarios baseline --workers 16   # one slice (e.g. one SLURM array task)
```

It is safe to interrupt and re-run: every step whose output already exists
(Stage-1 parquet files, a validated marker, an experiment archive matching
that exact country/asset/scenario/sampler/n) is skipped rather than
regenerated - `--force` overrides this. `analyze`/`analyze_sobol` always
re-run (cheap, and just re-derive a summary from the archive that the skip
logic already protects). Progress is logged to
`results/run_study_log.jsonl` in addition to stdout. By default a failed
step is logged and the study continues with the next combination; pass
`--fail-fast` to abort immediately instead.

**Nothing is ever overwritten**: every experiment archive filename includes
a timestamp, and every summary CSV/figure is prefixed with
`{country}_{asset}_{scenario}`, so different countries, assets, and
scenarios - and repeated re-runs of the same one - never collide.

**Cluster / multi-processor notes**: `--workers N` is passed straight
through to `ema_workbench.MultiprocessingEvaluator`. Each worker process
independently loads its own copy of the cached Stage-1 arrays for that
(country, asset) - `--workers` trades CPU parallelism for RAM, which matters
more for the larger asset/country combinations (e.g. power, or a large
country) than it did for LUX roads. Each pipeline stage runs as its own
`python -m src.<stage>` subprocess (via `run_study.py`), so a crash in one
combination cannot corrupt or hang a later one.

## Data (S: drive = /scistor/ivm/eks510)

- Exposure: `S:\eks510\MIRACA_EXPOSURE\{ISO3}_{asset}_exposure.parquet` (EPSG:3035, harmonized OSM)
- River hazard: `S:\eks510\Hazard_data\River_floods\Europe_RP{10..500}_filled_depth.tif` (EPSG:4326, ~90 m depth in m)
- Earthquake hazard: `S:\eks510\Hazard_data\Earthquakes\PGA_1_{50..5000}_vs30.tif` (EPSG:4326, ~550 m, PGA in g, vs30 soil-adjusted)
- Vulnerability curves: MIRACA Table D2 xlsx, sheet `F_Vuln_Depth` (flood depth-damage) and `E_Frag_PGA` (earthquake fragility) - all curve IDs per asset live in `src/curves.py::FLOOD_CURVES_RAW` / `EQ_CURVES_RAW`, transcribed from AssetRisk_PanEU `src/constants.py`
- Cost values (`MAXDAM_RAW` in `src/curves.py`): also transcribed from AssetRisk_PanEU `src/constants.py`. Cross-checked against the primary source, `Table_D3_Costs_V1.1.0.xlsx` (Nirandjan et al. 2024) - roads' trunk/tertiary/other-roads figures matched the "maximum damage, reconstruction cost" rows (Kok et al. 2005; Briene et al. 2002) closely, but motorway's mean was ~3x that source and the `Cost_Database`'s own curve-ID cross-reference for the F7.4-F7.7 curves we use points instead at van Ginkel et al. (2021) "construction cost" rows, which run far higher still. Kept as-is for consistency with the rest of the AssetRisk_PanEU pipeline; flagged here as a number worth another look if you use `cost_level` results for roads' high-class classes in a decision context.
- Protection standards: `floodProtection_v2019_paper3.tif` (FLOPROS-based, EPSG:3035, 500 m)
- Warming shifts: `basins_abs_shift_return_periods.parquet` (HydroBASINS lev07 + new RPs for RP10/100/500 at 1.5/2/3/4 degC)

## Method notes / deliberate choices

- **EAD integration bounds** follow AssetRisk_PanEU: trapezoid over
  p = 1/RP between the smallest and largest mapped RP only (river 10-500,
  earthquake 50-5000) - no tail beyond, no damage for more frequent events.
- **Earthquake damage** uses the reference's damage-state model: exceedance
  probabilities per state are collapsed into an expected damage ratio (loss
  weights 0.05 / 0.20 / 0.70 / 0.85 / 1.00) and applied like a vulnerability
  curve. No protection standard for earthquakes (as in the reference).
- **Curve uncertainty is an explicit factor** (pick one curve per group),
  whereas AssetRisk_PanEU averages over the curve ensemble. Explicit sampling
  lets Sobol/feature scoring attribute variance to it.
- **`depth_offset` only perturbs mapped inundated cells**; a positive offset
  cannot expand the flood extent beyond the hazard-map footprint (a negative
  one can shrink it). Extent uncertainty would need multiple hazard maps.
- **Protection standards** are sampled at the raster's native 500 m at
  feature centroids (the pan-EU pipeline coarsens to 5 km for memory
  reasons).
- **`protection_scale` cannot vary protection for baseline-unprotected
  features** (see "Why `abs_protection` exists" above) - use that scenario
  specifically for countries/assets with a nonzero unprotected share.

## Extending

- **Another country**: `--country {ISO3}` on any script / `run_study.py`,
  no code changes.
- **Another asset type**: add an entry to `FLOOD_CURVES_RAW`, `EQ_CURVES_RAW`,
  and `MAXDAM_RAW` in `src/curves.py` (object_type -> curve list / [min,
  mean, max] cost, transcribed from AssetRisk_PanEU `src/constants.py`,
  restricted to the object types actually present in that asset's
  `MIRACA_EXPOSURE` parquet). Everything else - geometry handling, curve
  grouping, scenario building, the orchestrator - is already generic.
- **Windstorm** (for an asset with non-zero W curves, e.g. rail/power W3.x):
  add a `wind` block to `config.yml`'s `hazards`, a `wind_curve_groups`
  concept mirroring flood/eq in `curves.py`/`preprocess.py`/`risk_model.py`,
  and a `pga_scale`-style `wind_scale` factor.
- **More factors**: add a parameter to `compute_risk()` + a `Parameter`/
  `Constant` per scenario in `src/ema_model.py::build_model()` (e.g. EQ
  damage-state loss ratios, EAD tail assumptions, a fragility-of-defenses
  model around the protection threshold, an additive protection floor for
  baseline-unprotected features as an alternative to `abs_protection`).

## Layout

```
config.yml            paths + hazard blocks (country/asset/scenario are CLI overrides, not here)
run_study.py           orchestrator: every (asset, country, scenario) combination, resumable
src/paths.py           config loader + country/asset/scenario override + filename-stem helpers
src/curves.py          AssetConfig registry: curve groups, costs, geometry-aware, any asset type
src/preprocess.py      Stage 1 (GIS, run once per country+asset, all geometry types)
src/risk_model.py      Stage 2 (fast vectorized multi-hazard, multi-scenario model)
src/ema_model.py       EMA Workbench model + per-scenario factor/outcome/Constant definitions
src/run_experiments.py LHS / Sobol experiment runner for one (country, asset, scenario)
src/validate.py        equivalence checks vs damagescanner / reference logic, any asset
src/analyze.py         feature scoring + figures (LHS results), factor set auto-detected
src/analyze_sobol.py   Sobol indices (S1/ST) + figure (Sobol results), outcomes auto-detected
src/compare_countries.py  cross-country Sobol comparison (baseline scenario)
data/intermediate/     Stage 1 outputs (parquet, regenerable, gitignored)
results/               per-country subfolders + cross-country roll-ups:
  results/<ISO3>/        one folder per country: its experiment archives
                         (.tar.gz, gitignored), per-combo Sobol/feature CSVs
                         and figures/ (kept)
  results/*.jsonl        global run_study / sobol_convergence logs (all countries)
  results/*.csv, figures/  compare_countries + vulnerability-curve outputs
  (the aggregated workbook is written to the project root)
```
