# Uncertainty modeling for critical-infrastructure risk (MIRACA)

Which modeling choices and uncertainties matter most when estimating natural-
hazard risk for critical infrastructure? This project wraps the MIRACA
[AssetRisk_PanEU](https://github.com/miracaEU/AssetRisk_PanEU) risk methodology
(damagescanner-based) in the
[EMA Workbench](https://emaworkbench.readthedocs.io) for exploratory
uncertainty analysis, including Sobol variance decomposition.

**Asset types** (10, registered in `src/curves.py::ASSET_CONFIGS` - extendable):
airports, education, gas, healthcare, oil, ports, power, rail, roads, telecom.
Line, polygon, and point geometries are all supported and can coexist within
one asset (e.g. power mixes cables/lines with point towers/poles and polygon
substations/plants). Object types with no vulnerability curve are dropped from
the analysis rather than given an assumed one - see "Method notes".

**Hazards** (4; which apply to a given asset is decided by
`src/curves.py::applicable_hazards`, not by config):

| Hazard | Return periods | Applies to |
|---|---|---|
| River flood | 10-500, 9 maps | every asset |
| Earthquake | 50-5000, 6 maps | every asset |
| Windstorm | 5-500, 7 maps | every asset except roads and ports, whose wind curve (W7.2) is identically zero - wind does not damage pavements |
| Coastal flood | 1 / 100 / 1000 | coastal countries only (landlocked ones are skipped); streamed from the CoCLiCo STAC catalogue, reuses the river depth-damage curves |

Heat, wildfire and landslide have no damage curves in the MIRACA database
(exposure-only in the reference pipeline).

## Two-stage architecture

The expensive geospatial work and the uncertainty analysis are strictly
separated, so no GIS operation is ever repeated across experiments or
scenarios:

```
Stage 1  (src/preprocess.py, run once per country x asset - shared by every
          scenario below, since scenarios only change Stage-2 sampling)
  exposure parquet x hazard rasters (9 flood RPs + 6 PGA RPs + 7 gust RPs,
  plus coastal fragments streamed from STAC)  ->  per-feature, per-RP list of
  (intensity, exposed quantity) raster-cell fragments. "Quantity" is already
  unit-matched to each feature's geometry (metres / m^2 / count for line /
  polygon / point) so Stage 2 never has to know about geometry at all.
  + FLOPROS protection standard per feature (river design return period)
  + COASTPROS protection per NUTS2 region (coastal)
  + HydroBASINS id + basin RP-shift anchors for 4 warming levels (river)
  ==> data/intermediate/{country}_{asset}_segments.parquet
      data/intermediate/{country}_{asset}_{river,earthquake,windstorm,coastal}_profiles.parquet
      data/intermediate/{country}_{asset}_meta.json

Stage 2  (src/risk_model.py, pure numpy, well under a second per evaluation)
  cached fragments + one sampled parameterization -> damage per (feature, RP)
  per hazard -> protection cutoff (river/coastal/wind) -> warming RP shift
  (river) -> trapezoidal EAD integration
  ==> scalar outcomes (per-hazard & total EAD, per-class EAD, RP100 metrics)
```

Stage 2 reproduces damagescanner's `VectorScanner` damage formula exactly for
every geometry type, the EAD integration / protection / climate-shift logic of
AssetRisk_PanEU's `risk_integration.py`, and the earthquake fragility ->
expected-damage-ratio pipeline of `hazard_earthquake.py`. `src/validate.py`
checks this per asset against fresh reference runs:

| Check | What it verifies |
|---|---|
| A | River RP damage vs a fresh `VectorScanner` run |
| B | EAD integration vs a scalar port of the reference, to machine precision |
| C | Earthquake damage vs a fresh `VectorExposure` + EDR lookup |
| D | Windstorm damage vs a fresh `VectorExposure` + wind-curve interpolation |
| E | Coastal fragments loaded and non-empty (coastal reuses the river-validated damage/EAD path, so there is no separate numerical check) |

Checks skip themselves cleanly when a hazard does not apply to the asset or
country.

## Modeling scenarios

**Every scenario computes exactly ONE hazard** - hazards are never combined, so
their uncertainty structures do not mix. All 15 are built from the same
`compute_risk()`; see `src/ema_model.py::build_model()`.

| Family | Scenarios | Protection treatment | Depth treatment |
|---|---|---|---|
| River flood | `flood_baseline`, `flood_absprot`, `flood_noprot`, each with a `_ds` twin | scale / abs / fixed | additive `depth_offset`, or multiplicative `depth_scale` in the `_ds` twin |
| Coastal flood | `coastal_*`, the same six mirrored | same, against COASTPROS instead of FLOPROS | same |
| Earthquake | `earthquake` | none (as in the reference) | n/a |
| Windstorm | `windstorm`, `windstorm_absprot` | fixed RP50 design standard, or sampled absolute RP | n/a |

The three protection treatments:

- **scale** - `prot_eff = FLOPROS_rp * protection_scale`, `protection_scale in [0, 2]`
- **abs** - `protection_abs_rp in [5, 200]` years applied uniformly to every
  feature, replacing the multiplier entirely
- **fixed** - `protection_scale` held at exactly 1.0 (the recorded FLOPROS
  standards), not sampled, isolating how much the *other* factors matter

**Why the `absprot` variants exist:** the multiplier can never express
uncertainty for a feature FLOPROS records as unprotected, since `0 * x = 0`.
`LUX`/`SVK` roads happen to have 0% unprotected features so this never mattered
there, but `DNK` roads have 7.2% - for those, only the absolute return period
actually varies their protection level.

Coastal scenarios carry no `warming` factor: sea-level rise is a separate
mechanism and is not modelled here.

`run_study.py` runs a default subset of 7 (`DEFAULT_SCENARIOS`) - for each flood
type the absolute-protection and no-protection `_ds` variants, plus earthquake
and the two windstorm scenarios. The rest stay available via `--scenarios`.

## Uncertainty factors

| Factor | Type | Range | Hazard | Scenario(s) |
|---|---|---|---|---|
| `curve_<group>` | categorical | asset- and hazard-specific curve IDs (see below) | all | all (only groups with >1 curve option) |
| `cost_level` | real | [-1, 1] | all | all |
| `aggregation` | categorical | per_cell, mean_depth | all | all |
| `warming` | categorical | current, 1.5C, 2.0C, 3.0C, 4.0C | river | all `flood_*` |
| `protection_scale` | real | [0, 2] | river, coastal | `*_baseline` (held at 1.0 in `*_noprot`) |
| `protection_abs_rp` | real | [5, 200] yr; [25, 200] for wind | river, coastal, windstorm | `*_absprot` |
| `depth_offset` | real | [-0.5, 0.5] m | river, coastal | additive variants |
| `depth_scale` | real | [0.9, 1.1] | river, coastal | `_ds` variants |
| `pga_scale` | real | [0.8, 1.2] | earthquake | `earthquake` |
| `gust_scale` | real | [0.9, 1.1] | windstorm | both windstorm scenarios |

### Curve groups

Within one asset, object types that share the *exact same* curve list are
automatically collapsed into one named group (`src/curves.py::_derive_groups`),
named after the group's first curve ID. A group with only one available curve
carries no uncertainty and is held fixed as an `ema_workbench.Constant` rather
than sampled - only groups with >1 curve become a factor. Roads, for example,
have flood groups `curve_F7_4` (motorway/trunk/primary, 4 curves) and
`curve_F7_8` (secondary and below, 2 curves), plus one earthquake group
`curve_E7_10` (9 curves, shared by every road class); power's 13 object types
split into 6 flood groups (3 samplable), 5 earthquake groups and 4 wind groups.
Run
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
& $py -m src.preprocess  --country LUX --asset roads    # Stage 1, once per (country, asset)
& $py -m src.validate    --country LUX --asset roads    # Checks A-E vs damagescanner + reference

& $py -m src.run_experiments --country LUX --asset roads --scenario flood_absprot_ds --n 3000 --workers 8
& $py -m src.analyze         --country LUX --asset roads --scenario flood_absprot_ds

& $py -m src.run_experiments --country LUX --asset roads --scenario flood_absprot_ds --sampler sobol --n 512 --workers 8
& $py -m src.analyze_sobol   --country LUX --asset roads --scenario flood_absprot_ds
```

### The full study

`run_study.py` drives every (asset, country, scenario) combination in one call:

```powershell
& $py run_study.py --dry-run                           # preview the plan, run nothing
& $py run_study.py --workers 16                        # the full study
& $py run_study.py --assets roads --countries LUX --scenarios flood_absprot_ds --workers 16
```

It is safe to interrupt and re-run: every step whose output already exists
(Stage-1 parquet files, a `_validated.ok` marker, an experiment archive matching
that exact country/asset/scenario/sampler/n) is skipped rather than regenerated
- `--force` overrides this. `analyze`/`analyze_sobol` always re-run, since they
only re-derive a summary from an archive the skip logic already protects.
Progress is logged to `results/run_study_log.jsonl`. A failed step is logged and
the study continues; `--fail-fast` aborts instead.

**Nothing is ever overwritten**: experiment archive filenames include a
timestamp, and every summary CSV/figure is prefixed with
`{country}_{asset}_{scenario}`, so re-runs never collide.

### On the cluster (SLURM)

`submit_miraca_uncertainty_study.sh` submits the study as two chained jobs per
combination - `prep_<C>_<A>` (Stage 1: single-threaded, memory-hungry) and
`run_<C>_<A>` (Stage 2: many CPUs, `--dependency=afterok` on the prep job).
Running them as one job would idle 8-16 CPUs through the whole GIS stage.

```bash
./submit_miraca_uncertainty_study.sh dry        # print the plan, submit nothing
./submit_miraca_uncertainty_study.sh all        # venv + every combination + summary workbook
./submit_miraca_uncertainty_study.sh status     # queue + progress summary
./submit_miraca_uncertainty_study.sh resubmit   # only the combos in failed_combos.txt
./submit_miraca_uncertainty_study.sh addscen    # back-fill newly added scenarios
./submit_miraca_uncertainty_study.sh aggregate  # rebuild the summary workbook
```

`--workers N` is passed straight to `ema_workbench.MultiprocessingEvaluator`.
Each worker loads its own copy of the cached Stage-1 arrays, so `--workers`
trades CPU parallelism for RAM - which matters far more for power or a large
country than for LUX roads. Each stage runs as its own `python -m src.<stage>`
subprocess, so a crash in one combination cannot corrupt or hang a later one.

## Reporting and figures

Built from results already on disk - no new model runs:

```powershell
& $py -m src.aggregate_results   # one workbook across the whole study (project root)
& $py -m src.reference_ead       # cache MIRACA_RISK deterministic totals
& $py -m src.cascade             # nested "unfreeze" ensembles with common random numbers
& $py -m src.plot_pyramid        # trickle-down pyramids + summary tables
& $py -m src.ead_ranges          # full-uncertainty EAD ranges per combination
```

`cascade.py` widens one factor at a time on a shared draw matrix, so the
step-to-step widening is attributable to the newly freed factor rather than to
resampling noise. Because the dominant factors are epistemic (one curve
database, one cost table, one flood map), a pan-European total is built by
evaluating every country at the *same* draw and summing - never by summing
independently drawn per-country samples, and never by summing per-country
percentiles. `plot_pyramid.py` reports all three side by side so the difference
is visible.

## Data (S: drive = /scistor/ivm/eks510)

- Exposure: `S:\eks510\MIRACA_EXPOSURE\{ISO3}_{asset}_exposure.parquet` (EPSG:3035, harmonized OSM)
- River hazard: `S:\eks510\Hazard_data\River_floods\Europe_RP{10..500}_filled_depth.tif` (EPSG:4326, ~90 m, depth in m)
- Earthquake hazard: `S:\eks510\Hazard_data\Earthquakes\PGA_1_{50..5000}_vs30.tif` (EPSG:4326, ~550 m, PGA in g, vs30 soil-adjusted)
- Windstorm hazard: `S:\eks510\Hazard_data\Windstorms\{5..500}yr_wisc_nao_0.59.tif` (gust speed)
- Coastal hazard: streamed from the CoCLiCo STAC catalogue (`cfhp_all`, 2010 LOW_DEFENDED); needs `pystac-client` and network access at preprocessing time
- Vulnerability curves: MIRACA Table D2 xlsx, sheets `F_Vuln_Depth` (flood) and `E_Frag_PGA` (earthquake); curve IDs per asset live in `src/curves.py::FLOOD_CURVES_RAW` / `EQ_CURVES_RAW` / `WIND_CURVES_RAW`, transcribed from AssetRisk_PanEU `src/constants.py`
- Cost values (`MAXDAM_RAW` in `src/curves.py`): also transcribed from AssetRisk_PanEU. Cross-checked against the primary source `Table_D3_Costs_V1.1.0.xlsx` (Nirandjan et al. 2024): most road classes matched its reconstruction-cost rows, but **motorway's mean is ~3x that source**, and the Cost_Database's own cross-reference for the F7.4-F7.7 curves points instead at van Ginkel et al. (2021) construction-cost rows, higher still. Kept as-is for consistency with AssetRisk_PanEU - worth another look before using `cost_level` results for high-class roads in a decision context.
- Protection: `floodProtection_v2019_paper3.tif` (FLOPROS, EPSG:3035, 500 m) for river; `COASTPROS-EU.xlsx` joined to NUTS2 for coastal
- Warming shifts: `basins_abs_shift_return_periods.parquet` (HydroBASINS lev07, new RPs for RP10/100/500 at 1.5/2/3/4 degC)

## Method notes / deliberate choices

- **Object types with no curve are excluded, not assumed.** An object_type
  missing any curve its asset requires (river + earthquake, plus windstorm for
  wind-bearing assets) is dropped in Stage 1 and excluded identically by
  `src/validate.py` and `src/reference_ead.py` - one shared predicate,
  `src/curves.py::is_mapped`. This stays visible: preprocess prints the dropped
  counts and the reference table carries `n_assets_excluded_unmapped`. It can be
  substantial - `oil` has curves for pipelines and storage tanks but not
  petroleum wells, which are most of Albania's oil exposure.
- **EAD integration bounds** follow AssetRisk_PanEU: trapezoid over p = 1/RP
  between the smallest and largest mapped RP only - no tail beyond, no damage
  for more frequent events.
- **Earthquake damage** uses the reference's damage-state model: exceedance
  probabilities per state are collapsed into an expected damage ratio (loss
  weights 0.05 / 0.20 / 0.70 / 0.85 / 1.00) and applied like a vulnerability
  curve. No protection standard for earthquakes.
- **Curve uncertainty is an explicit factor** (pick one curve per group),
  whereas AssetRisk_PanEU averages over the curve ensemble. Explicit sampling
  lets Sobol/feature scoring attribute variance to it.
- **`depth_offset` only perturbs mapped inundated cells**; a positive offset
  cannot expand the flood extent beyond the hazard-map footprint (a negative one
  can shrink it). Extent uncertainty would need multiple hazard maps.
- **Protection standards** are sampled at the raster's native 500 m at feature
  centroids (the pan-EU pipeline coarsens to 5 km for memory reasons).
- **`protection_scale` cannot vary protection for baseline-unprotected
  features** - use an `absprot` scenario for countries/assets with a nonzero
  unprotected share.

## Extending

- **Another country**: `--country {ISO3}` on any script / `run_study.py`, no
  code changes.
- **Another asset type**: add an entry to `FLOOD_CURVES_RAW`, `EQ_CURVES_RAW`,
  `WIND_CURVES_RAW` and `MAXDAM_RAW` in `src/curves.py` (object_type -> curve
  list / [min, mean, max] cost, transcribed from AssetRisk_PanEU
  `src/constants.py`, restricted to the object types actually present in that
  asset's `MIRACA_EXPOSURE` parquet). Everything else - geometry handling, curve
  grouping, scenario building, the orchestrator - is already generic.
- **Curves for a currently unmapped object type**: adding it to those same
  tables is all that is needed; it stops being dropped automatically.
- **Another hazard**: add a block to `config.yml`'s `hazards` (local rasters via
  `dir`/`filename_template`, or `kind: stac` as coastal does), a curve group
  concept in `curves.py`/`preprocess.py`/`risk_model.py`, and a `pga_scale`-style
  intensity factor.
- **More factors**: add a parameter to `compute_risk()` + a `Parameter`/
  `Constant` per scenario in `src/ema_model.py::build_model()` (e.g. EQ
  damage-state loss ratios, EAD tail assumptions, a fragility-of-defenses model
  around the protection threshold).

## Layout

```
config.yml             paths + hazard blocks (country/asset/scenario are CLI overrides)
config.cluster.yml     the same, with cluster-side paths
run_study.py           orchestrator: every (asset, country, scenario), resumable
submit_miraca_uncertainty_study.sh   SLURM submitter (prep + run chained per combo)
src/paths.py           config loader + country/asset/scenario override + filename stems
src/curves.py          AssetConfig registry: curve groups, costs, exclusion predicate
src/preprocess.py      Stage 1 (GIS, once per country+asset, all geometry types)
src/coastal.py         Stage-1 coastal extraction from the CoCLiCo STAC catalogue
src/risk_model.py      Stage 2 (fast vectorized multi-hazard, multi-scenario model)
src/ema_model.py       EMA Workbench model + per-scenario factor/outcome definitions
src/run_experiments.py LHS / Sobol runner for one (country, asset, scenario)
src/adaptive_sobol.py  keep doubling N until the Sobol estimate is precise enough
src/validate.py        equivalence checks A-E vs damagescanner / reference logic
src/analyze.py         feature scoring + figures (LHS results)
src/analyze_sobol.py   Sobol indices (S1/ST) + figures
src/compare_countries.py  cross-country Sobol comparison
src/aggregate_results.py  whole-study roll-up into one workbook
src/cascade.py         nested unfreeze ensembles with common random numbers
src/plot_pyramid.py    trickle-down pyramid figures + summary tables
src/ead_ranges.py      full-uncertainty EAD ranges from archives on disk
src/reference_ead.py   cache of the deterministic MIRACA_RISK totals
src/plot_curves.py     plot every curve the pipeline samples among
data/intermediate/     Stage 1 outputs (parquet, regenerable, gitignored)
results/               per-country subfolders + cross-country roll-ups:
  results/<ISO3>/        experiment archives (.tar.gz), per-combo Sobol/feature
                         CSVs and figures
  results/*.jsonl        global run_study / sobol_convergence logs
  results/cascade/       cascade ensembles feeding the pyramid figures
overview_figures/      pyramids + EAD range figures (regenerable, gitignored)
MIRACA_uncertainty_study_summary.xlsx   the aggregated workbook (project root)
```
