# Uncertainty modeling for critical-infrastructure risk (MIRACA)

Which modeling choices and uncertainties matter most when estimating natural-
hazard risk for critical infrastructure? This project wraps the MIRACA
[AssetRisk_PanEU](https://github.com/miracaEU/AssetRisk_PanEU) risk methodology
(damagescanner-based) in the
[EMA Workbench](https://emaworkbench.readthedocs.io) for exploratory
uncertainty analysis, including Sobol variance decomposition.

**Minimal example implemented here: roads in Luxembourg.**
Hazards: **river flooding + earthquakes** — the two hazards with non-trivial
vulnerability/fragility curves for roads in the MIRACA database:

- *Windstorm* is excluded on purpose: the roads wind curve (W7.2, sheet
  `W_Vuln_V10m_3sec`) is **identically zero** — wind does not damage
  pavements — so windstorm EAD for roads is always 0. (Wind becomes relevant
  when expanding to rail/power/telecom, whose W3.x curves are non-zero.)
- *Extreme heat, wildfire, landslide* have no damage curves in the database
  (the pan-EU pipeline treats them as exposure-only metrics).
- *Coastal flooding* does not apply to landlocked Luxembourg.

The structure generalizes to other countries, hazards and infrastructure
types (see *Extending* below).

## Two-stage architecture

The expensive geospatial work and the uncertainty analysis are strictly
separated, so no GIS operation is ever repeated across experiments:

```
Stage 1  (src/preprocess.py, run once per country x asset, ~5 min for LUX)
  roads parquet x hazard rasters (9 flood RPs + 6 PGA RPs)  ->  per-segment,
  per-RP list of (intensity, exposed length) raster-cell fragments
  + FLOPROS protection standard per segment (design return period)
  + HydroBASINS id + basin RP-shift anchors for 4 warming levels (river)
  ==> data/intermediate/LUX_roads_segments.parquet
      data/intermediate/LUX_roads_{river,earthquake}_profiles.parquet

Stage 2  (src/risk_model.py, pure numpy, ~0.1-0.2 s per evaluation)
  cached fragments + one sampled parameterization -> damage per (segment, RP)
  per hazard -> protection cutoff (river) -> warming RP shift (river)
  -> trapezoidal EAD integration
  ==> scalar outcomes (per-hazard & total EAD, per-class EAD, RP100 metrics)
```

Stage 2 reproduces damagescanner's `VectorScanner` damage formula exactly
(validated to ~1e-8 relative, see `src/validate.py`), the EAD integration /
protection / climate-shift logic of AssetRisk_PanEU's `risk_integration.py`
(validated to machine precision against a scalar port), and the earthquake
fragility->expected-damage-ratio pipeline of `hazard_earthquake.py`
(validated by independent recomputation).

## Uncertainty factors (EMA Workbench)

| Factor | Type | Range | Hazard | What it represents |
|---|---|---|---|---|
| `warming` | categorical | current, 1.5C, 2.0C, 3.0C, 4.0C | river | Climate change: basin-level shifts of every return period (anchor maps RP10/100/500 from discharge projections) |
| `curve_main` | categorical | F7.4-F7.7 | river | Depth-damage curve, motorway/trunk/primary |
| `curve_other` | categorical | F7.8-F7.9 | river | Depth-damage curve, secondary and below |
| `eq_curve` | categorical | E7.2-E7.10 | earthquake | Fragility curve (3 damage states, collapsed to expected damage ratio vs PGA) |
| `cost_level` | real | [-1, 1] | both | Reconstruction cost per metre: min (-1) / mean (0) / max (+1), piecewise linear |
| `protection_scale` | real | [0, 2] | river | Design standard: multiplier on the FLOPROS protection return period (0 = none) |
| `depth_offset` | real | [-0.5, 0.5] m | river | Systematic bias in the flood-map water depths |
| `pga_scale` | real | [0.8, 1.2] | earthquake | Systematic bias in the seismic hazard map (PGA) |
| `aggregation` | categorical | per_cell, mean_depth | both | Exposure aggregation order: curve per raster cell then sum, vs. segment-mean intensity first, curve once |

Outcomes: `total_EAD_MEUR`, `EAD_river_MEUR`, `EAD_earthquake_MEUR`,
`EAD_{class}_MEUR` for 5 road classes, `damage_RP100_river_MEUR`,
`exposed_km_RP100_river`.

## How to run

Environment (venv lives outside OneDrive on purpose):

```powershell
uv venv $env:USERPROFILE\.venvs\miraca_uq --python 3.12
uv pip install --python $env:USERPROFILE\.venvs\miraca_uq\Scripts\python.exe -r requirements.txt
$py = "$env:USERPROFILE\.venvs\miraca_uq\Scripts\python.exe"
```

Pipeline (from this folder; paths/config in `config.yml`):

```powershell
& $py -m src.preprocess               # Stage 1: build intermediate data (once)
& $py -m src.validate                 # checks vs damagescanner + reference logic

# Exploratory run (Latin Hypercube) + feature scoring:
& $py -m src.run_experiments --n 3000 --workers 4
& $py -m src.analyze

# Sobol variance decomposition (N * (2k+2) = 20N runs for k=9 factors):
& $py -m src.run_experiments --sampler sobol --n 512 --workers 4
& $py -m src.analyze_sobol
```

## Data (S: drive = /scistor/ivm/eks510)

- Exposure: `S:\eks510\MIRACA_EXPOSURE\{ISO3}_{asset}_exposure.parquet` (EPSG:3035, harmonized OSM)
- River hazard: `S:\eks510\Hazard_data\River_floods\Europe_RP{10..500}_filled_depth.tif` (EPSG:4326, ~90 m depth in m)
- Earthquake hazard: `S:\eks510\Hazard_data\Earthquakes\PGA_1_{50..5000}_vs30.tif` (EPSG:4326, ~550 m, PGA in g, vs30 soil-adjusted)
- Vulnerability curves: MIRACA Table D2 xlsx, sheet `F_Vuln_Depth` (roads: F7.x)
- Fragility curves: `EQ_fragility.xlsx`, sheet `E_Frag_PGA` (roads: E7.2-E7.10, exceedance probabilities on a 0-4 g grid)
- Protection standards: `floodProtection_v2019_paper3.tif` (FLOPROS-based, EPSG:3035, 500 m)
- Warming shifts: `basins_abs_shift_return_periods.parquet` (HydroBASINS lev07 + new RPs for RP10/100/500 at 1.5/2/3/4 degC)

## Method notes / deliberate choices

- **EAD integration bounds** follow AssetRisk_PanEU: trapezoid over
  p = 1/RP between the smallest and largest mapped RP only (river 10-500,
  earthquake 50-5000) — no tail beyond, no damage for more frequent events.
- **Earthquake damage** uses the reference's damage-state model: exceedance
  probabilities per state (minor/moderate/extensive) are collapsed into an
  expected damage ratio (loss weights 0.05 / 0.20 / 0.70) and applied like a
  vulnerability curve. No protection standard for earthquakes (as in the
  reference). LUX PGA is low (0.01-0.05 g at RP476), so absolute EQ risk is
  small — but not zero, and its uncertainty structure is still informative.
- **Curve uncertainty is an explicit factor** (pick one curve per class
  group/hazard), whereas AssetRisk_PanEU averages over the curve ensemble.
  Explicit sampling lets Sobol/feature scoring attribute variance to it.
- **`depth_offset` only perturbs mapped inundated cells**; a positive offset
  cannot expand the flood extent beyond the hazard-map footprint (a negative
  one can shrink it). Extent uncertainty would need multiple hazard maps.
- **Protection standards** are sampled at the raster's native 500 m at segment
  centroids (the pan-EU pipeline coarsens to 5 km for memory reasons).
- In Luxembourg every segment gets flood protection RP ~100-157, which is why
  baseline river EAD is small: most of the integration range sits below the
  design standard. `protection_scale` re-opens that range.

## Extending

- **Another country**: change `country:` in `config.yml`, rerun Stage 1.
- **Another asset type**: extend `src/curves.py` (curve IDs + max damages per
  object type, from AssetRisk_PanEU `constants.py`) and the class grouping;
  Stages 1/2 are geometry-agnostic for line assets. Points/polygons need the
  polygon branch of damagescanner's formula (cell-area instead of length).
  For rail/power/telecom also add windstorm (W3.x curves are non-zero) as a
  third hazard block in `config.yml` + a curve set in `curves.py`.
- **More factors**: add a parameter to `compute_risk()` + a `Parameter` in
  `src/ema_model.py` (e.g. EQ damage-state loss ratios, EAD tail assumptions,
  a fragility-of-defenses model around the protection threshold).

## Layout

```
config.yml            paths + case-study selection + hazard blocks
src/paths.py          config loader
src/curves.py         flood curves, EQ fragility->EDR, unit costs
src/preprocess.py     Stage 1 (GIS, run once, per-hazard profiles)
src/risk_model.py     Stage 2 (fast vectorized multi-hazard model)
src/ema_model.py      EMA Workbench model + factor definitions
src/run_experiments.py  LHS / Sobol experiment runner (parallel capable)
src/validate.py       equivalence checks vs damagescanner / reference logic
src/analyze.py        feature scoring + figures (LHS results)
src/analyze_sobol.py  Sobol indices (S1/ST) + figure (Sobol results)
data/intermediate/    Stage 1 outputs (parquet, regenerable)
results/              experiment archives (.tar.gz) + figures
```
