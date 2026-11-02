# Future versions

Candidate changes for a later version of this study. Each entry says what the
current version does, why it might change, and what the change would take.

This is not a plan and nothing here is committed. It exists so that gaps found
while running the study are written down at the moment they are found, rather
than surviving as something somebody once said in a meeting.

Three neighbouring documents, so entries land in the right place:

- a choice that is **settled and expected to stay** belongs in README,
  "Method notes / deliberate choices";
- **how** to extend the pipeline mechanically belongs in README, "Extending";
- **what** might be worth extending, and why, belongs here.

An entry that gets decided against moves to the README method notes with the
reasoning. An entry that gets done is deleted.

---

## Hazard and asset coverage

### Windstorm damage to ports

**Now.** Not modelled. `ports` is deliberately absent from `WIND_CURVES_RAW`
(`src/curves.py`): the only MIRACA wind curve for the asset is `W7.2`
(port/harbour), which is identically zero in the table, so
`AssetConfig.supports_windstorm` is False and windstorm is skipped for ports
entirely. Every ports x windstorm cell in the heatmaps is blank as a result.

**Why it might change.** That zero comes from a missing curve, not from
evidence that gust damage to port infrastructure is negligible. Eight of the
ten asset types do take wind damage, so ports sitting at exactly zero across
36 countries is an assumption the study currently states nowhere.

**Candidate source.** Jasper Verschuur's port multi-hazard work, which ships
fragility curves as spreadsheets:

- `Port_specific_risk/Input/Fragility_curves/curves_TC.xlsx` —
  <https://github.com/jasperverschuur/Port_multi_hazard_risk/blob/main/Port_specific_risk/Input/Fragility_curves/curves_TC.xlsx>
- the same directory also holds `curves_flooding.xlsx` and
  `curves_earthquake.xlsx` (see the next entry)
- dataset: <https://data.mendeley.com/datasets/kdyt24tsh5>
- paper: Verschuur et al., "Multi-hazard risk to global port infrastructure and
  resulting trade and logistics losses", *Communications Earth & Environment*

**What it would take.**

1. **Check the intensity variable first.** Our windstorm hazard is a 3-second
   gust at 10 m in m/s (WISC rasters; MIRACA sheet `W_Vuln_V10m_3sec`, 0-120
   m/s, damage fraction 0-1). The Verschuur curves are **tropical cyclone**
   curves and European windstorms are extratropical. Gust speed may well
   transfer, but if the curves are keyed to sustained wind or a different
   averaging period they have to be converted before use, and the provenance
   has to be stated in the write-up either way.
2. **Get the curve into the curve database.** `load_wind_curves()` reads curves
   by ID out of the MIRACA workbook sheet, so naming an external curve in
   `WIND_CURVES_RAW` is not enough on its own - the curve has to be added to
   that sheet (or to a merged local copy) with its own ID first.
3. **Map it to the object types ports actually carries**: `port` and `harbour`.
4. **Give the group more than one curve** if the source offers alternatives. A
   single-curve group is held fixed and contributes no curve uncertainty, which
   in an uncertainty study is the part that matters.
5. Everything downstream is already generic: `supports_windstorm` flips by
   itself, `applicable_hazards` picks windstorm up, and the blank cells fill in.

**Scope note.** This adds a new (asset, hazard) family, so it is 36 countries x
the windstorm scenarios of new cluster runs, not a figure regeneration.

### Windstorm damage to roads

Same shape as ports and the same cause - the sole roads wind curve is the
identically-zero `W7.2`, so roads are excluded from windstorm. No candidate
source identified. Worth deciding explicitly rather than inheriting: road
damage from windstorm is plausibly small, and if that is the position, it
belongs in the README method notes instead of here.

---

## Vulnerability curves

### Ports flood risk has only one curve

`FLOOD_CURVES_RAW["ports"]` maps both `port` and `harbour` to `["F9.1"]` alone.
A single-curve group is held fixed, so `flood_curve_group` cannot vary for
ports and its Sobol decomposition carries one factor fewer than the other
assets'. Earthquake already has three curves for ports (`E9.2`-`E9.4`), so the
asymmetry is specific to flood. `curves_flooding.xlsx` in the repository linked
above is the obvious place to look for alternatives.

---

## Cost values

### Reconstruction costs carry an undocumented uplift, applied unevenly

**Now.** `MAXDAM_RAW` in `src/curves.py` holds a `[min, mean, max]` cost per
object type. The table is hard-coded, not loaded from a file the way the curves
are, and it is byte-identical to `INFRASTRUCTURE_DAMAGE_VALUES` in
AssetRisk_PanEU's `src/constants.py` - which carries no citations either. The
only statement of where any of it comes from is the prose in README, "Data".
Nothing about it is machine-checkable.

**What an audit found (October 2026).** Checked against the primary source,
`Table_D3_Costs_V1.1.0.xlsx` (Nirandjan et al. 2024), whose `Cost_Database`
sheet gives a Source per value keyed to curve ID, and whose `Reference_List`
names 22 primary references. The values do trace back - but through an
uplift of roughly 14% that is recorded nowhere, and that is not applied
consistently. Table_D3 states its own basis: all values in **2020 euros**, CPI
converted, USD to EUR at historical rates.

Where the rule holds, it holds exactly - min and max span the candidate rows,
the middle value is the row that actually describes the asset:

| ours | Table_D3 row | source |
| --- | --- | --- |
| airports `aerodrome` 113 / 135 / 165 | 98.97 / 118.72 / 144.48 x1.140 | Vanneuville 2006; Kok 2005 + Briene 2002; de Bruijn 2015 |
| healthcare `hospital` 591 / 1294 / 2227 | 517.92 / 1134.72 / 1953.39 x1.140 | Batica 2018; Vanneuville 2006; de Bruijn 2015 |

Where it does not:

- **`education`** takes its minimum (267) from Djordjevic (2014)'s 267.30
  **raw**, and its maximum (1294) from Vanneuville (2006)'s 1134.72
  **uplifted**. Two conventions inside one triple.
- **`rail`** takes its minimum (491) from Vanneuville's 491.21 raw, no uplift.
- **power `tower`** takes its maximum (275,472) from Quanta Technology (2009)'s
  245,717 at a factor of **1.121**, different again.
- **power `line`/`cable`/`pole` and all of `telecom`** cannot be reached from
  Table_D3 through their curve IDs at all. The only row those IDs hit is a FEMA
  aggregate quoted in bare "euro" with no unit basis. ICF Consulting (2002) and
  Quanta Technology (2009) are both in the reference list at plausible
  magnitudes, but the link could not be established.
- **roads `motorway`** `[1106, 2895, 3931]`: the minimum is Kok/Briene's Dutch
  motorway reconstruction cost (969.58) uplifted by 1.14, and the mean is
  **2.99x that same row** - not any row in the database. Meanwhile the van
  Ginkel et al. (2021) construction-cost rows that Table_D3 keys to our
  F7.4-F7.7 curves start at 3,470 EUR/m and run to 34,702, so the whole
  modelled range sits below where that source begins.

**Why it matters more here than in a deterministic study.** `cost_level` is a
pure per-segment multiplier sampled over exactly this min-max span, so the span
is not a detail of the central estimate - it *is* the cost uncertainty the
study reports. A span assembled by two different conventions does not just
shift a number, it misstates how uncertain that number is.

**What it would take.**

1. **Settle the uplift.** Which index, which base year, and whether it applies
   to every value or none. Then apply it once, and record it next to the table.
2. **Record provenance per value.** The Table_D3 cost ID and source belong in
   `MAXDAM_RAW` as comments, so the chain stops living only in README prose and
   a later reader can check a number without redoing this audit.
3. **Resolve `motorway`.** It is the one value with a factor of three that no
   source accounts for, and it is the highest-value road class.
4. **Find or replace the power and telecom point-asset costs**, which are
   currently untraceable.
5. **Check for a newer cost table.** Table_D2 is at V3.0.0 locally while
   Table_D3 is still V1.1.0, so a later release may exist.

Until at least (1) and (3) are done, `cost_level` results for high-class roads
should not be used in a decision context - which is what README, "Data",
already says, for narrower reasons than now apply.

---

## Protection standards

### Coastal: try the other COASTPROS layers before falling back to a country mean

**Now.** `src/coastal.py` assigns a NUTS2 region its `MODELLED RETURN PERIOD`,
falls back to the country mean where that is blank, and to 0 where the country
has no record at all. Recorded policy standards are patched in for Finland's
four coastal NUTS2 regions.

**Why it might change.** COASTPROS-EU carries GEOSPATIAL, POLICY and SUMMARY
layers besides MODELLED, and where MODELLED is blank one of the others often is
not. A published value from another layer is better evidence than this study's
own national average.

**Status.** Scoped in October 2026 as option (b) and deliberately deferred: the
rerun that was actually carried out was option (a), the regions that were
clearly wrong, plus Finland. Deferred on cost, not rejected - and the cost is a
coastal rerun of the affected regions, not of the whole hazard.

### River and coastal treat "no recorded standard" differently

River flood takes FLOPROS at face value: no record means unprotected. Coastal
substitutes the country mean. The asymmetry is deliberate (October 2026) - in
an uncertainty analysis, giving river flood a national-average standard
mid-study would introduce a new assumption rather than measure an existing one.

It is still an inconsistency between two hazards in one study, and a future
version should settle it as a deliberate pair, in whichever direction, rather
than leave it as the residue of two separate decisions.

---

## Hazard representation

### Flood extent uncertainty is not sampled

`depth_offset` perturbs only cells the hazard map already marks as inundated: a
positive offset deepens water but cannot push the flood outside the map's
footprint (a negative one can shrink it). Extent is therefore treated as
certain, which it is not. Sampling it needs more than one hazard map per return
period, which is a data question before it is a code question.

### No damage beyond the largest mapped return period

EAD is a trapezoid over p = 1/RP between the smallest and largest mapped RP,
with no tail beyond and no damage for more frequent events - following
AssetRisk_PanEU. README "Extending" already lists an EAD tail assumption as a
candidate factor; it would make the truncation visible as uncertainty instead
of silent.

---

## Open questions

### 85 combinations are exposed but never damaged

85 (country, asset, scenario) combinations have `exposed_qty_RP100_* > 0`,
often very large, yet `max(total_EAD) == 0` across every draw, so every Sobol
index is exactly zero. Split: windstorm 38, coastal 32, river flood 15; by
asset, airports 33, gas 28, oil 16. The worst cases are FIN, GRC, CZE, ROU and
HUN airports, each carrying roughly 2e8 m2 inside the windstorm footprint with
deterministically zero damage.

The figures handle this already - `plot_drivers.exposure_flags` and the
`no_variance` guard label them "Exposed, never damaged" (hatched) and
distinguish them from the 188 combinations with no exposure. So they are
visible. They are not explained.

**Check this first.** Airports map `aerodrome`, `apron` and `runway` to the
zero placeholder `W7.2`, and only `terminal` to real wind curves - so an
airport whose exposure is mostly surface area reports zero by construction,
which would account for the windstorm cases without any bug. Confirm it before
looking further: take FIN/airports/windstorm, pull the gust profile from
`data/intermediate/FIN_airports_windstorm_profiles.parquet`, and compare its
range against the damage onset of the curves `applicable_hazards` selects.

This matters because it is the difference between "correctly nothing to model"
and "the model cannot see damage it should see", and only the second needs
fixing. A whole asset class silently reporting zero is the shape a bug takes.

---

## Sampling

### Sobol sample size and convergence

The study is standardised on N = 8192. A handful of combinations were rerun at
N = 32768; both sets of archives are on disk and `src/restore_sobol_n.py` moves
the indices CSVs between them in either direction without re-running anything,
so neither N is lost.

`src/adaptive_sobol.py` accepts a combination once every relevant factor's
ST_conf/ST is below `DEFAULT_THRESHOLD = 0.2`. Combinations that never got
under it are the candidates for a higher N in a future version - they are the
only ones where the extra compute buys precision that the study currently
lacks.
