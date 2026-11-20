# River flood with the new protection standards: line and point assets (prepared 2026-11-11)

River protection is now the 5 km FLOPROS block mean under each asset's centroid, exactly as MIRACA's
AssetRisk_PanEU reads it, and where that block has no data, the nearest protected block whose edge is within
5 km (`src/river_protection.py`). It replaces the 500 m FLOPROS pixel. The 52 exposed road assets still at 0
after the fill (Norway, Iceland, Finland) stay at 0.

This round covers **roads, rail and telecom**, the asset types with only line and point features. Polygon
types wait for the DamageScanner polygon fix (commit f6599e00, not yet installed). Lines keep DamageScanner
1.0.1 now and later alike, so every line result comes from the same version.

## Files

- `river_protection_changes.csv`: per combination, river-exposed assets (any profile row with depth > 0) and
  how many change standard. Old values matched a fresh read of the 500 m pixel for every asset.
- `river_protection_applied.csv`: the segments update for roads, rail and telecom. The previous segments and
  meta files are in `data/intermediate_pixel_backup_20261111/`.
- `wave1_*.txt`, `wave2_*.txt`: the job lists, longest first. Wave 1 holds the jobs over 3 h.
- `snapshot_noprot.csv`: the `flood_noprot_ds` combinations whose derived files are copied to
  `results_snapshot_pixel_noprot_20261111/` before the forced reruns rewrite them.
- `update_segments.sh`: the cluster job that applies the new standards (step 1).
- `submit.sh`: checks the segments, records the existing archives, takes the snapshot and submits (step 2).

## What runs now

| Scenario | Protection | Combinations | Why |
|---|---|---|---|
| `flood_noprot_ds` | recorded standard | 90 (roads 33, rail 30, telecom 27) | standards of exposed assets changed; forced rerun |
| `flood_baseline_ds_sc0p5to2p0` | recorded × 0.5–2, log-uniform | 106 (all) | new scenario |
| `flood_baseline_ds_sc0p75to1p33` | recorded × 0.75–1.33, log-uniform | 106 (all) | new scenario |

302 jobs, about 479 job-h from the river RP 25-200 batch's runtimes, about 60 h with 8 slots free.

Not rerun for `flood_noprot_ds` because no exposed asset's standard changes: ALB rail, ALB telecom, AND roads,
AND telecom, BGR telecom, CYP rail, CYP roads, CYP telecom, ISL rail, ISL telecom, LUX telecom, MKD rail,
MKD telecom, MLT roads, MLT telecom, SWE telecom.

## Already valid, nothing to run

`flood_absprot_ds` (RP 5-200), `flood_absprot_ds_rp25to200` and `flood_absprot_ds_rp10to100` replace every
asset's standard with the sampled return period and never read the recorded one.

## Left for later

| What | Combinations | Note |
|---|---|---|
| DamageScanner polygon fix, Stage 1 river/earthquake/windstorm | airports, education, gas, healthcare, oil, ports, power | install f6599e00 first; coastal profiles unaffected |
| Segments update to the new protection | the same 243 combinations | `src.update_river_protection --apply`; 189 have exposed assets that change |
| `flood_noprot_ds` and both scaled scenarios | the same | after the two steps above; power is about 67 job-h per scenario |
| `flood_baseline_ds_sc0p5to1p5` (uniform, 99 combinations) | superseded | stays on the pixel standards; not rerun |
| Workbook, EAD ranges, convergence tests, paper figures | all | once the reruns land |
