# DamageScanner polygon fix: polygon asset types, river/earthquake/windstorm (prepared 2026-11-20)

DamageScanner 1.0.1 overstates polygon exposure on geographic rasters by about 1/cos(latitude), 1.22 for
Cyprus to 2.33 for Iceland (issue #82), and counts cells on tile edges twice (#83, lines +0.1-0.5%). The
fix is develop commit `f6599e00` (not on PyPI; it still reports version 1.0.1).

Each asset type stays on one version (`src/ds_version.py`):

| Asset types | DamageScanner | This batch |
|---|---|---|
| roads, rail, telecom (lines and points only) | 1.0.1, for good | not touched |
| airports, education, gas, healthcare, oil, ports, power | the fix, pinned to `f6599e00` | redone for river, earthquake, windstorm |

Coastal is a projected 25 m grid, which 1.0.1 already handled correctly; its profiles and results stay.

## Before submitting

1. Commit and push this code, then `git pull` on the cluster.
2. Build the second venv (copies every package version of the 1.0.1 venv, then adds the fix):
   `./setup_dsfix_venv.sh` (or `DRY=1 ./setup_dsfix_venv.sh` first). It stops if the target exists and never
   deletes anything.
3. `DRY=1 bash reruns/20261120_polygon_fix/submit.sh` checks the venv and the guard, and lists the jobs.

Then `bash reruns/20261120_polygon_fix/submit.sh` submits everything, chained.

## What runs

| Step | Jobs | Estimate |
|---|---|---|
| Stage 1, one job per combination (`stage1_combos.txt`) | 243 | at most 1.7 h each, 53 h in all |
| Gate: `src.stage1_rerun gate --require-polygon-fix` | 1 | minutes |
| Stage 2 at N = 8192, wave 1 (jobs over 3 h), then wave 2, default scenarios | 2,088 | about 570 job-h |
| Extra after both waves: the wider PGA range (`earthquake`, `extra_earthquake.txt`) | 243 | about 51 job-h |
| N = 32768 for the 13 combinations rerun at that N before | 13 | about 137 job-h, GBR power alone about 70 h |

The default earthquake scenario is the narrower PGA range, `earthquake_pga0p9to1p1` (x0.9-1.1), for
comparability. The wider `earthquake` is rerun too, as an extra queued after the defaults.

Each Stage 1 job: backs up the files it rewrites to `data/intermediate_ds101_backup_20261120/` (never
overwrites a backup, so a resubmitted job keeps the original), runs `preprocess --hazards river earthquake
windstorm`, checks the rebuilt segments against the backup, validates, and writes the validation marker.

- **Segments check:** only `prot_rp` may change, plus the new provenance columns `prot_source` and
  `prot_fill_km`. Rebuilding the segments also moves these types to the 5 km FLOPROS block standards (config
  `river_protection`), so `update_river_protection --apply` is not needed. Every other column, including
  the coastal standards with the Malta and Cyprus minimum, must be identical, or the coastal results would no
  longer match. Rebuilt in memory for LUX healthcare and power, MLT power, CYP airports and EST education,
  only `prot_rp` changed.
- **Gate:** passes only if every combination has its backup, a passed segments check, all three hazards
  extracted with the pinned fix (recorded per hazard in `_meta.json`), and a validation newer than the
  extraction. If it fails, Stage 2 never starts: its jobs sit with `DependencyNeverSatisfied`.
  `status/gate.csv` lists what is missing.

Stage 2 jobs are forced (`FORCE_SCEN=1`): the old archives keep their timestamped names, the newer ones win.
Each N = 32768 job waits for the N = 8192 job of the same combination and scenario. The 32768 runs must be
redone too, or "largest N wins" in the figures and convergence tests would keep the 1.0.1 result.

The estimates are each job's own last runtime, scaled to the S1/ST-only design now used. On the November
river batch the same method came out about a quarter high. With about 8 job slots free, Stage 2 takes
about 3 days, and the GBR power chain (about 9 h, then about 70 h) sets the end at about 4 days.

## Files

- `stage1_combos.txt`, `wave1_*.txt`, `wave2_*.txt`, `extra_*.txt`, `n32768_*.txt`: the job lists, longest
  first. Made by `python -m src.plan_polygon_fix --out reruns/20261120_polygon_fix --extras earthquake`.
- `plan.csv`: every Stage 2 job with its runtime estimate and the run it was timed from.
- `snapshot.csv`: the results the forced reruns overwrite; `submit.sh` copies their derived files to
  `results_snapshot_ds101_20261120/` first.
- `archives_before.txt`: the 3,283 LHS and Sobol archives these combinations had before.
- Written on submission: `jobs_*.log` (job ids), `gate_job.txt`, `status/` (segments checks, `gate.csv`),
  `submitted.txt` (time, venv, gate, commit). A second submission stops at `submitted.txt`.

## Not rerun

- Roads, rail and telecom, and every coastal scenario.
- `flood_absprot_ds_rp10to100` (69) and the superseded `flood_baseline_ds_sc0p5to1p5` (69): not in the
  paper, so they stay on 1.0.1 for polygon types. Rerun them by planning again with
  `python -m src.plan_polygon_fix --out reruns/<dir> --extras flood_absprot_ds_rp10to100 ...`.

## If the gate fails

Read `status/gate.csv` and the `out_s1_dsfix_<C>_<asset>` logs. Fix, then resubmit Stage 1 for the failed
combinations with the same `RESTAGE_BACKUP` (their backups are kept) and a new list, and run the gate by
hand: `python -m src.stage1_rerun gate --list reruns/20261120_polygon_fix/stage1_combos.txt --hazards river
earthquake windstorm --backup data/intermediate_ds101_backup_20261120 --status reruns/20261120_polygon_fix/status
--require-polygon-fix`. Cancel the pending Stage 2 jobs (`scancel -n scen_...` or by the ids in `jobs_*.log`)
and submit the waves again with `DEPEND` dropped, as in `submit.sh` step 4.

## Afterwards

Rebuild the summary workbook, EAD ranges, the paper quantile cache (`src.paper_ead_samples`), the convergence
tests (`src.convergence_tests --resume`), the cascades for power and the mid-size types, and the paper figures.
Polygon EAD should drop by 1.2-2.3x; relative widths and Sobol indices should barely move for the polygon-only
types (education, healthcare, ports) and can shift for the mixed ones.
