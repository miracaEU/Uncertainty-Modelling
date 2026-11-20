# Coastal reruns, prepared 2026-11-06

## A. Stale `coastal_absprot_ds` archives (`combos_absprot_pre_rpfix.txt`)

Five combinations still carry results from before the coastal RP-grid fix of 19 Aug 2026 (d1af891,
RPs [1, 100, 1000] instead of the phantom [10, 25, 50, 100, 250, 500, 1000]). They were never recomputed;
their indices CSVs were re-derived later from the old archives. Re-evaluating 4-6 rows of each archive with
the current code (scratch script `recheck_archive.py`) gives:

| combination | max relative EAD difference |
|---|---|
| DNK power | 8.1x |
| ESP roads | 8.0x |
| GBR power | 7.1x |
| FRA power | 2.6x |
| POL roads | 2.2x |

The other five pre-September archives (ALB/BGR/CYP/ESP oil, BGR power) match exactly. Their
`coastal_noprot_ds` and SLR runs date from after the fix. `archives_before.txt` lists the archives on disk
before the rerun. Snapshot the derived outputs first (`snapshot_absprot_pre_rpfix.csv`).

Only coastal runs WITH sea-level rise are reported, and these five are all present-day
(`coastal_absprot_ds`). Their SLR twins date from October and are current. Rerun them only if a present-day
coastal result is shown somewhere (e.g. the eu_totals reference against the deterministic run).

## B. Coastal range tests with SLR, all 26 coastal countries (`combos_coastal_all.txt`)

| scenario | protection |
|---|---|
| `coastal_baseline_ds_slr_sc0p5to2p0` | recorded x 0.5-2.0, log-uniform (median x1) |
| `coastal_absprot_ds_slr_rp25to200` | RP 25-200, uniform: the RP 5-200 default with a 25-year floor |

RP 10-100 was dropped (2026-11-06) as unrealistic; RP 25-200 replaces it, so that against
`coastal_absprot_ds_slr` (RP 5-200) it shows what a 25-year minimum standard removes.
The present-day twins (`coastal_baseline_ds_sc0p5to2p0`, `coastal_absprot_ds_rp25to200`) are registered
but not submitted, since only SLR runs are reported; the SLR factor's 2010_None state is the present day.
`coastal_baseline_ds_sc0p5to1p5` and the 7-country `coastal_absprot_ds_rp10to100` stay on disk, superseded.
ISL and ROU are included for uniform coverage although nothing is exposed there.
