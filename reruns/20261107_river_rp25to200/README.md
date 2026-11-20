# River flood, sampled protection RP 25-200 (prepared 2026-11-07)

`flood_absprot_ds_rp25to200`: as `flood_absprot_ds` (RP 5-200), with the protection floor raised to 25 years.
It mirrors the coastal `coastal_absprot_ds_slr_rp25to200`, so both flood hazards carry the same
minimum-standard test. River has no SLR twin; its climate factor `warming` is part of the scenario.

`combos_river_all.txt` lists all 349 river combinations, longest first. Expected runtime is from the original
`flood_absprot_ds` N=8192 runs, scaled by 0.56 for sampling without second-order rows (time-weighted
(k+2)/(2k+2) over k = 5-8; the earthquake range test showed 0.56 in practice): about 300 job-h, up to about
385 job-h if the bracket slows runs as RP 10-100 did (x1.28). Longest job is DEU roads, about 40 h. With 8
slots and the longest first, the batch takes about 40-51 h if the queue is otherwise empty.
