# Coastal range test x0.75-1.33, with SLR (prepared 2026-11-10)

`coastal_baseline_ds_slr_sc0p75to1p33`: the recorded COASTPROS standard times a multiplier, log-uniform on
[0.75, 1.33] (upper bound exactly 4/3; median x1, an increase as likely as a decrease of the same factor). It is the "recorded value is off,
but only a bit" companion to `coastal_baseline_ds_slr_sc0p5to2p0`. One multiplier per model run, applied to every
feature; a recorded standard of 0 stays 0.

`combos_coastal_all.txt` lists the 257 coastal combinations, longest first. Expected runtime, from the x0.5-2.0
batch: 50.8 job-h; about 8 h with all 8 slots free, set by DEU roads.

    R=reruns/20261110_coastal_sc0p75to1p33
    SCEN="coastal_baseline_ds_slr_sc0p75to1p33" ADDSCEN_LIST=$R/combos_coastal_all.txt \
      SCEN_TAG=coastal_slr_sc0p75to1p33 SCEN_TIME=120:00:00 SCEN_TIME_LIGHT=120:00:00 \
      EXCLUDE=node002,node009 ./submit_miraca_uncertainty_study.sh addscen
