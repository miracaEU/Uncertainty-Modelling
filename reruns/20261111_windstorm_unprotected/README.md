# Windstorm without a design standard (prepared 2026-11-11)

`windstorm_unprotected`: as `windstorm`, but with protection 0 instead of the fixed RP50 design standard, so every
wind map counts from the shortest return period (5 years) on. Same factors as `windstorm`: wind curve choice,
cost level, gust scale, aggregation.

This round: rail and telecom (lines and points), all 70 combinations, longest first in `combos_rail_telecom.txt`.
About 10 job-h from the logged `windstorm` runtimes, which used second-order sampling, so likely less.
Airports, education, gas, healthcare, oil and power follow after the DamageScanner polygon fix.

    SCEN="windstorm_unprotected" ADDSCEN_LIST=reruns/20261111_windstorm_unprotected/combos_rail_telecom.txt \
      SCEN_TAG=windstorm_unprotected SCEN_TIME=120:00:00 SCEN_TIME_LIGHT=120:00:00 EXCLUDE=node002,node009 \
      ./submit_miraca_uncertainty_study.sh addscen
