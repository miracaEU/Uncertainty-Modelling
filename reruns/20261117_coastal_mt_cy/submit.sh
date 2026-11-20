#!/bin/bash
# Step 2: rerun the coastal scenarios that read the recorded standard, for Malta and Cyprus.
#   DRY=1 bash reruns/20261117_coastal_mt_cy/submit.sh    # checks + preview
#   bash reruns/20261117_coastal_mt_cy/submit.sh          # submit
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-/scistor/ivm/${USER:-$(id -un)}/.venvs/miraca_uq/bin/python}
export MIRACA_CONFIG=${MIRACA_CONFIG:-$REPO/config.cluster.yml}
export EXCLUDE=${EXCLUDE:-node002,node009}
export SCEN_TIME=${SCEN_TIME:-120:00:00}
export SCEN_TIME_LIGHT=${SCEN_TIME_LIGHT:-120:00:00}

bad=0
for ca in CYP_airports CYP_education CYP_gas CYP_healthcare CYP_oil CYP_ports CYP_power CYP_rail CYP_roads CYP_telecom MLT_airports MLT_education MLT_healthcare MLT_oil MLT_ports MLT_power MLT_roads MLT_telecom; do
    grep -q '"coastal_minimum_standard"' "$REPO/data/intermediate/${ca}_meta.json" || { echo "!! $ca not updated" >&2; bad=1; }
done
[[ $bad -eq 0 ]] || { echo "Run update_coastal.sh first; nothing submitted." >&2; exit 1; }
echo "All 18 Malta and Cyprus segments carry the 25-year minimum."

if [[ -z "${DRY:-}" ]]; then
    [[ -f "$HERE/archives_before.txt" ]] && { echo "archives_before.txt exists: submitted before." >&2; exit 1; }
    {
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_airports_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_education_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_gas_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_healthcare_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_oil_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_ports_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_power_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_rail_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_roads_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_telecom_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_airports_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_education_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_healthcare_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_oil_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_ports_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_power_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_roads_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_telecom_coastal_noprot_ds_slr_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_airports_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_education_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_gas_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_healthcare_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_oil_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_ports_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_power_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_rail_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_roads_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_telecom_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_airports_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_education_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_healthcare_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_oil_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_ports_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_power_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_roads_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_telecom_coastal_baseline_ds_slr_sc0p5to2p0_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_airports_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_education_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_gas_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_healthcare_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_oil_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_ports_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_power_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_rail_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_roads_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CYP" -maxdepth 1 -name "experiments_CYP_telecom_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_airports_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_education_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_healthcare_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_oil_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_ports_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_power_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_roads_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MLT" -maxdepth 1 -name "experiments_MLT_telecom_coastal_baseline_ds_slr_sc0p75to1p33_*.tar.gz" -printf '%s %p\n'
    } > "$HERE/archives_before.txt"
    echo "Recorded $(wc -l < "$HERE/archives_before.txt") existing archive(s)"
fi
"$PYTHON" -m src.snapshot_results --list "$HERE/snapshot.csv" --tag coastal_min0_20261117 ${DRY:+--dry-run}

for scen in coastal_noprot_ds_slr coastal_baseline_ds_slr_sc0p5to2p0 coastal_baseline_ds_slr_sc0p75to1p33; do
    if [[ -n "${DRY:-}" ]]; then echo "$scen (forced): 18 combinations"; continue; fi
    FORCE_SCEN=1 SCEN="$scen" ADDSCEN_LIST="$HERE/combos_mt_cy.txt" SCEN_TAG="min25_$scen" \
        ./submit_miraca_uncertainty_study.sh addscen
done
