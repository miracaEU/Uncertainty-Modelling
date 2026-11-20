#!/bin/bash
# River flood with the 5 km FLOPROS blocks + 5 km fill (src/river_protection.py),
# for roads, rail and telecom. Generated 2026-11-11; see README.md.
#   DRY=1 bash reruns/20261111_river_protection/submit.sh    # checks + preview
#   bash reruns/20261111_river_protection/submit.sh          # submit
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-/scistor/ivm/${USER:-$(id -un)}/.venvs/miraca_uq/bin/python}
export MIRACA_CONFIG=${MIRACA_CONFIG:-$REPO/config.cluster.yml}
export EXCLUDE=${EXCLUDE:-node002,node009}
export SCEN_TIME=${SCEN_TIME:-120:00:00}
export SCEN_TIME_LIGHT=${SCEN_TIME_LIGHT:-120:00:00}

# 1. every listed combination must already carry the new standards
bad=0
for ca in ALB_rail AUT_rail BEL_rail BGR_rail CHE_rail CYP_rail CZE_rail DEU_rail DNK_rail ESP_rail EST_rail FIN_rail FRA_rail GBR_rail GRC_rail HRV_rail HUN_rail IRL_rail ISL_rail ITA_rail LIE_rail LTU_rail LUX_rail LVA_rail MKD_rail NLD_rail NOR_rail POL_rail PRT_rail ROU_rail SRB_rail SVK_rail SVN_rail SWE_rail ALB_roads AND_roads AUT_roads BEL_roads BGR_roads CHE_roads CYP_roads CZE_roads DEU_roads DNK_roads ESP_roads EST_roads FIN_roads FRA_roads GBR_roads GRC_roads HRV_roads HUN_roads IRL_roads ISL_roads ITA_roads LIE_roads LTU_roads LUX_roads LVA_roads MKD_roads MLT_roads NLD_roads NOR_roads POL_roads PRT_roads ROU_roads SRB_roads SVK_roads SVN_roads SWE_roads ALB_telecom AND_telecom AUT_telecom BEL_telecom BGR_telecom CHE_telecom CYP_telecom CZE_telecom DEU_telecom DNK_telecom ESP_telecom EST_telecom FIN_telecom FRA_telecom GBR_telecom GRC_telecom HRV_telecom HUN_telecom IRL_telecom ISL_telecom ITA_telecom LIE_telecom LTU_telecom LUX_telecom LVA_telecom MKD_telecom MLT_telecom NLD_telecom NOR_telecom POL_telecom PRT_telecom ROU_telecom SRB_telecom SVK_telecom SVN_telecom SWE_telecom; do
    if ! grep -q '"method": "block5km_fill"' "$REPO/data/intermediate/${ca}_meta.json"; then
        echo "!! $ca: segments still use the old protection method" >&2; bad=1
    fi
done
[[ $bad -eq 0 ]] || { echo "Run src.update_river_protection --apply first; nothing submitted." >&2; exit 1; }
echo "All 106 segments carry the 5 km block + fill standards."

# 2. back up what the forced flood_noprot_ds reruns rewrite in place
if [[ -z "${DRY:-}" ]]; then
    [[ -f "$HERE/archives_before.txt" ]] && { echo "archives_before.txt exists: submitted before." >&2; exit 1; }
    {
    find "$REPO/results/ITA" -maxdepth 1 -name "experiments_ITA_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/DEU" -maxdepth 1 -name "experiments_DEU_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ESP" -maxdepth 1 -name "experiments_ESP_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/FRA" -maxdepth 1 -name "experiments_FRA_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/GBR" -maxdepth 1 -name "experiments_GBR_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/POL" -maxdepth 1 -name "experiments_POL_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/AUT" -maxdepth 1 -name "experiments_AUT_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/NLD" -maxdepth 1 -name "experiments_NLD_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SWE" -maxdepth 1 -name "experiments_SWE_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/GRC" -maxdepth 1 -name "experiments_GRC_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ROU" -maxdepth 1 -name "experiments_ROU_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/PRT" -maxdepth 1 -name "experiments_PRT_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CZE" -maxdepth 1 -name "experiments_CZE_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/HUN" -maxdepth 1 -name "experiments_HUN_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/FIN" -maxdepth 1 -name "experiments_FIN_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CHE" -maxdepth 1 -name "experiments_CHE_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/BEL" -maxdepth 1 -name "experiments_BEL_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SRB" -maxdepth 1 -name "experiments_SRB_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/IRL" -maxdepth 1 -name "experiments_IRL_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SVK" -maxdepth 1 -name "experiments_SVK_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/NOR" -maxdepth 1 -name "experiments_NOR_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/DEU" -maxdepth 1 -name "experiments_DEU_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/BGR" -maxdepth 1 -name "experiments_BGR_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/HRV" -maxdepth 1 -name "experiments_HRV_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/DNK" -maxdepth 1 -name "experiments_DNK_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/FRA" -maxdepth 1 -name "experiments_FRA_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SVN" -maxdepth 1 -name "experiments_SVN_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LTU" -maxdepth 1 -name "experiments_LTU_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/GBR" -maxdepth 1 -name "experiments_GBR_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ITA" -maxdepth 1 -name "experiments_ITA_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LVA" -maxdepth 1 -name "experiments_LVA_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/POL" -maxdepth 1 -name "experiments_POL_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/EST" -maxdepth 1 -name "experiments_EST_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ALB" -maxdepth 1 -name "experiments_ALB_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ESP" -maxdepth 1 -name "experiments_ESP_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CZE" -maxdepth 1 -name "experiments_CZE_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/MKD" -maxdepth 1 -name "experiments_MKD_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/AUT" -maxdepth 1 -name "experiments_AUT_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LUX" -maxdepth 1 -name "experiments_LUX_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CHE" -maxdepth 1 -name "experiments_CHE_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ISL" -maxdepth 1 -name "experiments_ISL_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/FRA" -maxdepth 1 -name "experiments_FRA_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ROU" -maxdepth 1 -name "experiments_ROU_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SWE" -maxdepth 1 -name "experiments_SWE_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/NLD" -maxdepth 1 -name "experiments_NLD_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/BEL" -maxdepth 1 -name "experiments_BEL_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/HUN" -maxdepth 1 -name "experiments_HUN_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SVK" -maxdepth 1 -name "experiments_SVK_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/FIN" -maxdepth 1 -name "experiments_FIN_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/DEU" -maxdepth 1 -name "experiments_DEU_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/NOR" -maxdepth 1 -name "experiments_NOR_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/PRT" -maxdepth 1 -name "experiments_PRT_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/BGR" -maxdepth 1 -name "experiments_BGR_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/PRT" -maxdepth 1 -name "experiments_PRT_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ITA" -maxdepth 1 -name "experiments_ITA_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/GBR" -maxdepth 1 -name "experiments_GBR_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LVA" -maxdepth 1 -name "experiments_LVA_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SRB" -maxdepth 1 -name "experiments_SRB_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/POL" -maxdepth 1 -name "experiments_POL_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/HRV" -maxdepth 1 -name "experiments_HRV_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/NLD" -maxdepth 1 -name "experiments_NLD_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/GRC" -maxdepth 1 -name "experiments_GRC_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/DNK" -maxdepth 1 -name "experiments_DNK_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/IRL" -maxdepth 1 -name "experiments_IRL_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ESP" -maxdepth 1 -name "experiments_ESP_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SVN" -maxdepth 1 -name "experiments_SVN_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/NOR" -maxdepth 1 -name "experiments_NOR_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/FIN" -maxdepth 1 -name "experiments_FIN_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LIE" -maxdepth 1 -name "experiments_LIE_roads_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/AUT" -maxdepth 1 -name "experiments_AUT_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LTU" -maxdepth 1 -name "experiments_LTU_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LUX" -maxdepth 1 -name "experiments_LUX_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/HRV" -maxdepth 1 -name "experiments_HRV_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/EST" -maxdepth 1 -name "experiments_EST_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/ROU" -maxdepth 1 -name "experiments_ROU_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LIE" -maxdepth 1 -name "experiments_LIE_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CHE" -maxdepth 1 -name "experiments_CHE_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SVK" -maxdepth 1 -name "experiments_SVK_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LVA" -maxdepth 1 -name "experiments_LVA_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/IRL" -maxdepth 1 -name "experiments_IRL_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/HUN" -maxdepth 1 -name "experiments_HUN_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/BEL" -maxdepth 1 -name "experiments_BEL_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/CZE" -maxdepth 1 -name "experiments_CZE_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/GRC" -maxdepth 1 -name "experiments_GRC_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SVN" -maxdepth 1 -name "experiments_SVN_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/EST" -maxdepth 1 -name "experiments_EST_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/DNK" -maxdepth 1 -name "experiments_DNK_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/SRB" -maxdepth 1 -name "experiments_SRB_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LIE" -maxdepth 1 -name "experiments_LIE_rail_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    find "$REPO/results/LTU" -maxdepth 1 -name "experiments_LTU_telecom_flood_noprot_ds_*.tar.gz" -printf '%s %p\n'
    } > "$HERE/archives_before.txt"
    echo "Recorded $(wc -l < "$HERE/archives_before.txt") existing flood_noprot_ds archive(s)"
fi
"$PYTHON" -m src.snapshot_results --list "$HERE/snapshot_noprot.csv" --tag pixel_noprot_20261111 ${DRY:+--dry-run}

# 3. submit, heavy wave first; logs get their own tag so no old log is overwritten
sub() {   # scenario list force
    local scen="$1" list="$HERE/$2" force="${3:-}"
    if [[ -n "${DRY:-}" ]]; then
        echo "$scen${force:+ (forced)}: $(grep -vc '^#' "$list") combinations from $2"; return
    fi
    FORCE_SCEN="$force" SCEN="$scen" ADDSCEN_LIST="$list" SCEN_TAG="blk5_$scen" \
        ./submit_miraca_uncertainty_study.sh addscen
}
sub flood_baseline_ds_sc0p5to2p0 wave1_flood_baseline_ds_sc0p5to2p0.txt    # wave 1: 12 combinations
sub flood_baseline_ds_sc0p75to1p33 wave1_flood_baseline_ds_sc0p75to1p33.txt    # wave 1: 12 combinations
sub flood_noprot_ds wave1_flood_noprot_ds.txt 1   # wave 1: 9 combinations
sub flood_noprot_ds wave2_flood_noprot_ds.txt 1   # wave 2: 81 combinations
sub flood_baseline_ds_sc0p5to2p0 wave2_flood_baseline_ds_sc0p5to2p0.txt    # wave 2: 94 combinations
sub flood_baseline_ds_sc0p75to1p33 wave2_flood_baseline_ds_sc0p75to1p33.txt    # wave 2: 94 combinations
