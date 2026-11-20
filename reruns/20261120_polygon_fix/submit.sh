#!/bin/bash
# DamageScanner polygon fix: redo the polygon asset types for river, earthquake and
# windstorm. Generated 2026-11-20 by src.plan_polygon_fix; see README.md.
#   DRY=1 bash reruns/20261120_polygon_fix/submit.sh    # checks + preview, submits nothing
#   bash reruns/20261120_polygon_fix/submit.sh          # submit everything, chained
#
# Chain: Stage 1 (243 jobs, each: backup -> preprocess --hazards river earthquake
# windstorm -> segments check -> validation) -> gate -> Stage 2 waves, then the
# extras (afterok gate) -> the 13 N = 32768 reruns, each afterok its own N = 8192 job.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
TAG=dsfix
DATE=20261120
USER_DIR=/scistor/ivm/${USER:-$(id -un)}
# Every job of this batch runs in the polygon-fix venv (setup_dsfix_venv.sh).
export VENV=${VENV:-${USER_DIR}/.venvs/miraca_uq_dsfix}
export PYTHON=${VENV}/bin/python
export MIRACA_CONFIG=${MIRACA_CONFIG:-$REPO/config.cluster.yml}
export EXCLUDE=${EXCLUDE:-node002,node009}
export SCEN_TIME=${SCEN_TIME:-120:00:00}
export SCEN_TIME_LIGHT=${SCEN_TIME_LIGHT:-120:00:00}
export S1_TIME=${S1_TIME:-120:00:00}
BACKUP=data/intermediate_ds101_backup_${DATE}
STATUS=$HERE/status
HAZ="river earthquake windstorm"
SUB=./submit_miraca_uncertainty_study.sh
count() { grep -vc '^#' "$1"; }

# 1. the venv must hold the pinned fix, and the plan must be there
[[ -x "$PYTHON" ]] || { echo "No venv at $VENV - run ./setup_dsfix_venv.sh first." >&2; exit 1; }
"$PYTHON" -m src.ds_version --expect-fix --asset power > /dev/null \
    || { echo "$VENV does not hold the pinned polygon fix; nothing submitted." >&2; exit 1; }
"$PYTHON" -m src.ds_version --asset roads > /dev/null 2>&1 \
    && { echo "Guard check failed: roads would be allowed on the fix; nothing submitted." >&2; exit 1; }
echo "DamageScanner polygon fix in $VENV; the guard refuses line types there."
[[ -f "$HERE/stage1_combos.txt" ]] || { echo "No plan; run src.plan_polygon_fix first." >&2; exit 1; }

if [[ -n "${DRY:-}" ]]; then
    echo "Stage 1: $(count "$HERE/stage1_combos.txt") combinations ($HAZ), backup to $BACKUP"
    for f in "$HERE"/wave1_*.txt "$HERE"/wave2_*.txt "$HERE"/extra_*.txt "$HERE"/n32768_*.txt; do
        [[ -e "$f" ]] || continue
        echo "  $(basename "$f" .txt): $(count "$f") jobs"
    done
    "$PYTHON" -m src.snapshot_results --list "$HERE/snapshot.csv" --tag "ds101_${DATE}" --dry-run
    exit 0
fi
[[ -f "$HERE/submitted.txt" ]] && { echo "submitted.txt exists: this batch was submitted before." >&2; exit 1; }
mkdir -p "$STATUS"

# 2. the derived results the forced reruns overwrite (archives keep their own names)
"$PYTHON" -m src.snapshot_results --list "$HERE/snapshot.csv" --tag "ds101_${DATE}"

# 3. Stage 1 and its gate
RESTAGE_LIST="$HERE/stage1_combos.txt" RESTAGE_HAZARDS="$HAZ" RESTAGE_BACKUP="$BACKUP" \
RESTAGE_STATUS="$STATUS" RESTAGE_GATE_FILE="$HERE/gate_job.txt" RESTAGE_GATE_ARGS="--require-polygon-fix" \
SCEN_TAG="$TAG" "$SUB" restage1 | tee "$HERE/jobs_stage1.log"
GATE=$(cat "$HERE/gate_job.txt")

# 4. Stage 2 behind the gate: wave 1 (jobs over 3 h) of every default scenario first,
#    then wave 2, then the extras (the wider PGA range), so they never hold up the defaults
for prefix in wave1 wave2 extra; do
    for f in "$HERE"/${prefix}_*.txt; do
        [[ -e "$f" ]] || continue
        scen=$(basename "$f" .txt); scen=${scen#${prefix}_}
        DEPEND="afterok:${GATE}" FORCE_SCEN=1 SCEN="$scen" ADDSCEN_LIST="$f" SCEN_TAG="${TAG}_${scen}" \
            "$SUB" addscen | tee -a "$HERE/jobs_${scen}.log"
    done
done

# 5. N = 32768, each after the N = 8192 job of the same combination and scenario
mkdir -p "$HERE/n32768_single"
for f in "$HERE"/n32768_*.txt; do
    scen=$(basename "$f" .txt); scen=${scen#n32768_}
    grep -v '^#' "$f" | while read -r C A; do
        A=${A%$'\r'}
        jid=$(grep -oP "^submitted ${C}_${A}: run=\K[0-9]+" "$HERE/jobs_${scen}.log" | tail -1)
        [[ -n "$jid" ]] || { echo "!! no N = 8192 job for $C $A $scen; submit its N = 32768 run by hand" >&2; continue; }
        one="$HERE/n32768_single/${scen}_${C}_${A}.txt"
        echo "$C $A" > "$one"
        DEPEND="afterok:${jid}" FORCE_SOBOL=1 SOBOL_N=32768 SCEN="$scen" ADDSCEN_LIST="$one" \
            SCEN_TAG="${TAG}32k_${scen}" "$SUB" addscen | tee -a "$HERE/jobs_n32768.log"
    done
done

{
    echo "submitted $(date -u +%FT%TZ) by ${USER:-$(id -un)}"
    echo "venv $VENV"
    echo "gate $GATE"
    git -C "$REPO" rev-parse HEAD 2>/dev/null | sed 's/^/commit /'
} > "$HERE/submitted.txt"
echo
echo "All submitted. Gate verdict once Stage 1 drains:  cat \$LOG_DIR/out_s1_gate_${TAG}"
echo "If the gate fails, Stage 2 never starts; fix, then resubmit Stage 2 by hand (README)."
