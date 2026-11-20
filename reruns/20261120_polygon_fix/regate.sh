#!/bin/bash
# Redo Stage 1 for the combinations the gate found not ready, then point the
# waiting Stage 2 jobs at a new gate. Written 2026-11-20 after the first gate
# failed on 41 combinations: 12 stopped by the coastal segments check (features no
# coastal profile reaches; now allowed) and 29 by Check B (protection at or above
# the largest RP; now follows the model's rule).
#
#   DRY=1 bash reruns/20261120_polygon_fix/regate.sh   # list what it would do
#   bash reruns/20261120_polygon_fix/regate.sh         # submit and re-point
#
# 1. gate --write-failed lists the combinations not ready (from the live state).
# 2. restage1 for those: the extraction is skipped where it was already done with
#    the fix, so most jobs only rerun the segments check and validation.
# 3. Its new gate checks all 243 combinations.
# 4. Every Stage 2 job of this batch that is still pending on the old gate gets
#    Dependency=afterok:<new gate>. Job ids stay the same, so the N = 32768 jobs
#    (afterok on their N = 8192 job) need no change.
# Run it again after a further failure: each round gets its own number.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
USER_DIR=/scistor/ivm/${USER:-$(id -un)}
export VENV=${VENV:-${USER_DIR}/.venvs/miraca_uq_dsfix}
export PYTHON=${VENV}/bin/python
export MIRACA_CONFIG=${MIRACA_CONFIG:-$REPO/config.cluster.yml}
export EXCLUDE=${EXCLUDE:-node002,node009}
export S1_TIME=${S1_TIME:-120:00:00}
BACKUP=data/intermediate_ds101_backup_20261120
STATUS=${STATUS:-$HERE/status}
HAZ="river earthquake windstorm"
OLD_GATE=$(cat "$HERE/gate_job.txt")

"$PYTHON" -m src.ds_version --expect-fix --asset power > /dev/null \
    || { echo "$VENV does not hold the pinned polygon fix; nothing done." >&2; exit 1; }

# Stage 1 must have drained, or a still-running job would be listed as failed.
running=$(squeue -u "${USER:-$(id -un)}" -h -o "%j" | grep -c '^s1_' || true)
[[ "$running" -eq 0 ]] || { echo "$running Stage 1 job(s) still queued or running; wait until they end." >&2; exit 1; }

round=1
while [[ -e "$HERE/regate_${round}.txt" ]]; do round=$((round + 1)); done
FAILED="$HERE/failed_${round}.txt"
"$PYTHON" -m src.stage1_rerun gate --list "$HERE/stage1_combos.txt" --hazards $HAZ --backup "$BACKUP" \
    --status "$STATUS" --require-polygon-fix --write-failed "$FAILED" | tail -n 3 || true
[[ -f "$FAILED" ]] || { echo "The gate check did not finish (see above); nothing done." >&2; exit 1; }
n=$(grep -c . "$FAILED" || true)
[[ "$n" -gt 0 ]] || { echo "Every combination is ready. Release Stage 2 by pointing it at a passing gate instead." >&2; exit 1; }

# The Stage 2 jobs of this batch still waiting (the N = 32768 ones wait on their N = 8192 job instead)
mapfile -t S2 < <(cat "$HERE"/jobs_*.log | sed -nE 's/^submitted .* run=([0-9]+).*/\1/p' | sort -u)
mapfile -t N32 < <(sed -nE 's/^submitted .* run=([0-9]+).*/\1/p' "$HERE/jobs_n32768.log" | sort -u)
mapfile -t PENDING < <(squeue -u "${USER:-$(id -un)}" -h -t PENDING -o "%i" | sort -u)
mapfile -t TODO < <(comm -12 <(printf '%s\n' "${S2[@]}") <(printf '%s\n' "${PENDING[@]}") \
                    | comm -23 - <(printf '%s\n' "${N32[@]}"))
EXPECT=$(( ${#S2[@]} - ${#N32[@]} ))
echo "Round ${round}: ${n} combination(s) to redo; ${#TODO[@]} of ${EXPECT} Stage 2 jobs pending on gate ${OLD_GATE}"
if [[ ${#TODO[@]} -lt $EXPECT ]]; then
    echo "!! $(( EXPECT - ${#TODO[@]} )) Stage 2 job(s) are no longer pending (cancelled by SLURM?)." >&2
    echo "!! They have to be resubmitted; see README 'If the gate fails'. The rest is re-pointed." >&2
fi
if [[ -n "${DRY:-}" ]]; then
    echo "DRY: would resubmit Stage 1 for:"; sed 's/^/   /' "$FAILED"
    rm -f "$FAILED"
    exit 0
fi

RESTAGE_LIST="$FAILED" RESTAGE_GATE_LIST="$HERE/stage1_combos.txt" RESTAGE_HAZARDS="$HAZ" \
RESTAGE_BACKUP="$BACKUP" RESTAGE_STATUS="$STATUS" RESTAGE_GATE_FILE="$HERE/gate_job_${round}.txt" \
RESTAGE_GATE_ARGS="--require-polygon-fix" SCEN_TAG="dsfix_r${round}" \
    ./submit_miraca_uncertainty_study.sh restage1 | tee "$HERE/jobs_stage1_r${round}.log"
NEW_GATE=$(cat "$HERE/gate_job_${round}.txt")
# Mail when the new gate ends (Completed = Stage 2 starts, Failed = look at it), to
# the address notify_when_drained.sh uses.
MAIL_TO=${MAIL_TO:-$(sed -nE 's/^MAIL_TO=\$\{MAIL_TO:-(.*)\}$/\1/p' "$REPO/notify_when_drained.sh")}
[[ -n "$MAIL_TO" ]] && scontrol update JobId="$NEW_GATE" MailUser="$MAIL_TO" MailType=END,FAIL \
    && echo "gate ${NEW_GATE} mails ${MAIL_TO} when it ends"

fail=0
for id in "${TODO[@]}"; do
    scontrol update JobId="$id" Dependency="afterok:${NEW_GATE}" || fail=$((fail + 1))
done
{
    echo "round ${round} $(date -u +%FT%TZ): ${n} combination(s) redone ($(basename "$FAILED")), new gate ${NEW_GATE}"
    echo "re-pointed $(( ${#TODO[@]} - fail )) Stage 2 job(s) from gate ${OLD_GATE} to ${NEW_GATE}; ${fail} failed"
} | tee "$HERE/regate_${round}.txt"
echo "$NEW_GATE" > "$HERE/gate_job.txt"
echo
echo "New gate verdict once these finish:  cat \$LOG_DIR/out_s1_gate_dsfix_r${round}"
