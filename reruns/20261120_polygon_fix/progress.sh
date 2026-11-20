#!/bin/bash
# Progress of the polygon-fix batch, per step: how many jobs completed out of how
# many, the other states, the gate verdict, why jobs are pending, what failed,
# and the jobs running right now. Job ids come from the jobs_*.log files that
# submit.sh wrote, so other work in the queue is not counted.
#
#   bash reruns/20261120_polygon_fix/progress.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-/scistor/ivm/${USER:-$(id -un)}/MIRACA_UQ/logs}
SINCE=2026-11-20
ME=${USER:-$(id -un)}

ids() { sed -nE 's/^submitted .* (s1|run)=([0-9]+).*/\2/p' "$@" 2>/dev/null; }

DEFAULT_LOGS=()
for f in "$HERE"/jobs_*.log; do
    case $(basename "$f") in
        jobs_stage1.log | jobs_n32768.log | jobs_earthquake.log) ;;
        *) DEFAULT_LOGS+=("$f") ;;
    esac
done

ALL_IDS=$(mktemp)
trap 'rm -f "$ALL_IDS"' EXIT
{ ids "$HERE"/jobs_*.log; cat "$HERE/gate_job.txt" 2>/dev/null; } | sort -u > "$ALL_IDS"

summary() {   # $1 = step name, job ids on stdin
    local list
    list=$(paste -sd, -)
    if [[ -z "$list" ]]; then
        printf '%-30s no jobs\n' "$1"
        return
    fi
    sacct -X -n -S "$SINCE" -j "$list" -o State%30 | awk -v name="$1" '
        { s = $1; n[s]++; t++ }
        END {
            printf "%-30s %5d / %-5d completed", name, n["COMPLETED"], t
            for (k in n) if (k != "COMPLETED") printf "   %s %d", k, n[k]
            printf "\n"
        }'
}

echo "== polygon-fix batch, $(date '+%Y-%m-%d %H:%M') =="
ids "$HERE/jobs_stage1.log"     | summary "Stage 1"
cat "$HERE/gate_job.txt"        | summary "Gate"
ids "${DEFAULT_LOGS[@]}"        | summary "Stage 2, default scenarios"
ids "$HERE/jobs_earthquake.log" | summary "Stage 2, extra (wide PGA)"
ids "$HERE/jobs_n32768.log"     | summary "N = 32768"
summary "All" < "$ALL_IDS"

echo
GATE_LOG="$LOG_DIR/out_s1_gate_dsfix"
if [[ -f "$GATE_LOG" ]]; then
    echo "Gate verdict: $(grep -m1 'ready for Stage 2' "$GATE_LOG" || echo 'see '"$GATE_LOG")"
else
    echo "Gate verdict: not run yet"
fi

echo
echo "-- pending, by reason --"
squeue -u "$ME" -h -t PENDING -o "%F %r" | awk 'NR==FNR {w[$1]; next} ($1 in w) {print $2}' "$ALL_IDS" - \
    | sort | uniq -c | sort -rn

echo
echo "-- ended without COMPLETED (first 20) --"
sacct -X -n -S "$SINCE" -j "$(paste -sd, "$ALL_IDS")" -o JobID,JobName%28,State%20,Elapsed,ExitCode \
    | grep -Ev 'COMPLETED|PENDING|RUNNING' | head -n 20 | grep . || echo "   none"

echo
echo "-- running now --"
squeue -u "$ME" -h -t RUNNING -o "%F|%i %.28j %.11M %.8N" \
    | awk -F'|' 'NR==FNR {w[$1]; next} ($1 in w) {print $2; n++} END {print "   " n+0 " running"}' "$ALL_IDS" -
