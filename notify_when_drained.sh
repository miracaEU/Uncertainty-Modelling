#!/bin/bash -l
# ---------------------------------------------------------------------------
# Attach an email notification to work that is ALREADY submitted.
#
# Reads every job currently queued or running for this user and submits one
# small notify job that waits for all of them to leave the queue. It then builds
# a short report (how many jobs ended in which state, which ones did not
# COMPLETE, the newest gate verdicts, which cascade outputs landed) and mails it.
#
# Two ways to wait (MODE):
#   poll     (default) the notify job has no dependency and checks squeue every
#            POLL_EVERY seconds. Works for chained batches. If everything left
#            can never start (DependencyNeverSatisfied, e.g. a failed gate), it
#            mails at once with a "stuck" subject instead of waiting forever.
#            Holds one CPU for up to POLL_TIME; past that SLURM mails TIME_LIMIT,
#            and running this script again attaches a fresh watcher.
#   barrier  no-op barriers with afterany on chunks of BARRIER_CHUNK jobs, the
#            notify job behind them. Fails on chained batches (Stage 1 -> gate
#            -> Stage 2 -> N = 32768): SLURM's dependency scan gives up after
#            max_depend_depth jobs and reports "Circular job dependency" (seen
#            2026-11-20 with the polygon-fix batch).
#
# Usage:
#   ./notify_when_drained.sh                      # mail the default address
#   MAIL_TO=you@vu.nl ./notify_when_drained.sh    # override
#   DRY=1 ./notify_when_drained.sh                # show what it would submit
#   MODE=barrier ./notify_when_drained.sh         # the old dependency chain
#
# Run it again at any time to attach a fresh notification to whatever is then
# in the queue; the jobs it watches are fixed at submission.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$SCRIPT_DIR}
USER_DIR=${USER_DIR:-/scistor/ivm/${USER:-$(id -un)}}
VENV=${VENV:-${USER_DIR}/.venvs/miraca_uq}
PYTHON=${PYTHON:-${VENV}/bin/python}
LOG_DIR=${LOG_DIR:-${USER_DIR}/MIRACA_UQ/logs}
MODULES=${MODULES:-"gdal/3.6.2"}
export MIRACA_CONFIG=${MIRACA_CONFIG:-${REPO}/config.cluster.yml}

# The institutional address, not the git one: git identity here lives in the
# global config on the workstation, which the cluster home does not have, so
# reading it from git produced an empty default. An internal address is also the
# likelier to arrive - outbound relay to external providers is blocked at many
# sites even when internal delivery works, and no bounce comes back to say so.
MAIL_TO=${MAIL_TO:-v.k.c.weiwad@vu.nl}
MODE=${MODE:-poll}
POLL_EVERY=${POLL_EVERY:-600}
POLL_TIME=${POLL_TIME:-120:00:00}
BARRIER_CHUNK=${BARRIER_CHUNK:-50}
PARTITION=${PARTITION:-}
[[ "$MODE" == poll || "$MODE" == barrier ]] || { echo "MODE must be poll or barrier, not '$MODE'." >&2; exit 1; }

SB_PARTITION="# (no --partition set; using the cluster default)"
[[ -n "$PARTITION" ]] && SB_PARTITION="#SBATCH --partition=${PARTITION}"

if [[ -z "${MAIL_TO}" ]]; then
    echo "No address to mail. Pass MAIL_TO=you@example.org." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

# %F is the array job's base ID, and the plain job ID for non-array jobs, so
# this yields one entry per job array rather than one per task. Depending on the
# base ID waits for every task in that array.
mapfile -t IDS < <(squeue -h -u "${USER:-$(id -un)}" -o "%F" | sort -un)

if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "Nothing queued or running for ${USER:-$(id -un)} - the queue is already drained." >&2
    echo "Nothing to wait for, so no notification submitted." >&2
    exit 1
fi

echo "== attaching notification to ${#IDS[@]} queued job(s), mode ${MODE} =="
echo "   mail to: ${MAIL_TO}"
echo "   job ids: ${IDS[0]} ... ${IDS[-1]}"

JOBLIST=${LOG_DIR}/notify_watched_jobs.txt
if [[ -n "${DRY:-}" ]]; then
    echo "   DRY set - nothing submitted."
    exit 0
fi
printf '%s\n' "${IDS[@]}" > "${JOBLIST}"

# ── How the notify job waits ─────────────────────────────────────────────────
# poll: no dependency, a loop at the top of the job body. The watched list is
# matched against the user's own squeue, so finished ids simply drop out.
BARRIERS=()
SB_DEPEND="# (no dependency: the job polls squeue)"
SB_TIME="#SBATCH --time=00:30:00"
WAIT_BLOCK=""
if [[ "$MODE" == poll ]]; then
    SB_TIME="#SBATCH --time=${POLL_TIME}"
    WAIT_BLOCK=$(cat <<POLL
STUCK=""
while :; do
    LEFT=\$(squeue -h -u "\${USER:-\$(id -un)}" -o "%F %T %r" | awk 'NR==FNR {w[\$1]; next} (\$1 in w)' ${JOBLIST} -)
    [[ -z "\${LEFT}" ]] && break
    # Only jobs that can never start are left (a failed gate): report now.
    if ! grep -qv DependencyNeverSatisfied <<<"\${LEFT}"; then
        STUCK=\$(wc -l <<<"\${LEFT}")
        break
    fi
    sleep ${POLL_EVERY}
done
POLL
)
fi

# ── Barriers (MODE=barrier only) ─────────────────────────────────────────────
for ((i = 0; i < ${#IDS[@]}; i += BARRIER_CHUNK)); do
    [[ "$MODE" == barrier ]] || break
    chunk=$(IFS=:; echo "${IDS[*]:i:BARRIER_CHUNK}")
    bid=$(sbatch --parsable --dependency="afterany:${chunk}" <<SLURM
#!/bin/bash
#SBATCH --job-name=notify_barrier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
exit 0
SLURM
)
    BARRIERS+=("$bid")
done
if [[ "$MODE" == barrier ]]; then
    BARRIER_IDS=$(IFS=:; echo "${BARRIERS[*]}")
    SB_DEPEND="#SBATCH --dependency=afterany:${BARRIER_IDS}"
    echo "   barriers: ${BARRIERS[*]}"
fi

# ── The notify job ───────────────────────────────────────────────────────────
# afterany on the barriers, which are themselves afterany: the mail must arrive
# whether the work succeeded or failed - a silent failure is the case most worth
# being told about.
#
# --mail-type is set as well as sending mail from the body. SLURM's own message
# is terse but needs no MTA on the compute node; the body's report is useful but
# only arrives if a mail command exists. Belt and braces, because which of the
# two works is a property of the cluster, not of this script.
jid=$(sbatch --parsable <<SLURM
#!/bin/bash -l
#SBATCH --job-name=notify_drained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
${SB_DEPEND}
#SBATCH --mem=2G
${SB_TIME}
#SBATCH --output=${LOG_DIR}/out_notify_drained
#SBATCH --error=${LOG_DIR}/err_notify_drained
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=${MAIL_TO}
set -uo pipefail
cd ${REPO}
STUCK=""
${WAIT_BLOCK}

REPORT=${LOG_DIR}/drained_report.txt
WATCHED=\$(paste -sd, ${JOBLIST})

{
    if [[ -n "\${STUCK}" ]]; then
        echo "MIRACA: \${STUCK} job(s) can never start (DependencyNeverSatisfied) - check the gate"
        squeue -h -u "\${USER:-\$(id -un)}" -o "%F %j %r" | awk 'NR==FNR {w[\$1]; next} (\$1 in w)' ${JOBLIST} - | head -n 5
    else
        echo "MIRACA cluster queue drained"
    fi
    echo "\$(date '+%Y-%m-%d %H:%M:%S')  on \$(hostname)"
    echo
    echo "Watched \$(wc -l < ${JOBLIST}) submitted job(s)/array(s)."
    echo
    echo "--- final states ---"
    sacct -j "\${WATCHED}" -X -n -o State | awk '{print \$1}' | sort | uniq -c | sort -rn
    echo
    echo "--- anything that did not COMPLETE ---"
    sacct -j "\${WATCHED}" -X -n -o JobID,JobName%28,State,Elapsed,MaxRSS,ExitCode | grep -v COMPLETED || echo "  (nothing - every job completed)"
} > "\${REPORT}" 2>&1

# A batch's gate decides whether its Stage 2 was allowed to run at all, so its
# verdict is the single most informative line in the whole chain. Every batch
# names its gate log out_*gate* (out_slr_gate, out_s1_gate_<tag>); show the newest.
{
    echo
    echo "--- newest gate verdicts ---"
    GATES=\$(ls -t ${LOG_DIR}/out_*gate* 2>/dev/null | head -3)
    if [[ -n "\${GATES}" ]]; then
        for g in \${GATES}; do
            echo "  \${g}  (\$(date -r "\${g}" '+%Y-%m-%d %H:%M'))"
            grep -E 'ready for Stage 2|FAIL|missing' "\${g}" | head -n 5 | sed 's/^/    /'
        done
    else
        echo "  (no gate log yet)"
    fi

    echo
    echo "--- cascade outputs ---"
    for d in results/cascade/*/; do
        n=\$(ls "\${d}" 2>/dev/null | grep -c '_cascade.parquet' || true)
        echo "  \${d}  \${n} parquet(s)"
    done

    # The convergence reruns end in a retest job (src/plan_convergence_reruns.py);
    # its log says whether the old archives survived and what still fails.
    echo
    echo "--- newest convergence retest ---"
    LAST=\$(ls -t convergence/reruns/*/retest_*.out 2>/dev/null | head -1)
    if [[ -n "\${LAST}" ]]; then
        echo "  \${LAST}"
        tail -n 30 "\${LAST}"
    else
        echo "  (no retest log yet)"
    fi
} >> "\${REPORT}" 2>&1

cat "\${REPORT}"

# Try to mail the report itself. Which of these exists is cluster-specific, so
# fall through rather than assuming any one of them.
SUBJ="MIRACA: cluster queue drained"
[[ -n "\${STUCK}" ]] && SUBJ="MIRACA: jobs stuck behind a failed dependency - check the gate"
if command -v mail >/dev/null 2>&1; then
    mail -s "\${SUBJ}" "${MAIL_TO}" < "\${REPORT}" && echo "[sent via mail]"
elif command -v mailx >/dev/null 2>&1; then
    mailx -s "\${SUBJ}" "${MAIL_TO}" < "\${REPORT}" && echo "[sent via mailx]"
elif command -v sendmail >/dev/null 2>&1; then
    { echo "To: ${MAIL_TO}"; echo "Subject: \${SUBJ}"; echo; cat "\${REPORT}"; } | sendmail -t && echo "[sent via sendmail]"
else
    echo "[no mail command on this node - report is at \${REPORT};"
    echo " SLURM's own --mail-type message should still have been sent]"
fi
exit 0
SLURM
)

echo "   notify job: ${jid}"
echo
echo "You will get mail at ${MAIL_TO} once all ${#IDS[@]} watched jobs have ended."
echo "Report will also be at ${LOG_DIR}/drained_report.txt"
echo
echo "check mail is even configured here:  scontrol show config | grep -i MailProg"
echo "cancel the notification:             scancel ${jid} ${BARRIERS[*]}"
