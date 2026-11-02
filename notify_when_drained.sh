#!/bin/bash -l
# ---------------------------------------------------------------------------
# Attach an email notification to work that is ALREADY submitted.
#
# Reads every job currently queued or running for this user, chains a no-op
# barrier behind each chunk of them, and puts one notify job behind the
# barriers. That job builds a short report (how many jobs ended in which state,
# which ones did not COMPLETE, the SLR gate verdict, which cascade outputs
# landed) and mails it.
#
# Why barriers rather than one dependency: a Stage-2 back-fill is ~257 separate
# jobs, and a single afterany: list of that many IDs runs into SLURM's
# dependency-string limit. submit_miraca_uncertainty_study.sh already solves
# this with chunked barriers; same idea here, standalone so it can be run after
# the fact.
#
# Usage:
#   ./notify_when_drained.sh                      # mail the default address
#   MAIL_TO=you@vu.nl ./notify_when_drained.sh    # override
#   DRY=1 ./notify_when_drained.sh                # show what it would submit
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
BARRIER_CHUNK=${BARRIER_CHUNK:-50}
PARTITION=${PARTITION:-}

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

echo "== attaching notification to ${#IDS[@]} queued job(s) =="
echo "   mail to: ${MAIL_TO}"
echo "   job ids: ${IDS[*]}"

JOBLIST=${LOG_DIR}/notify_watched_jobs.txt
if [[ -n "${DRY:-}" ]]; then
    echo "   DRY set - nothing submitted."
    exit 0
fi
printf '%s\n' "${IDS[@]}" > "${JOBLIST}"

# ── Barriers ─────────────────────────────────────────────────────────────────
BARRIERS=()
for ((i = 0; i < ${#IDS[@]}; i += BARRIER_CHUNK)); do
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
BARRIER_IDS=$(IFS=:; echo "${BARRIERS[*]}")
echo "   barriers: ${BARRIERS[*]}"

# ── The notify job ───────────────────────────────────────────────────────────
# afterany on the barriers, which are themselves afterany: the mail must arrive
# whether the work succeeded or failed - a silent failure is the case most worth
# being told about.
#
# --mail-type is set as well as sending mail from the body. SLURM's own message
# is terse but needs no MTA on the compute node; the body's report is useful but
# only arrives if a mail command exists. Belt and braces, because which of the
# two works is a property of the cluster, not of this script.
jid=$(sbatch --parsable --dependency="afterany:${BARRIER_IDS}" <<SLURM
#!/bin/bash -l
#SBATCH --job-name=notify_drained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/out_notify_drained
#SBATCH --error=${LOG_DIR}/err_notify_drained
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_TO}
set -uo pipefail
cd ${REPO}

REPORT=${LOG_DIR}/drained_report.txt
WATCHED=\$(paste -sd, ${JOBLIST})

{
    echo "MIRACA cluster queue drained"
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

# The SLR gate decides whether Stage 2 was allowed to run at all, so its verdict
# is the single most informative line in the whole chain.
{
    echo
    echo "--- SLR state completeness ---"
    if [[ -f ${LOG_DIR}/out_slr_gate ]]; then
        tail -n 20 ${LOG_DIR}/out_slr_gate
    else
        echo "  (no gate log yet)"
    fi

    echo
    echo "--- cascade outputs ---"
    for d in results/cascade/*/; do
        n=\$(ls "\${d}" 2>/dev/null | grep -c '_cascade.parquet' || true)
        echo "  \${d}  \${n} parquet(s)"
    done
} >> "\${REPORT}" 2>&1

cat "\${REPORT}"

# Try to mail the report itself. Which of these exists is cluster-specific, so
# fall through rather than assuming any one of them.
SUBJ="MIRACA: cluster queue drained"
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
