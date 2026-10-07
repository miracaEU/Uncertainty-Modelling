#!/bin/bash -l
# ---------------------------------------------------------------------------
# Finish SLR Stage 1 for the combinations the 32 G run lost to OOM, then queue
# Stage 2 behind it in the same submission.
#
# Three jobs, chained:
#   1. slr_s1   array over slr_retry.txt  - preprocess --slr-only, on defq-fat
#   2. slr_gate afterany:1                - asserts EVERY configured state is on
#                                           disk for EVERY coastal combo
#   3. Stage 2  afterok:2                 - delegated to the addscen mode of
#                                           submit_miraca_uncertainty_study.sh,
#                                           which already has per-combo
#                                           resources tuned by country tier
#
# The gate exists because afterok alone is not enough. risk_model.load_model_data
# skips a missing state file without comment, and compute_risk resolves an
# unavailable state via .get(slr, hazards["coastal"]) - so a combo holding only
# some states still RUNS, silently substituting present-day coastal hazard for
# the missing horizons and understating warming. Its own guard only fires below
# two states, which catches the all-missing case and nothing else. The gate
# turns that silent degradation into a failed dependency.
#
# Usage:
#   ./submit_slr.sh              # Stage 1 retry + gate + Stage 2
#   GATE_ONLY=1 ./submit_slr.sh  # just check completeness now, submit nothing
#   NO_STAGE2=1 ./submit_slr.sh  # Stage 1 + gate only
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

RETRY_LIST=${RETRY_LIST:-${REPO}/slr_retry.txt}
STAGE2_LIST=${STAGE2_LIST:-${REPO}/slr_stage2.txt}
STAGE2_SCEN=${STAGE2_SCEN:-"coastal_absprot_ds_slr coastal_noprot_ds_slr"}

# ── Resources ────────────────────────────────────────────────────────────────
# 32 G was killed on 430 of 2313 extractions, worst at the 2100 horizons where
# the flood footprint is largest. 160 G on the fat partition, and a walltime far
# past the ~2 h a surviving combo took, because a wall kill costs a resubmission
# and idle reserved time costs nothing.
S1_MEM=${S1_MEM:-160G}
S1_CPUS=${S1_CPUS:-1}
S1_TIME=${S1_TIME:-48:00:00}
S1_PARTITION=${S1_PARTITION:-defq-fat}
EXCLUDE=${EXCLUDE:-node009}

mkdir -p "${LOG_DIR}"

[[ -x "${PYTHON}" ]] || { echo "No venv python at ${PYTHON}." >&2; exit 1; }
[[ -f "${RETRY_LIST}" ]] || { echo "No retry list at ${RETRY_LIST}." >&2; exit 1; }
[[ -f "${STAGE2_LIST}" ]] || { echo "No stage-2 list at ${STAGE2_LIST}." >&2; exit 1; }

if [[ -n "${GATE_ONLY:-}" ]]; then
    module load ${MODULES} 2>/dev/null || true
    cd "${REPO}"
    exec "${PYTHON}" -m src.check_slr_states
fi

# ── 1. Stage 1 retry ─────────────────────────────────────────────────────────
N_RETRY=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' "${RETRY_LIST}" || true)
[[ "${N_RETRY}" -gt 0 ]] || { echo "Retry list is empty." >&2; exit 1; }
echo "== SLR Stage 1 retry: ${N_RETRY} combos, ${S1_MEM} on ${S1_PARTITION} =="

jid_s1=$(sbatch --parsable <<SLURM
#!/bin/bash -l
#SBATCH --job-name=slr_s1
#SBATCH --array=1-${N_RETRY}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${S1_CPUS}
#SBATCH --partition=${S1_PARTITION}
#SBATCH --exclude=${EXCLUDE}
#SBATCH --mem=${S1_MEM}
#SBATCH --time=${S1_TIME}
#SBATCH --output=${LOG_DIR}/out_slr_s1_%a
#SBATCH --error=${LOG_DIR}/err_slr_s1_%a
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
module load ${MODULES}
cd ${REPO}

LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${RETRY_LIST})
C=\$(echo \${LINE} | awk '{print \$1}')
A=\$(echo \${LINE} | awk '{print \$2}')
echo "task \${SLURM_ARRAY_TASK_ID}: \${C} \${A}"

# Array form, no line continuations - a continuation whose backslash picks up a
# trailing space becomes an escaped space and the command ends there.
PREP_ARGS=(
    --country "\${C}"
    --asset "\${A}"
    --slr-only
)
echo "argv: \${PREP_ARGS[@]}"
${PYTHON} -m src.preprocess "\${PREP_ARGS[@]}"
SLURM
)
echo "submitted Stage 1 array: ${jid_s1}"

# ── 2. The gate ──────────────────────────────────────────────────────────────
# afterany, not afterok: if a task still OOMs the gate should SAY which combos
# are short, rather than leaving a dependency forever unsatisfiable with no
# diagnosis. The gate's own exit code is what blocks Stage 2.
jid_gate=$(sbatch --parsable --dependency=afterany:${jid_s1} <<SLURM
#!/bin/bash -l
#SBATCH --job-name=slr_gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/out_slr_gate
#SBATCH --error=${LOG_DIR}/err_slr_gate
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
module load ${MODULES}
cd ${REPO}
${PYTHON} -m src.check_slr_states
SLURM
)
echo "submitted gate:          ${jid_gate} (afterany the array)"

# ── 3. Stage 2 ───────────────────────────────────────────────────────────────
if [[ -n "${NO_STAGE2:-}" ]]; then
    echo
    echo "NO_STAGE2 set - Stage 2 not queued. Once the gate passes, run:"
    echo "  SCEN=\"${STAGE2_SCEN}\" ADDSCEN_LIST=${STAGE2_LIST} SCEN_TAG=slr \\"
    echo "    ./submit_miraca_uncertainty_study.sh addscen"
    exit 0
fi

N_S2=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' "${STAGE2_LIST}" || true)
echo "== SLR Stage 2: ${N_S2} combos, held on afterok:${jid_gate} =="
# Delegated so the per-combo cpu/mem/walltime tuning in that script is reused
# rather than reimplemented. SCEN_TAG keeps these logs clear of the river and
# baseline-coastal addscen runs, which overwrote each other once before.
DEPEND="afterok:${jid_gate}" \
SCEN="${STAGE2_SCEN}" \
ADDSCEN_LIST="${STAGE2_LIST}" \
SCEN_TAG="slr" \
    "${REPO}/submit_miraca_uncertainty_study.sh" addscen

echo
echo "chain: Stage1 ${jid_s1} -> gate ${jid_gate} -> Stage 2 (${N_S2} jobs)"
echo "watch:  squeue -u ${USER:-$(id -un)}"
echo "gate verdict afterwards:  cat ${LOG_DIR}/out_slr_gate"
