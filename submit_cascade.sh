#!/bin/bash -l
# ---------------------------------------------------------------------------
# Submit an uncertainty-cascade (trickle-down pyramid) run to SLURM.
#
# The cascade is a SINGLE pass that walks one factor at a time, so there is one
# job per (asset, scenario) - never one per factor or per curve group. Scenarios
# are run as a job array because their outputs are separate files
# ({asset}_{scenario}_cascade.parquet), so they parallelise cleanly; only
# cascade_order.txt is shared, and write_order_file() returns early once it
# exists with a matching signature.
#
# Usage:
#   ./submit_cascade.sh                 # the power/flood custom-order run
#   ASSET=rail ./submit_cascade.sh      # same ordering, different asset
#   NO_PLOT=1 ./submit_cascade.sh       # skip the chained figure job
#
# Defaults reproduce the run this script was written for: 3000 LHS draws,
# protection ordered last, and the frozen protection standard taken from the
# recorded FLOPROS value rather than the midpoint of the sampled range. That
# last choice is what makes the pre-protection steps reproduce the _noprot_
# scenario, so flood_noprot_ds does not need to be run separately.
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

# ── What to run ──────────────────────────────────────────────────────────────
ASSET=${ASSET:-power}
# One array task per entry. Keep them space-free.
SCENARIOS=${SCENARIOS:-"flood_baseline_ds_sc0p5to1p5 flood_absprot_ds"}
N_DRAWS=${N_DRAWS:-3000}
SCOPE_LABEL=${SCOPE_LABEL:-eu_sum}
PROT_NOMINAL=${PROT_NOMINAL:-recorded}
# Protection LAST so every step before it is free of the protection treatment,
# and warming after the curves so the curve/cost steps are readable without it.
FACTOR_ORDER=${FACTOR_ORDER:-"depth_scale aggregation curve_ cost_level warming protection_scale protection_abs_rp"}

# ── Resources ────────────────────────────────────────────────────────────────
# Walltime is deliberately far beyond the ~16 h a scenario needs: a job killed
# at the wall costs a full resubmission, idle reserved time costs nothing.
C_CPUS=${C_CPUS:-8}
C_MEM=${C_MEM:-8G}          # per cpu, so 8x8G = 64G per job
C_TIME=${C_TIME:-120:00:00}
# node009 runs ~2.3x slower per iteration than the other nodes; not worth it on
# a job this long.
EXCLUDE=${EXCLUDE:-node009}
PARTITION=${PARTITION:-}

SB_PARTITION="# (no --partition set; using the cluster default)"
[[ -n "$PARTITION" ]] && SB_PARTITION="#SBATCH --partition=${PARTITION}"
SB_EXCLUDE="# (no nodes excluded)"
[[ -n "$EXCLUDE" ]] && SB_EXCLUDE="#SBATCH --exclude=${EXCLUDE}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "No venv python at ${PYTHON}." >&2
    echo "Run './submit_miraca_uncertainty_study.sh setup' first." >&2
    exit 1
fi

read -r -a SCEN_ARR <<< "${SCENARIOS}"
N_SCEN=${#SCEN_ARR[@]}
echo "== cascade: ${ASSET} x ${N_SCEN} scenario(s), n=${N_DRAWS} =="

jid=$(sbatch --parsable <<SLURM
#!/bin/bash -l
#SBATCH --job-name=casc_${ASSET}
#SBATCH --array=0-$((N_SCEN - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${C_CPUS}
${SB_PARTITION}
${SB_EXCLUDE}
#SBATCH --mem-per-cpu=${C_MEM}
#SBATCH --time=${C_TIME}
#SBATCH --output=${LOG_DIR}/out_cascade_${ASSET}_%a
#SBATCH --error=${LOG_DIR}/err_cascade_${ASSET}_%a
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
# The parallelism is across cascade workers; nested BLAS threads would
# oversubscribe the allocation badly.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
module load ${MODULES}   # libgdal on LD_LIBRARY_PATH for osgeo
cd ${REPO}

SCENS=(${SCENARIOS})
SCEN=\${SCENS[\${SLURM_ARRAY_TASK_ID}]}

# Stagger the non-zero tasks so only one of them can be creating
# cascade_order.txt in a fresh output folder. The content is identical either
# way, so this only avoids a torn read, not a wrong result.
[[ \${SLURM_ARRAY_TASK_ID} -ne 0 ]] && sleep 45

echo "task \${SLURM_ARRAY_TASK_ID}: ${ASSET} / \${SCEN}"

# Arguments as an array, NOT backslash-continued lines. A continuation whose
# backslash picks up a trailing space silently becomes an escaped space - an
# empty argument - and the command ends there; that is what killed the first
# attempt at this job ("--assets: invalid choice: ' '"). Newlines inside ( )
# need no continuation character at all, so there is nothing left to mangle.
CASCADE_ARGS=(
    --assets ${ASSET}
    --scenarios "\${SCEN}"
    --n ${N_DRAWS}
    --workers ${C_CPUS}
    --protection-nominal ${PROT_NOMINAL}
    --factor-order ${FACTOR_ORDER}
    --scope-label ${SCOPE_LABEL}
)
echo "argv: \${CASCADE_ARGS[@]}"
${PYTHON} -m src.cascade "\${CASCADE_ARGS[@]}"
SLURM
)
echo "submitted cascade array: ${jid}"

if [[ -z "${NO_PLOT:-}" ]]; then
    # afterok on the whole array: figures only make sense once every scenario
    # is in, because --pair-protection needs both panels.
    jid_plot=$(sbatch --parsable --dependency=afterok:${jid} <<SLURM
#!/bin/bash -l
#SBATCH --job-name=casc_figs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/out_cascade_figs
#SBATCH --error=${LOG_DIR}/err_cascade_figs
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
export MPLBACKEND=Agg
module load ${MODULES}
cd ${REPO}

# The order folder is derived from the factor ordering, so ask the cascade
# itself rather than hard-coding the hash. Array form for the same reason as
# above: no continuation characters to pick up stray whitespace.
PROBE_ARGS=(
    --protection-nominal ${PROT_NOMINAL}
    --factor-order ${FACTOR_ORDER}
    --print-out-dir
)
CDIR=\$(${PYTHON} -m src.cascade "\${PROBE_ARGS[@]}" 2>/dev/null | tail -1)
echo "cascade dir: \${CDIR}"
if [[ ! -d "\${CDIR}" ]]; then
    echo "resolved cascade dir does not exist: \${CDIR}" >&2
    exit 1
fi

# Merged: one vulnerability-curve level, for reporting.
${PYTHON} -m src.plot_pyramid --cascade-dir "\${CDIR}" --scopes ${SCOPE_LABEL} --merge-curves
# Decomposed: which curve group actually carries the uncertainty. Re-reads the
# same parquet, so it costs no model time.
${PYTHON} -m src.plot_pyramid --cascade-dir "\${CDIR}" --scopes ${SCOPE_LABEL} --out-dir overview_figures/pyramids/custom_by_curve_group
SLURM
)
    echo "submitted figures:      ${jid_plot} (after the array)"
fi

echo
echo "watch with:  squeue -u ${USER:-$(id -un)}"
echo "logs:        ${LOG_DIR}/out_cascade_${ASSET}_*"
