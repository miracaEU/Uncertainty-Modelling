#!/bin/bash
# submit_miraca_uncertainty_study.sh
#
# Submit the MIRACA uncertainty study (EMA Workbench + Sobol) to SLURM for
# every (country, asset) combination that has exposure data.
#
# Each combination is submitted as TWO chained jobs, because Stage 1 and
# Stage 2 have opposite resource profiles:
#
#   prep_<C>_<A>   Stage 1: preprocess (GIS overlay of exposure x hazard
#                  rasters). Single-threaded but memory-hungry -> 1 CPU, lots
#                  of RAM. Writes data/intermediate/*.parquet.
#   run_<C>_<A>    Stage 2: LHS + Sobol experiments for every applicable
#                  default scenario, then the per-combo analyses. Pure numpy,
#                  embarrassingly parallel -> many CPUs, moderate RAM each.
#                  Submitted with --dependency=afterok on the prep job.
#
# Running them as one job would idle 8-16 CPUs for the whole GIS stage.
#
# IMPORTANT on memory: ema_workbench's MultiprocessingEvaluator gives every
# worker process its OWN copy of the loaded model data (src/ema_model.py
# caches per process), so Stage 2 memory scales with the number of workers.
# That is exactly what --mem-per-cpu expresses, so Stage 2 uses --mem-per-cpu
# and Stage 1 (a single process) uses a flat --mem.
#
# Usage:  ./submit_miraca_uncertainty_study.sh <mode> [asset ...]
#
# Modes:
#   all         THE ONE-COMMAND PATH. venv + every combination + the summary
#               workbook, all chained by SLURM dependencies. Returns
#               immediately; the cluster then runs the study unattended.
#   setup       one-time: create the venv and install requirements
#   dry         print the submission plan, submit nothing
#   pilot       2 combos end-to-end - worth running before `all`
#   submit      submit every combination, but do NOT chain the summary
#   resubmit    re-run only the combos listed in failed_combos.txt (resume-safe),
#               then chain the summary - use after a partial run / a code fix
#   recoastal   recompute ONLY the coastal scenario (Stage 2, forced) for the
#               finished combos in coastal_rerun.txt - after a coastal RP change
#   aggregate   build the summary workbook (`all` chains this for you)
#   status      show this user's queued/running jobs
#
# An optional asset list narrows the run, and ONLY_COUNTRIES narrows countries:
#   ./submit_miraca_uncertainty_study.sh all roads power
#   ONLY_COUNTRIES="DEU FRA" ./submit_miraca_uncertainty_study.sh all roads
#
set -euo pipefail

# ── Paths — auto-detected; override any from the environment ─────────────────
# Nothing here is tied to a specific user, so a colleague can run their own
# clone unchanged (useful for a test submission from a working account):
#   REPO    defaults to the directory THIS script lives in, so it always points
#           at the clone you launched it from - no editing, works for any user.
#   USER_DIR defaults to the SUBMITTING user's own space (/scistor/ivm/$USER),
#           so the venv and logs below land in their directory, not someone
#           else's (whoever runs it must be able to write there).
# Override explicitly if your layout differs, e.g.
#   VENV=/scistor/ivm/eks510/.venvs/miraca_uq ./submit_miraca_uncertainty_study.sh pilot
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$SCRIPT_DIR}
USER_DIR=${USER_DIR:-/scistor/ivm/${USER:-$(id -un)}}
VENV=${VENV:-${USER_DIR}/.venvs/miraca_uq}
PYTHON=${PYTHON:-${VENV}/bin/python}
LOG_DIR=${LOG_DIR:-${USER_DIR}/MIRACA_UQ/logs}

# Shared input data lives under eks510; only the repo and outputs are ours.
EXPOSURE_DIR=${EXPOSURE_DIR:-/scistor/ivm/eks510/MIRACA_EXPOSURE}
export MIRACA_CONFIG=${MIRACA_CONFIG:-${REPO}/config.cluster.yml}

# Partition is OPTIONAL and unset by default: on this cluster defq is already
# the default partition (the `*` in `sinfo -s`), and naming one explicitly can
# be rejected if your association doesn't grant it. Set it only when you need a
# specific one - e.g. PARTITION=defq-fat for the high-memory nodes, which the
# XL tier (64-128 GB) may require. Others: defq-thin, defq-gpu, binf, bw.
PARTITION=${PARTITION:-}
if [[ -n "$PARTITION" ]]; then
    SB_PARTITION="#SBATCH --partition=${PARTITION}"
else
    SB_PARTITION="# (no --partition set; using the cluster default)"
fi
# Some clusters require an explicit accounting group. Find yours with:
#   sacctmgr show assoc user=$USER format=account,partition,qos
# then export it, e.g. ACCOUNT=ivm ./submit_miraca_uncertainty_study.sh pilot
ACCOUNT=${ACCOUNT:-}
if [[ -n "$ACCOUNT" ]]; then
    SB_ACCOUNT="#SBATCH --account=${ACCOUNT}"
else
    SB_ACCOUNT="# (no --account set; export ACCOUNT=... if your cluster needs one)"
fi

# Cluster modules loaded for the build AND inside every job, so the venv's
# osgeo/GDAL bindings find libgdal at runtime. Deliberately ONLY gdal - NOT the
# python module, whose _ctypes is broken here (missing libffi.so.6); uv supplies
# a self-contained Python instead (see do_setup). Override if your cluster names
# the module differently: MODULES="gdal/3.6.2" ...
MODULES=${MODULES:-"gdal/3.6.2"}
UV_PYTHON=${UV_PYTHON:-3.12}   # uv-managed standalone CPython for the venv

SOBOL_N=${SOBOL_N:-8192}   # fixed base N for every combination
LHS_N=${LHS_N:-3000}

# ── Assets ───────────────────────────────────────────────────────────────────
ASSETS=(airports education gas healthcare oil ports power rail roads telecom)
# HEAVY = the geometry-dense ones. Roads alone is 85-90% of every country's
# exposure bytes (DEU_roads 4.0 GB, FRA_roads 3.9 GB); rail and power follow.
HEAVY=(roads rail power)

is_heavy() {
    local a="$1"
    for h in "${HEAVY[@]}"; do [[ "$a" == "$h" ]] && return 0; done
    return 1
}

# ── Country tiers, by total exposure size ────────────────────────────────────
# Measured from ${EXPOSURE_DIR} (total MB across all assets per country):
#   XL 1.4-4.5 GB | L 0.4-0.8 GB | M 0.1-0.35 GB | S < 0.1 GB
XL=(DEU FRA ESP ITA POL GBR)
L=(AUT PRT GRC CZE SWE NLD ROU CHE HUN FIN BEL)
M=(SVK SRB NOR BGR HRV IRL SVN DNK LTU LVA)
# S = everything else with exposure data (ALB CYP EST MKD LUX ISL MLT LIE AND)

tier_of() {
    local c="$1"
    for x in "${XL[@]}"; do [[ "$c" == "$x" ]] && { echo XL; return; }; done
    for x in "${L[@]}";  do [[ "$c" == "$x" ]] && { echo L;  return; }; done
    for x in "${M[@]}";  do [[ "$c" == "$x" ]] && { echo M;  return; }; done
    echo S
}

# Stage 1: 1 CPU, flat --mem, and a walltime that scales with the GIS work.
prep_res() {   # tier heavy -> "mem time"
    local t="$1" heavy="$2"
    case "${t}_${heavy}" in
        XL_1) echo "64G 24:00:00" ;;  XL_0) echo "24G 08:00:00" ;;
        L_1)  echo "32G 12:00:00" ;;  L_0)  echo "12G 04:00:00" ;;
        M_1)  echo "16G 06:00:00" ;;  M_0)  echo "8G  02:00:00" ;;
        S_1)  echo "8G  03:00:00" ;;  S_0)  echo "6G  02:00:00" ;;
    esac
}

# Stage 2: cpus + --mem-per-cpu (one model-data copy per worker, see header).
# Big countries get FEWER workers with more RAM each, precisely because of that
# per-worker duplication - 16 workers x a multi-GB copy would not fit.
#
# Sized to fit ONE node of the ivm partition: node240-242/244 have 64 cores and
# ~123 GB, so a request must stay under that. XL heavy is 8 x 14G = 112 GB,
# which fits with headroom (8 x 16G = 128 GB would NOT schedule there). If a
# combination needs more, node243 (ivm-fat, 768 GB) and node001-002 (defq-fat,
# 1031 GB) are the escape hatches: PARTITION=ivm-fat or defq-fat.
#
# Walltimes on the HEAVY tiers (_1 = roads/rail/power) were raised generously
# after the first run: a handful of the biggest road/power combos (DEU/ESP/FRA/
# GBR/ITA roads, FRA/GBR/BGR/CHE/LTU power, LTU/LVA/MKD roads) hit the previous
# limits mid-run and were cancelled, so these are now sized well above the
# longest observed runtimes to finish in one submission. Because run_study.py
# checkpoints per scenario (completed LHS/Sobol archives are skipped on re-run),
# even a combo that still overruns resumes on the next submission - the bump
# just avoids needing one.
# CAVEAT: these exceed the 48h that was verified to schedule on the ivm nodes.
# Check the partition cap first with `scontrol show partition <p> | grep MaxTime`
# (or `sinfo -o "%P %l"`); if a request is rejected, lower the offending tier or
# route it to the big-memory escape hatches PARTITION=ivm-fat / defq-fat.
run_res() {    # tier heavy -> "cpus mem_per_cpu time"
    local t="$1" heavy="$2"
    case "${t}_${heavy}" in
        XL_1) echo " 8 14G 120:00:00" ;;  XL_0) echo "16  4G 12:00:00" ;;
        L_1)  echo "12  8G 96:00:00" ;;   L_0)  echo "16  3G 08:00:00" ;;
        M_1)  echo "16  4G 60:00:00" ;;   M_0)  echo "16  2G 06:00:00" ;;
        S_1)  echo "16  2G 30:00:00" ;;   S_0)  echo "16  2G 04:00:00" ;;
    esac
}

# ── Discover countries from the exposure files themselves ────────────────────
# "All countries we have exposure data for" is defined by what is on disk, and
# not every country has every asset (AND has no oil/ports/rail), so the job
# list is derived from the actual filenames rather than hard-coded.
# nullglob (not `ls`) so an asset with no exposure files yields nothing rather
# than a non-zero exit that `set -o pipefail` would turn into a script abort.
countries_for_asset() {
    local asset="$1"
    (
        shopt -s nullglob
        for f in "${EXPOSURE_DIR}"/*_"${asset}"_exposure.parquet; do
            local base=${f##*/}
            echo "${base%_${asset}_exposure.parquet}"
        done
    ) | sort -u
}

# Optional country filter, e.g. to stage the expensive combos on their own:
#   ONLY_COUNTRIES="DEU FRA" ./submit_miraca_uncertainty_study.sh submit roads
ONLY_COUNTRIES=${ONLY_COUNTRIES:-}
country_selected() {
    [[ -z "$ONLY_COUNTRIES" ]] && return 0
    local c
    for c in $ONLY_COUNTRIES; do [[ "$c" == "$1" ]] && return 0; done
    return 1
}

DRY=0

submit_combo() {
    local country="$1" asset="$2"
    local tier heavy label
    tier=$(tier_of "$country")
    if is_heavy "$asset"; then heavy=1; else heavy=0; fi
    label="${country}_${asset}"

    read -r p_mem p_time <<<"$(prep_res "$tier" "$heavy")"
    read -r r_cpus r_mem r_time <<<"$(run_res "$tier" "$heavy")"

    if [[ $DRY -eq 0 ]]; then mkdir -p "$LOG_DIR"; fi

    if [[ $DRY -eq 1 ]]; then
        printf "%-18s tier=%-2s prep[1cpu %s %s]  run[%scpu %s/cpu %s]\n" \
            "$label" "$tier" "$p_mem" "$p_time" "$r_cpus" "$r_mem" "$r_time"
        return
    fi

    # ---- Stage 1: preprocess (1 CPU, high memory) ----------------------------
    local jid_prep
    jid_prep=$(sbatch --parsable <<SLURM
#!/bin/bash -l
#SBATCH --job-name=prep_${label}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem=${p_mem}
#SBATCH --time=${p_time}
#SBATCH --output=${LOG_DIR}/out_prep_${label}
#SBATCH --error=${LOG_DIR}/err_prep_${label}
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
# Coastal hazard maps are streamed from the CoCLiCo STAC catalogue over HTTPS.
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
module load ${MODULES}   # libgdal on LD_LIBRARY_PATH for osgeo
cd ${REPO}
${PYTHON} -m src.preprocess --country ${country} --asset ${asset}
SLURM
)

    # ---- Stage 2: experiments (many CPUs), only if Stage 1 succeeded --------
    # run_study.py re-checks Stage 1 and skips it, then runs every applicable
    # default scenario. --no-aggregate is essential: with hundreds of these in
    # flight, every job would otherwise rewrite the same summary .xlsx.
    local jid_run
    jid_run=$(sbatch --parsable --dependency=afterok:${jid_prep} <<SLURM
#!/bin/bash -l
#SBATCH --job-name=run_${label}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${r_cpus}
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem-per-cpu=${r_mem}
#SBATCH --time=${r_time}
#SBATCH --output=${LOG_DIR}/out_run_${label}
#SBATCH --error=${LOG_DIR}/err_run_${label}
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
# Keep BLAS single-threaded: the parallelism is across EMA workers, and nested
# threading oversubscribes the allocation badly.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
module load ${MODULES}   # libgdal on LD_LIBRARY_PATH for osgeo (inherited by subprocesses)
cd ${REPO}
${PYTHON} run_study.py \
    --countries ${country} \
    --assets ${asset} \
    --workers ${r_cpus} \
    --sobol-n ${SOBOL_N} \
    --lhs-n ${LHS_N} \
    --python ${PYTHON} \
    --no-aggregate
SLURM
)
    LAST_RUN_ID="$jid_run"   # picked up by `all` to chain the aggregation
    echo "submitted ${label}: prep=${jid_prep} run=${jid_run} (tier ${tier})"
}

LAST_RUN_ID=""

# ── Reusable steps ───────────────────────────────────────────────────────────

do_setup() {
    # Build the venv with uv, using uv's OWN standalone Python - NOT the cluster
    # python module, whose _ctypes is broken here (missing libffi.so.6). GDAL is
    # the one thing that must come from the module: gdal-config + libgdal build
    # the osgeo bindings, which have no usable PyPI wheel on this cluster.
    command -v uv >/dev/null 2>&1 || {
        echo "uv not found on PATH. Install it (user-space) first:" >&2
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        echo "  source \$HOME/.local/bin/env" >&2
        exit 1
    }
    module load ${MODULES}
    command -v gdal-config >/dev/null 2>&1 || {
        echo "gdal-config not on PATH after 'module load ${MODULES}'." >&2
        echo "Check the module name with: module avail gdal" >&2
        exit 1
    }

    uv python install "${UV_PYTHON}"
    rm -rf "$VENV"
    uv venv "$VENV" --python "${UV_PYTHON}" --python-preference only-managed

    # osgeo/GDAL is built from source against the module's libgdal. Two fixes:
    #  - LIBRARY_PATH: ld reads it (not LD_LIBRARY_PATH); the module puts libgdal
    #    in lib64 but GDAL's setup.py links -L.../lib, so mirror LD_LIBRARY_PATH
    #    across or the link fails with 'cannot find -lgdal'.
    #  - pin gdal==$(gdal-config --version) so the bindings match the module.
    export LIBRARY_PATH="${LD_LIBRARY_PATH:-}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    uv pip install --python "$PYTHON" "gdal==$(gdal-config --version)"
    uv pip install --python "$PYTHON" -r "${REPO}/requirements.txt"

    echo
    echo "Environment ready: $PYTHON"
    "$PYTHON" -c "from osgeo import gdal; import damagescanner, geopandas, ema_workbench, pandas; print('imports OK', gdal.__version__)"
}

# Submit the aggregation. With an argument, waits for that colon-separated list
# of job IDs (afterany - a few failed combos must not block the summary).
submit_aggregate() {
    local dep="${1:-}"
    local depflag=()
    if [[ -n "$dep" ]]; then depflag=(--dependency="afterany:${dep}"); fi
    mkdir -p "$LOG_DIR"
    sbatch --parsable ${depflag[@]+"${depflag[@]}"} <<SLURM
#!/bin/bash -l
#SBATCH --job-name=miraca_aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/out_aggregate
#SBATCH --error=${LOG_DIR}/err_aggregate
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
module load ${MODULES}   # src imports pull in geopandas/osgeo -> need libgdal
cd ${REPO}
${PYTHON} -m src.aggregate_results
SLURM
}

# Chain a follow-up behind MANY jobs without one enormous dependency string:
# every BARRIER_CHUNK run jobs get a trivial no-op "barrier" job, and the
# caller then depends on just those few barriers instead of all ~350 IDs.
BARRIER_CHUNK=50
submit_barriers() {   # args: job ids -> prints colon-separated barrier ids
    local ids=("$@") barriers=() i chunk bid
    for ((i = 0; i < ${#ids[@]}; i += BARRIER_CHUNK)); do
        chunk=$(IFS=:; echo "${ids[*]:i:BARRIER_CHUNK}")
        bid=$(sbatch --parsable --dependency="afterany:${chunk}" <<SLURM
#!/bin/bash
#SBATCH --job-name=miraca_barrier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem=256M
#SBATCH --time=00:02:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
exit 0
SLURM
)
        barriers+=("$bid")
    done
    (IFS=:; echo "${barriers[*]}")
}

# ── Modes ────────────────────────────────────────────────────────────────────
MODE="${1:-dry}"
shift || true
SEL_ASSETS=("$@")
[[ ${#SEL_ASSETS[@]} -eq 0 ]] && SEL_ASSETS=("${ASSETS[@]}")

case "$MODE" in

setup)
    do_setup
    ;;

all)
    # ── THE one-command path: environment -> every combination -> summary ────
    # Everything is submitted up front and chained with SLURM dependencies, so
    # this returns immediately and the cluster runs the whole study unattended.
    mkdir -p "$LOG_DIR"
    if [[ ! -x "$PYTHON" ]]; then
        echo "== no venv at ${PYTHON} - running setup first =="
        do_setup
        echo
    fi

    echo "== submitting all combinations =="
    RUN_IDS=()
    total=0
    for A in "${SEL_ASSETS[@]}"; do
        for C in $(countries_for_asset "$A"); do
            if country_selected "$C"; then
                submit_combo "$C" "$A"
                RUN_IDS+=("$LAST_RUN_ID")
                total=$((total + 1))
            fi
        done
    done

    if [[ $total -eq 0 ]]; then
        echo "Nothing matched - check the asset names / ONLY_COUNTRIES filter." >&2
        exit 1
    fi

    echo
    echo "== chaining the summary workbook behind all ${total} run jobs =="
    BARRIER_IDS=$(submit_barriers "${RUN_IDS[@]}")
    AGG_ID=$(submit_aggregate "$BARRIER_IDS")

    echo
    echo "Submitted: ${total} combinations = $((total * 2)) prep/run jobs,"
    echo "           barriers ${BARRIER_IDS//:/ }, aggregate ${AGG_ID}."
    echo "Nothing else to do - the workbook appears at"
    echo "  ${REPO}/MIRACA_uncertainty_study_summary.xlsx"
    echo "Watch progress with: $0 status"
    ;;

resubmit)
    # ── Re-run ONLY the combos listed in a file, then chain the summary ───────
    # One "COUNTRY ASSET" per line (# comments and blanks ignored); defaults to
    # ${REPO}/failed_combos.txt. Each combo goes through the normal prep->run
    # chain, but nothing is wasted on a re-run: src.preprocess rewrites Stage-1
    # (curve fixes now let the oil/ports prep-failures through), and run_study.py
    # skips Stage-1 if already valid and resumes Stage-2 at the first scenario
    # whose LHS/Sobol archive is missing - so the timed-out combos pick up exactly
    # where SLURM cut them off. Use this after fixing curves.py / risk_model.py:
    #   ./submit_miraca_uncertainty_study.sh resubmit
    #   RESUBMIT_LIST=/path/to/list.txt ./submit_miraca_uncertainty_study.sh resubmit
    LIST=${RESUBMIT_LIST:-${REPO}/failed_combos.txt}
    if [[ ! -f "$LIST" ]]; then
        echo "No combo list at '$LIST' (set RESUBMIT_LIST=/path/to/list)." >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR"
    echo "== re-submitting combos from ${LIST} =="
    RUN_IDS=()
    total=0
    while read -r C A _; do
        [[ -z "${C:-}" || "$C" == \#* ]] && continue
        submit_combo "$C" "$A"
        RUN_IDS+=("$LAST_RUN_ID")
        total=$((total + 1))
    done < "$LIST"

    if [[ $total -eq 0 ]]; then
        echo "No combos read from ${LIST}." >&2
        exit 1
    fi

    echo
    echo "== chaining the summary workbook behind all ${total} run jobs =="
    BARRIER_IDS=$(submit_barriers "${RUN_IDS[@]}")
    AGG_ID=$(submit_aggregate "$BARRIER_IDS")
    echo
    echo "Re-submitted ${total} combos ($((total * 2)) prep/run jobs),"
    echo "             barriers ${BARRIER_IDS//:/ }, aggregate ${AGG_ID}."
    echo "Watch progress with: $0 status"
    ;;

recoastal)
    # ── Recompute ONLY the coastal scenario for already-finished combos ───────
    # After correcting the coastal return periods to what CoCLiCo actually serves
    # ([1,100,1000]), the previously-completed coastal EAD must be recomputed so
    # it is comparable with the freshly-run combos. This re-runs Stage 2's
    # coastal_absprot_ds scenario with --force-scenarios (LHS/Sobol/analysis
    # only); Stage 1 is left intact - no re-streaming, no river/EQ/wind rework.
    # Combos come from $RECOASTAL_LIST (default coastal_rerun.txt); it must list
    # only combos whose Stage 1 is already on disk (the finished coastal ones -
    # the failed combos get their coastal from `resubmit` instead).
    #
    # Coastal-only is light (minutes to a couple of hours), so walltime is capped
    # low here rather than taken from the heavy run tiers - avoids the big-tier
    # MaxTime question entirely. No aggregate is chained: run `$0 aggregate` once
    # BOTH this and `resubmit` have drained.
    LIST=${RECOASTAL_LIST:-${REPO}/coastal_rerun.txt}
    if [[ ! -f "$LIST" ]]; then
        echo "No combo list at '$LIST' (set RECOASTAL_LIST=/path/to/list)." >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR"
    echo "== recomputing coastal_absprot_ds from ${LIST} =="
    total=0
    while read -r C A _; do
        [[ -z "${C:-}" || "$C" == \#* ]] && continue
        tier=$(tier_of "$C")
        if is_heavy "$A"; then heavy=1; else heavy=0; fi
        read -r r_cpus r_mem _ <<<"$(run_res "$tier" "$heavy")"
        if [[ $heavy -eq 1 ]]; then c_time="08:00:00"; else c_time="04:00:00"; fi
        label="${C}_${A}"
        jid=$(sbatch --parsable <<SLURM
#!/bin/bash -l
#SBATCH --job-name=coast_${label}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${r_cpus}
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem-per-cpu=${r_mem}
#SBATCH --time=${c_time}
#SBATCH --output=${LOG_DIR}/out_coast_${label}
#SBATCH --error=${LOG_DIR}/err_coast_${label}
set -euo pipefail
export MIRACA_CONFIG=${MIRACA_CONFIG}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
module load ${MODULES}
cd ${REPO}
${PYTHON} run_study.py \
    --countries ${C} \
    --assets ${A} \
    --scenarios coastal_absprot_ds \
    --force-scenarios \
    --workers ${r_cpus} \
    --sobol-n ${SOBOL_N} \
    --lhs-n ${LHS_N} \
    --python ${PYTHON} \
    --no-aggregate
SLURM
)
        echo "submitted coastal ${label}: run=${jid} (tier ${tier})"
        total=$((total + 1))
    done < "$LIST"

    if [[ $total -eq 0 ]]; then
        echo "No combos read from ${LIST}." >&2
        exit 1
    fi
    echo
    echo "Re-ran coastal for ${total} combos."
    echo "When BOTH this and 'resubmit' have drained, build the workbook with:"
    echo "  $0 aggregate"
    ;;

dry)
    DRY=1
    echo "Submission plan (nothing submitted):"
    echo
    total=0
    for A in "${SEL_ASSETS[@]}"; do
        for C in $(countries_for_asset "$A"); do
            if country_selected "$C"; then
                submit_combo "$C" "$A"
                total=$((total + 1))
            fi
        done
    done
    echo
    echo "${total} combinations -> $((total * 2)) SLURM jobs."
    ;;

pilot)
    # Validate the whole chain on two cheap combos before committing the farm:
    # LUX is landlocked (river/EQ/wind only), DNK is coastal, so between them
    # they exercise every hazard path including the STAC streaming.
    echo "Pilot: LUX/roads (landlocked) + DNK/power (coastal + windstorm)"
    submit_combo LUX roads
    submit_combo DNK power
    echo
    echo "Check both finish clean, then run: $0 submit"
    ;;

submit)
    total=0
    for A in "${SEL_ASSETS[@]}"; do
        for C in $(countries_for_asset "$A"); do
            if country_selected "$C"; then
                submit_combo "$C" "$A"
                total=$((total + 1))
            fi
        done
    done
    echo
    echo "Submitted ${total} combinations ($((total * 2)) jobs)."
    echo "When the queue drains: $0 aggregate"
    ;;

aggregate)
    # Standalone: run it yourself once the queue has drained (`all` chains this
    # automatically, so you only need this after a partial or resumed run).
    echo "aggregate job: $(submit_aggregate)"
    ;;

status)
    squeue -u "$USER" -o "%.10i %.28j %.9P %.8T %.10M %.6D %R" | head -50
    echo
    echo "queued/running: $(squeue -u "$USER" -h | wc -l)"
    ;;

*)
    echo "Unknown mode '$MODE'." >&2
    echo "Use: all | setup | dry | pilot | submit | resubmit | recoastal | aggregate | status" >&2
    exit 1
    ;;
esac
