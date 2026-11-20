#!/bin/bash -l
# ---------------------------------------------------------------------------
# Build the SECOND venv, for the polygon asset types: identical to the current
# one in every package, except DamageScanner at the polygon fix (develop commit
# f6599e00, issues #82 and #83). See src/ds_version.py for which asset type runs
# on which.
#
#   ./setup_dsfix_venv.sh           # build it (refuses if it already exists)
#   DRY=1 ./setup_dsfix_venv.sh     # show what would be installed, build nothing
#
# Unlike `submit_miraca_uncertainty_study.sh setup`, this never deletes anything:
# it stops if the target exists, and it refuses outright if the target is the
# 1.0.1 venv. To rebuild, remove the target yourself first.
#
# Then submit polygon-type jobs with VENV pointing at it:
#   VENV=/scistor/ivm/$USER/.venvs/miraca_uq_dsfix ./submit_miraca_uncertainty_study.sh ...
# Stage 1 and src.validate refuse a polygon type on 1.0.1, and a line type on
# the fix, so a forgotten VENV= fails fast instead of mixing versions.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$SCRIPT_DIR}
USER_DIR=${USER_DIR:-/scistor/ivm/${USER:-$(id -un)}}
OLD_VENV=${OLD_VENV:-${USER_DIR}/.venvs/miraca_uq}
NEW_VENV=${NEW_VENV:-${USER_DIR}/.venvs/miraca_uq_dsfix}
MODULES=${MODULES:-"gdal/3.6.2"}
DS_COMMIT=f6599e00d732c0b41858f7c12c07f4039ed5de80
# git+ records the commit in direct_url.json, which src/ds_version.py checks. If
# the nodes have no git, the archive URL works too (the hash is in its path):
#   DS_SOURCE="damagescanner @ https://github.com/VU-IVM/DamageScanner/archive/${DS_COMMIT}.tar.gz"
DS_SOURCE=${DS_SOURCE:-"damagescanner @ git+https://github.com/VU-IVM/DamageScanner@${DS_COMMIT}"}

OLD_PY=${OLD_VENV}/bin/python
NEW_PY=${NEW_VENV}/bin/python

die() { echo "!! $*" >&2; exit 1; }

[[ "$(readlink -f "$NEW_VENV" 2>/dev/null || echo "$NEW_VENV")" != "$(readlink -f "$OLD_VENV")" ]] \
    || die "NEW_VENV is the 1.0.1 venv ($OLD_VENV). Nothing done."
[[ -x "$OLD_PY" ]] || die "No 1.0.1 venv at $OLD_VENV to copy the package versions from."
[[ -e "$NEW_VENV" ]] && die "$NEW_VENV already exists. Remove it yourself to rebuild; nothing done."
command -v uv >/dev/null 2>&1 || die "uv not on PATH (see submit_miraca_uncertainty_study.sh setup)."

module load ${MODULES}
command -v gdal-config >/dev/null 2>&1 || die "gdal-config not on PATH after 'module load ${MODULES}'."
cd "$REPO"

# The 1.0.1 venv must really be 1.0.1, or the copy below would copy the wrong thing.
"$OLD_PY" -m src.ds_version --expect-old >/dev/null || die "$OLD_VENV does not hold DamageScanner 1.0.1."

# 1. Package versions of the 1.0.1 venv. Left out: damagescanner (installed at
#    the fix below), gdal (built against the module, as in setup) and any local
#    or editable install of this repo (jobs run it from the checkout).
STAMP=$(date +%Y%m%d_%H%M%S)
FREEZE_DIR=${FREEZE_DIR:-${USER_DIR}/.venvs/freeze}
mkdir -p "$FREEZE_DIR"
FULL=${FREEZE_DIR}/miraca_uq_${STAMP}.txt
PINS=${FREEZE_DIR}/miraca_uq_dsfix_pins_${STAMP}.txt
uv pip freeze --python "$OLD_PY" > "$FULL"
grep -viE '^(damagescanner|gdal)==|^-e |@ file:' "$FULL" > "$PINS" || true
GDAL_OLD=$(grep -iE '^gdal==' "$FULL" | cut -d= -f3 || true)
GDAL_MOD=$(gdal-config --version)
[[ -z "$GDAL_OLD" || "$GDAL_OLD" == "$GDAL_MOD" ]] \
    || die "the 1.0.1 venv has gdal $GDAL_OLD but module ${MODULES} gives $GDAL_MOD; load the matching module."
echo "1.0.1 venv: $(wc -l < "$FULL") packages, $(wc -l < "$PINS") pinned for the copy (lists in $FREEZE_DIR)"
grep -iE '^-e |@ file:' "$FULL" | sed 's/^/  not copied (local install): /' || true

PY_VER=$("$OLD_PY" -c 'import platform; print(platform.python_version())')
if [[ -n "${DRY:-}" ]]; then
    echo "DRY: would create $NEW_VENV with Python $PY_VER, gdal==$GDAL_MOD, the $(wc -l < "$PINS") pins,"
    echo "     and --no-deps \"$DS_SOURCE\""
    exit 0
fi

# 2. Same interpreter version, same gdal build recipe as `setup`.
uv python install "$PY_VER"
uv venv "$NEW_VENV" --python "$PY_VER" --python-preference only-managed
export LIBRARY_PATH="${LD_LIBRARY_PATH:-}${LIBRARY_PATH:+:$LIBRARY_PATH}"
uv pip install --python "$NEW_PY" "gdal==${GDAL_MOD}"
uv pip install --python "$NEW_PY" -r "$PINS"

# 3. DamageScanner at the fix. --no-deps: every dependency it needs is already
#    pinned above at the 1.0.1 venv's versions, which meet the fix's minimums.
uv pip install --python "$NEW_PY" --no-deps "$DS_SOURCE"

# 4. Verify: the pinned fix, and nothing else different.
"$NEW_PY" -m src.ds_version --expect-fix || die "the new venv does not hold the pinned polygon fix."
SKIP='^damagescanner|^-e |@ file:'
if ! diff <(grep -viE "$SKIP" "$FULL") \
          <(uv pip freeze --python "$NEW_PY" | grep -viE "$SKIP"); then
    die "package versions differ from the 1.0.1 venv (diff above, '<' old, '>' new)."
fi
"$NEW_PY" -c "from osgeo import gdal; import damagescanner, geopandas, ema_workbench, pandas; print('imports OK', gdal.__version__)"
echo
echo "Ready: $NEW_PY"
echo "Same packages as $OLD_VENV except damagescanner @ ${DS_COMMIT:0:8}."
