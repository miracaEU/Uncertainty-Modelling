#!/bin/bash -l
# Step 1: give Malta and Cyprus the 25-year coastal minimum (coastal.MINIMUM_STANDARD) in
# their Stage 1 segments. Backs up every segments and meta file first. From the repo root:
#   sbatch reruns/20261117_coastal_mt_cy/update_coastal.sh
#SBATCH --job-name=coastal_mtcy_update
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002,node009
#SBATCH --output=reruns/20261117_coastal_mt_cy/update_%j.out
#SBATCH --error=reruns/20261117_coastal_mt_cy/update_%j.err
set -euo pipefail
PYTHON=${PYTHON:-/scistor/ivm/${USER:-$(id -un)}/.venvs/miraca_uq/bin/python}
export MIRACA_CONFIG=${MIRACA_CONFIG:-$PWD/config.cluster.yml}
module load gdal/3.6.2
"$PYTHON" -W ignore -m src.update_coastal_protection --out reruns/20261117_coastal_mt_cy \
    --countries MLT CYP --apply --backup-dir data/intermediate_coastal_backup_20261117
echo "Done: see reruns/20261117_coastal_mt_cy/coastal_protection_applied.csv"
