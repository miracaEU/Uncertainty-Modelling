#!/bin/bash -l
# Switch the roads, rail and telecom segments to the 5 km block + fill river standards.
# Backs up every segments and meta file to data/intermediate_pixel_backup_20261111/
# first (never overwriting a backup), then replaces prot_rp and adds prot_source and
# prot_fill_km. Submit from the repo root:
#   sbatch reruns/20261111_river_protection/update_segments.sh
#SBATCH --job-name=river_prot_update
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --exclude=node002,node009
#SBATCH --output=reruns/20261111_river_protection/update_%j.out
#SBATCH --error=reruns/20261111_river_protection/update_%j.err
set -euo pipefail
PYTHON=${PYTHON:-/scistor/ivm/${USER:-$(id -un)}/.venvs/miraca_uq/bin/python}
export MIRACA_CONFIG=${MIRACA_CONFIG:-$PWD/config.cluster.yml}
module load gdal/3.6.2
"$PYTHON" -W ignore -m src.update_river_protection --out reruns/20261111_river_protection \
    --assets roads rail telecom --apply --backup-dir data/intermediate_pixel_backup_20261111 --workers 8
"$PYTHON" - <<'EOF'
import pandas as pd
d = pd.read_csv("reruns/20261111_river_protection/river_protection_applied.csv")
print(d.status.value_counts().to_string())
bad = d[~d.status.isin(["applied", "already applied"])]
if len(bad):
    print(bad[["country", "asset", "status", "error"]].to_string(index=False))
    raise SystemExit(f"{len(bad)} combination(s) not updated")
print("All", len(d), "combinations updated.")
EOF
