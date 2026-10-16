#!/bin/bash
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a10,a30
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem=4G
#SBATCH -t 00:40:00
#SBATCH -o record_%j.log
#
# Film the reference policies (no model, no cost); videos and stills land in previews/.
#   TASKS="protein_assay cell_culture" SEEDS=0-2 sbatch cluster/record_reference.sh
set -uo pipefail
RCAC_SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}
TASKS=${TASKS:-protein_assay cell_culture}
SEEDS=${SEEDS:-0-2}
cd "$RCAC_SCRATCH/biomedical-tasks"
export MUJOCO_GL=egl OMP_NUM_THREADS=1
"$RCAC_SCRATCH/capx/cap-x/.venv/bin/python" scripts/record_reference.py $TASKS --seeds "$SEEDS" --workers "${SLURM_NTASKS:-3}"
status=$?
echo "exit code: $status"
exit $status
