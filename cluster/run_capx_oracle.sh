#!/bin/bash
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a30,a10,a100-40gb,a100-80gb
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o capx_oracle_%j.log
#
# No LLM involved. Runs reference solutions through CaP-X's launcher:
#   1. a stock Robosuite task, to check the CaP-X install itself;
#   2. the YAM tube-rack task, to check our wrapper.
#   sbatch cluster/run_capx_oracle.sh
set -uo pipefail

C=$RCAC_SCRATCH/capx
PROJECT=${PROJECT:-$RCAC_SCRATCH/biomedical-tasks}
export MUJOCO_GL=egl
cd "$C/cap-x"

echo "=== 1/2 stock CaP-X task (Franka cube lift, privileged) ==="
.venv/bin/python capx/envs/launch.py \
    --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml \
    --use-oracle-code True --total-trials 3 --num-workers 1 \
    --output-dir "$C/outputs/stock_cube_lift_oracle"
echo "stock task exit code: $?"

echo "=== 2/2 YAM tube rack (privileged) ==="
.venv/bin/python capx/envs/launch.py \
    --config-path "$PROJECT/env_configs/tube_rack/yam_tube_rack_privileged.yaml" \
    --use-oracle-code True --total-trials 10 --num-workers 2 \
    --output-dir "$C/outputs/yam_tube_rack_oracle"
echo "tube rack exit code: $?"
