#!/bin/bash
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a10,a30
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -t 00:45:00
#SBATCH -o evolve_%j.log
#
# Library loop: practice rounds with a persistent skill library, then a held-out test.
# One model call per round. Re-submitting with the same RUN resumes it.
#
#   RUN=luna_loop     MODEL=gpt-6-luna sbatch cluster/run_evolve.sh
#   RUN=luna_restarts MODEL=gpt-6-luna EXTRA="--restarts --no-early-stop" sbatch cluster/run_evolve.sh
#   RUN=luna_task2    MODEL=gpt-6-luna EXTRA="--inherit $RCAC_SCRATCH/biolab_runs/luna_loop/library" sbatch cluster/run_evolve.sh
#   RUN=reference     EXTRA="--reference" sbatch cluster/run_evolve.sh      # no model, no cost
#   RUN=joined TASK=weigh_salt EXTRA="--compose $RCAC_SCRATCH/biolab_runs/a/library $RCAC_SCRATCH/biolab_runs/b/library" \
#       sbatch -n 12 --mem=10G cluster/run_evolve.sh    # no model: two learned scripts run in order
#   RUN=luna_cap      MODEL=gpt-6-luna TASK=snap_cap sbatch cluster/run_evolve.sh   # the harder task
#   RUN=luna_cap12    MODEL=gpt-6-luna TASK=snap_cap EXTRA="--practice-seeds 100-111 --video" sbatch cluster/run_evolve.sh
#   RUN=luna_weigh    MODEL=gpt-6-luna TASK=weigh_salt EXTRA="--practice-seeds 100-111 --video --no-early-stop" \
#       sbatch -n 12 --mem=12G cluster/run_evolve.sh    # two-arm task: 70 s a layout, so one core per layout
# Keep -N 1: with cores spread over two nodes the script sees only the first node's share and crawls.
#
# Measured with 12 layouts and --video: under one core, 0.9 GB, 13 minutes for restarts and 33 for a
# library loop (most of it waiting for the model). A job that runs out of time resumes when re-submitted.
set -uo pipefail

RUN=${RUN:?set RUN to a name for this run}
MODEL=${MODEL:-}
TASK=${TASK:-tube_rack}
ROUNDS=${ROUNDS:-8}
BASE_URL=${BASE_URL:-https://api.experientiallabs.ai/v1/}
KEY_FILE=${KEY_FILE:-$RCAC_SCRATCH/capx/.expkey}
EXTRA=${EXTRA:-}

# Reuses CaP-X's environment (it has MuJoCo and this project installed). EGL is only used by --video.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1     # one core per layout, no oversubscription
PY=$RCAC_SCRATCH/capx/cap-x/.venv/bin/python
OUT=$RCAC_SCRATCH/biolab_runs/$RUN

# shellcheck disable=SC2086
$PY -u -m biolab.evolve --run-dir "$OUT" --task "$TASK" --model "$MODEL" --base-url "$BASE_URL" --key-file "$KEY_FILE" \
    --rounds "$ROUNDS" --workers "${SLURM_NTASKS:-1}" $EXTRA
code=$?
echo "exit code: $code  run folder: $OUT"
exit $code      # so that a failed run shows as FAILED and jobs waiting on it with afterok do not start
