#!/bin/bash
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a30,a10,a100-40gb,a100-80gb
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o capx_llm_%j.log
#
# An LLM writes the policy. Needs an API key in a file (one line, chmod 600); the default is
# an OpenRouter key in $RCAC_SCRATCH/capx/cap-x/.openrouterkey. This spends API credit.
#   sbatch cluster/run_capx_llm.sh
#   MODEL=<model id> TRIALS=20 CONFIG=yam_tube_rack_privileged_multiturn sbatch cluster/run_capx_llm.sh
# Any OpenAI-compatible provider works: point BASE_URL at its /v1/ root and KEY_FILE at its key.
#   BASE_URL=https://example.com/v1/ KEY_FILE=$RCAC_SCRATCH/capx/.otherkey MODEL=<model id> sbatch cluster/run_capx_llm.sh
set -uo pipefail

C=$RCAC_SCRATCH/capx
PROJECT=${PROJECT:-$RCAC_SCRATCH/biomedical-tasks}
MODEL=${MODEL:-google/gemini-3.1-pro-preview}
TRIALS=${TRIALS:-5}
CONFIG=${CONFIG:-yam_tube_rack_privileged}
BASE_URL=${BASE_URL:-https://openrouter.ai/api/v1/}
KEY_FILE=${KEY_FILE:-.openrouterkey}
PORT=8110
export MUJOCO_GL=egl
cd "$C/cap-x"

[ -s "$KEY_FILE" ] || { echo "missing key file $KEY_FILE"; exit 1; }

.venv/bin/python capx/serving/openrouter_server.py --key-file "$KEY_FILE" --base-url "$BASE_URL" --port $PORT &
PROXY=$!
trap "kill $PROXY 2>/dev/null" EXIT
for _ in $(seq 60); do
    (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
    sleep 2
done

OUT="$C/outputs/${CONFIG}_$(echo "$MODEL" | tr '/:' '__')_$SLURM_JOB_ID"
.venv/bin/python capx/envs/launch.py \
    --config-path "$PROJECT/env_configs/tube_rack/$CONFIG.yaml" \
    --model "$MODEL" --server-url "http://127.0.0.1:$PORT/chat/completions" \
    --total-trials "$TRIALS" --num-workers 2 --output-dir "$OUT"
echo "exit code: $?  outputs: $OUT"
