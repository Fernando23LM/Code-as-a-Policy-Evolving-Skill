#!/bin/bash
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a30,a10,a100-40gb,a100-80gb
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o capx_install_%j.log
#
# Installs CaP-X and this project into /scratch/gilbreth/$USER/capx.
# Safe to re-run: finished steps are skipped.
#   sbatch cluster/install_capx.sh
set -euo pipefail

C=$RCAC_SCRATCH/capx
PROJECT=${PROJECT:-$RCAC_SCRATCH/biomedical-tasks}

# Keep uv, its Python builds and its cache out of the 25 GB home.
export UV_INSTALL_DIR=$C/uv_bin UV_CACHE_DIR=$C/uv_cache UV_PYTHON_INSTALL_DIR=$C/uv_python UV_NO_MODIFY_PATH=1
export PATH=$C/uv_bin:$PATH
mkdir -p "$C" && cd "$C"

[ -x uv_bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d cap-x ] || git clone --recurse-submodules https://github.com/capgym/cap-x
cd cap-x
git submodule update --init --recursive

uv python install 3.10
[ -d .venv ] || uv venv -p 3.10
# --frozen installs exactly what CaP-X's uv.lock lists. Without it, uv re-resolves every
# extra for every Python version and fails on flash-attn (a Python 3.12-only wheel we do not use).
uv sync --frozen --extra robosuite

# `uv sync` removes packages it does not know, so this must come after it,
# and later commands must use .venv/bin/python (or `uv run --no-sync`).
# --no-deps keeps the MuJoCo version CaP-X locked.
uv pip install --no-deps -e "$PROJECT"

.venv/bin/python - <<'PY'
import mujoco, numpy, torch
import capx.envs, capx.integrations          # CaP-X's own registrations
import biolab.capx_env                       # ours
from capx.envs.base import list_envs
print("mujoco", mujoco.__version__, "| numpy", numpy.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
assert "yam_tube_rack_low_level" in list_envs()
print("INSTALL OK")
PY
