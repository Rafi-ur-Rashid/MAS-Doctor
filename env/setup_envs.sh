#!/usr/bin/env bash
# Rebuild both conda environments exactly (C01, EXPERIMENT_PLAN.md §9.3).
#   mastrust : this runtime + monitor
#   xgguard  : XG-Guard's own pinned stack (kept separate: its pins conflict with AgentDojo's)
# Both envs set PYTHONNOUSERSITE=1 and PYTHONPATH= on activation, because this machine
# exports PYTHONPATH=/scratch/mur5028/python/site-packages and has packages in ~/.local,
# either of which silently shadows env packages. Always run through `conda run -n <env>`
# or `conda activate <env>`, never through the env's python binary directly.
set -euo pipefail
CONDA=${CONDA:-/scratch/mur5028/miniconda3/bin/conda}
HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/mur5028/cache/pip} PYTHONNOUSERSITE=1 PYTHONPATH=

"$CONDA" create -y -q -n mastrust python=3.12
# TRANSFORMERS_CACHE: this machine points it at a shared /scratch/hf_cache whose lock
# files are not writable by this account; use the private cache under HF_HOME instead.
# PYTHONHASHSEED=0 (C06): AgentDojo orders event participants through a set, so string
# hashing must be the same in every process or a replayed run sees different tool results.
"$CONDA" env config vars set -n mastrust PYTHONNOUSERSITE=1 PYTHONPATH= \
    TRANSFORMERS_CACHE=/scratch/mur5028/cache/hub PYTHONHASHSEED=0
"$CONDA" run -n mastrust python -m pip install -r "$HERE/mastrust.lock.txt"

"$CONDA" create -y -q -n xgguard python=3.11
"$CONDA" env config vars set -n xgguard PYTHONNOUSERSITE=1 PYTHONPATH= TRANSFORMERS_CACHE=/scratch/mur5028/cache/hub
"$CONDA" run -n xgguard python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
"$CONDA" run -n xgguard python -m pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
"$CONDA" run -n xgguard python -m pip install -r "$HERE/xgguard.lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu124 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
