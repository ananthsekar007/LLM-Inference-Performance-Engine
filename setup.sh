#!/bin/bash
# Environment setup for the CSE 598 LLM serving assignment on ASU Sol.
#
# Run on a GPU node, and SOURCE it so the activated env stays in your shell:
#     source setup.sh            # load modules, create/activate env, set caches
#     source setup.sh --env-info # also write results/env.txt

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with 'source setup.sh', not 'bash setup.sh'."
    echo "Otherwise the activated environment disappears when the script exits."
    exit 1
fi

ENV_NAME=llmserve
PY_VERSION=3.11
SCRATCH_DIR=/scratch/$USER

module load mamba/latest || { echo "Failed to load mamba module."; return 1; }

export HF_HOME=$SCRATCH_DIR/hf_cache
export PIP_CACHE_DIR=$SCRATCH_DIR/pip_cache
export CONDA_PKGS_DIRS=$SCRATCH_DIR/conda_pkgs
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating env '$ENV_NAME' with Python $PY_VERSION..."
    mamba create -n "$ENV_NAME" "python=$PY_VERSION" -y || return 1
fi

source activate "$ENV_NAME" || { echo "Failed to activate $ENV_NAME."; return 1; }

ACTUAL_PY=$(python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
if [[ "$ACTUAL_PY" != "$PY_VERSION" ]]; then
    echo "WARNING: env '$ENV_NAME' has Python $ACTUAL_PY, expected $PY_VERSION."
    echo "To rebuild it: conda deactivate && mamba env remove -n $ENV_NAME -y && source setup.sh"
fi

# vLLM pins its own torch/transformers, so install it before the other packages.
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm || return 1
fi

if ! python -c "import accelerate, pandas, numpy, matplotlib" 2>/dev/null; then
    echo "Installing remaining dependencies..."
    pip install accelerate pandas numpy matplotlib || return 1
fi

mkdir -p results plots

python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')" \
    >/dev/null 2>&1 || echo "WARNING: could not pre-download gpt2."

python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"GPU ready: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. Are you on a GPU node?")
EOF

if [[ "$1" == "--env-info" ]]; then
    nvidia-smi > results/env.txt
    python -c "import torch, transformers, vllm, platform; print('python', platform.python_version()); print('torch', torch.__version__); print('cuda (torch build)', torch.version.cuda); print('cudnn', torch.backends.cudnn.version()); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('gpu', torch.cuda.get_device_name(0)); print('gpu mem GB', torch.cuda.get_device_properties(0).total_memory/1e9)" >> results/env.txt
    echo "Wrote results/env.txt"
fi

echo "Environment '$ENV_NAME' active (Python $ACTUAL_PY). HF_HOME=$HF_HOME"
