#!/bin/bash
# =============================================================================
# install_flash_attention.sh
#
# Run this separately if setup_unity.sh skipped flash-attn installation.
# Flash Attention 2 gives ~30-50% speedup on A100 for attention operations.
# Non-critical — simulation works fine without it.
#
# Run on an interactive GPU node (NOT a login node):
#   srun --partition=gpu-preempt --gres=gpu:a100:1 --constraint=a100-80g \
#        --mem=16G --time=01:00:00 --pty bash
#   bash install_flash_attention.sh
# =============================================================================

echo "Installing Flash Attention 2..."

# Load correct modules
module purge
module load python/3.11.7
module load cuda/12.1

# Activate venv (NOT conda)
VENV_PATH="/home/fmahnoor1_umassd_edu/venv_gandhian_abm"
source "${VENV_PATH}/bin/activate"

echo "Python: $(which python)"
echo "CUDA: $(nvcc --version | grep release)"

# Verify GPU is available (flash-attn needs GPU at build time)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# Install flash-attn
# --no-build-isolation: uses the already-installed torch version
# This takes 5-10 minutes — it compiles C++ extensions
echo ""
echo "Building flash-attn (5-10 minutes)..."
pip install flash-attn --no-build-isolation

# Verify
echo ""
echo "Verifying installation..."
python -c "
import flash_attn
print(f'Flash Attention version: {flash_attn.__version__}')
print('Installation successful.')
"

echo ""
echo "Done. Flash Attention 2 is now available."
