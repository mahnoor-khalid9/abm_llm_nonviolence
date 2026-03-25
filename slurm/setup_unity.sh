#!/bin/bash
# =============================================================================
# setup_unity.sh — One-time environment setup for Unity cluster
#
# Run ONCE interactively on a login node (NOT as a SLURM job):
#   bash setup_unity.sh
#
# After this completes, submit jobs with:
#   sbatch run_unity.slurm
# =============================================================================

set -e

echo "=========================================="
echo "  ABM-LLM Environment Setup — Unity"
echo "=========================================="

# ── Paths ─────────────────────────────────────────────────────────────────────
# Home dir: /home/fmahnoor1_umassd_edu      (93GB quota — keep lean)
# Work dir: /work/pi_lfiondella_umassd_edu/Mahnoor  (1TB quota — models go here)

HOME_DIR="/home/fmahnoor1_umassd_edu"
WORK_DIR="/work/pi_lfiondella_umassd_edu/Mahnoor"
VENV_PATH="${HOME_DIR}/venv_abm_llm"
MODEL_CACHE="${WORK_DIR}/huggingface_cache"
MODEL_DIR="${WORK_DIR}/models"
OFFLOAD_DIR="${WORK_DIR}/offload"

# HuggingFace tokens (already set in your SLURM script)
HF_TOKEN="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"

# ── Load modules ──────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Loading modules..."
module purge
module load python/3.11.7
module load cuda/12.1
echo "  Python: $(python --version)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'loaded')"

# ── Create work directories ───────────────────────────────────────────────────
echo ""
echo "[2/6] Creating directories..."
mkdir -p "$WORK_DIR"
mkdir -p "$MODEL_CACHE"
mkdir -p "$MODEL_DIR"
mkdir -p "$OFFLOAD_DIR"
mkdir -p "${HOME_DIR}/logs"
echo "  Work dir:     $WORK_DIR"
echo "  Model cache:  $MODEL_CACHE"
echo "  Venv:         $VENV_PATH"

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "[3/6] Creating virtual environment..."
if [ ! -d "${VENV_PATH}" ]; then
    python -m venv "${VENV_PATH}"
    echo "  Created new venv."
else
    echo "  Venv already exists, updating packages."
fi
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ── Install packages ──────────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing packages..."

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# Simulation core
pip install \
    "mesa==2.4.0" \
    numpy pandas scipy dill matplotlib \
    pydantic jinja2 \
    pyyaml python-dotenv requests \
    --quiet

# HuggingFace + vLLM
pip install \
    "transformers>=4.43.0" \
    accelerate \
    bitsandbytes \
    safetensors \
    sentencepiece \
    huggingface-hub \
    --quiet

# vLLM — primary inference backend (fastest on cluster)
pip install vllm --quiet

# API providers (fallback / local runs)
pip install openai google-generativeai --quiet

# Flash Attention 2 — significant speedup on A100
# Runs in background, non-critical if it fails
echo "  Installing flash-attn (may take 5-10 min)..."
pip install flash-attn --no-build-isolation --quiet || \
    echo "  WARNING: flash-attn install failed — will run without it (still OK)"

echo "  All packages installed."

# ── Set persistent environment variables ──────────────────────────────────────
echo ""
echo "[5/6] Setting environment variables..."

# Add to .bashrc so they persist across sessions
cat >> "${HOME_DIR}/.bashrc" << EOF

# ── ABM-LLM Simulation Environment ──────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export HF_DATASETS_CACHE="${MODEL_CACHE}/datasets"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF

# Apply now
export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"

echo "  Environment variables written to ~/.bashrc"

# ── Download Llama 3.1 8B AWQ (pre-quantized — fastest on cluster) ────────────
echo ""
echo "[6/6] Downloading Llama 3.1 8B AWQ quantized weights..."
echo "  Downloading to: ${MODEL_DIR}/llama-3.1-8b-awq"
echo "  Size: ~5GB (AWQ 4-bit quantized)"
echo "  This may take 5-15 minutes..."

python - << PYEOF
import os
os.environ["HF_TOKEN"] = "${HF_TOKEN}"
os.environ["HF_HOME"] = "${MODEL_CACHE}"

from huggingface_hub import snapshot_download, login

login(token="${HF_TOKEN}", add_to_git_credential=False)

# Download AWQ pre-quantized 8B — best speed/quality for cluster
# This is the recommended model for vLLM on A100
snapshot_download(
    repo_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    local_dir="${MODEL_DIR}/llama-3.1-8b-awq",
    ignore_patterns=["*.pth", "*.bin"],   # prefer safetensors
)
print("  Model downloaded successfully.")
PYEOF

# ── Verify installation ───────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python - << PYEOF
import torch
import vllm
import transformers
import mesa

print(f"  torch:        {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  vllm:         {vllm.__version__}")
print(f"  transformers: {transformers.__version__}")
print(f"  mesa:         {mesa.__version__}")
PYEOF

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Model path: ${MODEL_DIR}/llama-3.1-8b-awq"
echo "  Venv path:  ${VENV_PATH}"
echo ""
echo "  Submit a job:"
echo "    sbatch run_unity.slurm"
echo "=========================================="
