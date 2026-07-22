#!/bin/bash
# =============================================================================
# setup_unity.sh — Robust environment setup for Unity cluster
#
# Usage:
#   For public models (Qwen, Llama, etc.):
#     bash setup_unity.sh
#
#   For gated models (Mistral, etc.):
#     export HF_TOKEN="hf_your_token_here"
#     bash setup_unity.sh
# =============================================================================

set -e

echo "=========================================="
echo "  ABM-LLM Environment Setup — Unity"
echo "=========================================="

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME_DIR="/home/fmahnoor1_umassd_edu"
WORK_DIR="/work/pi_lfiondella_umassd_edu/Mahnoor"
VENV_PATH="${HOME_DIR}/venv_abm_llm"
MODEL_CACHE="${WORK_DIR}/huggingface_cache"
MODEL_DIR="${WORK_DIR}/models"

export HF_MODEL_DIR="mistralai/Ministral-3-8B-Reasoning-2512"
export MODEL_NAME="Ministral-3-8B-Reasoning-2512"
# export QUANT="Q4_K_M"  # ← Only needed for GGUF repos

# ── Load modules ──────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Loading modules..."
module purge
module load python/3.11.7
module load cuda/12.1

# Force NVCC to accept GCC 13 for FlashInfer JIT compilation
export NVCC_ALLOW_UNSUPPORTED_COMPILER=1

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "${VENV_PATH}" ]; then
    python -m venv "${VENV_PATH}"
fi
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ── Install packages ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing core dependencies..."

# Install PyTorch with cu128 (matches GPU driver CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if [ -f "requirements.txt" ]; then
    echo "Installing remaining dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing core packages manually..."
    pip install vllm transformers accelerate bitsandbytes scipy mesa matplotlib
fi

# ── Set persistent environment variables ──────────────────────────────────────
echo ""
echo "[4/6] Setting environment variables..."
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"

mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_CACHE}"

if ! grep -q "ABM-LLM" "${HOME_DIR}/.bashrc"; then
cat >> "${HOME_DIR}/.bashrc" << EOF

# ── ABM-LLM Simulation Environment ──────────────────────────────────────────
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NVCC_ALLOW_UNSUPPORTED_COMPILER=1

EOF
fi

# ── Download model weights ─────────────────────────────────────────────────
echo ""
echo "[5/6] Downloading model weights..."

export MODEL_DIR="${WORK_DIR}/models"
export HF_HOME="${WORK_DIR}/huggingface_cache"

mkdir -p "${MODEL_DIR}"
mkdir -p "${HF_HOME}"

# ── Token check ───────────────────────────────────────────────────────────────
if [ -n "${HF_TOKEN}" ]; then
    echo "  HF_TOKEN detected — will authenticate with Hugging Face."
else
    echo "  No HF_TOKEN set — proceeding without authentication."
    echo "  (This works for public models like Qwen, Llama, etc.)"
    echo "  For gated models, run: export HF_TOKEN=\"hf_...\" before this script."
fi

python - << 'PYEOF'
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files, login

# ── ENV VARS ──────────────────────────────────────────────────────────────────
token    = os.environ.get("HF_TOKEN")
repo     = os.environ.get("HF_MODEL_DIR")
base_dir = os.environ.get("MODEL_DIR")
name     = os.environ.get("MODEL_NAME")
quant    = os.environ.get("QUANT")

# Validate required vars (token is optional)
for var_name, value in [("HF_MODEL_DIR", repo), ("MODEL_DIR", base_dir), ("MODEL_NAME", name)]:
    if value is None:
        raise ValueError(f"Environment variable {var_name} is missing!")

# ── LOGIN (optional) ─────────────────────────────────────────────────────────
if token:
    try:
        login(token=token, add_to_git_credential=False)
        print("  Authenticated with Hugging Face successfully.")
    except Exception as e:
        print(f"  WARNING: HF login failed: {e}")
        print("  Continuing without authentication (will work for public models)...")
else:
    print("  Skipping HF login (no token provided).")

local_dir = os.path.join(base_dir, name)
os.makedirs(local_dir, exist_ok=True)

# ── DETECT REPO TYPE ──────────────────────────────────────────────────────────
print(f"  Inspecting repo: {repo} ...")
try:
    all_files = list(list_repo_files(repo, token=token if token else None))
except Exception as e:
    print(f"\n  ERROR: Cannot access repo '{repo}'")
    if "401" in str(e) or "403" in str(e):
        print("  This is likely a gated model. Please provide a valid HF_TOKEN:")
        print(f"    export HF_TOKEN=\"hf_your_token_here\"")
        print(f"    bash {sys.argv[0] if sys.argv[0] else 'setup_unity.sh'}")
    else:
        print(f"  Error details: {e}")
    sys.exit(1)

gguf_files = [f for f in all_files if f.endswith(".gguf")]
is_gguf_repo = len(gguf_files) > 0

# ── GGUF REPO ─────────────────────────────────────────────────────────────────
if is_gguf_repo:
    if quant is None:
        print("\n  GGUF repo detected. Available quantizations:\n")
        for f in sorted(gguf_files):
            print(f"    {f}")
        raise ValueError(
            "\n  QUANT env var is not set! "
            "Pick one from the list above and export QUANT=<name>.\n"
            "  Example: export QUANT=Q4_K_M"
        )

    matches = [f for f in gguf_files if quant.upper() in f.upper()]

    if len(matches) == 0:
        print(f"\n  No GGUF file found matching '{quant}'. Available files:\n")
        for f in sorted(gguf_files):
            print(f"    {f}")
        raise ValueError(f"QUANT='{quant}' did not match any file in the repo.")

    if len(matches) > 1:
        print(f"\n  Multiple matches found for '{quant}':")
        for f in matches:
            print(f"    {f}")
        raise ValueError("Ambiguous QUANT — please be more specific (e.g. Q4_K_M instead of Q4).")

    filename = matches[0]
    print(f"\n  GGUF repo | Downloading: {filename}")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=local_dir,
        token=token if token else None,
    )
    print(f"  Done: {os.path.join(local_dir, filename)}")

# ── FULL MODEL REPO ───────────────────────────────────────────────────────────
else:
    print(f"\n  Standard model repo | Downloading all files to: {local_dir}")
    snapshot_download(
        repo_id=repo,
        local_dir=local_dir,
        ignore_patterns=["*.pth", "*.bin"],
        token=token if token else None,
    )
    print(f"  Done: {local_dir}")

PYEOF

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."

if [ -f "${MODEL_DIR}/${MODEL_NAME}/config.json" ]; then
    echo "  SUCCESS: config.json present."
    echo "  Total Model Size: $(du -sh ${MODEL_DIR}/${MODEL_NAME} | cut -f1)"
else
    echo "  WARNING: config.json not found in ${MODEL_DIR}/${MODEL_NAME}"
fi

python - << 'PYEOF'
import torch
import vllm
print(f"  Torch:  {torch.__version__}")
print(f"  CUDA:   {torch.cuda.is_available()}")
print(f"  vLLM:   {vllm.__version__}")
if torch.cuda.is_available():
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
PYEOF

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Run command: source ${VENV_PATH}/bin/activate"
echo "=========================================="