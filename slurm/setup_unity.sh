#!/bin/bash
# =============================================================================
# setup_unity.sh — Robust environment setup for Unity cluster
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

# Updated to the 3.3 Instruct version you requested
# export HF_MODEL_DIR="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
export HF_MODEL_DIR="mistralai/Ministral-3-8B-Instruct-2512"
export MODEL_NAME="Ministral-3-8B-Instruct"
# export QUANT="Q4_K_M"  # ← pick your quant here


# ── Load modules ──────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Loading modules..."
module purge
module load python/3.11.7
module load cuda/12.1

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

# Install PyTorch first with specific CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements if file exists
if [ -f "requirements.txt" ]; then
    echo "Installing remaining dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    # Fallback: Core packages for your Agent-Based Modeling & LLM research
    echo "requirements.txt not found. Installing core packages manually..."
    pip install vllm transformers accelerate bitsandbytes scipy mesa matplotlib
fi

# ── Set persistent environment variables ──────────────────────────────────────
echo ""
echo "[4/6] Setting environment variables..."
# We use export here so the current shell and the following Python block see them
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export HF_TOKEN="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"

mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_CACHE}"

if ! grep -q "ABM-LLM" "${HOME_DIR}/.bashrc"; then
cat >> "${HOME_DIR}/.bashrc" << EOF

# ── ABM-LLM Simulation Environment ──────────────────────────────────────────
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF
fi

# ── Download model weights ─────────────────────────────────────────────────
echo ""
echo "[5/6] Downloading Llama 3.3 70B weights..."

# CRITICAL: These must be EXPORTED so the Python subprocess can see them
export HF_TOKEN="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"
export MODEL_DIR="/work/pi_lfiondella_umassd_edu/Mahnoor/models"
export HF_HOME="${WORK_DIR}/huggingface_cache"

mkdir -p "${MODEL_DIR}"
mkdir -p "${HF_HOME}"

# Use 'PYEOF' to prevent shell expansion of $ variables inside the Python code
python - << 'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files, login

# ── ENV VARS ──────────────────────────────────────────────────────────────────
token    = os.environ.get("HF_TOKEN")
repo     = os.environ.get("HF_MODEL_DIR")
base_dir = os.environ.get("MODEL_DIR")
name     = os.environ.get("MODEL_NAME")
quant    = os.environ.get("QUANT")  # Optional — only for GGUF repos

for var_name, value in [("HF_TOKEN", token), ("HF_MODEL_DIR", repo), ("MODEL_DIR", base_dir), ("MODEL_NAME", name)]:
    if value is None:
        raise ValueError(f"Environment variable {var_name} is missing!")

login(token=token, add_to_git_credential=False)

local_dir = os.path.join(base_dir, name)
os.makedirs(local_dir, exist_ok=True)

# ── DETECT REPO TYPE ──────────────────────────────────────────────────────────
print(f"Inspecting repo: {repo} ...")
all_files = list(list_repo_files(repo))
gguf_files = [f for f in all_files if f.endswith(".gguf")]
is_gguf_repo = len(gguf_files) > 0

# ── GGUF REPO ─────────────────────────────────────────────────────────────────
if is_gguf_repo:
    if quant is None:
        print("\nGGUF repo detected. Available quantizations:\n")
        for f in sorted(gguf_files):
            print(f"  {f}")
        raise ValueError(
            "\nQUANT env var is not set! "
            "Pick one from the list above and export QUANT=<name>.\n"
            "Example: export QUANT=Q4_K_M"
        )

    # Find the matching file — flexible match (doesn't require exact naming)
    matches = [f for f in gguf_files if quant.upper() in f.upper()]

    if len(matches) == 0:
        print(f"\nNo GGUF file found matching '{quant}'. Available files:\n")
        for f in sorted(gguf_files):
            print(f"  {f}")
        raise ValueError(f"QUANT='{quant}' did not match any file in the repo.")

    if len(matches) > 1:
        print(f"\nMultiple matches found for '{quant}':")
        for f in matches:
            print(f"  {f}")
        raise ValueError("Ambiguous QUANT — please be more specific (e.g. Q4_K_M instead of Q4).")

    filename = matches[0]
    print(f"\nGGUF repo | Downloading: {filename}")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    print(f"Done: {os.path.join(local_dir, filename)}")

# ── FULL MODEL REPO ───────────────────────────────────────────────────────────
else:
    print(f"\nStandard model repo | Downloading all files to: {local_dir}")
    snapshot_download(
        repo_id=repo,
        local_dir=local_dir,
        ignore_patterns=["*.pth", "*.bin"],  # skip redundant formats if safetensors exist
    )
    print(f"Done: {local_dir}")

PYEOF


# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying installation and files..."

if [ -f "${MODEL_DIR}/${MODEL_NAME}/config.json" ]; then
    echo "  SUCCESS: config.json present."
    echo "  Total Model Size: $(du -sh ${MODEL_DIR}/${MODEL_NAME} | cut -f1)"
else
    echo "  WARNING: config.json not found in ${MODEL_DIR}/${MODEL_NAME}"
fi

python - << 'PYEOF'
import torch
import os
import vllm
print(f"  Torch CUDA available: {torch.cuda.is_available()}")
print(f"  vLLM version:        {vllm.__version__}")
PYEOF

echo "=========================================="
echo "  Setup complete!"
echo "  Run command: source ${VENV_PATH}/bin/activate"
echo "=========================================="

