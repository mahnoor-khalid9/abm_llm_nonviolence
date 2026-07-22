#!/bin/bash
# =============================================================================
# run_attention_extraction.slurm - Extract attention weights (ZS + CoT)
#
# Loads each model one at a time, runs 10 scenarios x 2 prompt types per model.
# Total: 3 models x 10 scenarios x 2 prompts = 60 forward passes + 60 generations.
# Runtime estimate: ~45-90 minutes for all 3 models on L40S/A100.
#
# Submit:
#   sbatch run_attention_extraction.slurm
# =============================================================================

#SBATCH --job-name=attn_extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=vram48
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/attn_%j.out
#SBATCH --error=logs/attn_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fmahnoor1@umassd.edu

set -e

echo "=========================================="
echo "  Attention Extraction (ZS + CoT)"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $(hostname)"
echo "  Started: $(date)"
echo "=========================================="

module purge
module load python/3.11.7
module load cuda/12.1

HOME_DIR="/home/fmahnoor1_umassd_edu"
VENV_PATH="${HOME_DIR}/venv_abm_llm"
PROJECT_DIR="${HOME_DIR}/pyabm-llm"

source "${VENV_PATH}/bin/activate"
echo "Python: $(which python) - $(python --version)"

# Log the assigned GPU explicitly for reproducibility
echo ""
echo "Assigned GPU:"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader

cd "$PROJECT_DIR"
mkdir -p attention_data logs

echo ""
echo "[$(date +%H:%M:%S)] Starting attention extraction..."
echo "  Models will be loaded sequentially (one in VRAM at a time)."
echo "  Each model runs ZeroShot AND CoT prompt types."
echo ""

python extract_attention.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "  Finished: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "  Output files:"
ls -lh attention_data/*.npz 2>/dev/null || echo "  (no .npz files found)"
echo "=========================================="

exit $EXIT_CODE
