#!/bin/bash
# =============================================================================
# run_unity.slurm — NV Resistance Simulation on Unity Cluster
#
# Architecture:
#   1. Start vLLM server in background (Llama 3.1 8B AWQ, ~2-4s/response)
#   2. Wait for server to be healthy
#   3. Run simulation sequentially (N runs, each isolated log directory)
#   4. Kill vLLM server on exit
#
# Submit:
#   sbatch run_unity.slurm
#
# Monitor:
#   squeue -u fmahnoor1_umassd_edu
#   tail -f logs/sim_<jobid>.out
#
# Resume from run K if interrupted:
#   Edit START_RUN=K below and resubmit
# =============================================================================

#SBATCH --job-name=nv_resistance
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fmahnoor1@umassd.edu

# =============================================================================
# CONFIGURATION — edit N_RUNS and START_RUN as needed
# =============================================================================

N_RUNS=10          # number of simulation runs
START_RUN=0        # change to resume from a specific run_id
VLLM_PORT=8000
PROMPT_TYPE="ZeroShot"   # ZeroShot or CoT
MAX_STEPS=200

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME_DIR="/home/fmahnoor1_umassd_edu"
WORK_DIR="/work/pi_lfiondella_umassd_edu/Mahnoor"
MODEL_PATH="${WORK_DIR}/models/llama-3.1-8b-awq"
MODEL_CACHE="${WORK_DIR}/huggingface_cache"
PROJECT_DIR="${HOME_DIR}/pyabm-llm"      # ← adjust if your repo is elsewhere
VENV_PATH="${HOME_DIR}/venv_abm_llm"
OUTPUT_DIR="${WORK_DIR}/SimulationLogs"        # logs go to work dir (1TB quota)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "=========================================="
echo "  Non-Violent Resistance Simulation"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $(hostname)"
echo "  Started: $(date)"
echo "  Runs:    $START_RUN → $((START_RUN + N_RUNS - 1))"
echo "=========================================="
echo ""

# Load modules
module purge
module load python/3.11.7
module load cuda/12.1

# Activate venv
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python) — $(python --version)"

# HuggingFace + cache configuration
export HF_TOKEN="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"
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Output directories
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# GPU check
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Move to project directory
cd "$PROJECT_DIR"

# =============================================================================
# START vLLM SERVER
# =============================================================================

echo "[$(date +%H:%M:%S)] Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Port:  $VLLM_PORT"

# Kill any stale process on this port
fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true
sleep 2

# Start vLLM in background — logs to separate file
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --quantization awq \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --port $VLLM_PORT \
    --disable-log-requests \
    --trust-remote-code \
    > "${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
echo "[$(date +%H:%M:%S)] vLLM starting (PID=$VLLM_PID)..."

# ── Wait for server to be healthy (max 5 minutes) ────────────────────────────
MAX_WAIT=300
ELAPSED=0
echo "[$(date +%H:%M:%S)] Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... ${ELAPSED}s elapsed"
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s. Last log lines:"
        tail -20 "${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi
done
echo "[$(date +%H:%M:%S)] vLLM server ready."
echo ""

# Trap to ensure vLLM is killed on exit (even on error)
trap "echo 'Shutting down vLLM...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# =============================================================================
# RUN SIMULATION
# =============================================================================

echo "[$(date +%H:%M:%S)] Starting simulation..."
echo ""

python run_cluster.py \
    --n_runs        $N_RUNS \
    --start_run     $START_RUN \
    --provider      vllm \
    --model         "$MODEL_PATH" \
    --vllm_port     $VLLM_PORT \
    --prompt_type   $PROMPT_TYPE \
    --output_dir    "$OUTPUT_DIR" \
    --max_steps     $MAX_STEPS \
    --pillar_strat  3 \
    --max_workers   4         


EXIT_CODE=$?

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=========================================="
echo "  Simulation finished: $(date)"
echo "  Exit code: $EXIT_CODE"
echo ""
echo "  Output directory: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    echo "  Run directories: $(ls -d ${OUTPUT_DIR}/**/run_* 2>/dev/null | wc -l)"
fi
echo "  vLLM log: ${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log"
echo "=========================================="

exit $EXIT_CODE
