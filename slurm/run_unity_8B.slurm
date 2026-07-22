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


# ─────────────────────────────────────────────────────────────────────────────
# OPTION B: Small Model (Commented out with ## so SLURM ignores them)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=nv_resistance_ministral_8B_zeroshot
#SBATCH --partition=gpu
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=vram48
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fmahnoor1@umassd.edu


# =============================================================================
# CONFIGURATION
# =============================================================================

N_RUNS=100
PARALLEL_RUNS=25
GPU_BATCH_SIZE=25      
START_RUN=1
VLLM_PORT=8000
MAX_WAIT=600
PROMPT_TYPE="ZeroShot"
MAX_STEPS=200

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME_DIR="/home/fmahnoor1_umassd_edu"
WORK_DIR="/work/pi_lfiondella_umassd_edu/Mahnoor"
MODEL_CACHE="${WORK_DIR}/huggingface_cache"
PROJECT_DIR="${HOME_DIR}/pyabm-llm"
VENV_PATH="${HOME_DIR}/venv_abm_llm"

MODEL_PATH="${WORK_DIR}/models/Ministral-3-8B-Reasoning-2512"
SERVED_MODEL_NAME="Ministral-3-8B-Reasoning-2512"
OUTPUT_DIR="${WORK_DIR}/SimulationLogs/${SERVED_MODEL_NAME}_${PROMPT_TYPE}"


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

module purge
# 1. Point the system to the URI module directory
module use /modules/uri_modulefiles

# 2. Now you can load it successfully
module load compiler/GCC/12.3.0
module load python/3.11.7
module load cuda/12.1

# 3. Clean the cache to remove the failed GCC 13 build logs
rm -rf ~/.cache/flashinfer

# It helps to explicitly point to the compilers
export CC=$(which gcc)
export CXX=$(which g++)

# Python was compiled with GCC 13 — force NVCC to accept it
export NVCC_ALLOW_UNSUPPORTED_COMPILER=1


echo "=== Module Check ==="
which gcc
gcc --version
which nvcc
nvcc --version
echo "=== End Module Check ==="


source "${VENV_PATH}/bin/activate"
echo "Python: $(which python) — $(python --version)"

echo "--- GPU DIAGNOSTICS ---"
nvidia-smi
echo ""

python -c "
import torch, sys
print(f'Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

export HF_TOKEN="HF_TOKEN=hf_------------"
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

cd "$PROJECT_DIR"

# =============================================================================
# START vLLM SERVER
# =============================================================================

echo "[$(date +%H:%M:%S)] Starting vLLM server on port ${VLLM_PORT}..."

# Check port is free
if lsof -i :${VLLM_PORT} > /dev/null 2>&1; then
    echo "ERROR: Port ${VLLM_PORT} already in use. Aborting."
    exit 1
fi

export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export EXTRA_NVCCFLAGS="--allow-unsupported-compiler"
export VLLM_USE_FLASHINFER=0

# NO trailing spaces after backslashes
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs $GPU_BATCH_SIZE \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --no-enable-log-requests \
    --port $VLLM_PORT \
    --trust-remote-code \
    --enforce-eager \
    > "${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
echo "[$(date +%H:%M:%S)] vLLM starting (PID=$VLLM_PID)..."

# Trap set IMMEDIATELY after PID is known
trap "echo 'Shutting down vLLM (PID=$VLLM_PID)...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# =============================================================================
# WAIT FOR SERVER
# =============================================================================

echo "[$(date +%H:%M:%S)] Waiting for vLLM (max ${MAX_WAIT}s)..."
ELAPSED=0
while true; do
    # Check process still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died! Last 50 lines:"
        tail -n 50 "${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi

    # Check timeout
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Timeout after ${MAX_WAIT}s. Last 50 lines:"
        tail -n 50 "${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi

    STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
             --connect-timeout 2 \
             http://localhost:${VLLM_PORT}/v1/models)

    if [ "$STATUS" = "200" ]; then
        echo "[$(date +%H:%M:%S)] vLLM ready after ${ELAPSED}s"
        break
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... ${ELAPSED}s elapsed (HTTP ${STATUS:-000})"
done

# =============================================================================
# RUN SIMULATION
# =============================================================================

echo "[$(date +%H:%M:%S)] Starting simulation..."

python run_cluster.py \
    --n_runs        $N_RUNS \
    --start_run     $START_RUN \
    --provider      vllm \
    --model         "$SERVED_MODEL_NAME" \
    --vllm_port     $VLLM_PORT \
    --prompt_type   $PROMPT_TYPE \
    --output_dir    "$OUTPUT_DIR" \
    --max_steps     $MAX_STEPS \
    --pillar_strat  3 \
    --max_workers   $PARALLEL_RUNS

EXIT_CODE=$?

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=========================================="
echo "  Simulation finished: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "  Output: $OUTPUT_DIR"
echo "  vLLM log: ${OUTPUT_DIR}/vllm_${SLURM_JOB_ID}.log"
echo "=========================================="

exit $EXIT_CODE
