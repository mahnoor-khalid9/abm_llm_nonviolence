"""
run_cluster.py — CLI entry point for SLURM cluster runs.

Called by the SLURM scripts. Handles provider selection, LogConfig
construction, and delegates to RunOrchestrator.

Usage (from SLURM script)
--------------------------
    python run_cluster.py \\
        --n_runs      10 \\
        --start_run   0 \\
        --provider    vllm \\
        --model       /work/pi_lfiondella_umassd_edu/Mahnoor/models/Llama-3.1-8B-Instruct \\
        --vllm_port   8000 \\
        --prompt_type CoT \\
        --output_dir  SimulationLogs

Usage (local testing)
---------------------
    python run_cluster.py --n_runs 2 --provider vllm --prompt_type CoT
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Parse arguments ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NV Resistance Simulation — Cluster Runner")
    p.add_argument("--n_runs",      type=int,   default=10,
                   help="Number of simulation runs")
    p.add_argument("--start_run",   type=int,   default=0,
                   help="Starting run_id (for resuming)")
    p.add_argument("--provider",    type=str,   default="vllm",
                   choices=["vllm", "huggingface", "gemini", "openai"],
                   help="LLM provider")
    p.add_argument("--model",       type=str,
                   default="/work/pi_lfiondella_umassd_edu/Mahnoor/models/Llama-3.3-70B-Instruct",
                   help="Model ID or path")
    p.add_argument("--vllm_port",   type=int,   default=8000,
                   help="vLLM server port (vllm provider only)")
    p.add_argument("--prompt_type", type=str,   default="CoT",
                   choices=["ZeroShot", "CoT"],
                   help="Prompt type")
    p.add_argument("--output_dir",  type=str,   default="SimulationLogs",
                   help="Base directory for all log output")
    p.add_argument("--max_workers", type=int,   default=10,
                   help="Parallel workers (1=sequential, >1=parallel)")
    p.add_argument("--max_steps",   type=int,   default=200)
    p.add_argument("--pillar_strat",type=int,   default=3,
                   help="PillarProxStrat (3=LLM-guided)")
    p.add_argument("--memory",      type=str,   default="stateful")
    return p.parse_args()


# ── Provider factory ──────────────────────────────────────────────────────────
def build_chat_model(args):
    """
    Instantiate the correct provider and return a ChatModel-compatible wrapper.
    Called inside _run_one so it happens in the worker process (not parent).
    """
    from LLM.chat_model import ChatModel

    provider_kwargs = {
        "prompt_dir":  "./Prompts",
        "prompt_type": args.prompt_type,
    }

    if args.provider == "vllm":
        provider_kwargs.update({
            "base_url":    f"http://localhost:{args.vllm_port}",
            "model":       args.model,
            "temperature": 0.0,
            "max_tokens":  150,
        })
    elif args.provider == "huggingface":
        provider_kwargs.update({
            "model_name":     args.model,
            "quantization":   "4bit",
            "max_new_tokens": 150,
            "device":         "cuda",
        })
    elif args.provider == "gemini":
        provider_kwargs.update({
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        })
    elif args.provider == "openai":
        provider_kwargs.update({
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        })

    return ChatModel(
        provider=args.provider,
        use_memory=True,
        **provider_kwargs,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    from Logging.log_config import LogConfig
    from run_simulation import RunOrchestrator, BASE_PARAMS

    # Identify the run by provider + model + prompt type
    model_short = args.model.split("/")[-1].lower().replace(".", "-")

    log_cfg = LogConfig(
        provider    = f"{args.provider}",
        prompt_type = args.prompt_type,
        memory      = args.memory,
        vision      = "vision-4",
        base_dir    = args.output_dir,
    )

    params = dict(BASE_PARAMS)
    params.update({
        "MaxSteps":       args.max_steps,
        "PillarProxStrat": args.pillar_strat,
    })

    print(f"\n{'='*60}")
    print(f"  Provider    : {args.provider}")
    print(f"  Model       : {args.model}")
    print(f"  Prompt type : {args.prompt_type}")
    print(f"  Runs        : {args.start_run} → {args.start_run + args.n_runs - 1}")
    print(f"  Output      : {args.output_dir}")
    print(f"{'='*60}\n")

    orchestrator = RunOrchestrator(
        base_config  = log_cfg,
        params       = params,
        max_workers  = args.max_workers,      # 4 parallel runs → vLLM batches their LLM calls
        start_run_id = args.start_run,
    )

    summary = orchestrator.run(n_runs=args.n_runs)
    return 0 if summary.n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
