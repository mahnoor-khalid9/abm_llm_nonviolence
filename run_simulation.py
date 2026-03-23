"""
run_simulation.py — Orchestrator for running the simulation N times reliably.

Why not Mesa's batch_run()?
----------------------------
Mesa's batch_run() uses multiprocessing.Pool, which pickles the model object
and sends it to worker processes. Our simulation has two resources that don't
survive pickling cleanly:

  1. SimLogger (open file handles — fixed by lazy opening + __getstate__)
  2. ChatModel / Gemini client (holds an authenticated HTTP session)

More importantly: each step of our simulation calls an LLM API, which is
rate-limited. Running many workers in parallel saturates the API instantly,
causing 429 errors that silently corrupt or drop logs.

This orchestrator runs runs SEQUENTIALLY by default, which:
  - Avoids API rate limiting
  - Gives each run a guaranteed isolated directory (run_id)
  - Produces clean, complete logs for all 100 runs
  - Lets you resume from run K if a crash happens mid-batch

For TRUE parallelism (e.g. multiple API keys, local models), set
`max_workers > 1`.  The code handles this safely because each run
has its own LogConfig(run_id=N).

Usage (from the notebook)
--------------------------
    from run_simulation import RunOrchestrator, BASE_PARAMS
    from log_config import LogConfig

    cfg = LogConfig(provider="gemini_flash", version=12, base_dir="logs")
    orch = RunOrchestrator(base_config=cfg, params=BASE_PARAMS)
    summary = orch.run(n_runs=100)
    print(summary)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from Logging.log_config import LogConfig


# ── Canonical default parameters ─────────────────────────────────────────────
# Change these here; they propagate to every run automatically.

BASE_PARAMS: Dict[str, Any] = {
    "MaxSteps":          200,
    "PctCitizen":        70,
    "PctActivist":       0.8,
    "PctPolice":         4,
    "PctPillar":         1.53,
    "MaxJailTerm":       10,
    "StartLegitimacy":   0.56,
    "PctFindNV":         40,
    "PctTargetNV":       25,
    "PctKillNV":         10,
    "BackfireCoeff":     0.99,
    "valueF":            0.0706,
    "ProtestCycle":      8.9,
    "ProtestDuration":   1,
    "nNVthreshold":      1,
    "DefectThresh":      0.05,
    "PeerPressureNum":   3.46,
    "DelayStartMax":     5,
    "NVSuccessPct":      10,
    "PillarProxStrat":   3,
    "DefectThreshStDev": 0.01,
    "ActivistSearchVision": 10,
    "PctFickle":         100,
    "PctImmedProtest":   5,
    "dataCollectorType": "batchRun",
}



# ── Per-run result record ─────────────────────────────────────────────────────

@dataclass
class RunResult:
    run_id:        int
    seed:          Optional[int]
    victory:       int              # 0=undefined 1=regime 2=resistance 3=timeout
    outcome:       str              # "resistance" | "regime" | "error"
    steps_run:     int
    duration_s:    float
    log_dir:       str
    error:         Optional[str]    = None


# ── Worker function (must be top-level for multiprocessing pickle) ────────────

def _run_one(run_id: int, params: Dict, cfg_kwargs: Dict, seed: Optional[int]) -> RunResult:
    """
    Execute a single simulation run.  Runs inside a worker process when
    max_workers > 1, or in the main process when sequential.

    We import NVResistanceModel HERE (inside the function) rather than at
    module top-level.  This means the import (and any module-level side
    effects) happens inside the worker process, not in the parent.
    """
    from resistance_model import NVResistanceModel
    cfg = LogConfig(**{**cfg_kwargs, "version": run_id})
    # cfg = LogConfig(**cfg_kwargs).with_run_id(run_id)
    cfg.create_all_dirs()
    cfg.reset_run_logs()

    t0 = time.time()
    model = None
    try:
        model = NVResistanceModel(
            log_config=cfg,
            seed=seed,
            **{k: v for k, v in params.items() if k != "dataCollectorType"},
            dataCollectorType="batchRun",
        )

        while model.running:
            model.step()

        outcome = model.get_final_victory_status()
        steps   = model.schedule.steps
        victory = model.victory

    except Exception as exc:
        tb = traceback.format_exc()
        # Write crash info to the model log before giving up
        if model is not None and hasattr(model, "logger"):
            model.logger.log_model_init(f"CRASH at step {getattr(model.schedule, 'steps', '?')}: {tb}")
        outcome = "error"
        steps   = getattr(getattr(model, "schedule", None), "steps", 0)
        victory = -1
        return RunResult(
            run_id=run_id, seed=seed, victory=victory, outcome=outcome,
            steps_run=steps, duration_s=time.time() - t0,
            log_dir=cfg._experiment_root, error=tb,
        )
    finally:
        if model is not None and hasattr(model, "logger"):
            model.logger.close()

    return RunResult(
        run_id=run_id, seed=seed, victory=victory, outcome=outcome,
        steps_run=steps, duration_s=time.time() - t0,
        log_dir=cfg._experiment_root, error=None,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class RunOrchestrator:
    """
    Runs the simulation N times, each in its own isolated log directory.

    Parameters
    ----------
    base_config  : LogConfig template.  run_id is overridden per run.
    params       : Model parameters dict (see BASE_PARAMS above).
    seeds        : Optional list of seeds.  If None, seeds 0…n_runs-1 are used.
    max_workers  : 1 = sequential (recommended for API-based LLMs).
                   N > 1 = N parallel workers (safe only with local models or
                   when you have N separate API keys configured).
    start_run_id : Resume from this run_id (skip already-completed runs).
    """

    def __init__(
        self,
        base_config:   LogConfig,
        params:        Dict[str, Any] = None,
        seeds:         Optional[List[int]] = None,
        max_workers:   int = 1,
        start_run_id:  int = 0,
    ):
        self.base_config  = base_config
        self.params       = params or BASE_PARAMS
        self.seeds        = seeds
        self.max_workers  = max_workers
        self.start_run_id = start_run_id
        self.base_log_dir = self.base_config._experiment_root

    def run(self, n_runs: int = 100) -> "BatchSummary":
        run_ids = list(range(self.start_run_id, self.start_run_id + n_runs))
        seeds   = (
            self.seeds[:n_runs]
            if self.seeds
            else list(range(self.start_run_id, self.start_run_id + n_runs))
        )

        # Serialise LogConfig to plain dict for safe pickling across processes
        cfg_kwargs = {
            "provider":    self.base_config.provider,
            "version":     self.base_config.version,
            "memory":      self.base_config.memory,
            "prompt_type": self.base_config.prompt_type,
            "vision":      self.base_config.vision,
            "base_dir":    self.base_config.base_dir,
        }

        results: List[RunResult] = []
        total   = len(run_ids)

        print(f"[RunOrchestrator] Starting {total} runs "
              f"({'sequential' if self.max_workers == 1 else f'{self.max_workers} workers'})")
        batch_t0 = time.time()

        if self.max_workers == 1:
            # ── Sequential path (default, safest for API-rate-limited LLMs) ──
            for i, (run_id, seed) in enumerate(zip(run_ids, seeds)):
                print(f"  Run {run_id:4d} / {run_ids[-1]}  (seed={seed})", end="  ", flush=True)
                result = _run_one(run_id, self.params, cfg_kwargs, seed)
                results.append(result)
                status = "✓" if result.error is None else "✗ ERROR"
                print(f"{status}  {result.outcome:<12}  {result.duration_s:.1f}s  "
                      f"[{i+1}/{total}]")
                if result.error:
                    print(f"    └─ {result.error.splitlines()[-1]}")

        else:
            # ── Parallel path (only for local / rate-unlimited models) ────────
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(_run_one, run_id, self.params, cfg_kwargs, seed): run_id
                    for run_id, seed in zip(run_ids, seeds)
                }
                for fut in as_completed(futures):
                    result = fut.result()
                    results.append(result)
                    status = "✓" if result.error is None else "✗ ERROR"
                    print(f"  Run {result.run_id:4d}  {status}  {result.outcome:<12}  "
                          f"{result.duration_s:.1f}s  [{len(results)}/{total}]")

        summary = BatchSummary(
            results=results,
            total_duration_s=time.time() - batch_t0,
            base_log_dir=self.base_log_dir,
        )
        summary.save(self.base_config)
        summary.print()
        return summary


# ── Batch summary ─────────────────────────────────────────────────────────────

@dataclass
class BatchSummary:
    results:          List[RunResult]
    total_duration_s: float
    base_log_dir:     str

    @property
    def n_resistance(self) -> int:
        return sum(1 for r in self.results if r.outcome == "resistance")

    @property
    def n_regime(self) -> int:
        return sum(1 for r in self.results if r.outcome == "regime")

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if r.outcome == "error")

    @property
    def n_total(self) -> int:
        return len(self.results)

    def print(self) -> None:
        print("\n" + "=" * 60)
        print(f"  Batch complete: {self.n_total} runs in {self.total_duration_s:.1f}s")
        print(f"  Resistance wins : {self.n_resistance:3d} / {self.n_total}")
        print(f"  Regime wins     : {self.n_regime:3d} / {self.n_total}")
        print(f"  Errors          : {self.n_errors:3d} / {self.n_total}")
        if self.n_errors:
            print("  Errored run_ids:", [r.run_id for r in self.results if r.outcome == "error"])
        print("=" * 60)

    def save(self, cfg: LogConfig) -> None:
        """Write a JSON summary file next to the run directories."""
        summary_dir = cfg._experiment_root
        # summary_dir = os.path.join(
        #     cfg.base_dir, cfg.provider, cfg.prompt_type, cfg.memory, cfg.vision
        # )
        os.makedirs(summary_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(summary_dir, f"batch_summary_{ts}.json")

        payload = {
            "timestamp":        ts,
            "n_runs":           self.n_total,
            "n_resistance":     self.n_resistance,
            "n_regime":         self.n_regime,
            "n_errors":         self.n_errors,
            "total_duration_s": round(self.total_duration_s, 2),
            "runs": [
                {
                    "run_id":     r.run_id,
                    "seed":       r.seed,
                    "victory":    r.victory,
                    "outcome":    r.outcome,
                    "steps_run":  r.steps_run,
                    "duration_s": round(r.duration_s, 2),
                    "log_dir":    r.log_dir,
                    "error":      r.error,
                }
                for r in self.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Batch summary saved → {path}")
