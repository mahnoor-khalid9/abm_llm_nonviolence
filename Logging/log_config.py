"""
log_config.py — Single source of truth for all simulation logging paths and settings.

Both agent_based.py and resistance_model.py import from here.
Changing provider / version / memory in ONE place propagates everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LogConfig:
    """
    Immutable configuration object.  Create once in your run script and pass it
    into NVResistanceModel — never mutate it during a run.

    Parameters
    ----------
    provider     : LLM provider label used in directory names.
    version      : Experiment version number.
    memory       : Memory mode label used in directory names.
    prompt_type  : Prompt type label used in directory names.
    vision       : Vision radius label used in directory names (e.g. "vision-4").
    base_dir     : Root directory for all log output.
    """

    provider:    str = "gemini_flash"
    version:     int = 2
    memory:      str = "stateful"
    prompt_type: str = "ZeroShot"
    vision:      str = "vision-4"
    base_dir:    str = "SimulationLogs"
    # folder_name: str = "SimulationLogs"

    # ── Derived paths (computed once at construction) ─────────────────────────

    @property
    def run_id(self) ->  str:
        return f"{self.version}"
    
    @property
    def _experiment_root(self) -> str:
        return os.path.join(
            self.base_dir,
            self.provider,
            self.prompt_type,
            self.memory,
            self.vision,
        )

    @property
    def agent_log_root(self) -> str:
        return os.path.join(self._experiment_root, "AgentsLogging")

    @property
    def model_log_root(self) -> str:
        return os.path.join(self._experiment_root, "ModelLogging")

    # ── Per-step agent events (one .txt per step, parsed by raw_memory.py) ──

    @property
    def simulation_log_dir(self) -> str:
        return os.path.join(
            self.agent_log_root, "simulation_log", "unstructured", f"v{self.version}"
        )

    def step_log_path(self, step: int) -> str:
        return os.path.join(self.simulation_log_dir, f"step_{step}.txt")

    # ── Unified single-file agent log (errors, retries, fallbacks) ──────────

    @property
    def agent_error_log(self) -> str:
        return os.path.join(
            self.agent_log_root,
            "unstructured-error-logs",
            f"agents_simulation_log_v{self.version}.txt",
        )

    # ── Memory outputs ───────────────────────────────────────────────────────

    @property
    def memory_raw_dir(self) -> str:
        return os.path.join(
            self.agent_log_root, "memory_log", "raw_json", f"v{self.version}"
        )

    @property
    def memory_nlp_dir(self) -> str:
        return os.path.join(
            self.agent_log_root, "memory_log", "nlp_json", f"v{self.version}"
        )

    def memory_raw_path(self, step: int) -> str:
        return os.path.join(self.memory_raw_dir, f"step_{step}.json")

    def memory_nlp_path(self, step: int) -> str:
        return os.path.join(self.memory_nlp_dir, f"step_{step}.json")

    @property
    def narrative_debug_log(self) -> str:
        return os.path.join(self.memory_nlp_dir, "narrative_log.txt")

    # ── Model-level log ──────────────────────────────────────────────────────

    @property
    def model_log(self) -> str:
        return os.path.join(
            self.model_log_root,
            "unstructured-logs",
            f"model_simulation_log_v{self.version}.txt",
        )

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def create_all_dirs(self) -> None:
        """Create every directory this config references.  Call once at run start."""
        for path in [
            self.simulation_log_dir,
            os.path.dirname(self.agent_error_log),
            self.memory_raw_dir,
            self.memory_nlp_dir,
            os.path.dirname(self.model_log),
        ]:
            os.makedirs(path, exist_ok=True)

    def reset_run_logs(self) -> None:
        """
        Remove stale files from a previous partial run of THIS run_id only.
        Safe because each run_id is an isolated directory.
        """
        for path in [self.agent_error_log, self.model_log, self.narrative_debug_log]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass  # file locked on Windows — SimLogger will append, still clean enough


# ── Module-level default (convenience for single-experiment scripts) ──────────
# Override by constructing your own LogConfig and passing it to NVResistanceModel.
DEFAULT_LOG_CONFIG = LogConfig()
