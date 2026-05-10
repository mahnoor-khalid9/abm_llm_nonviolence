"""
simulation_logger.py — Buffered, picklable logger for the NV Resistance simulation.

Core design decisions
----------------------

1. LAZY file handles — handles are opened on the FIRST write, not in __init__.
   This is the single most important change for multiprocessing safety.
   Mesa's batch_run pickles the model object (including self.logger) to send it
   to worker processes. A logger that holds open file handles at pickle time
   crashes with "cannot pickle '_io.TextIOWrapper' object". A logger that opens
   handles lazily has nothing to pickle except the LogConfig (a plain dataclass).

2. Pickle-safe via __getstate__ / __setstate__ — if the logger is pickled after
   some handles have been opened (e.g. in the parent process), __getstate__
   drops all open handles from the pickle payload. __setstate__ restores the
   config so handles can be re-opened lazily in the worker.

3. One file handle per output file — opened once, written to many times,
   closed once. No per-event open/close syscall overhead.

4. Sanitised field values — free-text LLM output (reasoning, cta) is sanitised
   before writing so embedded semicolons/newlines cannot corrupt the
   semicolon-delimited format that raw_memory.py parses.

5. Errors go to a dedicated error log, never to the step files that
   raw_memory.py parses. This keeps step files clean and parseable.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import suppress
from typing import Any, Dict, Optional

from Logging.log_config import LogConfig


# ── Sanitisation ──────────────────────────────────────────────────────────────

_CTRL = re.compile(r"[\r\n\t]+")


def _sanitise(value: Any, max_len: int = 400) -> str:
    """
    Make a value safe to embed in a key=value; log line:
    - Coerce to str, collapse whitespace, escape semicolons, truncate.
    """
    s = _CTRL.sub(" ", str(value)).strip()
    s = s.replace(";", "[SC]")
    return s[:max_len] + "…" if len(s) > max_len else s


def _json_field(obj: Any) -> str:
    """Compact JSON string with semicolons escaped."""
    try:
        raw = json.dumps(obj, separators=(",", ":"), default=str)
    except Exception:
        raw = str(obj)
    return raw.replace(";", "[SC]")


# ── Lazy file handle wrapper ───────────────────────────────────────────────────

class _LazyLog:
    """
    File handle that opens on first write and closes on demand.

    NOT directly picklable (holds an open fd after first use), but
    SimLogger's __getstate__ excludes all _LazyLog instances from the
    pickle payload so they are re-created lazily in the new process.
    """

    def __init__(self, path: str, flush_every: int = 50):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path        = path
        self._flush_every = flush_every
        self._fh          = None   # opened on first write
        self._count       = 0

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open(self._path, "a", encoding="utf-8", buffering=8192)

    def write(self, line: str) -> None:
        self._ensure_open()
        self._fh.write(line + "\n")
        self._count += 1
        if self._count >= self._flush_every:
            self._fh.flush()
            self._count = 0

    def flush(self) -> None:
        if self._fh is not None:
            with suppress(Exception):
                self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            with suppress(Exception):
                self._fh.flush()
                self._fh.close()
            self._fh = None


# ── SimLogger ─────────────────────────────────────────────────────────────────

class SimLogger:
    """
    Centralised logger for one simulation run.

    Thread / process safety
    -----------------------
    Each NVResistanceModel instance should have its own SimLogger pointing at
    its own run_id directory. Multiple runs must NEVER share a SimLogger
    instance. The RunOrchestrator (run_simulation.py) enforces this.

    Pickle safety
    -------------
    __getstate__ strips all open _LazyLog handles before pickling.
    __setstate__ restores only the config; handles re-open lazily in the
    worker process on first write.
    """

    def __init__(self, config: LogConfig, flush_every: int = 5):
        self.cfg          = config
        self._flush_every = flush_every
        # _step_logs opened lazily per step; _fixed opened lazily on first write
        self._step_logs: Dict[int, _LazyLog] = {}
        self._error_log     = _LazyLog(config.agent_error_log,    flush_every)
        self._model_log     = _LazyLog(config.model_log,          1)
        self._narrative_log = _LazyLog(config.narrative_debug_log, flush_every)

    # ── Pickle support ────────────────────────────────────────────────────────

    def __getstate__(self):
        """
        Drop all open file handles from the pickle payload.
        Only the LogConfig and flush_every setting are pickled — both are
        plain Python objects with no open resources.
        """
        return {"cfg": self.cfg, "_flush_every": self._flush_every}

    def __setstate__(self, state):
        """Re-initialise with lazy handles in the new process."""
        self.cfg          = state["cfg"]
        self._flush_every = state["_flush_every"]
        self._step_logs   = {}
        self._error_log     = _LazyLog(self.cfg.agent_error_log,    self._flush_every)
        self._model_log     = _LazyLog(self.cfg.model_log,          1)
        self._narrative_log = _LazyLog(self.cfg.narrative_debug_log, self._flush_every)

    # ── Step file access ──────────────────────────────────────────────────────

    def _get_step_log(self, step: int) -> _LazyLog:
        if step not in self._step_logs:
            self._step_logs[step] = _LazyLog(
                self.cfg.step_log_path(step), self._flush_every
            )
        return self._step_logs[step]

    def flush_step(self, step: int) -> None:
        """Flush the step's buffer to disk. Call before reading it in raw_memory."""
        if step in self._step_logs:
            self._step_logs[step].flush()

    def close_step(self, step: int) -> None:
        """Flush, close, and release the step's handle."""
        if step in self._step_logs:
            self._step_logs[step].close()
            del self._step_logs[step]

    # ── Agent event logging (goes to per-step file) ───────────────────────────

    def log_agent_event(
        self,
        agent,
        *,
        event: str,
        pos,
        context: Optional[Dict]     = None,
        from_pos                    = None,
        to_pos                      = None,
        target_agent: str           = "None",
        target_pillar_inst: str     = "None",
        primary_attribute: str       = "None",
        non_violent_action_method : str         = "None",
        response_tokens: str         = "None",
        elapsed: str    = "None",
        situational_read: str = "None",
        llm_response = "None",
        memory: str = "None",
    ) -> None:
        """
        Write one structured agent event to step_N.txt.

        Format: Step=N; agent_type=X; agent_id=Y; event=Z; pos=...; context={...}; ...
        This is the exact format raw_memory.py parses.
        Free-text fields are sanitised before writing.
        """

        step  = agent.model.schedule.steps
        parts = [
            f"Step={step}",
            f"agent_type={agent.__class__.__name__}",
            f"agent_id={agent.unique_id}",
            f"event={event}",
            f"pos={pos}",
            f"from_pos={from_pos}",
            f"to_pos={to_pos}",
            f"elapsed={elapsed}",
            f"response_tokens={response_tokens}",
            f"target_agent=\"{target_agent}\"",
            f"target_pillar_inst=\"{target_pillar_inst}\"",
            f"primary_attribute=\"{primary_attribute}\"",
            f"non_violent_action_method=\"{non_violent_action_method}\"",
        ]
        if situational_read:
            parts.append(f"situational_read=\"{situational_read}\"")
        if memory:
            parts.append(f"memory=\"{memory}\"")
        if llm_response:
            parts.append(f"llm_response=\"{llm_response}\"")
        if context:
            parts.append(f"context={_json_field(context or {})}")

        self._get_step_log(step).write(";".join(parts))

    # ── Error logging (goes to dedicated error log, NOT step files) ───────────

    def log_error(self, agent, message: str) -> None:
        step = agent.model.schedule.steps
        line = (
            f"Step={step}; agent_type={agent.__class__.__name__}; "
            f"agent_id={agent.unique_id}; ERROR={_sanitise(message, 1000)}"
        )
        self._error_log.write(line)

    def log_error_raw(self, message: str) -> None:
        """Log a raw error message without requiring an agent reference."""
        self._error_log.write(f"[ERROR] {_sanitise(message, 1000)}")

    # ── Model-level logging ───────────────────────────────────────────────────

    def log_model_state(self, step: int, **fields) -> None:
        """
        Write one model-state line to the single run-level model log file.
        All values sanitised — JSON fields have semicolons escaped.
        """
        parts = [f"[Step {step}]", "[MODEL]"]
        for k, v in fields.items():
            # JSON values (dicts/lists) get compact JSON encoding with ; escaped
            if isinstance(v, (dict, list)):
                parts.append(f"{k}={_json_field(v)}")
            else:
                parts.append(f"{k}={_sanitise(v, 300)}")
        self._model_log.write(" ".join(parts))

    def log_model_init(self, message: str) -> None:
        """
        Write a run-initialisation line. Higher char limit to accommodate
        PARAMS: JSON which can be several hundred characters.
        """
        self._model_log.write(f"[INIT] {_sanitise(message, 2000)}")
        
    # def log_model_state(self, step: int, **fields) -> None:
    #     kv = " ".join(f"{k}={v} " for k, v in fields.items())
    #     self._model_log.write(f"[Step {step}] [MODEL] {kv}")

    def log_model_init(self, message: str) -> None:
        self._model_log.write(f"[INIT] {_sanitise(message, 500)}")

    # ── Narrative debug logging ───────────────────────────────────────────────
    def log_narrative(self, agent_id: Any, text: str) -> None:
        self._narrative_log.write(f"Agent {agent_id}: {_sanitise(text, 300)}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def flush_all(self) -> None:
        for log in self._step_logs.values():
            log.flush()
        self._error_log.flush()
        self._model_log.flush()
        self._narrative_log.flush()

    def close(self) -> None:
        """Flush and close every open handle. Call at the end of each run."""
        for log in self._step_logs.values():
            log.close()
        self._step_logs.clear()
        self._error_log.close()
        self._model_log.close()
        self._narrative_log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
