"""
vllm_provider.py — Local vLLM server provider for Unity cluster.

Architecture
------------
vLLM runs Llama 3.1 8B as a background process exposing an
OpenAI-compatible REST API on localhost:8000.  This provider
calls that endpoint exactly like the OpenAI provider does —
zero code changes to the simulation.

Why vLLM over raw transformers
------------------------------
- Continuous batching: parallel activist calls within one step
- PagedAttention: 2-4x memory efficiency vs HuggingFace generate()
- Guided JSON decoding via outlines: output always valid JSON
- ~2-4s per 8B response (vs 15-30s raw transformers)
- 4-bit AWQ/GPTQ quantization: 8B fits on a single V100 16GB

Setup on Unity
--------------
    pip install vllm
    # Start server (handled by SLURM script):
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --quantization awq \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --port 8000

Usage
-----
    from providers.vllm_provider import VLLMProvider
    provider = VLLMProvider(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        prompt_dir="./Prompts",
        prompt_type="ZeroShot",
    )
"""

from __future__ import annotations

import json
import time
import os
from typing import Any, Optional, Type

import requests

from LLM.base import BaseProvider
from LLM.schemas import GandhianMovementOutputZeroShot


class VLLMProvider(BaseProvider):
    """
    Calls a local vLLM server via its OpenAI-compatible /v1/chat/completions
    endpoint.  The server must be running before the simulation starts —
    the SLURM script handles this.

    Parameters
    ----------
    model        : HuggingFace model ID or local path.
    base_url     : vLLM server URL (default localhost:8000).
    prompt_dir   : Directory containing prompt markdown files.
    prompt_type  : Subdirectory under prompt_dir (ZeroShot / CoT).
    temperature  : Sampling temperature. 0.0 = greedy (fastest, most consistent).
    max_tokens   : Max output tokens. 150 is sufficient for ZeroShot 5-field output.
    timeout_s    : Per-request timeout in seconds.
    """

    def __init__(
        self,
        model:       str  = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        base_url:    str  = "http://localhost:8000",
        prompt_dir:  str  = "./Prompts",
        prompt_type: str  = "ZeroShot",
        temperature: float = 0.0,
        max_tokens:  int   = 150,
        timeout_s:   int   = 30,
        **kwargs,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout_s   = timeout_s
        self._last_token_count = 0

        self._load_prompts(prompt_dir=prompt_dir, prompt_type=prompt_type)
        self._wait_for_server()

    # ── Server readiness ──────────────────────────────────────────────────────

    def _wait_for_server(self, max_wait: int = 300, poll_interval: int = 5) -> None:
        """
        Block until the vLLM server is ready.
        Raises RuntimeError if server does not come up within max_wait seconds.
        Called once at provider init — SLURM script starts the server first.
        """
        health_url = f"{self.base_url}/health"
        deadline   = time.time() + max_wait
        print(f"[VLLMProvider] Waiting for server at {self.base_url} ...", flush=True)
        while time.time() < deadline:
            try:
                r = requests.get(health_url, timeout=3)
                if r.status_code == 200:
                    print(f"[VLLMProvider] Server ready.", flush=True)
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(poll_interval)
        raise RuntimeError(
            f"[VLLMProvider] Server at {self.base_url} did not respond "
            f"within {max_wait}s. Check that vLLM started correctly."
        )

    # ── Core API call ─────────────────────────────────────────────────────────

    def _call_api(
        self,
        prompt: str,
        response_model: Type,
        **kwargs,
    ) -> str:
        """
        Call vLLM via OpenAI-compatible chat completions with guided JSON.

        Guided JSON decoding (via vLLM's --guided-decoding-backend outlines)
        constrains the output to exactly match the Pydantic model's JSON schema.
        This eliminates invalid position errors and parsing failures.
        """
        schema = response_model.model_json_schema()

        payload = {
            "model":       self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "temperature":    self.temperature,
            "max_tokens":     self.max_tokens,
            # vLLM guided JSON — output is always valid per the schema
            "guided_json":    schema,
            "guided_decoding_backend": "outlines",
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()

        # Token count from vLLM usage metadata
        self._last_token_count = (
            data.get("usage", {}).get("completion_tokens", 0)
        )

        return data["choices"][0]["message"]["content"]
