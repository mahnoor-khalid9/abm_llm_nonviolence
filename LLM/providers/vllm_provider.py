"""
vllm_provider.py — Local vLLM server provider for Unity cluster.

Architecture
------------
vLLM runs any compatible model as a background process exposing an
OpenAI-compatible REST API on localhost:8000.  This provider
calls that endpoint exactly like the OpenAI provider does —
zero code changes to the simulation.

Why vLLM over raw transformers
------------------------------
- Continuous batching: parallel activist calls within one step
- PagedAttention: 2-4x memory efficiency vs HuggingFace generate()
- Guided JSON decoding via outlines: output always valid JSON (optional)
- ~2-4s per 8B response (vs 15-30s raw transformers)
- 4-bit AWQ/GPTQ quantization: 8B fits on a single V100 16GB

Setup on Unity
--------------
    pip install vllm
    # Start server (handled by SLURM script):
    python -m vllm.entrypoints.openai.api_server \
        --model mistralai/Ministral-3-8B-Instruct \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --port 8000

Usage
-----
    from providers.vllm_provider import VLLMProvider
    provider = VLLMProvider(
        model="mistralai/Ministral-3-8B-Instruct",
        prompt_dir="./Prompts",
        prompt_type="ZeroShot",
    )
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import time
from typing import Optional, Type
import requests
from LLM.base import BaseProvider
from LLM.schemas import LLMResponse

class VLLMProvider(BaseProvider):
    """
    Calls a local vLLM server via its OpenAI-compatible /v1/chat/completions
    endpoint. Auto-discovers the port and model name from the server.

    Parameters
    ----------
    base_url     : vLLM server URL. None = auto-scan ports.
    port_start   : Start of port scan range.
    port_end     : End of port scan range.
    prompt_dir   : Directory containing prompt markdown files.
    prompt_type  : Subdirectory under prompt_dir (ZeroShot / CoT).
    temperature  : Sampling temperature. 0.0 = greedy.
    max_tokens   : Max output tokens.
    timeout_s    : Per-request timeout in seconds.
    supports_thinking : Whether model supports <think> tags (Qwen3 only).
    use_guided_json   : Whether to enable guided JSON decoding.
    """

    def __init__(
        self,
        base_url:          Optional[str] = None,
        port_start:        int           = 8000,
        port_end:          int           = 8020,
        prompt_dir:        str           = "./Prompts",
        prompt_type:       str           = "ZeroShot",
        temperature:       float         = 0.0,
        max_tokens:        int           = 1024,
        timeout_s:         int           = 300,
        supports_thinking: bool          = False,
        use_guided_json:   bool          = False,
        **kwargs,
    ):
        self.model             = None   # resolved from server
        self.base_url          = None   # resolved from server or provided
        self.temperature       = temperature
        self.max_tokens        = max_tokens
        self.timeout_s         = timeout_s
        self.supports_thinking = supports_thinking
        self.use_guided_json   = use_guided_json
        self._last_token_count = 0

        # Step 1: resolve base_url
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = self._scan_for_server(port_start, port_end)

        # Step 2: resolve model name from server
        self.model = self._resolve_model_name()

        # Step 3: auto-detect thinking support from model name
        if "qwen" in self.model.lower():
            self.supports_thinking = True
            print(f"[VLLMProvider] Qwen model detected - enabling thinking mode", flush=True)

        # Step 4: load prompts
        self._load_prompts(prompt_dir=prompt_dir, prompt_type=prompt_type)

        print(
            f"[VLLMProvider] Ready — "
            f"url={self.base_url}  model={self.model}  "
            f"prompt_type={prompt_type}  thinking={self.supports_thinking}",
            flush=True,
        )
        
    # ── Port discovery ────────────────────────────────────────────────────────

    def _scan_for_server(
        self,
        start_port: int,
        end_port:   int,
    ) -> str:
        """
        Scan ports until one responds to /v1/models with HTTP 200.
        Returns base_url of the first live vLLM server found.
        """
        print(
            f"[VLLMProvider] Scanning ports {start_port}–{end_port} for vLLM...",
            flush=True,
        )
        for port in range(start_port, end_port + 1):
            url = f"http://localhost:{port}"
            try:
                r = requests.get(f"{url}/v1/models", timeout=2)
                if r.status_code == 200:
                    print(f"[VLLMProvider] Found server at {url}", flush=True)
                    return url
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout):
                continue

        raise RuntimeError(
            f"[VLLMProvider] No vLLM server found on ports {start_port}–{end_port}. "
            f"Is the server running?"
        )

    # ── Model name resolution ─────────────────────────────────────────────────

    def _resolve_model_name(self) -> str:
        """
        Fetch the exact model id registered on the server.
        This must match what we send in the payload — vLLM 404s on mismatch.
        """
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            r.raise_for_status()
            models = r.json().get("data", [])
            if not models:
                raise RuntimeError(
                    f"[VLLMProvider] /v1/models returned empty list at {self.base_url}"
                )
            model_id = models[0]["id"]
            print(f"[VLLMProvider] Resolved model: {model_id}", flush=True)
            return model_id
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"[VLLMProvider] Failed to resolve model name from "
                f"{self.base_url}/v1/models: {e}"
            )

    # ── Core API call ─────────────────────────────────────────────────────────

    def _extract_thinking(self, raw: str) -> LLMResponse:
        """
        Split output into reasoning trace and JSON payload.
        Handles models with and without <think> tags gracefully.
        """
        thinking = None
        content  = raw

        # Only try to extract thinking if model supports it
        if self.supports_thinking:
            think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                # Remove the <think> block to isolate the JSON
                content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip accidental markdown fences (common in non-guided outputs)
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$",          "", content)
        content = content.strip()

        return LLMResponse(thinking=thinking, content=content)


    def _call_api(self, prompt: str, response_model: Type, **kwargs) -> LLMResponse:
        """
        Call vLLM server with robust error handling for different model types.
        Automatically handles thinking/non-thinking models and guided JSON.
        """
        schema = response_model.model_json_schema()

        # Base payload - compatible with all models
        payload = {
            "model":       self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }

        # Add guided JSON if enabled and supported
        if self.use_guided_json:
            payload["guided_json"] = schema
            payload["guided_decoding_backend"] = "outlines"

        # Add thinking support for Qwen models
        if self.supports_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            
            # If 400 error, log the actual error message from server
            if response.status_code == 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = {"error": response.text or "No detail available"}
                
                print(
                    f"\n[VLLMProvider] ═══ 400 Bad Request ═══",
                    flush=True,
                )
                print(f"Server error: {error_detail}", flush=True)
                print(f"\nPayload sent:", flush=True)
                for key, value in payload.items():
                    if key == "messages":
                        print(f"  {key}: [system: {len(value[0]['content'])} chars, user: {len(value[1]['content'])} chars]", flush=True)
                    else:
                        print(f"  {key}: {value}", flush=True)
                print("═" * 50 + "\n", flush=True)
            
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            print(
                f"[VLLMProvider] Request failed: {e}",
                flush=True,
            )
            raise

        # Extract token usage
        self._last_token_count = (
            data.get("usage", {}).get("completion_tokens", 0)
        )

        # Get raw response
        raw = data["choices"][0]["message"]["content"]

        return self._extract_thinking(raw)