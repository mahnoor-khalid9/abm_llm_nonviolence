"""
providers/ollama.py — Ollama local provider.
Talks to the Ollama REST API running on localhost.
"""

from __future__ import annotations
import json
from typing import Type
import requests
from LLM.base import BaseProvider
from LLM.schemas import GandhianMovementOutput


class OllamaProvider(BaseProvider):
    """
    Wraps a locally running Ollama instance.
    Uses /api/chat with structured JSON output (Ollama format: field).
    """

    DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        prompt_dir: str = "./Prompts",
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: int = 1000,
    ):
        self.active_model = model or self.DEFAULT_MODEL
        self.base_url     = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.temperature  = temperature
        self.timeout      = timeout

        self._load_prompts(prompt_dir)

    # ── BaseProvider contract ────────────────────────────────────────────────

    def _call_api(self, prompt: str, response_model: Type, **kwargs) -> str:
        payload = {
            "model": self.active_model,
            "messages": [{"role": "user", "content": prompt}],
            "format": response_model.model_json_schema(),
            "options": {"temperature": self.temperature},
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        data = response.json()

        if "message" not in data:
            raise RuntimeError(
                f"Unexpected Ollama response (no 'message' key):\n{json.dumps(data, indent=2)}"
            )
        message = data["message"]
        if "content" not in message:
            raise RuntimeError(
                f"Ollama message missing 'content':\n{json.dumps(message, indent=2)}"
            )

        content = message["content"]
        if isinstance(content, str):
            content = json.loads(content)

        # Merge in thinking trace if present (some models support it)
        result = {**content, "thinking": message.get("thinking")}
        return json.dumps(result)
