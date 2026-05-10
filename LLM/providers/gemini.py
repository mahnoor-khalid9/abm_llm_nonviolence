"""
providers/gemini.py — Google Gemini provider.
"""

from __future__ import annotations
import os
from typing import Type
import google.genai as genai
from google.genai import types
from LLM.base import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Wraps Google Gemini (gemini-2.5-flash by default).
    Supports structured JSON output via response_json_schema.
    """

    DEFAULT_MODEL = "gemini-3.1-flash-lite"
    CANDIDATE_MODELS = [
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.0-flash-exp",
        "gemini-2.5-pro",
    ]

    def __init__(
        self,
        prompt_dir: str = "./Prompts",
        prompt_type: str = "ZeroShot",
        model: str | None = None,
        temperature: float = 0.0,
        thinking_budget: int = -1,   # -1 = enabled (2.5 Pro), 0 = disabled
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GEMINI_API_KEY environment variable.")

        self.client          = genai.Client(api_key=api_key)
        self.active_model    = model or self.DEFAULT_MODEL
        self.temperature     = temperature
        self.thinking_budget = thinking_budget
        self.prompt_type     = prompt_type
        self._load_prompts()

    # ── BaseProvider contract ────────────────────────────────────────────────

    def _call_api(self, prompt: str, response_model: Type, **kwargs) -> str:
        response = self.client.models.generate_content(
            model=self.active_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=self.system_prompt,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=self.thinking_budget,
                ),
                response_mime_type="application/json",
                response_json_schema=response_model.model_json_schema(),
            ),
        )
        self._last_token_count = getattr(
            getattr(response, "usage_metadata", None),
            "candidates_token_count", 0
        ) or 0
        return response.text.strip()
