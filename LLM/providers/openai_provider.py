"""
providers/openai_provider.py — OpenAI provider.
"""

from __future__ import annotations
import os
import time
from typing import Type
from openai import OpenAI, BadRequestError
from LLM.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    Wraps OpenAI chat completions (gpt-4o by default).
    Includes model fallback logic and warmup-wait on pending deploys.
    """

    DEFAULT_MODEL = "gpt-4o"
    CANDIDATE_MODELS = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(
        self,
        prompt_dir: str = "./Prompts",
        model: str | None = None,
        temperature: float = 0.0,
        retries: int = 3,
        warmup_wait: int = 15,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

        self.client       = OpenAI(api_key=api_key)
        self.active_model = model or self.DEFAULT_MODEL
        self.temperature  = temperature
        self.retries      = retries
        self.warmup_wait  = warmup_wait
        self._candidates  = [self.active_model] + [
            m for m in self.CANDIDATE_MODELS if m != self.active_model
        ]

        self._load_prompts(prompt_dir)

    # ── BaseProvider contract ────────────────────────────────────────────────

    def _call_api(self, prompt: str, response_model: Type, **kwargs) -> str:
        for attempt in range(self.retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.active_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                self._last_token_count = getattr(
                        getattr(completion, "usage", None),
                        "completion_tokens", 0
                    ) or 0
                return completion.choices[0].message.content

            except BadRequestError as e:
                msg = str(e)
                if "model_pending_deploy" in msg:
                    time.sleep(self.warmup_wait)
                    continue
                next_model = self._next_model()
                if next_model:
                    self.active_model = next_model
                    continue
                raise

        raise RuntimeError(f"OpenAI provider failed after {self.retries} retries.")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _next_model(self) -> str | None:
        try:
            idx = self._candidates.index(self.active_model)
            return self._candidates[idx + 1] if idx + 1 < len(self._candidates) else None
        except ValueError:
            return None
