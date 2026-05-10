"""
chat_model.py — Public factory. The simulation only ever imports from here.

Usage
-----
from chat_model import build_provider

provider = build_provider("gemini")                    # or "openai", "ollama", "huggingface"
output   = provider.ask_activist_position(context, memory, use_memory=True)
"""

from __future__ import annotations

import os
import sys

if __package__ is None and __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from typing import Any, Optional
from LLM.base import BaseProvider
from LLM.schemas import GandhianMovementOutputZeroShot, GandhianMovementOutputCoT, LLMResponse

def build_provider(
    provider: str,
    prompt_dir: str = "./Prompts",
    prompt_type: str = "ZeroShot",
    use_memory: bool = True,
    **kwargs: Any,
) -> BaseProvider:
    """
    Factory function — returns a fully initialised provider instance.

    Parameters
    ----------
    provider    : "gemini" | "openai" | "ollama" | "huggingface"
    prompt_dir  : directory containing the .md prompt files
    use_memory  : passed through to ask_activist_position(); stored here for
                  convenience so callers don't have to repeat it every call
    **kwargs    : forwarded to the provider constructor (model name, temperature, etc.)

    Returns
    -------
    A BaseProvider subclass with .ask_activist_position() ready to call.
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        from LLM.providers.gemini import GeminiProvider
        instance = GeminiProvider(prompt_dir=prompt_dir, prompt_type=prompt_type, **kwargs)

    elif provider == "openai":
        from LLM.providers.openai_provider import OpenAIProvider
        instance = OpenAIProvider(prompt_dir=prompt_dir, **kwargs)

    elif provider == "ollama":
        from LLM.providers.ollama import OllamaProvider
        instance = OllamaProvider(prompt_dir=prompt_dir, **kwargs)

    elif provider == "huggingface":
        from LLM.providers.huggingface import HuggingFaceProvider
        instance = HuggingFaceProvider(prompt_dir=prompt_dir, **kwargs)

    # LLM/chat_model.py — add to build_provider()
    elif provider == "vllm":
        from LLM.providers.vllm_provider import VLLMProvider
        instance = VLLMProvider(prompt_dir=prompt_dir, prompt_type=prompt_type, **kwargs)


    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Choose from: gemini, openai, ollama, huggingface, vllm"
        )

    # Attach use_memory so callers can read it back if needed
    instance._use_memory = use_memory
    return instance


# ── Convenience wrapper ───────────────────────────────────────────────────────

class ChatModel:
    """
    Thin wrapper around build_provider() that preserves the original
    `ChatModel.ask_activist_position(context, memory)` call signature.
    Drop-in replacement for the old monolithic class.
    """

    def __init__(
        self,
        provider: str = "gemini",
        use_memory: bool = True,
        prompt_dir: str = "./Prompts",
        prompt_type: str = "ZeroShot",
        **kwargs: Any,
    ):
        self.use_memory = use_memory
        self._provider  = build_provider(
            provider=provider,
            prompt_dir=prompt_dir,
            prompt_type=prompt_type,
            use_memory=use_memory,
            **kwargs,
        )


    def ask_activist_position(
        self,
        activist_local_context: dict,
        activist_memory: Optional[dict] = None,
        previous_error: Optional[str] = None,
    ):
        return self._provider.ask_activist_position(
            activist_local_context=activist_local_context,
            activist_memory=activist_memory,
            use_memory=self.use_memory,
            previous_error=previous_error,
        )

    def fill_self_reflection_prompt(self, agent_output, activist_local_context: dict) -> str:
        return self._provider.fill_self_reflection_prompt(agent_output, activist_local_context)

    def ask_llm(self, prompt: str, response_model=None, **kwargs) -> str:
        """Direct LLM call — for self-reflection or other non-standard prompts."""
        rm = response_model or GandhianMovementOutputZeroShot
        return self._provider._call_api(prompt=prompt, response_model=rm, **kwargs)
