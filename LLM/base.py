"""
base.py — Abstract base class for all LLM providers.
Holds shared utilities: prompt rendering, template loading, position validation,
and the public ask_activist_position() interface every provider exposes.
"""

from __future__ import annotations

import os
import re
import sys
import json

if __package__ is None and __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
from jinja2 import Template

from LLM.schemas import (
    GandhianMovementOutputZeroShot,
    GandhianMovementOutputCoT,
    LLMResponse
)



class BaseProvider(ABC):
    """
    Every provider (Gemini, OpenAI, Ollama, HuggingFace) inherits from this.

    Subclasses must implement:
        _call_api(prompt, response_model, **kwargs) -> str   (raw JSON string)
    """

    # ── Subclass contract ────────────────────────────────────────────────────

    @abstractmethod
    def _call_api(
        self,
        prompt: str,
        response_model: Type,
        **kwargs,
    ) -> str:
        """Call the underlying model and return a raw JSON string."""
        self._last_token_count: int = 0

    # ── Shared setup helpers ─────────────────────────────────────────────────

    def _load_prompts(self, prompt_dir: str = "./Prompts", prompt_type: str = "ZeroShot") -> None:
        """Read all prompt markdown files into instance attributes."""
        self.prompt_type = prompt_type
        self.system_prompt          = self._read_file(f"{prompt_dir}/{prompt_type}/system_prompt.md")
        self.user_prompt            = self._read_file(f"{prompt_dir}/{prompt_type}/user_prompt_w_memory.md")
        self.user_prompt_no_memory  = self._read_file(f"{prompt_dir}/{prompt_type}/user_prompt_wo_memory.md")
        self.temperature = 0.0
        self.top_p = 1.0


    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ── Prompt rendering ─────────────────────────────────────────────────────

    def render_template(
        self,
        prompt: str,
        activist_context: dict,
        activist_memory: Optional[dict] = None,
    ) -> str:
        data = {
            "ACTIVIST_LOCAL_CONTEXT": activist_context,
            "SELF_REFLECTION": (
                activist_memory.get("cognitive_memory", {}).get("self_reflection", "")
                if activist_memory else ""
            ),
            "RECENT_EVENTS": (
                activist_memory.get("cognitive_memory", {}).get("environmental_observation", "")
                if activist_memory else ""
            ),
        }
        return Template(prompt).render(**data)

    def fill_user_prompt(
        self,
        activist_local_context: dict,
        activist_memory: Optional[dict],
        use_memory: bool,
    ) -> str:
        try:
            if use_memory:
                return self.render_template(
                    prompt=self.user_prompt,
                    activist_context=activist_local_context,
                    activist_memory=activist_memory,
                )
            return self.render_template(
                prompt=self.user_prompt_no_memory,
                activist_context=activist_local_context,
                activist_memory=None,
            )
        except Exception as e:
            print(f"Error rendering template: {e}")
            return ""

    def fill_self_reflection_prompt(
        self,
        agent_output: GandhianMovementOutputCoT,
        activist_local_context: dict,
    ) -> str:
        data = {
            "FULL_AGENT_OUTPUT":      agent_output.model_dump_json(indent=2),
            "ACTIVIST_LOCAL_CONTEXT": activist_local_context,
        }
        return Template(self.self_reflection).render(**data)

    def append_error_feedback(
        self,
        base_prompt: str,
        error_message: Optional[str] = None,
    ) -> str:
        if not error_message:
            return base_prompt
        return base_prompt + f"""
            <correction>
            The previous response was INVALID. Fix the error below and return only valid JSON.

            Error:
            {error_message}
            </correction>
        """

    # ── Position validation ───────────────────────────────────────────────────

    @staticmethod
    def _available_positions(activist_local_context: dict):
        return [
            tuple(pos)
            for d in activist_local_context["neighborhood"].values()
            for pos in d.get("can_move_to", [])
        ]

    def zershot_parse_decision_xml(self, xml_str: str) -> dict:
        """
        Parse the <decision> XML output from the LLM into a JSON-compatible dict.
        Handles malformed whitespace, missing tags, and list fields like chosen_position.
        Returns dict with top-level "decision" key.
        """
        xml_str = xml_str.strip()

        # Extract content between <decision> tags if present
        decision_match = re.search(r"<decision>(.*?)</decision>", xml_str, re.DOTALL)
        if decision_match:
            inner = decision_match.group(1)
        else:
            inner = xml_str

        # Extract all tag-value pairs
        fields = re.findall(r"<(\w+)>(.*?)</\1>", inner, re.DOTALL)
        result = {}
        for tag, value in fields:
            value = value.strip()
            if tag == "chosen_position":
                nums = re.findall(r"-?\d+", value)
                result[tag] = [int(n) for n in nums]
            else:
                result[tag] = value

        return {"decision": result}

    def xml_to_json(self, xml_str: str) -> str:
        """Return compact JSON string from decision XML."""
        if self.prompt_type == "ZeroShot":
            return json.dumps(self.zershot_parse_decision_xml(xml_str))
        else:
            pass
    
    def ask_activist_position(
        self,
        activist_local_context: dict,
        activist_memory: Optional[dict] = None,
        use_memory: bool = True,
        previous_error: Optional[str] = None,
    ) -> Union[GandhianMovementOutputZeroShot, GandhianMovementOutputCoT]:
        """
        Build the prompt, call the model, validate the position, return output.
        Retries up to 3 times — each retry appends the error to the prompt
        so the model knows exactly what it did wrong.
        """
        if self.prompt_type == "ZeroShot":
            response_model = GandhianMovementOutputZeroShot
        else:
            response_model = GandhianMovementOutputCoT

        user_prompt = self.fill_user_prompt(
            activist_local_context=activist_local_context,
            activist_memory=activist_memory if use_memory else None,
            use_memory=use_memory,
        )

        current_prompt = user_prompt
        # error related to incorrect response generated
        if previous_error is not None:
            current_prompt = self.append_error_feedback(
                base_prompt=user_prompt,
                error_message=previous_error,
            )

        last_error = None
        for attempt in range(3):

            if last_error is not None:
                combined = "\n".join(filter(None, [previous_error, last_error]))
                current_prompt = self.append_error_feedback(user_prompt, combined)

            try:
                llm_response = self._call_api(
                    prompt=current_prompt,
                    response_model=response_model,
                )
                
                # print(llm_response.content, llm_response.thinking)
                
                # Guard: wrap plain-string responses from non-vLLM providers
                if not isinstance(llm_response, LLMResponse):
                    llm_response = LLMResponse(thinking=None, content=llm_response)

                try:
                    agent_output = response_model.model_validate_json(
                        llm_response.content
                    )
                except Exception as parse_err:
                    try:
                        agent_output = response_model.model_validate_json(
                            self.xml_to_json(llm_response.content)
                        )
                    except Exception:
                        raise parse_err

                # ── Attach metadata ───────────────────────────────────────────
                agent_output._response_tokens = self._last_token_count
                agent_output._reasoning        = llm_response.thinking  

                return agent_output

            except Exception as exc:
                last_error = str(exc)
                if attempt < 2:
                    print(
                        f"[BaseProvider] attempt {attempt + 1}/3 failed: "
                        f"{exc.__class__.__name__}: {str(exc)[:120]}  — retrying",
                        flush=True,
                    )
                else:
                    print(
                        f"[BaseProvider] all 3 attempts failed. "
                        f"Last error: {exc.__class__.__name__}: {str(exc)[:200]}",
                        flush=True,
                    )
                    raise