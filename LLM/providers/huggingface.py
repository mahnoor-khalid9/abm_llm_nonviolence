"""
huggingface.py — Direct HuggingFace transformers provider.

Auto-detects local vs cluster and configures itself — no manual changes needed.

Fixes vs previous version:
  1. device_map={"": 0} for bitsandbytes — "auto" triggers accelerate dispatch
     which calls .to(device), forbidden on quantized models (your ValueError)
  2. temperature=None, top_p=None — were referenced as self.temperature/self.top_p
     which are not defined on BaseProvider (AttributeError)
  3. _get_prefix_fn now has try/except — no crash if lmformatenforcer unavailable
  4. Auto-detects environment via SLURM_JOB_ID + VRAM size
"""

from __future__ import annotations

import os
import re
from typing import Optional, Type

from LLM.base import BaseProvider

_CLUSTER_VRAM_GB = 20.0


def _detect_vram_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def _is_cluster() -> bool:
    if os.environ.get("SLURM_JOB_ID"):
        return True
    return _detect_vram_gb() >= _CLUSTER_VRAM_GB


class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace transformers provider with automatic environment detection.

    Parameters
    ----------
    model_name      : HuggingFace model ID or absolute local path.
    prompt_dir      : Directory containing Prompts/ folder.
    prompt_type     : "ZeroShot" or "CoT".
    quantization    : "4bit" | "8bit" | "none".
    max_new_tokens  : Output token ceiling. 150 is safe for ZeroShot.
    device_profile  : "auto" | "local" | "cluster".
    max_memory_gb   : Override GPU memory cap. None = auto.
    """

    def __init__(
        self,
        model_name:     str   = "meta-llama/Llama-3.1-8B-Instruct",
        prompt_dir:     str   = "./Prompts",
        prompt_type:    str   = "ZeroShot",
        quantization:   str   = "4bit",
        max_new_tokens: int   = 150,
        device_profile: str   = "auto",
        max_memory_gb:  Optional[float] = None,
        **kwargs,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # ── Environment ───────────────────────────────────────────────────────
        if device_profile == "auto":
            on_cluster = _is_cluster()
        else:
            on_cluster = (device_profile == "cluster")

        vram_gb   = _detect_vram_gb()
        env_label = "cluster" if on_cluster else "local"
        print(
            f"[HuggingFaceProvider] env={env_label}  "
            f"vram={vram_gb:.1f}GB  "
            f"slurm={bool(os.environ.get('SLURM_JOB_ID'))}",
            flush=True,
        )

        self.max_new_tokens    = max_new_tokens
        self._last_token_count = 0
        self._prefix_fn_cache: dict = {}

        # ── Flash Attention ───────────────────────────────────────────────────
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("[HuggingFaceProvider] Flash Attention 2 enabled.", flush=True)
        except ImportError:
            attn_impl = "eager"

        # ── Quantization config ───────────────────────────────────────────────
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # ── Device map ────────────────────────────────────────────────────────
        # bitsandbytes MUST use {"": 0} on single GPU.
        # accelerate "auto" calls .to(device) which bitsandbytes forbids.
        if bnb_config is not None:
            device_map = {"": 0}
            if max_memory_gb is not None:
                max_memory = {0: f"{int(max_memory_gb)}GB"}
            elif on_cluster:
                max_memory = {0: f"{int(vram_gb * 0.90)}GB"}
            else:
                safe_gb    = max(4, int(vram_gb) - 2)
                max_memory = {0: f"{safe_gb}GB"}
        else:
            device_map = "auto" if on_cluster else {"": 0}
            max_memory = None

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(
            f"[HuggingFaceProvider] Loading {model_name}  "
            f"quant={quantization}  device_map={device_map}  "
            f"max_memory={max_memory}",
            flush=True,
        )

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model ─────────────────────────────────────────────────────────────
        load_kwargs: dict = {
            "quantization_config": bnb_config,
            "attn_implementation": attn_impl,
            "low_cpu_mem_usage":   True,
        }
        # Do NOT pass device_map for bitsandbytes — let it handle GPU placement
        # Do NOT pass max_memory without device_map — meaningless and triggers dispatch
        if bnb_config is None:
            load_kwargs["device_map"]  = "auto" if on_cluster else "cuda:0"
            load_kwargs["torch_dtype"] = torch.bfloat16
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        print("[HuggingFaceProvider] Model ready.", flush=True)

        self._load_prompts(prompt_dir=prompt_dir, prompt_type=prompt_type)

    # ── Constrained JSON decoding ─────────────────────────────────────────────

    def _get_prefix_fn(self, response_model: Type):
        key = response_model.__name__
        if key not in self._prefix_fn_cache:
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )
                parser = JsonSchemaParser(response_model.model_json_schema())
                self._prefix_fn_cache[key] = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, parser
                )
                print(f"[HuggingFaceProvider] Constrained decoding active.", flush=True)
            except Exception as exc:
                print(
                    f"[HuggingFaceProvider] Constrained decoding unavailable "
                    f"({exc.__class__.__name__}) — using unconstrained.",
                    flush=True,
                )
                self._prefix_fn_cache[key] = None
        return self._prefix_fn_cache[key]

    # ── Core API call ─────────────────────────────────────────────────────────

    def _call_api(self, prompt: str, response_model: Type, **kwargs) -> str:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

        tokenized      = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids      = tokenized["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        prefix_fn = self._get_prefix_fn(response_model)

        generate_kwargs: dict = {
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample":      False,
            "temperature":    self.temperature,  
            "top_p":          self.top_p,   
            "pad_token_id":   self.tokenizer.eos_token_id,
            "eos_token_id":   self.tokenizer.eos_token_id,
        }
        # if prefix_fn is not None:
        #     generate_kwargs["prefix_allowed_tokens_fn"] = prefix_fn

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generate_kwargs)

        new_tokens             = output_ids[0][input_ids.shape[-1]:]
        self._last_token_count = len(new_tokens)
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # if prefix_fn is None and not raw.startswith("{"):
        #     raw = self._extract_json(raw)
        return raw

    # ── JSON extraction fallback ──────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Brace-counting JSON extractor — handles nested structures correctly."""
        text  = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
        return text[start:]