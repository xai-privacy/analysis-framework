"""Model adapters.

Every adapter implements the same `generate` signature so the local-SLM and
frontier-API cells of the 2x2 are interchangeable. Nothing here is
Colab-specific: moving to another host is a config change, not a rewrite.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

_MODEL_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_configs"
)

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@runtime_checkable
class ModelAdapter(Protocol):
    def generate(self, prompt: str, *, seed: int, config: dict) -> str: ...


def load_model_config(model_id: str) -> dict:
    """Load model-specific config from model_configs/<sanitized_model_id>.json.

    Mirrors run_benchmark.py: the file name is the model id with "/" replaced by
    "_", falling back to the Llama config when no model-specific file exists.
    """
    sanitized = model_id.replace("/", "_")
    config_path = os.path.join(_MODEL_CONFIGS_DIR, f"{sanitized}.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    fallback_path = os.path.join(_MODEL_CONFIGS_DIR, f"{DEFAULT_MODEL.replace('/', '_')}.json")
    with open(fallback_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class HFAdapter:
    """Local Hugging Face causal LM.

    The model is loaded once and reused across trials; the seed is applied per
    generation so that the five trials of a record differ only by seed.
    """

    def __init__(self, model_id: str, config: Optional[dict] = None, system_prompt: str = ""):
        import torch  # imported here so the package is importable without torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.config = config or load_model_config(model_id)
        self.system_prompt = system_prompt

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        dtype_str = self.config.get("torch_dtype", "float16")
        torch_dtype = (
            torch.float16
            if dtype_str == "float16"
            else torch.bfloat16
            if dtype_str == "bfloat16"
            else "auto"
        )
        trust_remote_code = self.config.get("trust_remote_code", False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, *, seed: int, config: dict) -> str:
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        generation_kwargs = dict(config)
        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs,
                )
        except Exception as exc:  # a failed generation is a datapoint, not a crash
            print(f"[Generation Error]: {exc}", file=sys.stderr)
            return ""

        return self.tokenizer.decode(
            output_tokens[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


class APIAdapter:
    """Frontier model over an HTTP API.

    Stub with the same signature as `HFAdapter` so the frontier cell of the 2x2
    is a drop-in later. There is no API key available for this run.
    """

    def __init__(self, model_id: str, config: Optional[dict] = None, system_prompt: str = ""):
        self.model_id = model_id
        self.config = config or {}
        self.system_prompt = system_prompt

    def generate(self, prompt: str, *, seed: int, config: dict) -> str:
        raise NotImplementedError(
            "APIAdapter is a stub. The frontier-model cell of the 2x2 is a separate "
            "task and no API key is available for this run."
        )


class StubAdapter:
    """Deterministic canned outputs, for smoke-testing the harness without a GPU.

    Not a baseline and not a model: it exists so the six-stage pipeline can be
    exercised end to end on a machine with no torch installed. It cycles through
    a well-formed answer, a prose answer, a self-contradicting answer, a refusal
    and an empty string, so that every parse status is reachable.
    """

    _CANNED = (
        "ANSWER: 3",
        "I think the correct option is 2 given the passage.",
        "ANSWER: 1. On reflection, ANSWER: 4 is the better reading.",
        "I am not able to determine which of these evaluations applies here.",
        "",
    )

    def __init__(self, model_id: str = "stub", config: Optional[dict] = None, system_prompt: str = ""):
        self.model_id = model_id
        self.config = config or {}
        self.system_prompt = system_prompt

    def generate(self, prompt: str, *, seed: int, config: dict) -> str:
        index = (hash((len(prompt), seed)) % len(self._CANNED) + len(self._CANNED)) % len(self._CANNED)
        return self._CANNED[index]


def build_adapter(kind: str, model_id: str, config: Optional[dict] = None, system_prompt: str = "") -> ModelAdapter:
    if kind == "hf":
        return HFAdapter(model_id, config=config, system_prompt=system_prompt)
    if kind == "api":
        return APIAdapter(model_id, config=config, system_prompt=system_prompt)
    if kind == "stub":
        return StubAdapter(model_id, config=config, system_prompt=system_prompt)
    raise ValueError(f"unknown adapter {kind!r}; expected one of hf, api, stub")
