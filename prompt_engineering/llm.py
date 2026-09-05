"""Ollama client wrapper plus answer parsing.

Everything that talks to a model goes through here, so token accounting and
decoding parameters are controlled in one place. That matters for this study:
CoT, self-consistency, ToT and GoT all spend very different amounts of compute,
and comparing their accuracy without reporting cost is misleading.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"
# The paper standardised max output tokens at 3000 (Section 4.1), but it evaluated
# frontier models over an API. A 3B model on CPU generates roughly 5-10 tokens/sec,
# so a 3000-token budget means 5-10 MINUTES per call and guarantees timeouts.
# 800 is enough for a reasoned answer on this task; raise it with --num-predict if
# the `truncated` count in the results is high.
DEFAULT_NUM_PREDICT = 800

# Keep the model resident between calls. Without this Ollama unloads it after 5
# minutes idle, and a slow run pays the reload cost on every single question.
DEFAULT_KEEP_ALIVE = "30m"


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_sec: float = 0.0
    #: Ollama's stop reason. "length" means the output budget ran out mid-sentence,
    #: which is the usual cause of an unparseable answer.
    done_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


@dataclass
class CallLog:
    """Cumulative compute accounting for one strategy run on one question."""
    n_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_sec: float = 0.0

    def add(self, r: LLMResponse):
        self.n_calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.output_tokens += r.output_tokens
        self.latency_sec += r.latency_sec

    def as_dict(self):
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            "latency_sec": round(self.latency_sec, 2),
        }


class OllamaLLM:
    def __init__(self, model: str, host: str = DEFAULT_HOST, timeout: int = 600,
                 num_predict: int = DEFAULT_NUM_PREDICT,
                 keep_alive: str = DEFAULT_KEEP_ALIVE):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_predict = num_predict
        self.keep_alive = keep_alive

    def generate(self, prompt: str, temperature: float = 0.0,
                 seed: Optional[int] = None,
                 num_predict: Optional[int] = None) -> LLMResponse:
        options = {"temperature": temperature,
                   "num_predict": self.num_predict if num_predict is None else num_predict}
        if seed is not None:
            options["seed"] = seed
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "keep_alive": self.keep_alive, "options": options},
                timeout=self.timeout,
            )
        except requests.exceptions.ReadTimeout:
            raise TimeoutError(
                f"No response after {self.timeout}s with num_predict="
                f"{options['num_predict']}. On CPU a 3B model runs at roughly 5-10 "
                f"tokens/sec, so this budget alone needs about "
                f"{options['num_predict'] // 8}s per call. Lower --num-predict, use a "
                f"smaller model, or raise --timeout."
            )
        if r.status_code == 404:
            raise RuntimeError(f"Model '{self.model}' not pulled. Run: ollama pull {self.model}")
        r.raise_for_status()
        d = r.json()
        return LLMResponse(
            text=d.get("response", ""),
            prompt_tokens=d.get("prompt_eval_count", 0),
            output_tokens=d.get("eval_count", 0),
            latency_sec=time.time() - t0,
            done_reason=d.get("done_reason", ""),
        )

    def measure_speed(self, num_predict: int = 48):
        """Short generation to gauge output speed, for runtime estimates.

        Also warms the model into memory, so the first real question doesn't pay
        the load cost and get mistaken for a slow prompt.
        """
        r = self.generate("Count from one to twenty in words.",
                          temperature=0.0, num_predict=num_predict)
        toks_per_sec = (r.output_tokens / r.latency_sec) if r.latency_sec > 0 else 0.0
        return toks_per_sec, r

    def check(self) -> Optional[str]:
        try:
            r = requests.get(f"{self.host}/api/version", timeout=5)
            r.raise_for_status()
            return r.json().get("version", "?")
        except Exception:
            return None


class MockLLM:
    """Deterministic fake model for offline testing of the whole harness.

    Cycles through a supplied list of responses so multi-call strategies can be
    exercised without an inference engine.
    """

    def __init__(self, responses: List[str], model: str = "mock"):
        self.responses = responses
        self.model = model
        self.i = 0
        self.prompts: List[str] = []

    def generate(self, prompt: str, temperature: float = 0.0,
                 seed: Optional[int] = None,
                 num_predict: Optional[int] = None) -> LLMResponse:
        self.prompts.append(prompt)
        text = self.responses[self.i % len(self.responses)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=len(prompt) // 4,
                           output_tokens=len(text) // 4, latency_sec=0.0,
                           done_reason="stop")

    def check(self):
        return "mock"

    def measure_speed(self, num_predict: int = 48):
        return 999.0, self.generate("warmup")


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

# LEET choices are rendered as circled numerals in the source text.
CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}

_LABEL = r"(?:final\s*answer|answer|정답)"
_CHOICE = r"([1-5]|[①②③④⑤])"

_PATTERNS = [
    # "Answer: 3" / "Answer- ③" / "Final Answer = 3" / "**Answer:** 3"
    re.compile(rf"{_LABEL}\s*\**\s*[:\-=]?\s*\**\s*\(?{_CHOICE}\)?", re.I),
    # "The answer is 3"
    re.compile(rf"{_LABEL}\s+is\s+\(?{_CHOICE}\)?", re.I),
    # bare circled numeral anywhere
    re.compile(r"([①②③④⑤])"),
]


def parse_answer(text: str) -> Optional[str]:
    """Extract the selected choice (as '1'..'5') from a free-form response.

    Returns None when no choice can be identified. None is meaningful -- an
    unparseable response is a real failure mode of a prompting strategy, and is
    counted as incorrect rather than silently dropped.
    """
    if not text:
        return None
    # Prefer the LAST labelled answer: reasoning often mentions candidate
    # choices before committing, and CoT especially tends to revise mid-stream.
    for pat in _PATTERNS:
        matches = pat.findall(text)
        if matches:
            val = matches[-1]
            return CIRCLED.get(val, val)
    # Last resort: a lone digit on its own line.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.fullmatch(r"\(?([1-5])\)?\.?", ln)
        if m:
            return m.group(1)
    return None
