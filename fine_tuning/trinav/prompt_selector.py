"""Resolve built-in system prompts by benchmark and experiment type."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

PROMPTS = {
    "leetarg": {
        "baseline": HERE / "baseline-prompt.md",
        "cot": HERE / "leetarg-cot-prompt.md",
    },
    "mbe": {
        "baseline": HERE / "mbe-baseline-prompt.md",
        "cot": HERE / "mbe-cot-prompt.md",
    },
    "legalbench_causality": {
        "baseline": HERE / "legalbench-causality-baseline-prompt.md",
        "cot": HERE / "legalbench-causality-cot-prompt.md",
    },
    "legalbench_privacy": {
        "baseline": HERE / "legalbench-privacy-baseline-prompt.md",
        "cot": HERE / "legalbench-privacy-cot-prompt.md",
    },
    "legalbench_causality_finetune": {
        "baseline": HERE / "legalbench-causality-finetune-prompt.md",
    },
    "legalbench_privacy_finetune": {
        "baseline": HERE / "legalbench-privacy-finetune-prompt.md",
    },
}


def resolve_prompt(benchmark: str, prompt_type: str) -> Path:
    try:
        path = PROMPTS[benchmark][prompt_type]
    except KeyError as error:
        raise ValueError(
            f"No {prompt_type!r} prompt configured for benchmark {benchmark!r}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Configured {prompt_type} prompt not found: {path}")
    return path
