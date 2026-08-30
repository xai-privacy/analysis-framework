# File: run_claude_benchmark.py
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from anthropic import Anthropic

from prompts import get_system_prompt


_MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs")
_QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmarks",
    "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json",
)
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_path(model_id: str) -> str:
    signature = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return os.path.join(_RESULTS_DIR, f"{signature}.json")


def _load_questions(year: Optional[str] = None, limit: Optional[int] = None):
    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        questions = json.load(handle)

    if year is not None:
        year_prefix = f"{year}_"
        questions = [
            question
            for question in questions
            if str(question.get("id", "")).startswith(year_prefix)
        ]

    if limit is not None:
        questions = questions[:limit]

    return questions


def _load_existing_results(result_path: str, overwrite: bool):
    if overwrite or not os.path.isfile(result_path):
        return []

    with open(result_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    if not isinstance(results, list):
        raise ValueError(f"Expected a JSON array in {result_path}")

    return results


def _save_result(result_path: str, results):
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _existing_question_run_pairs(results):
    completed = set()

    for row in results:
        qid = row.get("id") or row.get("question_id")
        run_id = row.get("run_id", 1)
        completed.add((qid, run_id))

    return completed


def _load_claude_model_config(model_id: str) -> Dict[str, Any]:
    sanitized = model_id.replace("/", "_")
    config_path = os.path.join(_MODEL_CONFIGS_DIR, f"{sanitized}.json")

    default_config = {
        "provider": "anthropic",
        "max_tokens": 3000,
        "temperature": 0,
        "sleep_seconds": 0.2
    }

    if not os.path.isfile(config_path):
        print(
            f"[Config Notice] No Anthropic config found for {model_id}; "
            f"using default baseline config: {default_config}",
            file=sys.stderr,
        )
        return default_config

    with open(config_path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)

    merged = dict(default_config)
    merged.update(user_config)
    return merged


def _extract_claude_visible_text(message) -> str:
    """
    Convert Claude Messages API content blocks into one text string.

    Baseline rule:
    - model_rationale should contain whatever visible text Claude returned.
    - We do not parse the answer here.
    - If Anthropic returns non-text blocks, we include a compact marker so the
      baseline file makes that visible rather than silently dropping it.
    """
    chunks = []

    for block in message.content:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            chunks.append(getattr(block, "text", "") or "")
        elif block_type == "thinking":
            # If extended thinking is ever enabled, preserve it visibly.
            chunks.append(getattr(block, "thinking", "") or "")
        elif block_type == "redacted_thinking":
            chunks.append("[REDACTED_THINKING_BLOCK]")
        else:
            chunks.append(f"[NON_TEXT_BLOCK:{block_type}]")

    return "\n".join(chunk for chunk in chunks if chunk is not None)


def _call_claude_model(
    client: Anthropic,
    model_id: str,
    original_question: str,
    max_tokens: int,
    temperature: Optional[float],
):
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "system": get_system_prompt(),
        "messages": [
            {
                "role": "user",
                "content": original_question,
            }
        ],
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    message = client.messages.create(**kwargs)

    usage = None
    if getattr(message, "usage", None) is not None:
        usage_obj = message.usage
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
        }

    return {
        "text": _extract_claude_visible_text(message),
        "response_id": getattr(message, "id", None),
        "model_returned": getattr(message, "model", None),
        "stop_reason": getattr(message, "stop_reason", None),
        "stop_sequence": getattr(message, "stop_sequence", None),
        "usage": usage,
    }


def execution_pipeline(
    model_id: str,
    year: Optional[str] = None,
    limit: Optional[int] = None,
    runs: int = 1,
    overwrite: bool = False,
):
    print("Starting benchmarking the model via Anthropic Claude Messages API ...\n")
    print(f"Model: {model_id}")
    print(f"Year: {year if year is not None else 'all'}")
    print(f"Limit: {limit if limit is not None else 'all'}")
    print(f"Runs: {runs}")

    model_cfg = _load_claude_model_config(model_id)
    print(f"Model config: {model_cfg}")

    max_tokens = int(model_cfg.get("max_tokens", 3000))
    temperature = model_cfg.get("temperature", 0)
    sleep_seconds = float(model_cfg.get("sleep_seconds", 0.2))

    if temperature is not None:
        temperature = float(temperature)

    questions = _load_questions(year=year, limit=limit)
    result_path = _result_path(model_id)
    results = _load_existing_results(result_path, overwrite)
    completed = _existing_question_run_pairs(results)

    print(f"Questions selected: {len(questions)}")
    print(f"Questions path: {_QUESTIONS_PATH}")
    print(f"Results file: {result_path}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print()

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for run_id in range(1, runs + 1):
        for question in questions:
            qid = question["id"]

            if (qid, run_id) in completed:
                print(f"[{qid}] run={run_id} already complete; skipping.")
                continue

            print(f"\n[{qid}] run={run_id}")

            result = dict(question)
            result.update(
                {
                    # Sebastian baseline instruction:
                    # Store raw visible model output here.
                    "model_rationale": "",

                    # No model_answer field.
                    "run_id": run_id,
                    "model_id": model_id,
                    "provider": "anthropic",
                    "backend": "api",
                    "prompt_version": "leet_arg_paper_prompt_from_prompts_py",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "created_at": _utc_now(),
                    "response_id": None,
                    "model_returned": None,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": None,
                    "error": None,
                }
            )

            try:
                response_payload = _call_claude_model(
                    client=client,
                    model_id=model_id,
                    original_question=question["original_question"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                result["model_rationale"] = response_payload["text"]
                result["response_id"] = response_payload["response_id"]
                result["model_returned"] = response_payload["model_returned"]
                result["stop_reason"] = response_payload["stop_reason"]
                result["stop_sequence"] = response_payload["stop_sequence"]
                result["usage"] = response_payload["usage"]

            except Exception as exc:
                print(f"[Question failed]: {exc}", file=sys.stderr)
                result["error"] = repr(exc)
                result["model_rationale"] = ""

            results.append(result)
            _save_result(result_path, results)

            print(
                f"Captured chars: {len(result['model_rationale'])}; "
                f"error: {result['error'] is not None}; "
                f"stop_reason: {result.get('stop_reason')}"
            )
            sys.stdout.flush()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LEET-Arg benchmark against an Anthropic Claude model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Claude model id, e.g. claude-sonnet-4-6.",
    )
    parser.add_argument("--year", help="Run only questions whose id starts with YEAR_.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected questions.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the model result file before writing responses.",
    )

    args = parser.parse_args()

    try:
        execution_pipeline(
            model_id=args.model,
            year=args.year,
            limit=args.limit,
            runs=args.runs,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)

