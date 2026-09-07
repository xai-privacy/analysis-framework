# File: run_openai_benchmark.py
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

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


def _load_openai_model_config(model_id: str) -> Dict[str, Any]:
    """
    Optional OpenAI config.

    This is intentionally lightweight. If a file exists in model_configs/,
    we read fields such as max_output_tokens and reasoning_effort.

    If it does not exist, we use a reasonable baseline default.
    """
    sanitized = model_id.replace("/", "_")
    config_path = os.path.join(_MODEL_CONFIGS_DIR, f"{sanitized}.json")

    default_config = {
        "provider": "openai",
        "max_output_tokens": 3000,
        "reasoning_effort": None,
        "sleep_seconds": 0.2,
    }

    if not os.path.isfile(config_path):
        print(
            f"[Config Notice] No OpenAI config found for {model_id}; "
            f"using default baseline config: {default_config}",
            file=sys.stderr,
        )
        return default_config

    with open(config_path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)

    merged = dict(default_config)
    merged.update(user_config)
    return merged


def _build_prompt(original_question: str) -> str:
    """
    Match the local runner's conceptual setup:
    - benchmark question text
    - LEET-Arg paper-style instruction prompt from prompts.py

    We do not add constrained decoding, JSON schema, or a special CoT prompt.
    """
    return f"{original_question.strip()}\n\n{get_system_prompt().strip()}"


def _call_openai_model(
    client: OpenAI,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
):
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }

    # Leave unset by default. If intentionally used for a reasoning model,
    # it is recorded in the output.
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)

    usage = None
    if getattr(response, "usage", None) is not None:
        usage = response.usage.model_dump()

    return {
        "text": response.output_text or "",
        "response_id": getattr(response, "id", None),
        "model_returned": getattr(response, "model", None),
        "usage": usage,
    }


def execution_pipeline(
    model_id: str,
    year: Optional[str] = None,
    limit: Optional[int] = None,
    runs: int = 1,
    overwrite: bool = False,
):
    print("Starting benchmarking the model via OpenAI Responses API ...\n")
    print(f"Model: {model_id}")
    print(f"Year: {year if year is not None else 'all'}")
    print(f"Limit: {limit if limit is not None else 'all'}")
    print(f"Runs: {runs}")

    model_cfg = _load_openai_model_config(model_id)
    print(f"Model config: {model_cfg}")

    max_output_tokens = int(model_cfg.get("max_output_tokens", 3000))
    reasoning_effort = model_cfg.get("reasoning_effort")
    sleep_seconds = float(model_cfg.get("sleep_seconds", 0.2))

    questions = _load_questions(year=year, limit=limit)
    result_path = _result_path(model_id)
    results = _load_existing_results(result_path, overwrite)
    completed = _existing_question_run_pairs(results)

    print(f"Questions selected: {len(questions)}")
    print(f"Questions path: {_QUESTIONS_PATH}")
    print(f"Results file: {result_path}")
    print(f"Max output tokens: {max_output_tokens}")
    print(f"Reasoning effort: {reasoning_effort}")
    print()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for run_id in range(1, runs + 1):
        for question in questions:
            qid = question["id"]

            if (qid, run_id) in completed:
                print(f"[{qid}] run={run_id} already complete; skipping.")
                continue

            print(f"\n[{qid}] run={run_id}")

            result = dict(question)

            # Important baseline behavior:
            # model_rationale is the raw model response as returned by the API.
            # We intentionally do not create model_answer and do not parse here.
            result.update(
                {
                    "model_rationale": "",
                    "run_id": run_id,
                    "model_id": model_id,
                    "provider": "openai",
                    "backend": "api",
                    "prompt_version": "leet_arg_paper_prompt_from_prompts_py",
                    "max_output_tokens": max_output_tokens,
                    "reasoning_effort": reasoning_effort,
                    "created_at": _utc_now(),
                    "response_id": None,
                    "model_returned": None,
                    "usage": None,
                    "error": None,
                }
            )

            prompt = _build_prompt(question["original_question"])

            try:
                response_payload = _call_openai_model(
                    client=client,
                    model_id=model_id,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )

                result["model_rationale"] = response_payload["text"]
                result["response_id"] = response_payload["response_id"]
                result["model_returned"] = response_payload["model_returned"]
                result["usage"] = response_payload["usage"]

            except Exception as exc:
                print(f"[Question failed]: {exc}", file=sys.stderr)
                result["error"] = repr(exc)
                result["model_rationale"] = ""

            results.append(result)
            _save_result(result_path, results)

            print(
                f"Captured chars: {len(result['model_rationale'])}; "
                f"error: {result['error'] is not None}"
            )
            sys.stdout.flush()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LEET-Arg benchmark against an OpenAI frontier model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model id, e.g. gpt-4.1, gpt-4.1-mini, o4-mini.",
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
