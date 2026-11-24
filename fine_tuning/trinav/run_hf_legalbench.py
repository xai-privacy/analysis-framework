#!/usr/bin/env python3
"""Run a base model or QLoRA adapter on LegalBench rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .chameleon_utils import (
        REPO,
        load_cuda_model,
        render_prompt,
        write_question_markdown,
    )
    from .prompt_selector import resolve_prompt
except ImportError:
    from chameleon_utils import (
        REPO,
        load_cuda_model,
        render_prompt,
        write_question_markdown,
    )
    from prompt_selector import resolve_prompt

DEFAULT_OUTPUT_ROOT = REPO / "testing" / "baseline" / "huggingface" / "legalbench"
TASKS = {
    "legal_reasoning_causality": {
        "benchmark": "legalbench_causality",
        "input": REPO / "benchmarks" / "LegalBench_legal_reasoning_causality_cleaned_baseline.json",
        "source": REPO / "benchmarks" / "LegalBench_legal_reasoning_causality_cleaned.json",
        "labels": {"Yes", "No"},
        "fields": {"index", "text"},
    },
    "privacy_policy_entailment": {
        "benchmark": "legalbench_privacy",
        "input": REPO / "benchmarks" / "LegalBench_privacy_policy_entailment_cleaned_baseline.json",
        "source": REPO / "benchmarks" / "LegalBench_privacy_policy_entailment_cleaned.json",
        "labels": {"Correct", "Incorrect"},
        "fields": {"index", "text", "description"},
    },
}


def read_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return rows


def row_id(row: dict[str, Any]) -> str:
    index = row.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"Invalid LegalBench index: {index!r}")
    return str(index)


def format_question(task: str, row: dict[str, Any]) -> str:
    if task == "legal_reasoning_causality":
        return f"Passage:\n{row['text'].strip()}"
    return (
        f"Privacy Policy Excerpt:\n{row['text'].strip()}\n\n"
        f"Statement:\n{row['description'].strip()}"
    )


def load_inputs(
    task: str, input_path: Path, include_all: bool = True
) -> list[dict[str, Any]]:
    config = TASKS[task]
    input_rows = read_json_array(input_path)
    input_by_id = {}
    for row in input_rows:
        qid = row_id(row)
        if qid in input_by_id:
            raise ValueError(f"Duplicate LegalBench baseline index: {qid}")
        actual_fields = set(row)
        missing_fields = config["fields"] - actual_fields
        unexpected_fields = actual_fields - config["fields"] - {"original_split"}
        if missing_fields or unexpected_fields:
            raise ValueError(
                f"{qid}: baseline record has missing fields "
                f"{sorted(missing_fields)} or unexpected fields "
                f"{sorted(unexpected_fields)}"
            )
        if include_all:
            input_by_id[qid] = row
        elif row.get("original_split") == "test":
            input_by_id[qid] = row
        elif row.get("original_split") != "train":
            raise ValueError(f"{qid}: invalid or missing original_split")

    if not input_by_id:
        description = "LegalBench rows" if include_all else "official test rows"
        raise ValueError(f"No {description} found in {input_path}")
    return sorted(input_by_id.values(), key=lambda row: row["index"])


def build_system_prompt(
    task: str,
    few_shot_rows: list[dict[str, Any]],
    exclude_index: str | None = None,
) -> str:
    examples = []
    for row in few_shot_rows:
        if exclude_index is not None and row_id(row) == exclude_index:
            continue
        examples.append(
            f"{format_question(task, row)}\n\n"
            f"Final Answer Option: {row['answer']}"
        )
    if not examples:
        return ""
    return (
        "Use these official LegalBench training examples as demonstrations. "
        "Answer the new item using the task instructions provided with it.\n\n"
        + "\n\n---\n\n".join(examples)
    )


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "run"


def save_results(path: Path, records: dict[str, dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--few-shot-data", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter-path", type=Path)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-type", choices=("baseline", "cot"))
    prompt_group.add_argument("--prompt", type=Path)
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="add official training examples to adapter test-run prompts",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, help="run only the first N rows")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.few_shot and not args.adapter_path:
        parser.error("--few-shot is only available for adapter test runs")
    if args.few_shot_data and not args.few_shot:
        parser.error("--few-shot-data requires --few-shot")

    task_config = TASKS[args.task]
    input_path = args.input or task_config["input"]
    prompt_type = args.prompt_type or ("custom" if args.prompt else "baseline")
    prompt_path = (
        args.prompt
        if args.prompt
        else resolve_prompt(task_config["benchmark"], prompt_type)
    )
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ValueError(f"Prompt is empty: {prompt_path}")

    rows = load_inputs(args.task, input_path, include_all=not args.adapter_path)
    if args.limit is not None:
        rows = rows[:args.limit]

    few_shot_rows = []
    if args.few_shot:
        source_path = args.few_shot_data or task_config["source"]
        source_rows = read_json_array(source_path)
        for row in source_rows:
            if row.get("original_split") == "train":
                if row.get("answer") not in task_config["labels"]:
                    raise ValueError(
                        f"Invalid or missing training answer for index {row.get('index')}"
                    )
                few_shot_rows.append(row)
        if not few_shot_rows:
            raise ValueError(f"No official training rows found in {source_path}")

    has_template_fields = "{text}" in prompt_text
    if not has_template_fields and not args.prompt:
        raise ValueError(f"Built-in prompt has no dataset placeholders: {prompt_path}")

    adapter_path = str(args.adapter_path) if args.adapter_path else None
    adapter_tag = args.adapter_path.name if args.adapter_path else "base"
    run_name = sanitize(args.run_name or f"{args.task}-{prompt_type}-{adapter_tag}")
    output_dir = args.output_root / args.task / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"{results_path} exists; choose another --run-name or pass --overwrite")

    import torch

    model, tokenizer, _ = load_cuda_model(args.model, adapter_path)
    model.eval()
    device = getattr(
        model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    records: dict[str, dict[str, str]] = {}

    for index, row in enumerate(rows, 1):
        qid = row_id(row)
        print(f"[{index}/{len(rows)}] Processing question {qid}", flush=True)
        system_prompt = build_system_prompt(args.task, few_shot_rows, qid)
        if has_template_fields:
            user_prompt = render_prompt(
                prompt_text,
                {"text": row["text"], "description": row.get("description", "")},
            )
            row_system_prompt = system_prompt
        else:
            user_prompt = format_question(args.task, row)
            row_system_prompt = prompt_text + (
                f"\n\n{system_prompt}" if system_prompt else ""
            )
        messages = []
        if row_system_prompt:
            messages.append({"role": "system", "content": row_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        records[qid] = {"prompt": rendered_prompt, "response": response}
        save_results(results_path, records)
        write_question_markdown(output_dir, qid, rendered_prompt, response)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
