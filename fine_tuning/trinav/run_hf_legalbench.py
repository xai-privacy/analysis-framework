#!/usr/bin/env python3
"""Evaluate a base model or QLoRA adapter on LegalBench test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .chameleon_utils import HERE, load_cuda_model
    from .prompt_selector import resolve_prompt
except ImportError:
    from chameleon_utils import HERE, load_cuda_model
    from prompt_selector import resolve_prompt

REPO = HERE.parents[1]
DEFAULT_OUTPUT_ROOT = REPO / "testing" / "baseline" / "huggingface" / "legalbench"
ANSWER_RE = re.compile(
    r"^\s*Final Answer Option\s*:\s*(Correct|Incorrect|Yes|No)\b",
    re.IGNORECASE | re.MULTILINE,
)

TASKS = {
    "legal_reasoning_causality": {
        "benchmark": "legalbench_causality",
        "input": REPO / "benchmarks" / "LegalBench_legal_reasoning_causality_cleaned_baseline.json",
        "gold": REPO / "benchmarks" / "LegalBench_legal_reasoning_causality_cleaned.json",
        "labels": {"Yes", "No"},
        "fields": {"index", "text", "original_split"},
    },
    "privacy_policy_entailment": {
        "benchmark": "legalbench_privacy",
        "input": REPO / "benchmarks" / "LegalBench_privacy_policy_entailment_cleaned_baseline.json",
        "gold": REPO / "benchmarks" / "LegalBench_privacy_policy_entailment_cleaned.json",
        "labels": {"Correct", "Incorrect"},
        "fields": {"index", "text", "description", "original_split"},
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
    task: str,
    input_path: Path,
    gold_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config = TASKS[task]
    input_rows = read_json_array(input_path)
    gold_rows = read_json_array(gold_path)

    gold_by_id = {}
    for row in gold_rows:
        qid = row_id(row)
        if qid in gold_by_id:
            raise ValueError(f"Duplicate LegalBench gold index: {qid}")
        if row.get("answer") not in config["labels"]:
            raise ValueError(f"{qid}: invalid gold answer {row.get('answer')!r}")
        if row.get("original_split") not in {"train", "test"}:
            raise ValueError(f"{qid}: invalid original_split")
        gold_by_id[qid] = row

    input_by_id = {}
    for row in input_rows:
        qid = row_id(row)
        if qid in input_by_id:
            raise ValueError(f"Duplicate LegalBench baseline index: {qid}")
        if set(row) != config["fields"]:
            raise ValueError(
                f"{qid}: baseline record must contain exactly "
                f"{sorted(config['fields'])}"
            )
        gold = gold_by_id.get(qid)
        if gold is None:
            raise ValueError(f"{qid}: baseline index is missing from gold data")
        if row["original_split"] != gold["original_split"]:
            raise ValueError(f"{qid}: baseline split does not match gold data")
        input_fields = {"index", "text"}
        if task == "privacy_policy_entailment":
            input_fields.add("description")
        if any(row[field] != gold[field] for field in input_fields):
            raise ValueError(f"{qid}: baseline input does not match gold data")
        input_by_id[qid] = row

    if set(input_by_id) != set(gold_by_id):
        raise ValueError("Baseline inputs and gold data have different ID sets")
    test_rows = [
        row for row in input_rows if row["original_split"] == "test"
    ]
    if not test_rows:
        raise ValueError(f"No official test rows found in {input_path}")
    return test_rows, gold_by_id


def build_system_prompt(
    task: str,
    prompt_text: str,
    few_shot_rows: list[dict[str, Any]],
) -> str:
    if not few_shot_rows:
        return prompt_text
    examples = []
    for row in few_shot_rows:
        examples.append(
            f"{format_question(task, row)}\n\n"
            f"Final Answer Option: {row['answer']}"
        )
    return (
        f"{prompt_text.rstrip()}\n\n"
        "Official LegalBench training examples:\n\n"
        + "\n\n---\n\n".join(examples)
    )


def parse_answer(output: str, allowed_labels: set[str]) -> str | None:
    match = ANSWER_RE.search(output)
    if not match:
        return None
    by_casefold = {label.casefold(): label for label in allowed_labels}
    return by_casefold.get(match.group(1).casefold())


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "run"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    return {
        "n_questions": total,
        "parse_rate": (
            sum(record["pred_answer"] is not None for record in records) / total
            if total else 0.0
        ),
        "accuracy": (
            sum(record["correct"] for record in records) / total if total else 0.0
        ),
    }


def write_results(
    path: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {**metadata, **summarize(records)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--gold-data", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter-path", type=Path)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-type", choices=("baseline", "cot"))
    prompt_group.add_argument("--prompt", type=Path)
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="add the official LegalBench training examples to the system prompt",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    task_config = TASKS[args.task]
    input_path = args.input or task_config["input"]
    gold_path = args.gold_data or task_config["gold"]
    prompt_type = args.prompt_type or ("custom" if args.prompt else "baseline")
    prompt_path = (
        args.prompt
        if args.prompt
        else resolve_prompt(task_config["benchmark"], prompt_type)
    )
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt is empty: {prompt_path}")

    test_rows, gold_by_id = load_inputs(args.task, input_path, gold_path)
    if args.limit is not None:
        test_rows = test_rows[:args.limit]

    few_shot_rows = []
    if args.few_shot:
        full_gold_rows = read_json_array(gold_path)
        few_shot_rows = [
            row for row in full_gold_rows if row["original_split"] == "train"
        ]
    system_prompt = build_system_prompt(args.task, prompt_text, few_shot_rows)

    adapter_path = str(args.adapter_path) if args.adapter_path else None
    adapter_tag = args.adapter_path.name if args.adapter_path else "base"
    run_name = sanitize(
        args.run_name or f"{args.task}-{prompt_type}-{adapter_tag}"
    )
    output_dir = args.output_root / args.task / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "model": args.model,
        "adapter_path": adapter_path,
        "input": str(input_path.resolve()),
        "gold_data": str(gold_path.resolve()),
        "prompt": str(prompt_path.resolve()),
        "prompt_type": prompt_type,
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "few_shot": args.few_shot,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
    }
    manifest_path = output_dir / "run.json"
    results_path = output_dir / "results.json"
    if manifest_path.exists():
        old_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_metadata != metadata and not args.overwrite:
            raise SystemExit(
                f"{output_dir} contains a different run. Choose another "
                "--run-name or pass --overwrite."
            )
    elif results_path.exists() and not args.overwrite:
        raise SystemExit(
            f"{results_path} exists without run metadata; choose another "
            "--run-name or pass --overwrite."
        )
    manifest_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    wanted_ids = {row_id(row) for row in test_rows}
    records = []
    if results_path.exists() and not args.overwrite:
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        records = [
            record for record in previous.get("records", [])
            if record.get("id") in wanted_ids
        ]
        if records:
            print(f"Resuming with {len(records)} completed questions")
    done = {record["id"] for record in records}

    model, tokenizer, _ = load_cuda_model(args.model, adapter_path)
    model.eval()
    import torch

    device = getattr(
        model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    for position, row in enumerate(test_rows, 1):
        qid = row_id(row)
        if qid in done:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_question(args.task, row)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        start = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()
        prediction = parse_answer(output, task_config["labels"])
        gold_answer = gold_by_id[qid]["answer"]
        record = {
            "id": qid,
            "gold_answer": gold_answer,
            "pred_answer": prediction,
            "correct": prediction == gold_answer,
            "seconds": round(time.time() - start, 2),
            "output": output,
        }
        records.append(record)
        write_results(results_path, metadata, records)
        print(
            f"[{position}/{len(test_rows)}] {qid}: predicted "
            f"{prediction or 'UNPARSED'} (gold {gold_answer}), "
            f"{record['seconds']}s",
            flush=True,
        )

    print(json.dumps(write_results(results_path, metadata, records), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
