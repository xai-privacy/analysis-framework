#!/usr/bin/env python3
"""Run MBE questions with a Hugging Face base model or PEFT adapter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .chameleon_utils import (
        HERE,
        load_cuda_model,
        read_json_or_jsonl,
        render_prompt,
        write_question_markdown,
    )
    from .prompt_selector import resolve_prompt
except ImportError:
    from chameleon_utils import (
        HERE,
        load_cuda_model,
        read_json_or_jsonl,
        render_prompt,
        write_question_markdown,
    )
    from prompt_selector import resolve_prompt

FORMAT_HINT = (
    "\n\nEnd your explanation with exactly one line:\n"
    "Final Answer Option: <A, B, C, or D>"
)
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
CHOICES = ("A", "B", "C", "D")
DEFAULT_INPUT = HERE / "mbe" / "data" / "test.jsonl"
DEFAULT_SPLIT = HERE / "mbe" / "data" / "split.json"
DEFAULT_OUTPUT_ROOT = HERE / "testing" / "baseline" / "huggingface" / "mbe"


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "run"


def load_test_data(
    input_path: Path, split_path: Path, all_data: bool = False
) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(input_path)
    if not rows:
        raise ValueError(f"No test examples found in {input_path}")

    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every MBE test record must be a JSON object")
        qid = row.get("id")
        if not isinstance(qid, str) or not ID_PATTERN.fullmatch(qid):
            raise ValueError(f"Invalid or missing MBE question ID: {qid!r}")
        if qid in seen:
            raise ValueError(f"Duplicate MBE question ID: {qid}")
        seen.add(qid)

        if all_data:
            baseline_fields = {
                "id", "question", "optionA", "optionB", "optionC", "optionD"
            }
            if (
                set(row) != baseline_fields
                or not isinstance(row["question"], str)
                or not row["question"].strip()
                or any(
                    not isinstance(row[key], str) or not row[key].strip()
                    for key in ("optionA", "optionB", "optionC", "optionD")
                )
            ):
                raise ValueError(
                    f"{qid}: baseline records must contain id, question, and "
                    "optionA through optionD"
                )
        else:
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or any(not isinstance(message, dict) for message in messages)
                or [message.get("role") for message in messages] != ["system", "user"]
            ):
                raise ValueError(f"{qid}: expected system/user messages only")
            if any(
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            ):
                raise ValueError(f"{qid}: message content cannot be blank")

    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"Expected a JSON object in {split_path}")
    if all_data:
        expected_ids = (
            set(map(str, split.get("train", [])))
            | set(map(str, split.get("valid", [])))
            | set(map(str, split.get("test", [])))
        )
        if not expected_ids or seen != expected_ids:
            missing = sorted(expected_ids - seen)
            unexpected = sorted(seen - expected_ids)
            raise ValueError(
                f"Full dataset input does not match {split_path}; "
                f"missing IDs: {missing[:10]}, unexpected IDs: {unexpected[:10]}"
            )
    else:
        expected_test_ids = split.get("test")
        if not isinstance(expected_test_ids, list) or not expected_test_ids:
            raise ValueError(f"{split_path} has no test ID list")
        expected_test_ids = set(map(str, expected_test_ids))
        if seen != expected_test_ids:
            missing = sorted(expected_test_ids - seen)
            unexpected = sorted(seen - expected_test_ids)
            raise ValueError(
                f"Test input does not match {split_path}; "
                f"missing IDs: {missing[:10]}, unexpected IDs: {unexpected[:10]}"
            )

        train_valid_ids = set(map(str, split.get("train", []))) | set(
            map(str, split.get("valid", []))
        )
        overlap = seen & train_valid_ids
        if overlap:
            raise ValueError(f"Refusing to evaluate train/validation IDs: {sorted(overlap)}")
    return rows


def fields_from_test_row(row: dict[str, Any]) -> dict[str, str]:
    if "question" in row:
        return {
            "question": row["question"],
            **{key: row[key] for key in ("optionA", "optionB", "optionC", "optionD")},
        }

    messages = row["messages"]
    user_content = messages[-1]["content"]
    question, separator, options_text = user_content.partition("\n\nChoices:\n")
    if not separator:
        raise ValueError(f"{row['id']}: test question has no Choices section")
    options = {}
    for line in options_text.splitlines():
        match = re.match(r"^\s*([A-D])\.\s*(.+?)\s*$", line)
        if match:
            options[f"option{match.group(1)}"] = f"{match.group(1)}. {match.group(2)}"
    if set(options) != {f"option{choice}" for choice in CHOICES}:
        raise ValueError(f"{row['id']}: expected choices A through D")
    return {"question": question.strip(), **options}


def build_messages(
    question: str, prompt_text: str, format_hint: bool
) -> list[dict[str, str]]:
    instructions = []
    if prompt_text.strip():
        instructions.append(prompt_text.strip())
    if format_hint:
        instructions.append(FORMAT_HINT.strip())
    messages = []
    if instructions:
        messages.append({"role": "system", "content": "\n\n".join(instructions)})
    messages.append({"role": "user", "content": question})
    return messages


def save_results(path: Path, records: dict[str, dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter-path", type=Path)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=Path)
    prompt_group.add_argument("--prompt-type", choices=("baseline", "cot"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format-hint", action="store_true")
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="run a complete baseline JSON dataset; base model only",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.all_data and args.adapter_path:
        parser.error("--all-data is only for base-model baseline runs")
    rows = load_test_data(args.input, args.split, all_data=args.all_data)
    if args.limit is not None:
        rows = rows[:args.limit]

    prompt_path = args.prompt or (
        resolve_prompt("mbe", args.prompt_type) if args.prompt_type else None
    )
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path else ""
    if prompt_path and not prompt_text.strip():
        raise ValueError(f"Prompt is empty: {prompt_path}")

    adapter_path = str(args.adapter_path) if args.adapter_path else None
    adapter_tag = args.adapter_path.name if args.adapter_path else "base"
    run_name = sanitize(args.run_name or f"{args.model}__{adapter_tag}")
    output_dir = args.output_root / run_name
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
        qid = row["id"]
        print(f"[{index}/{len(rows)}] Processing question {qid}", flush=True)
        system_prompt = ""
        question = ""
        if "{question}" in prompt_text:
            question = render_prompt(prompt_text, fields_from_test_row(row))
        else:
            if args.all_data:
                fields = fields_from_test_row(row)
                options = "\n".join(fields[f"option{choice}"] for choice in CHOICES)
                question = f"{fields['question']}\n\nChoices:\n{options}"
            else:
                question = row["messages"][-1]["content"]
            system_prompt = prompt_text
        messages = build_messages(question, system_prompt, args.format_hint)
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
