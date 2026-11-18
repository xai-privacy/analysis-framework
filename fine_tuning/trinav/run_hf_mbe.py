#!/usr/bin/env python3
"""Evaluate a Hugging Face base model or PEFT adapter on the held-out MBE set."""

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
    from .chameleon_utils import HERE, load_cuda_model, read_jsonl
    from .mbe.prepare_mbe_data import question_id, validate_question
except ImportError:
    from chameleon_utils import HERE, load_cuda_model, read_jsonl
    from mbe.prepare_mbe_data import question_id, validate_question

FORMAT_HINT = (
    "\n\nBegin your response with `Answer: <letter>` on its own first line, "
    "where the letter is A, B, C, or D. Then explain your answer."
)
ANSWER_PATTERNS = (
    re.compile(r"^\s*(?:\*\*)?Answer\s*[:=-]\s*\**([A-D])\b", re.I),
    re.compile(
        r"^\s*(?:The\s+)?(?:correct\s+)?answer\s+is\s+([A-D])\b", re.I
    ),
)
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
CHOICES = {"A", "B", "C", "D"}
DEFAULT_INPUT = HERE / "mbe" / "data" / "test.jsonl"
DEFAULT_SPLIT = HERE / "mbe" / "data" / "split.json"
DEFAULT_OUTPUT_ROOT = HERE / "testing" / "baseline" / "huggingface" / "mbe"


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "run"


def load_test_data(
    input_path: Path, split_path: Path, all_data: bool = False
) -> list[dict[str, Any]]:
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No test examples found in {input_path}")

    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every MBE test line must be a JSON object")
        qid = row.get("id")
        if not isinstance(qid, str) or not ID_PATTERN.fullmatch(qid):
            raise ValueError(f"Invalid or missing MBE question ID: {qid!r}")
        if qid in seen:
            raise ValueError(f"Duplicate MBE question ID: {qid}")
        seen.add(qid)

        if all_data:
            if (
                set(row) != {"id", "question"}
                or not isinstance(row["question"], str)
                or not row["question"].strip()
            ):
                raise ValueError(
                    f"{qid}: baseline lines must contain only id and question"
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

            evaluation = row.get("evaluation")
            if (
                not isinstance(evaluation, dict)
                or evaluation.get("gold_answer") not in CHOICES
                or not isinstance(evaluation.get("gold_explanation"), dict)
            ):
                raise ValueError(f"{qid}: missing or invalid MBE evaluation metadata")

    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"Expected a JSON object in {split_path}")
    if all_data:
        expected_ids = set(map(str, split.get("train", []))) | set(
            map(str, split.get("valid", []))
        ) | set(map(str, split.get("test", [])))
        if not expected_ids or seen != expected_ids:
            missing = sorted(expected_ids - seen)
            unexpected = sorted(seen - expected_ids)
            raise ValueError(
                f"Full dataset JSONL does not match {split_path}; "
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
                f"Test JSONL does not match {split_path}; "
                f"missing IDs: {missing[:10]}, unexpected IDs: {unexpected[:10]}"
            )

        train_valid_ids = set(map(str, split.get("train", []))) | set(
            map(str, split.get("valid", []))
        )
        overlap = seen & train_valid_ids
        if overlap:
            raise ValueError(f"Refusing to evaluate train/validation IDs: {sorted(overlap)}")
    return rows


def parse_answer(output: str) -> str | None:
    first_line = next((line for line in output.splitlines() if line.strip()), "")
    for pattern in ANSWER_PATTERNS:
        match = pattern.match(first_line)
        if match:
            return match.group(1).upper()
    return None


def build_messages(
    question: str, prompt_text: str, format_hint: bool
) -> list[dict[str, str]]:
    messages = []
    instructions = []
    if prompt_text.strip():
        instructions.append(prompt_text.strip())
    if format_hint:
        instructions.append(FORMAT_HINT.strip())
    if instructions:
        messages.append({"role": "system", "content": "\n\n".join(instructions)})
    messages.append({"role": "user", "content": question})
    return messages


def load_gold_data(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array in {path}")

    gold_by_id = {}
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Every MBE gold entry must be a JSON object")
        validate_question(question)
        qid = question_id(question)
        if qid in gold_by_id:
            raise ValueError(f"Duplicate MBE gold question ID: {qid}")
        gold_by_id[qid] = {
            "gold_answer": str(question["correct_answer"]).strip().upper(),
            "gold_explanation": question["explanation"],
        }
    return gold_by_id


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    parsed = sum(record["pred_answer"] is not None for record in records)
    return {
        "n_questions": total,
        "parse_rate": parsed / total if total else 0.0,
        "question_accuracy": (
            sum(record["correct"] for record in records) / total if total else 0.0
        ),
    }


def write_results(
    output_dir: Path, metadata: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = {**metadata, **score_records(records)}
    temporary = output_dir / "results.json.tmp"
    temporary.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_dir / "results.json")
    return summary


def write_question_markdown(
    output_dir: Path, row: dict[str, Any], record: dict[str, Any]
) -> None:
    explanation = row["evaluation"]["gold_explanation"]
    gold_explanation = "\n\n".join(
        f"**{label.replace('_', ' ')}:** {text}"
        for label, text in explanation.items()
        if isinstance(text, str) and text.strip()
    )
    content = (
        f"# Question {row['id']}\n\n"
        f"- Model: `{record['model']}`\n"
        f"- Adapter: `{record['adapter_path'] or 'base model'}`\n"
        f"- Gold answer: {record['gold_answer']}\n"
        f"- Predicted answer: {record['pred_answer'] or 'UNPARSED'}\n"
        f"- Correct: {record['correct']}\n\n"
        f"## Question\n\n{row['question']}\n\n"
        f"## Model response\n\n{record['output']}\n\n"
        f"## Gold explanation\n\n{gold_explanation}\n"
    )
    (output_dir / f"{row['id']}.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--gold-data",
        type=Path,
        help="full source benchmark JSON used for scoring --all-data questions",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, help="Omit to evaluate the base model")
    parser.add_argument("--prompt", help="Optional UTF-8 file containing extra system instructions")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N test questions")
    parser.add_argument("--format-hint", action="store_true")
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="evaluate the complete source dataset; base-model baselines only",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    prompt_path = Path(args.prompt) if args.prompt else None
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path else ""
    if args.all_data and args.adapter_path:
        parser.error("--all-data is only for base-model baseline runs")
    if args.all_data and not args.gold_data:
        parser.error("--gold-data is required with --all-data")
    if args.gold_data and not args.all_data:
        parser.error("--gold-data is only used with --all-data")
    rows = load_test_data(args.input, args.split, all_data=args.all_data)
    if args.all_data:
        gold_by_id = load_gold_data(args.gold_data)
        input_ids = {row["id"] for row in rows}
        if input_ids != set(gold_by_id):
            raise SystemExit(
                "Baseline questions do not match --gold-data; "
                f"missing IDs: {sorted(set(gold_by_id) - input_ids)[:10]}, "
                f"unexpected IDs: {sorted(input_ids - set(gold_by_id))[:10]}"
            )
        for row in rows:
            row["evaluation"] = gold_by_id[row["id"]]
    else:
        for row in rows:
            row["question"] = row["messages"][-1]["content"]
    if args.limit is not None:
        rows = rows[:args.limit]

    adapter_path = str(args.adapter_path) if args.adapter_path else None
    adapter_tag = args.adapter_path.name if args.adapter_path else "base"
    run_name = sanitize(args.run_name or f"{args.model}__{adapter_tag}")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "adapter_path": adapter_path,
        "input": str(args.input.resolve()),
        "split": str(args.split.resolve()),
        "gold_data": str(args.gold_data.resolve()) if args.gold_data else None,
        "prompt": str(prompt_path.resolve()) if prompt_path else None,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "format_hint": args.format_hint,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "all_data": args.all_data,
    }
    manifest_path = output_dir / "run.json"
    if manifest_path.exists():
        previous_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_metadata != metadata and not args.overwrite:
            raise SystemExit(
                f"{output_dir} contains a different run. Choose another --run-name "
                "or pass --overwrite."
            )
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    wanted_ids = {row["id"] for row in rows}
    results_path = output_dir / "results.json"
    records = []
    if results_path.exists() and not args.overwrite:
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        records = [
            record for record in previous.get("records", [])
            if record.get("id") in wanted_ids
        ]
        print(f"Resuming with {len(records)} completed questions")
    done = {record["id"] for record in records}

    model, tokenizer, _ = load_cuda_model(args.model, adapter_path)
    model.eval()
    import torch

    device = getattr(
        model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"Running {len(rows)} MBE questions with {args.model} "
        f"({adapter_path or 'base model'}); output: {output_dir}"
    )

    for index, row in enumerate(rows, 1):
        qid = row["id"]
        if qid in done:
            continue
        question = (
            row["question"] if args.all_data else row["messages"][-1]["content"]
        )
        messages = build_messages(question, prompt_text, args.format_hint)
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
        prediction = parse_answer(output)
        gold_answer = row["evaluation"]["gold_answer"]
        record = {
            "id": qid,
            "model": args.model,
            "adapter_path": adapter_path,
            "gold_answer": gold_answer,
            "pred_answer": prediction,
            "correct": prediction == gold_answer,
            "seconds": round(time.time() - start, 2),
            "output": output,
        }
        records.append(record)
        write_question_markdown(output_dir, row, record)
        write_results(output_dir, metadata, records)
        print(
            f"[{index}/{len(rows)}] {qid}: predicted "
            f"{prediction or 'UNPARSED'} (gold {gold_answer}), {record['seconds']}s",
            flush=True,
        )

    print(json.dumps(write_results(output_dir, metadata, records), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
