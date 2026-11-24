#!/usr/bin/env python3
"""Run LEET-Arg questions with a Hugging Face base model or PEFT adapter."""

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
    "\n\nProvide your explanation and end with exactly one line:\n"
    "Final Answer Option(s): <label(s) of the correct statement(s) or "
    "choice(s), exactly as shown in the passage>"
)
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def find_repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "benchmarks").is_dir():
            return candidate
    return HERE


REPO = find_repo_root()
DEFAULT_SPLIT = HERE / "data" / "split.json"
DEFAULT_OUTPUT_ROOT = REPO / "testing" / "baseline" / "huggingface"


def load_questions(
    path: Path, baseline_only: bool = False
) -> list[dict[str, str]]:
    rows = read_json_or_jsonl(path)
    if not rows:
        raise ValueError(f"No questions found in {path}")
    questions = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Question at index {index} is not a JSON object")
        qid = row.get("id")
        if not isinstance(qid, str) or not ID_PATTERN.fullmatch(qid):
            raise ValueError(f"Question at index {index} has an invalid or missing id")
        if qid in seen:
            raise ValueError(f"Duplicate question id: {qid}")
        seen.add(qid)

        if baseline_only and set(row) != {"id", "question"}:
            raise ValueError(
                f"{qid}: baseline record must contain only id and question"
            )
        question = row.get("question")
        if not isinstance(question, str):
            original = row.get("original_question")
            statements = row.get("statements")
            if not isinstance(original, str) or not isinstance(statements, dict):
                raise ValueError(f"{qid}: expected a question string")
            question = build_user_prompt(original, statements)
        if not question.strip():
            raise ValueError(f"{qid}: question cannot be blank")
        questions.append({"id": qid, "question": question})
    return questions


def build_user_prompt(original: str, statements: dict[str, Any]) -> str:
    text = original
    first_statement = next(iter(statements.values()), "")
    if not isinstance(first_statement, str) or not first_statement.strip():
        raise ValueError("Source question has no statement text")
    cut = text.find(first_statement.strip()[:30], max(text.find("<question>"), 0))
    head = text[:cut] if cut > 0 else text.split("<choices>")[0]
    head = re.sub(r"(\s*<(statements|choices)>\s*)+$", "", head.rstrip())
    if "<question>" in head:
        passage, _, stem = head.rpartition("<question>")
    else:
        end = head.rfind("?")
        start = max(head.rfind(".", 0, end), head.rfind(">", 0, end)) + 1
        passage, stem = head[:start], head[start:]
    passage = re.sub(r"(?<=[.!?”\"’)])\s*(?=[A-Z][a-z]+\d?:\s)", "\n", passage)
    statements_text = "\n".join(
        statement.strip()
        for statement in statements.values()
        if isinstance(statement, str)
    )
    return (
        f"{passage.strip()}\n\n<question> {stem.strip()}\n<statements>\n"
        f"{statements_text}"
    )


def load_test_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array of test IDs in {path}")
    ids = [str(row.get("id")) if isinstance(row, dict) else str(row) for row in rows]
    if any(not qid or qid == "None" for qid in ids):
        raise ValueError(f"Missing question ID in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate test IDs in {path}")
    return ids


def read_split(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_ids = set(map(str, split.get("train", split.get("train_ids", []))))
    valid_ids = set(map(str, split.get("valid", split.get("valid_ids", []))))
    return train_ids, valid_ids


def build_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter-path", type=Path)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=Path)
    prompt_group.add_argument("--prompt-type", choices=("baseline", "cot"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=int, default=1200)
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
    if args.all_data and (args.adapter_path or args.test_ids):
        parser.error("--all-data is only for full base-model baseline runs")

    questions = load_questions(args.input, baseline_only=args.all_data)
    if args.all_data:
        if args.input.suffix != ".json":
            parser.error("--all-data requires a JSON baseline file")
        with args.split.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        expected_count = split.get("source_count")
        if expected_count is not None and len(questions) != expected_count:
            raise ValueError(
                f"--all-data expected {expected_count} questions, found {len(questions)}"
            )
    else:
        if args.test_ids:
            requested_ids = load_test_ids(args.test_ids)
            questions_by_id = {row["id"]: row for row in questions}
            missing = sorted(set(requested_ids) - set(questions_by_id))
            if missing:
                raise ValueError(f"Test IDs missing from --input: {missing}")
            questions = [questions_by_id[qid] for qid in requested_ids]

        train_ids, valid_ids = read_split(args.split)
        overlap = {row["id"] for row in questions} & (train_ids | valid_ids)
        if overlap:
            raise ValueError(
                "Refusing to evaluate questions used for training or validation: "
                + ", ".join(sorted(overlap))
            )

    if args.limit is not None:
        questions = questions[:args.limit]

    prompt_path = args.prompt or (
        resolve_prompt("leetarg", args.prompt_type) if args.prompt_type else None
    )
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path else ""
    if prompt_path and not prompt_text.strip():
        raise ValueError(f"Prompt is empty: {prompt_path}")

    adapter_path = str(args.adapter_path) if args.adapter_path else None
    adapter_tag = args.adapter_path.name if args.adapter_path else "base"
    run_name = args.run_name or f"{args.model.replace('/', '_')}__{adapter_tag}"
    output_dir = args.output_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
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

    for index, row in enumerate(questions, 1):
        qid = row["id"]
        print(f"[{index}/{len(questions)}] Processing question {qid}", flush=True)
        question = row["question"]
        system_prompt = prompt_text
        if "{question}" in prompt_text:
            question = render_prompt(prompt_text, {"question": question})
            system_prompt = ""
        if args.format_hint:
            if system_prompt:
                system_prompt += FORMAT_HINT
            else:
                question += FORMAT_HINT
        messages = build_messages(question, system_prompt)
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
