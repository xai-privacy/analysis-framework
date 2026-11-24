#!/usr/bin/env python3
"""Prepare leakage-safe LegalBench chat data for QLoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .chameleon_utils import HERE, REPO
except ImportError:
    from chameleon_utils import HERE, REPO

TASKS = {
    "legal_reasoning_causality": {
        "source": REPO / "benchmarks" / "LegalBench_legal_reasoning_causality_cleaned.json",
        "answer_labels": {"Yes", "No"},
    },
    "privacy_policy_entailment": {
        "source": REPO / "benchmarks" / "LegalBench_privacy_policy_entailment_cleaned.json",
        "answer_labels": {"Correct", "Incorrect"},
    },
}

SYSTEM_PROMPTS = {
    "legal_reasoning_causality": (
        "Determine whether the legal passage's causal reasoning relies on "
        "statistical evidence. Answer Yes or No."
    ),
    "privacy_policy_entailment": (
        "Determine whether the statement is entailed by the privacy policy "
        "excerpt. Base your answer only on the excerpt. Answer Correct or Incorrect."
    ),
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_text(task: str, row: dict[str, Any]) -> str:
    if task == "legal_reasoning_causality":
        return f"Passage:\n{row['text'].strip()}"
    return (
        f"Privacy Policy Excerpt:\n{row['text'].strip()}\n\n"
        f"Statement:\n{row['description'].strip()}"
    )


def make_record(
    task: str,
    row: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    answer = row["answer"]
    return {
        "id": str(row["index"]),
        "task": task,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text(task, row)},
            {
                "role": "assistant",
                "content": f"Final Answer Option: {answer}",
            },
        ],
    }


def prepare_data(
    task: str,
    source_path: Path,
    output_dir: Path,
    valid_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"Unsupported LegalBench task: {task}")
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")

    config = TASKS[task]
    rows = read_json(source_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON array in {source_path}")

    seen = set()
    grouped = defaultdict(list)
    test_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every LegalBench record must be a JSON object")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"Invalid LegalBench index: {index!r}")
        if index in seen:
            raise ValueError(f"Duplicate LegalBench index: {index}")
        seen.add(index)
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"LegalBench record {index} has no text")
        if task == "privacy_policy_entailment" and (
            not isinstance(row.get("description"), str)
            or not row["description"].strip()
        ):
            raise ValueError(f"LegalBench record {index} has no description")
        answer = row.get("answer")
        if answer not in config["answer_labels"]:
            raise ValueError(f"LegalBench record {index} has invalid answer {answer!r}")

        original_split = row.get("original_split")
        if original_split == "train":
            grouped[answer].append(row)
        elif original_split == "test":
            test_rows.append(row)
        else:
            raise ValueError(
                f"LegalBench record {index} has invalid original_split "
                f"{original_split!r}"
            )

    rng = random.Random(seed)
    train_rows = []
    valid_rows = []
    for answer in sorted(grouped):
        examples = sorted(grouped[answer], key=lambda row: row["index"])
        if len(examples) < 2:
            raise ValueError(
                f"Cannot split label {answer!r}: need at least two official "
                "training examples to keep both train and validation nonempty"
            )
        rng.shuffle(examples)
        valid_count = max(1, round(len(examples) * valid_fraction))
        valid_count = min(valid_count, len(examples) - 1)
        valid_rows.extend(examples[:valid_count])
        train_rows.extend(examples[valid_count:])

    train_rows.sort(key=lambda row: row["index"])
    valid_rows.sort(key=lambda row: row["index"])
    test_rows.sort(key=lambda row: row["index"])
    if not train_rows or not valid_rows or not test_rows:
        raise ValueError("Expected nonempty train, validation, and official test splits")

    system_prompt = SYSTEM_PROMPTS[task]

    train_records = [make_record(task, row, system_prompt) for row in train_rows]
    valid_records = [make_record(task, row, system_prompt) for row in valid_rows]
    train_ids = [record["id"] for record in train_records]
    valid_ids = [record["id"] for record in valid_records]
    test_ids = [str(row["index"]) for row in test_rows]
    if set(train_ids) & set(valid_ids) or (
        set(train_ids + valid_ids) & set(test_ids)
    ):
        raise ValueError("LegalBench split contains overlapping IDs")

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "valid.jsonl", valid_records)
    split = {
        "task": task,
        "source": str(source_path.resolve()),
        "seed": seed,
        "valid_fraction": valid_fraction,
        "train": train_ids,
        "valid": valid_ids,
        "test_ids": test_ids,
        "official_train_ids": sorted(train_ids + valid_ids),
    }
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "task": task,
        "train": len(train_records),
        "valid": len(valid_records),
        "official_test": len(test_ids),
        "output": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--valid-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TASKS[args.task]
    source_path = args.source or config["source"]
    output_dir = args.out or HERE / "legalbench" / "data" / args.task
    print(json.dumps(
        prepare_data(
            args.task,
            source_path,
            output_dir,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
