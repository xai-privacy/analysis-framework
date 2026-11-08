#!/usr/bin/env python3
"""Create leakage-safe MBE train/validation/test chat datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_SOURCE = REPO / "benchmarks" / "MBE-exam-questions_distilled_dataset_cleaned.json"
DEFAULT_OUTPUT = HERE / "data"

OPTION_KEYS = ("optionA", "optionB", "optionC", "optionD")
EXPLANATION_FIELDS = (
    "Legal_Concept",
    "Fact_Analysis",
    "Rule_Application",
    "Legal_Conclusion",
)
SYSTEM_PROMPT = (
    "You are a careful U.S. bar-exam tutor. Choose the single best answer. "
    "Explain the answer using the supplied legal concepts, facts, and rules. "
    "Do not introduce facts that are not in the question."
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_id(question: dict[str, Any]) -> str:
    domain = str(question["domain"])
    number = int(question["question_number"])
    return f"{domain}_{number:03d}"


def validate_question(question: dict[str, Any]):
    qid = question_id(question)
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        raise ValueError(f"{qid}: missing question text")
    options = question.get("options")
    if not isinstance(options, dict) or any(
        not isinstance(options.get(key), str) or not options[key].strip()
        for key in OPTION_KEYS
    ):
        raise ValueError(f"{qid}: expected nonempty optionA through optionD")
    answer = str(question.get("correct_answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"{qid}: invalid correct_answer {answer!r}")
    explanation = question.get("explanation")
    if not isinstance(explanation, dict) or any(
        not isinstance(explanation.get(key), str) or not explanation[key].strip()
        for key in EXPLANATION_FIELDS
    ):
        raise ValueError(f"{qid}: incomplete IRAC explanation")


def allocate_stratified_counts(groups, target_count: int, denominator: int):
    """Allocate an exact total proportionally, with deterministic remainder handling."""
    counts = {}
    remainders = []
    allocated = 0
    for key in sorted(groups):
        exact = len(groups[key]) * target_count / denominator
        count = min(len(groups[key]), math.floor(exact))
        counts[key] = count
        allocated += count
        remainders.append((exact - count, key))

    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if allocated >= target_count:
            break
        if counts[key] < len(groups[key]):
            counts[key] += 1
            allocated += 1

    if allocated != target_count:
        raise ValueError(f"Could not allocate requested stratified count {target_count}")
    return counts


def split_questions(questions, test_count: int, valid_fraction: float, seed: int):
    by_domain = defaultdict(list)
    for question in questions:
        by_domain[str(question["domain"])].append(question)

    test_counts = allocate_stratified_counts(by_domain, test_count, len(questions))
    rng = random.Random(seed)
    train_valid_by_domain = {}
    test = []
    for domain in sorted(by_domain):
        group = sorted(by_domain[domain], key=question_id)
        rng.shuffle(group)
        count = test_counts[domain]
        test.extend(group[:count])
        train_valid_by_domain[domain] = group[count:]

    remaining = [question for group in train_valid_by_domain.values() for question in group]
    valid_count = round(len(remaining) * valid_fraction)
    valid_counts = allocate_stratified_counts(
        train_valid_by_domain, valid_count, len(remaining)
    )

    train, valid = [], []
    for domain in sorted(train_valid_by_domain):
        group = train_valid_by_domain[domain]
        count = valid_counts[domain]
        valid.extend(group[:count])
        train.extend(group[count:])

    key = lambda question: question_id(question)
    return tuple(sorted(partition, key=key) for partition in (train, valid, test))


def build_user_prompt(question: dict[str, Any]) -> str:
    options = "\n".join(question["options"][key].strip() for key in OPTION_KEYS)
    return f"{question['question'].strip()}\n\nChoices:\n{options}"


def build_answer(question: dict[str, Any]) -> str:
    explanation = question["explanation"]
    sections = [
        f"Answer: {str(question['correct_answer']).strip().upper()}",
        f"Legal Concept: {explanation['Legal_Concept'].strip()}",
        f"Fact Analysis: {explanation['Fact_Analysis'].strip()}",
        f"Rule Application: {explanation['Rule_Application'].strip()}",
        f"Legal Conclusion: {explanation['Legal_Conclusion'].strip()}",
    ]
    return "\n\n".join(sections)


def make_record(question: dict[str, Any], include_answer: bool):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": build_answer(question)})
    return {
        "id": question_id(question),
        "domain": question["domain"],
        "question_number": int(question["question_number"]),
        "messages": messages,
        "evaluation": {
            "gold_answer": str(question["correct_answer"]).strip().upper(),
            "gold_explanation": question["explanation"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--valid-fraction", type=float, default=15 / 73)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.test_count < 1:
        parser.error("--test-count must be positive")
    if not 0 < args.valid_fraction < 1:
        parser.error("--valid-fraction must be between 0 and 1")

    questions = read_json(args.source)
    if not isinstance(questions, list):
        raise SystemExit(f"Expected a JSON array in {args.source}")
    for question in questions:
        if not isinstance(question, dict):
            raise SystemExit("Every MBE dataset entry must be a JSON object")
        validate_question(question)

    ids = [question_id(question) for question in questions]
    duplicates = sorted(qid for qid, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate domain/question IDs: {duplicates[:20]}")
    if args.test_count >= len(questions):
        parser.error("--test-count must be smaller than the dataset size")

    train, valid, test = split_questions(
        questions, args.test_count, args.valid_fraction, args.seed
    )
    train_ids = {question_id(question) for question in train}
    valid_ids = {question_id(question) for question in valid}
    test_ids = {question_id(question) for question in test}
    if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
        raise SystemExit("Split leakage detected: partitions overlap")
    if train_ids | valid_ids | test_ids != set(ids):
        raise SystemExit("Split error: one or more questions were dropped")

    write_jsonl(args.out / "train.jsonl", [
        make_record(question, include_answer=True) for question in train
    ])
    write_jsonl(args.out / "valid.jsonl", [
        make_record(question, include_answer=True) for question in valid
    ])
    write_jsonl(args.out / "test.jsonl", [
        make_record(question, include_answer=False) for question in test
    ])

    train_domains = Counter(str(question["domain"]) for question in train)
    valid_domains = Counter(str(question["domain"]) for question in valid)
    test_domains = Counter(str(question["domain"]) for question in test)
    split = {
        "source": str(args.source),
        "seed": args.seed,
        "test_count": args.test_count,
        "valid_fraction_of_remaining": args.valid_fraction,
        "train_count": len(train),
        "valid_count": len(valid),
        "test_count_actual": len(test),
        "train": sorted(train_ids),
        "valid": sorted(valid_ids),
        "test": sorted(test_ids),
        "train_domains": dict(sorted(train_domains.items())),
        "valid_domains": dict(sorted(valid_domains.items())),
        "test_domains": dict(sorted(test_domains.items())),
    }
    write_json(args.out / "split.json", split)
    for name, ids_for_split in (
        ("train", train_ids), ("valid", valid_ids), ("test", test_ids)
    ):
        write_json(args.out / f"{name}_ids.json", sorted(ids_for_split))

    print(json.dumps({
        "source": len(questions),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "output": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()