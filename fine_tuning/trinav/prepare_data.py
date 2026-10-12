#!/usr/bin/env python3
"""Prepare leakage-safe LEET-Arg chat datasets for fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_PATH = (
    REPO / "benchmarks"
    / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
)
TEST_SET_PATH = (
    REPO / "testing" / "final_testing" / "test_set"
    / "LEET_Arg_Questions_Test_Set.json"
)

sys.path.insert(0, str(REPO))
from fine_tuning.jun.leet_arg_format import (  # noqa: E402
    SYSTEM_PROMPT,
    build_target,
    build_user_prompt,
    has_statement_rationales,
)

# 2021_13 was added after Jun's 93-question source snapshot.
SNAPSHOT_ONLY = {
    "2021_13": "added to the benchmark after Jun's 93-question source snapshot"
}
INCONSISTENT = {
    "2025_16": (
        "choice list contradicts the expert rationale "
        "(rationale: only (c) is correct; choice ② is (b))"
    )
}


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


def stratified_split(rows, valid_frac: float, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category", "UNKNOWN")].append(row)

    train, valid = [], []
    for category in sorted(groups):
        group = sorted(groups[category], key=lambda row: str(row["id"]))
        rng.shuffle(group)
        n_valid = round(len(group) * valid_frac)
        valid.extend(group[:n_valid])
        train.extend(group[n_valid:])

    return sorted(train, key=lambda row: str(row["id"])), sorted(
        valid, key=lambda row: str(row["id"])
    )


def make_record(question):
    return {
        "id": str(question["id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
            {"role": "assistant", "content": build_target(question)},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--test-set", type=Path, default=TEST_SET_PATH)
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    args = parser.parse_args()

    if not 0 < args.valid_frac < 1:
        parser.error("--valid-frac must be between 0 and 1")

    source = read_json(args.source)
    test_rows = read_json(args.test_set)
    source_ids = [str(question["id"]) for question in source]
    test_ids = {str(question["id"]) for question in test_rows}
    duplicates = sorted(qid for qid, count in Counter(source_ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate question IDs in source: {duplicates}")

    missing_test = sorted(test_ids - set(source_ids))
    if missing_test:
        raise SystemExit(f"Held-out test IDs missing from source: {missing_test}")

    excluded = {}
    trainable = []
    for question in source:
        qid = str(question["id"])
        if qid in test_ids:
            excluded[qid] = "final test question"
        elif qid in SNAPSHOT_ONLY:
            excluded[qid] = SNAPSHOT_ONLY[qid]
        elif qid in INCONSISTENT:
            excluded[qid] = INCONSISTENT[qid]
        elif not has_statement_rationales(question):
            excluded[qid] = "no per-statement expert rationale (diagram-format question)"
        else:
            trainable.append(question)

    train, valid = stratified_split(trainable, args.valid_frac, args.seed)
    train_ids = {str(question["id"]) for question in train}
    valid_ids = {str(question["id"]) for question in valid}
    if train_ids & valid_ids or (train_ids | valid_ids) & test_ids:
        raise SystemExit("Invalid split: overlap or held-out test leakage detected")
    if train_ids | valid_ids != {str(q["id"]) for q in trainable}:
        raise SystemExit("Invalid split: some eligible questions were dropped")

    write_jsonl(args.out / "train.jsonl", [make_record(question) for question in train])
    write_jsonl(args.out / "valid.jsonl", [make_record(question) for question in valid])

    write_json(args.out / "split.json", {
        "source": "benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json",
        "seed": args.seed,
        "valid_frac": args.valid_frac,
        "source_count": len(source),
        "heldout_test_count": len(test_ids),
        "train": [str(question["id"]) for question in train],
        "valid": [str(question["id"]) for question in valid],
        "excluded": excluded,
        "train_categories": dict(sorted(Counter(
            question.get("category", "UNKNOWN") for question in train
        ).items())),
        "valid_categories": dict(sorted(Counter(
            question.get("category", "UNKNOWN") for question in valid
        ).items())),
    })
    write_json(args.out / "valid_ids.json", sorted(valid_ids))
    write_json(args.out / "test_ids.json", sorted(test_ids))

    n_test = sum(reason == "final test question" for reason in excluded.values())
    print(json.dumps({
        "source": len(source),
        "final_test_removed": n_test,
        "other_exclusions": len(excluded) - n_test,
        "train": len(train),
        "valid": len(valid),
        "output": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()