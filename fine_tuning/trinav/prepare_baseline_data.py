#!/usr/bin/env python3
"""Prepare full-dataset baseline inputs without exposing answer choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

try:
    from . import run_hf_questions
    from .mbe import prepare_mbe_data
except ImportError:
    import run_hf_questions
    from mbe import prepare_mbe_data


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_leetarg(source_path: Path, output_path: Path) -> int:
    questions = run_hf_questions.load_questions(source_path)
    rows = []
    for question in questions:
        prompt = run_hf_questions.build_user_prompt(question)
        if "<choices>" in prompt:
            raise ValueError(f"{question['id']}: baseline prompt contains choices")
        rows.append({"id": question["id"], "question": prompt})
    write_jsonl(output_path, rows)
    return len(rows)


def prepare_mbe(source_path: Path, output_path: Path) -> int:
    questions = read_json(source_path)
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array in {source_path}")

    rows = []
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Every MBE dataset entry must be a JSON object")
        prepare_mbe_data.validate_question(question)
        qid = prepare_mbe_data.question_id(question)
        if qid in seen:
            raise ValueError(f"Duplicate MBE question ID: {qid}")
        seen.add(qid)
        rows.append({
            "id": qid,
            "question": question["question"].strip(),
        })

    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leetarg-source",
        type=Path,
        default=REPO / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json",
    )
    parser.add_argument(
        "--mbe-source",
        type=Path,
        default=REPO / "benchmarks" / "MBE-exam-questions_distilled_dataset_cleaned.json",
    )
    parser.add_argument(
        "--leetarg-output",
        type=Path,
        default=REPO / "benchmarks" / "LEET_Arg_Baseline.jsonl",
    )
    parser.add_argument(
        "--mbe-output",
        type=Path,
        default=REPO / "benchmarks" / "MBE-exam-questions-baseline.jsonl",
    )
    args = parser.parse_args()

    leetarg_count = prepare_leetarg(args.leetarg_source, args.leetarg_output)
    mbe_count = prepare_mbe(args.mbe_source, args.mbe_output)
    print(json.dumps({
        "leetarg_questions": leetarg_count,
        "leetarg_output": str(args.leetarg_output.resolve()),
        "mbe_questions": mbe_count,
        "mbe_output": str(args.mbe_output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
