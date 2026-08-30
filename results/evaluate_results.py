#!/usr/bin/env python3
"""Compare model answers with the LEET-Arg answer key.

By default, reads every JSON file in this directory and writes
``main_results.csv`` here. Each model is evaluated over every question in the
dataset; missing model records therefore count as unparseable responses.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RESULTS_DIR = Path(__file__).parent
DEFAULT_QUESTIONS = DEFAULT_RESULTS_DIR.parent / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "main_results.csv"

_CIRCLED_TO_NUMBER = {symbol: str(number) for number, symbol in enumerate("①②③④⑤", 1)}


def normalize_answer(value: Any) -> Optional[str]:
    """Return a valid answer number, or None when the value is unparseable.

    Letters such as "a"/"b"/"c" are not accepted here: in this dataset they
    always label sub-statements referenced inside a question, never a
    selectable choice, so a model answering with a bare letter has picked
    the wrong vocabulary rather than given an equivalent answer.
    """
    if value is None:
        return None

    answer = str(value).strip()
    if answer in _CIRCLED_TO_NUMBER:
        return _CIRCLED_TO_NUMBER[answer]
    if answer in {"1", "2", "3", "4", "5"}:
        return answer
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_model(result_path: Path, answer_key: Dict[str, str]) -> Dict[str, Any]:
    records = load_json(result_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {result_path}")

    # Keep the last record if a benchmark was resumed and an ID appears twice.
    answers_by_id = {record.get("id"): record.get("model_answer") for record in records}
    correct = 0
    unparseable = 0

    for question_id, correct_answer in answer_key.items():
        model_answer = normalize_answer(answers_by_id.get(question_id))
        if model_answer is None:
            unparseable += 1
        elif model_answer == correct_answer:
            correct += 1

    return {
        "model": result_path.stem,
        "correct_responses": correct,
        "incorrect_responses": len(answer_key) - correct - unparseable,
        "unparseable_responses": unparseable,
        "total_questions": len(answer_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    questions = load_json(args.questions)
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array in {args.questions}")
    answer_key = {question["id"]: normalize_answer(question["answer"]) for question in questions}
    if any(answer is None for answer in answer_key.values()):
        raise ValueError("The dataset contains an invalid answer key")

    result_paths = sorted(args.results_dir.glob("*.json"))
    summaries = [evaluate_model(path, answer_key) for path in result_paths]
    with args.output.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "correct_responses",
                "incorrect_responses",
                "unparseable_responses",
                "total_questions",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {args.output} for {len(summaries)} models")
    for summary in summaries:
        print(
            f"{summary['model']}: {summary['correct_responses']} correct, "
            f"{summary['incorrect_responses']} incorrect, "
            f"{summary['unparseable_responses']} unparseable, "
            f"{summary['total_questions']} total"
        )


if __name__ == "__main__":
    main()