#!/usr/bin/env python3
"""
Validate cleaned LEET-Arg question dataset.

Checks:
- 93 questions
- 301 total statement units
- no missing/null statements
- sequential statement keys
- no suspicious short statements
- no instruction leakage
- reasonable final punctuation
- relaxed source-presence check after cosmetic normalization

Usage
-----
python tools/validate_leet_arg.py \
  --input data/leet_arg/LEET_Arg_Questions.cleaned.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPECTED_QUESTION_COUNT = 93
EXPECTED_STATEMENT_COUNT = 301

# Known false positives from issue #14.
KNOWN_FALSE_POSITIVES = {
    "2022_36",
    "2023_19",
    "2024_25",
    "2025_21",
    "2025_01",
}

# Manual fix does not need to be byte-identical if we reconstructed it.
KNOWN_MANUAL = {"2025_05"}

# Cosmetic normalized records may not be byte-identical to original_question.
COSMETIC_FIX_IDS = {
    "2021_13",
    "2022_08",
    "2023_29",
    "2024_06",
    "2024_22",
    "2024_29",
    "2025_23",
}


def normalize_for_compare(s: str) -> str:
    """
    Normalize for relaxed comparison against original_question.

    This handles:
    - whitespace differences
    - "(a) ." vs "(a)."
    """
    s = s or ""

    # Normalize label-space-period artifact anywhere in the string.
    s = re.sub(r"\(([a-z])\)\s+\.", r"(\1).", s)

    # Normalize whitespace.
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def statement_count(statements: Optional[Dict[str, str]]) -> int:
    if not statements:
        return 0
    return len(statements)


def is_instruction_leak(statement: str) -> bool:
    """
    Detect obvious instruction text that should not be inside a statement.

    Avoid flagging normal use of the word "choose" because 2025_01 contains
    legitimate wording that triggered a false positive in the original lint pass.
    """
    lower = (statement or "").lower()

    suspicious = [
        "<statements>",
        "<choices>",
        "<question>",
        "which of the following",
        "choose all that apply",
        "select the most appropriate",
        "which statement",
        "which of the above",
    ]

    return any(token in lower for token in suspicious)


def has_final_punctuation(statement: str) -> bool:
    """
    Check for plausible final punctuation.

    Allow:
    - period/question/exclamation
    - closing brace for DOT graph
    - quotation marks
    - apostrophe
    """
    statement = (statement or "").strip()

    if not statement:
        return False

    return statement.endswith(
        (
            ".",
            "?",
            "!",
            "}",
            "}.",
            ".”",
            "’",
            "\"",
            "'",
            "”",
            "）",
            ")",
        )
    )


def sequential_keys_ok(statements: Dict[str, str]) -> bool:
    expected = [f"statement_{i}" for i in range(1, len(statements) + 1)]
    actual = list(statements.keys())
    return actual == expected


def validate_record(q: dict) -> List[Tuple[str, str]]:
    """
    Validate one question record.

    Returns list of (qid, message).
    """
    qid = q["id"]
    problems: List[Tuple[str, str]] = []

    statements = q.get("statements")

    if not statements:
        problems.append((qid, "missing statements"))
        return problems

    if not sequential_keys_ok(statements):
        problems.append((qid, f"non-sequential keys: {list(statements.keys())}"))

    original_norm = normalize_for_compare(q.get("original_question", ""))

    for key, stmt in statements.items():
        stmt = stmt or ""
        stmt_norm = normalize_for_compare(stmt)

        if len(stmt.strip()) < 10:
            problems.append((qid, f"{key} suspiciously short: {repr(stmt)}"))

        if is_instruction_leak(stmt):
            problems.append((qid, f"{key} possible instruction leak: {repr(stmt[:160])}"))

        if not has_final_punctuation(stmt):
            problems.append(
                (
                    qid,
                    f"{key} may be missing final punctuation: {repr(stmt[-80:])}",
                )
            )

        # Relaxed source consistency check.
        # Skip known manual reconstruction and known false positives.
        if (
            qid not in KNOWN_MANUAL
            and qid not in KNOWN_FALSE_POSITIVES
            and stmt_norm not in original_norm
        ):
            problems.append((qid, f"{key} not found verbatim in original_question"))

    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cleaned LEET_Arg_Questions JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    total_statements = sum(statement_count(q.get("statements")) for q in data)

    problems: List[Tuple[str, str]] = []

    for q in data:
        problems.extend(validate_record(q))

    print(f"Questions: {len(data)}")
    print(f"Total statements: {total_statements}")
    print(f"Problems found: {len(problems)}")

    for qid, msg in problems:
        print(f"{qid}: {msg}")

    if len(data) != EXPECTED_QUESTION_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, found {len(data)}"
        )

    if total_statements != EXPECTED_STATEMENT_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_STATEMENT_COUNT} statement units, found {total_statements}"
        )

    unexpected = [
        (qid, msg)
        for qid, msg in problems
        if qid not in KNOWN_FALSE_POSITIVES
        and qid not in KNOWN_MANUAL
        and qid not in COSMETIC_FIX_IDS
    ]

    if unexpected:
        raise SystemExit("Unexpected validation problems remain.")

    print("Validation passed.")


if __name__ == "__main__":
    main()
