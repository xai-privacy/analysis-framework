#!/usr/bin/env python3
"""
Clean LEET-Arg question statement segmentation.

Purpose
-------
This script fixes confirmed statement parsing errors in LEET_Arg_Questions.json.

Main fixes:
- Rebuilds statements only for confirmed auto-fix IDs.
- Uses the "next expected label only" segmentation rule.
- Fixes choice-based records like 2021_25 where statements are under <choices>.
- Manually patches 2025_05, where statements were null.
- Normalizes cosmetic "(a) ." -> "(a)." spacing for known cosmetic records.
- Avoids rewriting known false-positive DOT records unnecessarily.

Usage
-----
python tools/clean_leet_arg.py \
  --input data/leet_arg/LEET_Arg_Questions.json \
  --output data/leet_arg/LEET_Arg_Questions.cleaned.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# Records identified in issue #14 as auto-fixable segmentation errors.
AUTO_FIX_IDS = {
    "2021_25",
    "2021_34",
    "2022_19",
    "2023_11",
    "2023_17",
    "2023_29",
    "2024_27",
    "2025_17",
    "2025_39",
}

# Record identified in issue #14 as requiring manual reconstruction.
MANUAL_FIX_IDS = {"2025_05"}

# Records with cosmetic "(a) ." spacing artifacts.
COSMETIC_FIX_IDS = {
    "2021_13",
    "2022_08",
    "2023_29",
    "2024_06",
    "2024_22",
    "2024_29",
    "2025_23",
}

# Known false positives from the issue. We deliberately avoid rewriting them.
KNOWN_FALSE_POSITIVES = {
    "2022_36",
    "2023_19",
    "2024_25",
    "2025_21",
    "2025_01",
}


MANUAL_2025_05_STATEMENTS = {
    "statement_1": (
        "(a). If Alice, an adult, hands over a manual containing all information "
        "about her property to Bob, a minor who has graduated from college, and "
        "then enters into a sales contract with him, the control of this contract "
        "is not justified by A, but is justified by B."
    ),
    "statement_2": (
        "(b). If Charlie, a minor, sells his bicycle at a price much lower than "
        "the market price to David, an adult whom he has never met before through "
        "an online brokerage platform, the control of this contract is justified "
        "by A but not justified by B."
    ),
    "statement_3": (
        "(c). If adult Eve and State X both know all information regarding a "
        "certain piece of land and then enter into a sales contract regarding "
        "that land, the control of this contract is justified neither by A nor by B."
    ),
}


def normalize_statement_text(text: str) -> str:
    """
    Normalize cosmetic statement-label spacing without changing content meaning.

    Example:
    "(a) . If ..." -> "(a). If ..."
    """
    text = (text or "").strip()
    text = re.sub(r"^\(([a-z])\)\s+\.", r"(\1).", text)
    return text


def statement_count(statements: Optional[Dict[str, str]]) -> int:
    """Return statement count safely."""
    if not statements:
        return 0
    return len(statements)


def find_span_between_markers(
    text: str,
    start_marker: str,
    end_marker: str,
) -> Optional[str]:
    """
    Return text between two markers, or None if either marker is missing.
    """
    start = text.find(start_marker)
    if start == -1:
        return None

    start += len(start_marker)

    end = text.find(end_marker, start)
    if end == -1:
        return None

    span = text[start:end].strip()
    return span or None


def detect_label_scheme(span: str) -> Optional[List[str]]:
    """
    Detect label scheme used in the statement span.

    Supported schemes:
    - Circled numerals: ① ② ③ ④ ⑤
    - Letter labels: (a) (b) (c) (d) (e)

    Returns the full ordered label list for the detected scheme.
    """
    circled = ["①", "②", "③", "④", "⑤"]
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    circled_positions = [span.find(label) for label in circled if span.find(label) != -1]
    letter_positions = [span.find(label) for label in letters if span.find(label) != -1]

    first_circled = min(circled_positions) if circled_positions else -1
    first_letter = min(letter_positions) if letter_positions else -1

    if first_circled == -1 and first_letter == -1:
        return None

    if first_circled != -1 and (first_letter == -1 or first_circled < first_letter):
        return circled

    return letters


def split_only_on_expected_labels(span: str, labels: List[str]) -> Dict[str, str]:
    """
    Split only on the next expected label.

    This is the key rule from issue #14.

    Examples:
    - After "(a)", only "(b)" can start the next statement.
    - After "①", only "②" can start the next statement.

    This prevents false splits on:
    - "(B)" inside a sentence
    - "(2)" inside a sentence
    - "Group 2."
    - "$10."
    - "1.5% to 3.5%"
    - "chromosomes (n)"
    """
    positions = []
    search_start = 0

    for label in labels:
        pos = span.find(label, search_start)
        if pos == -1:
            break

        positions.append((label, pos))
        search_start = pos + len(label)

    statements: Dict[str, str] = {}

    for i, (_, pos) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(span)
        statement_text = span[pos:end].strip()
        statement_text = normalize_statement_text(statement_text)

        if statement_text:
            statements[f"statement_{i + 1}"] = statement_text

    return statements


def resegment_from_original_question(question: dict) -> Optional[Dict[str, str]]:
    """
    Rebuild statements from original_question.

    Handles three layouts:

    1. Statement-based questions:
       <statements> ... <choices>

    2. Choice-based questions:
       <choices>① ... ② ... ③ ...
       This fixes 2021_25.

    3. Unusual prefix layout:
       statements appear before <question>.
       This is kept as a fallback, although 2025_05 is manually patched.
    """
    text = question["original_question"]

    # Case 1: normal statement-based layout.
    span = find_span_between_markers(text, "<statements>", "<choices>")

    # Case 2: choices are the actual statements.
    # Important for 2021_25.
    if span is None:
        choices_marker = "<choices>"
        choices_pos = text.find(choices_marker)
        if choices_pos != -1:
            span = text[choices_pos + len(choices_marker):].strip()

    # Case 3: unusual layout where statements appear before <question>.
    if span is None:
        q_marker = "<question>"
        q_pos = text.find(q_marker)
        if q_pos != -1:
            prefix = text[:q_pos]

            candidates = [
                prefix.find("(a)"),
                prefix.find("(a)."),
                prefix.find("①"),
            ]
            candidates = [pos for pos in candidates if pos != -1]

            if candidates:
                span = prefix[min(candidates):].strip()

    if not span:
        return None

    labels = detect_label_scheme(span)
    if not labels:
        return None

    statements = split_only_on_expected_labels(span, labels)
    return statements or None


def normalize_existing_statements(statements: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Normalize cosmetic label spacing in an existing statements dict."""
    if not statements:
        return statements

    return {
        key: normalize_statement_text(value)
        for key, value in statements.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to LEET_Arg_Questions.json")
    parser.add_argument("--output", required=True, help="Path for cleaned JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    before_total = sum(statement_count(q.get("statements")) for q in data)

    changed = []
    manual_records = []
    warnings = []

    for q in data:
        qid = q["id"]
        old_statements = q.get("statements")
        old_count = statement_count(old_statements)

        if qid in MANUAL_FIX_IDS:
            q["statements"] = MANUAL_2025_05_STATEMENTS
            new_count = statement_count(q["statements"])
            changed.append((qid, old_count, new_count, "manual"))
            manual_records.append(qid)
            continue

        if qid in AUTO_FIX_IDS:
            rebuilt = resegment_from_original_question(q)

            if not rebuilt:
                warnings.append(f"{qid}: could not rebuild statements from original_question")
                continue

            q["statements"] = rebuilt
            new_count = statement_count(rebuilt)
            changed.append((qid, old_count, new_count, "auto"))
            continue

        if qid in COSMETIC_FIX_IDS:
            normalized = normalize_existing_statements(old_statements)
            if normalized != old_statements:
                q["statements"] = normalized
                new_count = statement_count(normalized)
                changed.append((qid, old_count, new_count, "cosmetic"))
            continue

        # Do not touch known false positives or unrelated records.

    after_total = sum(statement_count(q.get("statements")) for q in data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Questions: {len(data)}")
    print(f"Before statement total: {before_total}")
    print(f"After statement total: {after_total}")
    print()

    print("Changed records:")
    for qid, before, after, mode in changed:
        print(f"  {qid}: {before} -> {after} ({mode})")

    print()
    print("Manual records:", manual_records)

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    print()

    if after_total != 315:
        print(f"WARNING: expected 315 statement units, found {after_total}")
    else:
        print("Statement total looks correct: 315")


if __name__ == "__main__":
    main()
