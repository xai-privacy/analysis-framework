"""Dataset loading and validation for LEET-Arg.

The cleaned dataset is a JSON list of records. Validation runs at load time and
fails loudly: a record that does not match the expected shape is a data problem
worth stopping for, not something to silently skip.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "leet_arg",
    "leet_arg_clean_v1.json",
)

REQUIRED_KEYS = (
    "id",
    "year",
    "problem_idx",
    "objective",
    "domain",
    "category",
    "answer",
    "original_question",
    "statements",
    "original_rationale",
)

# The five choice markers used throughout the benchmark.
CHOICE_MARKERS = ("①", "②", "③", "④", "⑤")  # (1)..(5)

# The dataset has exactly two structural shapes.
SHAPE_THREE = "3_statement"
SHAPE_FIVE = "5_statement"


class DatasetError(ValueError):
    """Raised when a record does not match the documented dataset contract."""


@dataclass
class Record:
    """One LEET-Arg problem.

    `statements` maps "statement_1".."statement_N" to statement text, with the
    original list marker retained as a prefix ("(a) ..." for the 3-statement
    shape, "(1) ..." or the circled numeral for the 5-statement shape).
    """

    id: str
    year: str
    problem_idx: str
    objective: str
    domain: Optional[str]
    category: str
    answer: str
    original_question: str
    statements: Dict[str, str]
    original_rationale: str
    raw: Dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def n_statements(self) -> int:
        return len(self.statements)

    @property
    def shape(self) -> str:
        """Structural shape of the record.

        In the 3-statement shape the five choices are combinations of the
        sub-statements (a)/(b)/(c). In the 5-statement shape choice k simply is
        statement k. Prompt building and gold derivation branch on this.
        """
        return SHAPE_THREE if self.n_statements == 3 else SHAPE_FIVE

    def statement_text(self, index: int) -> str:
        """Text of statement `index` (1-based)."""
        return self.statements[f"statement_{index}"]

    def ordered_statements(self) -> List[str]:
        return [self.statement_text(i) for i in range(1, self.n_statements + 1)]


def _validate(payload: dict, position: int) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise DatasetError(
            f"record at position {position} is missing required keys: {missing}"
        )

    record_id = payload["id"]

    statements = payload["statements"]
    if not isinstance(statements, dict):
        raise DatasetError(f"{record_id}: 'statements' must be a dict, got {type(statements).__name__}")
    if len(statements) not in (3, 5):
        raise DatasetError(
            f"{record_id}: expected 3 or 5 statements, got {len(statements)}. "
            "Only these two structural shapes are supported."
        )
    for index in range(1, len(statements) + 1):
        key = f"statement_{index}"
        if key not in statements:
            raise DatasetError(f"{record_id}: statements missing key '{key}'")

    # `answer` is a string "1".."5" in the source data, not an int.
    answer = payload["answer"]
    if not isinstance(answer, str) or answer not in {"1", "2", "3", "4", "5"}:
        raise DatasetError(f"{record_id}: 'answer' must be a string '1'..'5', got {answer!r}")

    question = payload["original_question"]
    if not isinstance(question, str) or "<choices>" not in question:
        raise DatasetError(f"{record_id}: 'original_question' has no <choices> block")
    absent = [marker for marker in CHOICE_MARKERS if marker not in question]
    if absent:
        raise DatasetError(f"{record_id}: <choices> block is missing markers {absent}")


def load_records(path: Optional[str] = None, validate: bool = True) -> List[Record]:
    """Load and validate the cleaned LEET-Arg dataset."""
    data_path = path or DEFAULT_DATA_PATH
    if not os.path.isfile(data_path):
        raise DatasetError(
            f"dataset not found at {data_path}. The cleaned dataset is committed "
            "separately; see data/leet_arg/README.md for provenance."
        )

    with open(data_path, "r", encoding="utf-8") as handle:
        payloads = json.load(handle)

    if not isinstance(payloads, list):
        raise DatasetError(f"expected a JSON list at {data_path}, got {type(payloads).__name__}")

    records: List[Record] = []
    for position, payload in enumerate(payloads):
        if validate:
            _validate(payload, position)
        records.append(
            Record(
                id=payload["id"],
                year=payload["year"],
                problem_idx=payload["problem_idx"],
                objective=payload["objective"],
                domain=payload["domain"],  # may be None for 10 records
                category=payload["category"],
                answer=payload["answer"],
                original_question=payload["original_question"],
                statements=payload["statements"],
                original_rationale=payload["original_rationale"],
                raw=payload,
            )
        )
    return records
