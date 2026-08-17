"""Gold label derivation for LEET-Arg.

Two priorities, deliberately separated:

* **P0 (headline metric)** -- `gold_choice()` is just `int(record.answer)`. This
  is what choice-level accuracy is scored against and it cannot fail.
* **P1 (statement level)** -- `derive_statement_labels()` parses the `<choices>`
  block into the set of sub-statements each choice asserts, then combines that
  with the gold answer and the stem polarity to label every statement true or
  false.

P1 refuses to guess. Any record whose derivation is ambiguous raises
`AmbiguousDerivation` and is reported by id, because labels that are quietly
wrong for a handful of records are worse than labels that are missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

from .data import CHOICE_MARKERS, Record, SHAPE_THREE

POSITIVE = "positive"
NEGATIVE = "negative"

# Letters used for sub-statements in the 3-statement shape.
_LETTERS = {"a": 1, "b": 2, "c": 3}
_LETTER_RE = re.compile(r"\(([a-c])\)")

# Negative-polarity stems ask which option is *not* appropriate, which inverts
# the statement-level mapping. The benchmark writes the negation in a small
# number of fixed forms; "NOT" is capitalised for emphasis in the source.
_NEGATION_RE = re.compile(r"\bNOT\b|\bcannot\b|inappropriate")

# A statement list block opens with (a), (1) or the circled numeral. Used to cut
# the interrogative stem away from the enumerated statements, so that a "not"
# occurring inside statement text is never mistaken for stem polarity.
_STATEMENT_BLOCK_RE = re.compile(r"<statements>\s*(?=\(a\)|\(1\)|①)", re.IGNORECASE)


class AmbiguousDerivation(ValueError):
    """Raised when statement-level gold labels cannot be derived unambiguously."""


@dataclass
class GoldLabels:
    """Derived gold information for one record."""

    record_id: str
    choice: int
    polarity: str
    choice_sets: Dict[int, FrozenSet[int]]
    statement_labels: Dict[int, bool]

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "gold_choice": self.choice,
            "polarity": self.polarity,
            "choice_sets": {k: sorted(v) for k, v in sorted(self.choice_sets.items())},
            "statement_labels": {k: v for k, v in sorted(self.statement_labels.items())},
        }


# --------------------------------------------------------------------------- #
# P0
# --------------------------------------------------------------------------- #


def gold_choice(record: Record) -> int:
    """The gold choice, 1..5. `answer` is a string in the source data."""
    return int(record.answer)


# --------------------------------------------------------------------------- #
# P1 helpers
# --------------------------------------------------------------------------- #


def question_stem(record: Record) -> str:
    """The interrogative stem, with the enumerated statement block removed.

    Everything between `<question>` and `<choices>`, truncated at the statement
    list if one is present. Polarity must be read from this and nothing else.
    """
    question = record.original_question
    end = question.find("<choices>")
    start = question.find("<question>")
    region = question[start + len("<question>") : end] if start >= 0 else question[:end]
    match = _STATEMENT_BLOCK_RE.search(region)
    return region[: match.start()] if match else region


def detect_polarity(record: Record) -> str:
    """Whether the stem asks for the correct option or the incorrect one."""
    return NEGATIVE if _NEGATION_RE.search(question_stem(record)) else POSITIVE


def parse_choice_block(record: Record) -> Dict[int, str]:
    """Split the `<choices>` block into {choice_number: raw text}.

    Fails loudly rather than returning a partial map: a missing or duplicated
    marker means the split is unreliable for that record.
    """
    question = record.original_question
    start = question.find("<choices>")
    if start < 0:
        raise AmbiguousDerivation(f"{record.id}: no <choices> block")
    block = question[start + len("<choices>") :]

    positions: List[Tuple[int, int]] = []
    for number, marker in enumerate(CHOICE_MARKERS, start=1):
        occurrences = [m.start() for m in re.finditer(re.escape(marker), block)]
        if not occurrences:
            raise AmbiguousDerivation(f"{record.id}: choice marker {marker} not found in <choices>")
        positions.append((occurrences[0], number))

    positions.sort()
    if [number for _, number in positions] != [1, 2, 3, 4, 5]:
        raise AmbiguousDerivation(
            f"{record.id}: choice markers are out of order in <choices>; refusing to guess"
        )

    choices: Dict[int, str] = {}
    for index, (offset, number) in enumerate(positions):
        marker_len = len(CHOICE_MARKERS[number - 1])
        end = positions[index + 1][0] if index + 1 < len(positions) else len(block)
        choices[number] = block[offset + marker_len : end].strip()
    return choices


def choice_statement_sets(record: Record) -> Dict[int, FrozenSet[int]]:
    """Map each choice to the set of statement indices it asserts.

    5-statement shape: choice k asserts statement k.
    3-statement shape: the choice text is a combination such as "(a), (c)",
    which maps to statement indices via a=1, b=2, c=3.
    """
    if record.shape != SHAPE_THREE:
        return {number: frozenset({number}) for number in range(1, 6)}

    choices = parse_choice_block(record)
    sets: Dict[int, FrozenSet[int]] = {}
    for number, text in choices.items():
        letters = _LETTER_RE.findall(text.lower())
        if not letters:
            raise AmbiguousDerivation(
                f"{record.id}: choice {number} has no (a)/(b)/(c) reference: {text!r}"
            )
        if len(set(letters)) != len(letters):
            raise AmbiguousDerivation(
                f"{record.id}: choice {number} repeats a sub-statement: {text!r}"
            )
        # Guard against the choice text carrying trailing prose rather than a
        # bare combination; a stray sentence would mean we parsed the wrong span.
        residue = _LETTER_RE.sub("", text).strip(" ,.;:()、，")
        if residue:
            raise AmbiguousDerivation(
                f"{record.id}: choice {number} has unexpected text beyond the "
                f"sub-statement combination: {residue!r}"
            )
        sets[number] = frozenset(_LETTERS[letter] for letter in letters)

    if len(set(sets.values())) != len(sets):
        raise AmbiguousDerivation(f"{record.id}: two choices assert the same sub-statement set")
    return sets


def derive_statement_labels(record: Record) -> GoldLabels:
    """Derive per-statement truth labels (P1).

    For a positive-polarity stem the statements in the gold choice's set are
    true and the rest are false. For a negative-polarity stem the gold choice
    identifies the *incorrect* option, so the mapping inverts.
    """
    choice = gold_choice(record)
    sets = choice_statement_sets(record)
    if choice not in sets:
        raise AmbiguousDerivation(f"{record.id}: gold answer {choice} is not among the parsed choices")

    polarity = detect_polarity(record)
    asserted = sets[choice]

    if record.shape == SHAPE_THREE:
        if polarity == NEGATIVE:
            # No 3-statement record in v1 has a negative stem. If the data ever
            # grows one, the combination semantics ("which combination is NOT
            # correct") do not reduce to a per-statement inversion, so refuse.
            raise AmbiguousDerivation(
                f"{record.id}: negative-polarity stem in the 3-statement shape has no "
                "unambiguous statement-level reading"
            )
        indices = range(1, 4)
        labels = {index: (index in asserted) for index in indices}
    else:
        indices = range(1, 6)
        if polarity == NEGATIVE:
            # The gold choice is the one statement that is *not* correct.
            labels = {index: (index not in asserted) for index in indices}
        else:
            labels = {index: (index in asserted) for index in indices}

    return GoldLabels(
        record_id=record.id,
        choice=choice,
        polarity=polarity,
        choice_sets=sets,
        statement_labels=labels,
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


@dataclass
class VerificationReport:
    total: int
    passed: List[str]
    failures: List[Tuple[str, str]]
    polarity_counts: Dict[str, int]
    shape_counts: Dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [
            "Gold derivation verification",
            f"  records            : {self.total}",
            f"  P1 derived         : {len(self.passed)}",
            f"  P1 failed          : {len(self.failures)}",
            "  shapes             : " + ", ".join(f"{k}={v}" for k, v in sorted(self.shape_counts.items())),
            "  stem polarity      : " + ", ".join(f"{k}={v}" for k, v in sorted(self.polarity_counts.items())),
        ]
        if self.failures:
            lines.append("  FAILED record ids:")
            for record_id, reason in self.failures:
                lines.append(f"    - {record_id}: {reason}")
        return "\n".join(lines)


def verify_gold_derivation(records: Sequence[Record]) -> VerificationReport:
    """Run P1 derivation over every record and report successes and failures.

    Never swallows a failure: every record that cannot be derived is listed by
    id with the reason.
    """
    passed: List[str] = []
    failures: List[Tuple[str, str]] = []
    polarity_counts: Dict[str, int] = {}
    shape_counts: Dict[str, int] = {}

    for record in records:
        shape_counts[record.shape] = shape_counts.get(record.shape, 0) + 1
        try:
            polarity = detect_polarity(record)
            polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1
            derive_statement_labels(record)
        except AmbiguousDerivation as exc:
            failures.append((record.id, str(exc)))
        except Exception as exc:  # unexpected shape problems are still failures
            failures.append((record.id, f"{type(exc).__name__}: {exc}"))
        else:
            passed.append(record.id)

    return VerificationReport(
        total=len(records),
        passed=passed,
        failures=failures,
        polarity_counts=polarity_counts,
        shape_counts=shape_counts,
    )
