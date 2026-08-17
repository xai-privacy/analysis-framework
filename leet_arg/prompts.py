"""Prompt construction for LEET-Arg.

`build_prompt` is parameterised by `decomposition` because the published
LEET-Arg numbers use different protocols for different models. Results are only
comparable within a protocol, so the parameter is recorded on every result row
even though only `choice_level` is implemented today.
"""

from __future__ import annotations

import re

from .data import Record, SHAPE_THREE
from .gold import parse_choice_block, question_stem

CHOICE_LEVEL = "choice_level"
STATEMENT_LEVEL = "statement_level"
DECOMPOSITIONS = (CHOICE_LEVEL, STATEMENT_LEVEL)

SYSTEM_PROMPT = (
    "You are answering multiple-choice questions from a law school admission "
    "exam that tests argument analysis and evaluation. Answer with the single "
    "best option."
)

_INSTRUCTION = (
    "Read the passage and the question, then choose the single best option.\n"
    "Reply with exactly one line, in this form:\n"
    "ANSWER: <number>\n"
    "where <number> is one of 1, 2, 3, 4, 5."
)

_STATEMENT_BLOCK_RE = re.compile(r"<statements>\s*(?=\(a\)|\(1\)|①)", re.IGNORECASE)


def _context(record: Record) -> str:
    """The passage preceding the question stem, without the statement list.

    The statement list is re-rendered from `record.statements` instead of being
    carried through from the raw text, because one record (2025_05) has a
    malformed `<statements>` block while its statements dict is well formed.
    """
    question = record.original_question
    end = question.find("<choices>")
    region = question[:end] if end >= 0 else question

    match = _STATEMENT_BLOCK_RE.search(region)
    if match:
        region = region[: match.start()]

    stem_start = region.find("<question>")
    context = region[:stem_start] if stem_start >= 0 else region
    return context.strip()


def build_prompt(record: Record, decomposition: str = CHOICE_LEVEL) -> str:
    """Build the user-turn prompt for one record.

    Both structural shapes are handled explicitly: in the 3-statement shape the
    choices are combinations of sub-statements (a)/(b)/(c), which are listed
    separately; in the 5-statement shape each choice is a statement in its own
    right and no separate list is needed.
    """
    if decomposition not in DECOMPOSITIONS:
        raise ValueError(f"unknown decomposition {decomposition!r}; expected one of {DECOMPOSITIONS}")
    if decomposition == STATEMENT_LEVEL:
        # TODO: statement-level protocol -- ask the model to judge each statement
        # independently, then combine the judgements into a choice. Deliberately
        # not implemented for the plain-model baseline.
        raise NotImplementedError(
            "the statement_level decomposition is not implemented yet; "
            "only choice_level is available"
        )

    stem = " ".join(question_stem(record).split())
    choices = parse_choice_block(record)

    sections = [_INSTRUCTION, "", "PASSAGE:", _context(record), "", f"QUESTION: {stem}"]

    if record.shape == SHAPE_THREE:
        sections.extend(["", "STATEMENTS:"])
        sections.extend(record.ordered_statements())

    sections.extend(["", "CHOICES:"])
    for number in range(1, 6):
        sections.append(f"{number}. {' '.join(choices[number].split())}")

    sections.extend(["", "ANSWER:"])
    return "\n".join(sections)
