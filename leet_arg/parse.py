"""Parsing raw model text into a choice.

A small model frequently fails to emit a parseable answer at all, and "failed to
produce an answer" is a different finding from "produced a wrong answer". The
status is therefore first-class: `ok`, `no_choice_found` and `ambiguous` are
reported separately rather than collapsed into a single failure bucket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


class ParseStatus:
    OK = "ok"
    NO_CHOICE_FOUND = "no_choice_found"
    AMBIGUOUS = "ambiguous"


@dataclass
class ParsedAnswer:
    """The choice a model committed to, if any."""

    choice: Optional[int]
    status: str
    evidence: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == ParseStatus.OK


_CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}

# Tier 1: the requested output format.
_EXPLICIT_RE = re.compile(r"ANSWER\s*[:\-]?\s*\(?\s*([1-5])", re.IGNORECASE)
# Tier 2: a natural-language commitment to an option.
_PHRASE_RE = re.compile(
    r"(?:answer|option|choice|correct)\b[^\n0-9]{0,20}?([1-5])",
    re.IGNORECASE,
)
# Tier 3: any bare option number.
_BARE_RE = re.compile(r"(?<![0-9.])([1-5])(?![0-9])")


def _normalise(text: str) -> str:
    for marker, digit in _CIRCLED.items():
        text = text.replace(marker, digit)
    return text


def _decide(candidates: List[str], evidence: str) -> Optional[ParsedAnswer]:
    if not candidates:
        return None
    distinct = sorted(set(candidates))
    if len(distinct) == 1:
        return ParsedAnswer(choice=int(distinct[0]), status=ParseStatus.OK, evidence=evidence)
    return ParsedAnswer(
        choice=None,
        status=ParseStatus.AMBIGUOUS,
        evidence=f"{evidence}: conflicting options {distinct}",
    )


def parse(raw_output: str) -> ParsedAnswer:
    """Turn raw model text into a `ParsedAnswer`.

    Tiers are tried in order of how strongly they indicate a deliberate answer.
    The first tier that finds anything decides the outcome, so a stray digit
    later in the text cannot override an explicit `ANSWER:` line.
    """
    if raw_output is None:
        return ParsedAnswer(choice=None, status=ParseStatus.NO_CHOICE_FOUND, evidence="empty output")

    text = _normalise(raw_output).strip()
    if not text:
        return ParsedAnswer(choice=None, status=ParseStatus.NO_CHOICE_FOUND, evidence="empty output")

    for pattern, label in (
        (_EXPLICIT_RE, "explicit ANSWER line"),
        (_PHRASE_RE, "answer phrase"),
        (_BARE_RE, "bare option number"),
    ):
        parsed = _decide(pattern.findall(text), label)
        if parsed is not None:
            return parsed

    return ParsedAnswer(
        choice=None,
        status=ParseStatus.NO_CHOICE_FOUND,
        evidence="no option number in output",
    )
