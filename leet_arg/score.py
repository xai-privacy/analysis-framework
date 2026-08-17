"""Metrics and aggregation.

Accuracy alone is misleading for a small model on this benchmark, because
failing to emit a parseable answer and emitting a wrong answer are different
failures. Coverage is therefore reported alongside accuracy, and accuracy is
reported twice: over parseable responses and over all attempts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .data import Record
from .parse import ParseStatus


@dataclass
class ResultRow:
    """One trial of one record. Serialised verbatim to JSONL."""

    record_id: str
    model: str
    decomposition: str
    trial_index: int
    seed: int
    temperature: float
    raw_output: str
    parsed_choice: Optional[int]
    parse_status: str
    gold_choice: int
    correct: bool
    # Context that makes a row interpretable on its own.
    shape: str
    domain: Optional[str] = None
    category: Optional[str] = None
    polarity: str = ""
    reasoner: str = ""
    verdict_choice: Optional[int] = None
    adapter: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Metrics:
    total_attempts: int
    parseable: int
    correct: int
    accuracy_parseable: float
    accuracy_all: float
    coverage: float
    majority_baseline: float
    consistency: float
    consistency_parseable_only: float
    n_records: int
    n_trials: int
    parse_status_counts: Dict[str, int] = field(default_factory=dict)
    shape_counts: Dict[str, int] = field(default_factory=dict)
    accuracy_by_shape: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        def pct(value: float) -> str:
            return f"{value * 100:.1f}%"

        lines = [
            "=" * 70,
            "LEET-Arg results",
            "=" * 70,
            f"  records                    : {self.n_records}",
            f"  trials per record          : {self.n_trials}",
            f"  total attempts             : {self.total_attempts}",
            "",
            f"  accuracy (parseable only)  : {pct(self.accuracy_parseable)}  "
            f"({self.correct}/{self.parseable})",
            f"  accuracy (all attempts)    : {pct(self.accuracy_all)}  "
            f"({self.correct}/{self.total_attempts})",
            f"  coverage (parseable/total) : {pct(self.coverage)}  "
            f"({self.parseable}/{self.total_attempts})",
            f"  majority-class baseline    : {pct(self.majority_baseline)}   <- reference line",
            f"  consistency (all trials)   : {pct(self.consistency)}",
            f"  consistency (parseable)    : {pct(self.consistency_parseable_only)}",
            "",
            "  parse status breakdown:",
        ]
        for status, count in sorted(self.parse_status_counts.items(), key=lambda kv: -kv[1]):
            share = count / self.total_attempts if self.total_attempts else 0.0
            lines.append(f"    {status:<18}: {count:>4}  ({pct(share)})")

        if self.shape_counts:
            lines.append("")
            lines.append("  record shapes sampled:")
            for shape, count in sorted(self.shape_counts.items()):
                accuracy = self.accuracy_by_shape.get(shape)
                suffix = f"  accuracy(all)={pct(accuracy)}" if accuracy is not None else ""
                lines.append(f"    {shape:<18}: {count:>4} records{suffix}")

        lines.append("=" * 70)
        return "\n".join(lines)


def majority_class_baseline(records: Sequence[Record]) -> float:
    """Share of the most common gold answer across the given records."""
    if not records:
        return 0.0
    counts = Counter(record.answer for record in records)
    return counts.most_common(1)[0][1] / len(records)


def score(rows: Sequence[ResultRow], majority_baseline: float = 0.0) -> Metrics:
    """Aggregate per-trial rows into the headline metrics."""
    total = len(rows)
    parseable_rows = [row for row in rows if row.parse_status == ParseStatus.OK]
    parseable = len(parseable_rows)
    correct = sum(1 for row in rows if row.correct)

    by_record: Dict[str, List[ResultRow]] = defaultdict(list)
    for row in rows:
        by_record[row.record_id].append(row)

    consistent = 0
    consistent_parseable = 0
    parseable_eligible = 0
    for record_rows in by_record.values():
        # None (a parse failure) counts as its own value: five failures in a row
        # is a consistent model, even though it is a useless one.
        if len({row.verdict_choice for row in record_rows}) == 1:
            consistent += 1
        answered = [row.verdict_choice for row in record_rows if row.parse_status == ParseStatus.OK]
        if answered:
            parseable_eligible += 1
            if len(set(answered)) == 1:
                consistent_parseable += 1

    shape_counts: Dict[str, int] = {}
    for record_id, record_rows in by_record.items():
        shape = record_rows[0].shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    accuracy_by_shape: Dict[str, float] = {}
    for shape in shape_counts:
        shape_rows = [row for row in rows if row.shape == shape]
        if shape_rows:
            accuracy_by_shape[shape] = sum(1 for row in shape_rows if row.correct) / len(shape_rows)

    return Metrics(
        total_attempts=total,
        parseable=parseable,
        correct=correct,
        accuracy_parseable=(correct / parseable) if parseable else 0.0,
        accuracy_all=(correct / total) if total else 0.0,
        coverage=(parseable / total) if total else 0.0,
        majority_baseline=majority_baseline,
        consistency=(consistent / len(by_record)) if by_record else 0.0,
        consistency_parseable_only=(
            (consistent_parseable / parseable_eligible) if parseable_eligible else 0.0
        ),
        n_records=len(by_record),
        n_trials=max((len(v) for v in by_record.values()), default=0),
        parse_status_counts=dict(Counter(row.parse_status for row in rows)),
        shape_counts=shape_counts,
        accuracy_by_shape=accuracy_by_shape,
    )
