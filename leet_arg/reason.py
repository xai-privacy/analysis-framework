"""The reasoning stage.

This is the seam the solver drops into. Today the only implementation is
`PassthroughReasoner`, which returns the choice the model already picked. It is
a real stage in the call path rather than an optimisation to be removed: keeping
it visible means the model+solver condition can be added later by supplying a
different `Reasoner` and touching no other stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # Protocol is stdlib from 3.8; keep the import defensive for older hosts.
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from .data import Record
from .parse import ParsedAnswer


@dataclass
class Verdict:
    """The harness's final answer for one record on one trial."""

    choice: Optional[int]
    status: str
    reasoner: str
    notes: str = ""


@runtime_checkable
class Reasoner(Protocol):
    def reason(self, parsed: ParsedAnswer, record: Record) -> Verdict: ...


class PassthroughReasoner:
    """Return the model's own choice unchanged.

    The plain-model condition of the 2x2: no deterministic reasoning is applied,
    so the verdict is exactly what the model said.
    """

    name = "passthrough"

    def reason(self, parsed: ParsedAnswer, record: Record) -> Verdict:
        return Verdict(
            choice=parsed.choice,
            status=parsed.status,
            reasoner=self.name,
            notes=parsed.evidence,
        )


# TODO: SolverReasoner -- map parsed predicates onto a Dung argumentation
# framework and compute extensions to select a choice. Separate workstream.
