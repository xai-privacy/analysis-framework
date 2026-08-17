"""LEET-Arg evaluation harness.

A six-stage pipeline for evaluating models on the LEET-Arg legal argument benchmark:

    Record -> build_prompt() -> ModelAdapter.generate() -> parse()
           -> Reasoner.reason() -> score()

Each stage has a fixed signature so that the plain-model and model+solver
conditions can be swapped independently. This package deliberately contains no
Colab-specific or environment-specific code paths; anything host-dependent
belongs in the CLI entry point or a model config.
"""

from .data import Record, load_records
from .gold import gold_choice, derive_statement_labels, verify_gold_derivation
from .prompts import build_prompt, DECOMPOSITIONS
from .adapters import ModelAdapter, HFAdapter, APIAdapter, StubAdapter
from .parse import ParsedAnswer, ParseStatus, parse
from .reason import Reasoner, Verdict, PassthroughReasoner
from .score import score, ResultRow

__all__ = [
    "Record",
    "load_records",
    "gold_choice",
    "derive_statement_labels",
    "verify_gold_derivation",
    "build_prompt",
    "DECOMPOSITIONS",
    "ModelAdapter",
    "HFAdapter",
    "APIAdapter",
    "StubAdapter",
    "ParsedAnswer",
    "ParseStatus",
    "parse",
    "Reasoner",
    "Verdict",
    "PassthroughReasoner",
    "score",
    "ResultRow",
]
