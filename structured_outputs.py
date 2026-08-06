from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


@dataclass
class PredicateOutput:
    infringing_product_available: bool
    substitute_product_available: bool

    def to_dict(self) -> dict:
        return {
            "infringing_product_available": self.infringing_product_available,
            "substitute_product_available": self.substitute_product_available,
        }


@dataclass
class DecisionOutput:
    decision: Literal["AWARDED", "DENIED"]
    explanation: str

    def to_dict(self) -> dict:
        return {"decision": self.decision, "explanation": self.explanation}


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _iter_json_objects(text: str):
    depth = 0
    start = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:index + 1]
                    start = None


def _normalize_predicates(payload: dict) -> dict:
    normalized = {}
    for key, value in payload.items():
        if key in {"infringing_product_available", "substitute_product_available"}:
            normalized[key] = value
        elif key.lower() in {"infringing_product_available", "substitute_product_available"}:
            normalized[key.lower()] = value
    return normalized


def parse_predicate_response(raw_response: str) -> PredicateOutput:
    """Parse the model's structured predicate response into a Pydantic model."""
    cleaned = _strip_code_fences(raw_response)

    for candidate in _iter_json_objects(cleaned):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            normalized = _normalize_predicates(payload)
            if {"infringing_product_available", "substitute_product_available"}.issubset(normalized.keys()):
                return PredicateOutput(
                    infringing_product_available=bool(normalized["infringing_product_available"]),
                    substitute_product_available=bool(normalized["substitute_product_available"]),
                )

    raise ValueError("Response JSON does not contain the required predicate fields")


def reason_with_dsl(predicates: PredicateOutput) -> DecisionOutput:
    """Apply a simple DSL-style rule over the structured predicates."""
    if predicates.substitute_product_available:
        return DecisionOutput(decision="DENIED", explanation="A non-infringing substitute is available, so the claim is denied")
    if predicates.infringing_product_available:
        return DecisionOutput(decision="AWARDED", explanation="No substitute is available and the infringing product is present, so the claim is awarded")
    return DecisionOutput(decision="DENIED", explanation="The infringing product is absent, so the claim is denied")


def parse_and_reason(raw_response: str) -> DecisionOutput:
    predicates = parse_predicate_response(raw_response)
    return reason_with_dsl(predicates)
