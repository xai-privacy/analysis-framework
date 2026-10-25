#!/usr/bin/env python3
"""Run LEET-Arg questions sequentially through a local Ollama chat API."""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = (
    ROOT / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "baseline"
MAX_OUTPUT_TOKENS = 3000
QUESTION_PROMPT = (
    "Solve the problem and provide explanations for your solution. Select the most appropriate "
    "option from <choices> to answer the <question>. Do not repeat the statements. Answer only "
    "in English using this format: Answer-{choice}. Rationale 1. {explanation} "
    "2. {explanation} ..."
)


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array in {path}")

    seen_ids = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"Question at index {index} is not a JSON object")

        question_id = question.get("id")
        if not isinstance(question_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", question_id):
            raise ValueError(f"Question at index {index} has an invalid or missing id")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)

        if not isinstance(question.get("original_question"), str):
            raise ValueError(f"Question {question_id} has no string original_question")
        if "answer" not in question:
            raise ValueError(f"Question {question_id} has no answer")

    return questions


def render_original_answer(question: Dict[str, Any]) -> str:
    answer = question["answer"]
    rendered = f"**Answer choice:** {answer}"

    rationale = question.get("original_rationale")
    if isinstance(rationale, dict):
        explanation = rationale.get("answer")
        if isinstance(explanation, str) and explanation.strip():
            rendered += f"\n\n{explanation.strip()}"

    return rendered


def call_ollama(url: str, model: str, original_question: str, timeout: int) -> str:
    request_body = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{QUESTION_PROMPT}\n\n{original_question}"}
        ],
        "stream": False,
        "options": {"num_predict": MAX_OUTPUT_TOKENS},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error

    try:
        content = result["message"]["content"]
    except (KeyError, TypeError) as error:
        raise ValueError("Ollama response did not contain message.content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response")
    return content.strip()


def write_question(
    question: Dict[str, Any],
    output_dir: Path,
    model: str,
    url: str,
    timeout: int,
) -> Path:
    question_id = question["id"]
    output_path = output_dir / f"{question_id}.md"
    content = (
        f"# Question {question_id}\n\n"
        f"## Original Question\n\n{question['original_question'].strip()}\n\n"
        f"## Original Answer\n\n{render_original_answer(question)}\n\n"
        "## LLM Response\n\n"
    )
    output_path.write_text(content, encoding="utf-8")

    response = call_ollama(url, model, question["original_question"], timeout)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(response + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_QUESTIONS, help="Question JSON file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--timeout", type=int, default=900, help="Request timeout in seconds")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    try:
        questions = load_questions(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {len(questions)} questions with {args.model}")
        for index, question in enumerate(questions, start=1):
            path = write_question(
                question, args.output_dir, args.model, args.url, args.timeout
            )
            print(f"[{index}/{len(questions)}] Wrote {path}")
    except (OSError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
