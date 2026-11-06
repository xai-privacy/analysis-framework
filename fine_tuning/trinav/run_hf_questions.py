#!/usr/bin/env python3
"""Run LEET-Arg test questions with a Hugging Face base model or PEFT adapter."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .chameleon_utils import HERE, load_cuda_model, parse_verdicts
except ImportError:
    from chameleon_utils import HERE, load_cuda_model, parse_verdicts

SYSTEM_PROMPT = (
    "You are an expert legal reasoning auditor. Analyze the debate provided, "
    "extract the core premises and inference rules for each statement, and "
    "declare a final verdict (CORRECT or INCORRECT)."
)
FORMAT_HINT = (
    "\n\nAnswer in exactly this format, one block per statement, using the "
    "statement's own label:\n\nStatement (a)\nReasoning: <your reasoning>\n"
    "Verdict: CORRECT or INCORRECT"
)
CIRCLED = "①②③④⑤"
NEGATIVE_STEM = re.compile(r"\bNOT\b|cannot be included", re.I)


def find_repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "benchmarks").is_dir():
            return candidate
    return HERE


REPO = find_repo_root()
DEFAULT_SPLIT = HERE / "data" / "split.json"
DEFAULT_OUTPUT_ROOT = REPO / "testing" / "baseline" / "huggingface"


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array in {path}")

    seen = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"Question at index {index} is not a JSON object")
        qid = question.get("id")
        if not isinstance(qid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", qid):
            raise ValueError(f"Question at index {index} has an invalid or missing id")
        if qid in seen:
            raise ValueError(f"Duplicate question id: {qid}")
        seen.add(qid)
        if not isinstance(question.get("original_question"), str):
            raise ValueError(f"Question {qid} has no string original_question")
        if not isinstance(question.get("statements"), dict) or not question["statements"]:
            raise ValueError(f"Question {qid} has no statement mapping")
        if "answer" not in question:
            raise ValueError(f"Question {qid} has no gold answer")
    return questions


def split_question(question: dict[str, Any]):
    text = question["original_question"]
    _, _, choices = text.partition("<choices>")
    first_statement = next(iter(question["statements"].values())).strip()
    cut = text.find(first_statement[:30], max(text.find("<question>"), 0))
    head = text[:cut] if cut > 0 else text.split("<choices>")[0]
    head = re.sub(r"(\s*<(statements|choices)>\s*)+$", "", head.rstrip())
    if "<question>" in head:
        passage, _, stem = head.rpartition("<question>")
    else:
        end = head.rfind("?")
        start = max(head.rfind(".", 0, end), head.rfind(">", 0, end)) + 1
        passage, stem = head[:start], head[start:]
    return passage.strip(), stem.replace("<choices>", "Options").strip(), choices.strip()


def build_user_prompt(question: dict[str, Any]) -> str:
    passage, stem, _ = split_question(question)
    passage = re.sub(r"(?<=[.!?”\"’)])\s*(?=[A-Z][a-z]+\d?:\s)", "\n", passage)
    statements = "\n".join(text.strip() for text in question["statements"].values())
    return f"{passage}\n\n<question> {stem}\n<statements>\n{statements}"


def statement_label(text: str):
    match = re.match(r"\s*(\(\w\)|[①②③④⑤])", text)
    return match.group(1) if match else None


def parse_choices(question: dict[str, Any]) -> dict[int, str]:
    _, _, choices = split_question(question)
    parts = re.split(r"([①②③④⑤])", choices)
    return {
        CIRCLED.index(parts[index]) + 1: parts[index + 1].strip().rstrip(".").strip()
        for index in range(1, len(parts), 2)
    }


def is_combination(question: dict[str, Any]) -> bool:
    labels = [statement_label(text) for text in question["statements"].values()]
    if not all(label and label.startswith("(") for label in labels):
        return False
    return all(
        re.fullmatch(r"(\(\w\)[,\s]*(and\s*)?)+", value)
        for value in parse_choices(question).values()
    )


def is_negative(question: dict[str, Any]) -> bool:
    _, stem, _ = split_question(question)
    return bool(NEGATIVE_STEM.search(stem))


def gold_verdicts(question: dict[str, Any]) -> dict[str, bool]:
    labels = [statement_label(text) for text in question["statements"].values()]
    answer = int(question["answer"])
    negative = is_negative(question)
    if is_combination(question):
        selected = set(re.findall(r"\(\w\)", parse_choices(question)[answer]))
        picked = [label in selected for label in labels]
    else:
        picked = [index + 1 == answer for index in range(len(labels))]
    return {label: (selected != negative) for label, selected in zip(labels, picked)}


def answer_from_verdicts(question: dict[str, Any], verdicts: dict[str, bool]):
    labels = [statement_label(text) for text in question["statements"].values()]
    negative = is_negative(question)
    if is_combination(question):
        wanted = {label for label in labels if verdicts.get(label) is (not negative)}
        for number, option in parse_choices(question).items():
            if set(re.findall(r"\(\w\)", option)) == wanted:
                return number
        return None
    matches = [
        index + 1 for index, label in enumerate(labels)
        if verdicts.get(label) is (not negative)
    ]
    return matches[0] if len(matches) == 1 else None


def read_split(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_ids = set(map(str, split.get("train", split.get("train_ids", []))))
    valid_ids = set(map(str, split.get("valid", split.get("valid_ids", []))))
    return train_ids, valid_ids


def load_test_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array of test IDs in {path}")
    ids = [str(row.get("id")) if isinstance(row, dict) else str(row) for row in rows]
    if any(not qid or qid == "None" for qid in ids):
        raise ValueError(f"Missing question ID in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate test IDs in {path}")
    return ids


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "run"


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    statement_rows = [
        statement for record in records for statement in record["statements"].values()
    ]
    total_statements = len(statement_rows)
    return {
        "n_questions": len(records),
        "n_statements": total_statements,
        "parse_rate": sum(record["parsed_all"] for record in records) / max(len(records), 1),
        "statement_accuracy": sum(
            row["gold"] == row["pred"] for row in statement_rows
        ) / max(total_statements, 1),
        "question_accuracy": sum(record["question_correct"] for record in records)
        / max(len(records), 1),
    }


def write_results(output_dir: Path, metadata: dict[str, Any], records: list[dict[str, Any]]):
    summary = {**metadata, **score_records(records)}
    temporary = output_dir / "results.json.tmp"
    temporary.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_dir / "results.json")
    return summary


def write_question_markdown(output_dir: Path, question: dict[str, Any], record: dict[str, Any]):
    qid = question["id"]
    statements = "\n".join(
        f"- {label}: gold={'CORRECT' if item['gold'] else 'INCORRECT'}, "
        f"predicted={('CORRECT' if item['pred'] else 'INCORRECT') if item['pred'] is not None else 'UNPARSED'}"
        for label, item in record["statements"].items()
    )
    original_rationale = question.get("original_rationale")
    gold_rationale = ""
    if isinstance(original_rationale, dict):
        answer_rationale = original_rationale.get("answer")
        if isinstance(answer_rationale, str) and answer_rationale.strip():
            gold_rationale = f"\n\n{answer_rationale.strip()}"
    content = (
        f"# Question {qid}\n\n"
        f"## Model run\n\n"
        f"- Model: `{record['model']}`\n"
        f"- Adapter: `{record['adapter_path'] or 'base model'}`\n\n"
        f"## Original question\n\n{question['original_question'].strip()}\n\n"
        f"## Gold answer\n\n{question['answer']}{gold_rationale}\n\n"
        f"## Scores\n\n"
        f"- Predicted answer: {record['pred_answer']}\n"
        f"- All statements parsed: {record['parsed_all']}\n"
        f"- Question correct: {record['question_correct']}\n"
        f"- Statement verdicts:\n{statements}\n\n"
        f"## Model response\n\n{record['output']}\n"
    )
    (output_dir / f"{qid}.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON test questions file")
    parser.add_argument(
        "--test-ids",
        type=Path,
        help="optional JSON list of IDs/records to select from --input",
    )
    parser.add_argument("--split", type=Path, default=HERE / "data" / "split.json")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, help="PEFT adapter; omit to test the base model")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format-hint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    questions = load_questions(args.input)
    if args.test_ids:
        requested_ids = load_test_ids(args.test_ids)
        questions_by_id = {question["id"]: question for question in questions}
        missing = sorted(set(requested_ids) - set(questions_by_id))
        if missing:
            raise SystemExit(f"Test IDs missing from --input: {missing}")
        questions = [questions_by_id[qid] for qid in requested_ids]

    train_ids, valid_ids = read_split(args.split)
    overlap = {question["id"] for question in questions} & (train_ids | valid_ids)
    if overlap:
        raise SystemExit(
            "Refusing to evaluate questions used for training or validation: "
            + ", ".join(sorted(overlap))
        )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        questions = questions[:args.limit]

    adapter_path = str(args.adapter_path) if args.adapter_path else None
    adapter_tag = args.adapter_path.name if args.adapter_path else "base"
    run_name = args.run_name or f"{sanitize(args.model)}__{sanitize(adapter_tag)}"
    output_dir = args.output_root / sanitize(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "adapter_path": adapter_path,
        "input": str(args.input.resolve()),
        "split": str(args.split.resolve()),
        "format_hint": args.format_hint,
        "max_tokens": args.max_tokens,
    }
    manifest_path = output_dir / "run.json"
    if manifest_path.exists():
        previous_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_metadata != metadata and not args.overwrite:
            raise SystemExit(
                f"{output_dir} contains a different run. Choose another --run-name "
                "or pass --overwrite."
            )
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results_path = output_dir / "results.json"
    records = []
    wanted_ids = {question["id"] for question in questions}
    if results_path.exists() and not args.overwrite:
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        records = [
            record for record in previous.get("records", [])
            if record.get("id") in wanted_ids
        ]
        print(f"Resuming with {len(records)} completed questions")
    done = {record["id"] for record in records}

    model, tokenizer, _ = load_cuda_model(args.model, adapter_path)
    model.eval()
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(
        f"Running {len(questions)} questions with {args.model} "
        f"({adapter_path or 'base model'}); output: {output_dir}"
    )
    for index, question in enumerate(questions, 1):
        qid = question["id"]
        if qid in done:
            continue
        system_prompt = SYSTEM_PROMPT + (FORMAT_HINT if args.format_hint else "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(question)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        start = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()

        gold = gold_verdicts(question)
        predicted = parse_verdicts(output)
        parsed_all = all(label in predicted for label in gold)
        predicted_answer = answer_from_verdicts(question, predicted) if parsed_all else None
        gold_answer = int(question["answer"])
        record = {
            "id": qid,
            "model": args.model,
            "adapter_path": adapter_path,
            "gold_answer": gold_answer,
            "pred_answer": predicted_answer,
            "question_correct": predicted_answer == gold_answer,
            "parsed_all": parsed_all,
            "statements": {
                label: {"gold": verdict, "pred": predicted.get(label)}
                for label, verdict in gold.items()
            },
            "seconds": round(time.time() - start, 2),
            "output": output,
        }
        records.append(record)
        write_question_markdown(output_dir, question, record)
        write_results(output_dir, metadata, records)
        statement_hits = sum(
            item["gold"] == item["pred"] for item in record["statements"].values()
        )
        print(
            f"[{index}/{len(questions)}] {qid}: statements {statement_hits}/{len(gold)}, "
            f"answer {predicted_answer} (gold {gold_answer}), {record['seconds']}s",
            flush=True,
        )

    print(json.dumps(write_results(output_dir, metadata, records), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
