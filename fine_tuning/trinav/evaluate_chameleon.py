#!/usr/bin/env python3
"""Evaluate a base or QLoRA-adapted model on Trinav's validation split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .chameleon_utils import (
        DATA_DIR,
        HERE,
        load_cuda_model,
        parse_verdicts,
        read_jsonl,
    )
    from .train_chameleon import DEFAULT_MODEL
except ImportError:
    from chameleon_utils import (
        DATA_DIR,
        HERE,
        load_cuda_model,
        parse_verdicts,
        read_jsonl,
    )
    from train_chameleon import DEFAULT_MODEL

FORMAT_HINT = (
    "\n\nAnswer in exactly this format, one block per statement, using the "
    "statement's own label:\n\nStatement (a)\nReasoning: <your reasoning>\n"
    "Verdict: CORRECT or INCORRECT"
)


def save_results(path: Path, args, records):
    statements = [value for row in records for value in row["statements"].values()]
    count = len(records)
    summary = {
        "model": args.model,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "format_hint": args.format_hint,
        "data": str(args.data),
        "n_questions": count,
        "n_statements": len(statements),
        "parse_rate": sum(row["parsed_all"] for row in records) / max(count, 1),
        "statement_accuracy": sum(
            value["gold"] == value["pred"] for value in statements
        ) / max(len(statements), 1),
        "question_accuracy": sum(row["question_correct"] for row in records)
        / max(count, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "valid.jsonl")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format-hint", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[:args.limit]
    test_ids = set()
    split_path = DATA_DIR / "split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = set(map(str, split.get("test_ids", [])))
        if not test_ids:
            test_ids = {
                qid for qid, reason in split.get("excluded", {}).items()
                if reason == "final test question"
            }
    leaked = test_ids & {str(row["id"]) for row in rows}
    if leaked:
        raise SystemExit(f"Refusing to evaluate final-test IDs: {sorted(leaked)}")
    if any(not isinstance(row.get("evaluation"), dict) for row in rows):
        raise SystemExit("Evaluation metadata missing; regenerate train/valid data first")

    model, tokenizer, _ = load_cuda_model(
        args.model,
        str(args.adapter_path) if args.adapter_path else None,
    )
    model.eval()
    metadata = {
        "model": args.model,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "format_hint": args.format_hint,
    }

    records = []
    if args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        old_summary = previous.get("summary", {})
        if not all(old_summary.get(key) == value for key, value in metadata.items()):
            raise SystemExit(f"{args.out} contains results for a different model or prompt")
        wanted = {str(row["id"]) for row in rows}
        records = [row for row in previous.get("records", []) if str(row["id"]) in wanted]
        if records:
            print(f"Resuming with {len(records)} completed questions")
    done = {str(row["id"]) for row in records}

    for index, row in enumerate(rows, 1):
        qid = str(row["id"])
        if qid in done:
            continue
        messages = [dict(message) for message in row["messages"][:2]]
        if args.format_hint:
            messages[0]["content"] += FORMAT_HINT

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()

        evaluation = row["evaluation"]
        gold = evaluation["gold_verdicts"]
        predicted = parse_verdicts(output)
        parsed_all = all(label in predicted for label in gold)
        matched_answers = [
            int(answer)
            for answer, answer_verdicts in evaluation["answer_verdicts"].items()
            if all(predicted.get(label) is verdict for label, verdict in answer_verdicts.items())
            and len(predicted) == len(answer_verdicts)
        ] if parsed_all else []
        predicted_answer = matched_answers[0] if len(matched_answers) == 1 else None
        gold_answer = int(evaluation["gold_answer"])
        record = {
            "id": qid,
            "gold_answer": gold_answer,
            "pred_answer": predicted_answer,
            "question_correct": predicted_answer == gold_answer,
            "parsed_all": parsed_all,
            "statements": {
                label: {"gold": verdict, "pred": predicted.get(label)}
                for label, verdict in gold.items()
            },
            "seconds": round(time.time() - start, 1),
            "output": output,
        }
        records.append(record)
        save_results(args.out, args, records)
        statement_hits = sum(
            value["gold"] == value["pred"] for value in record["statements"].values()
        )
        print(
            f"[{index}/{len(rows)}] {qid}: statements "
            f"{statement_hits}/{len(gold)}, answer {predicted_answer} "
            f"(gold {gold_answer}), {record['seconds']}s",
            flush=True,
        )

    print(json.dumps(save_results(args.out, args, records), indent=2))


if __name__ == "__main__":
    main()