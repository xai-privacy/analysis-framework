#!/usr/bin/env python3
"""Evaluate base or LoRA-adapted Llama on LEET-Arg.

Changes from v1:
- Strict-format compliance is measured separately from answer extraction.
- Lenient parser understands ASCII digits and Unicode circled choices ①..⑤.
- It recognizes common base-model answer styles such as:
    ⑤
    ⑤ (a), (b), (c)
    Answer: ④
    Answer: ④ (b), (c)
    The correct answer is: ⑤ ...
- Ambiguous outputs with multiple candidate choices are left unparsed rather
  than silently choosing the last choice.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from mlx_lm import generate, load

SYSTEM_PROMPT = """You are solving LEET-Arg reasoning questions.
Use only the information in the question. Decide which answer choice is correct.
Do not invent facts that are not in the passage.
End your response with exactly one final line in this form:
Answer: <number from 1 to 5>"""

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
CHOICE_TOKEN = r"([1-5①②③④⑤])"

# Strict means EXACTLY the requested final line, using an ASCII digit.
STRICT_RE = re.compile(r"(?:^|\n)Answer:\s*([1-5])\s*$", re.I)

# Explicit Answer line, allowing a circled choice and optional choice text.
ANSWER_LINE_RE = re.compile(
    rf"(?im)^\s*Answer\s*[:：-]\s*{CHOICE_TOKEN}(?:\s|$)"
)

CORRECT_PHRASE_RE = re.compile(
    rf"(?i)(?:the\s+)?correct\s+(?:answer|choice|option|analysis)"
    rf"\s*(?:is|:)?\s*{CHOICE_TOKEN}"
)


def choice_to_int(token: str) -> int:
    return CIRCLED[token] if token in CIRCLED else int(token)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_answer(text: str):
    """Return (prediction, strict_format, parse_method).

    Accuracy parsing is intentionally more permissive than strict-format
    scoring. Strict-format scoring therefore remains a useful independent
    metric.
    """
    s = text.strip()
    strict = STRICT_RE.search(s)
    if strict:
        return int(strict.group(1)), True, "strict_answer_line"

    # Explicit Answer: line, but not exact requested format.
    matches = list(ANSWER_LINE_RE.finditer(s))
    if matches:
        return choice_to_int(matches[-1].group(1)), False, "answer_line_lenient"

    # "The correct answer is: ⑤"
    matches = list(CORRECT_PHRASE_RE.finditer(s))
    if matches:
        return choice_to_int(matches[-1].group(1)), False, "correct_answer_phrase"

    lines = [line.strip() for line in s.splitlines() if line.strip()]

    # A bare choice or choice-plus-set as first line is common for the base model.
    if lines:
        m = re.match(r"^([①②③④⑤])(?:\s|$|\()", lines[0])
        if m:
            return choice_to_int(m.group(1)), False, "first_line_circled"

    # "The correct options are:" followed by exactly ONE choice line.
    # If several candidate choices are listed, leave it unparsed.
    for i, line in enumerate(lines):
        if re.search(r"(?i)\bcorrect\s+(?:options?|answers?|analysis)\b", line):
            choices = []
            for nxt in lines[i + 1 : i + 5]:
                m = re.match(r"^([①②③④⑤])(?:\s|$|\()", nxt)
                if m:
                    choices.append(choice_to_int(m.group(1)))
                elif choices:
                    break
            if len(choices) == 1:
                return choices[0], False, "single_choice_after_correct_phrase"
            if len(choices) > 1:
                return None, False, "ambiguous_multiple_choices"

    return None, False, "unparsed"


def save(path: Path, meta, records):
    n = len(records)
    parsed = [r for r in records if r["pred_answer"] is not None]
    summary = {
        **meta,
        "n": n,
        "strict_format_rate": (
            sum(r["strict_format"] for r in records) / n if n else 0.0
        ),
        "parse_rate": len(parsed) / n if n else 0.0,
        "accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
    }

    cats = defaultdict(list)
    for r in records:
        cats[r["category"]].append(r)

    summary["per_category"] = {
        c: {
            "n": len(rs),
            "accuracy": sum(r["correct"] for r in rs) / len(rs),
            "strict_format_rate": sum(r["strict_format"] for r in rs) / len(rs),
            "parse_rate": sum(r["pred_answer"] is not None for r in rs) / len(rs),
        }
        for c, rs in sorted(cats.items())
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ids", required=True, help="JSON list of question IDs")
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    source = {str(q["id"]): q for q in read_json(args.source)}
    ids = [str(x) for x in read_json(args.ids)]
    missing = [x for x in ids if x not in source]
    if missing:
        raise SystemExit(f"IDs missing from source: {missing}")

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    out = Path(args.out)
    meta = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "ids_file": args.ids,
        "max_tokens": args.max_tokens,
        "parser_version": "v2",
    }

    records = []
    for idx, qid in enumerate(ids, 1):
        q = source[qid]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["original_question"].strip()},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        t0 = time.time()
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )

        pred, strict, method = parse_answer(output)
        gold = int(q["answer"])
        rec = {
            "id": qid,
            "domain": q.get("domain"),
            "category": q.get("category", "UNKNOWN"),
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": pred == gold,
            "strict_format": strict,
            "parse_method": method,
            "seconds": round(time.time() - t0, 2),
            "output": output,
        }
        records.append(rec)
        print(
            f"[{idx}/{len(ids)}] {qid}: pred={pred} gold={gold} "
            f"correct={rec['correct']} strict={strict} parser={method}"
        )

    print(json.dumps(save(out, meta, records), indent=2))


if __name__ == "__main__":
    main()
