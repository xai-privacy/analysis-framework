#!/usr/bin/env python3
"""Evaluate base or LoRA-adapted Llama on a fixed LEET-Arg ID manifest.

Primary metric is strict final-answer accuracy. Generation is greedy because
MLX-LM defaults to greedy decoding when no sampler is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from mlx_lm import generate, load

SYSTEM_PROMPT = """You are solving LEET-Arg reasoning questions.
Use only the information in the question. Decide which answer choice is correct.
Do not invent facts that are not in the passage.
End your response with exactly one final line in this form:
Answer: <number from 1 to 5>"""

STRICT_RE = re.compile(r"(?:^|\n)Answer:\s*([1-5])\s*$", re.I)
LENIENT_RE = re.compile(r"\bAnswer\s*[:：-]?\s*([1-5])\b", re.I)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_answer(text):
    m = STRICT_RE.search(text.strip())
    if m:
        return int(m.group(1)), True
    ms = LENIENT_RE.findall(text)
    return (int(ms[-1]), False) if ms else (None, False)


def save(path: Path, meta, records):
    n = len(records)
    parsed = [r for r in records if r["pred_answer"] is not None]
    summary = {
        **meta,
        "n": n,
        "strict_format_rate": sum(r["strict_format"] for r in records) / n if n else 0.0,
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
    }

    records = []
    if out.exists():
        prev = read_json(out)
        if all(prev["summary"].get(k) == v for k, v in meta.items()):
            records = [r for r in prev["records"] if r["id"] in set(ids)]
            print(f"Resuming with {len(records)} completed questions")
        else:
            raise SystemExit(f"{out} belongs to a different evaluation configuration")
    done = {r["id"] for r in records}

    for idx, qid in enumerate(ids, 1):
        if qid in done:
            continue
        q = source[qid]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["original_question"].strip()},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        pred, strict = parse_answer(output)
        gold = int(q["answer"])
        rec = {
            "id": qid,
            "domain": q.get("domain"),
            "category": q.get("category", "UNKNOWN"),
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": pred == gold,
            "strict_format": strict,
            "seconds": round(time.time() - t0, 2),
            "output": output,
        }
        records.append(rec)
        summary = save(out, meta, records)
        print(f"[{idx}/{len(ids)}] {qid}: pred={pred} gold={gold} correct={rec['correct']} strict={strict}")

    print(json.dumps(save(out, meta, records), indent=2))


if __name__ == "__main__":
    main()
