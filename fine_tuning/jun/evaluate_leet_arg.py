"""Evaluate Llama 3.2 3B (base or fine-tuned) on the held-out 20% split.

Runs the model on every question in data/valid.jsonl, with greedy decoding,
using exactly the same system and user prompt as training. It then scores:

  parse_rate          share of questions where every statement got a verdict
  statement_accuracy  per-statement CORRECT/INCORRECT vs. the gold label
  question_accuracy   verdicts mapped back to a choice (1-5) vs. the gold answer

Results are saved after every question. If you stop it (Ctrl+C) and run the
same command again, it skips the questions already in --out.

The final test questions are never loaded here (the script refuses to run if
one appears in the split).

Usage (from fine_tuning/jun/):
  # fine-tuned
  python3 evaluate_leet_arg.py --adapter-path ./adapters_leet --out results/finetuned.json
  # base model, same prompt plus the answer format, for a fair comparison
  python3 evaluate_leet_arg.py --format-hint --out results/base_hint.json
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load

from leet_arg_format import (DATA_DIR, answer_from_verdicts, gold_verdicts,
                             load_dataset, load_final_test_ids, parse_verdicts)

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# The base model never saw the training answer format, so without this it is
# partly scored on formatting. Use --format-hint for a fair base-model run.
FORMAT_HINT = (
    "\n\nAnswer in exactly this format, one block per statement, using the "
    "statement's own label:\n\nStatement (a)\nReasoning: <your reasoning>\n"
    "Verdict: CORRECT or INCORRECT"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter-path", default=None, help="omit for the base model")
    ap.add_argument("--data", default=str(DATA_DIR / "valid.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (for a quick check)")
    ap.add_argument("--format-hint", action="store_true",
                    help="append the answer format to the system prompt (for the base model)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    leaked = load_final_test_ids() & {r["id"] for r in rows}
    if leaked:
        raise SystemExit(f"refusing to run: final test questions in {args.data}: {sorted(leaked)}")

    questions = {q["id"]: q for q in load_dataset()}
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    # Resume: keep finished questions from an earlier run with the same
    # model and adapter, so an interrupted run (Ctrl+C) does not start over.
    out = Path(args.out)
    records = []
    if out.exists():
        prev = json.load(open(out, encoding="utf-8"))
        same = (prev["summary"]["model"] == args.model
                and prev["summary"]["adapter_path"] == args.adapter_path
                and prev["summary"].get("format_hint", False) == args.format_hint)
        if not same:
            raise SystemExit(f"{out} is from a different model/adapter; use another --out")
        wanted = {r["id"] for r in rows}
        records = [r for r in prev["records"] if r["id"] in wanted]
        if records:
            print(f"resuming: {len(records)} questions already done in {out}")
    done = {r["id"] for r in records}

    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        q = questions[row["id"]]
        messages = [dict(m) for m in row["messages"][:2]]
        if args.format_hint:
            messages[0]["content"] += FORMAT_HINT
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
        gold = gold_verdicts(q)
        pred = parse_verdicts(output)
        pred_answer = answer_from_verdicts(q, pred) if len(pred) == len(gold) else None
        rec = {
            "id": q["id"],
            "gold_answer": int(q["answer"]),
            "pred_answer": pred_answer,
            "question_correct": pred_answer == int(q["answer"]),
            "parsed_all": all(l in pred for l in gold),
            "statements": {l: {"gold": gold[l], "pred": pred.get(l)} for l in gold},
            "seconds": round(time.time() - t0, 1),
            "output": output,
        }
        records.append(rec)
        save(out, args, records)  # after every question
        ok = sum(v["gold"] == v["pred"] for v in rec["statements"].values())
        print(f"[{i}/{len(rows)}] {q['id']}: statements {ok}/{len(gold)}, "
              f"answer {pred_answer} (gold {q['answer']}), {rec['seconds']}s", flush=True)

    print(json.dumps(save(out, args, records), indent=2))


def save(out, args, records):
    stmts = [v for r in records for v in r["statements"].values()]
    n = max(len(records), 1)
    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "format_hint": args.format_hint,
        "data": args.data,
        "n_questions": len(records),
        "n_statements": len(stmts),
        "parse_rate": sum(r["parsed_all"] for r in records) / n,
        "statement_accuracy": sum(v["gold"] == v["pred"] for v in stmts) / max(len(stmts), 1),
        "question_accuracy": sum(r["question_correct"] for r in records) / n,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)
    tmp.replace(out)  # never leaves a half-written file if interrupted mid-write
    return summary


if __name__ == "__main__":
    main()
