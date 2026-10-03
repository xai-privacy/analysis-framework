"""Re-score saved evaluation results without running the model again.

Prints two scores per file:
  strict   only answers in the training format count (what evaluate_leet_arg.py reports)
  lenient  any clear CORRECT/INCORRECT verdict counts, whatever the format

Usage (from fine_tuning/jun/):
  python3 rescore_results.py results/base.json results/finetuned.json
"""

import json
import sys

from leet_arg_format import (answer_from_verdicts, gold_verdicts, load_dataset,
                             parse_verdicts, parse_verdicts_lenient)


def score(records, questions, lenient):
    n_stmt = ok_stmt = ok_q = full = 0
    rows = []
    for r in records:
        q = questions[r["id"]]
        gold = gold_verdicts(q)
        pred = (parse_verdicts_lenient(r["output"], set(gold)) if lenient
                else parse_verdicts(r["output"]))
        ans = answer_from_verdicts(q, pred) if all(l in pred for l in gold) else None
        hits = sum(pred.get(l) is g for l, g in gold.items())
        n_stmt += len(gold); ok_stmt += hits
        ok_q += ans == int(q["answer"]); full += all(l in pred for l in gold)
        rows.append((r["id"], hits, len(gold), ans, int(q["answer"]),
                     "".join({True: "C", False: "I", None: "-"}[pred.get(l)] for l in gold),
                     "".join("C" if g else "I" for g in gold.values())))
    n = len(records)
    return {"parse_rate": full / n, "statement_accuracy": ok_stmt / n_stmt,
            "statements": f"{ok_stmt}/{n_stmt}", "question_accuracy": ok_q / n,
            "questions": f"{ok_q}/{n}"}, rows


def main():
    questions = {q["id"]: q for q in load_dataset()}
    for path in sys.argv[1:]:
        records = json.load(open(path, encoding="utf-8"))["records"]
        print(f"\n== {path}")
        for mode in (False, True):
            s, rows = score(records, questions, mode)
            print(f"{'lenient' if mode else 'strict ':8} parse {s['parse_rate']:.2f} | "
                  f"statements {s['statements']} = {s['statement_accuracy']:.3f} | "
                  f"questions {s['questions']} = {s['question_accuracy']:.3f}")
        print(f"{'id':8} {'ok':>5}  pred  gold  answer")
        for rid, hits, tot, ans, gold_ans, pv, gv in rows:  # lenient rows
            print(f"{rid:8} {hits}/{tot}  {pv:5} {gv:5} {ans} (gold {gold_ans})")


if __name__ == "__main__":
    main()
