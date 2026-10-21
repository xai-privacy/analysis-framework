#!/usr/bin/env python3
"""Rescore saved eval_aspic.py results, with and without a bracket repair.

No model runs here: it reads the outputs already saved in eval_results/*.json.

The repair is mechanical and the same for every model:
  - it is applied only to outputs that ended on their own (the last character
    is } or ]). Outputs cut off at the token limit, usually in a repetition
    loop, are left as they are, since their missing statements cannot be
    recovered.
  - a closing bracket of the wrong kind is replaced with the expected one
    (for example "]]}}" where "}]}}" was meant)
  - missing closing brackets at the end are added
Nothing else is changed: no quotes, commas or content.

Columns:
  valid          as in eval_aspic.py, raw outputs
  valid+rep      after the repair
  stmt           statements right, out of all statements (invalid = wrong)
  stmt+rep       the same after the repair
  stmt in valid  statements right, out of the statements of valid translations
                 only (after the repair). This is the judgment part: is the
                 structure right when the format is right?
  rule in valid  what "call every statement false" gets on exactly the same
                 statements. Compare it with stmt in valid: each model's valid
                 translations cover different statements, so this is the fair
                 reference for that model
  questions+rep  computed answer = gold, after the repair

Usage (from fine_tuning/jun/aspic/, venv active):
  python rescore_aspic.py                     # every file in eval_results/
  python rescore_aspic.py --show-repairs      # also list which outputs were repaired
"""
import argparse
import json
import sys
from pathlib import Path

from check_aspic import load_bench
from eval_aspic import BENCH, SPLIT, pct, score

HERE = Path(__file__).resolve().parent
PAIR = {"{": "}", "[": "]"}


def repair(raw):
    """Return (text, changes) or (None, reason) when the output is not repaired."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0].strip()
    start = s.find("{")
    if start < 0:
        return None, "no JSON object"
    s = s[start:].rstrip()
    if not s or s[-1] not in "}]":
        return None, "cut off (does not end with } or ])"
    out, stack, changes = [], [], []
    in_str = esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in PAIR:
            stack.append(PAIR[ch])
        elif ch in "}]":
            if not stack:
                changes.append(f"removed extra {ch}")
                continue
            want = stack.pop()
            if ch != want:
                changes.append(f"{ch} -> {want}")
                ch = want
        out.append(ch)
    if in_str:
        return None, "unclosed string"
    if stack:
        changes.append(f"added {''.join(reversed(stack))}")
        out.extend(reversed(stack))
    return "".join(out), changes


def rescore(path, bench, split, show):
    d = json.load(open(path, encoding="utf-8"))
    raw_recs, rep_recs, repaired = [], [], []
    for r in d["records"]:
        q = bench[r["id"]]
        base = score(r["output"], q, bench, split)
        raw_recs.append(base)
        if base["valid"]:
            rep_recs.append(base)
            continue
        text, changes = repair(r["output"])
        if text is None or not changes:
            rep_recs.append(base)
            continue
        new = score(text, q, bench, split)
        rep_recs.append(new)
        repaired.append((r["id"], changes, new["valid"]))
    if show and repaired:
        for qid, changes, ok in repaired:
            print(f"  {Path(path).stem} {qid}: {', '.join(changes)} -> {'valid' if ok else 'still invalid'}")
    return raw_recs, rep_recs


def frac(xs):
    return f"{sum(xs)}/{len(xs)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--results", default=str(HERE / "eval_results"))
    ap.add_argument("--show-repairs", action="store_true")
    args = ap.parse_args()

    bench = load_bench(args.bench)
    split = json.load(open(args.split))
    files = sorted(p for p in Path(args.results).glob("*.json"))
    if not files:
        sys.exit(f"no results in {args.results}/")

    cols = ["valid", "valid+rep", "stmt", "stmt+rep", "stmt in valid", "rule in valid", "questions+rep"]
    lines = ["| model | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for f in files:
        raw, rep = rescore(f, bench, split, args.show_repairs)
        st_raw = [s for r in raw for s in r["statements"].values()]
        st_rep = [s for r in rep for s in r["statements"].values()]
        st_valid = [s for r in rep if r["valid"] for s in r["statements"].values()]
        row = {
            "valid": frac([r["valid"] for r in raw]),
            "valid+rep": frac([r["valid"] for r in rep]),
            "stmt": frac([s["ok"] for s in st_raw]),
            "stmt+rep": frac([s["ok"] for s in st_rep]),
            "stmt in valid": frac([s["ok"] for s in st_valid]) if st_valid else "0/0",
            "rule in valid": frac([not s["gold"] for s in st_valid]) if st_valid else "0/0",
            "questions+rep": frac([r["question_ok"] for r in rep]),
        }
        lines.append(f"| {f.stem} | {len(raw)} | " + " | ".join(pct(row[c]) for c in cols) + " |")
    table = "\n".join(lines)
    note = ("Rescored from saved outputs. +rep: after the bracket repair (only outputs that ended on their own; "
            "a wrong closing bracket is swapped, missing ones are added; nothing else changes). "
            "stmt in valid: statements right among valid translations only. "
            "rule in valid: 'call every statement false' on the same statements.\n\n")
    (Path(args.results) / "summary_repaired.md").write_text(note + table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwritten to {args.results}/summary_repaired.md")


if __name__ == "__main__":
    main()
