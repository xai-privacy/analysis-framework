#!/usr/bin/env python3
"""Audit LEET-Arg source/test files before fine-tuning."""

import argparse
import json
from collections import Counter


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--test-set", required=True)
    args = ap.parse_args()
    rows = load(args.source)
    test = load(args.test_set)
    ids = [str(q["id"]) for q in rows]
    test_ids = {str(q["id"]) for q in test}
    dup = [k for k,v in Counter(ids).items() if v>1]
    missing_fields = []
    bad_answers = []
    empty_rationale = []
    for q in rows:
        missing = [k for k in ("id","original_question","answer") if not q.get(k)]
        if missing:
            missing_fields.append((q.get("id"), missing))
        try:
            a=int(q.get("answer"))
            if a not in range(1,6): bad_answers.append(q.get("id"))
        except Exception:
            bad_answers.append(q.get("id"))
        if not (q.get("original_rationale") or "").strip():
            empty_rationale.append(q.get("id"))

    dev = [q for q in rows if str(q["id"]) not in test_ids]
    print(f"source questions: {len(rows)}")
    print(f"held-out test IDs: {len(test_ids)}")
    print(f"non-test questions: {len(dev)}")
    print(f"duplicate IDs: {dup or 'none'}")
    print(f"missing test IDs from source: {sorted(test_ids-set(ids)) or 'none'}")
    print(f"bad answer fields: {bad_answers or 'none'}")
    print(f"empty rationales: {empty_rationale or 'none'}")
    print("\nNon-test category counts:")
    for k,v in sorted(Counter(q.get('category','UNKNOWN') for q in dev).items()):
        print(f"  {k}: {v}")
    print("\nNon-test answer distribution:")
    for k,v in sorted(Counter(str(q.get('answer')) for q in dev).items()):
        print(f"  {k}: {v}")
    if dup or missing_fields or bad_answers or (test_ids-set(ids)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
