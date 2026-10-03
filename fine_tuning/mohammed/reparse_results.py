#!/usr/bin/env python3
"""Reparse existing evaluation JSON without regenerating model outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Import parse_answer from evaluate_v2.py in the same directory.
from evaluate_v2 import parse_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    data = json.loads(src.read_text(encoding="utf-8"))
    records = data["records"]

    for r in records:
        pred, strict, method = parse_answer(r.get("output", ""))
        r["pred_answer"] = pred
        r["strict_format"] = strict
        r["parse_method"] = method
        r["correct"] = pred == int(r["gold_answer"])

    n = len(records)
    parsed = [r for r in records if r["pred_answer"] is not None]
    summary = dict(data.get("summary", {}))
    summary["parser_version"] = "v2_reparse"
    summary["n"] = n
    summary["strict_format_rate"] = sum(r["strict_format"] for r in records) / n if n else 0
    summary["parse_rate"] = len(parsed) / n if n else 0
    summary["accuracy"] = sum(r["correct"] for r in records) / n if n else 0

    cats = defaultdict(list)
    for r in records:
        cats[r.get("category", "UNKNOWN")].append(r)
    summary["per_category"] = {
        c: {
            "n": len(rs),
            "accuracy": sum(r["correct"] for r in rs) / len(rs),
            "strict_format_rate": sum(r["strict_format"] for r in rs) / len(rs),
            "parse_rate": sum(r["pred_answer"] is not None for r in rs) / len(rs),
        }
        for c, rs in sorted(cats.items())
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
