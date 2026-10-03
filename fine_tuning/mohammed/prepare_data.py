#!/usr/bin/env python3
"""Prepare leakage-safe MLX chat datasets for LEET-Arg fine-tuning.

Creates two training conditions on exactly the same question split:
  1) direct    -> assistant target is only `Answer: N`
  2) rationale -> expert rationale followed by `Answer: N`

The final test IDs are removed before the train/validation split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SYSTEM_PROMPT = """You are solving LEET-Arg reasoning questions.
Use only the information in the question. Decide which answer choice is correct.
Do not invent facts that are not in the passage.
End your response with exactly one final line in this form:
Answer: <number from 1 to 5>"""


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_id_set(test_rows):
    return {str(x["id"]) for x in test_rows}


def stratified_split(rows, valid_frac: float, seed: int):
    """Split by question, approximately stratified by category.

    Singleton categories stay in train. Every category with >=2 examples gets
    at least one validation item. This keeps all examples from one question in
    exactly one partition because each question is one row here.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category", "UNKNOWN")].append(row)

    train, valid = [], []
    for category in sorted(groups):
        g = groups[category][:]
        rng.shuffle(g)
        if len(g) == 1:
            n_valid = 0
        else:
            n_valid = max(1, round(len(g) * valid_frac))
            n_valid = min(n_valid, len(g) - 1)
        valid.extend(g[:n_valid])
        train.extend(g[n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def make_record(q, mode: str):
    gold = int(q["answer"])
    if mode == "direct":
        assistant = f"Answer: {gold}"
    elif mode == "rationale":
        rationale = (q.get("original_rationale") or "").strip()
        if not rationale:
            raise ValueError(f"{q['id']}: missing original_rationale")
        assistant = rationale + f"\n\nAnswer: {gold}"
    else:
        raise ValueError(mode)

    return {
        "id": str(q["id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["original_question"].strip()},
            {"role": "assistant", "content": assistant},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="LEET_Arg_Questions_cleaned.json")
    ap.add_argument("--test-set", required=True, help="JSON containing held-out test IDs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--valid-frac", type=float, default=0.20)
    ap.add_argument(
        "--exclude-ids",
        default="",
        help="optional comma-separated IDs to exclude for documented data-quality reasons",
    )
    args = ap.parse_args()

    source = read_json(Path(args.source))
    test_rows = read_json(Path(args.test_set))
    test_ids = test_id_set(test_rows)
    extra_exclude = {x.strip() for x in args.exclude_ids.split(",") if x.strip()}

    ids = [str(q["id"]) for q in source]
    duplicates = sorted(x for x, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise SystemExit(f"Duplicate question IDs in source: {duplicates}")

    missing_test = sorted(test_ids - set(ids))
    if missing_test:
        raise SystemExit(f"Held-out IDs missing from source: {missing_test}")

    dev = [q for q in source if str(q["id"]) not in test_ids | extra_exclude]
    heldout = [q for q in source if str(q["id"]) in test_ids]

    bad = [q["id"] for q in dev if not q.get("original_question") or not q.get("answer")]
    if bad:
        raise SystemExit(f"Missing question/answer fields: {bad}")

    train, valid = stratified_split(dev, args.valid_frac, args.seed)

    train_ids = {str(q["id"]) for q in train}
    valid_ids = {str(q["id"]) for q in valid}
    assert not (train_ids & valid_ids)
    assert not (train_ids & test_ids)
    assert not (valid_ids & test_ids)

    out = Path(args.out)
    for mode in ("direct", "rationale"):
        write_jsonl(out / mode / "train.jsonl", [make_record(q, mode) for q in train])
        write_jsonl(out / mode / "valid.jsonl", [make_record(q, mode) for q in valid])

    split = {
        "seed": args.seed,
        "valid_frac": args.valid_frac,
        "source_count": len(source),
        "heldout_test_count": len(heldout),
        "extra_excluded_ids": sorted(extra_exclude),
        "train_count": len(train),
        "valid_count": len(valid),
        "train_ids": sorted(train_ids),
        "valid_ids": sorted(valid_ids),
        "test_ids": sorted(test_ids),
        "train_categories": dict(sorted(Counter(q.get("category", "UNKNOWN") for q in train).items())),
        "valid_categories": dict(sorted(Counter(q.get("category", "UNKNOWN") for q in valid).items())),
    }
    write_json(out / "split.json", split)

    # Small evaluation manifests. They contain IDs only, so the gold data stays
    # in the source benchmark and no training file contains held-out examples.
    write_json(out / "valid_ids.json", sorted(valid_ids))
    write_json(out / "test_ids.json", sorted(test_ids))

    print(json.dumps({
        "source": len(source),
        "heldout_test": len(heldout),
        "extra_excluded": len(extra_exclude),
        "train": len(train),
        "valid": len(valid),
        "output": str(out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
