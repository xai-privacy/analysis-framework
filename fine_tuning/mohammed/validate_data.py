#!/usr/bin/env python3
"""Validate prepared LEET-Arg MLX chat data and detect leakage."""

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON: {e}")
    return rows


def check_records(path, rows):
    errors = []
    ids = []
    for i, r in enumerate(rows, 1):
        rid = str(r.get("id", ""))
        ids.append(rid)
        msgs = r.get("messages")
        if not rid:
            errors.append(f"{path}:{i}: missing id")
        if not isinstance(msgs, list) or len(msgs) != 3:
            errors.append(f"{path}:{i}: messages must contain system/user/assistant")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"{path}:{i}: wrong roles {roles}")
        if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs):
            errors.append(f"{path}:{i}: blank message content")
        if "Answer:" not in msgs[-1]["content"]:
            errors.append(f"{path}:{i}: assistant target missing Answer:")
    if len(ids) != len(set(ids)):
        errors.append(f"{path}: duplicate IDs")
    return errors, set(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    root = Path(args.data)
    split = load_json(root / "split.json")
    heldout = set(split["test_ids"])

    all_errors = []
    mode_ids = {}
    for mode in ("direct", "rationale"):
        train = load_jsonl(root / mode / "train.jsonl")
        valid = load_jsonl(root / mode / "valid.jsonl")
        e1, train_ids = check_records(root / mode / "train.jsonl", train)
        e2, valid_ids = check_records(root / mode / "valid.jsonl", valid)
        all_errors += e1 + e2
        if train_ids & valid_ids:
            all_errors.append(f"{mode}: train/valid overlap: {sorted(train_ids & valid_ids)}")
        if train_ids & heldout:
            all_errors.append(f"{mode}: TEST LEAKAGE in train: {sorted(train_ids & heldout)}")
        if valid_ids & heldout:
            all_errors.append(f"{mode}: TEST LEAKAGE in valid: {sorted(valid_ids & heldout)}")
        mode_ids[mode] = (train_ids, valid_ids)

    if mode_ids["direct"] != mode_ids["rationale"]:
        all_errors.append("direct and rationale conditions do not use identical splits")

    if all_errors:
        print("VALIDATION FAILED")
        for e in all_errors:
            print(" -", e)
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"train={len(mode_ids['direct'][0])}, valid={len(mode_ids['direct'][1])}, heldout_test={len(heldout)}")
    print("No held-out test ID appears in train or validation.")


if __name__ == "__main__":
    main()
