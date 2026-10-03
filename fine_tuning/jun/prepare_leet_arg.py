"""Convert LEET-Arg into MLX fine-tuning data (issue #13).

Writes:
  data/train.jsonl   80% of the usable non-test questions
  data/valid.jsonl   20% of the usable non-test questions
  data/split.json    which question IDs went where, and why some were left out

The 16 final test questions in
testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json are removed
before anything else and are never written to any file here.

Each line uses MLX's chat format ({"messages": [...]}), so mlx_lm applies the
Llama 3.2 chat template itself, both in training and in mlx_lm.generate.

Usage (works from any folder; output goes next to this script):
  python3 prepare_leet_arg.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from leet_arg_format import (DATA_DIR, SYSTEM_PROMPT, build_target,
                             build_user_prompt, has_statement_rationales,
                             load_dataset, load_final_test_ids)

# The choice list says ② = (b), but the expert rationale says only (c) is
# correct and that the answer is ②. Gold labels cannot be trusted.
INCONSISTENT = {"2025_16": "choice list contradicts the expert rationale (rationale: only (c) is correct; choice ② is (b))"}


def to_example(q):
    return {
        "id": q["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(q)},
            {"role": "assistant", "content": build_target(q)},
        ],
    }


def stratified_split(questions, valid_frac, seed):
    """80/20 split by question, stratified by category."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)
    train, valid = [], []
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda q: q["id"])
        rng.shuffle(group)
        n_valid = round(len(group) * valid_frac)
        valid += group[:n_valid]
        train += group[n_valid:]
    return sorted(train, key=lambda q: q["id"]), sorted(valid, key=lambda q: q["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    questions = load_dataset()
    test_ids = load_final_test_ids()

    excluded = {}
    usable = []
    for q in questions:
        if q["id"] in test_ids:
            excluded[q["id"]] = "final test question"
        elif q["id"] in INCONSISTENT:
            excluded[q["id"]] = INCONSISTENT[q["id"]]
        elif not has_statement_rationales(q):
            excluded[q["id"]] = "no per-statement expert rationale (diagram-format question)"
        else:
            usable.append(q)

    train, valid = stratified_split(usable, args.valid_frac, args.seed)

    # Hard check: no final test question may appear in any output.
    leaked = test_ids & {q["id"] for q in train + valid}
    assert not leaked, f"final test questions leaked into the split: {sorted(leaked)}"

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, split in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for q in split:
                f.write(json.dumps(to_example(q), ensure_ascii=False) + "\n")

    manifest = {
        "source": "benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json",
        "seed": args.seed,
        "valid_frac": args.valid_frac,
        "train": [q["id"] for q in train],
        "valid": [q["id"] for q in valid],
        "excluded": excluded,
    }
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_test = sum(1 for r in excluded.values() if r == "final test question")
    print(f"{len(questions)} questions: {n_test} final test removed, "
          f"{len(excluded) - n_test} other exclusions, {len(usable)} usable")
    print(f"train: {len(train)}  valid: {len(valid)}  -> {out}/train.jsonl, {out}/valid.jsonl")


if __name__ == "__main__":
    main()
