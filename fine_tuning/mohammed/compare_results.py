#!/usr/bin/env python3
"""Compare paired LEET-Arg evaluation result files.

Reports accuracy, format compliance, paired accuracy difference, a question-level
bootstrap CI, and exact McNemar p-value. Small test sets produce wide intervals;
report them as uncertainty, not as proof of equivalence.
"""

import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        x = json.load(f)
    return x["summary"], {r["id"]: r for r in x["records"]}


def exact_mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial test under p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(a, b, ids, seed=7, n_boot=10000):
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        da = sum(a[i]["correct"] for i in sample) / len(sample)
        db = sum(b[i]["correct"] for i in sample) / len(sample)
        diffs.append(db - da)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sa, a = load(args.baseline)
    sb, b = load(args.candidate)
    ids = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise SystemExit("Result files do not contain the same question IDs")

    acc_a = sum(a[i]["correct"] for i in ids) / len(ids)
    acc_b = sum(b[i]["correct"] for i in ids) / len(ids)
    b_only = sum((not a[i]["correct"]) and b[i]["correct"] for i in ids)
    a_only = sum(a[i]["correct"] and (not b[i]["correct"]) for i in ids)
    lo, hi = bootstrap_diff(a, b, ids, args.seed, args.bootstrap)
    p = exact_mcnemar_p(a_only, b_only)

    print(f"N questions: {len(ids)}")
    print(f"Baseline accuracy:  {acc_a:.3f} ({sum(a[i]['correct'] for i in ids)}/{len(ids)})")
    print(f"Candidate accuracy: {acc_b:.3f} ({sum(b[i]['correct'] for i in ids)}/{len(ids)})")
    print(f"Difference candidate-baseline: {acc_b-acc_a:+.3f}")
    print(f"Bootstrap 95% CI for difference: [{lo:+.3f}, {hi:+.3f}]")
    print(f"Paired discordant cases: baseline-only={a_only}, candidate-only={b_only}")
    print(f"Exact McNemar p-value: {p:.4f}")
    print(f"Baseline strict format rate:  {sa['strict_format_rate']:.3f}")
    print(f"Candidate strict format rate: {sb['strict_format_rate']:.3f}")

    print("\nChanged questions:")
    for i in ids:
        if a[i]["correct"] != b[i]["correct"]:
            print(f"  {i}: baseline={a[i]['pred_answer']} candidate={b[i]['pred_answer']} gold={a[i]['gold_answer']}")


if __name__ == "__main__":
    main()
