"""Scoring, following the LEET-Arg paper's methodology (Sections 5.1-5.2).

Metrics implemented:
  * Answer accuracy over N runs x 97 problems.
  * Inverse-frequency weighted accuracy (paper Eq. 1).
  * Robustness table: avg/max SD, unique answers, consistent errors, perfect
    performance -- matching the columns of the paper's Table 3.
  * McNemar paired test between two strategies (the paper compares models with
    ANOVA/GLM; for comparing prompting strategies on the *same* items a paired
    test is the appropriate choice).
  * Answer base-rate check, to catch a strategy that scores well by collapsing
    onto one choice rather than by reasoning.

Two dataset notes, both verified against the released JSON:

1. The JSON carries four `category` values, but the paper reports three.
   Merging 'Argument Evaluation & Analysis' (24) into 'Argument Evaluation &
   Problem Solving' (15) yields 39, and reproduces the paper's exact counts of
   43 / 15 / 39. `merge_categories=True` (default) does this.
2. Ten items have `domain: null`. The paper lists a 'Science and Technology'
   domain and states that Argument Analysis has n=0 there. The null-domain group
   is exactly the group containing no Argument Analysis items, so these are
   treated as Science and Technology.
"""

import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

MERGE_MAP = {"Argument Evaluation & Analysis": "Argument Evaluation & Problem Solving"}
MISSING_DOMAIN = "Science and Technology"


def category_of(q: dict, merge_categories: bool = True) -> str:
    c = q.get("category") or "Unspecified"
    return MERGE_MAP.get(c, c) if merge_categories else c


def domain_of(q: dict) -> str:
    return q.get("domain") or MISSING_DOMAIN


def compute_weights(questions: Sequence[dict], merge_categories: bool = True) -> Dict[str, float]:
    """Inverse-frequency weights per question id (paper Eq. 1).

        w(c,d) = N / n(c,d),  then normalised to mean 1.0

    N is the number of problems in the set being scored. Combinations with no
    problems contribute nothing (you cannot weight an empty cell), so the
    normalisation runs over items actually present.
    """
    N = len(questions)
    if N == 0:
        return {}
    counts = Counter((category_of(q, merge_categories), domain_of(q)) for q in questions)
    raw = {q["id"]: N / counts[(category_of(q, merge_categories), domain_of(q))]
           for q in questions}
    mean_raw = sum(raw.values()) / N
    return {qid: w / mean_raw for qid, w in raw.items()}


def accuracy(records: Sequence[dict]) -> float:
    """Plain accuracy over all (question, run) records."""
    if not records:
        return 0.0
    return sum(1 for r in records if r["correct"]) / len(records)


def weighted_accuracy(records: Sequence[dict], weights: Dict[str, float]) -> float:
    """Accuracy with each record weighted by its category-domain weight."""
    if not records:
        return 0.0
    num = sum(weights.get(r["question_id"], 1.0) * (1.0 if r["correct"] else 0.0)
              for r in records)
    den = sum(weights.get(r["question_id"], 1.0) for r in records)
    return num / den if den else 0.0


def robustness(records: Sequence[dict]) -> Dict:
    """Reproduces the columns of the paper's Table 3.

    SD is the *sample* standard deviation (ddof=1) of the 0/1 correctness vector
    across runs for one problem. This is verified by the paper's own reported
    values: 3-of-5 correct gives 0.548 and 4-of-5 gives 0.447, both of which
    match ddof=1 exactly (ddof=0 would give 0.490 and 0.400).
    """
    by_q = defaultdict(list)
    for r in records:
        by_q[r["question_id"]].append(r)

    sds, uniques, consistent_errors, perfect, high_var = [], [], 0, 0, 0
    for qid, rs in by_q.items():
        correct = [1 if r["correct"] else 0 for r in rs]
        sd = statistics.stdev(correct) if len(correct) > 1 else 0.0
        sds.append(sd)
        if sd > 0.4:
            high_var += 1
        # None (unparseable) is its own distinct answer value.
        uniques.append(len({r["predicted"] for r in rs}))
        if sum(correct) == 0:
            consistent_errors += 1
        if sum(correct) == len(correct):
            perfect += 1

    n_runs = max((len(v) for v in by_q.values()), default=0)
    return {
        "n_problems": len(by_q),
        "n_runs": n_runs,
        "avg_sd": round(sum(sds) / len(sds), 4) if sds else 0.0,
        "max_sd": round(max(sds), 4) if sds else 0.0,
        "avg_unique_answers": round(sum(uniques) / len(uniques), 3) if uniques else 0.0,
        "high_variability_cases": high_var,
        "consistent_errors": consistent_errors,
        "perfect_performance": perfect,
    }


def answer_distribution(records: Sequence[dict], questions: Sequence[dict]) -> Dict:
    """Predicted-vs-gold choice distribution, plus an unparseable count.

    A strategy that answers '2' for everything will score ~26% on this dataset
    without reasoning at all, so this guards against reading such a result as
    genuine competence.
    """
    pred = Counter(r["predicted"] if r["predicted"] is not None else "unparsed"
                   for r in records)
    gold = Counter(q["answer"] for q in questions)
    total = len(records)
    return {
        "predicted_pct": {k: round(100 * v / total, 1) for k, v in sorted(pred.items(), key=str)},
        "gold_pct": {k: round(100 * v / len(questions), 1) for k, v in sorted(gold.items())},
        "unparsed": pred.get("unparsed", 0),
        "most_common_share": round(100 * max(pred.values()) / total, 1) if total else 0.0,
    }


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar with n = b + c, p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(records_a: Sequence[dict], records_b: Sequence[dict]) -> Dict:
    """Paired comparison of two strategies on the same problems.

    Aggregates each strategy to one score per problem (majority-correct across
    runs) so the pairing is per problem, not per run. Uses the exact binomial
    test, which is appropriate for the small discordant counts expected at
    n = 97 (the chi-square approximation is unreliable below ~25).
    """
    def per_q(records):
        by_q = defaultdict(list)
        for r in records:
            by_q[r["question_id"]].append(1 if r["correct"] else 0)
        return {q: (sum(v) / len(v)) > 0.5 for q, v in by_q.items()}

    a, b_ = per_q(records_a), per_q(records_b)
    shared = sorted(set(a) & set(b_))
    b = sum(1 for q in shared if a[q] and not b_[q])   # A right, B wrong
    c = sum(1 for q in shared if b_[q] and not a[q])   # B right, A wrong
    return {
        "n_paired": len(shared),
        "a_only_correct": b,
        "b_only_correct": c,
        "p_value": round(_binom_two_sided(b, c), 4),
        "significant_at_05": _binom_two_sided(b, c) < 0.05,
    }


def summarize(records: Sequence[dict], questions: Sequence[dict],
              merge_categories: bool = True) -> Dict:
    weights = compute_weights(questions, merge_categories)
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "?")].append(r)
    out = {
        "n_records": len(records),
        "accuracy": round(accuracy(records), 4),
        "weighted_accuracy": round(weighted_accuracy(records, weights), 4),
        "robustness": robustness(records),
        "answer_distribution": answer_distribution(records, questions),
        "accuracy_by_category": {k: round(accuracy(v), 3) for k, v in sorted(by_cat.items())},
        "weight_range": [round(min(weights.values()), 3), round(max(weights.values()), 3)]
                        if weights else [0, 0],
    }
    return out
