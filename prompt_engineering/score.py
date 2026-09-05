#!/usr/bin/env python3
"""Score and compare the prompting techniques.

Two modes:

    python score.py                          # compare every technique run so far
    python score.py --detail zero_shot_cot   # dig into one technique

The comparison view is designed to be run repeatedly as you work through the
techniques one at a time -- it simply reports on whatever result files exist.

    python score.py --detail self_refine --show-trace 2   # inspect model output
"""

import argparse
import json
from pathlib import Path

from dataset import get_split
from scoring import mcnemar


def load(outdir: Path, split: str):
    out = {}
    for p in sorted(outdir.glob(f"*__{split}.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out[d["strategy"]] = d
    return out


# Presentation order matches the run script numbering.
ORDER = ["zero_shot", "zero_shot_cot", "few_shot_cot", "self_consistency",
         "plan_and_solve", "self_refine", "tree_of_thoughts", "graph_of_thoughts"]
REDUCTIONS = {"tree_of_thoughts", "graph_of_thoughts"}


def _banner(data, questions, split):
    covered = {r["question_id"] for d in data.values() for r in d["records"]}
    n_runs = max((d.get("runs", 1) for d in data.values()), default=1)
    models = {d.get("model", "?") for d in data.values()}
    partial = "" if len(covered) == len(questions) else f" of {len(questions)}"
    print(f"\n=== {split} split | {len(covered)}{partial} questions | {n_runs} run(s) "
          f"| model={'/'.join(sorted(models))} ===")
    if "mock" in models:
        print("!! DRY-RUN DATA PRESENT: mock responses, not a real model.")
        print("   Those accuracy numbers say nothing about prompting techniques.")
    if len(models) > 1:
        print("!! Results came from DIFFERENT MODELS -- not comparable.")
    if n_runs < 2:
        print("!! runs=1, so avgSD is 0 by construction and the 0/N and N/N columns")
        print("   are just wrong/right counts. Use --runs 3 or more for robustness.")
    return n_runs


def compare(data, questions, split, baseline):
    n_runs = _banner(data, questions, split)

    rows = []
    for name in sorted(data, key=lambda n: ORDER.index(n) if n in ORDER else 99):
        s = data[name]["summary"]
        rb, cp = s["robustness"], s.get("compute", {})
        rows.append((name, s["accuracy"], s["weighted_accuracy"], rb["avg_sd"],
                     rb["consistent_errors"], rb["perfect_performance"],
                     s["answer_distribution"]["unparsed"],
                     cp.get("tokens_per_question", 0), cp.get("calls_per_question", 0)))

    hdr = (f"{'technique':<20}{'acc':>7}{'w.acc':>8}{'avgSD':>7}{'0/N':>6}"
           f"{'N/N':>6}{'unpar':>7}{'tok/q':>9}{'calls/q':>9}")
    print(hdr); print("-" * len(hdr))
    base_acc = data[baseline]["summary"]["accuracy"] if baseline in data else None
    for r in rows:
        mark = " *" if r[0] in REDUCTIONS else ""
        print(f"{r[0] + mark:<20}{r[1]:>7.3f}{r[2]:>8.3f}{r[3]:>7.3f}"
              f"{r[4]:>6}{r[5]:>6}{r[6]:>7}{r[7]:>9.0f}{r[8]:>9.2f}")
    if any(r[0] in REDUCTIONS for r in rows):
        print("\n* single-level reduction, not the published algorithm (see the run script)")

    if base_acc is not None:
        print(f"\n=== Change vs {baseline} ===")
        for r in rows:
            if r[0] == baseline:
                continue
            delta = r[1] - base_acc
            cost = (r[7] / rows[[x[0] for x in rows].index(baseline)][7]) \
                if rows[[x[0] for x in rows].index(baseline)][7] else 0
            print(f"  {r[0]:<20} {delta:+.3f} accuracy   {cost:>5.1f}x tokens")

        print(f"\n=== McNemar vs {baseline} (paired, exact binomial) ===")
        base = data[baseline]["records"]
        for r in rows:
            if r[0] == baseline:
                continue
            m = mcnemar(data[r[0]]["records"], base)
            direction = ("better" if m["a_only_correct"] > m["b_only_correct"]
                         else "worse" if m["a_only_correct"] < m["b_only_correct"] else "tied")
            star = " *" if m["significant_at_05"] else ""
            print(f"  {r[0]:<20} {direction:<7} (+{m['a_only_correct']}/-{m['b_only_correct']}) "
                  f"p={m['p_value']:.4f}{star}")
        print("  * significant at 0.05. With small discordant counts this test has very")
        print("    low power -- non-significant usually means too little data, not no effect.")

    print("\n=== Answer base rates ===")
    for r in rows:
        ad = data[r[0]]["summary"]["answer_distribution"]
        flag = "  <-- collapsed onto one choice" if ad["most_common_share"] > 50 else ""
        print(f"  {r[0]:<20} most common choice: {ad['most_common_share']:>5.1f}%{flag}")
    print("  gold:", data[rows[0][0]]["summary"]["answer_distribution"]["gold_pct"])

    missing = [n for n in ORDER if n not in data]
    if missing:
        print(f"\nNot yet run: {', '.join(missing)}")
    print()


def detail(d, show_trace):
    s = d["summary"]
    name = d["strategy"]
    print(f"\n=== {name} | {d['split']} split | model={d.get('model','?')} "
          f"| {d.get('runs',1)} run(s) ===")
    if name in REDUCTIONS:
        print("NOTE: single-level reduction, not the published algorithm.")
    print(f"\naccuracy           : {s['accuracy']:.3f}")
    print(f"weighted accuracy  : {s['weighted_accuracy']:.3f}")
    cp = s.get("compute", {})
    print(f"calls / question   : {cp.get('calls_per_question','?')}")
    print(f"tokens / question  : {cp.get('tokens_per_question','?')}")
    print(f"wall clock         : {cp.get('wall_sec_total','?')}s")

    print("\naccuracy by category:")
    for k, v in s["accuracy_by_category"].items():
        print(f"  {k:<40} {v:.3f}")

    ad = s["answer_distribution"]
    print(f"\npredicted choice distribution : {ad['predicted_pct']}")
    print(f"gold choice distribution      : {ad['gold_pct']}")
    if ad["most_common_share"] > 50:
        print("  !! collapsed onto one choice -- accuracy here may not reflect reasoning")
    if ad["unparsed"]:
        print(f"  !! {ad['unparsed']} unparseable response(s), counted as incorrect.")
        print("     If this is high, the model is likely running out of output budget")
        print("     before emitting its final answer -- raise num_predict in llm.py.")

    errs = [r for r in d["records"] if r.get("error")]
    if errs:
        print(f"\n{len(errs)} error(s):")
        for r in errs[:5]:
            print(f"  {r['question_id']}: {r['error']}")

    print("\nper-question results:")
    print(f"  {'id':<10}{'run':>4}{'gold':>6}{'pred':>7}{'ok':>5}{'sec':>8}")
    for r in d["records"]:
        print(f"  {r['question_id']:<10}{r['run']:>4}{r['gold']:>6}"
              f"{str(r['predicted']):>7}{('Y' if r['correct'] else 'n'):>5}"
              f"{r['wall_sec']:>8.1f}")

    wrong = [r for r in d["records"] if not r["correct"] and not r.get("error")]
    if wrong:
        print(f"\n{len(wrong)} incorrect. First few, with gold vs predicted:")
        for r in wrong[:10]:
            print(f"  {r['question_id']} run{r['run']}: gold={r['gold']} pred={r['predicted']}")

    if show_trace:
        print(f"\n=== model output for the first {show_trace} question(s) ===")
        for r in d["records"][:show_trace]:
            print(f"\n--- {r['question_id']} (gold={r['gold']}, pred={r['predicted']}) ---")
            for t in r.get("trace", []):
                print(f"\n  [stage: {t['stage']}]")
                body = (t.get("response") or "").strip()
                for line in body.splitlines()[:40]:
                    print(f"    {line}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--baseline", default="zero_shot")
    ap.add_argument("--detail", metavar="TECHNIQUE",
                    help="show a full breakdown for one technique")
    ap.add_argument("--show-trace", type=int, default=0, metavar="N",
                    help="with --detail, print raw model output for N questions")
    args = ap.parse_args()

    data = load(Path(args.outdir), args.split)
    if not data:
        print(f"No results in '{args.outdir}' for split '{args.split}'.")
        print("Run a technique first, e.g.:  python run_01_zero_shot.py --limit 10")
        return

    if args.detail:
        if args.detail not in data:
            print(f"No results for '{args.detail}'. Available: {', '.join(sorted(data))}")
            return
        detail(data[args.detail], args.show_trace)
    else:
        compare(data, get_split(args.split), args.split, args.baseline)


if __name__ == "__main__":
    main()
