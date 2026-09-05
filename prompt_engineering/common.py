"""Shared plumbing for the eight per-technique run scripts.

Each `run_NN_*.py` script is deliberately thin: it names a strategy, optionally
adds a couple of technique-specific flags, and calls `execute()`. Everything
common -- argument parsing, model setup, the run loop, scoring, writing results
-- lives here, so the eight scripts stay readable and cannot drift apart.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dataset import example_pool, get_split
from llm import DEFAULT_NUM_PREDICT, MockLLM, OllamaLLM
from scoring import category_of, summarize
from strategies import build_strategy

# Qwen2.5-3B-Instruct. In Ollama the `qwen2.5:3b` tag is the instruct variant.
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://localhost:11434"
RESULTS_DIR = "results"

MOCK_RESPONSES = [
    "Considering each option in turn, option 3 best fits.\nAnswer: 3",
    "The competing views differ on scope.\nAnswer: 1",
    "Best: 2",
    "A partial analysis of the assumptions involved.",
    "After merging the analyses.\nAnswer: 2",
]


def build_parser(description: str, extra=None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="dev = 2021-2023 (60 q); test = 2024-2025 (37 q), hold this out")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeats per question; use >=3 for the robustness metrics")
    ap.add_argument("--limit", type=int, default=0, help="cap questions (0 = all)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per model call")
    ap.add_argument("--num-predict", type=int, default=None,
                    help=f"max output tokens per call (default {DEFAULT_NUM_PREDICT}). "
                         "The dominant cost on CPU -- lower it to go faster.")
    ap.add_argument("--yes", action="store_true",
                    help="skip the runtime estimate confirmation prompt")
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="use a mock model; validates plumbing, produces meaningless scores")
    ap.add_argument("--quiet", action="store_true", help="suppress per-question lines")
    if extra:
        extra(ap)
    return ap


def _make_llm(args):
    if args.dry_run:
        print("DRY RUN: mock model. Scores below are artifacts of canned responses,")
        print("         not evidence about this technique.\n")
        return MockLLM(MOCK_RESPONSES)
    npred = args.num_predict or DEFAULT_NUM_PREDICT
    llm = OllamaLLM(args.model, args.host, timeout=args.timeout, num_predict=npred)
    v = llm.check()
    if v is None:
        print(f"ERROR: cannot reach Ollama at {args.host}.")
        print("Start it with `ollama serve`, then retry.")
        sys.exit(1)
    print(f"Connected to Ollama {v} at {args.host}  |  model={args.model}")
    print(f"max output tokens per call: {npred}\n")
    return llm


def execute(strategy_name: str, args) -> dict:
    """Run one strategy over the chosen split; write results/<name>__<split>.json."""
    questions = get_split(args.split)
    if args.limit:
        questions = questions[:args.limit]

    llm = _make_llm(args)
    kw = {k: v for k, v in vars(args).items()
          if k in ("temperature", "k_examples", "k_samples", "k_branches",
                    "k_thoughts", "sc_temperature")}
    strat = build_strategy(strategy_name, examples=example_pool(), **kw)

    print(f"technique : {strategy_name}")
    print(f"split     : {args.split} ({len(questions)} questions) x {args.runs} run(s)")
    print(f"expected  : ~{strat.calls_per_question} model call(s) per question")

    # Warm the model and measure output speed, so a long run can be aborted before
    # it starts rather than after an hour. This also loads the model into memory,
    # so question 1 isn't slowed by the load and mistaken for a slow prompt.
    if not args.dry_run:
        print("\nwarming up and measuring speed...", flush=True)
        try:
            tok_s, _ = llm.measure_speed()
        except Exception as e:
            print(f"  speed probe failed ({type(e).__name__}); continuing anyway.")
            tok_s = 0.0
        if tok_s > 0:
            npred = llm.num_predict
            per_call = npred / tok_s
            per_q = per_call * strat.calls_per_question
            total_min = per_q * len(questions) * args.runs / 60
            print(f"  output speed      : {tok_s:.1f} tokens/sec")
            print(f"  worst case / call : {per_call:.0f}s  (at the full {npred}-token budget)")
            print(f"  worst case / q    : {per_q:.0f}s")
            print(f"  worst case total  : {total_min:.0f} min "
                  f"({len(questions)} q x {args.runs} run(s))")
            print("  (worst case assumes every call runs to the token limit; typical")
            print("   responses stop earlier, so real time is usually well under this.)")
            if per_call > args.timeout:
                print(f"\n  !! A full-budget call needs ~{per_call:.0f}s but --timeout is "
                      f"{args.timeout}s.")
                print(f"     Raise --timeout, or lower --num-predict to about "
                      f"{int(tok_s * args.timeout * 0.8)}.")
            if total_min > 45 and not args.yes:
                try:
                    if input("\n  Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                        print("  Aborted.")
                        sys.exit(0)
                except EOFError:
                    pass
    print()

    records = []
    total = len(questions) * args.runs
    done = 0
    t_start = time.time()
    interrupted = False

    for run_i in range(args.runs):
        if interrupted:
            break
        for q in questions:
            seed = args.seed + run_i * 1000
            t0 = time.time()
            try:
                out = strat.run(q, llm, temperature=args.temperature, seed=seed)
                predicted, calls = out.predicted, out.calls.as_dict()
                error, trace = None, out.trace
            except KeyboardInterrupt:
                print("\nInterrupted -- writing partial results.")
                interrupted = True
                break
            except Exception as e:
                predicted, calls, error, trace = None, {}, f"{type(e).__name__}: {e}", []
            done += 1
            rec = {
                "strategy": strategy_name, "run": run_i,
                "question_id": q["id"], "year": q["year"], "category": category_of(q),
                "gold": q["answer"], "predicted": predicted,
                "correct": predicted == q["answer"], "error": error,
                "calls": calls, "wall_sec": round(time.time() - t0, 2), "trace": trace,
            }
            records.append(rec)
            if not args.quiet:
                mark = "OK " if rec["correct"] else ("ERR" if error else "-  ")
                print(f"  [{done}/{total}] {q['id']} run{run_i}  {mark} "
                      f"pred={predicted}  gold={q['answer']}  ({rec['wall_sec']}s)",
                      flush=True)

    if not records:
        print("No records produced.")
        return {}

    summ = summarize(records, questions)
    calls_total = sum(r["calls"].get("n_calls", 0) for r in records)
    tokens_total = sum(r["calls"].get("total_tokens", 0) for r in records)
    summ["compute"] = {
        "total_calls": calls_total,
        "total_tokens": tokens_total,
        "calls_per_question": round(calls_total / len(records), 2),
        "tokens_per_question": round(tokens_total / len(records), 1),
        "wall_sec_total": round(time.time() - t_start, 1),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{strategy_name}__{args.split}.json"
    payload = {"strategy": strategy_name, "split": args.split, "model": llm.model,
               "runs": args.runs, "partial": interrupted,
               "summary": summ, "records": records}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    trunc = sum(1 for r in records
                for t in r.get("trace", []) if t.get("stage") == "extract_answer")
    summ["answer_extraction_fallbacks"] = trunc

    ad = summ["answer_distribution"]
    print(f"\n{'=' * 58}")
    print(f"  {strategy_name}   ({summ['n_records']} responses)")
    print(f"{'=' * 58}")
    print(f"  accuracy            : {summ['accuracy']:.3f}")
    print(f"  weighted accuracy   : {summ['weighted_accuracy']:.3f}")
    print(f"  unparseable         : {ad['unparsed']}")
    if trunc:
        print(f"  answer-extraction   : {trunc} response(s) needed a follow-up call")
        print(f"                        (raise --num-predict if this is most of them)")
    print(f"  most common choice  : {ad['most_common_share']:.1f}% of responses"
          + ("   <-- collapsed onto one choice" if ad["most_common_share"] > 50 else ""))
    print(f"  calls / question    : {summ['compute']['calls_per_question']}")
    print(f"  tokens / question   : {summ['compute']['tokens_per_question']:.0f}")
    print(f"  wall clock          : {summ['compute']['wall_sec_total']}s")
    if args.runs > 1:
        rb = summ["robustness"]
        print(f"  avg SD across runs  : {rb['avg_sd']:.3f}")
        print(f"  wrong in every run  : {rb['consistent_errors']}")
        print(f"  right in every run  : {rb['perfect_performance']}")
    print(f"\n  -> {path}")
    print(f"\n  Inspect this run : python score.py --detail {strategy_name}")
    print(f"  Compare all      : python score.py\n")
    return payload
