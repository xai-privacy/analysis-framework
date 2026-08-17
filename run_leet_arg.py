# File: run_leet_arg.py
"""CLI entry point for the LEET-Arg plain-model harness.

Runs the six-stage pipeline over a sample of LEET-Arg records:

    Record -> build_prompt() -> ModelAdapter.generate() -> parse()
           -> Reasoner.reason() -> score()

This builds the plain-model cell of the 2x2 (local SLM, no solver). The solver
condition plugs in by supplying a different Reasoner; the frontier condition by
supplying a different ModelAdapter. No other stage changes.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

from leet_arg.adapters import DEFAULT_MODEL, build_adapter, load_model_config
from leet_arg.data import SHAPE_FIVE, SHAPE_THREE, load_records
from leet_arg.gold import detect_polarity, gold_choice, verify_gold_derivation
from leet_arg.parse import parse
from leet_arg.prompts import CHOICE_LEVEL, DECOMPOSITIONS, SYSTEM_PROMPT, build_prompt
from leet_arg.reason import PassthroughReasoner
from leet_arg.score import ResultRow, majority_class_baseline, score

### Usage:
###   python3 run_leet_arg.py [--model <hf-model-id>] [--n <records>] [--trials <k>]
###
### --model         : Hugging Face model id. Dense text decoder models only.
### --n             : number of records to sample (-1 for all 97).
### --trials        : repeats per record, each with a distinct seed.
### --decomposition : prompt protocol. Only choice_level is implemented; the
###                   parameter exists because LEET-Arg results are comparable
###                   only within a protocol.
### --adapter       : hf (default), api (stub), or stub (no-GPU smoke test).

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_SEED_BASE = 1000
DEFAULT_SAMPLE_SEED = 7


def select_records(records, n, sample_seed):
    """Sample `n` records, guaranteeing both structural shapes are exercised.

    The 3-statement and 5-statement shapes are prompted differently, so a sample
    that happens to contain only one of them would leave half the prompt builder
    untested. Counts are proportional to the dataset, floored at one per shape.
    """
    if n is None or n < 0 or n >= len(records):
        return list(records)

    three = [record for record in records if record.shape == SHAPE_THREE]
    five = [record for record in records if record.shape == SHAPE_FIVE]

    if n < 2:
        n_five = 0
    else:
        proportional = int(round(n * len(five) / float(len(records))))
        n_five = min(len(five), max(1, proportional))
    n_three = min(len(three), n - n_five)
    # If the 3-statement pool were ever smaller than requested, spend the
    # remainder on 5-statement records rather than returning a short sample.
    n_five = min(len(five), n - n_three)

    rng = random.Random(sample_seed)
    chosen = rng.sample(three, n_three) + rng.sample(five, n_five)
    chosen.sort(key=lambda record: record.id)
    return chosen


def build_generation_config(model_config, temperature, max_new_tokens):
    """Generation kwargs, with sampling forced on.

    The committed model configs use `do_sample: false` and set no temperature.
    Greedy decoding would make all trials of a record identical and the
    consistency metric meaningless, so temperature is set explicitly here and
    must be greater than zero.
    """
    generation = dict(model_config.get("generation", {}))
    generation.pop("do_sample", None)

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero: greedy decoding makes every "
            "trial identical and the consistency metric meaningless"
        )

    generation["do_sample"] = True
    generation["temperature"] = float(temperature)
    generation["max_new_tokens"] = int(max_new_tokens)
    return generation


def run(args):
    records = load_records(args.data)
    baseline = majority_class_baseline(records)

    print("=" * 70)
    print("LEET-Arg plain-model harness")
    print("=" * 70)
    print(f"  dataset            : {len(records)} records")
    print(f"  majority baseline  : {baseline * 100:.1f}%")
    print()

    # Gold derivation verification runs every time: P0 accuracy is meaningless if
    # the labels underneath it are unsound.
    report = verify_gold_derivation(records)
    print(report.render())
    if not report.ok:
        print(
            f"\nWARNING: statement-level (P1) derivation failed for "
            f"{len(report.failures)} record(s), listed above. "
            "Choice-level (P0) scoring is unaffected.",
            file=sys.stderr,
        )
    print()

    if args.verify_gold_only:
        return 0 if report.ok else 1

    sample = select_records(records, args.n, args.sample_seed)
    model_config = load_model_config(args.model)
    generation = build_generation_config(model_config, args.temperature, args.max_new_tokens)
    seeds = [args.seed_base + index for index in range(args.trials)]

    shape_summary = {}
    for record in sample:
        shape_summary[record.shape] = shape_summary.get(record.shape, 0) + 1

    print(f"  model              : {args.model}")
    print(f"  adapter            : {args.adapter}")
    print(f"  decomposition      : {args.decomposition}")
    print(f"  sampled records    : {len(sample)} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(shape_summary.items()))})")
    print(f"  trials per record  : {args.trials}  seeds={seeds}")
    print(f"  temperature        : {generation['temperature']}")
    print(f"  max_new_tokens     : {generation['max_new_tokens']}")
    print()

    try:
        adapter = build_adapter(
            args.adapter, args.model, config=model_config, system_prompt=SYSTEM_PROMPT
        )
    except ImportError as exc:
        print(f"\nMissing runtime dependencies for the '{args.adapter}' adapter: {exc}", file=sys.stderr)
        print("Install torch and transformers, or use --adapter stub to smoke-test "
              "the harness without a model.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUnable to initialise adapter '{args.adapter}' for {args.model}: {exc}", file=sys.stderr)
        print("Authenticate with Hugging Face for gated models, or pass a public model id.",
              file=sys.stderr)
        return 1

    reasoner = PassthroughReasoner()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"{timestamp}_{args.model.replace('/', '_')}.jsonl")

    rows = []
    with open(out_path, "w", encoding="utf-8") as handle:
        for position, record in enumerate(sample, start=1):
            gold = gold_choice(record)
            polarity = detect_polarity(record)
            prompt = build_prompt(record, decomposition=args.decomposition)

            for trial_index, seed in enumerate(seeds):
                raw_output = adapter.generate(prompt, seed=seed, config=generation)

                # parse -> reason are separate stages. PassthroughReasoner is in
                # the call path even though it returns the choice unchanged.
                parsed = parse(raw_output)
                verdict = reasoner.reason(parsed, record)

                row = ResultRow(
                    record_id=record.id,
                    model=args.model,
                    decomposition=args.decomposition,
                    trial_index=trial_index,
                    seed=seed,
                    temperature=generation["temperature"],
                    raw_output=raw_output,
                    parsed_choice=parsed.choice,
                    parse_status=parsed.status,
                    gold_choice=gold,
                    correct=(verdict.choice == gold),
                    shape=record.shape,
                    domain=record.domain,
                    category=record.category,
                    polarity=polarity,
                    reasoner=verdict.reasoner,
                    verdict_choice=verdict.choice,
                    adapter=args.adapter,
                )
                rows.append(row)
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()

                flag = "ok " if row.correct else "   "
                print(
                    f"  [{position:>3}/{len(sample)}] {record.id} "
                    f"trial {trial_index} seed {seed} -> "
                    f"{str(parsed.choice):>4} (gold {gold}) "
                    f"{parsed.status:<16} {flag}"
                )
                sys.stdout.flush()

    metrics = score(rows, majority_baseline=baseline)
    print()
    print(metrics.render())
    print(f"\n  rows written to: {out_path}")

    summary_path = out_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "adapter": args.adapter,
                "decomposition": args.decomposition,
                "n_records": len(sample),
                "trials": args.trials,
                "seeds": seeds,
                "generation": generation,
                "gold_verification": {
                    "total": report.total,
                    "passed": len(report.passed),
                    "failures": report.failures,
                },
                "metrics": metrics.to_dict(),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  summary written to: {summary_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the LEET-Arg plain-model baseline against an HF model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model id (dense text decoder models only). "
             f"Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of records to sample; -1 runs all 97. Default: 10",
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Repeats per record, each with a distinct seed. Default: 5",
    )
    parser.add_argument(
        "--decomposition", default=CHOICE_LEVEL, choices=list(DECOMPOSITIONS),
        help="Prompt protocol. Only choice_level is implemented. "
             "Recorded on every result row because LEET-Arg results are "
             "comparable only within a protocol.",
    )
    parser.add_argument(
        "--adapter", default="hf", choices=["hf", "api", "stub"],
        help="hf: local Hugging Face model (default). api: frontier stub, not "
             "implemented. stub: canned outputs for smoke-testing without a GPU.",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature; must be > 0. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation length cap. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                        help=f"First seed; trial k uses seed_base + k. Default: {DEFAULT_SEED_BASE}")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help=f"Seed for record sampling. Default: {DEFAULT_SAMPLE_SEED}")
    parser.add_argument("--data", default=None, help="Path to the cleaned dataset JSON.")
    parser.add_argument("--out-dir", default="results", help="Directory for JSONL output.")
    parser.add_argument("--verify-gold-only", action="store_true",
                        help="Run gold derivation verification and exit without loading a model.")
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"LEET-Arg run failed: {exc}", file=sys.stderr)
        sys.exit(1)
