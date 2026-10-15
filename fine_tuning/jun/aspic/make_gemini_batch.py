#!/usr/bin/env python3
"""Build one Markdown file to attach to Gemini for drafting ASPIC+ translations.

The file has three parts:
  1. the format spec (ASPIC_FORMAT.md)
  2. worked examples: question, expert rationale, finished translation
  3. the questions to translate, each with its expert rationale

A question is never translated while its own finished translation is shown as
an example (Gemini would copy it). Final test questions are refused.
Validation questions are refused unless --allow-valid is given.

Usage (from fine_tuning/jun/aspic/), with B = the cleaned benchmark file:

  # Round A: Gemini redoes the three example questions, shown two other examples
  python make_gemini_batch.py --bench B --split ../data/split.json \
      --examples-dir translations_claude --examples 2022_29 2023_26 \
      --ids 2022_17 2021_37 2024_15 --out gemini_input_A.md

  # Round B and later: examples come from Gemini's own finished translations
  python make_gemini_batch.py --bench B --split ../data/split.json \
      --ids 2022_29 2023_26 2021_28 2021_34 2023_22 --out gemini_input_B.md
  python make_gemini_batch.py --bench B --split ../data/split.json --next 5 --out gemini_input_C.md
"""
import argparse
import json
import sys
from pathlib import Path

from check_aspic import FINAL_TEST_IDS, load_bench

HERE = Path(__file__).resolve().parent
EXAMPLES = ["2022_17", "2021_37", "2024_15"]


def rationale_text(q):
    r = q.get("original_rationale")
    if isinstance(r, dict):
        return "\n".join(f"- {k}: {v}" for k, v in r.items())
    return str(r or "")


def question_block(q):
    return (
        f"#### Question {q['id']}\n\n"
        f"Question text:\n\n{q['original_question']}\n\n"
        f"Expert rationale:\n\n{rationale_text(q)}\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", required=True, help="fine_tuning/jun/data/split.json")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--next", type=int, default=0, help="add the next N train questions without a translation")
    ap.add_argument("--translations", default=str(HERE / "translations"),
                    help="folder of finished translations (the training data)")
    ap.add_argument("--examples", nargs="+", default=EXAMPLES, help="question IDs to show as worked examples")
    ap.add_argument("--examples-dir", help="folder to take the examples from (default: --translations)")
    ap.add_argument("--allow-valid", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench = load_bench(args.bench)
    split = json.load(open(args.split))
    train, valid = set(split["train"]), set(split["valid"])
    done = {p.stem for p in Path(args.translations).glob("*.json")}
    ex_dir = Path(args.examples_dir or args.translations)

    ids = list(args.ids)
    if args.next:
        ids += [i for i in sorted(train)
                if i not in done and i not in ids and i not in args.examples][: args.next]
    if not ids:
        sys.exit("no questions selected: use --ids or --next")

    for i in ids:
        if i in FINAL_TEST_IDS:
            sys.exit(f"{i} is a final test question. It must never be translated or trained on.")
        if i in args.examples:
            sys.exit(f"{i} is also a worked example in this batch, so it would be copied. "
                     f"Pick other --examples for it.")
        if i not in bench:
            sys.exit(f"{i} is not in {args.bench}")
        if i in valid and not args.allow_valid:
            sys.exit(f"{i} is a validation question. Use --allow-valid if you mean it.")
        if i not in train and i not in valid:
            sys.exit(f"{i} is not in the train or validation split (excluded question)")
        if i in done:
            print(f"note: {i} already has a translation in {args.translations}; it will be replaced if you save the new one")

    spec = (HERE / "ASPIC_FORMAT.md").read_text(encoding="utf-8").split("\n", 1)[1]
    spec = "\n".join("#" + line if line.startswith("##") else line for line in spec.splitlines())
    parts = ["# ASPIC+ translation task\n",
             "## Part 1. Format\n",
             spec,
             "\n## Part 2. Worked examples\n"]
    for ex in args.examples:
        f = ex_dir / f"{ex}.json"
        if not f.exists():
            sys.exit(f"example {ex} not found in {ex_dir}/")
        t = json.load(open(f, encoding="utf-8"))
        parts.append(question_block(bench[ex]))
        parts.append("Translation:\n\n```json\n" + json.dumps(t, ensure_ascii=False, indent=1) + "\n```\n")
    parts.append("\n## Part 3. Questions to translate\n")
    for i in ids:
        parts.append(question_block(bench[i]))

    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}: {len(ids)} questions ({', '.join(ids)}); "
          f"examples {', '.join(args.examples)} from {ex_dir}/")


if __name__ == "__main__":
    main()
