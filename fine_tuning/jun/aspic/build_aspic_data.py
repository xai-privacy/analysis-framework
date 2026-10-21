#!/usr/bin/env python3
"""Build the MLX training files from the matched teacher translations.

Each line of train.jsonl / valid.jsonl is one question in MLX chat format:
  system     the frozen instruction from ASPIC_FORMAT.md
  user       passage, question and statements, without the choice list
  assistant  the teacher's translation, with note/summary/source fields and
             the id removed (compact JSON on one line)

This is the same prompt the base-model test used, so training, the base model
and the fine-tuned model all see the same input.

Safety checks: final test questions are refused, train and validation stay in
their split, and every translation is re-checked against the gold answer.

It also writes:
  <out>/stats.json            token lengths, the --max-seq-length to train with,
                              and the --max-tokens eval_aspic.py should use
  statement_types.json        the type of every validation statement (structure
                              or weakens/strengthens) for eval_aspic.py. Written
                              only if it does not exist yet, so your hand edits stay

Usage (from fine_tuning/jun/aspic/, venv active):
  python build_aspic_data.py                          # translations/ -> data_aspic/
  python build_aspic_data.py --smoke --translations translations_claude --out data_smoke
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from check_aspic import FINAL_TEST_IDS, check_translation, load_bench
from test_base_model import frozen_instruction, question_text, strip

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = REPO / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
SPLIT = HERE.parent / "data" / "split.json"
MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TYPES_FILE = HERE / "statement_types.json"

# Statements whose wording asks whether something weakens or strengthens a
# claim. ASPIC+ has no support relation, and for these the translation itself
# mostly decides the answer, so they are scored separately.
WEAK_STRONG = re.compile(r"weaken|strengthen|support|undermin", re.I)
CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}


def target(t):
    out = strip(t)
    out.pop("id", None)
    return out


def row_for(q, t):
    return {"id": q["id"], "messages": [
        {"role": "system", "content": frozen_instruction()},
        {"role": "user", "content": question_text(q)},
        {"role": "assistant", "content": json.dumps(target(t), ensure_ascii=False)},
    ]}


def statement_key(text):
    m = re.match(r"\s*(?:\((\w)\)|([①②③④⑤]))", text)
    if not m:
        return None
    return m.group(1) if m.group(1) else CIRCLED[m.group(2)]


def statement_types(bench, ids):
    out = {}
    for qid in sorted(ids):
        out[qid] = {}
        for text in bench[qid]["statements"].values():
            k = statement_key(text)
            if k is None:
                continue
            out[qid][k] = {"type": "weakens/strengthens" if WEAK_STRONG.search(text) else "structure",
                           "text": text.strip()}
    return out


def token_stats(rows):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # no transformers or no network: rough estimate
        print(f"note: tokenizer not available ({type(e).__name__}); estimating 4 characters per token")
        tok = None
    out = []
    for r in rows:
        m = r["messages"]
        if tok:
            def count(msgs, gen):
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
                return len(tok(text, add_special_tokens=False)["input_ids"])
            total = count(m, False)
            prompt = count(m[:-1], True)
        else:
            total = sum(len(x["content"]) for x in m) // 4 + 40
            prompt = sum(len(x["content"]) for x in m[:-1]) // 4 + 30
        out.append({"id": r["id"], "prompt": prompt, "target": total - prompt, "total": total})
    return out, tok is not None


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--translations", nargs="+", default=[str(HERE / "translations")],
                    help="folders of matched translations; a later folder wins for the same question")
    ap.add_argument("--out", default=str(HERE / "data_aspic"))
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline test: every translation goes to train, and the first two are "
                         "copied to valid so MLX can run. Not for real results")
    args = ap.parse_args()

    bench = load_bench(args.bench)
    split = json.load(open(args.split))
    train_ids, valid_ids = set(split["train"]), set(split["valid"])
    if (train_ids | valid_ids) & FINAL_TEST_IDS:
        sys.exit(f"the split contains final test questions: {sorted((train_ids | valid_ids) & FINAL_TEST_IDS)}")

    found = {}
    for d in args.translations:
        for f in sorted(Path(d).glob("*.json")):
            found[f.stem] = json.load(open(f, encoding="utf-8"))
    if not found:
        sys.exit(f"no translations in {', '.join(args.translations)}")

    train, valid, problems = [], [], []
    for qid, t in sorted(found.items()):
        if qid in FINAL_TEST_IDS:
            sys.exit(f"{qid} is a final test question. Remove it from the translations and stop.")
        if t.get("id") != qid:
            problems.append(f"{qid}: file name and id ({t.get('id')}) differ")
            continue
        rec = check_translation(t, bench, split, FINAL_TEST_IDS, use_pyarg=False)
        if not rec.get("match"):
            problems.append(f"{qid}: does not reproduce the gold answer ({'; '.join(rec.get('errors', [])) or rec.get('pred')})")
            continue
        if qid in train_ids:
            train.append(row_for(bench[qid], t))
        elif qid in valid_ids:
            if args.smoke:
                problems.append(f"{qid}: validation question, left out of the smoke test")
            else:
                valid.append(row_for(bench[qid], t))
        else:
            problems.append(f"{qid}: not in the train or validation split")
    for p in problems:
        print(f"skipped {p}")
    if args.smoke:
        valid = train[:2]
    if not train or not valid:
        sys.exit(f"not enough data: {len(train)} train, {len(valid)} validation")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)

    lens, exact = token_stats(train + valid)
    mx_total = max(l["total"] for l in lens)
    mx_target = max(l["target"] for l in lens)
    max_seq = int(math.ceil((mx_total + 64) / 256.0) * 256)
    max_tokens = int(math.ceil(mx_target * 1.5 / 100.0) * 100)
    stats = {
        "train": len(train), "valid": len(valid), "smoke": args.smoke,
        "train_ids": [r["id"] for r in train], "valid_ids": [r["id"] for r in valid],
        "tokens_exact": exact,
        "longest_total": mx_total, "longest_target": mx_target,
        "mean_target": round(sum(l["target"] for l in lens) / len(lens)),
        "max_seq_length": max_seq, "eval_max_tokens": max_tokens,
        "per_question": lens,
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=1)

    missing_train = sorted(train_ids - {r["id"] for r in train})
    missing_valid = sorted(valid_ids - {r["id"] for r in valid}) if not args.smoke else []
    print(f"\nwrote {out}/train.jsonl ({len(train)}) and {out}/valid.jsonl ({len(valid)})"
          + ("  [SMOKE TEST: valid is a copy of two train rows]" if args.smoke else ""))
    if not args.smoke:
        print(f"train questions without a translation: {len(missing_train)}"
              + (f" ({', '.join(missing_train)})" if missing_train else ""))
        print(f"validation questions without a translation: {len(missing_valid)} "
              "(they are still evaluated; they are only missing from the validation loss)")
    print(f"\ntokens{'' if exact else ' (estimated)'}: longest example {mx_total}, "
          f"longest answer {mx_target}, mean answer {stats['mean_target']}")
    print(f"train with:    --max-seq-length {max_seq}")
    print(f"eval_aspic.py will generate up to {max_tokens} tokens per answer (1.5x the longest answer)")

    if not TYPES_FILE.exists() and not args.smoke:
        types = statement_types(bench, valid_ids)
        json.dump(types, open(TYPES_FILE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        n = sum(len(v) for v in types.values())
        w = sum(s["type"] != "structure" for v in types.values() for s in v.values())
        print(f"\nwrote {TYPES_FILE.name}: {n} validation statements, {w} tagged weakens/strengthens by keyword.")
        print("Read it once and fix any tag that is wrong; this script will not overwrite it.")


if __name__ == "__main__":
    main()
