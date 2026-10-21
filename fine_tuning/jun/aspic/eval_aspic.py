#!/usr/bin/env python3
"""Score the base model and fine-tuned checkpoints by running the solver on their output.

For each model and each validation question, the model writes an ASPIC+
translation from the raw question (greedy decoding). check_aspic.py then
computes the statements and the answer. No one reads the outputs by hand to
get the numbers.

Scores (per model):
  parsed       the output contains a JSON object
  valid        the JSON is a usable translation (IDs defined, every statement checked)
  stmt         statements whose computed true/false matches the gold label
               (an invalid translation or an unsupported statement counts as wrong)
  structure    the same, only for structure statements (agree, assume, hold, deny, ...)
  weak/str     the same, only for weakens/strengthens statements
  questions    computed answer = gold answer

Statement types come from statement_types.json (written by build_aspic_data.py,
hand-checked by you). If a question is missing there, a keyword rule is used.

Prompts:
  fine-tuned   exactly the training prompt (frozen instruction + question), no example
  base         the same, plus one worked example (2022_17), because the base model
               has never seen the format. --shots 0 runs it without the example.

Usage (from fine_tuning/jun/aspic/, venv active):
  python eval_aspic.py --base                                   # base model
  python eval_aspic.py --adapters adapters_aspic --iters 40 80 120 160 200 240
  python eval_aspic.py --adapters adapters_aspic                # every saved checkpoint
  python eval_aspic.py --summary                                # only print the table

Results go to eval_results/<name>.json (one per model, saved after every
question, so a stopped run continues where it left off) and the table to
eval_results/summary.md.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from check_aspic import FINAL_TEST_IDS, check_translation, gold_number, load_bench
from test_base_model import extract_json, frozen_instruction, gold_labels, question_text, strip

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = REPO / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
SPLIT = HERE.parent / "data" / "split.json"
MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TYPES_FILE = HERE / "statement_types.json"
WEAK_STRONG = re.compile(r"weaken|strengthen|support|undermin", re.I)
EXAMPLE = HERE / "translations_claude" / "2022_17.json"


def statement_types(bench, qid):
    types = {}
    if TYPES_FILE.exists():
        types = json.load(open(TYPES_FILE, encoding="utf-8")).get(qid, {})
    if types:
        return {k: v["type"] for k, v in types.items()}
    from build_aspic_data import statement_key
    out = {}
    for text in bench[qid]["statements"].values():
        k = statement_key(text)
        if k:
            out[k] = "weakens/strengthens" if WEAK_STRONG.search(text) else "structure"
    return out


def score(raw, q, bench, split):
    qid = q["id"]
    gold = gold_labels(q)
    rec = {"id": qid, "parsed": False, "valid": False, "pred": None,
           "gold": gold_number(q["answer"]), "statements": {}, "error": ""}
    truth = {}
    try:
        t = extract_json(raw)
        rec["parsed"] = True
        t["id"] = qid
        res = check_translation(t, bench, split, FINAL_TEST_IDS, use_pyarg=False)
        if res["errors"]:
            rec["error"] = "; ".join(res["errors"])[:200]
        else:
            rec["valid"] = True
            rec["pred"] = res["pred"]
            truth = res["truth"]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    types = statement_types(bench, qid)
    for k, g in gold.items():
        p = truth.get(k)
        rec["statements"][k] = {"gold": g, "pred": p, "ok": p == g, "type": types.get(k, "structure")}
    rec["question_ok"] = rec["valid"] and rec["pred"] == rec["gold"]
    return rec


def summarize(records):
    st = [s for r in records for s in r["statements"].values()]

    def frac(xs):
        return f"{sum(xs)}/{len(xs)}"
    return {
        "n": len(records),
        "parsed": frac([r["parsed"] for r in records]),
        "valid": frac([r["valid"] for r in records]),
        "stmt": frac([s["ok"] for s in st]),
        "structure": frac([s["ok"] for s in st if s["type"] == "structure"]),
        "weak/str": frac([s["ok"] for s in st if s["type"] != "structure"]),
        "unsupported": sum(r["valid"] and s["pred"] is None for r in records for s in r["statements"].values()),
        "questions": frac([r["question_ok"] for r in records]),
    }


def pct(s):
    a, b = (int(x) for x in s.split("/"))
    return f"{s} ({100 * a / b:.0f}%)" if b else s


def print_summary(outdir):
    files = sorted(Path(outdir).glob("*.json"))
    if not files:
        print("no results yet")
        return
    cols = ["parsed", "valid", "stmt", "structure", "weak/str", "questions"]
    lines = ["| model | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        s = summarize(d["records"])
        lines.append(f"| {f.stem} | {s['n']} | " + " | ".join(pct(s[c]) for c in cols) + " |")
    table = "\n".join(lines)
    (Path(outdir) / "summary.md").write_text(
        "Validation results. Statements: invalid translations and unsupported statements count as wrong.\n\n"
        + table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwritten to {outdir}/summary.md")


def checkpoints(adapters, iters):
    adapters = Path(adapters)
    if not (adapters / "adapter_config.json").exists():
        sys.exit(f"{adapters}/adapter_config.json not found")
    found = {int(p.name.split("_")[0]): p for p in adapters.glob("[0-9]*_adapters.safetensors")}
    if iters:
        missing = [i for i in iters if i not in found]
        if missing:
            sys.exit(f"no checkpoint for iteration(s) {missing} in {adapters}/ (have {sorted(found)})")
        return [(i, found[i]) for i in iters]
    if not found:
        sys.exit(f"no NNNNNNN_adapters.safetensors checkpoints in {adapters}/")
    return sorted(found.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--model", default=MODEL)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", action="store_true", help="evaluate the base model")
    g.add_argument("--adapters", help="adapter folder from mlx_lm.lora; evaluates its checkpoints")
    g.add_argument("--summary", action="store_true", help="only print the results table")
    ap.add_argument("--iters", nargs="*", type=int, help="checkpoint iterations to evaluate (default: all)")
    ap.add_argument("--shots", type=int, default=1, choices=[0, 1], help="worked examples for the base model")
    ap.add_argument("--ids", nargs="*", help="questions (default: all validation questions)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="default: eval_max_tokens from data_aspic/stats.json, else 3000")
    ap.add_argument("--stats", default=str(HERE / "data_aspic" / "stats.json"))
    ap.add_argument("--out", default=str(HERE / "eval_results"))
    ap.add_argument("--tag", default="", help="added to result file names, e.g. 'smoke'")
    args = ap.parse_args()

    if args.summary:
        print_summary(args.out)
        return

    bench = load_bench(args.bench)
    split = json.load(open(args.split))
    ids = args.ids or sorted(split["valid"])
    if args.limit:
        ids = ids[: args.limit]
    bad = [i for i in ids if i in FINAL_TEST_IDS]
    if bad:
        sys.exit(f"refusing to run on final test questions: {bad}")
    train_overlap = [i for i in ids if i in split["train"]]
    if train_overlap:
        print(f"warning: these are training questions, so their scores are not a fair test: {train_overlap}")

    max_tokens = args.max_tokens
    if not max_tokens:
        try:
            max_tokens = json.load(open(args.stats))["eval_max_tokens"]
        except Exception:
            max_tokens = 3000
    print(f"generating up to {max_tokens} tokens per answer, {len(ids)} questions")

    from mlx_lm import generate, load  # imported here so --summary works without MLX
    import mlx.core as mx

    system = frozen_instruction()
    runs = []  # (name, adapter weights file or None, shots)
    if args.base:
        runs.append((f"base_{args.shots}shot", None, args.shots))
    else:
        for it, f in checkpoints(args.adapters, args.iters):
            runs.append((f"{Path(args.adapters).name}_iter{it:04d}", f, 0))

    shots_msgs = []
    if any(r[2] for r in runs):
        ex = json.load(open(EXAMPLE, encoding="utf-8"))
        if ex["id"] in ids:
            sys.exit(f"{ex['id']} is the worked example; it cannot also be evaluated")
        ex_out = strip(ex)
        ex_out.pop("id", None)
        shots_msgs = [{"role": "user", "content": question_text(bench[ex["id"]])},
                      {"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)}]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    stage = None
    for name, weights, shots in runs:
        name = f"{name}_{args.tag}" if args.tag else name
        out = outdir / f"{name}.json"
        records = []
        if out.exists():
            records = [r for r in json.load(open(out, encoding="utf-8"))["records"] if r["id"] in ids]
        done = {r["id"] for r in records}
        todo = [i for i in ids if i not in done]
        if not todo:
            print(f"{name}: already done ({len(records)} questions)")
            continue

        # Load the model once; for each checkpoint, swap in its adapter weights.
        if weights is None:
            model, tokenizer = load(args.model)
            stage = None
        elif model is None or stage is None:
            stage = outdir / "_adapter_stage"
            stage.mkdir(exist_ok=True)
            shutil.copy(Path(args.adapters) / "adapter_config.json", stage / "adapter_config.json")
            shutil.copy(weights, stage / "adapters.safetensors")
            model, tokenizer = load(args.model, adapter_path=str(stage))
        else:
            model.load_weights(list(mx.load(str(weights)).items()), strict=False)

        print(f"\n{name}: {len(todo)} questions" + (f" (resuming, {len(done)} done)" if done else ""))
        for qid in todo:
            q = bench[qid]
            msgs = [{"role": "system", "content": system}] + (shots_msgs if shots else []) + \
                   [{"role": "user", "content": question_text(q)}]
            prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            t0 = time.time()
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
            rec = score(raw, q, bench, split)
            rec["seconds"] = round(time.time() - t0, 1)
            rec["output"] = raw
            records.append(rec)
            tmp = out.with_suffix(".tmp")
            json.dump({"model": args.model, "adapter": str(weights) if weights else None, "shots": shots,
                       "max_tokens": max_tokens, "records": records},
                      open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            tmp.replace(out)
            ok = sum(s["ok"] for s in rec["statements"].values())
            status = "valid" if rec["valid"] else ("invalid" if rec["parsed"] else "no JSON")
            print(f"  {qid}: {status:7} statements {ok}/{len(rec['statements'])}  "
                  f"answer {rec['pred']} (gold {rec['gold']})  {rec['seconds']}s"
                  + (f"  {rec['error'][:80]}" if rec["error"] else ""), flush=True)
        s = summarize(records)
        print(f"{name}: valid {s['valid']}, statements {s['stmt']} "
              f"(structure {s['structure']}, weak/str {s['weak/str']}), questions {s['questions']}")
    if stage and stage.exists():
        shutil.rmtree(stage)
    print()
    print_summary(outdir)


if __name__ == "__main__":
    main()
