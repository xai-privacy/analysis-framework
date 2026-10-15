#!/usr/bin/env python3
"""Quick test: can base Llama 3.2 3B write ASPIC+ translations from one example?

For each question, the model (through Ollama) gets:
  system:    the frozen instruction from ASPIC_FORMAT.md
  user:      one example question            assistant: its translation
  user:      the question to translate
and must answer with the translation JSON. The script then scores it with
check_aspic.py. No training is involved.

Scores per question and in total:
  parsed      the answer contains valid JSON
  valid       the JSON is a usable translation (every ID defined, every statement checked)
  statements  true/false per statement compared with the gold labels
              (derived from the gold answer, the choice list and the question wording)
  answer      the choice the check script computes, compared with the gold answer

Setup (once):  ollama pull llama3.2:3b      (Ollama must be running)
Usage, from fine_tuning/jun/aspic/ with the venv active:
  python test_base_model.py --bench $B --split $S --limit 5
  python test_base_model.py --bench $B --split $S --json-mode     # Ollama forces valid JSON syntax
  python test_base_model.py --bench $B --split $S --ids 2021_28 2021_34

By default it runs on the validation questions, so the answer numbers can be put
next to your first fine-tuning run (base: 27/51 statements, 3/15 questions).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from check_aspic import (FINAL_TEST_IDS, asks_for, check_translation, gold_number,
                         load_bench, parse_choices)

HERE = Path(__file__).resolve().parent
DROP = {"note", "summary", "source"}


def frozen_instruction():
    text = (HERE / "ASPIC_FORMAT.md").read_text(encoding="utf-8")
    m = re.search(r"## Frozen instruction.*?```\n(.*?)\n```", text, re.S)
    if not m:
        sys.exit("could not find the frozen instruction in ASPIC_FORMAT.md")
    return m.group(1).strip()


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in DROP}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def question_text(q):
    """Passage, question and statements; the (a)/(b)/(c) choice list is removed."""
    text = q["original_question"]
    mode, _ = parse_choices(text)
    if mode == "combo":
        text = re.split(r"<choices>", text, flags=re.I)[0].rstrip()
    return text


def gold_labels(q):
    text = q["original_question"]
    mode, choices = parse_choices(text)
    gold = gold_number(q["answer"])
    want = asks_for(text) == "correct"
    if mode == "combo":
        letters = sorted(set().union(*choices.values()))
        return {l: (l in choices[gold]) == want for l in letters}
    return {str(k): (k == gold) == want for k in range(1, 6)}


def extract_json(raw):
    s = re.sub(r"```(?:json)?", "", raw)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer")
    return json.loads(s[start:end + 1])


def ask_ollama(url, model, messages, json_mode, num_ctx, num_predict):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ids", nargs="*", help="questions to test (default: the validation questions)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--example", default=None,
                    help="translation file used as the one example "
                         "(default: translations/2022_17.json, else translations_claude/2022_17.json)")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--json-mode", action="store_true", help="let Ollama force valid JSON syntax")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=2048)
    ap.add_argument("--out", default=None, help="results folder (default: base_test_<time>)")
    args = ap.parse_args()

    bench = load_bench(args.bench)
    split = json.load(open(args.split))

    ex_path = Path(args.example) if args.example else next(
        (p for p in (HERE / "translations/2022_17.json", HERE / "translations_claude/2022_17.json") if p.exists()), None)
    if not ex_path or not ex_path.exists():
        sys.exit("no example translation found; pass --example")
    example = json.load(open(ex_path, encoding="utf-8"))
    ex_id = example["id"]

    ids = args.ids or split["valid"]
    if args.limit:
        ids = ids[: args.limit]
    for i in ids:
        if i in FINAL_TEST_IDS:
            sys.exit(f"{i} is a final test question. Do not run it.")
        if i == ex_id:
            sys.exit(f"{i} is the example question; pick another --example or drop it from --ids")
        if i not in bench:
            sys.exit(f"{i} not in {args.bench}")

    system = frozen_instruction()
    shots = [
        {"role": "user", "content": question_text(bench[ex_id])},
        {"role": "assistant", "content": json.dumps(strip(example), ensure_ascii=False)},
    ]

    out = Path(args.out or f"base_test_{time.strftime('%Y%m%d_%H%M%S')}")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    print(f"model {args.model} | example {ex_id} ({ex_path}) | json-mode {args.json_mode} | {len(ids)} questions")
    print(f"results in {out}/\n")

    rows = []
    for n, qid in enumerate(ids, 1):
        q = bench[qid]
        gold = gold_labels(q)
        messages = [{"role": "system", "content": system}] + shots + [
            {"role": "user", "content": question_text(q)}]
        t0 = time.time()
        try:
            raw = ask_ollama(args.url, args.model, messages, args.json_mode, args.num_ctx, args.num_predict)
        except Exception as e:
            sys.exit(f"Ollama call failed ({e}). Is Ollama running, and did you run 'ollama pull {args.model}'?")
        secs = time.time() - t0
        (out / "raw" / f"{qid}.txt").write_text(raw, encoding="utf-8")

        row = {"id": qid, "secs": round(secs, 1), "parsed": False, "valid": False,
               "stmt_ok": 0, "stmt_n": len(gold), "pred": None, "gold": gold_number(q["answer"]), "error": ""}
        try:
            t = extract_json(raw)
            t["id"] = qid
            row["parsed"] = True
            rec = check_translation(t, bench, split, FINAL_TEST_IDS, use_pyarg=False)
            if rec["errors"]:
                row["error"] = "; ".join(rec["errors"])[:120]
            else:
                row["valid"] = True
                row["pred"] = rec["pred"]
                row["stmt_ok"] = sum(rec["truth"].get(k) == v for k, v in gold.items())
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:120]
        rows.append(row)

        status = ("valid" if row["valid"] else "parsed, invalid" if row["parsed"] else "not JSON")
        ans = f"answer {row['pred']} (gold {row['gold']})" if row["valid"] else row["error"]
        print(f"[{n}/{len(ids)}] {qid}: {status:15} statements {row['stmt_ok']}/{row['stmt_n']}  {ans}  {secs:.0f}s")

    N = len(rows)
    parsed = sum(r["parsed"] for r in rows)
    valid = sum(r["valid"] for r in rows)
    s_ok = sum(r["stmt_ok"] for r in rows)
    s_n = sum(r["stmt_n"] for r in rows)
    q_ok = sum(r["valid"] and r["pred"] == r["gold"] for r in rows)
    print(f"\nparsed JSON      {parsed}/{N}")
    print(f"valid translation {valid}/{N}")
    print(f"statements       {s_ok}/{s_n}   (invalid translations count as 0)")
    print(f"questions        {q_ok}/{N}")
    json.dump({"model": args.model, "example": ex_id, "json_mode": args.json_mode, "rows": rows},
              open(out / "report.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
