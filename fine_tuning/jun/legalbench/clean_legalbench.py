#!/usr/bin/env python3
"""Merge, reindex, clean and convert the two LegalBench benchmarks (issue #33).

For each task (legal_reasoning_causality, privacy_policy_entailment):
  1. Load the train and test TSVs (train first, then test).
  2. Clean the text fields. Every fix is counted, with record indices, in
     legalbench_cleaning_log.json:
       - whitespace: non-breaking spaces become plain spaces, invisible
         characters (U+200B-U+200F, U+FEFF) are removed, runs of spaces become
         one space, spaces around line breaks and at the start/end are removed;
       - court-reporter page markers such as [*27] or [**6] are removed;
       - legal-database headnote markers such as HN26 are removed;
       - HTML: <wbr> tags are removed and <a ...>link text</a> becomes its link text;
       - the PDF ligature characters U+FB01 and U+FB02 become "fi" and "fl";
       - in privacy descriptions, the source typo "by by" becomes "by".
     Answers are never changed.
  3. Drop duplicates: two rows are duplicates if their content fields match
     after whitespace collapsing and lowercasing. The first occurrence is kept.
     Rows whose duplicates disagree on the answer are kept and flagged.
  4. Reindex from 0. Output columns are the TSV columns in TSV order, plus
     original_split and original_index pointing back to the source row.
  5. Write one JSON file per task and a cleaning log, verify every output
     record against its source TSV row, and write a seeded random sample
     (100 per task, or all if fewer) for a manual spot check.

Usage:
  python3 clean_legalbench.py --in-dir <repo>/benchmarks --out-dir <out> [--log-dir <dir>]
From the repository root, the files in the repo were made with:
  python3 fine_tuning/jun/legalbench/clean_legalbench.py --in-dir benchmarks \
      --out-dir benchmarks --log-dir fine_tuning/jun/legalbench
The cleaned JSON files go to --out-dir; legalbench_cleaning_log.json and the
spot_check_*.md files go to --log-dir (default: --out-dir).
legalbench_review_desk.html shows the same cleaning record by record, before and after.
The script is deterministic: same inputs give byte-identical outputs.
Only the Python standard library is used.
"""

import argparse
import csv
import difflib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

TASKS = {
    "legal_reasoning_causality": {
        "content_fields": ["text"],
        "labels": ["Yes", "No"],
        "tsv_columns": ["answer", "text"],  # same order as the TSV, after "index"
    },
    "privacy_policy_entailment": {
        "content_fields": ["text", "description"],
        "labels": ["Correct", "Incorrect"],
        "tsv_columns": ["text", "description", "answer"],
    },
}

SPLITS = ["train", "test"]
SAMPLE_SIZE = 100
SEED = 33  # issue number, fixed so the spot-check sample is reproducible

NBSP = "\u00a0"
INVISIBLE = re.compile("[\u200b-\u200f\ufeff]")
LIGATURES = {"\ufb01": "fi", "\ufb02": "fl"}
PAGE_MARKER = re.compile(r"[ \t]*\[\*+\d+\][ \t]*")
HEADNOTE_MARKER = re.compile(r"\bHN\d+\b[ \t]*")
HTML_WBR = re.compile(r"<wbr\s*/?>", re.I)
HTML_LINK = re.compile(r"<a\b[^>]*>(.*?)</a>", re.I | re.S)
BY_BY = re.compile(r"\bby by\b")

# Still reported after cleaning, so anything left over is visible in the log.
REPORT_PATTERNS = {
    "page_marker": re.compile(r"\[\*+\d+\]"),
    "headnote_marker": re.compile(r"\bHN\d+\b"),
    "html_tag": re.compile(r"<[a-zA-Z/][^>]*>"),
    "html_entity": re.compile(r"&[a-zA-Z]+;|&#\d+;"),
    "ligature": re.compile("[\ufb01\ufb02]"),
    "internal_newline": re.compile(r"\n"),
}


def clean_field(field, s):
    """Return the cleaned string and the names of the fixes that changed it."""
    fixes = []

    def step(name, new):
        nonlocal s
        if new != s:
            fixes.append(name)
            s = new

    step("whitespace: non-breaking space -> space", s.replace(NBSP, " "))
    step("whitespace: removed invisible characters", INVISIBLE.sub("", s))
    step("html: removed <wbr> tags", HTML_WBR.sub("", s))
    step("html: replaced <a> link with its link text", HTML_LINK.sub(r"\1", s))
    for lig, plain in LIGATURES.items():
        step(f"ligature: {lig} -> {plain}", s.replace(lig, plain))
    step("removed page markers like [*27]", PAGE_MARKER.sub(" ", s))
    step("removed headnote markers like HN26", HEADNOTE_MARKER.sub("", s))
    if field == "description":
        step("description: 'by by' -> 'by'", BY_BY.sub("by", s))
    step("whitespace: collapsed repeated spaces", re.sub(r" {2,}", " ", s))
    step("whitespace: removed spaces around line breaks", re.sub(r" *\n *", "\n", s))
    step("whitespace: stripped start/end", s.strip())
    return s, fixes


def dup_key(rec, fields):
    return tuple(re.sub(r"\s+", " ", rec[f]).strip().lower() for f in fields)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        if None in r:
            raise ValueError(f"{path}: row with extra columns at index {r.get('index')}")
    return rows


def process_task(task, cfg, in_dir, out_dir):
    fields = cfg["content_fields"]
    log = {"task": task, "input_rows": {}, "dropped_duplicates": [], "conflicting_duplicates": [],
           "unexpected_labels": [], "empty_rows": []}

    merged = []
    for split in SPLITS:
        path = in_dir / f"LegalBench_{task}_{split}.tsv"
        rows = read_tsv(path)
        log["input_rows"][split] = len(rows)
        if [r["index"] for r in rows] != [str(i) for i in range(len(rows))]:
            raise ValueError(f"{path}: index column is not 0..{len(rows) - 1}")
        for r in rows:
            rec = {"original_split": split, "original_index": int(r["index"]), "fixes": {}}
            for f in fields:
                rec[f], rec["fixes"][f] = clean_field(f, r[f])
            rec["answer"] = r["answer"].strip()
            merged.append(rec)

    kept, seen = [], {}
    for rec in merged:
        ref = f"{rec['original_split']}:{rec['original_index']}"
        if any(not rec[f] for f in fields) or not rec["answer"]:
            log["empty_rows"].append(ref)
            continue
        if rec["answer"] not in cfg["labels"]:
            log["unexpected_labels"].append({"row": ref, "answer": rec["answer"]})
        k = dup_key(rec, fields)
        if k in seen:
            first = seen[k]
            first_ref = f"{first['original_split']}:{first['original_index']}"
            if first["answer"] == rec["answer"]:
                same = all(first[f] == rec[f] for f in fields)
                log["dropped_duplicates"].append({
                    "dropped": ref, "kept": first_ref,
                    "match": "identical after cleaning" if same else "differs only in case/whitespace"})
                continue
            log["conflicting_duplicates"].append({"rows": [first_ref, ref],
                                                  "answers": [first["answer"], rec["answer"]]})
        else:
            seen[k] = rec
        kept.append(rec)

    out = []
    fixes_by_name = defaultdict(list)
    left_over = defaultdict(list)
    for i, rec in enumerate(kept):
        o = {"index": i}
        for col in cfg["tsv_columns"]:
            o[col] = rec[col]
        o["original_split"] = rec["original_split"]
        o["original_index"] = rec["original_index"]
        out.append(o)
        for f in fields:
            for name in rec["fixes"][f]:
                fixes_by_name[f"{f}: {name}"].append(i)
            for name, pat in REPORT_PATTERNS.items():
                if pat.search(o[f]):
                    left_over[f"{f}: {name}"].append(i)

    out_path = out_dir / f"LegalBench_{task}_cleaned.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    log["output_rows"] = len(out)
    log["label_counts"] = dict(Counter(o["answer"] for o in out))
    log["split_counts"] = dict(Counter(o["original_split"] for o in out))
    log["records_changed_per_fix"] = {k: {"count": len(v), "indices": v}
                                      for k, v in sorted(fixes_by_name.items())}
    log["still_present_after_cleaning"] = {k: {"count": len(v), "indices": v}
                                           for k, v in sorted(left_over.items())}
    return out_path, log


# ---------- independent verification ----------

ALLOWED_REMOVED_WORD = re.compile(r"\[\*+\d+\]|HN\d+")


def words(s):
    for lig, plain in LIGATURES.items():
        s = s.replace(lig, plain)
    s = INVISIBLE.sub("", s.replace(NBSP, " "))
    return re.findall(r"\S+", s)


def check_field(field, raw, cleaned):
    """Compare word by word. Only markers, HTML and 'by by' may differ."""
    a, b = words(raw), words(cleaned)
    problems = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        removed, added = a[i1:i2], b[j1:j2]
        # A marker may be its own word or stuck to punctuation, e.g. '"HN16'.
        if "".join(ALLOWED_REMOVED_WORD.sub("", x) for x in removed) == "".join(added):
            continue
        if field == "description" and not added and removed == ["by"]:
            continue
        raw_chunk = " ".join(removed)
        if "<" in raw_chunk:  # HTML link: kept words must come from the raw text with tags removed
            visible = HTML_WBR.sub("", raw_chunk)
            visible = re.sub(r"<[^>]*>", "", visible)
            if all(w in visible for w in added):
                continue
        problems.append(f"{op} {removed[:4]} -> {added[:4]}")
    return problems


def verify(task, cfg, in_dir, out_path):
    fields = cfg["content_fields"]
    src = {}
    for split in SPLITS:
        for r in read_tsv(in_dir / f"LegalBench_{task}_{split}.tsv"):
            src[(split, int(r["index"]))] = r
    data = json.loads(out_path.read_text(encoding="utf-8"))
    errors = []
    if [o["index"] for o in data] != list(range(len(data))):
        errors.append("index is not contiguous from 0")
    if len({(o["original_split"], o["original_index"]) for o in data}) != len(data):
        errors.append("an original row appears twice")
    for o in data:
        s = src.get((o["original_split"], o["original_index"]))
        if s is None:
            errors.append(f"index {o['index']}: no source row")
            continue
        if o["answer"] != s["answer"].strip():
            errors.append(f"index {o['index']}: answer differs from source")
        for f in fields:
            for p in check_field(f, s[f], o[f]):
                errors.append(f"index {o['index']} {f}: unexpected change {p}")
    return data, errors


def write_spot_check(task, cfg, data, out_dir):
    rng = random.Random(SEED)
    n = min(SAMPLE_SIZE, len(data))
    sample = sorted(rng.sample(range(len(data)), n))
    lines = [f"# Spot check: LegalBench {task}", "",
             f"{n} of {len(data)} records, random seed {SEED}. For each, compare against the source TSV row "
             f"named in `original_split` / `original_index` and tick the box.", ""]
    for i in sample:
        o = data[i]
        lines.append(f"## [ ] index {i} (source: {o['original_split']}.tsv index {o['original_index']}) "
                     f"answer: {o['answer']}")
        for f in cfg["content_fields"]:
            lines += ["", f"**{f}:**", "", "> " + o[f].replace("\n", "\n> ")]
        lines.append("")
    path = out_dir / f"spot_check_{task}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True, help="folder with the four LegalBench TSVs")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the cleaned JSON files go")
    ap.add_argument("--log-dir", type=Path, help="where the log and spot checks go (default: --out-dir)")
    args = ap.parse_args()
    args.log_dir = args.log_dir or args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    logs, ok = {}, True
    for task, cfg in TASKS.items():
        out_path, log = process_task(task, cfg, args.in_dir, args.out_dir)
        data, errors = verify(task, cfg, args.in_dir, out_path)
        spot_path, sample = write_spot_check(task, cfg, data, args.log_dir)
        log["verification_errors"] = errors
        log["spot_check_sample_indices"] = sample
        logs[task] = log
        ok &= not errors
        print(f"{task}: train {log['input_rows']['train']} + test {log['input_rows']['test']} "
              f"= {sum(log['input_rows'].values())} -> {log['output_rows']} "
              f"({len(log['dropped_duplicates'])} duplicate(s) dropped); labels {log['label_counts']}; "
              f"verification {'OK' if not errors else 'FAILED: ' + '; '.join(errors[:5])}")
        for name, v in log["records_changed_per_fix"].items():
            print(f"    {v['count']:>5}  {name}")
        for name, v in log["still_present_after_cleaning"].items():
            print(f"    left: {v['count']:>3}  {name}")
    (args.log_dir / "legalbench_cleaning_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print("wrote the cleaned JSON files to", args.out_dir)
    print("wrote legalbench_cleaning_log.json and spot_check_*.md to", args.log_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
