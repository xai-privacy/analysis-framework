"""Clean the MBE-exam-questions dataset (HolySaint/MBE-exam-questions).

Inputs (paths are arguments, see --help):
  distilled  MBE-exam-questions_distilled_dataset.json   (IRAC explanations)
  review     review notes exported from mbe_review_desk.html ("Copy review notes (JSON)")
  dups       mbe_duplicate_drops.json                     (exact-duplicate copies to drop)

What it does, in order:
  1. Drops records marked "drop" in the review notes.
  2. Drops the exact-duplicate copies listed in the dups file.
  3. Applies answer-key corrections (KEY_FIXES below).
  4. Strips leading/trailing spaces and collapses repeated spaces in `question`.
The raw file (human explanations) is not changed; it stays the reference for checking keys.
Everything else (field names, order, options, explanations) is left unchanged,
so the cleaned files load anywhere the originals do.

Outputs: <out>/MBE-exam-questions_distilled_dataset_cleaned.json,
         <out>/mbe_cleaning_log.json (one entry per change).
"""
import argparse, json, re, collections

# Record ids follow the viewer: "<domain lowercased, _ -> ->-<question_number>",
# with "-dup2" for the second record that shares a domain and number.
def record_ids(records):
    seen = collections.Counter()
    ids = []
    for r in records:
        base = f"{r['domain'].lower().replace('_', '-')}-{r['question_number']}"
        seen[base] += 1
        ids.append(base if seen[base] == 1 else f"{base}-dup{seen[base]}")
    return ids

# Answer-key corrections. Each one is listed for review by Sebastian (legal side).
KEY_FIXES = {
    "constitutional-law-217": {
        "from": "A", "to": "B",
        "why": "Human explanation says B is correct; the IRAC reasoning (state acted as a market "
               "participant) also matches option B, only its last words named A.",
        # the IRAC conclusion's reasoning already supports B; only the option letter changes
        "irac_conclusion": ("supporting option A.", "supporting option B."),
    },
}

def norm_q(s):
    return re.sub(r" {2,}", " ", s).strip()

def is_drop(note):
    # "drop" as a decision; "dropping X" describes a different record
    return bool(re.search(r"\bdrop\b", note, re.I)) and "dropping" not in note.lower()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--dups", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    D = json.load(open(a.distilled, encoding="utf-8"))
    review = json.load(open(a.review, encoding="utf-8"))
    dups = json.load(open(a.dups, encoding="utf-8"))
    ids = record_ids(D)
    by_id = dict(zip(ids, D))
    log = []

    review_drop = {x["id"]: x["note"] for x in review if is_drop(x.get("note", ""))}
    dup_drop = {x["id"]: x for x in dups}
    unknown = [i for i in list(review_drop) + list(dup_drop) if i not in by_id]
    if unknown:
        raise SystemExit(f"ids not found in the distilled file: {unknown}")

    kept = []
    for i, r in zip(ids, D):
        if i in review_drop:
            log.append({"id": i, "action": "drop", "source": "manual review", "reason": review_drop[i]})
        elif i in dup_drop:
            log.append({"id": i, "action": "drop", "source": "duplicate", "kept_copy": dup_drop[i]["keep"],
                        "reason": dup_drop[i]["reason"]})
        else:
            kept.append((i, r))

    out_d = []
    for i, r in kept:
        r = json.loads(json.dumps(r))  # copy
        if i in KEY_FIXES:
            fx = KEY_FIXES[i]
            assert r["correct_answer"] == fx["from"], (i, r["correct_answer"])
            r["correct_answer"] = fx["to"]
            old, new = fx["irac_conclusion"]
            assert r["explanation"]["Legal_Conclusion"].count(old) == 1, i
            r["explanation"]["Legal_Conclusion"] = r["explanation"]["Legal_Conclusion"].replace(old, new)
            log.append({"id": i, "action": "change key", "from": fx["from"], "to": fx["to"], "reason": fx["why"],
                        "also_changed": f"IRAC Legal_Conclusion: '{old}' -> '{new}'", "review": "Sebastian (legal)"})
        q = norm_q(r["question"])
        if q != r["question"]:
            log.append({"id": i, "action": "fix whitespace", "field": "question"})
            r["question"] = q
        out_d.append(r)

    pre = "MBE-exam-questions"
    json.dump(out_d, open(f"{a.out}/{pre}_distilled_dataset_cleaned.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    summary = collections.Counter((e["action"], e.get("source", "")) for e in log)
    json.dump({"input": len(D),
               "output": len(out_d),
               "summary": {f"{k[0]} ({k[1]})" if k[1] else k[0]: v for k, v in summary.items()},
               "changes": log},
              open(f"{a.out}/mbe_cleaning_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"distilled {len(D)} -> {len(out_d)}")
    for k, v in summary.items():
        print(f"  {k[0]}{' (' + k[1] + ')' if k[1] else ''}: {v}")

if __name__ == "__main__":
    main()
