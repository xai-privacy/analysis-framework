#!/usr/bin/env python3
"""Can a frontier model write the ASPIC+ structure WITHOUT the expert rationale?

README "Next steps" 1: Gemini translates the 15 validation questions without
the expert rationale, in one try with no feedback, and the solver scores it.
If the solver cannot reach good answers from a strong model's structures,
training a small model to write them will not help.

How this differs from teacher.py (do not use teacher.py for this run):
  - no expert rationale anywhere: not for the question, not for the examples
  - the question text is question_text(): exactly what the fine-tuned model
    saw (passage, question and statements, without the (a)/(b)/(c) choice list)
  - the worked examples are the same three training questions teacher.py used
    (2022_17, 2021_37, 2024_15), shown without their rationale. In their
    translations, "source": "rationale" is relabelled "background" and the
    "note" fields are removed, since they refer to the rationale
  - one try per question, no feedback, the gold answer is never shown
  - nothing is written to translations/, gemini_raw/ or teacher_log.csv

Commands (run from fine_tuning/jun/aspic/ with the venv active):

  python gemini_norationale.py prompts     write norationale/prompts/<id>.md (15 validation questions)
  python gemini_norationale.py run         macOS, several Gemini tabs: Enter copies the next prompt;
                                           every Gemini answer you copy is found by its "id" and saved
  python gemini_norationale.py next [ID]   copy one prompt to the clipboard
  python gemini_norationale.py save ID     save Gemini's answer from the clipboard (or --file answer.txt)
  python gemini_norationale.py status      what is saved and what is left
  python gemini_norationale.py score       score the saved answers with the solver
                                           -> eval_results/gemini_norationale.json
  python gemini_norationale.py reference   score Gemini WITH rationale, first try, from gemini_raw/<id>_try1.txt
                                           -> eval_results/gemini_withrationale_try1.json

Then `python rescore_aspic.py` puts both next to the fine-tuned model in
eval_results/summary_repaired.md (same columns as the README table).

The "frontier, no solver" cell of the 2x2 (Gemini answers the question itself):

  python gemini_norationale.py --cell direct prompts    -> direct/prompts/<id>.md
  python gemini_norationale.py --cell direct run        same workflow, new Gemini chat per question
  python gemini_norationale.py --cell direct score      -> direct/results.json, and the comparison
  python gemini_norationale.py compare                  print the comparison again (direct/compare.md)

Gemini setting: the same as teacher.py (Flash 3.8, medium effort), a NEW chat for
every question. Write the setting with --teacher if it is different.
"""
import argparse
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

from check_aspic import FINAL_TEST_IDS, asks_for, gold_number, load_bench, parse_choices, predict
from eval_aspic import pct, score, summarize
from teacher import (EXAMPLE_POOL, EXAMPLES_DIR, copy_to_clipboard, format_spec, read_clipboard)
from test_base_model import gold_labels, question_text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = REPO / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
SPLIT = HERE.parent / "data" / "split.json"
OUT = HERE / "norationale"
PROMPTS = OUT / "prompts"
RAW = OUT / "raw"
LOG = OUT / "log.csv"
RESULTS = HERE / "eval_results"
RESULT_FILE = RESULTS / "gemini_norationale.json"
REFERENCE_FILE = RESULTS / "gemini_withrationale_try1.json"
EXAMPLES = EXAMPLE_POOL[:3]  # 2022_17, 2021_37, 2024_15: all training questions
LOG_FIELDS = ["time", "id", "event", "chars", "teacher"]

INTRO = """I need you to translate one LEET argument question into a JSON format based on ASPIC+ argument structure. This message has three parts: (1) the format, (2) worked examples with finished translations, and (3) the question to translate.

Translate the question in Part 3 into the format in Part 1, in the same style as the examples in Part 2.

Rules:
1. Build the structure from the passage. There is no expert rationale in this task. If the reasoning needs a premise or rule the passage does not state, supply it yourself and mark it "source": "background". Ignore the "rationale" source value in Part 1; use "passage", "statement" or "background".
2. Do not write verdicts anywhere: no "correct", "incorrect", "true", "false", and no answer number. Whether a statement is true must follow only from the structure and its check.
3. Every element must be justified by the passage or by background knowledge needed to read it. Do not add a rule, contrary or premise only to make a check come out a certain way.
4. If a statement is false because a tempting inference fails, write that inference and the reason it fails (often an undercut), as in statement (c) of 2021_37.
5. Keep claim text close to the passage wording. Do not invent speakers or facts. Use the position labels the passage uses.
6. Single-choice questions (the choices are the statements themselves) use the keys "1" to "5" in "statements".
7. If a statement cannot be expressed with the check operations in Part 1, use {"op": "unsupported", "reason": "..."} instead of forcing it.
8. Before answering, check that every claim ID, rule ID, position label and hypothetical name you reference is defined.

Output: one JSON code block containing one object with "id": "QID". After the code block, list in plain text every statement you were unsure about, and why.
"""

OUTRO = """
---

Now translate question QID from Part 3. Follow rules 1 to 8 above. Output one JSON code block containing one object with "id": "QID", then list the statements you were unsure about.
"""

ID_RE = re.compile(r'"id"\s*:\s*"(\d{4}_\d{2})"')


# --------------------------------------------------------------------------
# prompts


def no_rationale(obj):
    """The example translation as shown to Gemini: rationale -> background, notes removed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "note":
                continue
            if k == "source" and v == "rationale":
                v = "background"
            out[k] = no_rationale(v)
        return out
    if isinstance(obj, list):
        return [no_rationale(v) for v in obj]
    return obj


def question_block(q):
    return f"#### Question {q['id']}\n\nQuestion text:\n\n{question_text(q)}\n"


def build_prompt(qid, bench):
    parts = [INTRO.replace("QID", qid), "## Part 1. Format\n", format_spec(), "\n## Part 2. Worked examples\n"]
    for ex in EXAMPLES:
        t = no_rationale(json.load(open(EXAMPLES_DIR / f"{ex}.json", encoding="utf-8")))
        parts.append(question_block(bench[ex]))
        parts.append("Translation:\n\n```json\n" + json.dumps(t, ensure_ascii=False, indent=1) + "\n```\n")
    parts.append("\n## Part 3. Question to translate\n")
    parts.append(question_block(bench[qid]))
    parts.append(OUTRO.replace("QID", qid))
    return "\n".join(parts)


def leak_check(text, qid, bench):
    """Stop if anything from the rationale or the answer made it into a prompt."""
    q = bench[qid]
    problems = []
    if re.search(r"expert rationale:", text, re.I):
        problems.append("contains an 'Expert rationale:' block")
    r = q.get("original_rationale")
    pieces = r.values() if isinstance(r, dict) else [r or ""]
    for p in pieces:
        for sent in re.split(r"(?<=[.!?])\s+", str(p)):
            if len(sent) > 60 and sent in text:
                problems.append(f"contains a rationale sentence: {sent[:60]}...")
                break
    if re.search(r'"source"\s*:\s*"rationale"', text):
        problems.append('an example still has "source": "rationale"')
    if re.search(r"<choices>", text) and "<choices>" not in question_text(q):
        problems.append("contains the choice list")
    return problems


def load_ids(bench):
    split = json.load(open(SPLIT))
    ids = sorted(split["valid"])
    for i in ids:
        if i in FINAL_TEST_IDS:
            sys.exit(f"{i} is a final test question but it is in the validation split. Stop and check the split.")
        if i in EXAMPLES:
            sys.exit(f"{i} is a worked example and a validation question. Stop and check.")
        if i not in bench:
            sys.exit(f"{i} is not in {BENCH}")
    return split, ids


def cmd_prompts(args, bench):
    split, ids = load_ids(bench)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    for qid in ids:
        text = build_prompt(qid, bench)
        problems = leak_check(text, qid, bench)
        if problems:
            sys.exit(f"{qid}: " + "; ".join(problems))
        (PROMPTS / f"{qid}.md").write_text(text, encoding="utf-8")
    sizes = [len((PROMPTS / f"{q}.md").read_text(encoding="utf-8")) for q in ids]
    print(f"wrote {len(ids)} prompts to {PROMPTS.relative_to(HERE)}/ "
          f"({min(sizes) // 1000}-{max(sizes) // 1000}k characters each); leak check passed")
    if build_prompt is build_direct_prompt:
        print("direct cell: no format, no examples, no rationale")
    else:
        print(f"examples: {', '.join(EXAMPLES)} (training questions, shown without rationale)")


# --------------------------------------------------------------------------
# collecting answers


def read_log():
    if not LOG.exists():
        return []
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_log(row):
    OUT.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def saved(qid):
    return (RAW / f"{qid}.txt").exists()


def sent_ids():
    return {r["id"] for r in read_log() if r["event"] == "sent"}


def is_prompt(text):
    return ("Part 1. Format" in text and "Part 3. Question to translate" in text) or DIRECT_MARK in text


def save_answer(qid, raw, args, replace=False):
    """Save Gemini's answer once. Returns True if saved."""
    if saved(qid) and not replace:
        print(f"{qid} already has an answer. This run is one try per question, so it is not replaced.")
        print(f"(Only if you pasted the wrong thing: save {qid} --replace, which is logged.)")
        return False
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{qid}.txt").write_text(raw, encoding="utf-8")
    append_log({"time": now(), "id": qid, "event": "replaced" if replace else "saved",
                "chars": len(raw), "teacher": args.teacher})
    ids = ID_RE.findall(raw)
    if ids and qid not in ids:
        print(f"warning: the answer has id {ids[0]}, not {qid}. Check that you copied the right tab.")
    has_block = bool(re.search(r"```(?:json)?\s*\n", raw))
    print(f"{qid}: saved ({len(raw)} characters{'' if has_block else ', no ```json block found'})")
    return True


def cmd_next(args, bench):
    split, ids = load_ids(bench)
    if args.id:
        if args.id not in ids:
            sys.exit(f"{args.id} is not a validation question")
        qid = args.id
    else:
        left = [q for q in ids if not saved(q) and q not in sent_ids()]
        if not left:
            print("every prompt has been sent. Waiting for: "
                  + (", ".join(q for q in ids if not saved(q)) or "-"))
            return
        qid = left[0]
    p = PROMPTS / f"{qid}.md"
    if not p.exists():
        sys.exit("prompts are missing. Run: python gemini_norationale.py prompts")
    append_log({"time": now(), "id": qid, "event": "sent", "teacher": args.teacher})
    if not args.file and copy_to_clipboard(p.read_text(encoding="utf-8")):
        print(f"{qid}: prompt copied. Paste it into a NEW Gemini chat.")
    else:
        print(f"{qid}: prompt is {p}")
    print(f"then copy Gemini's whole answer and run: python gemini_norationale.py save {qid}")


def cmd_save(args, bench):
    split, ids = load_ids(bench)
    if args.id not in ids:
        sys.exit(f"{args.id} is not a validation question")
    raw = Path(args.file).read_text(encoding="utf-8") if args.file else read_clipboard()
    if raw is None:
        sys.exit("no clipboard on this machine; use --file")
    if not raw.strip():
        sys.exit("the clipboard is empty")
    if is_prompt(raw):
        sys.exit("the clipboard holds the prompt, not Gemini's answer")
    save_answer(args.id, raw, args, replace=args.replace)


RUN_HELP = """  Enter        copy the next prompt (paste it into a NEW Gemini tab)
  r ID         copy ID's prompt again (if you lost its tab)
  a ID         save the answer now on the clipboard as ID (if its id was not found)
  s            status          h  this help          q  quit"""


def cmd_run(args, bench):
    import queue
    import shutil
    import threading
    if not shutil.which("pbpaste"):
        sys.exit("run needs the macOS clipboard (pbcopy/pbpaste). Use next and save --file instead.")
    split, ids = load_ids(bench)
    if not all((PROMPTS / f"{q}.md").exists() for q in ids):
        sys.exit("prompts are missing. Run: python gemini_norationale.py prompts")
    seen, unclaimed = set(), [None]

    def put(text):
        seen.add(text)
        copy_to_clipboard(text)

    def waiting():
        return [q for q in ids if q in sent_ids() and not saved(q)]

    def show():
        done = sum(saved(q) for q in ids)
        print(f"saved {done}/{len(ids)}; waiting for: {', '.join(waiting()) or '-'}")

    def send(qid):
        put((PROMPTS / f"{qid}.md").read_text(encoding="utf-8"))
        append_log({"time": now(), "id": qid, "event": "sent", "teacher": args.teacher})
        print(f"prompt for {qid} copied: paste it into a NEW Gemini tab")

    lines = queue.Queue()

    def reader():
        while True:
            try:
                lines.put(input())
            except EOFError:
                lines.put("q")
                return

    seen.add(read_clipboard() or "")
    print("one try per question, no feedback. Copy each finished answer with Gemini's copy button;\n"
          "it is saved here by its \"id\".\n" + RUN_HELP)
    show()
    threading.Thread(target=reader, daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=0.7).strip()
        except queue.Empty:
            line = None
        if line is not None:
            cmd, _, rest = line.partition(" ")
            if line == "":
                left = [q for q in ids if not saved(q) and q not in sent_ids()]
                if left:
                    send(left[0])
                    print(f"{len(left) - 1} not sent yet")
                else:
                    print("nothing left to send.")
                show()
            elif cmd == "q":
                print("bye. Answers copied from now on are not saved until you run this again.")
                return
            elif cmd == "h":
                print(RUN_HELP)
            elif cmd == "s":
                cmd_status(args, bench)
            elif cmd == "r" and rest in ids:
                send(rest)
            elif cmd == "a" and rest in ids:
                raw = read_clipboard() or ""
                if raw.strip() and not is_prompt(raw) and (raw == unclaimed[0] or raw not in seen):
                    seen.add(raw)
                    unclaimed[0] = None
                    save_answer(rest, raw, args)
                    show()
                else:
                    print("the clipboard does not hold a new Gemini answer")
            else:
                print("unknown command. Type h for help.")
        clip = read_clipboard() or ""
        if clip in seen or not clip.strip():
            continue
        seen.add(clip)
        if is_prompt(clip) or not ('"statements"' in clip or '"positions"' in clip or '"answer"' in clip):
            continue
        found = set(ID_RE.findall(clip)) & set(ids)
        if len(found) == 1:
            if save_answer(found.pop(), clip, args):
                show()
        else:
            unclaimed[0] = clip
            print(f"\ncopied an answer but could not tell which question it is for. "
                  f"Type  a ID  to save it (waiting for: {', '.join(waiting()) or '-'})")


def cmd_status(args, bench):
    split, ids = load_ids(bench)
    done = [q for q in ids if saved(q)]
    sent = sent_ids()
    print(f"saved {len(done)}/{len(ids)}")
    left = [q for q in ids if not saved(q)]
    if left:
        print("not saved: " + ", ".join(f"{q}{' (sent)' if q in sent else ''}" for q in left))
    replaced = [r["id"] for r in read_log() if r["event"] == "replaced"]
    if replaced:
        print("replaced by hand (say so when reporting): " + ", ".join(replaced))


# --------------------------------------------------------------------------
# scoring


def answer_json(raw):
    """The text to score: the first ```json block if there is one, else the whole answer.

    Gemini writes a plain-text list of unsure statements after the block, which
    can contain braces, so only the block is passed to the solver.
    """
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    return m.group(1) if m else raw


def score_files(files, bench, split, out, meta, ids):
    records = []
    for qid in ids:
        f = files.get(qid)
        q = bench[qid]
        if f is None:
            raw = ""
        else:
            raw = f.read_text(encoding="utf-8")
        rec = score(answer_json(raw), q, bench, split)
        rec["output"] = answer_json(raw)
        rec["source_file"] = str(f.relative_to(HERE)) if f else None
        if f is None:
            rec["error"] = "no answer saved"
        records.append(rec)
        ok = sum(s["ok"] for s in rec["statements"].values())
        status = "valid" if rec["valid"] else ("invalid" if rec["parsed"] else "no JSON")
        print(f"  {qid}: {status:7} statements {ok}/{len(rec['statements'])}  "
              f"answer {rec['pred']} (gold {rec['gold']})"
              + (f"  {rec['error'][:90]}" if rec["error"] else ""))
    RESULTS.mkdir(exist_ok=True)
    json.dump({**meta, "records": records}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    s = summarize(records)
    st_valid = [x for r in records if r["valid"] for x in r["statements"].values()]
    in_valid = f"{sum(x['ok'] for x in st_valid)}/{len(st_valid)}"
    all_false = f"{sum(not x['gold'] for x in st_valid)}/{len(st_valid)}"
    print(f"\nvalid {pct(s['valid'])}; statements {pct(s['stmt'])} "
          f"(structure {s['structure']}, weak/str {s['weak/str']}); questions {pct(s['questions'])}")
    print(f"in valid translations: model {pct(in_valid)} vs calling every statement false {pct(all_false)}")
    print(f"unsupported statements in valid translations: {s['unsupported']}")
    print(f"written to {out.relative_to(HERE)}. For the full table: python rescore_aspic.py")


def cmd_score(args, bench):
    split, ids = load_ids(bench)
    files = {q: RAW / f"{q}.txt" for q in ids if saved(q)}
    missing = [q for q in ids if q not in files]
    if missing and not args.partial:
        sys.exit(f"no answer yet for {', '.join(missing)}. Use --partial to score anyway "
                 f"(missing answers count as wrong).")
    print(f"Gemini without rationale, one try ({len(files)}/{len(ids)} answers)")
    meta = {"model": args.teacher, "adapter": None, "shots": len(EXAMPLES), "condition": "no rationale, one try",
            "examples": EXAMPLES, "scored": now()}
    score_files(files, bench, split, RESULT_FILE, meta, ids)


def cmd_reference(args, bench):
    split, ids = load_ids(bench)
    files = {q: HERE / "gemini_raw" / f"{q}_try1.txt" for q in ids}
    files = {q: f for q, f in files.items() if f.exists()}
    print(f"Gemini WITH rationale, first try, from gemini_raw/ ({len(files)}/{len(ids)} answers)")
    print("caveat: questions re-run after the tab mix-up may not have their true first try in try1")
    meta = {"model": "gemini-flash-medium (teacher.py)", "adapter": None, "shots": 3,
            "condition": "with rationale, first try (teacher prompt)", "scored": now()}
    score_files(files, bench, split, REFERENCE_FILE, meta, ids)


# --------------------------------------------------------------------------
# --cell direct: the same Gemini answers the question itself, no ASPIC+, no solver
#
# This is the "frontier model, no solver" cell of the 2x2. Same 15 questions,
# same input text (question_text(), no (a)/(b)/(c) choice list), no rationale,
# no examples, one try. Like Trinav's baseline: (a)/(b)/(c) questions are
# judged statement by statement; five-option questions pick one option, and
# the option labels follow from that pick (the same way gold_labels() does).
# Results go to direct/results.json, NOT eval_results/, because
# rescore_aspic.py would try to read these answers as ASPIC+ translations.

DIRECT_MARK = "## Question to answer"
DIRECT_DIR = HERE / "direct"
DIRECT_RESULT = DIRECT_DIR / "results.json"

DIRECT_INTRO = """Answer one question from LEET, the Korean law school entrance exam (English translation).

TASK

Use only the passage and the ordinary background knowledge needed to read it. Think it through before you answer.

Output: one JSON code block with exactly this shape:

SHAPE

Nothing is needed after the code block.

""" + DIRECT_MARK + """

#### Question QID

Question text:

"""

TASK_COMBO = ("For each of the statements KEYS, decide whether the statement itself is correct (true) or not "
              "(false). Judge each statement itself, even if the question asks which statements are "
              "incorrect. The list of answer choices was left out on purpose: do not answer with a choice number.")
TASK_SINGLE = "Choose the one option (1 to 5) that answers the question."


def build_direct_prompt(qid, bench):
    q = bench[qid]
    mode, _ = parse_choices(q["original_question"])
    if mode == "combo":
        keys = list(gold_labels(q))
        task = TASK_COMBO.replace("KEYS", ", ".join(f"({k})" for k in keys))
        # neutral placeholders, so the shape does not suggest an answer
        shape = json.dumps({"id": qid, "statements": {k: "TF" for k in keys},
                            "reasons": {k: "one sentence" for k in keys}}, indent=1)
        shape = shape.replace('"TF"', "<true or false>")
    else:
        task = TASK_SINGLE
        shape = json.dumps({"id": qid, "answer": "N", "reason": "one or two sentences"}, indent=1)
        shape = shape.replace('"N"', "<a number from 1 to 5>")
    return (DIRECT_INTRO.replace("TASK", task).replace("SHAPE", "```json\n" + shape + "\n```")
            .replace("QID", qid) + question_text(q) + "\n")


def to_bool(v):
    if isinstance(v, dict):
        v = v.get("verdict", v.get("value"))
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def score_direct(raw, q):
    from eval_aspic import statement_types
    qid = q["id"]
    text = q["original_question"]
    mode, choices = parse_choices(text)
    polarity = asks_for(text)
    gold = gold_labels(q)
    rec = {"id": qid, "parsed": False, "valid": False, "pred": None, "gold": gold_number(q["answer"]),
           "statements": {}, "error": "", "mode": mode}
    truth = {}
    try:
        s = answer_json(raw)
        d = json.loads(s[s.find("{"): s.rfind("}") + 1])
        rec["parsed"] = True
        if mode == "combo":
            st = d.get("statements") or {}
            st = {str(k).strip("() ").lower(): v for k, v in st.items()}
            truth = {k: to_bool(st.get(k)) for k in gold}
            missing = [k for k, v in truth.items() if v is None]
            if missing:
                # like an invalid translation in the solver cell: every statement counts as wrong
                rec["error"] = f"no true/false for {', '.join(missing)}"
                truth = {}
            else:
                rec["valid"] = True
                rec["pred"] = predict(truth, mode, choices, polarity)
        else:
            picked = gold_number(d.get("answer", ""))  # accepts 3, "3" and "③"
            if picked not in range(1, 6):
                rec["error"] = f"no option 1-5 in answer: {d.get('answer')!r}"
            else:
                want = polarity == "correct"
                truth = {k: (int(k) == picked) == want for k in gold}
                rec["valid"] = True
                rec["pred"] = picked
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    types = statement_types(None, qid)
    for k, g in gold.items():
        p = truth.get(k)
        rec["statements"][k] = {"gold": g, "pred": p, "ok": p == g, "type": types.get(k, "structure")}
    rec["question_ok"] = rec["valid"] and rec["pred"] == rec["gold"]
    return rec


def cmd_score_direct(args, bench):
    split, ids = load_ids(bench)
    missing = [q for q in ids if not saved(q)]
    if missing and not args.partial:
        sys.exit(f"no answer yet for {', '.join(missing)}. Use --partial to score anyway.")
    print(f"Gemini answering directly, no solver ({len(ids) - len(missing)}/{len(ids)} answers)")
    records = []
    for qid in ids:
        raw = (RAW / f"{qid}.txt").read_text(encoding="utf-8") if saved(qid) else ""
        rec = score_direct(raw, bench[qid])
        rec["output"] = answer_json(raw)
        records.append(rec)
        ok = sum(s["ok"] for s in rec["statements"].values())
        print(f"  {qid}: {'valid' if rec['valid'] else 'INVALID':7} statements {ok}/{len(rec['statements'])}  "
              f"answer {rec['pred']} (gold {rec['gold']})" + (f"  {rec['error'][:90]}" if rec["error"] else ""))
    DIRECT_DIR.mkdir(exist_ok=True)
    json.dump({"model": args.teacher, "condition": "direct answer, no solver, no rationale, one try",
               "scored": now(), "records": records},
              open(DIRECT_RESULT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"written to {DIRECT_RESULT.relative_to(HERE)}\n")
    compare(ids)


def compare(ids):
    """The 2x2 frontier half on the same questions, with and without 2022_36 (DOT graphs)."""
    rows = [("Gemini direct, no solver", DIRECT_RESULT),
            ("Gemini + solver, no rationale", RESULT_FILE),
            ("Gemini + solver, with rationale (try 1)", REFERENCE_FILE)]
    loaded = {}
    for name, f in rows:
        if f.exists():
            loaded[name] = {r["id"]: r for r in json.load(open(f, encoding="utf-8"))["records"]}
    if not loaded:
        print("nothing to compare yet")
        return
    any_recs = next(iter(loaded.values()))
    sets = [("all 15", ids), ("without 2022_36", [i for i in ids if i != "2022_36"])]
    lines = ["| condition | questions | statements | statements w/o 2022_36 | questions w/o 2022_36 |",
             "|---|---|---|---|---|"]
    for name, recs in loaded.items():
        cells = []
        for _, qs in sets:
            st = [s for q in qs for s in recs[q]["statements"].values()]
            cells.append((f"{sum(s['ok'] for s in st)}/{len(st)}", f"{sum(recs[q]['question_ok'] for q in qs)}/{len(qs)}"))
        lines.append(f"| {name} | {pct(cells[0][1])} | {pct(cells[0][0])} | {pct(cells[1][0])} | {pct(cells[1][1])} |")
    ref = []
    for _, qs in sets:
        st = [s for q in qs for s in any_recs[q]["statements"].values()]
        ref.append(f"{sum(not s['gold'] for s in st)}/{len(st)}")
    lines.append(f"| reference: call every statement false | | {pct(ref[0])} | {pct(ref[1])} | |")
    print("\n".join(lines))
    a, b = "Gemini direct, no solver", "Gemini + solver, no rationale"
    if a in loaded and b in loaded:
        both = only_a = only_b = neither = 0
        for q in ids:
            for k, s in loaded[a][q]["statements"].items():
                x, y = s["ok"], loaded[b][q]["statements"][k]["ok"]
                both += x and y
                only_a += x and not y
                only_b += y and not x
                neither += not x and not y
        print(f"\nsame statements, paired: both right {both}, only direct right {only_a}, "
              f"only solver right {only_b}, both wrong {neither}")
        print("the solver adds value only if 'only solver right' is clearly larger than 'only direct right'")
    (DIRECT_DIR / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--teacher", default="gemini-flash-3.8-medium",
                    help="model and setting, written to the log and the result file")
    ap.add_argument("--cell", choices=["solver", "direct"], default="solver",
                    help="solver: Gemini writes ASPIC+ and the solver answers (default). "
                         "direct: Gemini answers the question itself, no solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts")
    sub.add_parser("run")
    p = sub.add_parser("next")
    p.add_argument("id", nargs="?")
    p.add_argument("--file", action="store_true", help="print the prompt path instead of copying it")
    p = sub.add_parser("save")
    p.add_argument("id")
    p.add_argument("--file", help="read the answer from this file instead of the clipboard")
    p.add_argument("--replace", action="store_true", help="replace a saved answer (only for a paste mistake)")
    sub.add_parser("status")
    p = sub.add_parser("score")
    p.add_argument("--partial", action="store_true")
    sub.add_parser("reference")
    sub.add_parser("compare")
    args = ap.parse_args()
    bench = load_bench(args.bench)
    if args.cell == "direct":
        if args.cmd == "reference":
            sys.exit("reference is for the solver cell only")
        g = globals()
        g.update(OUT=DIRECT_DIR, PROMPTS=DIRECT_DIR / "prompts", RAW=DIRECT_DIR / "raw",
                 LOG=DIRECT_DIR / "log.csv", build_prompt=build_direct_prompt)
    score_cmd = cmd_score_direct if args.cell == "direct" else cmd_score
    {"prompts": cmd_prompts, "run": cmd_run, "next": cmd_next, "save": cmd_save, "status": cmd_status,
     "score": score_cmd, "reference": cmd_reference,
     "compare": lambda a, b: compare(load_ids(b)[1])}[args.cmd](args, bench)


if __name__ == "__main__":
    main()
