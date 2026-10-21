#!/usr/bin/env python3
"""Gemini teacher loop: one question at a time, with copy and paste.

Each training translation is drafted by Gemini (one model, one setting for the
whole set), checked by the solver, and kept only if it reproduces the gold
answer. This script does the bookkeeping around the copy and paste.

Commands (run from fine_tuning/jun/aspic/ with the venv active):

  python teacher.py prompts            build prompts/<id>.md for every train and
                                       validation question (test set excluded)
  python teacher.py run                for several Gemini tabs at once (macOS):
                                       Enter copies the next prompt; every Gemini
                                       answer you copy is found by its "id",
                                       checked and saved without a command, and
                                       feedback is copied for you to paste back
  python teacher.py next [ID]          copy the next question's prompt (one not
                                       sent yet), or ID's prompt, to the clipboard
  python teacher.py save ID            read Gemini's whole answer from the clipboard,
                                       check it, save it if it matches; if not, copy
                                       a feedback message for Gemini to the clipboard
  python teacher.py drop ID "reason"   give up on a question (logged)
  python teacher.py status             what is done, dropped and left

Without a clipboard (not on a Mac), use --file: `save ID --file answer.txt`
reads the answer from a file, and `next` prints the prompt's path instead.

Files it writes:
  prompts/<id>.md          the full prompt to paste into a new Gemini chat
  gemini_raw/<id>_tryN.txt Gemini's whole answer for each try, unchanged
  translations/<id>.json   the translation, only when it matches the gold answer
  teacher_log.csv          one line per try, drop and setting
  teacher_sent.json        which prompts were handed out, so none is sent twice

A question gets at most 3 tries (the first answer plus 2 corrections). After
the 3rd failed try it is dropped automatically.
"""
import argparse
import csv
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from check_aspic import FINAL_TEST_IDS, check_translation, load_bench
from test_base_model import gold_labels

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = REPO / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
SPLIT = HERE.parent / "data" / "split.json"
PROMPTS = HERE / "prompts"
RAW = HERE / "gemini_raw"
TRANSLATIONS = HERE / "translations"
LOG = HERE / "teacher_log.csv"
SENT = HERE / "teacher_sent.json"  # questions whose prompt was copied and that wait for an answer
EXAMPLES_DIR = HERE / "translations_claude"
# Worked examples shown in every prompt. A question never sees itself as an
# example: the first three in this list that are not the question are used.
EXAMPLE_POOL = ["2022_17", "2021_37", "2024_15", "2022_29", "2023_26"]
MAX_TRIES = 3
LOG_FIELDS = ["time", "id", "try", "result", "truth", "gold", "detail", "teacher"]

INTRO = """I need you to translate one LEET argument question into a JSON format based on ASPIC+ argument structure. This message has three parts: (1) the format, (2) worked examples with finished translations, and (3) the question to translate, with its expert rationale.

Translate the question in Part 3 into the format in Part 1, in the same style as the examples in Part 2.

Rules:
1. Build the structure from the passage. Use the expert rationale to understand the reasoning. Anything you take from the rationale gets "source": "rationale", and general knowledge stated nowhere gets "source": "background".
2. Do not write verdicts anywhere: no "correct", "incorrect", "true", "false", and no answer number. Whether a statement is true must follow only from the structure and its check.
3. When the checks are evaluated as described in Part 1, each check must give the same result as the expert rationale. Do not add a rule, contrary or premise only to make a check come out right. Every element must be justified by the passage or the rationale.
4. If a statement is false because a tempting inference fails, write that inference and the reason it fails (often an undercut), as in statement (c) of 2021_37 if that example is included.
5. Keep claim text close to the passage wording. Do not invent speakers or facts. Use the position labels the passage uses.
6. Single-choice questions (the choices are the statements themselves) use the keys "1" to "5" in "statements".
7. If a statement cannot be expressed with the check operations in Part 1, use {"op": "unsupported", "reason": "..."} instead of forcing it.
8. You may add a short "note" to a statement to explain a choice that is not obvious.
9. Before answering, check that every claim ID, rule ID, position label and hypothetical name you reference is defined.

Output: one JSON code block containing one object with "id": "QID". After the code block, list in plain text every statement you were unsure about, and why.
"""

OUTRO = """
---

Now translate question QID from Part 3. Follow rules 1 to 9 above. Output one JSON code block containing one object with "id": "QID", then list the statements you were unsure about.
"""


# --------------------------------------------------------------------------
# helpers


def rationale_text(q):
    r = q.get("original_rationale")
    if isinstance(r, dict):
        return "\n".join(f"- {k}: {v}" for k, v in r.items())
    return str(r or "")


def question_block(q):
    return (f"#### Question {q['id']}\n\nQuestion text:\n\n{q['original_question']}\n\n"
            f"Expert rationale:\n\n{rationale_text(q)}\n")


def format_spec():
    spec = (HERE / "ASPIC_FORMAT.md").read_text(encoding="utf-8").split("\n", 1)[1]
    # Drop the frozen training instruction: it is for the student, not the teacher.
    spec = spec.split("## Frozen instruction")[0].rstrip()
    return "\n".join("#" + line if line.startswith("##") else line for line in spec.splitlines())


def build_prompt(qid, bench):
    examples = [e for e in EXAMPLE_POOL if e != qid][:3]
    parts = [INTRO.replace("QID", qid), "## Part 1. Format\n", format_spec(), "\n## Part 2. Worked examples\n"]
    for ex in examples:
        t = json.load(open(EXAMPLES_DIR / f"{ex}.json", encoding="utf-8"))
        parts.append(question_block(bench[ex]))
        parts.append("Translation:\n\n```json\n" + json.dumps(t, ensure_ascii=False, indent=1) + "\n```\n")
    parts.append("\n## Part 3. Question to translate\n")
    parts.append(question_block(bench[qid]))
    parts.append(OUTRO.replace("QID", qid))
    return "\n".join(parts)


def load_split():
    split = json.load(open(SPLIT))
    order = sorted(split["train"]) + sorted(split["valid"])
    for i in order:
        if i in FINAL_TEST_IDS:
            sys.exit(f"{i} is a final test question but it is in {SPLIT}. Stop and check the split.")
    return split, order


def read_log():
    if not LOG.exists():
        return []
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_log(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def state(log):
    """id -> (status, tries). status: MATCH, DROPPED, or IN PROGRESS."""
    out = {}
    for r in log:
        st, tries = out.get(r["id"], (None, 0))
        if r["result"] == "DROPPED":
            out[r["id"]] = ("DROPPED", tries)
        elif r["result"] == "MATCH":
            out[r["id"]] = ("MATCH", tries + 1)
        else:
            out[r["id"]] = (st or "IN PROGRESS", tries + 1)
    return out


def load_sent():
    try:
        return json.load(open(SENT, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_sent(qid):
    sent = load_sent()
    sent[qid] = dt.datetime.now().isoformat(timespec="seconds")
    json.dump(sent, open(SENT, "w", encoding="utf-8"), indent=1)


def waiting(order):
    """Questions sent to Gemini that are not finished yet, in split order."""
    st, sent = state(read_log()), load_sent()
    return [q for q in order if q in sent and st.get(q, ("",))[0] not in ("MATCH", "DROPPED")]


def next_unsent(order):
    st, sent = state(read_log()), load_sent()
    left = [q for q in order if st.get(q, ("",))[0] not in ("MATCH", "DROPPED") and q not in sent]
    return left[0] if left else None, len(left)


def copy_to_clipboard(text):
    if shutil.which("pbcopy"):
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        return True
    return False


def read_clipboard():
    if shutil.which("pbpaste"):
        return subprocess.run(["pbpaste"], capture_output=True, check=True).stdout.decode("utf-8")
    return None


def extract_json(raw):
    """The JSON from Gemini's answer: the first ```json block, else the outermost {...} or [...]."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    s = m.group(1) if m else raw
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object in the answer")
    start = min(starts)
    end = max(s.rfind("}"), s.rfind("]"))
    data = json.loads(s[start:end + 1])
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"expected one translation, got a list of {len(data)}")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("the JSON is not an object")
    return data


def label(k):
    return f"({k})" if k.isalpha() else f"option {k}"


def tf(v):
    return "true" if v else ("unsupported" if v is None else "false")


def feedback(qid, rec, gold):
    """The message to paste back into the same Gemini chat."""
    tail = (" Keep rules 1 to 9 from my first message: every element must be justified by the passage "
            "or the rationale, and do not add anything only to make a check pass. "
            "Output the full corrected translation in one JSON code block with \"id\": \"" + qid + "\".")
    if rec.get("errors"):
        return ("The solver could not read your translation: " + "; ".join(rec["errors"]) +
                ". Please fix this." + tail)
    truth = rec["truth"]
    got = ", ".join(f"{label(k)} {tf(v)}" for k, v in truth.items())
    wrong = [k for k in gold if truth.get(k) != gold[k]]
    if not wrong:  # every statement is right but the pattern did not map to a choice
        return (f"Checking your translation with the solver gives: {got}. These do not map to any of the "
                f"choices. Please check the statement keys and the question wording." + tail)
    should = "; ".join(f"the expert rationale says {label(k)} is {tf(gold[k])}" for k in wrong)
    msg = f"Checking your translation with the solver gives: {got}. But {should}. "
    unsup = [k for k in wrong if truth.get(k) is None]
    if unsup:
        msg += ("If an unsupported statement can be expressed with the check operations in Part 1, "
                "write the check; if it truly cannot, say so and explain why. ")
    msg += ("Please find what in the structure or the check makes these come out differently from the "
            "rationale, and fix only that.")
    return msg + tail


# --------------------------------------------------------------------------
# commands


def cmd_prompts(args, bench):
    split, order = load_split()
    PROMPTS.mkdir(exist_ok=True)
    for qid in order:
        (PROMPTS / f"{qid}.md").write_text(build_prompt(qid, bench), encoding="utf-8")
    sizes = [len((PROMPTS / f"{q}.md").read_text(encoding="utf-8")) for q in order]
    print(f"wrote {len(order)} prompts to {PROMPTS}/ ({len(split['train'])} train, {len(split['valid'])} validation); "
          f"{min(sizes) // 1000}-{max(sizes) // 1000}k characters each")
    stray = sorted(p.stem for p in TRANSLATIONS.glob("*.json")
                   if state(read_log()).get(p.stem, ("",))[0] != "MATCH")
    if stray:
        print(f"\nwarning: translations/ has files that this loop did not make: {', '.join(stray)}")
        print("They were made with another setting. Move them out before training (see the guide).")


def cmd_next(args, bench):
    split, order = load_split()
    if args.id:
        qid, left = args.id, None
        if qid not in order:
            sys.exit(f"{qid} is not in the train or validation split")
    else:
        qid, left = next_unsent(order)
        if not qid:
            w = waiting(order)
            print("every question is done, dropped or already sent."
                  + (f" Waiting for answers: {', '.join(w)}" if w else ""))
            return
    p = PROMPTS / f"{qid}.md"
    if not p.exists():
        sys.exit("prompts are missing. Run: python teacher.py prompts")
    which = "validation" if qid in split["valid"] else "train"
    print(f"next: {qid} ({which}" + (f", {left} not sent yet including this one)" if left else ")"))
    tries = state(read_log()).get(qid, ("", 0))[1]
    if tries:
        print(f"note: {qid} already has {tries} failed tr{'y' if tries == 1 else 'ies'}; the try count continues.")
    mark_sent(qid)
    if not args.file and copy_to_clipboard(p.read_text(encoding="utf-8")):
        print("prompt copied to the clipboard: paste it into a NEW Gemini chat")
    else:
        print(f"prompt: {p}")
    print(f"then copy Gemini's whole answer and run: python teacher.py save {qid}")


def evaluate_answer(qid, raw, args, bench, split):
    """Check one Gemini answer for qid, log it, and save it if it matches.

    Returns (result, n, feedback message or None, dropped). Prints the details.
    """
    st = state(read_log())
    tries = st.get(qid, ("", 0))[1]
    n = tries + 1
    RAW.mkdir(exist_ok=True)
    (RAW / f"{qid}_try{n}.txt").write_text(raw, encoding="utf-8")
    gold = gold_labels(bench[qid])
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "id": qid, "try": n,
           "gold": " ".join(f"{k}={'T' if v else 'F'}" for k, v in gold.items()), "teacher": args.teacher}

    try:
        t = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        rec = {"errors": [f"could not read the JSON ({e})"]}
        msg = ("I could not read the JSON in your answer. Please output the full translation again in one "
               "JSON code block, with nothing else inside the block.")
        return log_failure(args, qid, n, row, rec, msg, "ERROR")
    if t.get("id") not in (None, qid):
        print(f"warning: Gemini wrote id {t.get('id')!r}; using {qid}")
    t = {"id": qid, **{k: v for k, v in t.items() if k != "id"}}

    rec = check_translation(t, bench, split, FINAL_TEST_IDS, use_pyarg=args.pyarg)
    warnings = [w for w in rec.get("warnings", []) if not w.startswith("validation question")]
    for w in warnings:
        print(f"warning: {w}")
    if rec.get("match"):
        TRANSLATIONS.mkdir(exist_ok=True)
        (TRANSLATIONS / f"{qid}.json").write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
        row.update(result="MATCH", truth=" ".join(f"{k}={'T' if v else 'F'}" for k, v in rec["truth"].items()))
        append_log(row)
        print(f"{qid} try {n}: MATCH, saved to translations/{qid}.json")
        if warnings:
            print("read this one against its passage: the warnings above mean a 'passage' claim may not be in the text")
        return "MATCH", n, None, False
    result = "ERROR" if rec.get("errors") else ("UNRESOLVED" if rec.get("pred") is None else "WRONG")
    return log_failure(args, qid, n, row, rec, feedback(qid, rec, gold), result)


def log_failure(args, qid, n, row, rec, msg, result):
    truth = rec.get("truth") or {}
    row.update(result=result, truth=" ".join(f"{k}={'T' if v else ('?' if v is None else 'F')}" for k, v in truth.items()),
               detail="; ".join(rec.get("errors", []))[:200])
    append_log(row)
    print(f"{qid} try {n}: {result}")
    if rec.get("errors"):
        print("  " + "; ".join(rec["errors"]))
    if truth:
        print(f"  solver: {row['truth']}")
        print(f"  gold:   {row['gold']}")
    if n >= MAX_TRIES:
        append_log({"time": dt.datetime.now().isoformat(timespec="seconds"), "id": qid, "try": n,
                    "result": "DROPPED", "detail": f"{MAX_TRIES} tries failed", "teacher": args.teacher})
        print(f"{qid} has failed {MAX_TRIES} tries and is dropped. You can close its Gemini tab.")
        return result, n, None, True
    return result, n, f"[{qid}] " + msg, False


def check_saveable(qid, order):
    """None if answers for qid can be saved, else the reason why not."""
    if qid in FINAL_TEST_IDS:
        return f"{qid} is a final test question. It must never be translated or trained on."
    if qid not in order:
        return f"{qid} is not in the train or validation split"
    status = state(read_log()).get(qid, ("",))[0]
    if status in ("MATCH", "DROPPED"):
        return f"{qid} is already {status}. Nothing saved."
    return None


def is_prompt(text):
    return "Part 1. Format" in text and "Part 3. Question to translate" in text


def cmd_save(args, bench):
    qid = args.id
    split, order = load_split()
    why = check_saveable(qid, order)
    if why:
        sys.exit(why)
    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
    else:
        raw = read_clipboard()
        if raw is None:
            sys.exit("no clipboard on this machine; use --file")
    if not raw.strip():
        sys.exit("the clipboard is empty: copy Gemini's whole answer first")
    if is_prompt(raw):
        sys.exit("the clipboard still holds the prompt, not Gemini's answer. Copy the answer first.")
    result, n, msg, dropped = evaluate_answer(qid, raw, args, bench, split)
    if result == "MATCH" or dropped:
        print("next: python teacher.py next")
        return
    print(f"\nfeedback for Gemini (try {n + 1} of {MAX_TRIES}):\n\n{msg}\n")
    if not args.file and copy_to_clipboard(msg):
        print("feedback copied to the clipboard: paste it into the SAME Gemini chat, copy the new answer, then run:")
    else:
        print("paste the feedback above into the SAME Gemini chat, copy the new answer, then run:")
    print(f"python teacher.py save {qid}")


ID_RE = re.compile(r'"id"\s*:\s*"(\d{4}_\d{2})"')
RUN_HELP = """  Enter        copy the next question's prompt (paste it into a NEW Gemini tab)
  r ID         copy ID's prompt again (if you lost its tab; the try count continues)
  a ID         save the answer now on the clipboard as ID (if its id was not found)
  d ID reason  drop ID
  s            status          h  this help          q  quit"""


def cmd_run(args, bench):
    """One terminal for many Gemini tabs: Enter hands out prompts, copied answers are saved by themselves."""
    import queue
    import threading
    import time
    if not shutil.which("pbpaste"):
        sys.exit("run needs the macOS clipboard (pbcopy/pbpaste). Use next and save --file instead.")
    split, order = load_split()
    if not all((PROMPTS / f"{q}.md").exists() for q in order):
        sys.exit("prompts are missing. Run: python teacher.py prompts")

    seen = set()  # clipboard texts already handled, so nothing is processed twice
    unclaimed = [None]  # an answer whose question could not be found, for "a ID"

    def put(text):
        seen.add(text)
        copy_to_clipboard(text)

    def show_waiting():
        w = waiting(order)
        print(f"waiting for: {', '.join(w) if w else '-'}")

    def handle(qid, raw):
        why = check_saveable(qid, order)
        if why:
            print(why)
            return
        print(f"\n--- answer for {qid}")
        result, n, msg, dropped = evaluate_answer(qid, raw, args, bench, split)
        if msg:
            put(msg)
            print(f"feedback copied (try {n + 1} of {MAX_TRIES}): paste it into the tab for {qid}")
        show_waiting()

    def send(qid):
        put((PROMPTS / f"{qid}.md").read_text(encoding="utf-8"))
        mark_sent(qid)
        which = "validation" if qid in split["valid"] else "train"
        print(f"prompt for {qid} ({which}) copied: paste it into a NEW Gemini tab")

    lines = queue.Queue()

    def reader():
        while True:
            try:
                lines.put(input())
            except EOFError:
                lines.put("q")
                return

    first = read_clipboard() or ""
    seen.add(first)
    print("teacher run: one terminal for several Gemini tabs. Copy each finished answer with Gemini's")
    print("copy button; it is checked and saved here without a command.\n" + RUN_HELP)
    show_waiting()
    threading.Thread(target=reader, daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=0.7).strip()
        except queue.Empty:
            line = None
        if line is not None:
            cmd, _, rest = line.partition(" ")
            if line == "":
                qid, left = next_unsent(order)
                if qid:
                    send(qid)
                    print(f"{left - 1} not sent yet")
                else:
                    print("nothing left to send.")
                show_waiting()
            elif cmd == "q":
                print("bye. Answers you copy from now on are not saved until you run this again.")
                return
            elif cmd == "h":
                print(RUN_HELP)
            elif cmd == "s":
                cmd_status(args, bench)
                show_waiting()
            elif cmd == "r" and rest in order:
                send(rest)
            elif cmd == "a" and rest in order:
                raw = read_clipboard() or ""
                if raw.strip() and not is_prompt(raw) and (raw == unclaimed[0] or raw not in seen):
                    seen.add(raw)
                    unclaimed[0] = None
                    handle(rest, raw)
                else:
                    print("the clipboard does not hold a new Gemini answer")
            elif cmd == "d" and rest:
                qid, _, reason = rest.partition(" ")
                if qid in order:
                    append_log({"time": dt.datetime.now().isoformat(timespec="seconds"), "id": qid, "try": "",
                                "result": "DROPPED", "detail": reason or "dropped by hand", "teacher": args.teacher})
                    print(f"{qid} dropped")
                    show_waiting()
                else:
                    print(f"unknown question {qid}")
            else:
                print("unknown command. Type h for help.")

        clip = read_clipboard() or ""
        if clip in seen or not clip.strip():
            continue
        seen.add(clip)
        if is_prompt(clip):
            continue
        ids = set(ID_RE.findall(clip))
        looks_like_answer = '"statements"' in clip or '"positions"' in clip
        if not looks_like_answer:
            continue  # something else was copied
        if len(ids) == 1:
            handle(ids.pop(), clip)
        else:
            w = waiting(order)
            unclaimed[0] = clip
            print(f"\ncopied an answer but could not tell which question it is for. "
                  f"Type  a ID  to save it (waiting for: {', '.join(w) or '-'})")


def cmd_drop(args, bench):
    append_log({"time": dt.datetime.now().isoformat(timespec="seconds"), "id": args.id, "try": "",
                "result": "DROPPED", "detail": args.reason, "teacher": args.teacher})
    print(f"{args.id} dropped: {args.reason}")


def cmd_status(args, bench):
    split, order = load_split()
    st = state(read_log())
    for name in ("train", "valid"):
        ids = sorted(split[name])
        done = [q for q in ids if st.get(q, ("",))[0] == "MATCH"]
        dropped = [q for q in ids if st.get(q, ("",))[0] == "DROPPED"]
        busy = [q for q in ids if st.get(q, ("",))[0] == "IN PROGRESS"]
        left = len(ids) - len(done) - len(dropped) - len(busy)
        print(f"{name:6} {len(done):3} matched, {len(dropped):2} dropped, {len(busy):2} in progress, "
              f"{left:3} not started (of {len(ids)})")
        if dropped:
            print(f"       dropped: {', '.join(dropped)}")
    log = read_log()
    firsts = [r for r in log if r["try"] == "1"]
    if firsts:
        n1 = sum(r["result"] == "MATCH" for r in firsts)
        print(f"first-try match rate: {n1}/{len(firsts)}")
    files = {p.stem for p in TRANSLATIONS.glob("*.json")}
    stray = sorted(f for f in files if st.get(f, ("",))[0] != "MATCH")
    if stray:
        print(f"warning: translations/ has files this loop did not make: {', '.join(stray)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--teacher", default="gemini-flash-medium",
                    help="model and setting, written to the log (default: gemini-flash-medium)")
    ap.add_argument("--pyarg", action="store_true", help="cross-check with PyArg (slower, same result)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts")
    p = sub.add_parser("next")
    p.add_argument("id", nargs="?", help="a specific question (default: the next one not sent yet)")
    p.add_argument("--file", action="store_true", help="print the prompt path instead of copying it")
    sub.add_parser("run", help="one terminal for several Gemini tabs (macOS)")
    p = sub.add_parser("save")
    p.add_argument("id")
    p.add_argument("--file", help="read Gemini's answer from this file instead of the clipboard")
    p = sub.add_parser("drop")
    p.add_argument("id")
    p.add_argument("reason")
    sub.add_parser("status")
    args = ap.parse_args()
    bench = load_bench(args.bench)
    {"prompts": cmd_prompts, "next": cmd_next, "save": cmd_save, "run": cmd_run,
     "drop": cmd_drop, "status": cmd_status}[args.cmd](args, bench)


if __name__ == "__main__":
    main()
