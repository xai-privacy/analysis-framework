"""Shared helpers for turning LEET-Arg records into fine-tuning examples.

Used by prepare_leet_arg.py (builds data/train.jsonl and data/valid.jsonl) and
evaluate_leet_arg.py (runs the model on the held-out split and scores it), so
training and evaluation use exactly the same prompt and the same gold labels.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def _find_repo_root(start):
    """The repo root is the first parent folder that has benchmarks/ in it,
    so this file works wherever it sits inside the repo."""
    for p in [start, *start.parents]:
        if (p / "benchmarks").is_dir():
            return p
    raise FileNotFoundError("could not find the repo root (no benchmarks/ folder above this file)")


REPO = _find_repo_root(HERE)
DATASET_PATH = REPO / "benchmarks" / "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json"
FINAL_TEST_PATH = REPO / "testing" / "final_testing" / "test_set" / "LEET_Arg_Questions_Test_Set.json"

SYSTEM_PROMPT = (
    "You are an expert legal reasoning auditor. Analyze the debate provided, "
    "extract the core premises and inference rules for each statement, and "
    "declare a final verdict (CORRECT or INCORRECT)."
)

CIRCLED = "①②③④⑤"

# Stems that ask for the statement that is NOT correct. For these, the chosen
# option is the incorrect statement.
NEGATIVE_STEM = re.compile(r"\bNOT\b|cannot be included", re.I)


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_final_test_ids():
    with open(FINAL_TEST_PATH, encoding="utf-8") as f:
        return {q["id"] for q in json.load(f)}


# ---------------------------------------------------------------- question text

def split_question(q):
    """Return (passage, stem, choices_text) from original_question.

    The tag "<statements>" also appears inside many stems ("Choose all that
    apply from <statements>."), so the statement list is located by finding
    the first statement's own text, not by the tag.
    """
    text = q["original_question"]
    _, _, choices = text.partition("<choices>")
    first = next(iter(q["statements"].values())).strip()
    # Search after <question>: a few records also quote the statements earlier.
    cut = text.find(first[:30], max(text.find("<question>"), 0))
    head = text[:cut] if cut > 0 else text.split("<choices>")[0]
    head = re.sub(r"(\s*<(statements|choices)>\s*)+$", "", head.rstrip())
    if "<question>" in head:
        passage, _, stem = head.rpartition("<question>")
    else:
        # No <question> tag: the stem is the last sentence ending in "?".
        end = head.rfind("?")
        start = max(head.rfind(".", 0, end), head.rfind(">", 0, end)) + 1
        passage, stem = head[:start], head[start:]
    # Diagram questions say "<choices> (in DOT)"; the options are the statements.
    stem = stem.replace("<choices>", "Options")
    return passage.strip(), stem.strip(), choices.strip()


def statement_label(text):
    """'(a) ...' -> '(a)', '① ...' -> '①', '(1) ...' -> '(1)'."""
    m = re.match(r"\s*(\(\w\)|[①②③④⑤])", text)
    return m.group(1) if m else None


def add_speaker_newlines(passage):
    # The dataset joins turns without a break ("...neutral.Bob: While...").
    # Put each speaker turn on its own line, as in the README example.
    return re.sub(r"(?<=[.!?”\"’)])\s*(?=[A-Z][a-z]+\d?:\s)", "\n", passage)


def build_user_prompt(q):
    passage, stem, _ = split_question(q)
    statements = "\n".join(s.strip() for s in q["statements"].values())
    return f"{add_speaker_newlines(passage)}\n\n<question> {stem}\n<statements>\n{statements}"


# ----------------------------------------------------------------- gold labels

def parse_choices(q):
    """{1: '(a), (c)', ...} for combination format, {1: 'Someone who ...'} otherwise."""
    _, _, choices = split_question(q)
    parts = re.split(r"([①②③④⑤])", choices)
    return {CIRCLED.index(parts[i]) + 1: parts[i + 1].strip().rstrip(".").strip()
            for i in range(1, len(parts), 2)}


def is_combination(q):
    labels = [statement_label(s) for s in q["statements"].values()]
    if not all(l and l.startswith("(") for l in labels):
        return False
    return all(re.fullmatch(r"(\(\w\)[,\s]*(and\s*)?)+", v) for v in parse_choices(q).values())


def is_negative(q):
    _, stem, _ = split_question(q)
    return bool(NEGATIVE_STEM.search(stem))


def gold_verdicts(q):
    """Per-statement truth (True = CORRECT analysis), derived from answer,
    choice map, and stem polarity. Keys are statement labels in order."""
    labels = [statement_label(s) for s in q["statements"].values()]
    answer = int(q["answer"])
    neg = is_negative(q)
    if is_combination(q):
        selected = set(re.findall(r"\(\w\)", parse_choices(q)[answer]))
        picked = [l in selected for l in labels]
    else:
        # Single-answer format: statement i is option i.
        picked = [i + 1 == answer for i in range(len(labels))]
    return {l: (p != neg) for l, p in zip(labels, picked)}


def answer_from_verdicts(q, verdicts):
    """Map predicted per-statement verdicts back to a choice number (1-5),
    or None if no option matches."""
    labels = [statement_label(s) for s in q["statements"].values()]
    neg = is_negative(q)
    if is_combination(q):
        want = {l for l in labels if verdicts.get(l) is (not neg)}
        for num, opt in parse_choices(q).items():
            if set(re.findall(r"\(\w\)", opt)) == want:
                return num
        return None
    hits = [i + 1 for i, l in enumerate(labels) if verdicts.get(l) is (not neg)]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------- target answer

# A sentence that names the final choice ("...so the correct answer is ②.").
# The model never sees the <choices> list, so these sentences are removed.
ANSWER_SENTENCE = re.compile(r"[^.]*(\banswer is\b[^.]*[①②③④⑤]|<choices>)[^.]*\.?", re.I)


def clean_rationale(text, preliminaries=False):
    if preliminaries:
        text = re.split(r"<choices>", text, flags=re.I)[0]  # "... <Choices> Analysis" residue
    text = ANSWER_SENTENCE.sub("", text)
    text = re.sub(r"\s*\b(Thus|Therefore|So),?\s*$", "", text.strip())  # dangling connective
    return re.sub(r"\s+", " ", text).strip()


def has_statement_rationales(q):
    r = q.get("original_rationale")
    if not isinstance(r, dict):
        return False
    keys = [f"statement_{i}_rationale" for i in range(1, len(q["statements"]) + 1)]
    return all((r.get(k) or "").strip() for k in keys)


def build_target(q):
    r = q["original_rationale"]
    verdicts = gold_verdicts(q)
    parts = []
    prelim = clean_rationale(r.get("preliminaries") or "", preliminaries=True)
    if prelim:
        parts.append(f"Summary: {prelim}")
    for i, label in enumerate(verdicts, 1):
        reasoning = clean_rationale(r[f"statement_{i}_rationale"])
        verdict = "CORRECT" if verdicts[label] else "INCORRECT"
        parts.append(f"Statement {label}\nReasoning: {reasoning}\nVerdict: {verdict}")
    return "\n\n".join(parts)


HEADER_RE = re.compile(r"^\W*Statement\s*(\(\w\)|[①②③④⑤])", re.M)
VERDICT_WORD_RE = re.compile(r"Verdict\W*(INCORRECT|CORRECT)", re.I)


def parse_verdicts(output):
    """Extract {label: bool} from a model answer in the training format.
    Each block starts with a line "Statement (x)" and holds one "Verdict:"."""
    out = {}
    heads = list(HEADER_RE.finditer(output))
    for h, nxt in zip(heads, heads[1:] + [None]):
        block = output[h.end(): nxt.start() if nxt else len(output)]
        m = VERDICT_WORD_RE.search(block)
        if m and h.group(1) not in out:
            out[h.group(1)] = m.group(1).upper() == "CORRECT"
    return out


# ------------------------------------------------- lenient parse (any format)
# The strict parser above only reads the training format. A model that never
# saw that format (the base model) often gives clear verdicts in other shapes:
#   "(a) **INCORRECT**: ...", "Correctness: CORRECT", "This statement is CORRECT.",
#   "* Statement (a) is INCORRECT", "Statements ①, ②, and ③ are INCORRECT".
# This parser accepts those. Only upper-case CORRECT/INCORRECT count, so prose
# like "is a correct analysis" is never read as a verdict.

LABEL = r"(?:\(\w\)|[①②③④⑤])"
SUMMARY_RE = re.compile(
    rf"Statements?\s+\**\s*({LABEL}(?:\s*(?:,\s*and|,|and)\s*{LABEL})*)\**\s+(?:is|are)\s+\**\s*(INCORRECT|CORRECT)\b")
ANY_HEADER_RE = re.compile(rf"^\W*(?:Statement\s*)?({LABEL})", re.M)
UPPER_VERDICT_RE = re.compile(r"\b(INCORRECT|CORRECT)\b")


def parse_verdicts_lenient(output, labels):
    out = parse_verdicts(output)
    # 1. explicit summary lines ("Statement (a) is INCORRECT")
    for group, word in SUMMARY_RE.findall(output):
        for label in re.findall(LABEL, group):
            if label in labels:
                out.setdefault(label, word == "CORRECT")
    # 2. per-statement blocks headed by "(a) ..." or "Statement (a)": first verdict word
    heads = [h for h in ANY_HEADER_RE.finditer(output) if h.group(1) in labels]
    for h, nxt in zip(heads, heads[1:] + [None]):
        if h.group(1) in out:
            continue
        block = output[h.end(): nxt.start() if nxt else len(output)]
        m = UPPER_VERDICT_RE.search(block)
        if m:
            out[h.group(1)] = m.group(1) == "CORRECT"
    return out
