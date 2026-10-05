#!/usr/bin/env python3
"""Shared helpers for LEET-Arg fine-tuning experiments."""
from __future__ import annotations
import json, re, random, hashlib
from pathlib import Path

CIRCLED = ["①","②","③","④","⑤"]
SPEAKER_NAMES = ["Alice", "Bob", "Charlie"]
NAME_POOL = [
    "Nora", "Victor", "Priya", "Daniel", "Maya", "Ethan", "Lena", "Omar",
    "Sofia", "Marcus", "Iris", "Noah", "Amina", "Felix", "Rina", "Leo",
    "Tara", "Jonas", "Mei", "Caleb", "Anika", "Hugo", "Sara", "Dylan"
]

def read_json(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")

def split_choices(question: str):
    if "<choices>" not in question:
        return question.strip(), None
    a,b=question.split("<choices>",1)
    return a.strip(), b.strip()

def parse_choice_options(choice_text: str):
    """Return five option texts following ①..⑤."""
    if not choice_text: return []
    pat=re.compile(r"([①②③④⑤])")
    ms=list(pat.finditer(choice_text))
    out=[]
    for i,m in enumerate(ms):
        start=m.end(); end=ms[i+1].start() if i+1<len(ms) else len(choice_text)
        out.append((m.group(1), choice_text[start:end].strip()))
    return out

def is_not_question(question: str):
    # LEET wording varies; these phrases indicate that the selected standalone
    # option is the false/invalid one rather than the true/valid one.
    q=question.lower()
    patterns=[
        "which of the following is not", "which of the following cannot",
        "which of the following cannot be", "which of the following is incorrect",
        "which of the following is not a correct", "which of the following is not an appropriate",
        "which of the following is least", "which of the following does not"
    ]
    return any(p in q for p in patterns)

def statement_items(q):
    st=q.get("statements") or {}
    def kidx(k):
        m=re.search(r"(\d+)$", k); return int(m.group(1)) if m else 999
    return [(k,st[k]) for k in sorted(st,key=kidx)]

def derive_gold_statement_labels(q):
    """Derive one bool per statement from the gold answer and choice encoding.

    Combination questions: selected choice lists the correct statement letters.
    Standalone-option questions: the selected option is the single correct item,
    except for NOT/incorrect/cannot questions where it is the single incorrect item.
    """
    items=statement_items(q)
    n=len(items)
    if not n: raise ValueError(f"{q['id']}: no statements")
    base, choices_text=split_choices(q["original_question"])
    opts=parse_choice_options(choices_text)
    gold=int(q["answer"])
    if len(opts)<gold: raise ValueError(f"{q['id']}: cannot parse choice {gold}")
    chosen=opts[gold-1][1]
    letters=re.findall(r"\(([a-z])\)", chosen.lower())
    # Treat as combination encoding only when chosen option itself is made from
    # statement-letter references such as (a), (b), (c).
    if letters and n <= 10:
        labels=[]
        for i in range(n):
            letter=chr(ord('a')+i)
            labels.append(letter in letters)
        return labels, "combination"
    # Otherwise each option is itself one statement.
    if n != len(opts):
        raise ValueError(f"{q['id']}: standalone choice count {len(opts)} != statements {n}")
    negative=is_not_question(base)
    if negative:
        labels=[True]*n; labels[gold-1]=False
    else:
        labels=[False]*n; labels[gold-1]=True
    return labels, "standalone_not" if negative else "standalone_correct"

def strip_leading_choice_marker(text: str):
    return re.sub(r"^\s*[①②③④⑤]\s*", "", text).strip()

def strip_choice_numbers(q):
    """Remove ①..⑤ answer choices but keep the statements being judged.

    Most combination questions already contain a <statements> block before
    <choices>. Some standalone-choice/diagram questions store their candidate
    statements only in q["statements"], so we append those explicitly after
    removing their circled choice markers.
    """
    base,_=split_choices(q["original_question"])
    base=base.strip()
    if "<statements>" not in base.lower():
        items=statement_items(q)
        if items:
            lines=["<statements>"]
            for i,(_,txt) in enumerate(items,1):
                lines.append(f"Statement {i}: {strip_leading_choice_marker(txt)}")
            base += "\n" + "\n".join(lines)
    return base

def deterministic_name_map(qid: str, seed: int):
    digest=hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()
    rng=random.Random(int(digest[:16],16))
    pool=NAME_POOL[:]; rng.shuffle(pool)
    return dict(zip(SPEAKER_NAMES,pool[:len(SPEAKER_NAMES)]))

def rename_speakers(text: str, mapping: dict):
    # Alice1 becomes Nora1 because the word boundary occurs before the digit.
    out=text
    for old,new in mapping.items():
        out=re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out

def labels_to_target(labels):
    lines=[f"Statement {i}: {'CORRECT' if v else 'INCORRECT'}" for i,v in enumerate(labels,1)]
    return "\n".join(lines)

def labels_to_compact_structured_target(labels):
    # This is intentionally not called an argumentation-framework rationale.
    # It only adds a deterministic aggregation step without inventing premises.
    lines=[f"Statement {i}: {'CORRECT' if v else 'INCORRECT'}" for i,v in enumerate(labels,1)]
    correct=[str(i) for i,v in enumerate(labels,1) if v]
    lines.append("Correct statements: " + (", ".join(correct) if correct else "none"))
    return "\n".join(lines)
