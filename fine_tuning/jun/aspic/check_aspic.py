#!/usr/bin/env python3
"""Check ASPIC+ translations of LEET-Arg questions against the gold answers.

No language model is involved. For each translation the script:
  1. builds the ASPIC+ arguments and attacks from premises, rules, contraries
     and undercuts,
  2. computes the grounded labelling (with PyArg if it is installed, otherwise
     with the built-in solver, which gives the same result),
  3. runs the check of every statement and gets True / False,
  4. maps the true/false pattern to a choice (1-5) using the question's
     choice list and wording, and compares it with the gold answer.

Usage:
  python check_aspic.py --bench benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
      --split fine_tuning/jun/data/split.json  translations/
  python check_aspic.py --bench ... gemini_output.json      # a JSON array also works

The format is described in ASPIC_FORMAT.md.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

# The 16 final test questions (testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json).
# They must never be translated or trained on. --test-set reads the file instead.
FINAL_TEST_IDS = set(
    "2021_18 2021_20 2021_33 2021_36 2022_30 2022_38 2023_08 2023_15 "
    "2024_01 2024_06 2024_21 2024_24 2024_29 2025_08 2025_13 2025_22".split()
)

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
RANK = {"IN": 3, "UNDEC": 2, "OUT": 1, "NONE": 0}
STAR = "*"
MAX_ARGS = 5000


class TranslationError(Exception):
    pass


# --------------------------------------------------------------------------
# Theory


def as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


class Theory:
    def __init__(self, claims, premises, rules, contraries, undercuts, positions):
        self.claims = dict(claims)
        self.positions = positions
        # claim -> {"owners": set or None (None = everyone), "kind": ...}
        self.premises = {}
        for p in premises:
            self._add_premise(p)
        self.rules = {}
        for r in rules:
            self._add_rule(r)
        self.contraries = set()
        for pair in contraries:
            if len(pair) != 2:
                raise TranslationError(f"contrary pair must have two claims: {pair}")
            for c in pair:
                self._need_claim(c, "contraries")
            self.contraries.add(frozenset(pair))
        self.undercuts = set()
        for u in undercuts:
            self._need_claim(u["claim"], "undercuts")
            if u["rule"] not in self.rules:
                raise TranslationError(f"undercut names unknown rule {u['rule']}")
            self.undercuts.add((u["claim"], u["rule"]))

    # -- building -----------------------------------------------------------

    def _need_claim(self, c, where):
        if c not in self.claims:
            raise TranslationError(f"unknown claim {c!r} in {where}")

    def _owners(self, by, where):
        labels = as_list(by)
        if not labels:
            raise TranslationError(f"missing 'by' in {where}")
        if STAR in labels:
            return None
        for b in labels:
            if b not in self.positions:
                raise TranslationError(f"unknown position {b!r} in {where}")
        return set(labels)

    def _add_premise(self, p):
        c = p["claim"]
        self._need_claim(c, "premises")
        kind = p.get("kind", "ordinary")
        if kind not in ("ordinary", "axiom"):
            raise TranslationError(f"premise kind must be ordinary or axiom, got {kind!r}")
        owners = self._owners(p.get("by"), f"premise {c}")
        if c in self.premises:
            old = self.premises[c]
            old["owners"] = None if (old["owners"] is None or owners is None) else old["owners"] | owners
            if kind == "axiom":
                old["kind"] = "axiom"
        else:
            self.premises[c] = {"owners": owners, "kind": kind}

    def _add_rule(self, r):
        rid = r["id"]
        if rid in self.rules:
            raise TranslationError(f"duplicate rule id {rid}")
        if r.get("type") not in ("strict", "defeasible"):
            raise TranslationError(f"rule {rid}: type must be strict or defeasible")
        ante = as_list(r.get("if"))
        if not ante:
            raise TranslationError(f"rule {rid} has no 'if' claims")
        for c in ante + [r["then"]]:
            self._need_claim(c, f"rule {rid}")
        self.rules[rid] = {
            "if": ante,
            "then": r["then"],
            "type": r["type"],
            "owners": self._owners(r.get("by"), f"rule {rid}"),
        }

    def extended(self, add):
        """A copy of this theory with a hypothetical 'add' block merged in."""
        if not add:
            return self
        claims = dict(self.claims)
        for k, v in (add.get("claims") or {}).items():
            if k in claims and claims[k] != v:
                raise TranslationError(f"'add' redefines claim {k}")
            claims[k] = v
        premises = [
            {"claim": c, "by": sorted(p["owners"]) if p["owners"] else STAR, "kind": p["kind"]}
            for c, p in self.premises.items()
        ] + list(add.get("premises") or [])
        rules = [
            {"id": rid, "if": r["if"], "then": r["then"], "type": r["type"],
             "by": sorted(r["owners"]) if r["owners"] else STAR}
            for rid, r in self.rules.items()
        ] + list(add.get("rules") or [])
        contraries = [sorted(p) for p in self.contraries] + list(add.get("contraries") or [])
        undercuts = [{"claim": c, "rule": r} for c, r in self.undercuts] + list(add.get("undercuts") or [])
        return Theory(claims, premises, rules, contraries, undercuts, self.positions)

    # -- arguments ----------------------------------------------------------

    def arguments(self):
        if hasattr(self, "_args"):
            return self._args
        args = {}  # key -> argument dict

        def owners_meet(a, b):
            if a is None:
                return None if b is None else set(b)
            if b is None:
                return set(a)
            return a & b

        for c, p in self.premises.items():
            key = f"P({c})"
            args[key] = {
                "key": key, "concl": c, "rule": None, "type": None, "kind": p["kind"],
                "subs": [], "all": {key}, "claims_used": {c}, "owners": p["owners"],
            }
        changed = True
        while changed:
            changed = False
            by_concl = {}
            for a in args.values():
                by_concl.setdefault(a["concl"], []).append(a)
            for rid, r in self.rules.items():
                pools = [by_concl.get(c, []) for c in r["if"]]
                if any(not pool for pool in pools):
                    continue
                for combo in itertools.product(*pools):
                    key = f"{rid}(" + ",".join(s["key"] for s in combo) + ")"
                    if key in args:
                        continue
                    used = set().union(*(s["claims_used"] for s in combo))
                    if r["then"] in used:  # no circular arguments
                        continue
                    owners = r["owners"]
                    for s in combo:
                        owners = owners_meet(owners, s["owners"])
                    if owners is not None and not owners:
                        owners = set()  # nobody can make it alone, but it still exists
                    args[key] = {
                        "key": key, "concl": r["then"], "rule": rid, "type": r["type"], "kind": None,
                        "subs": [s["key"] for s in combo],
                        "all": {key}.union(*(s["all"] for s in combo)),
                        "claims_used": used | {r["then"]}, "owners": owners,
                    }
                    changed = True
                    if len(args) > MAX_ARGS:
                        raise TranslationError("too many arguments; is a rule set circular?")
        self._args = args
        return args

    def is_contrary(self, x, y):
        return frozenset((x, y)) in self.contraries

    def attacks(self):
        if hasattr(self, "_attacks"):
            return self._attacks
        args = self.arguments()
        result = set()
        for x in args.values():
            for y in args.values():
                for sk in y["all"]:
                    s = args[sk]
                    if s["rule"] is None:
                        hit = s["kind"] == "ordinary" and self.is_contrary(x["concl"], s["concl"])
                    else:
                        hit = s["type"] == "defeasible" and (
                            self.is_contrary(x["concl"], s["concl"])
                            or (x["concl"], s["rule"]) in self.undercuts
                        )
                    if hit:
                        result.add((x["key"], y["key"]))
                        break
        self._attacks = result
        return result

    def labels(self, use_pyarg):
        if hasattr(self, "_labels"):
            return self._labels
        args = self.arguments()
        att = self.attacks()
        attackers = {k: set() for k in args}
        for a, b in att:
            attackers[b].add(a)
        lab = grounded_builtin(args, attackers)
        if use_pyarg:
            lab_pyarg = grounded_pyarg(args, att, attackers)
            if lab_pyarg != lab:
                raise RuntimeError("PyArg and the built-in solver disagree; please report this")
        self._labels = lab
        return lab

    def claim_status(self, claim, use_pyarg):
        lab = self.labels(use_pyarg)
        found = [lab[k] for k, a in self.arguments().items() if a["concl"] == claim]
        if not found:
            return "NONE"
        for s in ("IN", "UNDEC", "OUT"):
            if s in found:
                return s


def grounded_builtin(args, attackers):
    lab = {k: "UNDEC" for k in args}
    changed = True
    while changed:
        changed = False
        for k in args:
            if lab[k] != "UNDEC":
                continue
            if all(lab[a] == "OUT" for a in attackers[k]):
                lab[k] = "IN"
                changed = True
            elif any(lab[a] == "IN" for a in attackers[k]):
                lab[k] = "OUT"
                changed = True
    return lab


def grounded_pyarg(args, att, attackers):
    from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import (
        AbstractArgumentationFramework,
    )
    from py_arg.abstract_argumentation_classes.argument import Argument
    from py_arg.abstract_argumentation_classes.defeat import Defeat
    from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension

    nodes = {k: Argument(k) for k in args}
    af = AbstractArgumentationFramework("af", list(nodes.values()), [Defeat(nodes[a], nodes[b]) for a, b in att])
    ext = {a.name for a in get_grounded_extension(af)}
    lab = {}
    for k in args:
        if k in ext:
            lab[k] = "IN"
        elif attackers[k] & ext:
            lab[k] = "OUT"
        else:
            lab[k] = "UNDEC"
    return lab


# --------------------------------------------------------------------------
# Checks


class Checker:
    def __init__(self, base, positions, use_pyarg, hypotheticals=None):
        self.base = base
        self.positions = positions
        self.use_pyarg = use_pyarg
        self.hyp = hypotheticals or {}
        self.ops_used = []

    def add_block(self, check):
        add = check.get("add")
        if isinstance(add, str):
            if add not in self.hyp:
                raise TranslationError(f"unknown hypothetical {add!r}")
            return self.hyp[add]
        return add

    def resolve(self, claim, theory):
        if claim in self.positions:
            t = self.positions[claim].get("thesis")
            if not t:
                raise TranslationError(f"position {claim} has no thesis")
            return t
        if claim not in theory.claims:
            raise TranslationError(f"unknown claim {claim!r} in a check")
        return claim

    def run(self, check):
        op = check.get("op")
        self.ops_used.append(op)
        if op in ("and", "or"):
            vals = [self.run(c) for c in check["args"]]
            if any(v is None for v in vals):
                return None
            return all(vals) if op == "and" else any(vals)
        if op == "not":
            v = self.run(check["arg"])
            return None if v is None else not v
        if op == "unsupported":
            return None

        theory = self.base.extended(self.add_block(check))
        if op == "holds":
            pos, c = check["by"], self.resolve(check["claim"], theory)
            if pos not in self.positions:
                raise TranslationError(f"unknown position {pos!r} in holds")
            return any(
                a["concl"] == c and (a["owners"] is None or pos in a["owners"])
                for a in theory.arguments().values()
            )
        if op == "assumes":
            pos, c = check["by"], self.resolve(check["claim"], theory)
            p = theory.premises.get(c)
            return bool(p) and (p["owners"] is None or pos in p["owners"])
        if op == "status":
            c = self.resolve(check["claim"], theory)
            wanted = as_list(check.get("is"))
            if not wanted or any(w not in RANK for w in wanted):
                raise TranslationError("status check needs 'is' with IN/OUT/UNDEC/NONE")
            return theory.claim_status(c, self.use_pyarg) in wanted
        if op in ("weakens", "strengthens"):
            if not self.add_block(check):
                raise TranslationError(f"{op} needs an 'add' block")
            c = self.resolve(check["claim"], theory)
            before = RANK[self.base.claim_status(c, self.use_pyarg)] if c in self.base.claims else 0
            after_status = theory.claim_status(c, self.use_pyarg)
            after = RANK[after_status]
            old_keys = set(self.base.arguments()) if c in self.base.claims else set()
            lab = theory.labels(self.use_pyarg)
            targets = {k for k, a in theory.arguments().items() if a["concl"] == c}
            if op == "weakens":
                if after < before:
                    return True
                if after > before:
                    return False
                new_live_attackers = {
                    x for x, y in theory.attacks()
                    if y in targets and x not in old_keys and lab[x] != "OUT"
                }
                return bool(new_live_attackers)
            else:
                if after > before:
                    return True
                if after < before:
                    return False
                return any(k not in old_keys and lab[k] != "OUT" for k in targets)
        raise TranslationError(f"unknown check op {op!r}")


# --------------------------------------------------------------------------
# Benchmark side: choices, polarity, gold


def load_bench(path):
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or list(data.values())
    return {q["id"]: q for q in data}


def gold_number(ans):
    ans = str(ans).strip()
    if ans in CIRCLED:
        return CIRCLED[ans]
    m = re.search(r"\d", ans)
    return int(m.group()) if m else None


def parse_choices(question_text):
    """Return ('combo', {1: {'a'}, ...}) or ('single', None)."""
    parts = re.split(r"<choices>", question_text, flags=re.I)
    if len(parts) < 2:
        return "single", None
    tail = parts[-1]
    # markers are ①..⑤, or "1." .. "5." in a few records
    pieces = re.split(r"([①②③④⑤]|(?<![\w(])[1-5]\.(?=\s))", tail)
    choices = {}
    for i in range(1, len(pieces) - 1, 2):
        mark = pieces[i]
        num = CIRCLED.get(mark) or int(mark[0])
        letters = set(re.findall(r"\(([a-e])\)", pieces[i + 1]))
        choices[num] = letters
    if choices and all(choices.values()):
        return "combo", choices
    return "single", None


def asks_for(question_text):
    q = question_text.split("<question>", 1)[-1]
    m = re.search(r"\((?:a|1)\)|<choices>|①", q)
    q = q[: m.start()] if m else q
    neg = re.search(r"\bNOT\b|\bcannot\b|\bnot\b|\bincorrect\b|\binappropriate\b|\bfalse\b", q, re.I)
    return "incorrect" if neg else "correct"


def predict(truth, mode, choices, polarity):
    if any(v is None for v in truth.values()):
        return None
    wanted = polarity == "correct"
    picked = {k for k, v in truth.items() if v == wanted}
    if mode == "combo":
        hits = [n for n, s in choices.items() if s == picked]
        return hits[0] if len(hits) == 1 else None
    nums = [int(k) for k in picked if k.isdigit()]
    return nums[0] if len(nums) == 1 else None


# --------------------------------------------------------------------------
# Grounding: is passage text really in the passage?

STOP = set("""the a an and or of to in on for with by from that this these those is are was were be been
being it its as at not no than then there their they them which who whom whose what when where why how
can could would should may might must will shall do does did has have had more less such also only
each other into about over under after before between because however therefore thus""".split())


def grounding(text, source_text):
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in STOP]
    if not words:
        return 1.0
    src = source_text.lower()
    return sum(w in src or w[:-1] in src for w in words) / len(words)


def all_claims(t):
    out = dict(t.get("claims") or {})

    def walk(chk):
        if not isinstance(chk, dict):
            return
        add = chk.get("add")
        if isinstance(add, dict):
            out.update(add.get("claims") or {})
        for c in chk.get("args") or []:
            walk(c)
        walk(chk.get("arg"))

    for h in (t.get("hypotheticals") or {}).values():
        out.update(h.get("claims") or {})
    for s in (t.get("statements") or {}).values():
        walk(s.get("check"))
    return out


def all_rules(t):
    out = list(t.get("rules") or [])

    def walk(chk):
        if not isinstance(chk, dict):
            return
        add = chk.get("add")
        if isinstance(add, dict):
            out.extend(add.get("rules") or [])
        for c in chk.get("args") or []:
            walk(c)
        walk(chk.get("arg"))

    for h in (t.get("hypotheticals") or {}).values():
        out.extend(h.get("rules") or [])
    for s in (t.get("statements") or {}).values():
        walk(s.get("check"))
    return out


# --------------------------------------------------------------------------


def check_translation(t, bench, split, test_ids, use_pyarg):
    qid = t.get("id")
    rec = {"id": qid, "errors": [], "warnings": []}
    if qid in test_ids:
        rec["errors"].append("FINAL TEST QUESTION: must not be translated or trained on")
        return rec
    if qid not in bench:
        rec["errors"].append("id not found in the benchmark file")
        return rec
    if split:
        if qid in split.get("valid", []):
            rec["warnings"].append("validation question (use for evaluation, not training)")
        elif qid not in split.get("train", []):
            rec["warnings"].append("not in the train split")
    q = bench[qid]
    qtext = q["original_question"]
    mode, choices = parse_choices(qtext)
    polarity = t.get("asks_for") or asks_for(qtext)
    gold = gold_number(q["answer"])
    rec.update(mode=mode, polarity=polarity, gold=gold)

    for cid, c in all_claims(t).items():
        if c.get("source") in ("passage", "statement"):
            g = grounding(c["text"], qtext)
            if g < 0.6:
                rec["warnings"].append(f"{cid} marked {c['source']} but only {g:.0%} of its words are in the question")
    elems = list(all_claims(t).values()) + all_rules(t)
    rec["outside_passage"] = sum(e.get("source") in ("rationale", "background") for e in elems)
    rec["elements"] = len(elems)

    try:
        positions = t["positions"]
        base = Theory(t["claims"], t["premises"], t["rules"], t.get("contraries", []),
                      t.get("undercuts", []), positions)
        checker = Checker(base, positions, use_pyarg, t.get("hypotheticals"))
        truth = {}
        for label, s in t["statements"].items():
            truth[label] = checker.run(s["check"])
        rec["ops"] = checker.ops_used
        rec["n_args"] = len(base.arguments())
    except (TranslationError, KeyError, TypeError) as e:
        rec["errors"].append(f"invalid translation: {e}")
        return rec

    needed = set().union(*choices.values()) if mode == "combo" else {"1", "2", "3", "4", "5"}
    missing = needed - set(truth)
    if missing:
        rec["errors"].append(f"no check for statement(s) {sorted(missing)}")
        return rec
    rec["truth"] = truth
    rec["pred"] = predict(truth, mode, choices, polarity)
    rec["match"] = rec["pred"] is not None and rec["pred"] == gold
    return rec


def load_translations(paths):
    out = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            data = json.load(open(f, encoding="utf-8"))
            items = data if isinstance(data, list) else [data]
            for t in items:
                t.setdefault("_file", str(f))
                out.append(t)
    return out


def fmt_truth(truth):
    return " ".join(f"{k}={'T' if v else ('?' if v is None else 'F')}" for k, v in truth.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="translation files or folders")
    ap.add_argument("--bench", required=True, help="benchmark JSON (cleaned file recommended)")
    ap.add_argument("--split", help="fine_tuning/jun/data/split.json")
    ap.add_argument("--test-set", help="LEET_Arg_Questions_Test_Set.json (default: built-in list of 16 IDs)")
    ap.add_argument("--no-pyarg", action="store_true", help="use only the built-in solver")
    ap.add_argument("--out", help="write a JSON report here")
    ap.add_argument("--save-matches", metavar="DIR",
                    help="write every translation that reproduces the gold answer to DIR/<id>.json")
    ap.add_argument("-v", "--verbose", action="store_true", help="print warnings and ops per question")
    args = ap.parse_args()

    bench = load_bench(args.bench)
    split = json.load(open(args.split)) if args.split else None
    test_ids = FINAL_TEST_IDS
    if args.test_set:
        test_ids = {q["id"] for q in json.load(open(args.test_set))}
    use_pyarg = False
    if not args.no_pyarg:
        try:
            import py_arg  # noqa: F401
            use_pyarg = True
        except ImportError:
            pass
    print(f"solver: {'PyArg grounded extension (cross-checked with built-in)' if use_pyarg else 'built-in grounded labelling'}")

    translations = load_translations(args.paths)
    recs = [check_translation(t, bench, split, test_ids, use_pyarg) for t in translations]

    print(f"\n{'id':8} {'asks for':9} {'statements':22} {'pred':>4} {'gold':>4}  result")
    for r in recs:
        if r["errors"]:
            print(f"{r['id']:8} ERROR: {'; '.join(r['errors'])}")
            continue
        res = "MATCH" if r["match"] else ("UNRESOLVED" if r["pred"] is None else "WRONG")
        print(f"{r['id']:8} {r['polarity']:9} {fmt_truth(r['truth']):22} {str(r['pred']):>4} {r['gold']:>4}  {res}")
        if args.verbose:
            print(f"         {r['n_args']} arguments; {r['outside_passage']}/{r['elements']} claims+rules from outside the passage")
        for w in r["warnings"]:
            print(f"         warning: {w}")

    ok = [r for r in recs if not r["errors"]]
    n_match = sum(r["match"] for r in ok)
    n_unres = sum(r["pred"] is None for r in ok)
    n_stmt = sum(len(r["truth"]) for r in ok)
    n_unsup = sum(v is None for r in ok for v in r["truth"].values())
    print(f"\n{n_match}/{len(ok)} translations reproduce the gold answer"
          f" ({n_unres} unresolved, {len(recs) - len(ok)} with errors)")
    print(f"{n_unsup}/{n_stmt} statements marked unsupported")
    if args.out:
        json.dump(recs, open(args.out, "w"), indent=2, ensure_ascii=False, default=sorted)
        print(f"report written to {args.out}")
    if args.save_matches:
        d = Path(args.save_matches)
        d.mkdir(parents=True, exist_ok=True)
        saved = []
        for t, r in zip(translations, recs):
            if r.get("match"):
                clean = {k: v for k, v in t.items() if k != "_file"}
                dest = d / f"{t['id']}.json"
                if dest.exists() and Path(t["_file"]).resolve() == dest.resolve():
                    continue
                dest.write_text(json.dumps(clean, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                saved.append(t["id"])
        print(f"saved {len(saved)} matching translation(s) to {d}/: {', '.join(saved) or '-'}")
        print("only MATCH ones are saved; read each once against its passage before training")
    sys.exit(0 if not any(r["errors"] for r in recs) else 1)


if __name__ == "__main__":
    main()
