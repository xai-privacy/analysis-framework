"""Prompting strategies, all behind one interface.

Every strategy takes a question and an LLM and returns a predicted choice plus a
full trace and compute accounting. Strategies differ enormously in cost -- zero-shot
is one call, self-consistency is k, ToT and GoT are k+2 -- which is why CallLog
travels with every result.

On ToT and GoT: the published methods are search procedures over a thought tree /
graph with scoring, backtracking and aggregation across many levels. Running that
faithfully against an 8B model on CPU is not practical, and arguably not meaningful
for a single multiple-choice judgement. What is implemented here are *single-level*
reductions that preserve the characteristic shape of each method:

  ToT -> branch into k candidate lines of reasoning, score them, expand the best.
  GoT -> generate k independent thoughts, aggregate them, then refine.

These are labelled as reductions everywhere they appear. Treat results as evidence
about the reduced variant, not about the published algorithm.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from llm import CallLog, parse_answer

# ---------------------------------------------------------------------------
# Shared prompt pieces
# ---------------------------------------------------------------------------

# Adapted from the paper's "Problem Solving Prompt" (Section 4.2). The original
# asked for Korean output; the released dataset is the English translation, so
# this asks for English. That deviation is deliberate and documented in the README.
BASE_INSTRUCTION = (
    "Solve the problem and provide explanations for your solution.\n"
    "Select the most appropriate option to answer the question.\n"
    "Do not repeat the statements."
)

ANSWER_FORMAT = (
    "Keep your explanation under about 150 words, then end your response with your "
    "final choice on its own line, exactly as:\n"
    "Answer: <the number of the option you choose, 1-5>"
)

# Asked as a cheap follow-up when the first response has no parseable answer --
# usually because generation hit the output budget mid-sentence. This mirrors the
# two-stage form of zero-shot CoT (reason, then extract the answer), and costs
# only a handful of tokens.
EXTRACT_SUFFIX = "\n\nTherefore, among options 1 through 5, the answer is:"


def _answer_or_extract(res, llm, reasoning_text: str, q: dict,
                       temperature: float = 0.0, seed=None):
    """Parse an answer; if that fails, make one short extraction call.

    Without this, a response that reasons well but runs out of output budget
    before printing its final line is scored as incorrect, which measures the
    token budget rather than the technique.
    """
    ans = parse_answer(reasoning_text)
    if ans is not None:
        return ans
    prompt = (f"{_problem_block(q)}\n\n<reasoning>\n{reasoning_text}\n</reasoning>"
              + EXTRACT_SUFFIX)
    r = llm.generate(prompt, temperature=temperature, seed=seed, num_predict=12)
    res.calls.add(r)
    res.log("extract_answer", prompt, r.text)
    return parse_answer(r.text)


def _problem_block(q: dict) -> str:
    return f"<problem>\n{q['original_question']}\n</problem>"


@dataclass
class StrategyResult:
    predicted: Optional[str]
    calls: CallLog
    trace: List[Dict] = field(default_factory=list)

    def log(self, stage: str, prompt: str, response: str):
        self.trace.append({"stage": stage, "prompt": prompt, "response": response})


class Strategy:
    name = "base"
    #: Rough number of model calls per question, for cost expectations.
    calls_per_question = 1

    def run(self, q: dict, llm, temperature: float = 0.0,
            seed: Optional[int] = None) -> StrategyResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Zero-shot (the paper's baseline)
# ---------------------------------------------------------------------------

class ZeroShot(Strategy):
    name = "zero_shot"

    def build_prompt(self, q: dict) -> str:
        return f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n{ANSWER_FORMAT}\n"

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())
        p = self.build_prompt(q)
        r = llm.generate(p, temperature=temperature, seed=seed)
        res.calls.add(r)
        res.log("answer", p, r.text)
        res.predicted = _answer_or_extract(res, llm, r.text, q, temperature, seed)
        return res


# ---------------------------------------------------------------------------
# 2. Zero-shot chain-of-thought
# ---------------------------------------------------------------------------

class ZeroShotCoT(Strategy):
    """Kojima et al.-style 'let's think step by step', in a single call.

    Kept as one call (rather than the original two-stage extract-answer variant)
    because the answer format is already constrained to a final line, which the
    parser reads from the end of the response.
    """
    name = "zero_shot_cot"

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())
        p = (f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n"
             "Let's think step by step. Work through each option in turn, stating for "
             "each whether it is a correct analysis and why, before committing to an "
             "overall choice.\n\n" + ANSWER_FORMAT + "\n")
        r = llm.generate(p, temperature=temperature, seed=seed)
        res.calls.add(r)
        res.log("cot", p, r.text)
        res.predicted = _answer_or_extract(res, llm, r.text, q, temperature, seed)
        return res


# ---------------------------------------------------------------------------
# 3. Few-shot chain-of-thought
# ---------------------------------------------------------------------------

class FewShotCoT(Strategy):
    """Worked examples drawn from the dev split, using the dataset's own
    expert-authored rationales as the demonstration reasoning.

    Those rationales are gold-standard legal reasoning, which makes them strong
    demonstrations -- but it also means example provenance matters. The example
    pool must come from the dev split only, and the current question is always
    excluded, so evaluation items never appear as their own demonstration.
    """
    name = "few_shot_cot"

    def __init__(self, examples: List[dict], k: int = 2, max_rationale_chars: int = 1200):
        self.examples = examples
        self.k = k
        self.max_rationale_chars = max_rationale_chars

    def _select(self, q: dict) -> List[dict]:
        pool = [e for e in self.examples if e["id"] != q["id"]]
        return pool[:self.k]

    def build_prompt(self, q: dict) -> str:
        parts = [BASE_INSTRUCTION, ""]
        for ex in self._select(q):
            rationale = (ex.get("original_rationale") or "").strip()
            if len(rationale) > self.max_rationale_chars:
                rationale = rationale[:self.max_rationale_chars].rsplit(" ", 1)[0] + " ..."
            parts += [
                "<example>",
                _problem_block(ex),
                f"Reasoning: {rationale}",
                f"Answer: {ex['answer']}",
                "</example>",
                "",
            ]
        parts += ["Now solve this problem in the same way.", "",
                  _problem_block(q), "", ANSWER_FORMAT, ""]
        return "\n".join(parts)

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())
        p = self.build_prompt(q)
        r = llm.generate(p, temperature=temperature, seed=seed)
        res.calls.add(r)
        res.log("few_shot", p, r.text)
        res.predicted = _answer_or_extract(res, llm, r.text, q, temperature, seed)
        return res


# ---------------------------------------------------------------------------
# 4. Self-consistency
# ---------------------------------------------------------------------------

class SelfConsistency(Strategy):
    """Sample k CoT paths at non-zero temperature, take the majority answer.

    Note this interacts with the robustness metric: self-consistency reduces
    run-to-run variance by construction, so it should look good on exactly the
    dimension where small models are weakest. Whether that is worth k times the
    compute is the question the token accounting is there to answer.
    """
    name = "self_consistency"

    def __init__(self, k: int = 5, temperature: float = 0.7):
        self.k = k
        self.sc_temperature = temperature
        self.calls_per_question = k

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())
        base = ZeroShotCoT()
        votes = []
        for i in range(self.k):
            p = (f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n"
                 "Let's think step by step. Work through each option in turn before "
                 "committing to an overall choice.\n\n" + ANSWER_FORMAT + "\n")
            s = None if seed is None else seed + i
            r = llm.generate(p, temperature=self.sc_temperature, seed=s)
            res.calls.add(r)
            res.log(f"sample_{i}", p, r.text)
            a = parse_answer(r.text)
            if a is None and r.text.strip():
                a = _answer_or_extract(res, llm, r.text, q, temperature, s)
            if a is not None:
                votes.append(a)
        if votes:
            # Majority vote; ties break toward the earliest-sampled answer.
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            best = max(counts.values())
            for v in votes:
                if counts[v] == best:
                    res.predicted = v
                    break
        res.trace.append({"stage": "vote", "prompt": "", "response": str(votes)})
        return res


# ---------------------------------------------------------------------------
# 5. Plan-and-solve
# ---------------------------------------------------------------------------

class PlanAndSolve(Strategy):
    """Wang et al. plan-and-solve, targeting CoT's missing-step errors -- which is
    the failure the LEET-Arg paper specifically documents ('logical leaps between
    premises and conclusions, omission of necessary premises')."""
    name = "plan_and_solve"

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())
        p = (f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n"
             "Let's first understand the problem and devise a plan to solve it. "
             "Identify the competing positions, the premises each depends on, and what "
             "would have to be true for each option to hold. Then carry out the plan, "
             "solving step by step and stating every premise you rely on.\n\n"
             + ANSWER_FORMAT + "\n")
        r = llm.generate(p, temperature=temperature, seed=seed)
        res.calls.add(r)
        res.log("plan_and_solve", p, r.text)
        res.predicted = _answer_or_extract(res, llm, r.text, q, temperature, seed)
        return res


# ---------------------------------------------------------------------------
# 6. Self-refine
# ---------------------------------------------------------------------------

class SelfRefine(Strategy):
    """Answer, then critique that answer, then revise.

    Directly aimed at the paper's headline finding -- correct answers supported by
    invalid reasoning. Watch for sycophantic collapse, where the critique stage
    simply agrees; the trace records all three stages so this is checkable.
    """
    name = "self_refine"
    calls_per_question = 3

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())

        p1 = (f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n"
              "Reason step by step.\n\n" + ANSWER_FORMAT + "\n")
        r1 = llm.generate(p1, temperature=temperature, seed=seed)
        res.calls.add(r1); res.log("initial", p1, r1.text)

        p2 = (f"{_problem_block(q)}\n\n<attempt>\n{r1.text}\n</attempt>\n\n"
              "Critique the attempt above. Identify specifically: any premise asserted "
              "without support, any step that does not follow from what precedes it, any "
              "option dismissed without a reason, and any place where the reasoning merely "
              "restates the question instead of analysing it. Do not give an answer yet.")
        r2 = llm.generate(p2, temperature=temperature, seed=seed)
        res.calls.add(r2); res.log("critique", p2, r2.text)

        p3 = (f"{_problem_block(q)}\n\n<attempt>\n{r1.text}\n</attempt>\n\n"
              f"<critique>\n{r2.text}\n</critique>\n\n"
              "Using the critique, produce a corrected analysis. If the critique found no "
              "real fault, keep the original conclusion.\n\n" + ANSWER_FORMAT + "\n")
        r3 = llm.generate(p3, temperature=temperature, seed=seed)
        res.calls.add(r3); res.log("refined", p3, r3.text)

        res.predicted = (parse_answer(r3.text) or parse_answer(r1.text)
                         or _answer_or_extract(res, llm, r3.text, q, temperature, seed))
        return res


# ---------------------------------------------------------------------------
# 7. Tree-of-Thoughts (single-level reduction)
# ---------------------------------------------------------------------------

class TreeOfThoughts(Strategy):
    """Branch -> score -> expand best. One level, beam width 1.

    A faithful ToT would search a multi-level tree with backtracking. For a single
    5-way multiple-choice judgement there is little to backtrack over, so this
    reduction branches once into k candidate lines of reasoning, has the model score
    them, and expands the highest-scored one.
    """
    name = "tree_of_thoughts"

    def __init__(self, k: int = 3, branch_temperature: float = 0.8):
        self.k = k
        self.branch_temperature = branch_temperature
        self.calls_per_question = k + 2

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())

        thoughts = []
        for i in range(self.k):
            p = (f"{_problem_block(q)}\n\n"
                 "Propose ONE distinct line of attack on this problem: which position or "
                 "option you would examine first and why it is the most promising place to "
                 "start. Give only the approach, three sentences at most. Do not solve it yet.")
            s = None if seed is None else seed + i
            r = llm.generate(p, temperature=self.branch_temperature, seed=s)
            res.calls.add(r); res.log(f"branch_{i}", p, r.text)
            thoughts.append(r.text.strip())

        listing = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(thoughts))
        p_eval = (f"{_problem_block(q)}\n\n<approaches>\n{listing}\n</approaches>\n\n"
                  "Judge which approach is most likely to reach the correct answer. "
                  "Reply with the number of the best approach on its own line as:\n"
                  "Best: <number>")
        r_eval = llm.generate(p_eval, temperature=temperature, seed=seed)
        res.calls.add(r_eval); res.log("evaluate", p_eval, r_eval.text)

        import re
        m = re.search(r"Best:\s*\[?(\d+)", r_eval.text, re.I)
        idx = int(m.group(1)) - 1 if m else 0
        if not (0 <= idx < len(thoughts)):
            idx = 0
        res.trace.append({"stage": "selected_branch", "prompt": "", "response": str(idx + 1)})

        p_final = (f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n"
                   f"<chosen_approach>\n{thoughts[idx]}\n</chosen_approach>\n\n"
                   "Follow the chosen approach and solve the problem step by step.\n\n"
                   + ANSWER_FORMAT + "\n")
        r_final = llm.generate(p_final, temperature=temperature, seed=seed)
        res.calls.add(r_final); res.log("expand", p_final, r_final.text)
        res.predicted = _answer_or_extract(res, llm, r_final.text, q, temperature, seed)
        return res


# ---------------------------------------------------------------------------
# 8. Graph-of-Thoughts (single-level reduction)
# ---------------------------------------------------------------------------

class GraphOfThoughts(Strategy):
    """Generate -> aggregate -> refine.

    The defining GoT operation is aggregation: merging several thoughts into a
    stronger combined one, rather than only branching as in ToT. This reduction
    keeps that operation but drops the scored, multi-round graph.
    """
    name = "graph_of_thoughts"

    def __init__(self, k: int = 3, gen_temperature: float = 0.8):
        self.k = k
        self.gen_temperature = gen_temperature
        self.calls_per_question = k + 2

    def run(self, q, llm, temperature=0.0, seed=None):
        res = StrategyResult(None, CallLog())

        thoughts = []
        for i in range(self.k):
            p = (f"{_problem_block(q)}\n\n"
                 "Give ONE partial analysis of this problem, examining a different aspect "
                 "than an obvious first reading would: for example the assumptions a "
                 "position depends on, or the conditions under which an option fails. "
                 "Five sentences at most. Do not give a final answer.")
            s = None if seed is None else seed + i
            r = llm.generate(p, temperature=self.gen_temperature, seed=s)
            res.calls.add(r); res.log(f"thought_{i}", p, r.text)
            thoughts.append(r.text.strip())

        listing = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(thoughts))
        p_agg = (f"{_problem_block(q)}\n\n<partial_analyses>\n{listing}\n</partial_analyses>\n\n"
                 "Merge these partial analyses into one coherent analysis. Keep what each "
                 "gets right, discard what is mistaken, and resolve any contradictions "
                 "between them explicitly. Do not give a final answer yet.")
        r_agg = llm.generate(p_agg, temperature=temperature, seed=seed)
        res.calls.add(r_agg); res.log("aggregate", p_agg, r_agg.text)

        p_ref = (f"{BASE_INSTRUCTION}\n\n{_problem_block(q)}\n\n"
                 f"<merged_analysis>\n{r_agg.text}\n</merged_analysis>\n\n"
                 "Refine the merged analysis into a final decision, checking each option "
                 "against it.\n\n" + ANSWER_FORMAT + "\n")
        r_ref = llm.generate(p_ref, temperature=temperature, seed=seed)
        res.calls.add(r_ref); res.log("refine", p_ref, r_ref.text)
        res.predicted = _answer_or_extract(res, llm, r_ref.text, q, temperature, seed)
        return res


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_strategy(name: str, examples: Optional[List[dict]] = None, **kw) -> Strategy:
    if name == "zero_shot":
        return ZeroShot()
    if name == "zero_shot_cot":
        return ZeroShotCoT()
    if name == "few_shot_cot":
        return FewShotCoT(examples or [], k=kw.get("k_examples", 2))
    if name == "self_consistency":
        return SelfConsistency(k=kw.get("k_samples", 5), temperature=kw.get("sc_temperature", 0.7))
    if name == "plan_and_solve":
        return PlanAndSolve()
    if name == "self_refine":
        return SelfRefine()
    if name == "tree_of_thoughts":
        return TreeOfThoughts(k=kw.get("k_branches", 3))
    if name == "graph_of_thoughts":
        return GraphOfThoughts(k=kw.get("k_thoughts", 3))
    raise ValueError(f"unknown strategy '{name}'. Available: {', '.join(STRATEGY_NAMES)}")


STRATEGY_NAMES = [
    "zero_shot", "zero_shot_cot", "few_shot_cot", "self_consistency",
    "plan_and_solve", "self_refine", "tree_of_thoughts", "graph_of_thoughts",
]
