"""Offline tests for parsing and scoring. No model or Ollama server needed.

    python tests/test_scoring.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from dataset import get_split, load_all
from llm import MockLLM, parse_answer
from scoring import (accuracy, category_of, compute_weights, domain_of,
                     mcnemar, robustness, weighted_accuracy)
from strategies import STRATEGY_NAMES, build_strategy


def _recs(pattern, qid="q1"):
    """Build records from a correctness pattern like [1,1,0]."""
    return [{"question_id": qid, "correct": bool(c), "predicted": str(c),
             "category": "c"} for c in pattern]


# --- parsing ---------------------------------------------------------------

def test_parse_common_formats():
    for text, want in [("Answer: 3", "3"), ("Answer-4.", "4"), ("**Answer:** 2", "2"),
                       ("Final Answer: 5", "5"), ("Answer ③", "3"), ("I choose ⑤", "5")]:
        assert parse_answer(text) == want, (text, parse_answer(text))


def test_parse_takes_last_answer():
    """CoT often revises mid-stream; the committed answer is the last one."""
    assert parse_answer("Answer: 2\nOn reflection, Answer: 4") == "4"


def test_parse_returns_none_when_absent():
    for text in ["", "no choice here", "Answer: 9"]:
        assert parse_answer(text) is None, text


# --- SD formula, verified against the paper's own reported values ----------

def test_sd_matches_paper_reported_values():
    """The paper reports max SD 0.548 and 0.447. Those are sample SDs (ddof=1)
    of the 0/1 correctness vector for 3-of-5 and 4-of-5 correct. ddof=0 would
    give 0.490 and 0.400, so this pins down the intended formula."""
    assert abs(robustness(_recs([1, 1, 1, 0, 0]))["max_sd"] - 0.5477) < 0.001
    assert abs(robustness(_recs([1, 1, 1, 1, 0]))["max_sd"] - 0.4472) < 0.001


def test_robustness_counts():
    recs = _recs([1, 1, 1, 1, 1], "a") + _recs([0, 0, 0, 0, 0], "b") + _recs([1, 0, 1, 0, 1], "c")
    r = robustness(recs)
    assert r["n_problems"] == 3 and r["n_runs"] == 5
    assert r["perfect_performance"] == 1
    assert r["consistent_errors"] == 1
    assert r["high_variability_cases"] == 1   # only 'c' has SD > 0.4


def test_unique_answers_counts_unparsed_as_distinct():
    recs = [{"question_id": "q", "correct": False, "predicted": p, "category": "c"}
            for p in ["1", "2", None, None]]
    assert robustness(recs)["avg_unique_answers"] == 3.0


# --- weighting -------------------------------------------------------------

def test_category_merge_reproduces_paper_counts():
    """Merging the JSON's two 'Argument Evaluation' categories yields the
    paper's stated 43 / 15 / 39 split."""
    c = Counter(category_of(q) for q in load_all())
    assert c["Argumentation and Rebuttal"] == 43
    assert c["Argument Analysis"] == 15
    assert c["Argument Evaluation & Problem Solving"] == 39


def test_missing_domain_group_has_no_argument_analysis():
    """The paper says Argument Analysis has n=0 in Science and Technology, which
    is what identifies the null-domain items as that domain."""
    st = [q for q in load_all() if domain_of(q) == "Science and Technology"]
    assert len(st) == 10
    assert not any(category_of(q) == "Argument Analysis" for q in st)


def test_weights_normalise_to_mean_one():
    qs = load_all()
    w = compute_weights(qs)
    assert abs(sum(w.values()) / len(w) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_weighted_accuracy_favours_rare_combinations():
    """Getting a rare-combination item right should outweigh a common one."""
    qs = load_all()
    w = compute_weights(qs)
    rare = max(w, key=lambda k: w[k])
    common = min(w, key=lambda k: w[k])
    only_rare = [{"question_id": rare, "correct": True, "predicted": "1", "category": "c"},
                 {"question_id": common, "correct": False, "predicted": "1", "category": "c"}]
    only_common = [{"question_id": rare, "correct": False, "predicted": "1", "category": "c"},
                   {"question_id": common, "correct": True, "predicted": "1", "category": "c"}]
    assert accuracy(only_rare) == accuracy(only_common) == 0.5
    assert weighted_accuracy(only_rare, w) > weighted_accuracy(only_common, w)


# --- split -----------------------------------------------------------------

def test_split_is_disjoint_and_complete():
    dev, test, allq = get_split("dev"), get_split("test"), load_all()
    assert len(dev) == 60 and len(test) == 37 and len(dev) + len(test) == len(allq) == 97
    assert not (set(q["id"] for q in dev) & set(q["id"] for q in test))


# --- McNemar ---------------------------------------------------------------

def test_mcnemar_detects_clear_difference():
    a = [{"question_id": f"q{i}", "correct": True, "predicted": "1"} for i in range(20)]
    b = [{"question_id": f"q{i}", "correct": False, "predicted": "2"} for i in range(20)]
    m = mcnemar(a, b)
    assert m["a_only_correct"] == 20 and m["b_only_correct"] == 0
    assert m["significant_at_05"]


def test_mcnemar_identical_is_not_significant():
    a = [{"question_id": f"q{i}", "correct": i % 2 == 0, "predicted": "1"} for i in range(20)]
    m = mcnemar(a, list(a))
    assert m["p_value"] == 1.0 and not m["significant_at_05"]


# --- strategies ------------------------------------------------------------

def test_all_strategies_run_and_account_for_compute():
    q = load_all()[0]
    examples = get_split("dev")[:3]
    for name in STRATEGY_NAMES:
        llm = MockLLM(["Reasoning here.\nAnswer: 3", "Best: 1", "A partial thought."])
        strat = build_strategy(name, examples=examples)
        out = strat.run(q, llm)
        assert out.calls.n_calls >= 1, name
        assert out.calls.prompt_tokens > 0, name
        assert out.trace, name


def test_multi_call_strategies_cost_more():
    """Base call cost, using responses that always parse so the answer-extraction
    fallback never fires and the counts are exact."""
    q = load_all()[0]
    def cost(name):
        llm = MockLLM(["Reasoning.\nAnswer: 3", "Best: 1\nAnswer: 1", "Thought.\nAnswer: 2"])
        return build_strategy(name, examples=[]).run(q, llm).calls.n_calls
    assert cost("zero_shot") == 1
    assert cost("zero_shot_cot") == 1
    assert cost("plan_and_solve") == 1
    assert cost("self_refine") == 3
    assert cost("self_consistency") == 5
    assert cost("tree_of_thoughts") == 5
    assert cost("graph_of_thoughts") == 5


def test_answer_extraction_rescues_unparseable_response():
    """A response that reasons but never prints 'Answer: N' -- the usual result of
    hitting the output budget -- must trigger one short extraction call rather than
    being silently scored incorrect."""
    q = load_all()[0]
    llm = MockLLM(["I was working through the options and ran out of room mid-sen",
                   "Answer: 4"])
    out = build_strategy("zero_shot").run(q, llm)
    assert [t["stage"] for t in out.trace] == ["answer", "extract_answer"]
    assert out.calls.n_calls == 2
    assert out.predicted == "4"


def test_extraction_not_triggered_when_answer_present():
    q = load_all()[0]
    llm = MockLLM(["Clear reasoning.\nAnswer: 2"])
    out = build_strategy("zero_shot").run(q, llm)
    assert out.calls.n_calls == 1
    assert not any(t["stage"] == "extract_answer" for t in out.trace)


def test_num_predict_default_is_cpu_viable():
    """3000 tokens at CPU speed is 5-10 minutes per call, which guarantees timeouts."""
    from llm import DEFAULT_NUM_PREDICT
    assert DEFAULT_NUM_PREDICT <= 1200


def test_few_shot_never_uses_the_question_as_its_own_example():
    q = get_split("dev")[0]
    strat = build_strategy("few_shot_cot", examples=get_split("dev"), k_examples=2)
    prompt = strat.build_prompt(q)
    # The gold answer line for this item must not appear in the demonstrations.
    assert prompt.count(q["original_question"]) == 1


def test_self_consistency_majority_vote():
    q = load_all()[0]
    llm = MockLLM(["Answer: 4", "Answer: 4", "Answer: 1", "Answer: 4", "Answer: 2"])
    out = build_strategy("self_consistency", k_samples=5).run(q, llm)
    assert out.predicted == "4"


def test_one_run_script_per_strategy():
    """Each strategy in the registry must have exactly one run_NN_*.py script,
    and each script must reference its strategy by name."""
    import pathlib
    root = pathlib.Path(__file__).parent.parent
    scripts = sorted(root.glob("run_0*.py"))
    assert len(scripts) == len(STRATEGY_NAMES), \
        f"{len(scripts)} scripts vs {len(STRATEGY_NAMES)} strategies"
    named = set()
    for sc in scripts:
        text = sc.read_text(encoding="utf-8")
        hits = [n for n in STRATEGY_NAMES if f'execute("{n}"' in text]
        assert len(hits) == 1, f"{sc.name} should execute exactly one strategy, got {hits}"
        named.add(hits[0])
    assert named == set(STRATEGY_NAMES), f"missing: {set(STRATEGY_NAMES) - named}"


def test_default_model_is_qwen_3b():
    from common import DEFAULT_MODEL
    assert DEFAULT_MODEL == "qwen2.5:3b"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
