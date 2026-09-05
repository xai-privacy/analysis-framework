## Contents

## File: Purpose

`run_01_zero_shot.py` … `run_08_graph_of_thoughts.py`: One runnable script per technique, basically just calls common with the correct arguments.
`strategies.py` : Prompts and techniques for each strategy. This is where the prompts are created.
`common.py` : Things that pertain to all strategies, like the calls to the model and the shared section of each prompt.
`llm.py` : Sets up Ollama, parses the answer, and does calculation on token use.
`scoring.py` : Computes accuracy, weighted accuracy, robustness, McNemar, and base-rate checks.
`score.py` : Generates comparison table across techniques.
`dataset.py` : Loads the dataset and splits between test and dev. Currently, the dev/test split is not the same as the one in [issue 24](https://github.com/xai-privacy/analysis-framework/issues/24#issuecomment-5530569978).
`check_env.py` : Diagnostic for enviroment/running.
`requirements.txt` The single dependency (`requests`).
`data/LEET_Arg_Questions.json` : The benchmark.
`tests/test_scoring.py` : Tests scoring with 22 offline tests (no model calls).
`results/*.json` : Model outputs and scores, with one file per technique (`<name>__<split>.json`).

---

## How to run

Requires Python ≥ 3.9 and a running Ollama server. No virtualenv is required, only `requests` is needed (requirements.txt)

In powershell:

```
# 1. One-time setup
ollama pull qwen2.5:3b
py -m pip install requests
py check_env.py                                   # expected "All good." output

# 2. Smoke test (mock model, ~1s, proves the plumbing)
py run_01_zero_shot.py --dry-run --limit 3

# 3. Real runs — baseline first, everything is measured against it
py run_01_zero_shot.py --split dev --runs 3 --yes
py run_02_zero_shot_cot.py --split dev --runs 3 --yes
py score.py                                        # compare everything run so far

# Offline tests, no model needed
py tests\test_scoring.py                           # 22/22 passed
```

After each script is ran, it writes `results/<technique>__<split>.json` and prints its own summary
Useful flags (common to all eight): `--model --host --split --runs --limit --seed
--temperature --timeout --num-predict --outdir --dry-run --quiet --yes`. Technique-
specific: `--k-examples` (03), `--k-samples`/`--sc-temperature` (04), `--k-branches`
(07), `--k-thoughts` (08).

Split. Dev = 2021–2023 (60 problems) for iteration; test = 2024–2025 (37 problems),
held out. Run `--split test` only once, at the very end — comparing eight techniques and
reporting the winner on the same data would be optimistically biased. Again, this isn't the same exact testing split as in [issue 24](https://github.com/xai-privacy/analysis-framework/issues/24#issuecomment-5530569978).

---

## Techniques and sources

Each technique is a "Strategy" subclass in `strategies.py`. The prompts are handwritten adaptations of the prompt from the associated paper. The shared block is an adapatation of the block from LEET-Arg paper's "Problem Solving Prompt," since the original prompt is in Korean, and not english. Each prompt is question agnostic and the same template is formulaically generated the same way, regardless of the question. There is not per-question tuning. Below is the origin of each technique and a brief description of it:

1. Zero-shot (baseline) - LEET-Arg's base prompt
2. Zero-shot CoT - Kojima et al. 2022, [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) - Adds "Let's think step by step. Work through each option in turn…" in one call.
3. Few-shot CoT - Wei et al. 2022, [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) - For these tests, k=2 worked examples from the **dev** split using the dataset's expert rationales.
4. Self-consistency - Wang et al. 2022, [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) - For these tests, k=5 CoT paths at temperature=0.7, then a majority vote on the correct answer.
5. Plan-and-Solve - Wang et al. 2023, [arXiv:2305.04091](https://arxiv.org/abs/2305.04091) - Adds "Devise a plan… identify premises… then carry it out," in one call.
6. Self-Refine - Madaan et al. 2023, [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) - Performs three different calls, first obtaining the answer, then asks the model to "revise," then to "finalize."
7. Tree-of-Thoughts - Yao et al. 2023, [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) - For these tests\*, branch into k=3 lines, then score each branch, then refine whichever performed the best.
8. Graph-of-Thoughts - Besta et al. 2024, [arXiv:2308.09687](https://arxiv.org/abs/2308.09687) - For these tests\*, generate k=3 thoughts, then aggregate them all and refine.

\* For GoT and ToT, the prompt and technique used is technically a slightly smaller version of the real technique. A full ToT, for example, would generate a tree with much much more levels, and a full GoT would aggregate numerous times. To do a full GoT or ToT outright would certainly have a negative impact (even this smaller one did), so that the is prompting technique I ended up testing.

## How "accuracy" is defined

All scoring is done by `scoring.py`

Each LEET-Arg problem has one gold option, so the `Accuracy` number is simply the (records where `predicted == gold`) / (total records). This ends up computing the accuracy of the final choice, but has contains nothing about the accuracy of the reasoning given. It is saved in the results, but does not get numerically evaluated anywhere (although we did discuss numerous ways to do this in our meeting, nothing ended up as a concrete answer). The `weighted accuracy` is also calculated, where a similar calculation to `accuracy` is performed but each question is weighted by the amount of questions in the same category and domain. So, questions that test rare skills or ask about uncommon domains are worth more.

---

## Results (dev split, 60 questions)

The baseline, zero-shot CoT, plan-and-solve and self-consistency were run over 3 runs, with 180 records each, and the rest were over 1 run. The cost is tokens/question relative to baseline.

Technique - Runs - Accuracy - Weighted - SD - Cost - McNemar (gain/loss)

1. Zero-shot CoT | 3 | 0.294 | 0.285 | 0.048 | 1.1x | +10 / -5, p=0.30
2. Plan-and-Solve | 3 | 0.250 | 0.231 | 0.077 | 1.2x | +6 / -4, p=0.75
3. Self-consistency | 3 | 0.250 | 0.240 | 0.106 | 5.4x | +5 / -4, p=1.0
4. Zero-shot (baseline) | 3 | 0.217 | 0.221 | 0.135 | 1.0x | -
5. Few-shot CoT | 1 | 0.217 | 0.251 | - | 2.6x | +5 / -5, p=1.0
6. Tree-of-Thoughts | 1 | 0.217 | 0.217 | - | 4.5x | +6 / -6, p=1.0
7. Self-Refine | 1 | 0.200 | 0.238 | - | 4.3x | +6 / -7, p=1.0
8. Graph-of-Thoughts | 1 | 0.183 | 0.151 | - | 8.3x | +7 / -9, p=0.80

Zero shot CoT performs the best. The baseline scored 21.7% accuracy, and zero shot CoT scored 29.4%. This technique is also not particularly expensive. It only used 1.1x the tokens the baseline did. In addition to the best accuracy, it also had a standard deviation of .048 when comparing across the runs. It seems as though CoT improved the model in both variance and accuracy.

## Graph of Thoughts, Tree of Thoughts, and Self Refine, all scored near or slightly below the baseline. This is unfortunate, given that they use 8.3, 4.3, and 4.5 the tokens compared to the baseline, respectively. It seems as though for a small 3b model (see below), these expensive prompting techniques are just not worth it.

## Why some techniques may not help

The most important constraint is the output-token budget. The paper standardised max output
at 3000 tokens over a frontier-model API. On CPU a 3B model runs at ~5-10
tokens/sec, so 3000 tokens is 5-10 minutes per call. The default here is
`--num-predict 800` (`llm.py`). This is enough for one reasoned answer, but it has a pretty big impact on what techniques are viable.
Each of the multi-stage techniques max out the budget. Self-refine, ToT, and GoT all reach the 800 token cap per call, and so the model has to fallback on a less optimized response. In their respective papers, it is shown that ToT and GoT rely on the repeated iteration to achieve any noticable gains. Since the version that we are using in these tests are reduced to one or two iterations, max, the gain from these techniques is simply not obeserved. Similarly, the papers show that we should actually expect a GREATER gain in performance from CoT. While zero-shot CoT was measured to be the best, the 3b parameter model doesn't actually get the gains that the paper suggests.

---
