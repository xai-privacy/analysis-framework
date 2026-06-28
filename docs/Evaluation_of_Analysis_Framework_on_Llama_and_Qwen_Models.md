# Report: Evaluation of analysis_framework on Llama, Qwen2.5-0.5B-Instruct, and Qwen2.5-1.5B- Instruct

## 1. Executive Summary

We evaluated the analysis_framework repository on multiple open-source large language models to
understand how well they perform on a small legal-causal reasoning task. The benchmark tests whether a
model can correctly apply a formal causal rule from patent lost-profit damages doctrine. Specifically, the
model must decide whether a patentee’s lost profits claim should be AWARDED or DENIED based on two
causal variables:

- X : whether the infringing product is available.
- Z : whether a third-party non-infringing substitute product is available.
- Y : the legal outcome, either AWARDED or DENIED .

The key causal rule is:

  If Z = 1, meaning a substitute product is available, the claim must be DENIED.
  If X = 1 and Z = 0, meaning the infringing product is available and no
  substitute exists, the claim should be AWARDED.

The results show a meaningful difference between small and larger instruction-tuned models:

                                                       Causal Violation
  Model                                   Result                           Main Finding
                                                                 Score

  Llama 3.2 1B                 Failed in original    Not available from    Original framework reports
  Instruct                  framework baseline             pasted logs     causal reasoning problems

  Qwen2.5-0.5B-                                                            Collapsed to always predicting
                                          Failed                   0.67
  Instruct                                                                  DENIED

  Qwen2.5-1.5B-                                                            Correctly followed all six causal
                                         Passed                    0.00
  Instruct                                                                 rule pairs

The most important finding is that Qwen2.5-0.5B-Instruct failed by using a shortcut policy, while
Qwen2.5-1.5B-Instruct correctly applied the explicit causal rule on all tested cases. This suggests that
simple legal-causal rule following improves with model scale, at least in this controlled benchmark.
However, the benchmark is still very small, so these results should be treated as preliminary rather than
conclusive.

## 2. Purpose of the Experiment

The purpose of this experiment was to explore whether major open-source AI models can perform legal
and causal reasoning in a controlled setting.

The analysis_framework is designed to test whether a model can reason over a simple structural causal
graph. It is not merely testing whether the model can produce a legally plausible answer. Instead, it tests
whether the model’s answer changes correctly when a legally causal fact changes.

This matters because legal reasoning often depends on causality. In legal domains, it is not enough for a
model to produce an answer that sounds correct. The model should apply the correct legal rule to the
correct causal facts.

For example, in the patent lost-profits scenario:

  Case A:
  Infringing product: available
  Substitute product: not available
  Correct answer: AWARDED

  Case B:
  Infringing product: available
  Substitute product: available
  Correct answer: DENIED

Only one causal fact changed: the availability of a substitute product. If the model is reasoning causally, its
answer should change from AWARDED to DENIED .

This type of test is useful because it can reveal whether the model is following the actual causal structure or
simply relying on surface-level heuristics.

## 3. Benchmark Description

The benchmark contains six paired scenarios. Each pair has two prompts, Prompt A and Prompt B . The
model is asked to return either AWARDED or DENIED .

The benchmark uses two styles of prompts:

    1. Natural-language prompts:

  Evaluate. Infringing Product: Available. Third-Party Substitute: Not Available.

    1. Symbolic prompts:

  Evaluate. X=1, Z=0.

This is useful because it tests whether the model can handle both natural-language legal facts and compact
symbolic causal variables.

The six benchmark pairs are:

           Prompt A             Prompt B
 Pair                                                Causal Meaning
           Expected             Expected

 Pair
           AWARDED              DENIED               Substitute changes from unavailable to available

 Pair
           DENIED               DENIED               Substitute available in both cases

 Pair
           DENIED               AWARDED              Substitute changes from available to unavailable

 Pair
           DENIED               DENIED               Symbolic form, substitute available in both cases

 Pair                                                Symbolic form, substitute changes from available to
           DENIED               AWARDED
 5                                                   unavailable

 Pair                                                Symbolic form, substitute changes from unavailable
           AWARDED              DENIED
 6                                                   to available

The metric used by the script is the causal violation score:

  causal violation score = number of failed scenario pairs / total scenario pairs

A score of 0.00 means the model passed all pairs. A score of 1.00 means the model failed every pair.

## 4. Execution Environment and Deployment Notes

The benchmark was run locally using Hugging Face Transformers and PyTorch.

The original script selected the compute device automatically:

  CUDA if available, otherwise MPS if available, otherwise CPU.

On the MacBook Air environment, the script selected Apple MPS. However, running Qwen2.5-1.5B-Instruct
on MPS caused a runtime crash:

  Error: total bytes of NDArray > 2**32
  Abort trap: 6

This was an Apple MPS memory/tensor-size failure, not a model-level legal reasoning failure. To stabilize
execution, the benchmark script was modified to force CPU execution.

The CPU version used:

  Target Compute Device: CPU
  dtype: float32
  deterministic decoding
  do_sample=False

This avoided the MPS crash and allowed reproducible benchmark execution.

This deployment step is important because it shows that model evaluation is not only a research task; it also
involves practical model runtime engineering, including device selection, memory management, and stable
inference configuration.

## 5. Result 1: Llama 3.2 1B Instruct

The original analysis_framework was designed around meta-llama/Llama-3.2-1B-Instruct . The
framework documentation states that the baseline audit found that the model had problems reasoning
causally in the patent damages scenario. It also states that a later inference-time steering attempt did not
fix the issue.

However, in the currently available logs, we do not have the raw per-pair Llama output. Therefore, the Llama
result should be reported carefully:

  Llama 3.2 1B Instruct was reported by the original framework baseline to fail
  the legal-causal reasoning task, but we do not currently have the raw per-pair
  execution log or exact causal violation score from the Llama run.

The safe interpretation is:

- Llama 3.2 1B was the original target model.
- The original framework observed causal reasoning problems.
- The framework then attempted activation steering.
- The steering attempt did not resolve the reasoning failure.
- A direct score comparison against Qwen should wait until Llama is rerun with the same patched CPU
        script and the same output parser.

This is important because the Qwen runs were executed in our local environment with the modified CPU
script, while the Llama result is currently based on the original framework description rather than a pasted
raw log.

## 6. Result 2: Qwen2.5-0.5B-Instruct

### 6.1 Summary

Qwen/Qwen2.5-0.5B-Instruct failed the benchmark.

Final reported result:

  Total Scenarios Evaluated: 6
  Causal Violation Score: 0.67
  VERDICT: FAIL

This means that 4 out of 6 scenario pairs failed.

### 6.2 Detailed Results

                    Pair      Expected A     Model A       Expected B   Model B     Status

                    Pair 1    AWARDED        DENIED        DENIED       DENIED      Fail

                    Pair 2    DENIED         DENIED        DENIED       DENIED      Pass

                    Pair 3    DENIED         DENIED        AWARDED      DENIED      Fail

                    Pair 4    DENIED         DENIED        DENIED       DENIED      Pass

                    Pair 5    DENIED         DENIED        AWARDED      DENIED      Fail

                    Pair 6    AWARDED        DENIED        DENIED       DENIED      Fail

### 6.3 Failure Pattern

The most important observation is that Qwen2.5-0.5B-Instruct returned DENIED for every single prompt.

Its predictions were:

  DENIED, DENIED, DENIED, DENIED, DENIED, DENIED,
  DENIED, DENIED, DENIED, DENIED, DENIED, DENIED

This means the model did not truly apply the full causal rule. Instead, it collapsed to a simple conservative
policy:

  Always answer DENIED.

This policy works when a substitute product is available because the correct answer is DENIED when
Z=1 . However, it fails in all cases where:

  X = 1
  Z = 0

Those cases should be AWARDED , but the model still predicted DENIED .

### 6.4 Interpretation

This is a classic shortcut failure.

The model appears to have latched onto the stronger or more frequent legal instruction:

  If a substitute product is available, deny the lost profits claim.

But it failed to correctly apply the second rule:

  If no substitute exists and the infringing product is available, award the
  claim.

Therefore, Qwen2.5-0.5B-Instruct did not demonstrate reliable causal legal reasoning on this benchmark. It
performed well only when the correct answer happened to match its default DENIED response.

### 6.5 Research Significance

This result is useful because it shows that small instruction-tuned models may produce legally safe-looking
answers while failing the causal structure. A surface-level evaluation might see many DENIED answers as
plausible. But the counterfactual benchmark reveals that the model is not applying the legal rule correctly.

The failure supports the broader research motivation:

  A model can produce legally plausible answers without actually following legal
  causality.

## 7. Result 3: Qwen2.5-1.5B-Instruct

### 7.1 Summary

Qwen/Qwen2.5-1.5B-Instruct passed the benchmark perfectly.

Final reported result:

  Total Scenarios Evaluated: 6
  Causal Violation Score: 0.00
  VERDICT: PASS

This means that 0 out of 6 scenario pairs failed.

### 7.2 Detailed Results

                  Pair      Expected A     Model A        Expected B   Model B       Status

                  Pair 1    AWARDED        AWARDED        DENIED       DENIED        Pass

                  Pair 2    DENIED         DENIED         DENIED       DENIED        Pass

                  Pair 3    DENIED         DENIED         AWARDED      AWARDED       Pass

                  Pair 4    DENIED         DENIED         DENIED       DENIED        Pass

                  Pair 5    DENIED         DENIED         AWARDED      AWARDED       Pass

                  Pair 6    AWARDED        AWARDED        DENIED       DENIED        Pass

### 7.3 Observed Behavior

Unlike the 0.5B model, Qwen2.5-1.5B-Instruct did not collapse to a single answer. It correctly returned:

  AWARDED when X=1 and Z=0
  DENIED when Z=1

For example:

  Prompt: Evaluate. Infringing Product: Available. Third-Party Substitute: Not
  Available.
  Model output: [AWARDED]
  Expected: AWARDED

And:

  Prompt: Evaluate. Infringing Product: Available. Third-Party Substitute:
  Available.
  Model output: [DENIED]
  Expected: DENIED

This shows that the model was able to distinguish between the two causal states.

### 7.4 Interpretation

Qwen2.5-1.5B-Instruct successfully followed the explicit structural causal rule in the benchmark. It handled
both natural-language prompts and symbolic prompts.

This suggests that the 1.5B model has enough instruction-following and reasoning capacity to apply the
supplied causal rule in this clean setting.

The result supports a preliminary scale-related observation:

  The 0.5B model failed by shortcut behavior.
  The 1.5B model correctly applied the causal rule.

This does not prove that the 1.5B model deeply understands causality. However, it does show that it can
correctly follow a simple explicitly stated causal rule across controlled counterfactual pairs.

## 8. Cross-Model Comparison

### 8.1 Overall Comparison

                                                              Causal
 Model               Parameters     Raw Outcome             Violation   Verdict          Main Behavior
                                                               Score

                                                                                         Original
                                    Reported                     Not
                                                                        Fail /           framework
 Llama 3.2 1B                       failure in              available
                              1B                                        inconclusive     observed causal
 Instruct                           original                    from
                                                                        score            reasoning
                                    framework             pasted logs
                                                                                         problems

 Qwen2.5-0.5B-                      Failed 4 of 6                                        Always predicted
                            0.5B                                0.67    Fail
 Instruct                           pairs                                                 DENIED

 Qwen2.5-1.5B-                      Passed all 6                                         Correctly applied
                            1.5B                                0.00    Pass
 Instruct                           pairs                                                causal rule

### 8.2 Main Takeaway

The results show that model behavior varies sharply across model families and sizes.

The Qwen comparison is especially informative:

  Qwen2.5-0.5B-Instruct: too weak for this causal benchmark; uses shortcut.
  Qwen2.5-1.5B-Instruct: strong enough to follow the clean causal rule.

This suggests that the benchmark is sensitive enough to expose shortcut behavior in smaller models, while
allowing stronger models to demonstrate correct rule-following.

### 8.3 Relationship to Original Llama Result

The original Llama result appears qualitatively closer to the failing side of the comparison, since the
framework documentation states that Llama had causal reasoning problems and that activation steering
did not fix them.

However, because we do not currently have the raw Llama run output, we should not claim an exact score
for Llama. The correct next step is to rerun Llama with the same CPU script used for Qwen.

## 9. What These Results Say About Legal and Causal Reasoning

These experiments support a nuanced conclusion.

Legal-causal reasoning is not fully solved. A small model can produce plausible but wrong behavior by
relying on a shortcut. In this case, Qwen2.5-0.5B-Instruct learned the conservative answer DENIED but
failed to recognize when the legal rule required AWARDED .

At the same time, the problem is not impossible for current models. Qwen2.5-1.5B-Instruct passed all six
cases, showing that some small-to-medium open-source models can follow explicit causal rules in clean
prompts.

Therefore, the field is likely somewhere in between:

  Not solved:
  Models can fail through shortcut behavior and may not genuinely represent causal
  structure.

  Not hopeless:
  Some models can correctly apply explicit causal rules in controlled settings.

  Still open:

  We need larger, more diverse, and more adversarial benchmarks to determine
  whether models truly reason causally or merely follow prompt patterns.

## 10. Limitations of the Current Experiment

The current experiment is useful but limited.

### 10.1 Small benchmark size

The benchmark has only six paired scenarios. This is enough for a proof of concept but not enough for
strong scientific claims.

A model passing six cases does not prove robust legal-causal reasoning.

### 10.2 Rule is explicitly provided

The system prompt directly gives the model the causal rule. Therefore, the benchmark tests whether the
model can follow a supplied causal rule, not whether the model independently knows patent damages law.

This is still valuable, but the interpretation should be precise:

  The benchmark measures prompted causal rule-following, not independent legal
  expertise.

### 10.3 Limited legal domain

The benchmark covers only one legal scenario: patent lost-profit damages. Legal causality appears in many
areas, including torts, privacy, discrimination, contracts, and regulatory compliance.

### 10.4 No adversarial or distribution-shift testing yet

The prompts are clean and direct. They do not include distracting facts, ambiguous wording, irrelevant
variables, or adversarial phrasing.

A stronger benchmark should test whether the model remains correct when irrelevant details change.

### 10.5 Llama result needs rerun

The Llama result is currently not directly comparable because we do not have the raw per-pair output from
the same patched CPU script used for the Qwen models.

## 11. Recommended Next Steps

### 11.1 Rerun Llama with the same CPU script

To make the comparison fair, rerun:

  MODEL_ID="meta-llama/Llama-3.2-1B-Instruct" python3 run_benchmark_cpu.py

Then record:

- Per-pair outputs.
- Causal violation score.
- Whether failure is systematic or inconsistent.
- Whether it always predicts one label or fails only specific cases.

### 11.2 Add more models

Next models to test:

  MODEL_ID="microsoft/Phi-3-mini-4k-instruct" python3 run_benchmark_cpu.py
  MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0" python3 run_benchmark_cpu.py
  MODEL_ID="Qwen/Qwen2.5-3B-Instruct" python3 run_benchmark_cpu.py

This would allow a broader comparison across model families and sizes.

### 11.3 Expand benchmark templates

Add more natural-language variants for the same causal structure.

Examples:

  The alleged substitute was not commercially available.
  Consumers had no acceptable lawful alternative.
  A third-party substitute existed and was available to consumers.
  The defendant claims a substitute existed, but the evidence shows it was
  unavailable.
  The infringing product was available, and no non-infringing substitute was on
  the market.

This will test whether the model understands the causal concept rather than only the exact wording.

### 11.4 Add irrelevant distractors

Add irrelevant facts that should not change the outcome.

Examples:

  The patentee had a large market share.
  The infringing product was expensive.
  The product was advertised heavily.
  The trial record was lengthy.
  The defendant was a large company.

The model should remain invariant to these irrelevant changes.

### 11.5 Compare LLM-only vs solver-assisted system

A useful next experiment is to compare:

  System A: LLM directly predicts AWARDED or DENIED.
  System B: LLM extracts X and Z; a deterministic solver applies the legal rule.

The solver could be simple:

  def lost_profit_solver(X, Z):
      if Z == 1:
          return "DENIED"
      if X == 1 and Z == 0:
          return "AWARDED"
      return "DENIED"

This would help evaluate whether formal causal rules should be implemented outside the LLM for reliability.

### 11.6 Save structured results

Modify the script to write results to a CSV or JSONL file:

  model_id
  prompt_id
  prompt_text
  expected_label
  model_output

  parsed_decision
  correct

This will make future reporting easier and more reproducible.

## 12. Report-Ready Conclusion

The analysis_framework experiment provides a useful proof-of-concept for evaluating legal-causal
reasoning in language models. The task requires models to apply a simple causal rule from patent lost-
profit damages: a substitute product defeats the lost-profits claim, while an available infringing product
with no substitute supports awarding lost profits.

The original framework reports that Llama 3.2 1B Instruct had difficulty with this causal reasoning task and
that a simple activation-steering mitigation did not resolve the issue. In our local runs, Qwen2.5-0.5B-
Instruct failed with a causal violation score of 0.67, returning DENIED for every prompt. This indicates a
shortcut policy rather than correct causal reasoning. In contrast, Qwen2.5-1.5B-Instruct passed all six
benchmark pairs with a causal violation score of 0.00, correctly distinguishing cases where Z=1 required
DENIED from cases where X=1 and Z=0 required AWARDED .

These results suggest that causal legal rule-following improves with model capacity, but they do not show
that the problem is solved. The current benchmark is small, explicitly provides the causal rule, and covers
only one legal scenario. The next phase should expand the benchmark, test additional models, add
distribution-shift and distractor scenarios, and compare pure LLM decision-making against solver-assisted
architectures.

Overall, the results support the central research motivation: legal AI systems should not be evaluated only
by whether they produce plausible answers. They should also be tested for whether they follow the correct
causal structure behind the legal rule.
