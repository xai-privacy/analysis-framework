# Judge Selection for Evaluating Legal Reasoning

## 1. Goal

We want to evaluate the reasoning mistakes made by the models in our LEETArg and MBE experiments.

The selected candidate models are:

- Llama-3.2-3B-Instruct
- DeepSeek-R1-Distill-Llama-8B
- Qwen3-8B
- Mistral-Small 24B
- gpt-oss-20b or gpt-oss-120b
- Claude Sonnet 5.5

The main question is:

> If one of these models gives an answer and a reasoning explanation, which model should judge whether the reasoning is correct?

The main conclusion is:

> We should not use one single frontier model as the only judge.

A better approach is to use a judge from a different model family, use more than one judge where possible, give the judge the gold answer and a fixed rubric, and validate the judge against human labels.

---

## 2. What does not need an LLM judge?

We should not use an LLM judge for things that we can calculate directly from the benchmark.

### MBE

The benchmark already gives the correct answer option.

So we can calculate:

- whether the model selected A, B, C, or D correctly
- overall answer accuracy
- parse failures
- format failures

No judge is needed for these.

### LEETArg

LEETArg also gives gold labels.

We can calculate:

- whether the final answer is correct
- whether each statement is CORRECT or INCORRECT
- which statement the model got wrong
- parse rate
- format failures

Again, no LLM judge is needed for these.

### What does need a judge?

A judge is useful when we want to know **why** the model made a reasoning mistake.

For example:

- Did the model use the wrong legal rule?
- Did it understand the facts incorrectly?
- Did it apply the correct rule incorrectly?
- Did it ignore an exception?
- Did it contradict itself?
- Did it invent a legal rule?
- Did it reach the right answer using bad reasoning?

This is the main role of the LLM judge in our study.

---

## 3. Why one frontier model is not enough

A frontier model can still make judging mistakes.

For example, an LLM judge may:

- prefer answers written in a style similar to its own
- prefer longer or more confident answers
- reward good formatting even when the reasoning is wrong
- miss subtle reasoning errors
- disagree with another strong judge
- behave differently on different model families

This is especially important in our project because we are studying the difference between:

> good-looking reasoning

and

> actually correct reasoning.

If the judge rewards a polished explanation, it could repeat the same mistake we are trying to study.

---

## 4. Avoid using the same model family as the judge

A judge should preferably not grade a model from its own family.

For our candidate models:

| Candidate model | Family | Judge family to avoid |
|---|---|---|
| Llama-3.2-3B-Instruct | Meta | Llama-based judge |
| DeepSeek-R1-Distill-Llama-8B | DeepSeek + Llama lineage | DeepSeek and Llama judges |
| Qwen3-8B | Alibaba | Qwen judge |
| Mistral-Small 24B | Mistral | Mistral judge |
| gpt-oss-20b / 120b | OpenAI | OpenAI judge |
| Claude Sonnet 5.5 | Anthropic | Claude judge |

The most useful family for a common primary judge is therefore **Google Gemini**, because Gemini is not one of the candidate model families.

---

## 5. Recommended judge setup

### Primary judge

Use a current strong **Gemini Pro / Gemini reasoning model** as the primary judge.

Why?

- It is from a different family than all selected candidate models.
- We can use the same primary judge for every candidate.
- This makes comparisons across candidate models more consistent.

### Secondary judge for most models

Use **Claude Opus 5.5** as a second judge for:

- Llama
- DeepSeek-R1-Distill
- Qwen3
- Mistral
- gpt-oss

But do **not** use Claude Opus as the main judge for Claude Sonnet 5.5.

### Secondary judge for Claude Sonnet 5.5

For Claude Sonnet 5.5, use a strong **OpenAI frontier model** as the second judge.

So the basic design is:

```text
Candidate model
      |
      v
Gemini judge
      +
Second judge from another family
      |
      v
Agreement / disagreement
```

For Claude Sonnet 5.5:

```text
Claude Sonnet 5.5
        |
        +----> Gemini judge
        |
        +----> OpenAI frontier judge
```

For Llama, Qwen, Mistral, DeepSeek, or gpt-oss:

```text
Candidate
   |
   +----> Gemini judge
   |
   +----> Claude Opus 5.5
```

---

## 6. What should the judge see?

The judge should not be asked:

> "Is this reasoning good?"

That is too vague.

Instead, give the judge:

1. the original question
2. the answer choices or statements
3. the gold answer
4. the gold per-statement labels for LEETArg
5. the candidate model's final answer
6. the candidate model's reasoning
7. the reasoning-error taxonomy
8. a fixed judging rubric

The judge should already know which answer or statement is wrong.

Then its job is only to classify **why** it is wrong.

This makes judging easier and more reliable.

---

## 7. Example judge task

Suppose the model chose the wrong MBE answer.

The judge could receive:

```text
Question:
...

Gold answer:
D

Candidate answer:
B

Candidate reasoning:
...

Task:
Identify the main reasoning error.

Allowed categories:
- WRONG_RULE
- WRONG_FACT_INTERPRETATION
- WRONG_RULE_APPLICATION
- IGNORED_EXCEPTION
- INVALID_INFERENCE
- INTERNAL_CONTRADICTION
- FABRICATED_RULE
- UNSUPPORTED_ASSERTION
- OTHER_UNCLEAR
```

The judge should return structured output such as:

```json
{
  "error_type": "WRONG_RULE_APPLICATION",
  "short_reason": "The model states the correct rule but applies it incorrectly to the facts."
}
```

This is better than asking for a general score such as 7/10.

---

## 8. Judge one thing at a time

The judge should answer narrow questions.

Bad question:

> Rate the overall reasoning quality from 1 to 10.

Better questions:

- Is the legal rule correct?
- Is the fact analysis correct?
- Is the rule applied correctly?
- Does the conclusion follow from the reasoning?
- Is there an unsupported or fabricated claim?
- What is the main reasoning-error category?

Narrow questions are easier to evaluate and easier to compare.

---

## 9. Right answer, wrong reasoning

A model may select the correct final answer but still use bad reasoning.

For example:

```text
Gold answer: D
Model answer: D
```

But the explanation may contain:

- a wrong legal rule
- a false factual claim
- a contradiction
- an invalid reasoning step

Therefore, answer accuracy alone is not enough.

For a sample of correct answers, the judge should also check whether the reasoning is actually supported by the gold explanation.

This helps identify:

> right answer, wrong reasoning

which is one of the main questions in our research.

---

## 10. Blind the judge to model identity

The judge should not know which model produced the response.

Do not say:

```text
This response was generated by Claude Sonnet 5.5.
```

Instead say:

```text
Candidate response:
...
```

Also remove model-specific formatting where possible:

- `<think>` tags
- special analysis markers
- model-family names
- system-prompt text
- unnecessary Markdown differences

This reduces bias.

---

## 11. Do not reward style or length

The judge prompt should clearly say:

> Do not reward the response for being longer, more confident, more detailed, or better formatted. Judge only whether the legal reasoning is correct.

This is important because a model can produce perfect IRAC-style formatting while still applying the wrong rule.

Formatting should therefore be measured separately from reasoning correctness.

---

## 12. One trace at a time

Judge each answer separately.

Do not mainly use:

> Which answer is better, A or B?

Pairwise judging can introduce position bias.

Instead use:

> Does this response contain a reasoning error? If yes, classify the error.

If pairwise comparison is ever needed, run both orders:

```text
A vs B
B vs A
```

and only trust the preference if the result stays the same.

---

## 13. Use more than one judge

The safest design is:

```text
Primary judge
+
Secondary judge
```

If both agree:

```text
accept the category
```

If they disagree:

```text
send to human review
```

or mark:

```text
UNRESOLVED
```

Do not silently choose one judge when they disagree.

We should report the disagreement rate.

---

## 14. Human validation is still necessary

LLM judges are not ground truth.

We should manually label a subset of the data.

A good design is:

```text
All errors
   |
   v
LLM judge(s)

plus

Stratified sample
   |
   v
Human reviewers
```

The human sample should include different:

- candidate models
- benchmarks
- error categories
- easy and difficult examples

Ideally, two people independently label the sample.

Then compare:

- human vs human
- Gemini vs human
- second judge vs human
- Gemini vs second judge

Useful measures include:

- Cohen's kappa
- macro-F1
- simple agreement percentage

---

## 15. Validate each judge separately

A judge may work well for one model and poorly for another.

For example:

```text
Gemini vs human on Llama errors: high agreement
Gemini vs human on Claude Sonnet errors: lower agreement
```

Therefore judge accuracy should be reported by candidate model, not only as one overall number.

---

## 16. Different models expose different amounts of reasoning

Some models expose long reasoning traces.

For example:

- DeepSeek-R1-Distill can produce long reasoning
- Qwen3 thinking mode can produce reasoning
- gpt-oss can expose an analysis/reasoning channel

Other models may expose only a final written explanation.

Claude's visible thinking also depends on the API configuration.

Therefore, we should not assume that every model gives the same kind of reasoning trace.

Two possible solutions are:

### Option A

Divide error categories into:

- answer-level errors
- trace-dependent errors

Only compare trace-dependent categories across models that expose comparable reasoning.

### Option B

Use the final written rationale from every model as the common artifact.

Then analyze raw reasoning traces separately for models that provide them.

For the main cross-model study, Option B is simpler and fairer.

---

## 17. Recommended evaluation workflow

```text
Candidate model generates answer + rationale
                    |
                    v
Program checks final answer against gold
                    |
                    v
Correct or incorrect?
        |
        +----------------------+
        |                      |
      wrong                  correct
        |                      |
        v                      v
Judge classifies         Sample some for
reasoning error          right-answer-wrong-
                         reasoning analysis
        |
        v
Second judge checks
        |
        v
agreement?
   /          \
 yes           no
  |             |
  v             v
accept       human review
```

This keeps objective scoring separate from subjective reasoning evaluation.

---

## 18. Recommended setup for this study

### Candidate models

- Llama-3.2-3B-Instruct
- DeepSeek-R1-Distill-Llama-8B
- Qwen3-8B
- Mistral-Small 24B
- gpt-oss-20b or gpt-oss-120b
- Claude Sonnet 5.5

### Primary judge

**Current strong Gemini reasoning model**

Use it for every candidate.

### Secondary judge

For Llama, DeepSeek, Qwen, Mistral, and gpt-oss:

**Claude Opus 5.5**

For Claude Sonnet 5.5:

**Current strong OpenAI frontier model**

### Human validation

Use a stratified subset with preferably two human reviewers.

---

## 19. What should be reported in the paper?

At minimum, report:

1. exact judge model/version
2. judge prompt
3. temperature and reasoning settings
4. gold information given to the judge
5. reasoning-error taxonomy
6. number of items judged
7. inter-judge agreement
8. judge-vs-human agreement
9. human-human agreement
10. judge parse failures
11. `OTHER / UNCLEAR` rate
12. disagreement rate
13. whether candidate model identity was hidden
14. how visible reasoning traces differed across candidate models

This makes the evaluation reproducible.

---

## 20. Main recommendation

The main idea is:

> Use normal code to check whether the answer is correct. Do not ask an LLM judge to do something the benchmark already tells us.

Then:

> Use LLM judges only to understand why the reasoning failed.

For those reasoning judgments:

> Use a strong judge from a different model family, preferably two judges, give them the gold answer and a fixed taxonomy, hide the candidate model identity, and validate the judges against human labels.

For our selected models, the simplest design is:

```text
Primary judge:
Gemini

Second judge:
Claude Opus for non-Claude candidates

Second judge for Claude Sonnet:
OpenAI frontier model

Disagreement:
Human review
```

This is stronger than simply choosing one frontier model and treating its judgment as ground truth.

---

## 21. References

The judge-selection design is motivated by work on LLM-as-a-judge reliability, self-preference, judge panels, and reasoning-trace evaluation, including:

- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- Panickssery, Bowman & Feng (2024), *LLM Evaluators Recognize and Favor Their Own Generations*
- Verga et al. (2024), *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models*
- Tan et al. (2025), *JudgeBench: A Benchmark for Evaluating LLM-Based Judges*
- Zheng et al. (2025), *ProcessBench: Identifying Process Errors in Mathematical Reasoning*
- Kenton et al. (2024), *On Scalable Oversight with Weak LLMs Judging Strong LLMs*
- Lee & Hockenmaier (2025), *Evaluating Step-by-step Reasoning Traces: A Survey*
- *Reasoning Jury: Multi-Model Consensus for Evaluating Reasoning Traces* (2026)

The exact judge model IDs and versions should be pinned when the experiment is run, because frontier model names and versions can change.
