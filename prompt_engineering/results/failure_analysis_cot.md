# CoT failure analysis — LEET-Arg, qwen2.5:3b

Manual deep-dive requested on issue #19: take a question where zero-shot CoT fails,
understand it fully, locate where the model's reasoning goes wrong, and judge whether
the failure is an artifact of our prompt formulation or a genuine capability limit.

## Headline: the errors are not random — CoT over-selects "combination" options

The LEET-Arg items in this set share a fixed shape. Three statements (a), (b), (c) each
claim something about the debate, and the five options are *combinations* of them:

| Option | ① | ② | ③ | ④ | ⑤ |
|---|---|---|---|---|---|
| meaning | one statement correct | one statement correct | two correct | two correct | all three correct |
| example (2021_02) | (a) | (b) | (a),(c) | (b),(c) | (a),(b),(c) |

So the option number is a direct function of *how many* statements the model judges
correct. That makes the answer distribution diagnostic:

| Choice | CoT predicted (180 records) | Gold (180 records) |
|---|---|---|
| ① | 19 | 36 |
| ② | **6** | **51** |
| ③ | 44 | 42 |
| ④ | **83** | **27** |
| ⑤ | 28 | 24 |

The model picks ④ three times too often and picks ② almost never — while ② is the single
most common gold answer. Since ② means "exactly one statement is correct" and ④ means "two
are correct," the model is **systematically judging more statements 'correct' than it should.**
It over-credits statements, which inflates it toward the two-correct and all-correct options.
This is a false-positive bias on statement validity, and it is the mechanism behind most CoT
failures here.

## Worked example: 2021_02 (gold ②, CoT answered ④)

### The question, understood fully
Three people debate whether obscene materials can be protected under a Copyright Act.
- **Alice**: creativity is the *only* condition; recognition of a work should be value-neutral (so obscene works can be copyrighted).
- **Bob**: protecting obscene materials rewards "dirty hands" and violates legal unity/fairness, so legality is a requirement of authorship (obscene works cannot be copyrighted).
- **Charlie**: clearly harmful materials (child pornography, filmed rape) are not works, but *other* obscene materials can be recognized, to minimize infringement on expression/property rights — a compromise keyed to **social harmfulness**.

The three statements each assert a *presupposition* of a debater:
- **(a)** Alice presupposes creativity *cannot* be acknowledged for obscene expression.
- **(b)** Bob does not regard illegal works (e.g. murals in prohibited locations, National-Security-Act-violating incitement) as protectable.
- **(c)** Charlie presupposes the legal evaluation of *obscenity* varies by purpose, method, and audience of distribution.

**Gold = ② (only (b) is correct).** Why the others are wrong, per the expert rationale:
- (a) is backwards — Alice presupposes creativity *can* be acknowledged for obscene works (that's the whole point of value-neutrality).
- (c) is a subtle misattribution. Charlie's criterion is **social harmfulness**, and it governs the evaluation of **copyrightability/authorship**, *not* the evaluation of **obscenity itself**. Charlie's argument holds even if the obscenity judgment never changes with purpose/method/audience. So (c) attaches the varying evaluation to the wrong object.

### Where the model went wrong
The model handled (a) and (b) correctly. It failed on (c):

> **Statement (c):** Charlie suggests that obscenity should be evaluated based on the purpose,
> method, and audience of distribution… This aligns with Charlie's argument… Therefore, (c) is correct.

That is a surface-level match: Charlie *does* talk about treating obscene materials differently,
so a statement mentioning "purpose, method, audience" *looks* aligned. The model never checks the
finer question the item is actually testing — *what* is being evaluated (copyrightability, not
obscenity). It credits (c), reaches "(b) and (c) correct," and outputs ④ instead of ②.

This is the general pattern in miniature: the distractor statement is a **near-miss paraphrase**
— right topic, subtly wrong object/scope — and the model accepts it because it reads as
topically consistent.

## Is this our CoT prompt's fault, or the model's?

Both contribute, and it's worth separating them because the reviewer's concern is that our CoT
implementation might be idiosyncratic.

**Mostly capability.** The core miss on (c) is a substantive legal-reasoning distinction
(evaluation-of-obscenity vs. evaluation-of-copyrightability). A standard Kojima-style "let's think
step by step" would almost certainly make the same error; nothing about our wording causes it.

**But our prompt does two things that plausibly amplify the false-positive bias**, and both are
fixable *generically* (no question-specific tweaking):
1. It frames the sub-task as *"for each option, state whether it is a correct analysis."* Asking
   "is this correct?" is an acceptance-biased framing — it invites confirmation rather than
   disconfirmation. It does not push the model to actively find the flaw.
2. It says "work through each **option**," but the atomic judgments are over **statements** (a/b/c),
   and the option is a downstream mapping. The mismatch may blur the statement-level checks.

## Proposed next test (generic, question-agnostic — per the issue's rule)

Two changes, both uniform across every item and neither hand-crafted to any question:

- **Disconfirmation framing:** "For each statement, first try to find a reason it *misrepresents*
  what the speaker actually presupposes. Mark it correct only if you cannot find such a reason."
- **Explicit statement→option mapping:** "List which of (a), (b), (c) survive, then choose the
  option matching exactly that set."

Prediction: if the ④-bias shrinks and accuracy rises, part of the failure was our acceptance-biased
formulation; if it persists, it's a capability limit of the 3B model. Either result directly answers
whether our baseline CoT is "standard enough." This should be run as a *new* named strategy (e.g.
`zero_shot_cot_disconfirm`) alongside the existing one, on dev, before moving to the 14B model — so
the two are compared on identical items.

## Notes for the write-up
- Accuracy here is **question-level, final-answer-only, computed by exact match** (predicted option
  == gold option) — no statement-level scoring and no model-as-judge. See the README/`scoring.py`.
- The reasoning text shown above is saved verbatim in each record's `trace`, so any of these
  failures can be re-inspected with `python score.py --show-trace N`.
- Questions analyzed in depth so far: **2021_02**. (Extend this list as more are examined.)
