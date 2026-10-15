# ASPIC+ translation format for LEET-Arg (pilot, v1)

One LEET-Arg question becomes one JSON object. The object describes the **argument structure of the passage** and gives **one check per statement**. It never says whether a statement is correct. `check_aspic.py` computes that from the structure.

At test time the fine-tuned model writes this JSON from the raw question, and the same script decides the answer.

## Top-level fields

```json
{
  "id": "2022_17",
  "positions": { ... },
  "claims": { ... },
  "premises": [ ... ],
  "rules": [ ... ],
  "contraries": [ ... ],
  "undercuts": [ ... ],
  "statements": { ... }
}
```

`undercuts` and `hypotheticals` can be left out when empty. `asks_for` (`"correct"` or `"incorrect"`) is optional; the script detects it from the question wording, so use it only if the detection is wrong.

### `positions`

Who argues what. Use the labels the passage uses: `A`, `B`, `Alice`, `Hypothesis 1`, `ⓐ`, `Method A`.

```json
"positions": {
  "A": {"thesis": "c4", "summary": "Evil is not a privation, because evil has degrees and privation does not."}
}
```

- `thesis`: the claim ID of the position's main conclusion. Checks can then refer to a position by name.
- `summary`: one sentence, for human readers.
- `*` is a reserved label for content that belongs to no single position: facts stated by the narrator, facts given in a statement ("The fact that ..."), and general background knowledge. Every position may use `*` content.

### `claims`

Every proposition used anywhere gets an ID and short text.

```json
"claims": {
  "c1": {"text": "Privation does not admit of more or less.", "source": "passage"}
}
```

- IDs: `c1`, `c2`, ... for the passage; `h1`, `h2`, ... for claims that only appear inside a statement check.
- `text`: keep it close to the passage wording. Shorten, do not paraphrase freely.
- `source`, one of:
  - `passage`: stated in the passage
  - `statement`: stated in one of the statements (usually evidence: "The fact that ...")
  - `rationale`: not stated in the passage, taken from the expert rationale
  - `background`: general knowledge needed but stated nowhere
- Negations are separate claims ("Evil is a privation" and "Evil is not a privation"), linked in `contraries`. There is no negation operator.

### `premises`

Claims a position starts from.

```json
{"claim": "c3", "by": ["A", "B"], "kind": "ordinary"}
```

- `by`: a position label, a list of labels, or `"*"`.
- `kind`: `ordinary` (an assumption that can be attacked) or `axiom` (a fact that cannot be attacked: scenario facts, numbers, evidence given in a statement).

### `rules`

Steps from claims to a claim.

```json
{"id": "r2", "by": "B", "if": ["c5"], "then": "c6", "type": "defeasible", "source": "passage"}
```

- `type`: `strict` (follows with certainty: logic, definitions, a criterion the statement says to apply) or `defeasible` (holds usually: generalizations, causal or empirical inferences, analogies).
- `source`: same values as for claims.
- Rule IDs are `r1`, `r2`, ... in the passage and `ra1`, `rb1`, ... inside the check of statement (a), (b), ...

### `contraries`

Pairs of claims that cannot both be true: `[["c1", "c6"], ["c4", "c7"]]`. Symmetric.

### `undercuts`

A claim that says a rule does not apply here: `[{"claim": "c2", "rule": "rc1"}]`. Use it when a fact blocks an inference without denying its conclusion.

### How attacks follow (computed, not written)

No preferences, so every attack succeeds.

- **Rebut**: an argument concluding X attacks any argument whose *defeasible* step concludes a contrary of X.
- **Undermine**: an argument concluding X attacks any argument that uses an *ordinary* premise that is a contrary of X.
- **Undercut**: an argument concluding a claim listed in `undercuts` attacks any argument that uses that rule.
- Strict steps and axioms cannot be attacked directly, only through what they are built on.

The script then computes the grounded labelling (via PyArg when installed): each argument is IN, OUT or UNDEC. A claim is IN if some argument for it is IN, else UNDEC if some argument is UNDEC, else OUT if arguments exist, else NONE.

## `statements`

One entry per statement, keyed by its label as printed: `a`, `b`, `c` (or `1` to `5` when the choices themselves are the statements).

```json
"statements": {
  "a": {"check": {"op": "holds", "by": "B", "claim": "c7"}, "note": "optional, for human reviewers"}
}
```

`note` is ignored by the script and removed from training targets.

### Check operations

| op | fields | true when |
|---|---|---|
| `holds` | `by`, `claim`, optional `add` | the position can build an argument for the claim from its own premises and rules plus `*` content |
| `assumes` | `by`, `claim`, optional `add` | the claim is one of the position's premises |
| `status` | `claim`, `is` (list of `IN`, `OUT`, `UNDEC`, `NONE`), optional `add` | the claim's status is in the list. "X is denied" = `"is": ["OUT"]` |
| `weakens` | `claim`, `add` | after adding `add`, the claim's status drops, or stays equal but gains a new attacker that is not OUT |
| `strengthens` | `claim`, `add` | after adding `add`, the claim's status rises, or stays equal but gains a new argument that is not OUT |
| `and`, `or` | `args` (list of checks) | all / any are true |
| `not` | `arg` (one check) | the check is false |
| `unsupported` | `reason` | the statement cannot be expressed in this format; counted separately |

In every check, `claim` can also be a position label. It then means that position's thesis.

`add` is a hypothetical extension of the theory, with the same fields as the top level: `claims`, `premises`, `rules`, `contraries`, `undercuts`. It is used for "if X ...", "the fact that X ...", "if we apply B's reasoning ...". It only applies inside that check.

When several checks use the same hypothetical (for example one statement tests two criteria, and another statement reuses one of them), define it once in an optional top-level `hypotheticals` object and refer to it by name: `"add": "greater_total"`. See 2024_15.

### Typical statement patterns

| Statement wording | Check |
|---|---|
| "B argues X", "Unlike A, B claims X" | `holds` (and `not holds` for A) |
| "A assumes X", "A acknowledges X" | `assumes` |
| "Both A and B agree that X" | `and` of two `holds` |
| "The fact that E weakens ⓐ" | `weakens` with E as an axiom premise in `add`, plus the rules that connect E to a contrary of ⓐ |
| "If E, the hypothesis is strengthened" | `strengthens` with E and the supporting rules in `add` |
| "If criterion K is adopted, Alice's conclusion is denied" | `status ... is OUT`, with K as an axiom plus strict rules in `add` |
| "If we apply Bob's reasoning, Bob would agree with Charlie" | `holds` with Bob's rule applied to the new case in `add` (`by: Bob`) |

If a statement is false because a naive inference fails, write the naive inference and the reason it fails (often an undercut). Then the check shows why it is false instead of just leaving the attack out. See statement (c) of 2021_37.

## Writing rules

1. Do not write any verdict, "correct/incorrect", or the answer number anywhere.
2. Every `passage` claim must be findable in the passage. Do not invent speakers or facts.
3. Use the fewest claims and rules that make every check meaningful. Do not force a fixed number.
4. General logic that holds for every question (for example "if both use the same premise, they agree") belongs in `check_aspic.py`, not in the translation.
5. Mark anything not in the passage as `rationale` or `background`. The share of such elements tells us how much the model must supply on its own at test time.

## Frozen instruction for training and inference

Use this exact text as the system message in the training data, for the base model, and for the fine-tuned model. Do not edit it after the first run.

```
Translate the LEET argument passage into ASPIC+ JSON with these fields: positions, claims, premises, rules, contraries, undercuts, hypotheticals, statements. Keep claim text close to the passage wording. Mark rules as strict or defeasible. For each statement, write one check (holds, assumes, status, weakens, strengthens, and, or, not). Do not decide whether any statement is correct. Output only the JSON.
```

User message: the passage, question and statements, without the ① to ⑤ choice list (same as the first fine-tuning run). Assistant message: the translation JSON with `note` fields removed.
