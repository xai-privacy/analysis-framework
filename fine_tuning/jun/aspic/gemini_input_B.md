# ASPIC+ translation task

## Part 1. Format


One LEET-Arg question becomes one JSON object. The object describes the **argument structure of the passage** and gives **one check per statement**. It never says whether a statement is correct. `check_aspic.py` computes that from the structure.

At test time the fine-tuned model writes this JSON from the raw question, and the same script decides the answer.

### Top-level fields

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

#### `positions`

Who argues what. Use the labels the passage uses: `A`, `B`, `Alice`, `Hypothesis 1`, `ⓐ`, `Method A`.

```json
"positions": {
  "A": {"thesis": "c4", "summary": "Evil is not a privation, because evil has degrees and privation does not."}
}
```

- `thesis`: the claim ID of the position's main conclusion. Checks can then refer to a position by name.
- `summary`: one sentence, for human readers.
- `*` is a reserved label for content that belongs to no single position: facts stated by the narrator, facts given in a statement ("The fact that ..."), and general background knowledge. Every position may use `*` content.

#### `claims`

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

#### `premises`

Claims a position starts from.

```json
{"claim": "c3", "by": ["A", "B"], "kind": "ordinary"}
```

- `by`: a position label, a list of labels, or `"*"`.
- `kind`: `ordinary` (an assumption that can be attacked) or `axiom` (a fact that cannot be attacked: scenario facts, numbers, evidence given in a statement).

#### `rules`

Steps from claims to a claim.

```json
{"id": "r2", "by": "B", "if": ["c5"], "then": "c6", "type": "defeasible", "source": "passage"}
```

- `type`: `strict` (follows with certainty: logic, definitions, a criterion the statement says to apply) or `defeasible` (holds usually: generalizations, causal or empirical inferences, analogies).
- `source`: same values as for claims.
- Rule IDs are `r1`, `r2`, ... in the passage and `ra1`, `rb1`, ... inside the check of statement (a), (b), ...

#### `contraries`

Pairs of claims that cannot both be true: `[["c1", "c6"], ["c4", "c7"]]`. Symmetric.

#### `undercuts`

A claim that says a rule does not apply here: `[{"claim": "c2", "rule": "rc1"}]`. Use it when a fact blocks an inference without denying its conclusion.

#### How attacks follow (computed, not written)

No preferences, so every attack succeeds.

- **Rebut**: an argument concluding X attacks any argument whose *defeasible* step concludes a contrary of X.
- **Undermine**: an argument concluding X attacks any argument that uses an *ordinary* premise that is a contrary of X.
- **Undercut**: an argument concluding a claim listed in `undercuts` attacks any argument that uses that rule.
- Strict steps and axioms cannot be attacked directly, only through what they are built on.

The script then computes the grounded labelling (via PyArg when installed): each argument is IN, OUT or UNDEC. A claim is IN if some argument for it is IN, else UNDEC if some argument is UNDEC, else OUT if arguments exist, else NONE.

### `statements`

One entry per statement, keyed by its label as printed: `a`, `b`, `c` (or `1` to `5` when the choices themselves are the statements).

```json
"statements": {
  "a": {"check": {"op": "holds", "by": "B", "claim": "c7"}, "note": "optional, for human reviewers"}
}
```

`note` is ignored by the script and removed from training targets.

#### Check operations

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

#### Typical statement patterns

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

### Writing rules

1. Do not write any verdict, "correct/incorrect", or the answer number anywhere.
2. Every `passage` claim must be findable in the passage. Do not invent speakers or facts.
3. Use the fewest claims and rules that make every check meaningful. Do not force a fixed number.
4. General logic that holds for every question (for example "if both use the same premise, they agree") belongs in `check_aspic.py`, not in the translation.
5. Mark anything not in the passage as `rationale` or `background`. The share of such elements tells us how much the model must supply on its own at test time.

### Frozen instruction for training and inference

Use this exact text as the system message in the training data, for the base model, and for the fine-tuned model. Do not edit it after the first run.

```
Translate the LEET argument passage into ASPIC+ JSON with these fields: positions, claims, premises, rules, contraries, undercuts, hypotheticals, statements. Keep claim text close to the passage wording. Mark rules as strict or defeasible. For each statement, write one check (holds, assumes, status, weakens, strengthens, and, or, not). Do not decide whether any statement is correct. Output only the JSON.
```

User message: the passage, question and statements, without the ① to ⑤ choice list (same as the first fine-tuning run). Assistant message: the translation JSON with `note` fields removed.

## Part 2. Worked examples

#### Question 2022_17

Question text:

A: There are people who claim that evil is not a real existence but merely a privation. However, when it comes to privation, we cannot speak of "more" or "less." This is because the statement "this is lacking" and the statement "this is not lacking" are in a contradictory relationship. In a contradiction, no middle ground is allowed. On the other hand, with respect to existence, we can speak of "more" or "less," since there can be degrees of perfection in existence. Then what about evil? Among evil things, some are more evil than others.B: When we say that something is more evil or less evil than another, we are referring to how far removed it is from the good. In this sense, even for concepts that involve privation, such as “inequality” or “dissimilarity,” we can speak of “more” or “less.” That is, we say something is “more unequal” when it is farther from equality, and “more dissimilar” when it is farther from similarity. Therefore, something that is more lacking in goodness, as if it were farther from the good, can be described as “more evil.” A privation is not a property that exists in itself, but only increases or decreases according to the increase or decrease in the cause that produces it. Darkness does not exist or become intensified on its own; rather, the more light is blocked, the darker it becomes, and the farther it is removed from brightness. <question> Which of the following <statements> is correct regarding the evaluations of A and B? <statements> (a) Unlike A, B argues that evil is a privation. (b) A acknowledges the degree of evil, and B agrees with this. (c) The view that good can exist without evil, but evil cannot exist without good, is better supported by B than by A. <choices> ① (a) ② (c) ③ (a), (b) ④ (b), (c) ⑤ (a), (b), (c)

Expert rationale:

- preliminaries: A is an argument that refutes the perspective that evil is not a real existence but merely a privation, while B is an argument that, in order to maintain the privation view, refutes A. The issue at stake in both arguments is the ambiguity of the concept of privation. A understands the relationship of privation with respect to existence simply as the negation of a statement, and holds that, since affirmative and negative statements are contradictory, there can be no “more” or “less” in negation or privation. That is, A claims that in the contradictory statements “it exists” and “it does not exist (is lacking),” there is no room for an intermediate, that is, for “more” or “less.” Nevertheless, the existence of “more evil” and “less evil” is commonly accepted as an empirical fact or in ordinary language, and therefore, A concludes that evil cannot be a mere privation. In contrast, B does not simply understand the relationship of privation with respect to existence as a negation of a statement, but as a state where an essence or property is removed or diminished due to a particular cause. In this sense, degrees are naturally accepted in privation. <choices> Explanation
- statement_1_rationale: (a) A claims that privation does not accept degrees, but existence does, and since evil accepts degrees, it must be a kind of existence. In contrast, B defends the concept of evil as privation by interpreting privation as something that can accept degrees, opposing A. (a) is a correct evaluation.
- statement_2_rationale: (b) The fact that evil can be more or less is a key premise in A’s argument. B also agrees with this premise. What B rejects is A’s claim that privation does not accept degrees. Therefore, (b) is a correct evaluation.
- statement_3_rationale: (c) According to B, privation is a partial negation of existence (or a deleted part of existence), and thus, is a concept dependent on existence. Therefore, B’s view, which sees evil as privation, better supports the idea that good can exist without evil, but evil cannot exist without good. On the other hand, A’s view, which sees evil as a kind of existence, tends to regard good and evil as ontologically equal, thus implying that evil can also exist independently of good. Therefore, (c) is correct.
- answer: Since all of (a), (b), and (c) are correct, the right answer is ⑤.

Translation:

```json
{
 "id": "2022_17",
 "positions": {
  "A": {
   "thesis": "c7",
   "summary": "Evil is an existence and not a privation, because evil admits of degrees while privation does not."
  },
  "B": {
   "thesis": "c13",
   "summary": "Evil is a privation that admits of degrees based on distance from the good, and evil depends on good."
  }
 },
 "claims": {
  "c1": {
   "text": "Privation does not admit of more or less.",
   "source": "passage"
  },
  "c2": {
   "text": "'This is lacking' and 'this is not lacking' are contradictory, allowing no middle ground.",
   "source": "passage"
  },
  "c3": {
   "text": "Existence admits of more or less, since there can be degrees of perfection in existence.",
   "source": "passage"
  },
  "c4": {
   "text": "Evil admits of degrees (some evil things are more evil than others).",
   "source": "passage"
  },
  "c5": {
   "text": "Evil is not a privation.",
   "source": "passage"
  },
  "c6": {
   "text": "Evil is a privation.",
   "source": "passage"
  },
  "c7": {
   "text": "Evil is an existence.",
   "source": "rationale"
  },
  "c8": {
   "text": "Concepts involving privation (such as inequality or dissimilarity) admit of more or less depending on distance from perfection/good.",
   "source": "passage"
  },
  "c9": {
   "text": "Something more lacking in goodness is more evil.",
   "source": "passage"
  },
  "c10": {
   "text": "Privation does not exist in itself, but increases or decreases according to the cause producing it.",
   "source": "passage"
  },
  "c11": {
   "text": "Good can exist without evil, but evil cannot exist without good.",
   "source": "statement"
  },
  "c12": {
   "text": "Evil can exist independently of good.",
   "source": "rationale"
  },
  "c13": {
   "text": "Evil is a privation that admits of degrees.",
   "source": "passage"
  }
 },
 "premises": [
  {
   "claim": "c2",
   "by": "A",
   "kind": "ordinary"
  },
  {
   "claim": "c3",
   "by": "A",
   "kind": "ordinary"
  },
  {
   "claim": "c4",
   "by": [
    "A",
    "B"
   ],
   "kind": "axiom"
  },
  {
   "claim": "c8",
   "by": "B",
   "kind": "ordinary"
  },
  {
   "claim": "c10",
   "by": "B",
   "kind": "ordinary"
  }
 ],
 "rules": [
  {
   "id": "r1",
   "by": "A",
   "if": [
    "c2"
   ],
   "then": "c1",
   "type": "strict",
   "source": "passage"
  },
  {
   "id": "r2",
   "by": "A",
   "if": [
    "c1",
    "c4"
   ],
   "then": "c5",
   "type": "strict",
   "source": "passage"
  },
  {
   "id": "r3",
   "by": "A",
   "if": [
    "c3",
    "c4",
    "c5"
   ],
   "then": "c7",
   "type": "defeasible",
   "source": "rationale"
  },
  {
   "id": "r4",
   "by": "A",
   "if": [
    "c7"
   ],
   "then": "c12",
   "type": "defeasible",
   "source": "rationale"
  },
  {
   "id": "r5",
   "by": "B",
   "if": [
    "c8",
    "c4"
   ],
   "then": "c9",
   "type": "defeasible",
   "source": "passage"
  },
  {
   "id": "r6",
   "by": "B",
   "if": [
    "c9"
   ],
   "then": "c13",
   "type": "defeasible",
   "source": "passage"
  },
  {
   "id": "r7",
   "by": "B",
   "if": [
    "c13"
   ],
   "then": "c6",
   "type": "strict",
   "source": "passage"
  },
  {
   "id": "r8",
   "by": "B",
   "if": [
    "c6",
    "c10"
   ],
   "then": "c11",
   "type": "defeasible",
   "source": "rationale"
  }
 ],
 "contraries": [
  [
   "c5",
   "c6"
  ],
  [
   "c1",
   "c8"
  ],
  [
   "c11",
   "c12"
  ]
 ],
 "statements": {
  "a": {
   "check": {
    "op": "and",
    "args": [
     {
      "op": "holds",
      "by": "B",
      "claim": "c6"
     },
     {
      "op": "not",
      "arg": {
       "op": "holds",
       "by": "A",
       "claim": "c6"
      }
     }
    ]
   },
   "note": "B argues evil is a privation (c6), whereas A concludes evil is not a privation (c5, contrary of c6)."
  },
  "b": {
   "check": {
    "op": "and",
    "args": [
     {
      "op": "assumes",
      "by": "A",
      "claim": "c4"
     },
     {
      "op": "assumes",
      "by": "B",
      "claim": "c4"
     }
    ]
   },
   "note": "Both A and B take the degree of evil (c4) as an established premise."
  },
  "c": {
   "check": {
    "op": "and",
    "args": [
     {
      "op": "holds",
      "by": "B",
      "claim": "c11"
     },
     {
      "op": "not",
      "arg": {
       "op": "holds",
       "by": "A",
       "claim": "c11"
      }
     }
    ]
   },
   "note": "B derives that good can exist without evil but not vice versa (c11), whereas A views evil as an independent existence (c12)."
  }
 }
}
```

#### Question 2021_37

Question text:

At the end of the 18th century, the French chemist Lavoisier argued that the combustion of a substance was a process in which the substance combined with another substance he named “oxygen.” However, this claim met with opposing arguments that it could not explain why heat was generated when a substance combined with oxygen. In response, he made the following assumptions: First, heat is actually a massless substance called “caloric,” and the generation of heat is a process by which caloric, which had been combined with a substance, is released. Second, gases contain far more caloric than solids or liquids. The reason why water turns into steam when an enormous amount of heat is supplied is because water and a large amount of caloric combine together. Similarly, gaseous oxygen is also a compound of oxygen and caloric. Based on these two assumptions, Lavoisier claimed that ‘ⓐ in the process of combustion, the oxygen in gaseous oxygen combines with the burning substance to form a compound, but the caloric in gaseous oxygen does not combine with the substance and escapes into the air, thus generating heat.’<question> Which of the following from <statements> are appropriate evaluations of ⓐ?<statements> (a) The fact that when many solid substances combust, heat is generated and gas is produced at the same time, strengthens ⓐ.(b) The fact that gunpowder, which contains compounds of oxygen, can combust explosively and release a massive amount of heat even without gaseous oxygen, weakens ⓐ.(c) The fact that, no matter how much heat generated during the combustion process is released into the air, the mass of air does not increase, weakens ⓐ. <choices> ① (a) ② (b) ③ (a), (c) ④ (b), (c) ⑤ (a), (b), (c)

Expert rationale:

- preliminaries: Lavoisier’s theory of combustion was accepted because of its strong explanatory power concerning the change in mass during the process of combustion. However, there were several weaknesses in his theory, one of which was the difficulty in explaining why heat is generated in the process of combustion. He proposed theory ⓐ, in which gaseous oxygen contains caloric, and during combustion, caloric escapes into the atmosphere and thus causes heat to be released. Even at the time, his theory was subject to criticism. According to ⓐ, the reason heat is generated in the process in which gaseous oxygen combines with a substance is that the caloric within the gaseous oxygen escapes into the air. Note that Lavoisier’s concept of caloric is not equivalent to the modern scientific concept of energy. Caloric was considered a type of material that could combine with other substances (e.g., oxygen) to form a compound; a gas, in this view, was itself a compound resulting from a sufficient quantity of caloric binding to a material. Thus, unlike in modern science, for Lavoisier a gas was a type of compound.
- statement_1_rationale: (a) In Lavoisier’s theory of combustion, in the process of combustion, a substance combines only with oxygen. In this process, the caloric contained in gaseous oxygen escapes into the air, generating heat. This may be represented as: Substance + gaseous oxygen (oxygen + caloric) → oxide (substance + oxygen) + released caloric However, let us consider that a solid substance such as coal combusts and produces a gaseous compound. This is a process in which a solid substance containing little caloric is converted into a gaseous compound containing much more caloric, so it would have to combine with both oxygen and caloric. This process can be represented as: solid substance + gaseous oxygen (oxygen + caloric) → gaseous oxide (substance + oxygen + caloric) or solid substance + gaseous oxygen (oxygen + caloric) → gaseous oxide (substance + oxygen + caloric) + a small amount of remaining caloric This would result in little or no caloric escaping to the outside, so little or no heat should be generated in the combustion process. Therefore, the fact that heat is generated and gas is produced when a solid substance combusts does not strengthen ⓐ. (a) is not an appropriate evaluation.
- statement_2_rationale: (b) ⓐ assumes that the source of heat generated during combustion is the caloric originally contained in gaseous oxygen. However, gunpowder can combust and release large amounts of heat even without gaseous oxygen because it contains oxygen compounds. These compounds exist as solids within the gunpowder and contain little caloric. Thus, the explanation in ⓐ—that heat is generated by the caloric escaping from gaseous oxygen during combustion—is weakened. (b) is an appropriate evaluation.
- statement_3_rationale: (c) Lavoisier regarded caloric as a massless substance; therefore, even if caloric is released into the air during combustion, the mass of the air does not increase. Thus, the fact that the mass of air does not increase even when heat is released into it does not weaken ⓐ, which seeks the source of generated heat in the caloric combined in gaseous oxygen during combustion. (c) is not an appropriate evaluation.Only (b) is an appropriate evaluation in <choices>, so the correct answer is ②.
- answer: Only ② is the correct analysis among the <choices>, so the answer is ②.

Translation:

```json
{
 "id": "2021_37",
 "positions": {
  "ⓐ": {
   "thesis": "c1",
   "summary": "Combustion occurs when oxygen in gaseous oxygen combines with a substance, while caloric escapes into the air, generating heat."
  }
 },
 "claims": {
  "c1": {
   "text": "ⓐ In combustion, oxygen in gaseous oxygen combines with the burning substance, and caloric escapes into the air, generating heat.",
   "source": "passage"
  },
  "c2": {
   "text": "Heat is a massless substance called caloric, and heat generation is the release of combined caloric.",
   "source": "passage"
  },
  "c3": {
   "text": "Gaseous oxygen is a compound of oxygen and caloric.",
   "source": "passage"
  },
  "c4": {
   "text": "Gases contain far more caloric than solids or liquids.",
   "source": "passage"
  },
  "c5": {
   "text": "Caloric is massless.",
   "source": "passage"
  }
 },
 "premises": [
  {
   "claim": "c2",
   "by": "ⓐ",
   "kind": "ordinary"
  },
  {
   "claim": "c3",
   "by": "ⓐ",
   "kind": "ordinary"
  },
  {
   "claim": "c4",
   "by": "*",
   "kind": "axiom"
  },
  {
   "claim": "c5",
   "by": "*",
   "kind": "axiom"
  }
 ],
 "rules": [
  {
   "id": "r1",
   "by": "ⓐ",
   "if": [
    "c2",
    "c3"
   ],
   "then": "c1",
   "type": "defeasible",
   "source": "passage"
  }
 ],
 "contraries": [],
 "statements": {
  "a": {
   "check": {
    "op": "strengthens",
    "claim": "c1",
    "add": {
     "claims": {
      "h1": {
       "text": "When many solid substances combust, heat is generated and gas is produced at the same time.",
       "source": "statement"
      },
      "h2": {
       "text": "Since the produced gas requires large amounts of caloric, little or no caloric escapes, so heat should not be generated under ⓐ.",
       "source": "rationale"
      }
     },
     "premises": [
      {
       "claim": "h1",
       "by": "*",
       "kind": "axiom"
      }
     ],
     "rules": [
      {
       "id": "ra1",
       "by": "*",
       "if": [
        "h1",
        "c4"
       ],
       "then": "h2",
       "type": "defeasible",
       "source": "rationale"
      }
     ],
     "contraries": [
      [
       "h2",
       "c1"
      ]
     ]
    }
   },
   "note": "Solid-to-gas combustion consumes caloric rather than releasing it, attacking rather than strengthening ⓐ."
  },
  "b": {
   "check": {
    "op": "weakens",
    "claim": "c1",
    "add": {
     "claims": {
      "h3": {
       "text": "Gunpowder containing oxygen compounds combusts explosively and releases massive heat without gaseous oxygen.",
       "source": "statement"
      },
      "h4": {
       "text": "Caloric in gaseous oxygen is not the source of heat generated during combustion.",
       "source": "rationale"
      }
     },
     "premises": [
      {
       "claim": "h3",
       "by": "*",
       "kind": "axiom"
      }
     ],
     "rules": [
      {
       "id": "rb1",
       "by": "*",
       "if": [
        "h3"
       ],
       "then": "h4",
       "type": "defeasible",
       "source": "rationale"
      }
     ],
     "contraries": [
      [
       "h4",
       "c1"
      ]
     ]
    }
   },
   "note": "Combustion releasing heat without gaseous oxygen undermines the claim that caloric from gaseous oxygen is the source of combustion heat."
  },
  "c": {
   "check": {
    "op": "weakens",
    "claim": "c1",
    "add": {
     "claims": {
      "h5": {
       "text": "No matter how much heat released during combustion enters the air, the mass of air does not increase.",
       "source": "statement"
      },
      "h6": {
       "text": "If heat were a substance added to the air, air mass would increase.",
       "source": "rationale"
      }
     },
     "premises": [
      {
       "claim": "h5",
       "by": "*",
       "kind": "axiom"
      }
     ],
     "rules": [
      {
       "id": "rc1",
       "by": "*",
       "if": [
        "h5"
       ],
       "then": "h6",
       "type": "defeasible",
       "source": "rationale"
      }
     ],
     "contraries": [
      [
       "h6",
       "c1"
      ]
     ],
     "undercuts": [
      {
       "claim": "c5",
       "rule": "rc1"
      }
     ]
    }
   },
   "note": "The naive attack rc1 assumes caloric has mass, but since caloric is massless (c5), rc1 is undercut."
  }
 }
}
```

#### Question 2024_15

Question text:

Let’s suppose there are three worlds, A, B, and C. A: 100 million people live here. Each of them enjoys a high welfare level of 100 units. B: 10 billion people live here, each enjoying a modest yet worthwhile welfare level of 5 units. Then, 100 million people who were living in A migrate to B. The 100 million migrants from A enjoy a welfare level of 105 units each. The original 10 billion inhabitants of B continue to enjoy 5 units of welfare each. C: All 10.1 billion people residing in B migrate to C, where previously no one lived. All 10.1 billion inhabitants of C enjoy a welfare level of 10 units each. <Opinions> Alice: The people who migrated from A to B now have a higher level of welfare than when they lived in A. Additionally, the remaining people in B continue to enjoy their worthwhile welfare level of 5 units each. Therefore, B is better than A. Bob: In C, there is complete equality, and the average welfare in C is also higher than in B. Therefore, C is better than B. Charlie: The total sum of welfare is greater in C than in A, but the average welfare is higher in A than in C. Therefore, A is better than C. <question> Choose all the correct evaluations of the <Opinions> from the statements listed below. <statements> (a) If a world with less inequality is considered a better world, then Alice’s conclusion is denied while Bob’s conclusion is not denied. (b) If we apply Bob’s reasoning for why C is better than B, then Bob would agree with Charlie’s conclusion and would not agree with Alice’s conclusion. (c) If a world with a higher average welfare is considered a better world, Alice’s conclusion is denied; if a world with a greater total welfare is considered a better world, then Bob’s conclusion is not denied and Charlie’s conclusion is denied. <choices> ① (a) ② (c) ③ (a), (b) ④ (b), (c) ⑤ (a), (b), (c)

Expert rationale:

- preliminaries: The worlds A, B, and C can be illustrated as follows. In the diagram, the horizontal width of a rectangle represents the population size, and the height represents the welfare level.
- statement_1_rationale: (a) Worlds A and C are completely equalitarian, but B is a world with inequality. Alice’s conclusion is that B (the unequal world) is better than A (the equal world), whereas Bob’s conclusion is that C (the equal world) is better than B (the unequal world). Therefore, if a world with less inequality is considered better, Alice’s conclusion is denied while Bob’s conclusion is not denied. Thus, (a) is a correct evaluation.
- statement_2_rationale: (b) The reasons Bob argues that C is better than B are (1) there is complete equality, and (2) the average welfare is higher. Applying these reasons (1) and (2) to Charlie’s conclusion that A is better than C: since A and C are equally equalitarian, but the average welfare in A is higher than in C, the conclusion would be that A is better than C. Therefore, if Bob were to apply (1) and (2), he would agree with Charlie’s conclusion. On the other hand, Alice’s conclusion is that B is better than A. By applying (1) and (2) here, since A is completely equal but B is unequal and A’s average welfare is higher than B’s, the conclusion reached is that A is better than B. Therefore, if Bob were to apply (1) and (2), he would not agree with Alice’s conclusion. Thus, (b) is a correct evaluation.
- statement_3_rationale: (c) The average and total welfare in each world are as follows: A B C Population 100,000,000 10,100,000,000 10,100,000,000 Average welfare 100  approx. 6  10 Total welfare 10,000,000,000 60,500,000,000 101,000,000,000 Since the average welfare in A is higher than that in B, if a higher average welfare is the criterion for a better world, A is better than B. Therefore, Alice’s conclusion that B is better than A is denied. The total welfare increases from A to B to C, so if a greater total welfare is considered better, C is better than both B and A. Therefore, Bob’s conclusion that C is better than B is not denied, and Charlie’s conclusion that A is better than C is denied. Hence, (c) is a correct evaluation.
- answer: Since all of (a), (b), and (c) in <Explanation> are correct, the answer is ⑤.

Translation:

```json
{
 "id": "2024_15",
 "positions": {
  "Alice": {
   "thesis": "c7",
   "summary": "B is better than A."
  },
  "Bob": {
   "thesis": "c10",
   "summary": "C is better than B."
  },
  "Charlie": {
   "thesis": "c13",
   "summary": "A is better than C."
  }
 },
 "claims": {
  "c1": {
   "text": "In A, 100M people enjoy 100 units each (total: 10B, avg: 100, completely equal).",
   "source": "passage"
  },
  "c2": {
   "text": "In B, 100M people enjoy 105 units and 10B people enjoy 5 units (total: 60.5B, avg: ~6, unequal).",
   "source": "passage"
  },
  "c3": {
   "text": "In C, 10.1B people enjoy 10 units each (total: 101B, avg: 10, completely equal).",
   "source": "passage"
  },
  "c4": {
   "text": "Migrants in B have higher welfare (105) than in A (100), and original B inhabitants maintain worthwhile welfare (5).",
   "source": "passage"
  },
  "c5": {
   "text": "In C there is complete equality and higher average welfare than in B.",
   "source": "passage"
  },
  "c6": {
   "text": "Total welfare is greater in C than in A, but average welfare is higher in A than in C.",
   "source": "passage"
  },
  "c7": {
   "text": "B is better than A.",
   "source": "passage"
  },
  "c8": {
   "text": "A is better than B.",
   "source": "rationale"
  },
  "c9": {
   "text": "C is not better than B.",
   "source": "rationale"
  },
  "c10": {
   "text": "C is better than B.",
   "source": "passage"
  },
  "c11": {
   "text": "C is better than A.",
   "source": "rationale"
  },
  "c12": {
   "text": "A is not better than C.",
   "source": "rationale"
  },
  "c13": {
   "text": "A is better than C.",
   "source": "passage"
  }
 },
 "premises": [
  {
   "claim": "c1",
   "by": "*",
   "kind": "axiom"
  },
  {
   "claim": "c2",
   "by": "*",
   "kind": "axiom"
  },
  {
   "claim": "c3",
   "by": "*",
   "kind": "axiom"
  },
  {
   "claim": "c4",
   "by": "Alice",
   "kind": "ordinary"
  },
  {
   "claim": "c5",
   "by": "Bob",
   "kind": "ordinary"
  },
  {
   "claim": "c6",
   "by": "Charlie",
   "kind": "ordinary"
  }
 ],
 "rules": [
  {
   "id": "r1",
   "by": "Alice",
   "if": [
    "c4"
   ],
   "then": "c7",
   "type": "defeasible",
   "source": "passage"
  },
  {
   "id": "r2",
   "by": "Bob",
   "if": [
    "c5"
   ],
   "then": "c10",
   "type": "defeasible",
   "source": "passage"
  },
  {
   "id": "r3",
   "by": "Charlie",
   "if": [
    "c6"
   ],
   "then": "c13",
   "type": "defeasible",
   "source": "passage"
  }
 ],
 "contraries": [
  [
   "c7",
   "c8"
  ],
  [
   "c10",
   "c9"
  ],
  [
   "c13",
   "c11"
  ],
  [
   "c13",
   "c12"
  ]
 ],
 "hypotheticals": {
  "less_inequality": {
   "claims": {
    "h1": {
     "text": "A world with less inequality is considered a better world.",
     "source": "statement"
    }
   },
   "premises": [
    {
     "claim": "h1",
     "by": "*",
     "kind": "axiom"
    }
   ],
   "rules": [
    {
     "id": "rh1",
     "by": "*",
     "if": [
      "h1",
      "c1",
      "c2"
     ],
     "then": "c8",
     "type": "strict",
     "source": "rationale"
    }
   ]
  },
  "higher_average": {
   "claims": {
    "h2": {
     "text": "A world with a higher average welfare is considered a better world.",
     "source": "statement"
    }
   },
   "premises": [
    {
     "claim": "h2",
     "by": "*",
     "kind": "axiom"
    }
   ],
   "rules": [
    {
     "id": "rh2",
     "by": "*",
     "if": [
      "h2",
      "c1",
      "c2"
     ],
     "then": "c8",
     "type": "strict",
     "source": "rationale"
    }
   ]
  },
  "greater_total": {
   "claims": {
    "h3": {
     "text": "A world with a greater total welfare is considered a better world.",
     "source": "statement"
    }
   },
   "premises": [
    {
     "claim": "h3",
     "by": "*",
     "kind": "axiom"
    }
   ],
   "rules": [
    {
     "id": "rh3",
     "by": "*",
     "if": [
      "h3",
      "c3",
      "c1"
     ],
     "then": "c11",
     "type": "strict",
     "source": "rationale"
    }
   ]
  }
 },
 "statements": {
  "a": {
   "check": {
    "op": "and",
    "args": [
     {
      "op": "status",
      "claim": "Alice",
      "is": [
       "OUT"
      ],
      "add": "less_inequality"
     },
     {
      "op": "not",
      "arg": {
       "op": "status",
       "claim": "Bob",
       "is": [
        "OUT"
       ],
       "add": "less_inequality"
      }
     }
    ]
   },
   "note": "Under the equality criterion, A is better than B, rebutting Alice's conclusion (Alice is OUT), while Bob's conclusion (C is better than B) remains undefeated."
  },
  "b": {
   "check": {
    "op": "and",
    "args": [
     {
      "op": "holds",
      "by": "Bob",
      "claim": "c13",
      "add": {
       "claims": {
        "h4": {
         "text": "Between worlds with equal distribution, the world with higher average welfare is better.",
         "source": "rationale"
        }
       },
       "premises": [
        {
         "claim": "h4",
         "by": "Bob",
         "kind": "ordinary"
        }
       ],
       "rules": [
        {
         "id": "rb1",
         "by": "Bob",
         "if": [
          "h4",
          "c1",
          "c3"
         ],
         "then": "c13",
         "type": "defeasible",
         "source": "rationale"
        }
       ]
      }
     },
     {
      "op": "not",
      "arg": {
       "op": "holds",
       "by": "Bob",
       "claim": "c7",
       "add": {
        "claims": {
         "h5": {
          "text": "A world with both complete equality and higher average welfare is better than an unequal world with lower average welfare.",
          "source": "rationale"
         }
        },
        "premises": [
         {
          "claim": "h5",
          "by": "Bob",
          "kind": "ordinary"
         }
        ],
        "rules": [
         {
          "id": "rb2",
          "by": "Bob",
          "if": [
           "h5",
           "c1",
           "c2"
          ],
          "then": "c8",
          "type": "defeasible",
          "source": "rationale"
         }
        ]
       }
      }
     }
    ]
   },
   "note": "Applying Bob's equality and average welfare criteria leads Bob to agree that A is better than C (c13) and reject that B is better than A (c7)."
  },
  "c": {
   "check": {
    "op": "and",
    "args": [
     {
      "op": "status",
      "claim": "Alice",
      "is": [
       "OUT"
      ],
      "add": "higher_average"
     },
     {
      "op": "not",
      "arg": {
       "op": "status",
       "claim": "Bob",
       "is": [
        "OUT"
       ],
       "add": "greater_total"
      }
     },
     {
      "op": "status",
      "claim": "Charlie",
      "is": [
       "OUT"
      ],
      "add": "greater_total"
     }
    ]
   },
   "note": "Higher average welfare makes A better than B, denying Alice. Greater total welfare makes C better than A (denying Charlie) and C better than B (leaving Bob not denied)."
  }
 }
}
```


## Part 3. Questions to translate

#### Question 2022_29

Question text:

Violent and property crimes, which had surged explosively in the United States since the 1960s, entered a sharp declining trend from the early 1990s onwards. Between 1991 and 2012, violent crime decreased by 49% and property crime by 44%. Moreover, this decline was observed in all regions and across all population groups, and the trend has continued up until 2020. In this context, there is the claim ⓐ, ‘that the decline in crime in the United States is closely related to lead exposure.’ According to this claim, from the end of World War II until the early 1970s, the increase in lead emissions from cars was followed by a rise in violent crime. However, after lead began to be removed from gasoline in the 1970s, violent crime subsequently decreased. Tetraethyl lead was developed in the 1920s to prevent engine knocking in gasoline engines, and its use exploded in the postwar period with the popularity of automobiles. Rates of violent and property crime are highest in the late teens to early twenties; unlike adolescents or adults, however, for children, exposure to lead affects brain development and future potential for criminal behavior. In particular, lead is known to induce increased aggression and impulsivity. <question> Which of the following evaluations of the above passage is correct? Select all that apply from <statements>.<statements> (a) The fact that the average blood lead levels of American children aged 1–5 in 2000 had dropped to half the level of 1990 strengthens claim ⓐ. (b) The fact that the sharp decline in violent crime in the US began in the 1990s rather than the 1970s weakens claim ⓐ. (c) The research finding that adolescents in the United States who committed crimes had four times the bone lead content of those who did not strengthens claim ⓐ. <choices> ① (a) ② (b) ③ (a), (c) ④ (b), (c) ⑤ (a), (b), (c)

Expert rationale:

- preliminaries: In the passage, the statement “rates of violent and property crime are highest in the late teens to early twenties; unlike adolescents or adults, for children, exposure to lead affects brain development and future potential for criminal behavior. In particular, lead is known to induce increased aggression and impulsivity” serves as a key clue for solving the problem. From this, we can infer that lead exposure affects the likelihood of criminal behavior in childhood, but its effects emerge roughly 20 years later, when the affected children reach their late teens or early twenties.
- statement_1_rationale: (a) Suppose it is true that the average blood lead level among American children aged 1–5 in 2000 was half the level of a decade earlier in 1990. If claim ⓐ is true, and there is a positive correlation between lead poisoning and criminal behavior, this implies a reduction in the relative size of the population at risk for future crime. Children aged 1–5 in 1990 would likely express potential criminal behavior around 2005–2010, and those aged 1–5 in 2000 would do so around 2015–2020. The passage states that the decline in American crime rates has continued from the early 1990s until 2020. Therefore, the fact that average blood lead levels among 1–5 year-olds in 2000 fell to half the 1990 level strengthens claim ⓐ. Thus, (a) is a correct evaluation.
- statement_2_rationale: (b) According to the reasoning for ⓐ, lead negatively impacts brain development in children, thus affecting the probability of criminal behavior when they later become adolescents or adults; that is, there should be a time lag between exposure to lead and the manifestation of criminal behavior—about 20 years, until children reach their late teens or early twenties. Therefore, the effect of reducing lead in gasoline in the 1970s should appear some 20 years later, i.e., in the 1990s. Thus, the fact that crime rates started dropping in the 1990s rather than the 1970s is actually consistent with ⓐ, and does not weaken it. Thus, (b) is not a correct evaluation.
- statement_3_rationale: (c) The claim that the reduction in American crime is closely linked to lead means that there is a positive correlation between lead poisoning and criminal behavior. The research showing that adolescent offenders in the US have much higher bone lead levels than non-offenders supports this correlation and thus strengthens claim ⓐ. Therefore, (c) is a correct evaluation.
- answer: Among the <choices>, only (a) and (c) are correct, so the answer is ③.

#### Question 2023_26

Question text:

An election exit poll is a method of increasing the accuracy of predicting election results by surveying actual voters instead of eligible voters. In exit polls, it is important how to select the polling stations (within an electoral district) that will be surveyed. Once the polling stations are selected, the surveys are conducted at the polling places that belong to those polling stations. There are three exit poll methods: A, B, and C.A: Compare the overall vote tally of the district and the individual results from each polling station in the previous election, then select the polling place of the station where the difference is smallest as the target for the survey. B: Group together polling stations that showed similar party-specific results in the previous election into several strata, and select a certain number of polling stations randomly from each stratum according to the proportion of voters, then survey the polling places corresponding to those stations. C: Rather than pre-selecting polling stations and conducting surveys within the voting hours only at those places, travel to all polling stations in the district, conducting surveys at each by staying for a certain period of time before moving to the next. In this way, surveyors travel between polling stations, surveying for a set amount of time at each. The sample size for each polling station is determined in proportion to its number of eligible voters. <question> Which of the following analyses about the text above is/are correct? Select the correct options from <statements> as they are given. <statements> (a) If there has been significant change in the population’s social characteristics in a polling station since the last election, method A is difficult to apply. (b) Method B assumes that polling stations within the same district are generally homogeneous in terms of voters’ political orientation. (c) In method C, data from the previous election about vote counts per polling station is essential. <choices> ① (a) ② (c) ③ (a), (b) ④ (b), (c) ⑤ (a), (b), (c)

Expert rationale:

- preliminaries: As with other surveys, ensuring the representativeness of the sample is key in election exit polls. Method A is an approach seeking representativeness by selecting the polling station that, based on results from the previous election, had the most similar results to the district as a whole. Method B first stratifies all polling stations by grouping those with similar results in the previous election, then selects a certain number of polling stations from each stratum. This method seeks representativeness by ensuring that polling stations with diverse characteristics are evenly included in the sample. Method C is a method where all polling stations are visited in sequence, and a small number of voters are surveyed at each, aiming to ensure that voters from every polling station are included in the sample.
- statement_1_rationale: (a) Method A assumes that voting behavior at a particular polling station will be similar to that in the previous election. If there has been a significant change in the population or social characteristics of a polling station since the last election, voting behavior may differ considerably from the past, making method A difficult to apply. Therefore, (a) is a correct analysis.
- statement_2_rationale: (b) Method B assumes that polling stations within the same district are heterogeneous, which is why it first groups homogeneous polling stations into strata, then selects a certain number of polling stations from each stratum. While polling stations within the same stratum are homogeneous, polling stations in different strata are heterogeneous. By this approach, polling stations with various characteristics are evenly surveyed, allowing for better prediction of the district's overall voting results. If polling stations’ political inclinations were homogeneous, there would be no need to divide them into strata and select polling stations from each. Therefore, (b) is an incorrect analysis.
- statement_3_rationale: (c) Both A and B require data from the previous election's polling station results in order to pre-select representative stations; however, C surveys all polling stations anyway, so previous vote count data is not necessary. Therefore, (c) is an incorrect analysis.
- answer: Thus, only (a) is a correct analysis, so the answer is ①.

#### Question 2021_28

Question text:

Country X applies the following progressive billing system for monthly household electricity rates.- Tier 1: Usage up to 200 kWh, base charge is 900 won, rate per kWh is 90 won.- Tier 2: Usage over 200 kWh and up to 400 kWh, base charge is 1,600 won, rate per kWh is 180 won.- Tier 3: Usage over 400 kWh, base charge is 7,300 won, rate per kWh is 280 won.For example, a household consuming 300 kWh in a month will pay a base charge of 1,600 won, and rates of 90 won per kWh for Tier 1 and 180 won per kWh for Tier 2, resulting in a total electricity bill of 37,600 won (=1,600 + 200×90 + 100×180).Recently, Country X introduced a “Cool Summer Policy” for electricity used in the summer, changing the tiers to ‘up to 300 kWh’, ‘over 300 kWh up to 450 kWh’, and ‘over 450 kWh’, while maintaining the same rate structure as before in each tier.<Debate> Alice: In order to ensure a stable electricity supply, power generation facilities must be able to meet the peak hourly electricity demand. The introduction of the Cool Summer Policy will reduce the stability of electricity supply.Bob: Air conditioning is a necessity, not a luxury. To lower the summer electricity bill burden for most households, it would be better to switch from the Cool Summer Policy to a flat rate system with a base charge of 1,600 won and a per kWh rate of 180 won.Charlie: It is better to focus on welfare for vulnerable groups rather than all households. The Cool Summer Policy should be modified so that it applies only to vulnerable groups. <question> Which of the following, as presented in <statements>, are appropriate evaluations of the <Debate> above?<statements> (a) Data showing that peak hourly electricity consumption in Country X occurs in summer weakens Alice’s argument.(b) Data showing that most households consume 400–450 kWh per month weakens Bob’s argument.(c) Data showing that most vulnerable individuals, such as elderly people living alone, use less than 200 kWh per month weakens Charlie’s argument. <choices> ① (a) ② (b) ③ (a), (c) ④ (b), (c) ⑤ (a), (b), (c)

Expert rationale:

- preliminaries: Unlike a flat rate system, a progressive billing system applies different rates for each tier, and even for higher tiers, the lower tier rates are applied cumulatively for the respective amount of consumption in each tier. When the usage range of each tier is expanded by changing the tier criteria, households in higher tiers receive greater discounts. For example, in the Cool Summer Policy, households that originally belonged to Tier 1 before the change already benefit from the lowest rates, so expanding the tier does not affect them. However, households that were in Tier 2 prior to the change enjoy a rate reduction of 90 won (180–90 won) per kWh for usage between 200 and 300 kWh, and also a 700-won decrease in the base charge. Households in what was formerly Tier 3 benefit not only from this but also from a rate reduction of 100 won (280–180 won) per kWh for usage between 400 and 450 kWh, and a 5,700-won decrease in the base charge for households consuming 400–450 kWh.
- statement_1_rationale: (a) Alice claims that if the Cool Summer Policy reduces rates, summer electricity consumption will increase, leading to higher peak hourly demand in summer and thus reduced supply stability. Therefore, if data show that peak hourly demand in Country X is actually highest not in summer, and that even with increased summer consumption due to the policy this ranking does not change, Alice’s argument would be weakened. However, data that show summer peak demand is highest do not weaken Alice’s argument. Thus, (a) is not an appropriate evaluation.
- statement_2_rationale: (b) Under a progressive system, households with high consumption still enjoy lower rates for their usage in the lower tiers. A household using 400–450 kWh per month will fall into Tier 2 under the Cool Summer Policy, paying a base charge of 1,600 won, 90 won per kWh up to 300 kWh, and 180 won per kWh beyond 300 kWh. Under Bob’s proposed flat rate (1,600 won base, 180 won/kWh for all usage), all consumption is charged at the higher rate, leading to a bigger bill than under the Cool Summer Policy. Therefore, data showing most households are in this range would weaken Bob’s argument. So (b) is an appropriate evaluation.
- statement_3_rationale: c) Charlie argues the Cool Summer Policy should target only vulnerable populations for welfare reasons; this only makes sense if vulnerable groups actually benefit. If most vulnerable individuals, such as elderly people living alone, use less than 200 kWh per month, their bills would be unaffected whether the policy applies universally or only to them. Therefore, data showing this would weaken Charlie’s argument. Thus, (c) is an appropriate evaluation.
- answer: Since only (b) and (c) are appropriate, the answer is ④.

#### Question 2021_34

Question text:

To test the following hypotheses, [Experiment 1] and [Experiment 2] were conducted.(Hypothesis 1) People choose actions that maximize their expected payoff*.(Hypothesis 2) The more people believe that their advantageous but unfair actions are unlikely to be detected by the other party, the more likely they are to engage in such actions.[Experiment 1] Participants are randomly assigned as proposers and responders. The proposer is to suggest one of the following ways to split 10,000 won, which they will receive from the experimenter, with the responder:• 5-5 Option: The proposer and responder each get 5,000 won.• 8-2 Option: The proposer gets 8,000 won and the responder gets 2,000 won.• Coin Option: Toss a fair coin. If heads, the 5-5 option applies; if tails, the 8-2 option applies.The responder can accept or reject the proposer’s offer. If the responder accepts the proposed 5-5 or 8-2 option, the money is split as offered. If the Coin Option is proposed and the responder accepts, the experimenter tosses the coin in front of the responder and they split the money according to the coin’s outcome. If the responder rejects any offer, both proposer and responder receive 0 won. The rules are provided to the participants in advance.[Experiment 2] Everything is the same as [Experiment 1] except for the following. If the proposer chooses the Coin Option, the experimenter tosses the coin without the responder’s knowledge and informs the responder that either the 5-5 or 8-2 option has been offered, depending on whether the coin came up heads or tails. For example, if the responder is offered the 8-2 option, they do not know whether the proposer selected it directly or the coin toss resulted in the 8-2 outcome.* Expected payoff: The sum of the values obtained by multiplying the “payoff received in each scenario due to an action” by the “probability the person believes each scenario will occur.”<question> Which of the following <statements> is/are appropriate evaluation(s) of the passage above? Choose as listed from among <statements>.<statements>(a) If 60% of responders rejected the 8-2 offer in [Experiment 1], (Hypothesis 1) is weakened.(b) In [Experiment 1], if a proposer believes that the probabilities of the responder accepting the 5-5, 8-2, and Coin Options are 100%, 20%, and 80%, respectively, and chooses the Coin Option, (Hypothesis 1) is strengthened.(c) If participants believe that the 5-5 and Coin Options are fair but the 8-2 option is unfair, and if the rate of proposers choosing the 8-2 Option is higher in [Experiment 2] than in [Experiment 1], then (Hypothesis 2) is strengthened. <choices> ① (a) ② (c) ③ (a), (b) ④ (b), (c) ⑤ (a), (b), (c).

Expert rationale:

- preliminaries: (Hypothesis 1) is a common-sense view that people act to maximize their expected payoff. (Hypothesis 2) states that people will choose unfair actions that benefit themselves when they think such unfairness is unlikely to be detected by the other party.In [Experiment 1], the proposer chooses one of three possible options:5-5 Option: Both proposer and responder get 5,000 won.8-2 Option: The proposer gets 8,000 won, the responder gets 2,000 won.Coin Option: A fair coin is tossed. If heads, the 5-5 Option is selected; if tails, the 8-2 Option is selected.The responder can accept or reject the offer. If rejected, both get 0 won. If the proposer selects the 5-5 or 8-2 Option and the responder accepts, the money is split as proposed.If the proposer selects the Coin Option and the responder accepts, the experimenter tosses the coin in the responder’s presence; if heads, they split 5,000 won each; if tails, 8,000 won to the proposer and 2,000 won to the responder.In [Experiment 2], just like [Experiment 1], the proposer selects one of the three options. However, if the Coin Option is selected, the experimenter tosses the coin without the responder’s knowledge; if heads, informs the responder the 5-5 Option is being offered; if tails, informs the responder of the 8-2 Option. Everything else proceeds as in [Experiment 1].For example, when the responder is offered the 8-2 Option, they do not know if the proposer chose it directly or if it was the result of a coin toss. The proposer also knows that the responder cannot distinguish the source of the offer.
- statement_1_rationale: (a) In [Experiment 1], if the responder is offered the 8-2 Option, their expected payoff is 0 won if they reject and 2,000 won if they accept. According to (Hypothesis 1), which states people act to maximize expected payoff, the responder should accept the 8-2 offer. Therefore, if a majority—60% of responders—reject the offer, (Hypothesis 1) is weakened. Thus, (a) is an appropriate evaluation.
- statement_2_rationale: (b) In [Experiment 1], if a proposer believes the responder will accept the 5-5, 8-2, and Coin Options with probabilities 100%, 20%, and 80%, respectively, the expected payoffs for each are:5-5 Option: 5,000 × 1.00 = 5,000 won8-2 Option: 8,000 × 0.20 = 1,600 wonCoin Option: {(5,000 × 0.5) + (8,000 × 0.5)} × 0.8 = (2,500 + 4,000) × 0.8 = 6,500 × 0.8 = 5,200 wonIf the proposer chooses the Coin Option, which yields the highest expected payoff, this aligns with (Hypothesis 1)—the principle of maximizing expected payoff. So (b) is an appropriate evaluation.
- statement_3_rationale: (c) In [Experiment 1], the responder knows which option the proposer chose. In [Experiment 2], however, when offered the 8-2 Option, the responder cannot know if it was chosen directly by the proposer or resulted from the coin toss. Therefore, in [Experiment 2], the proposer can obscure the fact that they offered the 8-2 Option directly. That is, the proposer can believe that their unfair action (offering the 8-2 Option) is less likely to be detected by the other party. In [Experiment 1], such belief is not possible. Thus, if the rate of proposers choosing the 8-2 Option is higher in [Experiment 2] than in [Experiment 1], (Hypothesis 2) is strengthened. So (c) is also an appropriate evaluation.
- answer: All of (a), (b), and (c) are appropriate evaluations, so the correct answer is ⑤.

#### Question 2023_22

Question text:

According to 'conceptual role semantics', understanding the meaning of a word depends on the ability to follow the usage rules of that word. To follow the usage rules of a word does not merely mean to use the word according to those rules, but to use it based on an understanding of those rules. If that is the case, failing to understand the usage rules of a word means failing to understand the meaning of that word. However, in order to refute this theory, 'the following argument' has been raised. Let us suppose we try to understand the meaning of the word 'ppyojukhada' (which means 'to be sharp' or 'pointed'). According to this theory, in order to understand the meaning of the word, one must understand the usage rules of the word. However, for such understanding to be established, those rules must first be expressed linguistically and clearly, for example, as in "Apply 'ppyojukhada' to objects with a very thin end that can pierce through something." The problem, however, is that several words have been used even in the expression of this rule. In order to understand this rule, one would have to understand the meanings of all the words included, and for instance, if one does not understand the meaning of 'to pierce' as it appears in this rule, one cannot understand the rule. Then, what is required to understand the meaning of 'to pierce'? It would be an understanding of the usage rules for that word. But the usage rules for the word 'to pierce' will also be composed of several words, and to understand that rule, one must again understand the meaning of the words used to express those rules, and this kind of regression will continue infinitely. The occurrence of such infinite regression means that ultimately, we cannot understand the meaning of the word 'ppyojukhada', and such a problem would arise for all other words as well. Therefore, if we accept conceptual role semantics, we end up with the very unreasonable conclusion that no one truly understands the meaning of any word that we use. <question> Which of the following is NOT a correct assessment of the passage? <statements> (1) Even if an artificial intelligence appears that can use Korean perfectly, as well as a native Korean, conceptual role semantics is not weakened. (2) If the rules for word usage do not necessarily have to be expressed linguistically, the following argument is weakened. (3) If there is in fact a person who understands the meaning of every word included in “Apply 'ppyojukhada' to objects with a very thin end that can pierce through something,” conceptual role semantics is strengthened. (4) If it is possible to understand the meaning of a statement even if it contains words whose meanings one does not understand, the following argument is weakened. (5) Even if there is an agent who does not understand the meaning of a word but can use it according to the usage rules, conceptual role semantics is not weakened. <choices> ①(1) ②(2) ③(3) ④(4) ⑤(5).

Expert rationale:

- preliminaries: Conceptual role semantics makes the following claim: To understand the meaning of a word is to follow the usage rules of that word. To follow the usage rules of a word is to use it in accordance with those rules and to understand those rules. Therefore, according to conceptual role semantics, if one does not understand the usage rules of a word, one cannot understand the meaning of the word. The structure of the counterargument to conceptual role semantics is as follows: A language user must understand the usage rules of a word to understand the word’s meaning. (Claim of conceptual role semantics, assumption for reductio) To understand the usage rules of a word, the usage rules must be linguistically clear. Expressing the usage rules linguistically involves including many words. To understand the usage rules, one must understand the meaning of all the words included in those rules. To understand the meaning of the words included in the rules, one must understand the usage rules of those words. Steps 2–5 repeat, resulting in an infinite regress. If an infinite regress arises, one cannot understand the meaning of any word. Therefore, no one can understand the meaning of any word (from 6 and 7). That no one can understand the meaning of any word is highly unreasonable. Therefore, the claim that a language user must understand the usage rules of a word to understand its meaning (the claim of conceptual role semantics) cannot be accepted.
- statement_1_rationale: ① Whether or not an AI that uses Korean as well as a native understands Korean meanings based on rules is the key issue, so the mere existence of such an AI is irrelevant to the core claim of conceptual role semantics. Therefore, it is not weakened. ① is a correct assessment.
- statement_2_rationale: ② If it is not necessary for word usage rules to be expressed linguistically, premise 2 of the refutation argument is denied, so (c) is weakened. ② is a correct assessment.
- statement_3_rationale: ③ Conceptual role semantics is a theory that attempts to explain what it means to understand the meaning of a word, provided that there exist people who do understand the meaning of words. Therefore, the mere fact that there exists someone who understands the meaning of every word included in (c) does not strengthen or weaken the theory. Furthermore, the existence of such a person can be seen as evidence supporting premise 9 of the refutation argument against conceptual role semantics, so it cannot strengthen the theory. Thus, ③ is NOT a correct assessment.
- statement_4_rationale: ④ If it is possible to understand the meaning of a statement containing words whose meanings are unknown, premise 4 of the refutation argument is denied, so (d) is weakened. ④ is a correct assessment.
- statement_5_rationale: ⑤ A true counterexample to conceptual role semantics would be a person who understands the meaning of a word without understanding its usage rules. The scenario in which an agent does not understand the meaning of a word but uses it according to the rules is therefore not a counterexample, and thus does not weaken the theory. ⑤ is a correct assessment.
- answer: Only ③ is the incorrect analysis among the <choices>, so the answer is ③.
