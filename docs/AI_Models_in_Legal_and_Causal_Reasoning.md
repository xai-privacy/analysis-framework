# How Well Do Major AI Models Perform on Legal and Causal Reasoning Tasks?

## Abstract

Major AI models have made substantial progress on legal and reasoning-oriented tasks. They can
summarize legal documents, draft legal-style arguments, answer many benchmark questions, and often
produce fluent explanations that appear logically structured. However, this progress should not be mistaken
for a complete solution to legal reasoning, causal reasoning, or formal logic. The current state is best
described as partial competence with important blind spots. Models are no longer at the beginning: they
can perform many useful tasks, especially when the problem is narrow, well-scaffolded, and similar to
patterns seen during training. But they are also far from fully reliable reasoners. Their weaknesses become
visible when tasks require consistency under prompt variation, correct application of novel rules, rigorous
counterfactual reasoning, grounding in legal authority, or faithful causal explanation. Legal reasoning is
further along than causal reasoning, but neither is solved. The most difficult and important open problem is
the intersection of the two: legal-causal reasoning, where a model must not only produce the correct legal
answer but also follow the correct causal structure behind the legal rule.

## 1. Introduction

The question “How do major AI models perform in legal and causal reasoning tasks?” cannot be answered
with a simple yes or no. The answer depends on what we mean by legal reasoning, what we mean by
causal reasoning, and what standard we use for saying a problem is “resolved.”

Modern large language models can produce answers that look very impressive. They can write legal
memos, summarize cases, explain statutes, identify issues, and produce arguments in familiar legal formats
such as IRAC: Issue, Rule, Application, Conclusion. They can also talk fluently about causality and produce
explanations involving words such as “because,” “therefore,” “caused,” and “would have happened.” But
producing reasoning-like language is not the same as reasoning reliably.

This distinction matters because legal and causal reasoning are high-stakes forms of reasoning. In law, the
validity of an answer depends not only on the final conclusion but also on whether the conclusion follows
from the correct legal rule, the correct facts, and the correct application of doctrine. In causal reasoning, the
issue is not merely whether two events are associated, but whether changing one factor would actually
change the outcome.

Therefore, the central claim of this paper is:

       Major AI models are useful and increasingly capable in legal and causal reasoning tasks, but
       these are not solved problems. Legal reasoning is somewhere in the middle: models are
       strong assistants but weak autonomous legal reasoners. Causal reasoning is less mature:

       models are often good at association but still unreliable at intervention and counterfactual
       reasoning. Legal-causal reasoning remains an early-stage research frontier.

## 2. Definitions and Background

### 2.1 What counts as a “major AI model”?

In this discussion, “major AI models” refers to large language models and instruction-tuned models that are
widely used for general reasoning, professional assistance, and domain-specific tasks. Examples include
frontier commercial systems such as GPT-4-class models, Claude models, Gemini models, and open-source
or open-weight models such as Llama, Qwen, Mistral, Phi, and DeepSeek variants.

These models are trained primarily to predict and generate text. They learn patterns from massive corpora
and are then instruction-tuned to follow user requests. This gives them broad linguistic and reasoning-like
capability, but it does not automatically give them formal legal reasoning, causal world models, or
guaranteed logical consistency.

### 2.2 What is legal reasoning?

Legal reasoning is not one skill. It is a bundle of different tasks, including:

     1. Legal text classification
        Example: determining whether a clause is an arbitration clause.

     2. Issue spotting
        Example: identifying which legal issues arise from a fact pattern.

     3. Rule recall
        Example: recalling the elements of negligence or contract formation.

     4. Rule interpretation
        Example: determining what a statute means in context.

     5. Rule application
       Example: applying a legal standard to a new factual scenario.

     6. Analogical reasoning
        Example: comparing a current case to prior cases.

     7. Legal drafting and summarization
        Example: summarizing a brief or drafting a memo.

     8. Legal outcome prediction
        Example: predicting whether a claim will succeed.

     9. Legal-causal reasoning
        Example: determining whether a legally relevant fact caused the legal outcome.

The distinction is important because models can be good at some legal tasks and weak at others. A model
may summarize legal text well but still fail to apply a statute correctly to a novel fact pattern. It may recall a
rule but fail to identify which fact triggers that rule. It may produce a correct answer but justify it with
incorrect or hallucinated authority.

### 2.3 What is causal reasoning?

Causal reasoning asks whether one thing actually causes another, not merely whether the two are
associated.

A useful framework is Pearl’s ladder of causation:

            Type of
  Level                           Question                               Example
            reasoning

  Level
            Association           What is observed?                      Are X and Y correlated?

  Level                                                                  What happens to Y if we force X to
            Intervention          What happens if we do X?
  2                                                                      change?

  Level                           What would have happened               Would Y have occurred if X had not
            Counterfactual
  3                               otherwise?                             happened?

Large language models are generally strongest at Level 1 association. They are trained on text and
therefore learn statistical relationships between concepts. For example, they know that “rain” and “wet
roads” are related, or that “discrimination” and “protected class” are related in legal contexts.

They become less reliable at Level 2 and Level 3. Intervention and counterfactual reasoning require more
than recognizing patterns. The model must reason about what would change if a variable were actively
changed, while holding other factors fixed. That is much closer to formal causal reasoning than ordinary
text prediction.

### 2.4 What is logic?

Logic is formal rule-following. In a logical system, a conclusion follows from premises according to valid
rules of inference.

For example:

  Premise 1: If Z = 1, then Y = DENIED.
  Premise 2: Z = 1.
  Conclusion: Y = DENIED.

A fully reliable logical reasoner should produce the same conclusion every time the same premises are
given. It should not be distracted by irrelevant wording, stylistic changes, or facts that do not affect the rule.

LLMs can often imitate logical reasoning and solve many short reasoning problems, but they remain brittle
under longer chains, adversarial phrasing, hidden assumptions, or tasks requiring strict global consistency.
This is why they are not equivalent to theorem provers or symbolic solvers.

### 2.5 What does it mean for a model to “understand” causality or law?

The word “understand” is difficult. In practical terms, a model would demonstrate meaningful
understanding of legal causality if it could:

     1. Identify the legally relevant causal variables.
     2. Distinguish causal facts from irrelevant or merely correlated facts.
     3. Apply the legal rule consistently.
     4. Change its answer when a causal fact changes.
     5. Keep its answer stable when irrelevant facts change.
     6. Explain the reasoning using the correct legal doctrine.
     7. Maintain this behavior under paraphrase, distribution shift, and novel scenarios.

Most current models do not satisfy all these requirements reliably. They may pass clean examples but fail
under stress tests. Therefore, it is safer to say that current models show partial causal competence, not
robust causal understanding.

## 3. Legal Reasoning: Strong Surface Capability, Weak Autonomy

### 3.1 What models do well in legal tasks

Major AI models are already useful in many legal workflows. They can summarize long documents, extract
important facts, draft legal-style text, organize arguments, and answer many structured legal questions.
They can also follow common legal formats such as IRAC and produce professional-sounding explanations.

For example, if given a contract, a model may be able to identify termination clauses, summarize
obligations, and flag potentially relevant provisions. If given a case, it may summarize the holding and
identify key facts. If asked a general legal question, it may provide a plausible overview of the legal doctrine.

These are valuable capabilities. They can improve productivity in legal research, compliance review, contract
analysis, and first-pass issue spotting.

However, these tasks mostly involve text processing plus pattern recognition. They do not necessarily
prove that the model can reason like a lawyer.

### 3.2 Why legal benchmark success can be misleading

Some legal benchmarks show strong performance by frontier models. However, a high score on a
benchmark does not mean legal reasoning is solved.

There are several reasons.

First, many legal benchmarks test the final answer, not the reasoning process. A model may get the answer
right because the problem resembles something in training data, because the answer is statistically
common, or because it recognizes surface cues.

Second, many legal benchmark questions are shorter and cleaner than real legal problems. Real legal work
involves ambiguous facts, incomplete records, conflicting authorities, jurisdictional differences, and
strategic interpretation.

Third, legal reasoning often depends on the quality of the reasoning chain. A model can reach the correct
conclusion through invalid reasoning. In law, that is a serious problem because the explanation is often part
of the legal product.

Fourth, models may hallucinate legal authority. They can cite non-existent cases, misstate statutes, or
confidently assert rules that are not applicable. This makes them risky as autonomous legal agents.

Therefore, legal benchmark success should be interpreted as evidence of useful capability, not as proof of
reliable legal reasoning.

### 3.3 Legal reasoning failure modes

The main legal reasoning failure modes are:

  1. Hallucinated legal authority
  2. Incorrect citation or misquotation
  3. Confusion between jurisdictions
  4. Misinterpretation of statutory language
  5. Failure to identify controlling facts
  6. Overconfident legal conclusions
  7. Inconsistent answers under prompt variation
  8. Correct conclusion with invalid reasoning
  9. Weak handling of novel or edge-case fact patterns

These failures show why models are best understood as legal assistants rather than autonomous legal
decision-makers.

A useful phrase is:

       Current models are strong legal drafting and analysis aids, but weak autonomous legal
       reasoners.

## 4. Causal Reasoning: The Deeper Unsolved Problem

### 4.1 Association is not causation

The most important distinction in causal reasoning is the difference between association and causation.

An association means two things occur together. Causation means changing one thing would change the
other.

For example:

  Association: People carrying umbrellas are often seen when streets are wet.
  Causation: Carrying umbrellas does not cause wet streets; rain causes both.

LLMs are very good at learning associations because they are trained on statistical patterns in text. But
causal reasoning requires more than association. It requires a model of how variables affect one another.

This is where current AI models remain weak.

### 4.2 Intervention and counterfactual reasoning

Intervention asks:

  What happens if I change X?

Counterfactual reasoning asks:

  What would have happened if X had been different?

These are central to law, science, medicine, policy, and compliance.

For example, in a hiring discrimination case, a causal question might be:

  Would the applicant have been rejected if their protected attribute had been
  different?

This is not just a classification question. It requires reasoning about an alternative world where one fact
changes and other relevant conditions remain fixed.

LLMs often struggle here because they do not naturally maintain a formal causal graph. They may answer
based on the most familiar textual pattern rather than performing the intervention or counterfactual
calculation.

### 4.3 Why causal reasoning is hard for LLMs

Causal reasoning is hard for LLMs because their core training objective is not causal inference. They learn to
predict text, not to compute interventions over structural causal models.

A model may know the sentence:

  Smoking causes cancer.

But that does not mean it can correctly reason through a novel causal graph with confounding, mediation,
and counterfactual conditions.

For causal reasoning, the model must often:

- Identify variables
- Determine causal direction
- Separate cause from correlation
- Ignore irrelevant variables
- Handle confounders
- Evaluate interventions
- Compare actual and counterfactual worlds
- Maintain consistency across scenarios

These are not guaranteed by next-token prediction.

## 5. Logic and Formal Reasoning: Capable but Brittle

Major AI models can solve many logic-like problems, especially short ones. They can follow simple
syllogisms, solve basic math problems, and produce step-by-step explanations. Reasoning-tuned models
improve this further by spending more computation on intermediate steps.

But logic remains brittle. Models can lose track of assumptions, contradict earlier statements, accept invalid
steps, or fail when a problem is adversarially phrased.

The difference between an LLM and a formal solver is important.

An LLM predicts a likely answer from text. A solver computes the answer from rules.

For example:

  def lost_profit_solver(X, Z):
      if Z == 1:
          return "DENIED"
      if X == 1 and Z == 0:
          return "AWARDED"
      return "DENIED"

This solver does not guess. If the input is X=1, Z=0 , it deterministically returns AWARDED . If the input is
Z=1 , it returns DENIED .

An LLM may produce the same answer, but it may also be distracted by wording, prior associations, or
default tendencies. This is why solver-assisted architectures are promising for legal-causal reasoning.

## 6. The Intersection: Legal-Causal Reasoning

The most important research frontier is not legal reasoning alone or causal reasoning alone. It is their
intersection.

Legal-causal reasoning occurs when a legal outcome depends on a causal relationship.

Examples include:

  Tort law: Did the defendant’s act cause the plaintiff’s injury?
  Employment discrimination: Would the adverse decision have occurred but for a
  protected attribute?
  Privacy law: Was the use of sensitive data necessary to provide the service?
  Patent damages: Would the patentee have made the sales but for the infringement?
  Contract damages: Did the breach cause the claimed loss?

These are not merely language tasks. They require identifying legally relevant causal variables and applying
a rule consistently.

A model that produces a fluent explanation but fails to track the causal variable is not reliable.

## 7. Example: Patent Lost-Profits Causality

Consider a simple legal-causal rule from patent lost-profits damages.

Let:

  X = infringing product is available
  Z = third-party non-infringing substitute is available
  Y = outcome: AWARDED or DENIED

The rule is:

  If Z = 1, the lost-profits claim is DENIED.
  If X = 1 and Z = 0, the lost-profits claim is AWARDED.

Now compare two cases:

  Case A:
  X = 1
  Z = 0
  Correct outcome: AWARDED

  Case B:
  X = 1
  Z = 1
  Correct outcome: DENIED

Only one fact changed: the substitute product became available. A causally aligned model should change its
answer from AWARDED to DENIED .

This is a simple example of a counterfactual legal benchmark. The question is not only whether the model
can answer one prompt correctly. The question is whether it changes its answer correctly when the causal
fact changes.

## 8. Empirical Illustration: Qwen Model Results

A small experiment using the analysis_framework benchmark shows why this distinction matters.

The benchmark tested six paired scenarios using the patent lost-profits rule above.

The results were:

  Qwen2.5-0.5B-Instruct:
  Causal violation score = 0.67
  Verdict = FAIL
  Observed behavior = always predicted DENIED

  Qwen2.5-1.5B-Instruct:
  Causal violation score = 0.00
  Verdict = PASS
  Observed behavior = correctly predicted AWARDED for X=1, Z=0 and DENIED for Z=1

The 0.5B model failed in an informative way. It did not produce random errors. Instead, it collapsed to a
simple policy:

  Always answer DENIED.

This policy works for all cases where Z=1 , because the presence of a substitute product means the claim
should be denied. But it fails whenever X=1 and Z=0 , because those cases require AWARDED .

This is shortcut behavior. The model learned or followed a safe-looking legal default, but it did not apply the
full causal rule.

The 1.5B model performed better. It correctly changed its answer depending on the causal variable. This
suggests that modest increases in model capacity and instruction-following ability can improve
performance on clean legal-causal reasoning tasks.

However, the conclusion must be cautious. Passing six clean benchmark pairs does not prove general causal
understanding. It shows that the model can follow an explicitly provided causal rule in a controlled setting.
More difficult tests are needed.

## 9. Are These Problems Resolved?

The answer is no.

Legal reasoning is not resolved because models still fail on grounding, authority, consistency, subtle
doctrine, and reasoning-chain validity.

Causal reasoning is not resolved because models still struggle with intervention, counterfactuals,
confounding, and novel causal structures.

Logic is not resolved because models remain brittle under long chains, adversarial examples, and tasks
requiring strict formal consistency.

Legal-causal reasoning is especially unresolved because it combines all of these difficulties.

The best answer is:

       We are somewhere in between. Major AI models are far beyond the beginning for legal
       assistance and simple reasoning, but they are still far from robust legal-causal
       understanding.

## 10. A Maturity Model

The current state can be organized into levels.

Level 1: Pattern matching

Models are strong here.

Examples:

- Summarizing documents
- Recognizing common legal phrases
- Producing standard legal explanations
- Answering familiar questions

Level 2: Structured reasoning with scaffolding

Models are often useful here.

Examples:

- Applying an explicitly provided rule
- Following a structured prompt
- Producing a controlled JSON answer
- Answering simple counterfactual pairs

Level 3: Robust causal and legal reasoning

Models are not reliably here yet.

Examples:

- Handling novel statutes
- Maintaining consistency under paraphrase

- Ignoring irrelevant distractors
- Performing intervention reasoning
- Performing counterfactual reasoning
- Producing legally valid reasoning chains
- Explaining decisions without hallucinated authority

This maturity model explains why models can look strong in some settings and weak in others.

## 11. Why Scaling Alone Is Not Enough

Larger models often perform better than smaller models. The Qwen example supports this: the 1.5B model
passed a task that the 0.5B model failed.

But scaling alone is unlikely to fully solve legal-causal reasoning.

The reason is structural. LLMs are trained primarily on text prediction. They do not automatically maintain
formal causal graphs, legal rule engines, or proof systems. Larger models may approximate these
behaviors better, but approximation is not the same as reliability.

For high-stakes legal or compliance use cases, we need more than plausible answers. We need verifiable
reasoning.

## 12. The Likely Path Forward: Hybrid Systems

The most promising direction is hybrid AI architecture.

A reliable legal-causal AI system should not depend only on an LLM’s internal reasoning. Instead, it should
combine:

  1. LLMs for natural language understanding
  2. Retrieval systems for grounding in legal sources
  3. Structured representations such as DAGs or SCMs
  4. Rule-based solvers for deterministic legal logic
  5. Formal verification for checking reasoning
  6. Human expert review for high-stakes decisions

In such a system, the LLM would read and structure the facts. The solver would apply the rule. The verifier
would check consistency. The human expert would review the result.

For example:

  Legal text → LLM extracts X and Z → solver applies causal rule → explanation
  generated → verifier checks consistency

This approach reduces the risk that the LLM will produce a plausible but causally wrong answer.

## 13. Research Implications

The key research implication is that future evaluation should move beyond ordinary answer accuracy.

Current benchmarks often ask:

  Did the model give the right answer?

A stronger legal-causal benchmark asks:

  Did the model give the right answer for the right causal reason?

This requires counterfactual testing.

For every scenario, we can create a paired scenario where only one causal variable changes. Then we test
whether the model’s answer changes correctly.

A good benchmark should also include:

- Paraphrases
- Distractors
- Ambiguous facts
- Negation
- Long fact patterns
- Multiple jurisdictions
- Irrelevant variables
- Solver-generated ground truth

This would expose whether a model is truly tracking the legal causal structure or merely responding to
surface patterns.

## 14. Practical Answer to the Original Question

So how do major AI models do in legal and causal reasoning tasks?

They do well on many surface and structured tasks. They are useful for summarization, drafting,
classification, issue spotting, and first-pass legal analysis. They can often follow explicit rules in clean
prompts. Larger and reasoning-tuned models are increasingly capable.

But they are not reliable autonomous reasoners. They remain vulnerable to hallucination, inconsistency,
prompt sensitivity, shallow reasoning, and shortcut behavior. They struggle most when the task requires
true intervention or counterfactual reasoning.

Are these resolved problems?

No.

Are we just at the beginning?

Not exactly. For general legal AI assistance, we are beyond the beginning. The tools are already useful. For
causal reasoning and legal-causal reasoning, however, we are still early.

Are we somewhere in between?

Yes. That is the most accurate answer.

The field is in a middle stage for legal reasoning and an early-to-middle stage for causal reasoning. The
frontier is moving from fluent answer generation toward verifiable reasoning systems.

## 15. Conclusion

Major AI models have become impressive legal and reasoning assistants, but they do not yet reliably
understand causality, logic, or legal rule structure in the strong sense required for high-stakes decision-
making.

Legal reasoning is partially mature: models can summarize, draft, classify, and answer many structured
questions. But they still fail on grounding, subtle doctrine, consistency, and valid reasoning chains.

Causal reasoning is less mature: models can discuss cause and effect, but they remain weak at
interventions, counterfactuals, and formal causal inference.

Legal-causal reasoning is the critical open problem. Many legal outcomes depend on causal facts. A model
that cannot reliably track those facts may produce correct-looking but legally invalid conclusions.

The central lesson is:

       The next stage of AI progress should not be measured only by whether models sound more
       intelligent. It should be measured by whether their answers are grounded, consistent,
       causally valid, legally faithful, and verifiable.

For that reason, the most promising path is not simply larger models. It is the combination of LLMs with
structured legal representations, causal graphs, rule-based solvers, retrieval, formal verification, and expert
oversight.

In short: major AI models are useful, impressive, and improving — but legal and causal reasoning are not
solved. We are somewhere in between, and the hardest work is still ahead.
