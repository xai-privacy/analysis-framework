# Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness on Tax Law - Paper Summary

## Introduction

"Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness on Tax Law" - this paper investigates whether large language models are genuinely good at legal reasoning or whether their strong performance on legal benchmarks is partly caused by memorization and benchmark contamination. The authors focus on **tax law**, because tax law is rule-based, structured, and often requires exact calculations. This makes it a good domain for testing whether AI models can truly follow legal rules.

The central question of the paper is: "Are LLMs better understood as legal reasoners, or are they more reliable as translators that convert natural language into formal logic for a solver?" The paper shows that LLMs are useful, but they are not fully reliable standalone legal reasoners. They are often more reliable when used as part of a "neuro-symbolic system", where the LLM translates legal text into formal logic and a symbolic solver such as Prolog performs the actual reasoning.

The paper finds that:

- Direct LLM performance on legal benchmarks may be inflated by data contamination.
- LLMs can perform well on original benchmark examples but become unstable when rules or cases are changed.
- LLMs are relatively robust to simple paraphrasing.
- LLMs struggle more with numerical legal reasoning than with simpler entailment tasks.
- Prolog-based neuro-symbolic systems are more stable and robust, especially for numerical tax reasoning.
- The best role for LLMs in high-stakes legal reasoning may be as language-to-logic translators, not as final decision-makers.

The paper’s main conclusion is that legal reasoning is compositional, structured, and high-stakes. Therefore, reliable legal AI should combine LLMs with symbolic reasoning, verification, and possibly human-in-the-loop review.


## Motivation

Legal reasoning is important because mistakes can have serious consequences. In tax law, an incorrect answer can affect someone’s tax liability, benefits, financial aid, credit, or legal compliance. LLMs have become very strong at generating legal-sounding text. They can summarize laws, answer questions, and explain rules. However, the paper argues that this does not prove they are actually reasoning correctly. There are two major concerns:

**Hallucination** : LLMs may invent facts, rules, or legal conclusions that are not supported by the input. For example, if the law does not say Alice qualifies for a deduction, the model might still say she qualifies because it has seen similar examples before.

**Data contamination** : Data contamination means that the model may have seen the benchmark examples during training. If a model performs well on a public benchmark, it may not be because it learned to reason. It may be because the benchmark or similar examples were included in its training data. This is especially important for legal datasets because many legal benchmarks are public and may have been scraped into web-scale training corpora. The paper asks whether high performance on tax-law benchmarks reflects real reasoning or memorization.


## Main Research Questions

The paper is organized around four main questions.

**Q1: Which approach is more effective for legal reasoning?**

The authors compare:

- **Monolithic LLMs**, where the LLM directly answers the legal question.
- **Neuro-symbolic models**, where the LLM translates text into formal logic and a solver performs reasoning.

A monolithic LLM is a system where the LLM does everything itself. The LLM reads the law, understands the facts, performs the reasoning, does the calculation, and gives the final answer. The problem is that the model may hallucinate, miscalculate, misread the statute, or rely on memorized examples.

A neuro-symbolic system combines a neural model, such as an LLM and a symbolic reasoning system, such as Prolog. Prolog applies formal tax rules and computes the answer. The advantage is that Prolog is deterministic and verifiable. It does not guess. It applies the rules exactly. The paper argues that this separation is valuable as  LLM is good at language understanding and Prolog is good at exact rule-based reasoning


**Q2: Do new in-house experiments match prior results?**

The authors reproduce and extend prior results using newer state-of-the-art models, including GPT, Claude, Gemini, DeepSeek, Llama, and Qwen-style models.

**Q3: Is LLM performance inflated by contamination?**

The authors test whether models recognize original SARA benchmark examples, which would suggest they may have seen them during training.

**Q4: Do LLMs generalize under changed rules and changed cases?**

The authors create SARA+, a new benchmark with modified rules, modified cases, paraphrases, and combined changes to test whether models can reason beyond memorized examples.



## What the Paper Does

The paper does five major things.

- It studies tax-law reasoning: It focuses on statutory tax reasoning, where models must apply tax-law rules to case facts.

- It compares LLM-only and neuro-symbolic approaches.

- It tests data contamination: The authors design multiple-choice tests to check whether LLMs recognize original SARA examples.

- It creates SARA+ : The authors create new benchmark variants by modifying the original SARA dataset. SARA+ includes: Rule changes, Case numerical changes, Case paraphrases, Combined rule and case changes.

- It evaluates robustness and generalization: The authors test whether models still work when examples are no longer exactly like the original benchmark.


## Experiment Pipeline

The paper’s pipeline has two main stages.

**Stage 1: Case formalization** : The LLM reads a natural-language case and converts it into Prolog facts. For example:

Alice made $100,000 in 2015 -> income(alice, 2015, 100000).
Alice and Bob were married in 2015. -> married(alice, bob, 2015).

**Stage 2: Logic-based verification** : The Prolog solver combines:

1. Prolog facts from the case.
2. Prolog rules from the tax statutes.
3. A Prolog query.

Then it computes the answer. Example query: owes_tax(alice, 2015, Tax). Possible answer: Tax = 14000.

This means the LLM is not responsible for final reasoning. It is responsible for translating text into structured facts.


## Data Contamination Experiment

The goal is to check whether models may have memorized SARA examples. If a model can identify the exact original SARA text among several paraphrases, this suggests possible contamination.

 **Bias Detector Quiz** : In the Bias Detector Quiz, the model receives four paraphrased versions of a SARA example and a fifth option "None of the above". In this setting, the original example is not included, so the correct answer is "None of the above". This helps detect whether the model has positional bias, such as preferring A or C.

**BCQ: Bias Compensator Quiz** :  In the Bias Compensator Quiz, one of the paraphrased options is replaced with the exact original SARA example. If the model chooses the exact original, it may indicate that the model recognizes it from training. The authors use this method to estimate contamination while accounting for positional bias.

The paper finds that contamination varies widely across models. Some newer frontier models show high contamination estimates. For example, Gemini 3 Pro has very high contamination estimates, while older models such as GPT-4o and GPT-4.1 show lower estimates.

The important finding is: Contamination is strongly associated with high Direct QA performance, especially on entailment. This suggests that some strong results on original SARA may be inflated by memorization. However, contamination is less strongly related to Prolog-based performance. This suggests that solver-backed systems provide more stable and reliable evaluation.


## SARA+: New Test Data

SARA+ is created because the original SARA benchmark may be contaminated. Also, original benchmark examples may be too familiar to modern models. To test true generalization, the authors create SARA+. SARA+ introduces controlled changes to the original benchmark.

**Types of SARA+ variations** :

- Original SARA : This is the original dataset.

- Rule change : The legal rule is modified. For example:

  Original rule: threshold is $50,000. -> Changed rule: threshold is $60,000.
  This tests whether the model follows the new rule or relies on memorized old rules.

- Case numerical change : The facts of the case are modified. For example:

  Original case: Alice earned $100,000 -> Changed case: Alice earned $130,000.

  This tests whether the model recomputes the answer from the new facts.

- Case paraphrasing : The wording is changed while the meaning stays the same. For example:

  Alice earned $100,000 in 2015. -> Alice received $100,000 of income during 2015.

  This tests whether the model is robust to different wording.

- Rule and case change : Both the statute and the case are modified. This is the hardest setting because the model cannot rely on memorized rules or memorized cases.


## Experimental Setup


The paper evaluates a wide range of models, including:

- GPT models.
- OpenAI reasoning models.
- Claude models.
- Gemini models.
- DeepSeek models.
- Llama models.
- Qwen models.

The authors compare both older and newer models.

**Direct QA setting** : In Direct QA, the model receives the statutes, case, and question in natural language. It must directly answer the question. For entailment, the output is: Entailment or Contradiction. For numerical reasoning, the output is the tax amount.

**Prolog-based setting** : In the Prolog-based setting, the model translates the case into Prolog facts. The human-coded Prolog statutes are then combined with the generated facts. The Prolog solver executes the rules and returns the answer. This allows the paper to test whether LLMs are better as translators than as standalone reasoners.

**Metrics** : The paper uses different metrics for different tasks. "Entailment accuracy" is used for binary entailment tasks. "Exact match" is used for numerical tax computation. The model must produce the exact correct tax amount. Some prior work uses a relaxed metric where answers within 10% of the correct value count as correct. The paper notes that this metric is less strict and not directly comparable to exact match.

**Error / abstention** :  Some solver-based systems can abstain if generated code fails verification. This is useful because abstaining may be safer than giving a wrong answer.


## Experiment 1: Prior Baselines

The authors first review prior work to understand the existing state of tax-law reasoning. They compare prior direct LLM results, agentic solver-based methods, and Prolog-based methods. Prior work shows that strong reasoning LLMs can achieve high scores on original SARA, especially on entailment. For example, DeepSeek-R1 performs strongly on entailment. Other models such as GPT-4o, o1-preview, Qwen2-72B, and Llama3.1-405B are also included in prior comparisons. However, metrics differ across papers, making direct comparison difficult. Some solver-based approaches report relaxed numerical metrics such as M10%, while other results use exact match or mean squared error.


The authors conclude that prior work suggests direct LLMs can perform well, but existing comparisons are difficult because:

- Models differ.
- Metrics differ.
- Some tasks are easier than others.
- Contamination may inflate results.
- Solver-based methods may improve reliability but not always raw accuracy.

This motivates the authors’ in-house controlled evaluation.


## Experiment 2: In-House Direct QA vs Prolog

The goal is to compare direct LLM answering and Prolog-based reasoning using the same models and more consistent evaluation.

**Direct QA findings** : The paper finds that Direct QA performs well on entailment tasks. This means LLMs are good at deciding whether simple legal claims follow from given statutes and facts. However, Direct QA performs much worse on numerical tax reasoning. Numerical reasoning requires exact calculation, and LLMs vary widely in performance. For example:

- Some newer models achieve strong numerical performance.
- GPT-4o performs very poorly on numerical Direct QA.
- GPT-5.1 and Gemini 3 Pro perform much better.
- Llama 3.1 405B performs poorly on numerical Direct QA.

The main point is that numerical legal reasoning remains difficult for standalone LLMs.

**Prolog-based findings** : When the same models are used as translators into Prolog, performance improves substantially for numerical reasoning. This means that even if the LLM is not good at computing the answer directly, it may still be good enough to extract the facts into Prolog. Then Prolog performs the actual reasoning and arithmetic. The largest improvements occur for weaker Direct QA numerical models. For example, GPT-4o improves dramatically when used in the Prolog-based pipeline.

This experiment supports the paper’s title.

For difficult numerical legal reasoning, LLMs are often better as "translators into formal logic" than as "standalone reasoners".  Direct QA is strong for simpler entailment, but Prolog-based reasoning is more reliable for exact numerical computation.


## Experiment 3: Data Contamination Analysis

The goal is to determine whether high model performance on original SARA is partly due to contamination. The authors use contamination quizzes. They generate meaning-preserving perturbations of SARA examples. Then they ask models to identify the exact original example. If the model can reliably identify original text, it may have seen the benchmark before.

The paper finds substantial contamination signals in several modern models. Some newer frontier models show high contamination estimates, while some earlier models show lower estimates. The paper also finds that contamination correlates with Direct QA performance, especially on entailment. This means high Direct QA scores may partly reflect benchmark memorization.

This is one of the paper’s most important findings. The paper warns that benchmark performance can be misleading. A model may look like it is reasoning, but it may actually be recognizing examples. Therefore, legal AI evaluation must be contamination-aware.


## Experiment 4: Generalization on SARA+

The goal is to test whether models generalize when the rules or cases are changed. This is important because real legal reasoning requires applying rules to new cases, not just answering familiar examples.

**Direct QA results on SARA+** :  Direct QA performance drops when rules or numerical facts are changed. The drop is especially large for:

- Numerical reasoning.
- Entailment tasks that require numerical reasoning.
- Combined rule and case changes.

This suggests that monolithic LLMs rely partly on memorized patterns and struggle when the underlying legal logic changes.

**Prolog-based results on SARA+** :  Prolog-based performance remains much more stable across SARA+ variations. This is because once the facts are correctly translated, the solver applies the current rules exactly. Changing the rules or case values does not confuse the solver in the same way it confuses a direct LLM.

**Paraphrase results** : Both Direct QA and Prolog-based systems remain fairly stable under paraphrasing, especially when the reasoning is simple. This means LLMs are not mainly failing because of wording changes. They are failing more when the logical or numerical structure changes.

This experiment shows that LLMs are good at language variation but weaker at logical generalization. They can handle new wording, but they struggle when the actual legal computation changes. The neuro-symbolic approach generalizes better because the solver performs rule execution.


## Findings Summary

**Finding 1: LLMs are not fully reliable legal reasoners** : LLMs can answer many legal questions, but their performance is inconsistent and can be inflated by memorization.

**Finding 2: Contamination is a serious problem** : Some models appear to recognize original SARA examples, which means benchmark scores may not reflect true reasoning ability.

**Finding 3: Direct QA is strong on easy entailment** : When the task is binary and relatively simple, direct LLMs often do well.

**Finding 4: Direct QA is weak on exact numerical reasoning** : Tax law often requires exact calculations. LLMs struggle with this when they must reason directly.

**Finding 5: LLMs are better as translators** : When LLMs translate case facts into Prolog, the Prolog solver can compute answers more reliably.

**Finding 6: Neuro-symbolic systems are more robust** : When rules or cases are changed, Prolog-based systems remain more stable than direct LLMs.

**Finding 7: Legal reasoning needs compositional systems** : Legal reasoning is not just text generation. It requires combining facts, rules, calculations, and verification.


## Why This Paper Matters

This paper matters because it challenges the assumption that high LLM benchmark performance means true legal reasoning. It shows that models may perform well because of contamination or memorization, and that direct LLMs become less stable when rules or case facts change. The paper also shows that neuro-symbolic systems can improve reliability by separating language understanding from formal reasoning. This is important for high-stakes legal AI, where answers must be grounded, explainable, and verifiable.


## Conclusion

The paper shows that LLMs have real value in legal AI, but their role should be carefully defined.

They are good at:

- Reading legal text
- Translating natural language into structured representations
- Handling paraphrases
- Producing legal-style explanations

They are weaker at:

- Exact numerical reasoning
- Generalizing to changed rules
- Generalizing to changed case facts
- Avoiding contamination effects
- Providing verifiable reasoning as standalone systems

So the conclusion is: LLMs are not yet reliable standalone legal reasoners. They can sometimes reason directly, especially on simpler entailment tasks, but they are less reliable for numerical tax reasoning, changed rules, changed case facts, and contamination-free evaluation. The paper argues that LLMs are more robust when used as part of neuro-symbolic systems, where they translate legal language into formal logic and symbolic solvers perform the actual reasoning. For high-stakes legal AI, the most reliable path is not LLM-only reasoning, but solver-backed, verifiable, and human-reviewable systems.

## Bibliography

- Kordjamshidi, P., Aslan, S., Seshadri, M., Barrett, L., & Santus, E. (2026). *Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness on Tax Law*. Proceedings of the First Workshop on Structured Understanding, Retrieval, and Generation in the LLM Era (SURGeLLM 2026), 344–360. https://arxiv.org/abs/2605.16052
