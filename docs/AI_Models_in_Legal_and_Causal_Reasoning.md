# How Well Do Major AI Models Perform on Legal and Causal Reasoning Tasks?

## Introduction

Major AI models have made clear progress on legal and reasoning-oriented tasks. They can summarize legal documents, draft legal-style arguments, identify issues, answer many benchmark questions, and produce explanations that look logically structured. However, strong language ability should not be confused with reliable legal or causal reasoning. In high-stakes domains such as law, it is not enough for a model to give a plausible answer. The model must apply the correct rule to the correct facts, avoid hallucinated authority, remain consistent when wording changes, and correctly handle causal changes in the scenario.

The central question is whether current AI models truly understand legal causality and logic, or whether they mostly recognize patterns from training data and produce fluent answers. The answer is mixed: models are no longer at the beginning, because they are already useful for many legal-support tasks, but the problems of legal reasoning, causal reasoning, and legal-causal reasoning are not solved.


## How Well Do Major AI Models Perform on Legal and Causal Reasoning Tasks?

Major AI models perform well on many surface-level and structured legal tasks. They are useful for summarization, drafting, classification, issue spotting, legal document review, and first-pass legal analysis. They can often follow explicit rules in clean prompts, especially when the problem is short, well-scaffolded, and similar to patterns they have seen before. Larger models and reasoning-tuned models usually perform better than smaller models.

However, major AI models are still not reliable autonomous legal or causal reasoners. They can hallucinate legal authority, misstate statutes, apply the wrong rule, produce inconsistent answers under prompt variation, and reach the right answer for the wrong reason. Their weakness becomes more serious when the task requires counterfactual reasoning, intervention reasoning, novel rule application, or strict logical consistency.

Causal reasoning is less mature than general legal reasoning. Models are often good at recognizing associations, but they are weaker when asked what would happen if one causal variable changed while other facts stayed fixed. This is a major issue for legal AI because many legal outcomes depend on causality, such as whether a breach caused damages, whether discrimination caused an adverse decision, whether a privacy violation caused harm, or whether a substitute product defeats a patent lost-profits claim.

A small experiment illustrates the problem. In a legal-causal benchmark about patent lost-profits damages, Qwen2.5-0.5B-Instruct failed with a causal violation score of 0.67 because it always predicted DENIED. This was shortcut behavior: the model produced a legally safe-looking answer but did not apply the full causal rule. Qwen2.5-1.5B-Instruct passed the clean six-pair benchmark with a causal violation score of 0.00, correctly predicting AWARDED when X=1 and Z=0, and DENIED when Z=1. This result suggests that stronger instruction-tuned models may handle clean legal-causal rules better, but it should be interpreted cautiously because the benchmark is small and controlled. Passing six clean pairs does not prove general legal-causal reasoning.

Therefore, the best answer is: we are somewhere in between. AI models are already useful legal assistants, but they are not dependable legal decision-makers. Legal reasoning is in a middle stage: useful but not fully reliable. Causal reasoning and legal-causal reasoning are earlier and more difficult. The next step is not only bigger models, but better evaluation and more verifiable systems, including counterfactual tests, perturbation tests, retrieval grounding, symbolic solvers, formal verification, and human expert oversight.


## Related Work and Existing Research

Several studies have investigated how well LLMs support legal reasoning, statutory interpretation, tax-law reasoning, formal logic translation, and contamination-aware evaluation. These works show that LLMs can perform many useful legal-language tasks, but they also reveal important limitations: models may hallucinate, memorize benchmark examples, fail under rule or fact changes, and struggle with formal reasoning. The following papers provide the research foundation for understanding whether major AI models are true legal-causal reasoners or mainly powerful language processors that need solver-backed verification.

### 1. Legal Reasoning Benchmarks

#### Guha et al. (2023): LegalBench 

Guha et al. introduced LegalBench, a large benchmark for evaluating legal reasoning in LLMs. LegalBench includes many different legal task types, such as rule recall, rule application, legal interpretation, issue spotting, and rhetorical understanding. LegalBench showed that LLMs can perform many legal reasoning tasks, but performance varies significantly depending on the types of reasoning required. Models may do well on some legal tasks but struggle on others, especially tasks requiring careful legal interpretation or rule application.

#### Chalkidis et al. (2022): LexGLUE

Chalkidis et al. introduced LexGLUE, a benchmark for legal language understanding in English. It evaluates models on several legal NLP tasks. LexGLUE helped standardize legal NLP evaluation and showed that transformer models can perform well on legal classification and understanding tasks. However, these tasks are mostly language-understanding tasks rather than full formal legal reasoning tasks.

#### Shi et al. (2026): PLawBench

Shi et al. introduced PLawBench, a rubric-based benchmark for evaluating LLMs in real-world legal practice. The benchmark focuses on practical legal performance, showing that legal AI evaluation must consider real-world use cases and not only abstract benchmark tasks.

Using an LLM-based evaluator aligned with human expert judgments, they evaluate 10 state-ofthe-art LLMs. Experimental results show that none achieves strong performance on PLAWBENCH, revealing substantial limitations in the fine-grained legal reasoning capabilities of current LLMs and highlighting important directions for future evaluation and development of legal LLMs.

### 2. Legal Domain Models

#### Chalkidis et al. (2020): LEGAL-BERT

Chalkidis et al. developed LEGAL-BERT, a BERT-style language model trained or adapted for legal text. The work investigated whether domain-specific legal pretraining improves performance on legal NLP tasks. It showed that legal-domain pretraining helps models better handle legal language compared with general-purpose models. Legal language has specialized vocabulary, structure, and style, so models trained on legal corpora can perform better on legal classification and understanding tasks.


### 3. Tax-Law Reasoning

#### Holzenberger et al. (2020): SARA

Holzenberger et al. introduced SARA, the StAtutory Reasoning Assessment dataset. SARA focuses on U.S. tax-law reasoning. It includes natural-language statutes, natural-language case descriptions, formal Prolog representations, queries, and answers. SARA showed that tax law is a useful testbed for formal legal reasoning because tax statutes are structured and rule-based. The dataset allows researchers to compare natural-language reasoning with formal logic-based reasoning.

#### Nay et al. (2024): LLMs as Tax Attorneys

Nay et al. studied whether LLMs can perform tax-law reasoning tasks, treating LLMs like tax-law assistants or “tax attorneys.” The work reported strong LLM performance on some tax-law reasoning tasks, with results approaching high accuracy in some settings. This suggested that LLMs may have emerging capabilities in legal reasoning.

#### Hu et al. (2025): Test-Time Scaling LLMs for Legal Reasoning

Hu et al. evaluated newer reasoning-oriented LLMs, including OpenAI o-series models and DeepSeek-R1-style models, on legal reasoning tasks in both Chinese and English. Their evaluation included tax-law reasoning on SARA. It showed that "reasoning-optimized models" often perform better than ordinary instruction-following models. In the uploaded paper’s summary of prior results, DeepSeek-R1 performs strongly on SARA entailment and numerical reasoning.

Together, these tax-law studies show that tax law is a useful domain for evaluating legal reasoning because it combines statutes, facts, and numerical computation. They also show that high performance on tax-law benchmarks should be interpreted carefully because models may rely on familiar patterns rather than true rule application.

### 4. Hallucination and Reliability

#### Dahl et al. (2024): Large Legal Fictions

Dahl et al. studied hallucinations in legal LLM outputs. They investigated how often LLMs generate legal statements, citations, or claims that are unsupported or false. It found that LLMs can hallucinate legal content, including incorrect or nonexistent legal authorities. This is dangerous because legal users may trust confident but false outputs.

#### Magesh et al. (2025): Reliability of AI Legal Research Tools

Magesh et al. evaluated the reliability of leading AI legal research tools and whether they are truly hallucination-free. They found that even specialized legal AI tools can produce unreliable or hallucinated outputs. Legal AI systems may sound authoritative while still making factual or legal errors.

### 5. Symbolic and Neuro-Symbolic Legal Reasoning

#### Buchanan and Headrick (1970): Early AI and Legal Reasoning

Buchanan and Headrick discussed early possibilities for using artificial intelligence in legal reasoning. They helped establish the idea that legal reasoning can be modeled computationally, especially when legal rules can be made explicit.

#### Sergot et al. (1986): British Nationality Act as a Logic Program

Sergot et al. represented the British Nationality Act as a logic program. They showed that legal statutes can sometimes be encoded as formal logic rules and executed computationally.

#### Schild (1990): Open-Textured Law and Logic Programming

Schild studied how logic programming can handle legal reasoning, including the difficulty of open-textured legal terms. Their finding is that logic programming is useful for formal legal rules, but real legal language often includes ambiguous or open-textured concepts that are hard to formalize.

#### Jurayj et al. (2025): Language Models and Logic Programs for Trustworthy Tax Reasoning

Jurayj et al. developed a Prolog-based framework for tax-law reasoning. Their system uses LLMs to generate formal logic programs and uses Prolog to reason over them. Their results demonstrated the effectiveness of applying semantic parsing methods to statutory reasoning, and showed promising economic feasibility of neuro-symbolic architectures for increasing access to reliable tax assistance.

#### Lorenzo et al. (2025): Translating Tax Law to Code with LLMs

Lorenzo et al. studied the translation of tax law into executable code using LLMs. They found that LLMs can assist with converting legal rules into code, but translating complex legal text into executable programs remains challenging.

#### Sadowski and Chudziak (2025): SOLAR / Multi-Agent Verifiable Legal Reasoning

Sadowski and Chudziak developed an agentic legal reasoning framework that constructs formal knowledge representations and uses external solvers, such as SAT or SMT solvers, to compute answers. Their system achieved strong results on numerical tax reasoning using a relaxed metric, but the evaluation was limited mainly to the numerical task and did not report full entailment results.

#### Kordjamshidi et al. (2026): Contamination-aware Evaluation and Neuro-Symbolic Robustness in Tax Law

This research studies whether LLMs truly reason over tax law or mainly perform better as translators into formal logic. It compares direct LLM question answering with neuro-symbolic systems where LLMs translate cases into Prolog and a solver performs reasoning. The paper also tests data contamination and introduces SARA+, a perturbed benchmark with changed rules, changed cases, and paraphrases. Its main finding is that direct LLM performance can be inflated by contamination and becomes unstable under rule/case changes, while Prolog-based systems are more robust. The paper supports the view that LLMs are useful, but high-stakes legal reasoning should be solver-backed and verifiable.

This paper is especially important for the question of legal-causal reasoning because it shows that high LLM performance on legal benchmarks may not be enough; models must also be tested under contamination controls, changed rules, changed facts, and solver-checkable reasoning.

### 6. Logic Translation and Solver-Augmented LLMs

#### Yang et al. (2024): Natural Language to First-Order Logic

Yang et al. studied how LLMs can translate natural language into first-order logic. Their work showed that LLMs can help convert language into formal logical representations, but the translation process remains difficult and requires careful validation.

#### Putra et al. (2026): NL2Logic

Putra et al. studied AST-guided translation from natural language into first-order logic. Their finding was that structured guidance can help LLMs produce better formal logic, but robust translation from natural language into formal representations remains challenging.

#### Pan et al. (2023): Logic-LM

Pan et al. introduced Logic-LM, a framework that empowers LLMs with symbolic solvers for logical reasoning. They found that LLMs can be improved by delegating formal reasoning to symbolic solvers. The LLM can help translate or prepare the problem, while the solver performs faithful logical inference.

#### Jiang et al. (2024): LeanReasoner

Jiang et al. studied how theorem provers such as Lean can support complex logical reasoning for LLMs. Their finding is that formal theorem provers can improve reasoning reliability because they verify whether logical steps are valid.

#### Feng et al. (2026): VeriCoT

Feng et al. introduced a neuro-symbolic method for validating chain-of-thought reasoning using logical consistency checks. They showed that logical consistency checks can help verify whether generated reasoning is valid.

### 7. Contamination Detection

#### Golchin and Surdeanu (2025): Data Contamination Quiz

Golchin and Surdeanu developed a method to detect and estimate data contamination in LLMs using quiz-style tests. Their method can detect whether models may have memorized benchmark examples by asking them to identify original examples among perturbed versions.

### 8. Broader Legal Reasoning Frameworks

#### Nguyen et al. (2025): LLMs for Legal Reasoning

Nguyen et al. provided a broader framework and future perspective on LLMs for legal reasoning. Legal reasoning includes multiple reasoning types, such as case-based reasoning, abductive reasoning, and deductive reasoning. LLMs can help with some of these tasks but are not fully reliable across all forms of legal reasoning.

#### Mochales and Moens (2011): Argumentation Mining

Mochales and Moens studied how to extract legal arguments and argumentative structures from text. They showed that legal reasoning often involves arguments, supporting claims, objections, and evidence. Computational systems can identify some of this structure, but legal argumentation remains complex.

#### Collenette et al. (2023): Explainable AI for Legal Case Reasoning

Collenette et al. studied explainable AI tools for legal reasoning about cases, including work related to the European Court of Human Rights. They argued that Legal AI systems need explanations, not just answers. Users need to know why a model reached a legal conclusion.

#### Zou et al. (2024): Tax Law Entailment as Analogical Reasoning

Zou et al. reframed tax-law entailment as an analogical reasoning problem. Instead of directly applying statutes, the model compares a new case to previous cases. The paper notes that, analogical reasoning can be useful but this approach is less competitive than more recent LLM-based methods for SARA-style statutory reasoning.

#### Savelka (2023): Semantic Annotation of Legal Texts

Savelka evaluated GPT-style models for zero-shot semantic annotation of legal texts. The main finding is that LLMs can annotate legal text and extract useful structure, showing that they can help convert legal language into more structured forms.

## Conclusion

Major AI models have become powerful legal and reasoning assistants, but the evidence does not support treating them as fully reliable legal, causal, or logical reasoners. Across the reviewed work, the same pattern appears: LLMs can perform well on many legal-language tasks, such as summarization, drafting, classification, issue spotting, and answering structured benchmark questions, but their reliability drops when tasks require grounded authority, exact rule application, numerical reasoning, consistency under prompt variation, or reasoning over changed facts and rules.

Legal reasoning is therefore partially mature. Models are already useful as assistants in legal research, compliance review, document analysis, and first-pass reasoning. However, hallucination studies and legal benchmark evaluations show that they can still misstate law, invent citations, confuse legal rules, and produce correct-looking answers with weak or invalid reasoning. This means benchmark success should be interpreted as evidence of useful capability, not proof of dependable legal reasoning.

Causal reasoning is less mature than general legal reasoning. Major models can discuss cause and effect fluently and often recognize common associations, but they remain weaker at intervention, counterfactual reasoning, confounding, and formal causal inference. This matters because legal reasoning often depends on causality: whether an action caused harm, whether discrimination caused an adverse decision, whether a breach caused damages, or whether a substitute product changes a patent damages outcome.

Legal-causal reasoning is the most important open problem. A model must not only produce the correct legal conclusion; it must reach that conclusion by tracking the correct causal variables and applying the correct legal rule. The Qwen legal-causal experiment illustrates this distinction: the smaller Qwen2.5-0.5B model failed by collapsing to an “always DENIED” shortcut, while Qwen2.5-1.5B passed the clean paired benchmark. This suggests that stronger models can improve on controlled legal-causal tasks, but passing small clean examples does not prove robust causal understanding.

The tax-law neuro-symbolic paper strengthens this conclusion. It shows that direct LLM performance on legal benchmarks can be inflated by data contamination and can become unstable when statutes or case facts are changed. In contrast, Prolog-based neuro-symbolic systems are more robust, especially for numerical tax reasoning, because they separate language understanding from formal rule execution. In this setup, the LLM is most useful as a translator from natural language into structured facts, while the symbolic solver performs the actual reasoning.

The central lesson is that the next stage of progress should not be measured only by whether models sound more intelligent or achieve high benchmark accuracy. It should be measured by whether their answers are grounded, consistent, causally valid, legally faithful, contamination-aware, and verifiable under changed rules, changed facts, paraphrases, and counterfactual scenarios.

For that reason, the most promising path is not simply larger models. Larger and reasoning-tuned models help, but high-stakes legal AI needs hybrid systems that combine LLMs with retrieval grounding, structured legal representations, causal graphs, rule-based solvers, formal verification, contamination-aware evaluation, and expert oversight.

In short, major AI models are useful, impressive, and improving, but legal and causal reasoning are not solved. We are somewhere in between: legal AI is already valuable as an assistant, but robust legal-causal reasoning remains an open research frontier. The hardest work ahead is building systems that do not merely produce plausible legal answers, but produce answers that are grounded, causally correct, legally valid, and verifiable.


## Bibliography

- Kordjamshidi, P., Aslan, S., Seshadri, M., Barrett, L., & Santus, E. (2026). *Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness on Tax Law*. Proceedings of the First Workshop on Structured Understanding, Retrieval, and Generation in the LLM Era (SURGeLLM 2026), 344–360.

- Holzenberger, N., Blair-Stanek, A., & Van Durme, B. (2020). *A Dataset for Statutory Reasoning in Tax Law Entailment and Question Answering*. Natural Legal Language Processing Workshop. https://arxiv.org/abs/2005.05257

- Guha, N., Nyarko, J., Ho, D. E., Ré, C., Chilton, A., Narayana, A., Chohlas-Wood, A., Peters, B., Waldon, B., Rockmore, D. N., Zambrano, D., Talisman, D., Hoque, E., Surani, F., Fagan, F., Sarfaty, G., Dickinson, G. M., Porat, H., Hegland, J., Wu, J., et al. (2023). *LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models*. Advances in Neural Information Processing Systems. https://arxiv.org/abs/2308.11462

- Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). *LEGAL-BERT: The Muppets Straight Out of Law School*. Findings of the Association for Computational Linguistics: EMNLP 2020, 2898–2904. https://aclanthology.org/2020.findings-emnlp.261/

- Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D., & Aletras, N. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 4310–4330. https://aclanthology.org/2022.acl-long.297/

- Nay, J. J., Karamardian, D., Lawsky, S. B., Tao, W., Bhat, M., Jain, R., Lee, A. T., Choi, J. H., & Kasai, J. (2024). *Large Language Models as Tax Attorneys: A Case Study in Legal Capabilities Emergence*. Philosophical Transactions of the Royal Society A, 382(2270), 20230159. https://arxiv.org/abs/2306.07075

- Hu, Y., Yu, Y., Gan, L., Wei, B., Kuang, K., & Wu, F. (2025). *Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond*. Findings of the Association for Computational Linguistics: EMNLP 2025, 13759–13781. https://aclanthology.org/2025.findings-emnlp.742/

- Dahl, M., Magesh, V., Suzgun, M., & Ho, D. E. (2024). *Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models*. Journal of Legal Analysis, 16(1), 64–93. https://arxiv.org/abs/2401.01301

- Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2025). *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*. Journal of Empirical Legal Studies, 22. https://arxiv.org/abs/2405.20362

- Buchanan, B. G., & Headrick, T. E. (1970). *Some Speculation About Artificial Intelligence and Legal Reasoning*. Stanford Law Review, 23(1), 40–62. https://digitalcommons.law.buffalo.edu/journal_articles/867/

- Sergot, M. J., Sadri, F., Kowalski, R. A., Kriwaczek, F., Hammond, P., & Cory, H. T. (1986). *The British Nationality Act as a Logic Program*. Communications of the ACM, 29(5), 370–386. https://doi.org/10.1145/5689.5920

- Schild, U. J. (1990). *Open-Textured Law, Expert Systems and Logic Programming*. PhD thesis, University of London. https://dblp.org/pid/94/4980.html

- Sartor, G. (2005). *Legal Reasoning: A Cognitive Approach to the Law*. Springer. https://link.springer.com/book/10.1007/1-4020-3505-5

- Mochales, R., & Moens, M.-F. (2011). *Argumentation Mining*. Artificial Intelligence and Law, 19(1), 1–22. https://doi.org/10.1007/s10506-010-9104-x

- Collenette, J., Atkinson, K., & Bench-Capon, T. (2023). *Explainable AI Tools for Legal Reasoning About Cases: A Study on the European Court of Human Rights*. Artificial Intelligence, 317, 103861. https://doi.org/10.1016/j.artint.2023.103861

- Zou, X., Zhang, M., Weir, N., Van Durme, B., & Holzenberger, N. (2024). *Reframing Tax Law Entailment as Analogical Reasoning*. arXiv preprint. https://arxiv.org/abs/2401.06715

- Yang, Y., Xiong, S., Payani, A., Shareghi, E., & Fekri, F. (2024). *Harnessing the Power of Large Language Models for Natural Language to First-Order Logic Translation*. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 6942–6959. https://aclanthology.org/2024.acl-long.375/

- Putra, R. R., Basuki, R. S. P., Cheng, Y., & Gao, P. (2026). *NL2Logic: AST-Guided Translation of Natural Language into First-Order Logic with Large Language Models*. Findings of the European Chapter of the Association for Computational Linguistics. https://arxiv.org/abs/2602.13237

- Savelka, J. (2023). *Unlocking Practical Applications in Legal Domain: Evaluation of GPT for Zero-Shot Semantic Annotation of Legal Texts*. Proceedings of the International Conference on Artificial Intelligence and Law. https://arxiv.org/abs/2305.04417

- Pan, L., Albalak, A., Wang, X., & Wang, W. Y. (2023). *Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning*. Findings of the Association for Computational Linguistics: EMNLP 2023, 3806–3824. 	

- Jiang, D., Fonseca, M., & Cohen, S. B. (2024). *LeanReasoner: Boosting Complex Logical Reasoning with Lean*. Proceedings of NAACL-HLT 2024, 7497–7510. https://aclanthology.org/2024.naacl-long.416/

- Jurayj, W., Holzenberger, N., & Van Durme, B. (2025). *Language Models and Logic Programs for Trustworthy Tax Reasoning*. AAAI Conference on Artificial Intelligence. https://ojs.aaai.org/index.php/AAAI/article/view/41212

- Lorenzo, G., Pietromatera, A., & Holzenberger, N. (2025). *Translating Tax Law to Code with LLMs: A Benchmark and Evaluation Framework*. Natural Legal Language Processing Workshop 2025, 31–47. https://aclanthology.org/2025.nllp-1.4/

- Sadowski, A., & Chudziak, J. A. (2025). *On Verifiable Legal Reasoning: A Multi-Agent Framework with Formalized Knowledge Representations*. arXiv preprint. https://arxiv.org/abs/2509.00710

- Feng, Y., Weir, N., Bostrom, K., Bayless, S., Cassel, D., Chaudhary, S., Kiesl-Reiter, B., & Rangwala, H. (2026). *VeriCoT: Neuro-Symbolic Chain-of-Thought Validation via Logical Consistency Checks*. International Conference on Learning Representations. https://arxiv.org/abs/2511.04662

- Golchin, S., & Surdeanu, M. (2025). *Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models*. Transactions of the Association for Computational Linguistics, 13, 809–830. https://aclanthology.org/2025.tacl-1.37/

- Nguyen, H. T., Fungwacharakorn, W., Zin, M. M., Goebel, R., Toni, F., Stathis, K., & Satoh, K. (2025). *LLMs for Legal Reasoning: A Unified Framework and Future Perspectives*. Computer Law & Security Review, 58, 106165. https://www.sciencedirect.com/science/article/pii/S2212473X25000380

- Shi, Y., Liu, H., Hu, Y., Song, G., Xu, X., Ma, Y., Tang, T., Zhang, L., Chen, Q., Feng, D., Lv, W., Wu, W., Yang, K., Yang, S., Wang, W., Shi, R., Qiu, Y., Qi, Y., Zhang, J., Sui, X., et al. (2026). *PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice*. arXiv preprint. https://arxiv.org/abs/2601.16669
