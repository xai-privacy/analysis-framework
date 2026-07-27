# Few-Shot Prompting and Constrained Decoding: Research Review

## 1. Introduction

Large language models (LLMs) can be used for a new task in two main ways: by giving them examples in the prompt (few-shot prompting), or by forcing their output to follow a fixed structure during generation (constrained decoding). Few-shot prompting became popular after Brown et al. (2020) showed that a large model could handle a new task after seeing only a few prompt examples, without retraining. Constrained decoding came from earlier work on controlling machine-translation output (Post & Vilar, 2018) and later developed into general tools that force an LLM's output to match a JSON schema or grammar (Willard & Louf, 2023). These two techniques solve two different problems, so combining them makes sense:

- Few-shot prompting teaches the model what the task means — the semantics (Brown et al., 2020).
- Constrained decoding guarantees the shape of the answer — the syntax (Scholak et al., 2021).

This report reviews several publications on constrained decoding and few-shot prompting. It gives a simple overview of both methods, explains the main research papers behind them (grouped by theme), and discusses current best practices — including the important question, "How many examples, or shots, should I use?"


## 2. Definitions

### 2.1 Few-Shot Prompting

Few-shot prompting means putting a small number of example input-output pairs directly in the prompt, before asking it to handle a new example. The model is not retrained — no weights change. It uses the examples as context to understand the task pattern. This ability is called in-context learning (ICL), clearly demonstrated at scale by Brown et al. (2020) using GPT-3. Later work found that this ability becomes much clearer when a model is large enough — smaller models may not benefit from examples in the same way (Touvron et al., 2023).

Example:


Classify the sentiment of the sentence.

Sentence: "This movie was fantastic!"
Sentiment: positive

Sentence: "I would not recommend this restaurant."
Sentiment: negative

Sentence: "The service was okay, nothing special."
Sentiment:


The model has seen two examples ("2-shot") and must now complete the third one. If no examples are given, this is called zero-shot; one example is one-shot; many examples (sometimes hundreds or thousands, made possible by modern long context windows) is called many-shot ICL.

### 2.2 Constrained Decoding

Constrained decoding is a method for controlling what the model may generate at each step, so the final output must follow a chosen rule — a JSON schema, a formal grammar, a regular expression, or a fixed set of allowed words. Instead of only hoping for valid output, the decoder blocks every token that would break the format. This idea started with early work that forced specific words into machine-translation output (Post & Vilar, 2018) and later led to systems that keep SQL queries valid during generation (Scholak et al., 2021). Modern general-purpose tools apply the same idea to JSON schemas and custom grammars, usually by removing illegal token choices at each step (Willard & Louf, 2023).

The key difference from prompting: prompting *asks nicely*; constrained decoding *makes it structurally impossible to answer wrong*.

### 2.3 Why Combine Them?

Prompting alone can still produce invalid output, even when the few-shot examples are well formatted, because the model is still predicting likely next tokens and may ignore the requested format (vLLM Blog, 2025). Constrained decoding solves the syntax problem completely but does not guarantee that the content is correct or meaningful. Few-shot prompting solves the content/semantics problem but cannot guarantee format on its own. Together, they address both failure types: the examples teach the model the meaning of the task, while the decoder guarantees the output always parses correctly. This exact combination — few-shot demonstrations plus schema- or grammar-based decoding — appears in several recent papers, most directly in Wang et al.'s Grammar Prompting (Wang et al., 2023), and is discussed further in Section 3.6.



## 3. Research Works

This section explains the main papers behind the later recommendations, grouped by theme.

### 3.1 Foundational Work

**Brown et al. (2020) — "Language Models are Few-Shot Learners" (GPT-3 paper), NeurIPS 2020.**
This is the paper that made modern few-shot prompting widely known. The authors showed that a large enough language model (GPT-3, 175B parameters) could handle many tasks — translation, question answering, arithmetic, and more — by seeing only a small number of prompt examples, without any fine-tuning. They found that few-shot performance improved much faster than zero-shot performance as the model got bigger, suggesting that learning from context becomes stronger as models grow. In their standard evaluation setup they typically compared performance across K = 0, 1, and a handful of examples per task, with K = 100 used for some smaller "toy" tasks like word-scrambling. This paper is the main starting point for most later work in this report.

**Touvron et al. (2023) — "LLaMA: Open and Efficient Foundation Language Models".**
Often cited as evidence that few-shot ("in-context learning") behavior is an emergent property: it becomes more useful after a model reaches a certain size, echoing the scaling-law observations of Kaplan et al. (2020).

### 3.2 How Many Examples? (Few-Shot vs. Many-Shot)

**Agarwal et al. (2024) — "Many-Shot In-Context Learning," NeurIPS 2024 (Spotlight).**
This paper directly studies the question "how many shots should I use?" The authors show that as context windows became larger from a few thousand tokens (GPT-3's 2048-token limit) to hundreds of thousands of tokens, it became possible to include *hundreds or thousands* of examples into a prompt — an approach they call **many-shot ICL**. Across several tasks, they found many-shot ICL often performs better than traditional few-shot ICL, especially on difficult tasks that are not ordinary language tasks (e.g., using the entire training set for a machine translation task produced large accuracy gains over a 1-shot prompt). Notably, for reasoning tasks like MATH, they still used a common shot count from earlier research (e.g., 4-shot) as a fair few-shot baseline, showing that even researchers studying many-shot regimes still use small few-shot settings as the normal comparison baseline.

**Meta-Tool paper (2026) — "Efficient Few-Shot Tool Adaptation for Small Language Models."**
This paper tested a controlled range from 0 to 5 shots and found the largest accuracy gain happens going from 0 to 1 example (+8 percentage points on average across benchmarks), with performance continuing to rise up to about 5 shots but with smaller gains after 3. Some tasks showed little change at any shot count, showing that the "right" number of shots is very task-dependent, not one fixed rule.

**DetPO (2026) — few-shot object detection with multi-modal LLMs.**
This work found a similar pattern in a very different domain (vision): performance improved noticeably from 3-shot to 5-shot, but gains from 5-shot to 10-shot were only marginal — which also supports "diminishing returns after roughly 5 examples."

**Med-HALT (2023) —  Medical Domain Hallucination Test for Large Language Models.**
On GPT-3.5, zero-shot accuracy was very low (about 7%) in a medical QA setting, and adding a few examples produced large gains, but the improvement flattened out once shot count went past 3.

**Kim et al. — "Can Language Models Explain Their Own Classification Behavior?"**
An important exception: for some harder rule-learning tasks, the authors found they needed as many as 16–32 examples before the model could reliably understand the hidden rule, and even then some rules were never fully learned. This is a useful reminder that "3–5 shots is enough" is a rule of thumb for easier tasks, not a hard law — truly complex or unclear tasks may need more.

### 3.3 Which Examples to Pick (Demonstration Selection)

**Liu et al. (2022) — "What Makes Good In-Context Examples for GPT-3?" (introduces KATE).**
This paper focuses on which examples to use, not only how many. The authors proposed **KATE** (kNN-Augmented in-conText Example selection): instead of picking examples randomly, retrieve examples from a pool that are *semantically closest* to the current test question, using sentence embeddings and nearest-neighbor search. They found this retrieval-based selection consistently performed better than random selection, with especially large gains on table-to-text generation and open-domain question answering. This idea — pick examples similar to the current query, not a fixed random set — has become a common approach for demonstration selection in later work.

**Rubin et al. (2022) — EPR (Efficient Prompt Retrieval).**
This improves KATE-style retrieval by training a dedicated retriever model (rather than using an off-the-shelf sentence encoder) to find examples that are useful to the final language model, using a two-stage retrieve-then-rerank process.

**Cheng et al. (2023) — UPRISE.**
This extends retrieval-based selection to work across many different tasks and domains at once, training one universal retriever instead of a separate one per task.

**Luo et al. (2023) — Dr.ICL (Demonstration-Retrieved In-Context Learning).**
Surveys and compares these retrieval approaches, and confirms that semantically similar demonstrations usually beat random ones across benchmarks.

### 3.4 Order and Formatting Sensitivity

**Lu et al. (2022) — "Fantastically Ordered Prompts and Where to Find Them."**
Shows that changing only the *order* in which the same set of few-shot examples appears in the prompt can greatly change performance — sometimes making the difference between state-of-the-art results and near-random performance on the exact same examples. They also show this order effect is dataset-dependent: an order that works for one task may not work for another.

**Zhao et al. (2021) — "Calibrate Before Use: Improving Few-Shot Performance of Language Models."**
Shows that models may have systematic biases toward certain answers even when the input changes — for example, a tendency to prefer whichever label appeared most recently or most often among the demonstrations. They propose a calibration procedure (using a "content-neutral" placeholder input to measure and correct for this bias) to fix it.

**Min et al. (2022) — "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"**
This widely cited paper reports a surprising finding: replacing the correct labels in the few-shot examples with *random, incorrect* labels often causes only a small performance drop on many tasks. What matters much more is the overall **format** — the fact that inputs and labels appear in a consistent structure — rather than whether the specific label values shown are actually correct. In simple terms: demonstrations often teach the model "what shape an answer should have," more than they teach it new facts.

### 3.5 Constrained Decoding Foundations

**Hokamp & Liu (2017); Post & Vilar (2018) — early lexically-constrained decoding.**
Early work in machine translation that forced specific words or phrases to appear in the output during decoding — an early foundation of modern constrained decoding.

**Scholak et al. (2021) — PICARD.**
A widely cited system that integrates incremental parsing directly into the decoding loop for SQL generation, rejecting any token that would make the growing SQL query invalid — making the syntax valid by design.

**Willard & Louf (2023) — Outlines; and related tools (XGrammar, Guidance, vendor "structured outputs" features).**
Modern general-purpose libraries and frameworks that extend the same idea to JSON Schemas, type systems, and arbitrary domain-specific languages, using optimized finite-state machines to filter which tokens are legal at each decoding step, making constrained decoding practical for real systems.

**JSONSchemaBench (cited in several 2025–2026 papers).**
A benchmark designed to test how well constrained-decoding frameworks handle real-world JSON schemas. An important finding in several recent papers: even *with* constrained decoding turned on, some frameworks and schema types still struggle — constrained decoding removes invalid-syntax errors but still does not guarantee semantically correct answers.

### 3.6 Combining Few-Shot Prompting with Constrained Decoding


**Wang et al. (2023) — "Grammar Prompting for Domain-Specific Language Generation with Large Language Models."**
 The authors combine few-shot demonstrations with a formal grammar (written in Backus–Naur Form): each in-context example (input, output) is paired with a minimal grammar snippet that describes the exact syntax needed to produce that particular output. Given a new input, the model first predicts the small grammar it will need, then generates the final answer *while the grammar constrains decoding*, guaranteeing syntactic validity. The main goal of the paper is to use few-shot learning to teach meaning while using a grammar to guarantee structure.

**Recent structured-generation survey/appendix work (2026) — "The Hidden Cost of Structured Generation in LLMs" (Draft-Conditioned Constrained Decoding, DCCD).**
This paper separates and compares different ways of getting structured output: prompt-only format control (including few-shot demonstrations), plain constrained decoding, and a hybrid approach. It observes that prompt-based methods (schema instructions, few-shot demonstrations, reminders) can improve how well the model follows the format, but do not *guarantee* correctness and can still produce invalid outputs — this is precisely the gap that constrained decoding is meant to close. The paper runs a specific "constrained few-shot" baseline as a baseline, treating few-shot prompting plus constraints as a normal baseline in this space.

**Multi-view Prompting for Aspect-Based Sentiment Analysis (2026).**
This is an applied example: the authors combine schema-constrained decoding (via a context-free grammar) with few-shot-style prompting techniques, and report that this combination reduces much of the performance gap between plain few-shot prompting and models that were fully fine-tuned for the task — while being much cheaper to run than fine-tuning.

**vLLM engineering blog (2025) — "Structured Decoding in vLLM: A Gentle Introduction."**
A practice-oriented (non-academic but widely read) explanation of exactly why prompting alone is not enough: because LLMs are probabilistic generators, there is no guarantee that a few-shot-prompted request for JSON actually returns valid JSON every time — hence the need for constrained/structured decoding as a second safety layer added to prompting.


## 4. Best Practices and Discussion

### 4.1 How many shots should we use?

There is no single correct number, but many studies show a similar pattern:

0 -> 1           :  Usually the largest accuracy gain 
1 -> 3-5         :  Usually continued improvement 
5 → 10           :  Often only small improvements ("diminishing returns")
10 → 100s/1000s  :  Most useful with a long context window and a large, good example pool ("many-shot" regime)


Moving from zero examples to one often gives the largest single gain. A recent study on tool-use tasks found the jump from 0 to 1 example gave an average gain of about 8 percentage points, with gains slowing down noticeably after 3 examples (Meta-Tool paper, 2026). A study on object detection found a similar shape: real improvement from 3 to 5 examples, but almost no extra benefit from 5 to 10 (DetPO paper, 2026). A medical question-answering study reported a similar pattern — strong early gains, then a flattening out after about 3 examples (Pal et al., 2023).

Some newer work shows that when a model can process very long prompts, using hundreds or even thousands of examples ("many-shot" prompting) can beat the usual few-shot setup, especially on harder tasks (Agarwal et al., 2024). But even in that same paper, when comparing against a "normal" few-shot baseline, the authors still used a small, typical shot count like 4 — showing that a small number of examples is still the normal starting point even in many-shot research (Agarwal et al., 2024).

**Simple starting rule: start with 3–5 examples.** This matches common benchmark practice (Agarwal et al., 2024) and the pattern seen across several independent studies (Meta-Tool paper, 2026; DetPO paper, 2026; Pal et al., 2023). Add more only when validation results continue to improve, and only move into the hundreds/thousands range if the task is truly hard and the system can afford the extra cost per request.

**One exception to keep in mind:** harder tasks may need more examples before the model "gets it," and the right number is not the same for every task or model. Chen et al. (2023) study exactly this question — how many demonstrations a task needs — and find that the answer varies noticeably depending on the task and the model being used, rather than being a single fixed number. So 3–5 is a good starting point, not a strict limit — we should always check the results before locking in a final number.

### 4.2 How should we choose which examples to show?

Random examples can work, but it is usually not the best choice. Liu et al. (2022) introduced KATE, a method that picks examples that are semantically similar to the specific question being asked, instead of using one fixed random set. This consistently performed better than random selection, with especially large gains on tasks like table-to-text writing and open-domain question answering (Liu et al., 2022). Later papers improved on this idea further — for example, by training a dedicated retriever model instead of using an off-the-shelf one (Rubin et al., 2022), or by building one retriever that works well across many different tasks at once (Cheng et al., 2023). If we can measure how similar our example pool is to a new input, we can use that to choose our examples — this can provide a useful gain for limited extra work.

### 4.3 Does the order of examples matter?

Yes, sometimes. Lu et al. (2022) showed that simply changing the order of the same set of examples — without changing the examples themselves — can change performance greatly, sometimes from near state-of-the-art down to close to random guessing. However, they also found that the best order is different for different tasks, so there is no universal rule like "always put the hardest example last" (Lu et al., 2022). This connects to a related finding by Zhao et al. (2021): models are biased toward whichever example sits near the end of the prompt or appears most often, which is part of why order can matter so much.

### 4.4 Prompting alone is not enough for guaranteed structure

Even a carefully written few-shot prompt cannot guarantee valid output — it can only increase the chance of valid output, because the model is still just predicting the next most probable token each time (vLLM Blog, 2025). A recent paper studying structured output generation makes this point directly: prompt-based methods, including few-shot examples, can improve how well a model follows a format, but they do not guarantee correctness and can still produce broken output (DCCD paper, 2026). This is exactly the gap that constrained decoding is built to close — by blocking invalid tokens from being generated in the first place, instead of just asking the model nicely (Cooper, 2024; Scholak et al., 2021).

### 4.5 Practical checklist — "what do we need to do concretely?"

1. Create a labeled pool of example input/output pairs — ideally several times larger than the maximum number of shots we plan to use, so that there is room to select from it for each query.
2. Pick a small starting shot count (3–5) and test it on a held-out validation set before choosing the final setting.
3. Retrieve demonstrations by similarity to the current input, rather than using one fixed static set for every request.
4. Fix the example order after choosing a good one; don't re-randomize every call.
5. Define the desired output format as an actual schema or grammar, and enforce it at the decoding layer (tool-forced/JSON-schema output, grammar-constrained decoding, etc.) — do not rely on prompt instructions alone to guarantee valid structure.
6. Track two separate numbers when evaluating: (a) task accuracy, and (b) the rate of malformed/unparseable outputs. These are different failure modes and constrained decoding is specifically meant to drive the second one to zero.
7. Re-check all of the above on the actual task and data — none of these numbers transfer perfectly from one task to another.


## 5. Conclusion

Few-shot prompting and constrained decoding solve two different problems, and so they work well together. Few-shot prompting teaches a model what a task *means* by showing it examples (Brown et al., 2020), while constrained decoding guarantees the *shape* of its output by blocking invalid tokens as the answer is generated (Scholak et al., 2021; Cooper, 2024). The research supports combining the two instead of using only one, and papers doing exactly this already exist — most directly, Wang et al.'s Grammar Prompting, which pairs few-shot examples with a formal grammar to guarantee valid structured output from only a handful of demonstrations (Wang et al., 2023).

For the specific question of "how many shots," several independent studies show a similar overall pattern: the jump from zero to a few examples matters the most, gains slow down noticeably after about 3–5 examples for most tasks (Meta-Tool paper, 2026; DetPO paper, 2026), and going much further into the hundreds or thousands ("many-shot" prompting) is a separate, newer technique that is mainly useful for harder tasks with a large context window and a large, reliable example pool (Agarwal et al., 2024). In addition to shot count, *which* examples are chosen matters too — similarity-based retrieval beats random selection (Liu et al., 2022) — and *how strictly the output format is enforced* can matter as much as the number of examples, since prompting alone cannot fully guarantee valid output (DCCD paper, 2026).

**In summary:** start with 3–5 well-chosen, similar examples, enforce output format at the decoding level rather than only asking for it in the prompt, and check both the shot count and selection method on the actual data before finalizing them.

---

## 6. References

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *NeurIPS 33*, 1877–1901. https://arxiv.org/abs/2005.14165
- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. https://arxiv.org/abs/2302.13971
- Agarwal, R., Singh, A., Zhang, L. M., Bohnet, B., Rosias, L., Chan, S., ... & Larochelle, H. (2024). Many-Shot In-Context Learning. *NeurIPS 2024* (Spotlight). https://arxiv.org/abs/2404.11018
- Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., & Chen, W. (2022). What Makes Good In-Context Examples for GPT-3? *DeeLIO 2022 (ACL Workshop)*. https://arxiv.org/abs/2101.06804
- Rubin, O., Herzig, J., & Berant, J. (2022). Learning to Retrieve Prompts for In-Context Learning. *NAACL 2022*. https://arxiv.org/abs/2112.08633
- Cheng, D., Huang, S., Bi, J., Zhan, Y., Liu, J., Wang, Y., Sun, H., Wei, F., Deng, D., & Zhang, Q. (2023). UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation. https://arxiv.org/abs/2303.08518
- Luo, M., Xu, X., Dai, Z., Pasupat, P., Kazemi, M., Baral, C., Imbrasaite, V., & Zhao, V. (2023). Dr.ICL: Demonstration-Retrieved In-Context Learning. https://arxiv.org/abs/2305.14128
- Lu, Y., Bartolo, M., Moore, A., Riedel, S., & Stenetorp, P. (2022). Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. *ACL 2022*. https://arxiv.org/abs/2104.08786
- Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *ICML 2021*. https://arxiv.org/abs/2102.09690
- Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., & Zettlemoyer, L. (2022). Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? *EMNLP 2022*. https://arxiv.org/abs/2202.12837
- Pal, A., Umapathi, L. K., & Sankarasubbu, M. (2023). Med-HALT: Medical Domain Hallucination Test for Large Language Models. https://arxiv.org/abs/2307.15343
- Chen, J., Chen, L., Zhu, C., & Zhou, T. (2023). How Many Demonstrations Do You Need for In-Context Learning? *Findings of EMNLP 2023*. https://arxiv.org/abs/2303.08119
- Hokamp, C., & Liu, Q. (2017). Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search. *ACL 2017*. https://aclanthology.org/P17-1141/
- Post, M., & Vilar, D. (2018). Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation. *NAACL 2018*. https://arxiv.org/abs/1804.06609
- Scholak, T., Schucher, N., & Bahdanau, D. (2021). PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models. *EMNLP 2021*. https://arxiv.org/abs/2109.05093
- Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large Language Models (Outlines). https://arxiv.org/abs/2307.09702
- Wang, B., Wang, Z., Wang, X., Cao, Y., Saurous, R. A., & Kim, Y. (2023). Grammar Prompting for Domain-Specific Language Generation with Large Language Models. https://arxiv.org/abs/2305.19234
- vLLM Blog (2025). Structured Decoding in vLLM: A Gentle Introduction. https://blog.vllm.ai/2025/01/14/struct-decode-intro.html
- Cooper, A. (2024). A Guide to Structured Outputs Using Constrained Decoding. https://www.aidancooper.co.uk/constrained-decoding/

