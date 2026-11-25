# Model Selection Report: Reasoning-Error Taxonomy on LEET_Arg, MBE, and LegalBench

Prepared 2026-10-31 · Revised 2026-11-22 (final model set after review feedback)

## 1. Purpose

This report justifies the model set for a taxonomy of reasoning mistakes that open-weight models (small general-purpose, mid-size, domain-adapted legal, and a modern large reasoning model) and a closed frontier model make on two benchmarks, **LEET_Arg** and **MBE**, with **LegalBench** as a supplementary check. For each model it explains why it fits the available hardware and why it is informative for these benchmarks.

**Revision summary.** Following review feedback, the following changes were made to the original set:

- **Dropped:** Qwen3-0.6B (its failures were expected to be dominated by format and comprehension noise rather than legal-reasoning errors), Llama-3.2-3B-Instruct, and gpt-oss-20b/120b.
- **Added:** Qwen3-4B as the small baseline, SaulLM-54B-Instruct as a domain-adapted legal model, and DeepSeek-V4-Flash as a modern open-weight reasoner.
- **Kept:** Qwen3-8B (thinking on and off), Mistral-Small-24B-Instruct-2501, and Claude Sonnet 5.5.

**Selected models**

| #   | Model                                   | Organization             | Parameters                       | Role in design                               | Access             | URL                                                                |
| --- | --------------------------------------- | ------------------------ | -------------------------------- | -------------------------------------------- | ------------------ | ------------------------------------------------------------------ |
| 1   | Qwen3-4B                                | Alibaba                  | 4B dense                         | Small baseline, thinking on/off              | Open weights       | <https://huggingface.co/Qwen/Qwen3-4B>                             |
| 2   | Qwen3-8B                                | Alibaba                  | 8B dense                         | Mid-small, thinking on/off                   | Open weights       | <https://huggingface.co/Qwen/Qwen3-8B>                             |
| 3   | Mistral-Small-24B-Instruct-2501         | Mistral AI               | 24B dense                        | Mid-size dense generalist, no reasoning mode | Open weights       | <https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501> |
| 4   | SaulLM-54B-Instruct                     | Equall / CentraleSupélec | 54B nominal, Mixtral-derived MoE | Domain-adapted legal LLM                     | Open weights (MIT) | <https://huggingface.co/Equall/SaulLM-54B-Instruct>                |
| 5   | DeepSeek-V4-Flash                       | DeepSeek                 | 284B MoE, 13B active             | Modern open-weight reasoner                  | Open weights (MIT) | <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>             |
| 6   | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | Anthropic                | Not disclosed                    | Closed frontier ceiling                      | Closed, API only   | <https://docs.claude.com>                                          |

## 2. Hardware fit

### 2.1 Available flavors

| Flavor                 | GPU memory | Role in this study                                                     |
| ---------------------- | ---------- | ---------------------------------------------------------------------- |
| `g1.h100.vgpu.1g.12gb` | 12 GB      | Prompt and parser development, debugging, short-context Qwen3-4B tests |
| `g1.h100.pci.1`        | 80 GB      | Main evaluation of Qwen3-4B, Qwen3-8B, and Mistral-Small-24B           |
| `g1.h100.pci.4`        | 4 x 80 GB  | Tensor-parallel evaluation of SaulLM-54B and DeepSeek-V4-Flash         |

### 2.2 Memory estimates

Estimates below are weights only, from parameter count times bytes per parameter. KV cache and activations come on top, and thinking-mode generations with long outputs make the KV cache substantial. Confirm actual fit with a smoke test before scheduling full runs.

| Model                           | Precision              | Approx. weight memory             | Planned flavor                  |
| ------------------------------- | ---------------------- | --------------------------------- | ------------------------------- |
| Qwen3-4B                        | bf16                   | about 8 GB                        | `pci.1` (`vgpu` for dev only)   |
| Qwen3-8B                        | bf16                   | about 16 GB                       | `pci.1`                         |
| Mistral-Small-24B-Instruct-2501 | bf16                   | about 48 GB                       | `pci.1`                         |
| SaulLM-54B-Instruct             | bf16                   | about 93 GB or more               | `pci.4`, tensor parallel 2 or 4 |
| DeepSeek-V4-Flash               | Released mixed FP4/FP8 | roughly 150 GB or more (estimate) | `pci.4`, tensor parallel 4      |
| Claude Sonnet 5.5               | n/a                    | none                              | API                             |

## 3. Benchmark relevance

### 3.1 MBE-exam-questions

The Multistate Bar Exam is a four-option multiple-choice test of US legal reasoning. Each item is a short fact pattern plus a question that requires recalling the right rule and applying it to the facts, across subjects such as torts, contracts, evidence, constitutional law, criminal law and procedure, real property, and civil procedure.

### 3.2 LEET_Arg

Items contain a passage (often a dialogue between named speakers or a short argument), several labeled statements, a per-statement correct/incorrect judgment, and a question-level answer option determined by which statements are correct. Statement sets are either (a)-(c) or ①-⑤.

### 3.3 LegalBench

LegalBench consists of multiple benchmarks, of which we use the legal reasoning causality and privacy policy entailment benchmarks (Guha et al., 2023).

## 4. Why each model is relevant

### 4.1 Qwen3-4B (Alibaba)

- **URL:** <https://huggingface.co/Qwen/Qwen3-4B>
- **Hardware:** about 8 GB in bf16. It fits the 12 GB vGPU for development with short contexts, but run the main evaluation on `pci.1` because thinking-mode outputs need KV headroom.
- **Benchmarks:** one set of weights supports thinking (`enable_thinking=True`) and non-thinking (`enable_thinking=False`) modes, with the same tokenizer and training lineage as Qwen3-8B.
- **Taxonomy value:** serves as the small-scale baseline. At 4B the model is generally capable of following the answer format, so most of its failures should be genuine reasoning errors rather than the formatting and repetition-loop noise expected at 0.6B. This keeps a clean lower bound on the scale axis. Pairing it with Qwen3-8B gives a within-family 4B vs. 8B comparison.
- **Practical notes:** follow the Qwen3 model card sampling settings per mode and avoid greedy decoding in thinking mode. Cap `max_tokens` so any residual repetition loops terminate and are recorded as token-budget failures.

### 4.2 Qwen3-8B (Alibaba)

- **URL:** <https://huggingface.co/Qwen/Qwen3-8B>
- **Hardware:** about 16 GB in bf16 on `pci.1`.
- **Benchmarks:** run both thinking on and thinking off. Because the weights are identical, this is the cleanest reasoning comparison in the study (Qwen Team, 2025). Qwen3 also has broad multilingual coverage, relevant to LEET's Korean origin.
- **Taxonomy value:** isolates the effect of explicit reasoning traces from the effect of the model, and shows how those traces change the distribution of legal error categories. Together with Qwen3-4B it provides a within-family scale comparison.
- **Practical notes:** the model card recommends different sampling settings per mode, so the two conditions will not share identical decoding parameters. Report this.

### 4.3 Mistral-Small-24B-Instruct-2501 (Mistral AI)

- **URL:** <https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501>
- **Hardware:** about 48 GB in bf16 fits a single 80 GB GPU with room for KV, so no quantization is needed.
- **Benchmarks:** a non-reasoning instruct model in the 24B dense tier, from a European lab with a different training mix. It fills the gap between the 8B models and the large models.
- **Taxonomy value:** shows whether errors seen at 4-8B persist at 24B without any reasoning mode, which separates scale from reasoning.
- **Practical notes:** pin the exact checkpoint (`2501`). Later releases exist, and the "Mistral Small" name now also covers much larger MoE models, so the name alone is ambiguous. Record the license for the pinned release.

### 4.4 SaulLM-54B-Instruct (Equall and MICS, CentraleSupélec)

- **URLs:** <https://huggingface.co/Equall/SaulLM-54B-Instruct> (model card) and <https://arxiv.org/abs/2407.19584> (Colombo et al., 2024)
- **Hardware:** the model is a Mixtral-derived architecture. The Hugging Face card lists it as a 54B model, while the base checkpoint reports about 46.7B parameters in safetensors. Either way, bf16 weights are roughly 93 GB or more, which exceeds a single 80 GB GPU. Run on `pci.4` with tensor parallelism. Avoid quantizing to fit one GPU, because that would add a precision confound relative to the other models.
- **Benchmarks:** it is the most widely cited open legal LLM, produced by continued pretraining and legal instruction tuning on US and European legal data. It directly tests whether legal domain adaptation changes the error profile on bar-exam and LEET-style reasoning.
- **Taxonomy value:** tests whether legal knowledge acquired through continued pretraining reduces rule-recall errors, rule-application errors, or neither, compared with general-purpose models of similar or smaller size.
- **Limitations to report:**
  - **Vintage.** Its weights date from 2024 (initial upload May 7, 2024; paper July 2024), so it predates the other open models in the study. State this explicitly as a variable, not only a caveat.
  - **Contamination risk on LegalBench.** The Hugging Face card lists `Equall/legalbench_instruct` as a training dataset. LegalBench results for this model may therefore be contaminated and should be flagged or reported separately from LEET_Arg and MBE.
  - **Architecture.** It is an MoE derived from Mixtral, so do not place it on the dense-model scale curve. Treat it as a separate domain-adaptation condition.
  - **Missing control.** Without the underlying Mixtral instruct model in the set, the domain-adaptation effect is confounded with architecture and training recipe. An optional Mixtral control (see Section 6) would close this gap.
  - **Jurisdiction.** The model card notes a focus on US and European legal systems, so it may be weaker on Korean-law content in LEET_Arg.

### 4.5 DeepSeek-V4-Flash (DeepSeek)

- **URLs:** <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash> (model card), <https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/> (specifications), <https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash> (NVIDIA reference page)
- **Specifications:** 284B total parameters with 13B active per token, a 1M-token context window, MIT license, and hybrid attention (compressed sparse plus heavily compressed). The initial release was announced on April 24, 2026. A later update, DeepSeek-V4-Flash-0731, is listed at 304B total and 13B active, with a Hugging Face release date of July 30, 2026 (<https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash-0731>).
- **Hardware:** the released weights use mixed FP4 (MoE experts) and FP8 precision, which puts the weights at an estimated 150 GB or more, so a single 80 GB GPU is not an option. Plan for `pci.4` with tensor parallelism 4, and note that NVIDIA's reference run used much larger multi-GPU nodes. Before committing, confirm that the pinned vLLM version can serve this checkpoint on H100 (the model card recommends vLLM with sparse-attention support). If it cannot, fall back to the hosted API and log that deviation.
- **Benchmarks:** three reasoning modes (non-think, think-high, think-max) give a reasoning-effort axis on a modern RL-trained reasoner.
- **Taxonomy value:** fills the gap between the 24B/54B open models and Claude Sonnet 5.5. It separates the effect of modern RL-based reasoning from the effect of being a proprietary frontier model.
- **Practical notes:** pin the checkpoint. The April release and the 0731 update differ in size and behavior, so choose one, state which, and do not mix them across runs. At minimum run non-think and think-high. Think-max is documented as needing a very large output budget (DeepSeek recommends 384K+ context), so cap and log `max_tokens` and report budget-exhaustion failures separately.

### 4.6 Claude Sonnet 5.5 (Anthropic, closed)

- **URL:** <https://docs.claude.com> (API documentation); model string `claude-sonnet-5-5`
- **Hardware:** none; API only.
- **Benchmarks:** serves as the closed frontier ceiling, showing which errors are specific to open-weight models and which are shared by frontier models.
- **Reproducibility:** pin and log the effort/reasoning setting (for example medium vs. high), temperature, `max_tokens`, and the date of each run, since API-served models can change.
- **Limits:** no weights, no logits, and reasoning visibility depends on the API configuration. Check what trace information you can retrieve before planning annotation.
- **Annotation caution:** if any LLM is used to assist error labeling, avoid using Sonnet 5.5 as the sole labeler of its own outputs (Panickssery et al., 2024).

## 5. Axis coverage and confounds

| Axis                    | Coverage                                                                       | Clean comparison?                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Scale (dense)           | 4B, 8B, 24B                                                                    | Partly; Qwen3 4B vs. 8B is within family, 24B is another family                                    |
| Scale (within family)   | Qwen3-4B to Qwen3-8B                                                           | Yes                                                                                                |
| Reasoning               | Qwen3 thinking on/off (4B, 8B); DeepSeek-V4 non-think / think-high / think-max | Yes                                                                                                |
| Domain adaptation       | SaulLM-54B (legal) vs. general models                                          | Partly; confounded with MoE architecture, size, and 2024 vintage unless a Mixtral control is added |
| Family                  | Alibaba, Mistral, Equall, DeepSeek, Anthropic                                  | Confounded with scale                                                                              |
| Open vs. closed         | Claude Sonnet 5.5                                                              | Single data point                                                                                  |
| Total vs. active params | SaulLM and DeepSeek are MoE                                                    | Treat separately from the dense-model scale curve                                                  |

## 6. Suggested evaluation conditions

| Condition                             | Flavor                         | Precision        | Notes                                                                  |
| ------------------------------------- | ------------------------------ | ---------------- | ---------------------------------------------------------------------- |
| Qwen3-4B (thinking)                   | `pci.1`                        | bf16             | Model-card sampling, `max_tokens` cap, presence penalty if loops occur |
| Qwen3-4B (non-thinking)               | `pci.1`                        | bf16             | Model-card sampling                                                    |
| Qwen3-8B (thinking)                   | `pci.1`                        | bf16             | Model-card sampling for thinking mode                                  |
| Qwen3-8B (non-thinking)               | `pci.1`                        | bf16             | Model-card sampling for non-thinking mode                              |
| Mistral-Small-24B-Instruct-2501       | `pci.1`                        | bf16             | Pin checkpoint                                                         |
| SaulLM-54B-Instruct                   | `pci.4` (TP 2 or 4)            | bf16             | Flag LegalBench contamination; 2024 vintage                            |
| DeepSeek-V4-Flash (non-think)         | `pci.4` (TP 4)                 | Released FP4/FP8 | Pin checkpoint (April release or 0731); verify vLLM support first      |
| DeepSeek-V4-Flash (think-high)        | `pci.4` (TP 4)                 | Released FP4/FP8 | Large output budget; log truncations                                   |
| Claude Sonnet 5.5                     | API                            | n/a              | Pin effort level, fixed temperature, logged settings                   |
| Optional: Mixtral-8x7B-Instruct       | `pci.4` or `pci.1` (quantized) | bf16 or 8-bit    | Control for SaulLM's base architecture; note any precision difference  |
| Optional: DeepSeek-V4-Flash think-max | `pci.4` (TP 4)                 | Released FP4/FP8 | Only if budget allows; very long outputs                               |

## 7. Release dates, training cutoffs, and dataset provenance

### 7.1 Models

| Model                           | Release date                                                                 | Training-data / knowledge cutoff                  |
| ------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------- |
| Qwen3-4B                        | Apr 28, 2025 (Qwen3 family release; verify per-model date on the model page) | **Not recorded.**                                 |
| Qwen3-8B                        | Apr 28, 2025 (Qwen3 family release; verify per-model date on the model page) | **Not recorded.**                                 |
| Mistral-Small-24B-Instruct-2501 | Jan 30, 2025                                                                 | Oct 2023                                          |
| SaulLM-54B-Instruct             | Weights uploaded May 7, 2024; paper Jul 2024                                 | **Not recorded.** (Mixtral-derived; 2024 vintage) |
| DeepSeek-V4-Flash               | Apr 23-24, 2026 (0731 update: Jul 30, 2026)                                  | **Not recorded.**                                 |
| Claude Sonnet 5.5               | Sep 28, 2026                                                                 | Jun 2026                                          |

### 7.2 Datasets

| Dataset    | Source       | Publication date | Date range of underlying questions |
| ---------- | ------------ | ---------------- | ---------------------------------- |
| LEET_Arg   | GitHub       | 2026             | LEET exams 2021-2025               |
| MBE        | Hugging Face | April 2025       | **Not recorded.**                  |
| barexam_qa | Hugging Face | May 2025         | **Not recorded.**                  |

Because several models have cutoffs before or near the dataset dates, record for each benchmark whether its question date range falls after each model's cutoff.

## 8. References

- Bommarito & Katz (2022). GPT Takes the Bar Exam. arXiv.
- Katz et al. (2024). GPT-4 passes the bar exam. Philosophical Transactions of the Royal Society A.
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Nature.
- Chen et al. (2024). Do NOT think that much for 2+3=? On the overthinking of o1-like LLMs. arXiv 2412.21187.
- Qwen Team (2025). Qwen3 Technical Report. arXiv.
- Colombo et al. (2024). SaulLM-54B & SaulLM-141B: Scaling Up Domain Adaptation for the Legal Domain. arXiv 2407.19584. <https://arxiv.org/abs/2407.19584>
- Panickssery, Bowman & Feng (2024). LLM evaluators recognize and favor their own generations. NeurIPS.
- Guha et al. (2023). LegalBench. NeurIPS Datasets and Benchmarks.
- Dettmers & Zettlemoyer (2023). The case for 4-bit precision: k-bit inference scaling laws. ICML.

**Model pages**

- <https://huggingface.co/Qwen/Qwen3-4B>
- <https://huggingface.co/Qwen/Qwen3-8B>
- <https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501>
- <https://huggingface.co/Equall/SaulLM-54B-Instruct>
- <https://huggingface.co/Equall/SaulLM-54-Base>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- <https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/>
- <https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash>
- <https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash-0731>
- <https://docs.claude.com>
