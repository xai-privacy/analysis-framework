# Model Selection Report: Reasoning-Error Taxonomy on LEETArg and MBE

Prepared 2026-10-31 (section 9 added 2026-11-04)

## 1. Purpose

This report justifies the model set for a taxonomy of reasoning mistakes that small open-weight models and a frontier model, make on two benchmarks, **LEETArg** and **MBE**. It explains, for each model, why it fits the available hardware and why it is informative for these benchmarks. We will try to use gpt-oss-120gb but revert to 20b if the former model does not fit within the capacity of the hardware we have access to.

**Selected models**

| # | Model | Organization | Parameters | Access |
|---|---|---|---|---|
| 1 | Llama-3.2-3B-Instruct | Meta | 3B dense | Open weights |
| 2 | DeepSeek-R1-Distill-Llama-8B | DeepSeek | 8B dense | Open weights |
| 3 | Qwen3-8B | Alibaba | 8B dense | Open weights |
| 4 | Mistral-Small (24B) | Mistral AI | 24B dense | Open weights |
| 5 | gpt-oss-20b or gpt-oss-120b | OpenAI | 21B / 117B MoE (about 3.6B / 5.1B active) | Open weights |
| 6 | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | Anthropic | Not disclosed | Closed, API only |


## 2. Hardware fit

### 2.1 Available flavors

| Flavor | GPU memory | Role in this study |
|---|---|---|
| `g1.h100.vgpu.1g.12gb` | 12 GB | Prompt and parser development, debugging, Llama-3.2-3B runs |
| `g1.h100.pci.1` | 80 GB | Main evaluation of every open model, LoRA experiments up to about 8B |
| `g1.h100.pci.4` | 4 x 80 GB | Parallel evaluation (one vLLM server per GPU), or tensor parallelism for gpt-oss-120b if more KV headroom is needed |

## 3. Benchmark relevance

### 3.1 MBE

The Multistate Bar Exam is a four-option multiple-choice test of US legal reasoning. Each item is a short fact pattern plus a question that requires recalling the right rule and applying it to the facts, across subjects such as torts, contracts, evidence, constitutional law, criminal law and procedure, real property, and civil procedure.

### 3.2 LEETArg

Items contain a passage (often a dialogue between named speakers or a short argument), several labeled statements, a per-statement correct/incorrect judgment, and a question-level answer option determined by which statements are correct. Statement sets are either (a)-(c) or ①-⑤.

## 4. Why each model is relevant

### 4.1 Llama-3.2-3B-Instruct (Meta)

- **Hardware:** fits all three flavors in bf16.
- **Benchmarks:** it is the model your existing adapter was trained on, so base-versus-adapter comparison is possible with the identical prompt. It also anchors the small end of the scale axis.
- **Taxonomy value:** expected to produce the most format and control failures, which makes it the main source for the format-failure branch.

### 4.2 DeepSeek-R1-Distill-Llama-8B (DeepSeek)

- **Hardware:** about 16 GB in bf16, comfortably on `pci.1`. Needs a large output budget (8-16k tokens), which shrinks batch size.
- **Benchmarks:** long chain-of-thought gives a different error profile than direct-answer models, with more room for self-correction and for over-reasoning.
- **Taxonomy value:** shows which errors reasoning removes and which it introduces, such as reversing a correct intermediate conclusion or failing to terminate inside the token budget.
- **Practical notes:** follow the model card's recommended sampling settings (temperature around 0.6), use no system prompt, and strip the reasoning block before parsing. Keep the full trace for annotation.
- **Confound:** this model is a Llama-3.1-8B derivative, while the baseline is Llama-3.2-3B. Differences from model 1 mix scale, version, and reasoning distillation. If you need a clean distillation comparison, add Llama-3.1-8B-Instruct as a control (about 16 GB, cheap).

### 4.3 Qwen3-8B (Alibaba)

- **Hardware:** about 16 GB in bf16 on `pci.1`.
- **Benchmarks:** one set of weights supports a thinking mode and a non-thinking mode (Qwen3 Technical Report, arXiv 2025), so thinking on versus off is the cleanest reasoning comparison in the study. Qwen3 also has broad multilingual coverage, relevant to LEET's Korean origin.
- **Taxonomy value:** isolates the effect of reasoning from the effect of the model.
- **Practical notes:** the model card recommends different sampling settings per mode and advises against greedy decoding in thinking mode, so the two conditions will not share identical decoding parameters. Report this.

### 4.4 Mistral-Small, 24B (Mistral AI)

- **Hardware:** about 48 GB in bf16 fits a single 80 GB GPU with room for KV, so no quantization is needed.
- **Benchmarks:** a non-reasoning instruct model at mid scale, from a European lab with a different training mix. It fills the gap between the 8B models and the large MoE.
- **Taxonomy value:** shows whether errors seen at 3-8B persist at 24B without any reasoning mode, so scale is separated from reasoning.
- **Practical notes:** several Mistral-Small 24B checkpoints exist (for example 2501 and later releases). Pin the exact checkpoint and record its license, since terms can differ between releases.

### 4.5 gpt-oss-20b or gpt-oss-120b (OpenAI)

- **Hardware:** both are MoE models released in native MXFP4. The 120B fits on one 80 GB GPU per the model card. The 20B is small enough to run alongside other jobs on `pci.1`.
- **Benchmarks:** configurable reasoning effort (low, medium, high) gives a second clean reasoning axis, and the 20B/120B pair gives a within-family scale axis.
- **Taxonomy value:** acts as the upper anchor among open-weight models and shows which error types remain at the top of the open-weight range.
- **Caveats:** active parameters are only about 3.6B and 5.1B, so "scale" here means total rather than active size. Treat it separately from the dense-model scale curve. The models use the harmony response format, so confirm that your vLLM version and prompt template handle it.

### 4.6 Claude Sonnet 5.5 (Anthropic, closed)

- **Hardware:** none; API only.
- **Benchmarks:** serves as a reference ceiling, showing which errors are specific to small open-weight models and which are shared by frontier models.
- **Limits:** no weights, no logits, and reasoning visibility depends on the API configuration. Check what trace information you can retrieve before planning annotation.

## 5. Axis coverage and confounds

| Axis | Coverage | Clean comparison? |
|---|---|---|
| Scale (dense) | 3B, 8B, 24B | Partly; each size is a different family |
| Scale (within family) | gpt-oss-20b to 120b | Yes, but MoE with small active parameters |
| Reasoning | Qwen3 thinking on/off; gpt-oss effort levels | Yes |
| Reasoning (distillation) | R1-Distill-Llama-8B vs Llama | Only with a Llama-3.1-8B control |
| Family | Meta, DeepSeek, Alibaba, Mistral, OpenAI, Anthropic | Confounded with scale |
| Open vs closed | Sonnet 5.5 | Single data point |

## 6. Suggested evaluation conditions

| Condition | Flavor | Precision | Notes |
|---|---|---|---|
| Llama-3.2-3B base | `vgpu` or `pci.1` | bf16 | Also run the adapter for comparison |
| R1-Distill-Llama-8B | `pci.1` | bf16 | 8-16k output tokens |
| Qwen3-8B (thinking) | `pci.1` | bf16 | Model-card sampling for thinking mode |
| Qwen3-8B (non-thinking) | `pci.1` | bf16 | Model-card sampling for non-thinking mode |
| Mistral-Small 24B | `pci.1` | bf16 | Pin checkpoint |
| gpt-oss-20b (low/high effort) | `pci.1` | MXFP4 | At minimum low and high |
| gpt-oss-120b (low/high effort) | `pci.1` (or `pci.4` TP) | MXFP4 | Small batch size |
| Claude Sonnet 5.5 | API | n/a | Fixed temperature, logged settings |
| Optional: OLMo 2 7B Instruct | `pci.1` | bf16 | Multi-lingual coverage |

## 7. Protocol recommendations

1. Give every model the same prompt, with explicit statement counts and labels in the gold format (including ①-⑤ items).
2. Report parse rate and parsed-only accuracy separately from overall accuracy, and include an always-CORRECT baseline.
3. Constrain the final verdict where the engine allows it, and strip reasoning traces before parsing while keeping them for annotation.
4. Pin the vLLM version, `max_model_len`, sampling parameters, and seeds, and log them with each result file.
5. Hand-validate a stratified sample of error labels and report annotator agreement.

## 8. Release dates, training cutoffs, and dataset provenance

### 8.1 Models

| Model | Release date | Training-data / knowledge cutoff |
|---|---|---|---|
| Llama-3.2-3B-Instruct | Sep 25, 2024 | Dec 2023 |
| DeepSeek-R1-Distill-Llama-8B | Jan 20, 2025 | **Not sure.** |
| Qwen3-8B | Apr 28, 2025 | **Not sure.** |
| Mistral-Small-24B-Instruct-2501 | Jan 30, 2025 | Oct 2023 |
| gpt-oss-20b and gpt-oss-120b | Aug 5, 2025 | Jun 2024 |
| Claude Sonnet 5.5 | Sep 28, 2026 | Jun 2026 |

### 8.2 Datasets

| Dataset | Source | Publication date | Date range of underlying questions |
|---|---|---|---|
| LEET-Arg | GitHub | 2026 | LEET exams 2021-2025 |
| MBE | Hugging Face | April 2025 | **Not recorded.** |
| barexam_qa | Hugging Face | May 2025 | **Not recorded.** |

## 9. References

- Bommarito & Katz (2022). GPT Takes the Bar Exam. arXiv.
- Katz et al. (2024). GPT-4 passes the bar exam. Philosophical Transactions of the Royal Society A.
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Nature.
- Chen et al. (2024). Do NOT think that much for 2+3=? On the overthinking of o1-like LLMs. arXiv 2412.21187.
- Qwen Team (2025). Qwen3 Technical Report. arXiv.
- OpenAI (2025). gpt-oss-120b and gpt-oss-20b Model Card. arXiv 2508.10925.
- Dettmers & Zettlemoyer (2023). The case for 4-bit precision: k-bit inference scaling laws. ICML.
- Panickssery, Bowman & Feng (2024). LLM evaluators recognize and favor their own generations. NeurIPS.
- Guha et al. (2023). LegalBench. NeurIPS Datasets and Benchmarks.
- https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
- https://huggingface.co/openai/gpt-oss-120b
- https://huggingface.co/openai/gpt-oss-20b
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct