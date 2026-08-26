# Analysis Framework

This repository evaluates the LEET-Arg benchmark dataset, which contains
statement-level legal reasoning questions, expert rationales, model responses,
and evaluator scores.

## Repository Contents

- `run_benchmark.py`: Runs the LEET-Arg questions against a Hugging Face causal language model and stores parsed responses in `slm_results/`.
- `prompts.py`: Defines the English reasoning prompt used by the benchmark.
- `model_configs/`: Per-model inference settings.
- `benchmarks/LEET_Arg_Questions_cleaned.json`: The current cleaned LEET-Arg question set: 93 questions and 301 statement units.
- `benchmarks/LEET_Arg_Model_Responses.json`: Responses and LLM-as-a-Judge evaluations for seven models across the LEET-Arg questions.
- `tools/clean_leet_arg.py`: Rebuilds statements for known segmentation issues in a source dataset.
- `tools/validate_leet_arg.py`: Validates statement structure, source consistency, punctuation, and suspicious content.
- `docs/`: Research notes and evaluation reports.

## Setup

### First-time setup

1. Install Python 3.10 or newer and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the runtime dependencies:

```bash
pip install torch transformers accelerate pyvene transformer-lens pydantic
```

3. Install the Hugging Face command-line tools. The `hf` command is included with the Hugging Face Hub package:

```bash
pip install -U huggingface_hub
hf --help
```

4. Create or sign in to a [Hugging Face](https://huggingface.co) account, then authenticate locally:

```bash
hf auth login
```

When prompted, paste a Hugging Face access token with permission to read models. Keep the token out of source files and shell history where possible.

The repository supports Linux and macOS. Use the activation command appropriate for your shell, for example `source .venv/bin/activate.fish` for Fish.

#### Troubleshooting Hugging Face imports

This benchmark uses text-only models. If Transformers reports `Could not import module 'LlamaConfig'` and the traceback ends with `torchvision::nms`, remove the unused incompatible vision package from the active environment:

```bash
pip uninstall -y torchvision
```

If you want to clean up models in your disk since we are testing one model at a time, use the hf CLI. For listing all saved models in the cache, use `hf cache list` and `hf rm <model>` to remove any of them.

### Downloading models from Hugging Face

The default model is `meta-llama/Llama-3.2-1B-Instruct`. It is gated, so first visit the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), accept its license, and wait for access to be approved. Then download it into a local directory:

```bash
hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir models/Llama-3.2-1B-Instruct
```

Public models can be downloaded in the same way. For example:

```bash
hf download microsoft/Phi-4-mini-instruct \
  --local-dir models/Phi-4-mini-instruct
```

You can either let Transformers download and cache a model automatically by passing its Hugging Face ID:

```bash
python run_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct
```

or use a previously downloaded local directory:

```bash
python run_benchmark.py --model models/Llama-3.2-1B-Instruct
```

For local directories, the model configuration lookup falls back to the default Llama settings unless a matching entry exists in `model_configs/`. Copy and adapt an existing file there if the model requires different dtype, trust, seed, or generation settings. Model downloads can be several gigabytes, so ensure sufficient disk space and use a GPU or other supported accelerator when available.

## LEET-Arg Benchmark

### Test setup

For each benchmark question, the model receives only:

1. The base system prompt defined in `prompts.py`.
2. The individual LEET-Arg question from
  `benchmarks/LEET_Arg_Questions_cleaned.json`.

The model is not given the dataset answer, the expert rationale, other model
responses, or demonstrations from other questions. The generated response is
parsed for its answer and saved with the question in `slm_results/`.

Run all questions with Llama 3.2 1B Instruct:

```bash
python3 run_benchmark.py
```

Run only questions whose IDs start with `2021_`:

```bash
python3 run_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct --year 2021
```

Results are written to `slm_results/<model-signature>.json`. Existing results are retained and new responses are appended by default. Use `--overwrite` to clear that model's result file before running:

```bash
python3 run_benchmark.py --model microsoft/Phi-4-mini-instruct --year 2021 --overwrite
```

Each stored record retains the source question and includes `model_answer` and `model_rationale`, parsed from the model's `Answer-<choice>.` response. The output terminal will not display the rationale for readibility purposes, storing it directly in the file.

### Running a subset

`--year` filters by ID prefix; the smallest year is 2021 with 15 questions. For
smaller or more targeted runs:

```bash
# 10 questions, sampled round-robin across years rather than all from one year
python3 run_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct --limit 10

# specific questions
python3 run_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct --ids 2021_02,2024_05

# continue an interrupted run without appending duplicate records
python3 run_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct --resume
```

Score a subset with `--present-only`, or every unrun question is counted as an
unparseable response:

```bash
python3 slm_results/evaluate_results.py --present-only
```

### Choosing `max_new_tokens`

`--max-new-tokens` overrides the model config for one run, and each record
stores `completion_tokens` plus a `stop_reason` of `eos`, `length`, `stop`, or
`error`. `tools/token_budget_report.py` turns that into a per-model
recommendation:

```bash
python3 tools/token_budget_report.py
```

Because every config decodes greedily (`do_sample: false`), a response generated
at cap N is a prefix of the same response at cap M > N, and generation stops at
EOS regardless of the cap. So a generous budget costs no extra time for
responses that would have finished anyway: run once at a generous cap, then read
the recommendation. Responses that stopped on `length` are right-censored, so the
report refuses to recommend a cap while any remain and instead names the ids to
rerun at a higher budget.

This matters most for thinking models. `parse_model_response` returns
`model_answer: None` whenever `<think>` has no closing `</think>`, so every
truncated response is a guaranteed unparseable — a token-budget problem that
looks like a reasoning failure in the scores.

### Run manifests

Every invocation writes `runs/<utc-timestamp>_<model-signature>/` containing:

- `manifest.json` — resolved Hugging Face revision SHA, chat-template hash,
  merged generation config, system prompt hash, dataset hash and the exact
  question IDs run, git commit and dirty flag, library versions, GPU and compute
  capability, and aggregate token/throughput totals.
- `console.log` — stdout and stderr, which is where the config warning and
  reasoning-tag warnings go.

Each result record carries the matching `run_id`, so any row traces back to the
run that produced it. This matters because `--resume` and `--overwrite` let one
result file accumulate records from several sessions at different token budgets.

Pin a model version with `--revision <sha>`; the resolved SHA is recorded either
way.

### Running on Colab

`notebooks/colab_ssh_bench.ipynb` opens an SSH tunnel into a Colab VM so the
benchmark can be driven from a local terminal. Use an **L4** runtime — three of
the four model configs request bfloat16, and T4 is `sm_75` with no bf16 support.

## LEET-Arg Dataset

The cleaned question file contains 93 questions from 2021-2025 and 301 statement-level tasks. The response file contains entries for the same question IDs and includes these models:

`claude_opus4`, `claude_sonnet4`, `deepseek_r1`, `gemini-2.5-pro`, `o3`, `o3-pro`, and `o4-mini`.

Validate the current cleaned dataset:

```bash
python tools/validate_leet_arg.py \
  --input benchmarks/LEET_Arg_Questions_cleaned.json
```

The validator checks sequential statement keys, missing statements, suspiciously short or leaked instruction text, punctuation, and relaxed source consistency. The source-cleaning utility is intended for rebuilding a cleaned file from an original dataset:

```bash
python tools/clean_leet_arg.py \
  --input path/to/LEET_Arg_Questions.json \
  --output benchmarks/LEET_Arg_Questions_cleaned.json
```

Evaluate saved model responses and create the comparison CSV:

```bash
python3 slm_results/evaluate_results.py
```

## Limitations

The LEET-Arg files contain model-generated answers and evaluator judgments,
which should be treated as benchmark data rather than verified legal advice.
Model availability, hardware, Hugging Face permissions, and model-specific
settings can affect reproducibility.

### Answer format is scored strictly

`parse_model_response` requires the literal `Answer-<choice>` marker the system
prompt asks for, and `normalize_answer` accepts only `1`-`5` or `①`-`⑤`. A model
that reasons well but reports its choice differently is scored as unparseable.
This is a deliberate choice — it measures instruction-following alongside
reasoning — but it means an unparseable count is not by itself evidence of poor
reasoning, and the two should be reported separately.

Two cases in the current results:

**Pharia-1-7B** opens 61 of 93 responses with a bare digit (`5 (a), (b), (c).`)
rather than `Answer-5`, giving it 65/93 unparseable. Accepting a leading bare
digit would move it from 6/93 (6.5%) to 18/93 (19.4%) — still below the
25.8% majority-class baseline (the most common gold answer, 24/93). So the
strict parser understates Pharia by roughly 13 points without changing the
conclusion.

**LFM2.5-1.2B-Thinking** answers with a bare letter (`a`, `b`, `c`, `e`) on 51 of
93 questions, which is 55% and the bulk of its 57 unparseable. Letters are
rejected on purpose: in this dataset they label sub-statements inside a
question, never a selectable choice, so a letter answer has picked the wrong
vocabulary rather than given an equivalent answer. Notably none of the 57 are
truncations — at an 8192-token budget the model finished every single response.

### Option coverage varies sharply by model

Gold answers are distributed 1:24 2:24 3:19 4:12 5:14, but no model comes close
to matching that spread. Counting only parseable answers:

| model | 1 | 2 | 3 | 4 | 5 | letters | none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | 60 | 4 | 8 | 0 | 6 | 10 | 5 |
| Phi-4-mini | 28 | 46 | 17 | 0 | 0 | 0 | 2 |
| Pharia-1-7B | 2 | 1 | 21 | 1 | 3 | 0 | 65 |
| LFM2.5-Thinking | 13 | 12 | 5 | 6 | 0 | 51 | 6 |

Llama collapses onto option 1 (65% of all questions) and Phi onto option 2.
Neither ever selects option 4, and Phi never selects 4 or 5 — together 28% of
the gold answers. The models pile onto different options, so this is a
per-model bias rather than an artifact of the prompt or choice ordering, but
the practical effect is the same: accuracy is bounded well below what the
rationales might suggest.

### Prompts are not byte-stable across days

Llama-3.2's chat template interpolates the current date (`Today Date: 26 Aug
2026`). Greedy decoding is therefore reproducible within a day but not across
days for that model family. Run manifests record the chat-template hash, but
not the rendered prompt, so this is worth pinning with `date_string=` before
publishing numbers.

See `docs/` for LEET-Arg research context and evaluation notes.
