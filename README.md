# Analysis Framework

This repository evaluates whether language models can follow explicit legal and causal rules. It contains two related workflows:

1. A runnable synthetic patent lost-profits benchmark using structured predicate extraction and deterministic rule application.
2. The LEET-Arg benchmark dataset, which contains statement-level legal reasoning questions, expert rationales, model responses, and evaluator scores.

## Repository Contents

- `run_benchmark.py`: Runs the LEET-Arg questions against a Hugging Face causal language model and stores parsed responses in `slm_results/`.
- `structured_outputs.py`: Parses predicate JSON and applies the deterministic `AWARDED`/`DENIED` rule.
- `prompts.py`: Defines the English reasoning prompt used by the benchmark.
- `probe_activations.py`: Extracts a model activation vector for the legacy patent benchmark.
- `steer_inference.py`: Tests inference-time activation steering for that benchmark.
- `model_configs/`: Per-model inference settings.
- `benchmarks/LEET_Arg_Questions_cleaned.json`: The current cleaned LEET-Arg question set: 93 questions and 301 statement units.
- `benchmarks/LEET_Arg_Model_Responses.json`: Responses and LLM-as-a-Judge evaluations for seven models across the LEET-Arg questions.
- `tools/clean_leet_arg.py`: Rebuilds statements for known segmentation issues in a source dataset.
- `tools/validate_leet_arg.py`: Validates statement structure, source consistency, punctuation, and suspicious content.
- `docs/`: Research notes and evaluation reports.

The LEET-Arg JSON files are currently dataset and analysis artifacts; `run_benchmark.py` runs the separate six-case legacy benchmark and does not execute the LEET-Arg response-generation workflow.

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

Run all questions with Llama 3.2 1B Instruct:

```bash
python run_benchmark.py
```

Run only questions whose IDs start with `2021_`:

```bash
python run_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct --year 2021
```

Results are written to `slm_results/<model-signature>.json`. Existing results are retained and new responses are appended by default. Use `--overwrite` to clear that model's result file before running:

```bash
python run_benchmark.py --model microsoft/Phi-4-mini-instruct --year 2021 --overwrite
```

Each stored record retains the source question and includes `model_answer` and `model_rationale`, parsed from the model's `Answer-<choice>.` response. The output terminal will not display the rationale for readibility purposes, storing it directly in the file.

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

## Activation Probing and Steering

The legacy patent benchmark also includes experimental activation analysis:

```bash
python probe_activations.py
python steer_inference.py
```

These scripts use the same model and DSL options as the benchmark and are exploratory research tools, not a general-purpose training or deployment pipeline.

## Limitations

The patent benchmark is small and synthetic, so passing it does not establish general legal or causal reasoning. The LEET-Arg files contain model-generated answers and evaluator judgments, which should be treated as benchmark data rather than verified legal advice. Model availability, hardware, Hugging Face permissions, and model-specific settings can affect reproducibility.

See `docs/` for research context, model evaluations, and formal reasoning notes.
