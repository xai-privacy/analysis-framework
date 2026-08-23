# LEET-Arg Documentation

This repository currently focuses on evaluating model responses to the LEET-Arg
benchmark dataset. The cleaned dataset contains 93 questions and 301
statement-level tasks from 2021 through 2025.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the runtime dependencies:

```bash
pip install torch transformers accelerate pyvene transformer-lens pydantic
```

Install the Hugging Face command-line tools and authenticate:

```bash
pip install -U huggingface_hub
hf auth login
```

## Run The Benchmark

Run all questions with the default model:

```bash
python run_benchmark.py
```

Choose a Hugging Face model and optionally restrict the run to one year:

```bash
python run_benchmark.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --year 2023
```

Use `--overwrite` to replace an existing result file. Otherwise, responses are
appended to the model's JSON file in `slm_results/`.

## Evaluate Results

Evaluate saved model responses with:

```bash
python slm_results/evaluate_results.py
```

The evaluator creates `slm_results/main_results.csv`, reporting correct,
incorrect, unparseable, and total responses for every model result file.

## Validate The Dataset

```bash
python tools/validate_leet_arg.py \
  --input benchmarks/LEET_Arg_Questions_cleaned.json
```
