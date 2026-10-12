# Trinav Fine-Tuning Data

This folder prepares LEET-Arg fine-tuning data in the per-statement rationale
format used by Jun's experiment. Each assistant target contains the rationale
summary, reasoning, and a `CORRECT` or `INCORRECT` verdict for every statement.
The answer-choice list is removed from the model input.

The source is
`benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`.
The 16 final-test questions, three diagram questions without per-statement
rationales, and inconsistent question `2025_16` are excluded before splitting.

The current source has 94 questions, while Jun's run used the earlier 93-question
snapshot. The only added ID is `2021_13`; it has complete rationales now, but is
excluded here to reproduce Jun's cohort and exact split. The split therefore
contains the same 58 training and 15 validation question IDs as Jun's seed-42
split. The extra snapshot exclusion is recorded in `data/split.json`.

Regenerate the data from the repository root or this directory:

```bash
python3 fine_tuning/trinav/prepare_data.py
```

The script writes `data/train.jsonl`, `data/valid.jsonl`, and `data/split.json`.
The ID-only validation and test manifests are also written under `data/`; no
test examples are included in the training or validation JSONL. Keep the test
set untouched when comparing checkpoints and select checkpoints using
validation results.

## Chameleon CUDA Fine-Tuning

Run these commands in a GPU-backed Chameleon instance, not in the local WSL
environment. Start with one NVIDIA GPU. The training script uses QLoRA: a
4-bit NF4 base model and LoRA adapters, with loss masked on the system/user
prompt so only the assistant rationale and verdicts are trained.

The default model is the gated `meta-llama/Llama-3.2-3B-Instruct`. Accept its
license on Hugging Face and authenticate from the Chameleon instance. Install
the CUDA-enabled PyTorch build recommended for the selected Chameleon image;
the requirements file intentionally does not replace PyTorch with a CPU build.

```bash
cd analysis-framework
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r fine_tuning/trinav/requirements-chameleon.txt
hf auth login
python fine_tuning/trinav/prepare_data.py
```

Train for 180 steps, evaluating and saving every 20 steps, matching Jun's run:

```bash
python fine_tuning/trinav/train_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--train-data fine_tuning/trinav/data/train.jsonl \
	--valid-data fine_tuning/trinav/data/valid.jsonl \
	--output-dir fine_tuning/trinav/adapters_chameleon \
	--max-steps 180 \
	--eval-steps 20 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

The script evaluates validation loss every 20 steps, keeps the best checkpoint,
and writes the final adapter to `adapters_chameleon/`. It uses batch size 1,
LoRA rank 8, and a constant learning rate. If an example exceeds the configured
sequence length, the script stops with its ID rather than silently truncating
the rationale.

Trainer logs include training loss and learning rate every 10 steps, plus
validation loss every 20 steps. Export the raw log rows and plot the curves
after training:

```bash
python fine_tuning/trinav/plot_training_metrics.py \
	--trainer-state fine_tuning/trinav/adapters_chameleon/trainer_state.json
```

This writes `training_metrics.csv` and `training_metrics.png` beside the
adapter. The plot shows training/validation loss, learning rate, and epoch by
step. Jun's report notes that falling training loss did not imply better
validation verdict accuracy, so compare these curves with the evaluator's
statement and question accuracy rather than selecting solely by training loss.

Evaluate the base model with the output-format hint, then evaluate the adapter
with the same prompt used in training:

```bash
python fine_tuning/trinav/evaluate_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--format-hint \
	--out fine_tuning/trinav/results/base_hint.json

python fine_tuning/trinav/evaluate_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path fine_tuning/trinav/adapters_chameleon \
	--out fine_tuning/trinav/results/finetuned.json
```

Evaluation reports parse rate, statement accuracy, and question accuracy, and
resumes after interruption. It refuses to evaluate any final-test question.
If CUDA, `bitsandbytes`, or model access is unavailable, fix the Chameleon
instance setup before starting a full run.