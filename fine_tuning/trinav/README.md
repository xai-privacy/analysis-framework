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

**Model compatibility:** Jun's run used
`mlx-community/Llama-3.2-3B-Instruct-4bit`, an MLX-format quantized checkpoint.
The Chameleon scripts use Transformers and bitsandbytes, which cannot load that
MLX artifact directly. They therefore default to the original
`meta-llama/Llama-3.2-3B-Instruct` weights and quantize those to NF4 during load.
This keeps the same Llama 3.2 3B Instruct model family, but it is **not the exact
same quantized checkpoint** Jun trained. The script rejects Jun's MLX model ID
instead of silently treating it as compatible. To fine-tune the exact MLX
checkpoint, use an MLX-compatible Apple Silicon environment; the current
CUDA/Chameleon workflow cannot promise identical base weights.

Accept the original Llama model's license on Hugging Face and authenticate from
the Chameleon instance. The model is fetched by Transformers when the run
starts; no model files need to be uploaded with your scripts. Install the
CUDA-enabled PyTorch build recommended for the selected Chameleon image; the
requirements file intentionally does not replace PyTorch with a CPU build.

### Upload the Trinav folder

From `analysis-framework/fine_tuning/chameleon` on your local machine, create
the destination and copy the complete personal workflow. Put SSH options before
the source/destination operands:

```bash
ssh -o StrictHostKeyChecking=no -i ../my-chi-key.pem \
	cc@129.114.26.200 'mkdir -p ~/xai'

scp -o StrictHostKeyChecking=no -i ../my-chi-key.pem -r ../trinav \
	cc@129.114.26.200:~/xai/
```

Then log in and enter the uploaded directory:

```bash
ssh -o StrictHostKeyChecking=no -i ../my-chi-key.pem cc@129.114.26.200
cd ~/xai/trinav
```

The uploaded `trinav` directory includes the train/validation JSONL, split
manifest, scripts, and Python requirements. That is sufficient for training,
evaluation, and plotting. `prepare_data.py` is for regenerating the data from
the repository; to run it remotely, upload the repository benchmark source, the
final-test ID file, and `fine_tuning/jun/leet_arg_format.py` as well.

```bash
nvidia-smi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch/GPU is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install --upgrade pip
python -m pip install -r requirements-chameleon.txt
hf auth login
```

Train for 180 steps, evaluating and saving every 20 steps, matching Jun's run:

```bash
python train_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--train-data data/train.jsonl \
	--valid-data data/valid.jsonl \
	--split data/split.json \
	--output-dir adapters_chameleon \
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
python plot_training_metrics.py \
	--trainer-state adapters_chameleon/trainer_state.json
```

This writes `training_metrics.csv` and `training_metrics.png` beside the
adapter. The plot shows training/validation loss, learning rate, and epoch by
step. Jun's report notes that falling training loss did not imply better
validation verdict accuracy, so compare these curves with the evaluator's
statement and question accuracy rather than selecting solely by training loss.

Evaluate the base model with the output-format hint, then evaluate the adapter
with the same prompt used in training:

```bash
python evaluate_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--format-hint \
	--out results/base_hint.json

python evaluate_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path adapters_chameleon \
	--out results/finetuned.json
```

Evaluation reports parse rate, statement accuracy, and question accuracy, and
resumes after interruption. It refuses to evaluate any final-test question.
If CUDA, `bitsandbytes`, or model access is unavailable, fix the Chameleon
instance setup before starting a full run.