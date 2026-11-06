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

### Fine-Tuning with Microsoft Phi-4 (or Phi-4-mini)

To fine-tune with `microsoft/Phi-4-mini-instruct`, pass `--model microsoft/Phi-4-mini-instruct`. The training script automatically detects Phi-4's fused projection architecture (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`) and applies LoRA to those layers, saving to `adapters_phi4/` by default:

```bash
python train_chameleon.py \
	--model microsoft/Phi-4-mini-instruct \
	--train-data data/train.jsonl \
	--valid-data data/valid.jsonl \
	--split data/split.json \
	--output-dir adapters_phi4 \
	--max-steps 180 \
	--eval-steps 20 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

Evaluate the base Phi-4 model and fine-tuned Phi-4 adapter:

```bash
# Evaluate base model with format hint
python evaluate_chameleon.py \
	--model microsoft/Phi-4-mini-instruct \
	--format-hint \
	--out results/phi4_base_hint.json

# Evaluate fine-tuned adapter
python evaluate_chameleon.py \
	--model microsoft/Phi-4-mini-instruct \
	--adapter-path adapters_phi4 \
	--out results/phi4_finetuned.json
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

## Upload the Adapter to Hugging Face

The training output is a PEFT adapter, not a standalone copy of the base model.
Upload the adapter configuration and weights; the base model is still loaded
separately when you evaluate or run the adapter. Keep the Hugging Face repo
private unless you intentionally want to publish it. Do not put access tokens in
the script or README.

On the VM, log in and run this from the directory containing the adapter:

```bash
hf auth login
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi

adapter_dir = Path("adapters_chameleon")
repo_id = "YOUR_HF_USERNAME/leetarg-llama32-lora"
config = adapter_dir / "adapter_config.json"
weights = next((adapter_dir / name for name in (
	"adapter_model.safetensors", "adapter_model.bin"
) if (adapter_dir / name).is_file()), None)

if not config.is_file() or weights is None:
	raise SystemExit(f"Adapter config/weights not found in {adapter_dir}")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
for path in (config, weights):
	api.upload_file(
		path_or_fileobj=str(path),
		path_in_repo=path.name,
		repo_id=repo_id,
		repo_type="model",
	)
print(f"Uploaded adapter to https://huggingface.co/{repo_id}")
PY
```

Replace `YOUR_HF_USERNAME` with your Hugging Face account or organization. The
base model weights are not uploaded. Preserve the adapter locally and copy it
off the VM before the lease ends.

## Run a Test Set with a Hugging Face Model

`run_hf_questions.py` tests a JSON array of LEET-Arg question records with
either a base Hugging Face model or a PEFT adapter. The input must include
`id`, `original_question`, `statements`, and `answer` for each question. Use
`--test-ids` to select the held-out IDs from a full benchmark file. The runner
refuses any ID listed in `data/split.json` as train or validation data.

From the repository root, test a base model:

```bash
python fine_tuning/trinav/run_hf_questions.py \
	--input benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json \
	--split fine_tuning/trinav/data/split.json \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--run-name llama32-base
```

Test the fine-tuned adapter with the same base model and prompt format:

```bash
python fine_tuning/trinav/run_hf_questions.py \
	--input benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json \
	--split fine_tuning/trinav/data/split.json \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path fine_tuning/trinav/adapters_chameleon \
	--run-name llama32-adapter-180
```

To load the uploaded adapter instead, log in to Hugging Face on the machine
running the test and pass its repo ID as `--adapter-path`:

```bash
python fine_tuning/trinav/run_hf_questions.py \
	--input benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json \
	--split fine_tuning/trinav/data/split.json \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path YOUR_HF_USERNAME/leetarg-llama32-lora \
	--run-name llama32-adapter-hf
```

By default, each run is saved under
`testing/baseline/huggingface/<run-name>/`: one Markdown file per question,
`results.json` with statement/question metrics, and `run.json` with settings.
The repository's final-test file contains IDs only. To run on the Chameleon VM,
also copy `benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`
and `testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json` into
`~/xai/trinav/data/`, then use `--input
data/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`, `--test-ids
data/LEET_Arg_Questions_Test_Set.json`, `--adapter-path adapters_chameleon`,
and `--output-root testing/baseline/huggingface`. The model argument always
names the base model; `--adapter-path` adds the fine-tuned adapter on top.

### Chameleon: Baseline Then Uploaded Adapter

First, from the repository root on your local machine, copy the full benchmark
and final-test ID list to the VM. Set the IP and key path for your instance:

```bash
scp -i <private-key.pem> \
	benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	cc@<public-ip>:~/xai/trinav/data/
scp -i <private-key.pem> \
	testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json \
	cc@<public-ip>:~/xai/trinav/data/
```

On the VM, run the base model first. This establishes the baseline on the same
held-out questions and with the same statement-level prompt used for the
adapter. No `--format-hint` is added in either run.

```bash
cd ~/xai/trinav
source .venv/bin/activate
hf auth login

python run_hf_questions.py \
	--input data/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids data/LEET_Arg_Questions_Test_Set.json \
	--split data/split.json \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--run-name llama32-base \
	--output-root testing/baseline/huggingface
```

After training and uploading the adapter using the section above, run it on the
same test IDs. Replace the placeholder with your Hugging Face username or
organization. You must have access to both the gated base model and the adapter
repo from the VM.

```bash
python run_hf_questions.py \
	--input data/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids data/LEET_Arg_Questions_Test_Set.json \
	--split data/split.json \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path YOUR_HF_USERNAME/leetarg-llama32-lora \
	--run-name llama32-adapter-hf \
	--output-root testing/baseline/huggingface
```

The two runs are written to separate directories under
`testing/baseline/huggingface/`. Compare their `results.json` summaries and
question Markdown files. Keep this final test set untouched while selecting
training steps or adapters; use the validation set for those decisions.